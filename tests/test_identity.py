"""Pair identity: compatibility classes, fingerprints, and runtime policy.

All identities here are synthetic fixtures; no real deployment values.
"""
from __future__ import annotations

import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace

from tests.util import install_network_guard, trained_once

install_network_guard()

from config import settings  # noqa: E402
from routing import identity as ident  # noqa: E402
from routing.artifact import ArtifactError  # noqa: E402
from routing.learned_router import LearnedRouter  # noqa: E402
from schemas import Task  # noqa: E402

_FIELDS = ("mock_mode", "tier_mode", "local_backend", "local_model_name",
           "strong_model_name", "remote_model_name", "local_context_tokens",
           "router_artifact_path", "fireworks_api_key", "system_prompt",
           "router_mode")

FAKE_FACTS = {"digest": "sha256:fixture-digest", "quantization": "Q4_K_M",
              "template_sha256": "1" * 64, "parameters_sha256": "2" * 64,
              "runtime_version": "0.0-fixture"}


class SettingsCase(unittest.TestCase):
    def setUp(self):
        self._saved = {f: getattr(settings, f) for f in _FIELDS}
        settings.mock_mode = False

    def tearDown(self):
        for f, v in self._saved.items():
            setattr(settings, f, v)


def _ollama_identity(probe=lambda _m: FAKE_FACTS, **overrides):
    settings.tier_mode = "local_remote"
    settings.local_backend = "ollama"
    settings.local_model_name = "fixture-local:27b"
    settings.local_context_tokens = 8192
    identity = ident.runtime_identity(probe=probe)
    for dotted, value in overrides.items():
        target = identity
        *parents, leaf = dotted.split(".")
        for p in parents:
            target = target[p]
        target[leaf] = value
    return identity


class TestCompare(SettingsCase):
    def test_identical_identity_is_compatible(self):
        runtime = _ollama_identity()
        result = ident.compare({"pair_identity": copy.deepcopy(runtime)}, runtime)
        self.assertEqual(result.status, ident.COMPATIBLE)
        self.assertEqual(runtime["cheap"]["context_tokens"], 8192)
        self.assertEqual(runtime["cheap"]["digest"], "sha256:fixture-digest")
        self.assertIs(runtime["cheap"]["think"], False)

    def test_retrain_fields(self):
        trained = _ollama_identity()
        for dotted, value in [("cheap.model", "other:27b"),
                              ("cheap.digest", "sha256:other"),
                              ("cheap.quantization", "Q8_0"),
                              ("cheap.system_prompt_sha256", "0" * 64),
                              ("strong.model", "accounts/x/models/other")]:
            with self.subTest(field=dotted):
                runtime = _ollama_identity(**{dotted: value})
                result = ident.compare({"pair_identity": trained}, runtime)
                self.assertEqual(result.status, ident.RETRAIN_REQUIRED)
                self.assertIn(dotted, [m["field"] for m in result.mismatches])

    def test_revalidate_fields(self):
        trained = _ollama_identity()
        for dotted, value in [("cheap.context_tokens", 32768),
                              ("cheap.max_output_tokens", 2048),
                              ("strong.max_output_tokens", 8192)]:
            with self.subTest(field=dotted):
                result = ident.compare({"pair_identity": trained},
                                       _ollama_identity(**{dotted: value}))
                self.assertEqual(result.status, ident.REVALIDATION_REQUIRED)

    def test_retrain_outranks_revalidate_and_unknown(self):
        runtime = _ollama_identity(**{"cheap.model": "other", "cheap.digest": None,
                                      "cheap.context_tokens": 1})
        result = ident.compare({"pair_identity": _ollama_identity()}, runtime)
        self.assertEqual(result.status, ident.RETRAIN_REQUIRED)

    def test_unknown_values_are_unverifiable_not_compatible(self):
        no_context = _ollama_identity(**{"cheap.context_tokens": None})

        def runtime_down(_model):
            raise RuntimeError("Ollama is unavailable")

        down = _ollama_identity(probe=runtime_down)
        self.assertIsNone(down["cheap"]["digest"])
        settings.local_context_tokens = 0  # Ollama default: nothing recorded
        self.assertIsNone(ident.runtime_identity(
            probe=lambda _m: FAKE_FACTS)["cheap"]["context_tokens"])
        for trained, runtime in [(no_context, no_context),
                                 (_ollama_identity(), down)]:
            result = ident.compare({"pair_identity": trained}, runtime)
            self.assertEqual(result.status, ident.UNVERIFIABLE)
            self.assertTrue(result.unknown)

    def test_synthetic_and_legacy_metadata(self):
        runtime = _ollama_identity()
        self.assertEqual(ident.compare({}, runtime).status, ident.SYNTHETIC)
        synthetic = dict(runtime, synthetic=True)
        self.assertEqual(ident.compare({"pair_identity": synthetic}, runtime).status,
                         ident.SYNTHETIC)
        legacy = {"tier_mode": "local_remote", "local_model": "fixture-local:27b",
                  "remote_model": settings.strong_model_name}
        self.assertEqual(ident.compare(legacy, runtime).status, ident.UNVERIFIABLE)
        legacy["local_model"] = "another"
        self.assertEqual(ident.compare(legacy, runtime).status, ident.RETRAIN_REQUIRED)

    def test_unknown_identity_version_is_unverifiable(self):
        trained = dict(_ollama_identity(), identity_version="99")
        result = ident.compare({"pair_identity": trained}, _ollama_identity())
        self.assertEqual(result.status, ident.UNVERIFIABLE)


class TestFingerprints(unittest.TestCase):
    def test_exact_vs_canonical(self):
        data = [{"task_id": "t1", "prompt": "Café — 東京"}]
        text = json.dumps(data, ensure_ascii=False, indent=1)
        with tempfile.TemporaryDirectory() as tmp:
            lf, crlf = os.path.join(tmp, "lf.json"), os.path.join(tmp, "crlf.json")
            with open(lf, "wb") as fh:
                fh.write(text.encode("utf-8"))
            with open(crlf, "wb") as fh:
                fh.write(text.replace("\n", "\r\n").encode("utf-8"))
            self.assertNotEqual(ident.file_sha256(lf), ident.file_sha256(crlf))
            self.assertEqual(ident.canonical_file_sha256(lf),
                             ident.canonical_file_sha256(crlf))
            self.assertEqual(ident.canonical_file_sha256(lf),
                             ident.canonical_json_sha256(data))

    def test_canonical_does_not_normalise_strings(self):
        a = [{"prompt": "line one\nline two"}]
        b = [{"prompt": "line one\r\nline two"}]
        c = [{"prompt": "Café"}]  # decomposed form, not normalised to NFC
        d = [{"prompt": "Café"}]
        self.assertNotEqual(ident.canonical_json_sha256(a), ident.canonical_json_sha256(b))
        self.assertNotEqual(ident.canonical_json_sha256(c), ident.canonical_json_sha256(d))


class TestRuntimePolicy(SettingsCase):
    @classmethod
    def setUpClass(cls):
        _, (cls.artifact, _, _, _) = trained_once()

    def _with(self, meta):
        return replace(self.artifact, dataset_meta=meta)

    def test_production_rejects_synthetic_and_unverifiable(self):
        settings.tier_mode = "remote_pair"
        legacy = {"tier_mode": "remote_pair", "local_model": settings.cheap_model_name,
                  "remote_model": settings.strong_model_name}
        for meta in ({}, legacy):
            with self.subTest(meta=meta), self.assertRaises(ArtifactError):
                LearnedRouter(artifact=self._with(meta))

    def test_mock_or_explicit_test_use_admits_synthetic_with_warning(self):
        settings.tier_mode = "remote_pair"
        settings.mock_mode = True
        err = io.StringIO()
        with redirect_stderr(err):
            router = LearnedRouter(artifact=self._with({}))
        self.assertEqual(router.compatibility.status, ident.SYNTHETIC)
        self.assertIn("mock/test use only", err.getvalue())
        settings.mock_mode = False
        with redirect_stderr(io.StringIO()):
            LearnedRouter(artifact=self._with({}), allow_unverified=True)

    def test_retrain_required_rejected_even_in_test_use(self):
        settings.tier_mode = "remote_pair"
        trained = ident.runtime_identity()
        trained["strong"]["model"] = "accounts/x/models/other"
        for mock in (False, True):
            settings.mock_mode = mock
            with self.subTest(mock=mock), self.assertRaises(ArtifactError):
                LearnedRouter(artifact=self._with({"pair_identity": trained}),
                              allow_unverified=True)

    def test_compatible_artifact_accepted_in_production(self):
        settings.tier_mode = "remote_pair"
        router = LearnedRouter(artifact=self._with(
            {"pair_identity": ident.runtime_identity()}))
        self.assertTrue(router.compatibility.ok)

    def test_webui_real_mode_requires_verified_identity(self):
        from routing.artifact import save_artifact
        from webui import service
        from webui.service import DemoConfig

        cfg = DemoConfig(mode="hybrid", hybrid_pair="remote_pair",
                         router_kind="learned", mock=False)
        legacy = {"tier_mode": "remote_pair", "local_model": cfg.cheap_model,
                  "remote_model": cfg.strong_model}
        settings.tier_mode = "remote_pair"
        settings.strong_model_name = cfg.strong_model
        verified = {"pair_identity": ident.runtime_identity(
            tier_mode="remote_pair", cheap_model=cfg.cheap_model,
            strong_model=cfg.strong_model)}
        with tempfile.TemporaryDirectory() as tmp:
            settings.router_artifact_path = os.path.join(tmp, "r.joblib")
            save_artifact(self._with(legacy), settings.router_artifact_path)
            status = service.artifact_status(cfg)
            self.assertEqual((status["pair_match"], status["verified"]), (True, False))
            usable, reason = service._learned_router_usable(cfg, status)
            self.assertFalse(usable)
            self.assertIn("unverifiable", reason)

            save_artifact(self._with(verified), settings.router_artifact_path)
            status = service.artifact_status(cfg)
            self.assertTrue(status["verified"])
            self.assertEqual(service._learned_router_usable(cfg, status), (True, ""))


# ── S2 correction: enforcement matrix (C) ────────────────────────────────

def _remote_pair_cases():
    """Artifact metadata for each status, against the default remote_pair."""
    settings.tier_mode = "remote_pair"
    good = ident.runtime_identity()
    retrain = copy.deepcopy(good)
    retrain["strong"]["model"] = "accounts/fireworks/models/other-strong"
    revalidate = copy.deepcopy(good)
    revalidate["strong"]["max_output_tokens"] = good["strong"]["max_output_tokens"] * 2
    unknown = copy.deepcopy(good)
    unknown["strong"]["request_sha256"] = None
    legacy = {"tier_mode": "remote_pair", "local_model": settings.cheap_model_name,
              "remote_model": settings.strong_model_name}
    return {
        ident.COMPATIBLE: {"pair_identity": good},
        ident.RETRAIN_REQUIRED: {"pair_identity": retrain},
        ident.REVALIDATION_REQUIRED: {"pair_identity": revalidate},
        "unverifiable (unknown field)": {"pair_identity": unknown},
        "unverifiable (names only)": legacy,
    }


class TestEnforcementMatrix(SettingsCase):
    @classmethod
    def setUpClass(cls):
        _, (cls.artifact, _, _, _) = trained_once()

    def setUp(self):
        super().setUp()
        self._saved_router = (settings.router_mode, settings.router_artifact_path)
        self.cases = _remote_pair_cases()

    def tearDown(self):
        settings.router_mode, settings.router_artifact_path = self._saved_router
        super().tearDown()

    def _artifact(self, meta):
        return replace(self.artifact, dataset_meta=meta)

    def test_learned_router(self):
        for name, meta in self.cases.items():
            with self.subTest(case=name):
                if name == ident.COMPATIBLE:
                    self.assertTrue(LearnedRouter(artifact=self._artifact(meta)).compatibility.ok)
                else:
                    with self.assertRaises(ArtifactError):
                        LearnedRouter(artifact=self._artifact(meta))

    def test_build_router_learned_and_auto(self):
        import main
        from routing.artifact import save_artifact

        for name, meta in self.cases.items():
            with tempfile.TemporaryDirectory() as tmp:
                settings.router_artifact_path = os.path.join(tmp, "r.joblib")
                save_artifact(self._artifact(meta), settings.router_artifact_path)
                for mode in ("learned", "auto"):
                    settings.router_mode = mode
                    err = io.StringIO()
                    with self.subTest(case=name, mode=mode), redirect_stderr(err):
                        if name == ident.COMPATIBLE:
                            self.assertIsInstance(main.build_router(), LearnedRouter)
                        elif mode == "learned":
                            with self.assertRaises(RuntimeError):
                                main.build_router()
                        else:
                            router = main.build_router()
                            self.assertNotIsInstance(router, LearnedRouter)
                            self.assertEqual(router.decide(Task("t", "hi")).router_kind,
                                             "heuristic")
                            self.assertIn("ROUTER_MODE=auto", err.getvalue())
                            self.assertIn("falling back to heuristic", err.getvalue())

    def test_web_artifact_status(self):
        from routing.artifact import save_artifact
        from webui import service
        from webui.service import DemoConfig

        cases = dict(self.cases, synthetic={})
        expected = {  # (pair_match, verified, usable in real mode)
            ident.COMPATIBLE: (True, True, True),
            ident.RETRAIN_REQUIRED: (False, False, False),
            ident.REVALIDATION_REQUIRED: (False, False, False),
            "unverifiable (unknown field)": (True, False, False),
            "unverifiable (names only)": (True, False, False),
            "synthetic": (None, False, False),
        }
        for name, meta in cases.items():
            with tempfile.TemporaryDirectory() as tmp, self.subTest(case=name):
                settings.router_artifact_path = os.path.join(tmp, "r.joblib")
                save_artifact(self._artifact(meta), settings.router_artifact_path)
                cfg = DemoConfig(mode="hybrid", hybrid_pair="remote_pair",
                                 router_kind="learned", mock=False)
                status = service.artifact_status(cfg)
                usable, reason = service._learned_router_usable(cfg, status)
                self.assertEqual((status["pair_match"], status["verified"], usable),
                                 expected[name])
                if name == ident.REVALIDATION_REQUIRED:
                    self.assertIn("No revalidation-acceptance workflow", reason)
                # Synthetic allowance only for explicit mock use.
                cfg.mock = True
                mock_usable = service._learned_router_usable(
                    cfg, service.artifact_status(cfg))[0]
                self.assertEqual(mock_usable, name not in (
                    ident.RETRAIN_REQUIRED, ident.REVALIDATION_REQUIRED))


# ── S2 correction: effective generation settings (B, E) ──────────────────

class TestEffectiveRequestIdentity(SettingsCase):
    def test_generation_and_identity_share_one_builder(self):
        import ollama_client
        import remote_client
        from unittest import mock

        settings.fireworks_api_key = "unit-test-placeholder"  # post is patched
        client = remote_client.StrongRemoteClient()
        body = {"choices": [{"message": {"content": "x"}}], "usage": {}}
        response = mock.Mock(status_code=200, json=mock.Mock(return_value=body))
        with mock.patch("requests.post", return_value=response) as post, \
                redirect_stderr(io.StringIO()):
            client.generate("Q?")
        self.assertEqual(post.call_args.kwargs["json"], remote_client.build_chat_payload(
            client.model_name, client.max_tokens, "Q?"))

        settings.local_context_tokens = 4096
        with mock.patch.object(ollama_client, "request",
                               return_value={"message": {"content": "x"}, "done": True}) as req:
            ollama_client.generate("fixture:1b", "Q?")
        from generation_context import completion_messages
        self.assertEqual(req.call_args[0][1], ollama_client.build_chat_payload(
            "fixture:1b", completion_messages("Q?", ollama_client.local_system_prompt("fixture:1b")),
            stream=False))

    def test_changed_effective_setting_changes_identity(self):
        import remote_client
        from unittest import mock

        settings.tier_mode = "remote_pair"
        base = ident.runtime_identity()
        trained = {"pair_identity": base}
        saved_prompt = settings.system_prompt
        try:
            settings.system_prompt = saved_prompt + " Extra instruction."
            result = ident.compare(trained, ident.runtime_identity())
            self.assertEqual(result.status, ident.RETRAIN_REQUIRED)
            self.assertIn("strong.request_sha256", [m["field"] for m in result.mismatches])
        finally:
            settings.system_prompt = saved_prompt

        real = remote_client.build_chat_payload

        def with_top_p(*args, **kwargs):  # a code change to what is sent
            return dict(real(*args, **kwargs), top_p=0.9)

        with mock.patch.object(remote_client, "build_chat_payload", with_top_p):
            result = ident.compare(trained, ident.runtime_identity())
        self.assertEqual(result.status, ident.RETRAIN_REQUIRED)

        trained = {"pair_identity": _ollama_identity()}
        settings.local_context_tokens = 16384
        changed = ident.runtime_identity(probe=lambda _m: FAKE_FACTS)
        self.assertEqual(ident.compare(trained, changed).status,
                         ident.REVALIDATION_REQUIRED)

    def test_request_scoped_overrides_make_path_unverifiable(self):
        from generation_context import LOCAL_CONTEXT, MESSAGES, OUTPUT_LIMIT, TOOLS

        settings.tier_mode = "remote_pair"
        trained = {"pair_identity": ident.runtime_identity()}
        for var, value, label in ((OUTPUT_LIMIT, 4096, "output_limit"),
                                  (MESSAGES, [{"role": "user", "content": "hi"}], "messages"),
                                  (LOCAL_CONTEXT, 8192, "local_context"),
                                  (TOOLS, [{"type": "function"}], "tools")):
            token = var.set(value)
            try:
                with self.subTest(override=label):
                    result = ident.compare(trained, ident.runtime_identity())
                    self.assertEqual(result.status, ident.UNVERIFIABLE)
                    self.assertIn(label, result.detail)
                    _, (artifact, _, _, _) = trained_once()
                    with self.assertRaises(ArtifactError):
                        LearnedRouter(artifact=replace(artifact, dataset_meta=trained))
            finally:
                var.reset(token)
        self.assertTrue(ident.compare(trained, ident.runtime_identity()).ok)

    def test_chat_auto_fallback_is_labelled(self):
        from chat import service as chat_service

        settings.router_mode, saved = "auto", settings.router_mode
        try:
            note = chat_service._auto_fallback_note("hybrid")
            self.assertEqual(note["router_requested"], "auto")
            self.assertIn("heuristic router was used", note["router_note"])
            self.assertEqual(chat_service._auto_fallback_note("local"), {})
            cfg = chat_service.conversation_config("hybrid", None, True)
            self.assertEqual(cfg.router_kind, "heuristic")
            settings.router_mode = "heuristic"
            self.assertEqual(chat_service._auto_fallback_note("hybrid"), {})
        finally:
            settings.router_mode = saved


# ── S2 correction: Ollama template/parameters and runtime version (F, G) ──

class TestOllamaMetadata(SettingsCase):
    def _facts(self, show):
        import ollama_client
        from unittest import mock

        replies = {"/api/show": show, "/api/tags": {"models": []},
                   "/api/version": {"version": "0.1-test"}}
        with mock.patch.object(ollama_client, "request",
                               side_effect=lambda path, *a, **k: replies[path]):
            return ollama_client.model_identity("fixture:27b")

    def test_template_and_parameters_hashed_without_modification(self):
        import hashlib
        import ollama_client

        template = "{{ .System }}\r\n<|user|> {{ .Prompt }}  "
        facts = self._facts({"template": template, "parameters": "stop <|end|>\ntemperature 0.6"})
        self.assertEqual(facts["template_sha256"],
                         hashlib.sha256(template.encode("utf-8")).hexdigest())
        stripped = self._facts({"template": template.strip(), "parameters": "x"})
        self.assertNotEqual(stripped["template_sha256"], facts["template_sha256"])
        a = self._facts({"parameters": {"stop": ["<|end|>"], "num_ctx": 4096}})
        b = self._facts({"parameters": {"num_ctx": 4096, "stop": ["<|end|>"]}})
        self.assertEqual(a["parameters_sha256"], b["parameters_sha256"])
        self.assertEqual(a["template_sha256"], ollama_client.ABSENT_IN_RESPONSE)
        self.assertIsNone(self._facts({"template": 42})["template_sha256"])
        self.assertIsNone(a["digest"])  # model not listed: unknown, not guessed

    def test_older_identity_version_is_unverifiable(self):
        old = _ollama_identity()
        old["identity_version"] = "1"
        for key in ("template_sha256", "parameters_sha256", "request_sha256"):
            old["cheap"].pop(key)
        result = ident.compare({"pair_identity": old}, _ollama_identity())
        self.assertEqual(result.status, ident.UNVERIFIABLE)
        self.assertIn("predates", result.detail)

    def test_runtime_version_change_is_a_notice_not_a_block(self):
        trained = _ollama_identity()
        newer = dict(FAKE_FACTS, runtime_version="9.9-fixture")
        runtime = _ollama_identity(probe=lambda _m: newer)
        result = ident.compare({"pair_identity": trained}, runtime)
        self.assertEqual(result.status, ident.COMPATIBLE)
        self.assertEqual(len(result.notices), 1)
        self.assertIn("not proof that outputs changed", result.notices[0])
        self.assertIn("review required", result.detail)
        same = ident.compare({"pair_identity": trained}, _ollama_identity())
        self.assertEqual(same.notices, [])

    def test_resume_ignores_informational_but_not_identity(self):
        from scripts.collect_outcomes import resume_problem

        identity = _ollama_identity()
        manifest = {"manifest_version": "1", "compatibility": {
            "pair_identity": identity, "label_identity": {"grader_version": "2"},
            "tasks": {"canonical_sha256": "x", "count": 1}}}
        newer = copy.deepcopy(manifest)
        newer["compatibility"]["pair_identity"]["informational"]["cheap_runtime_version"] = "9.9"
        self.assertIsNone(resume_problem(manifest, newer))
        other = copy.deepcopy(manifest)
        other["compatibility"]["pair_identity"]["cheap"]["template_sha256"] = "3" * 64
        self.assertIn("cheap.template_sha256", resume_problem(manifest, other))


if __name__ == "__main__":
    unittest.main()
