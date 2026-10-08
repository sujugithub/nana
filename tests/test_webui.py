"""Demo UI service + server: mode isolation, billing honesty, pair checks.

Everything runs in mock mode with the external-network guard active; the
HTTP tests talk only to a server on 127.0.0.1.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import urllib.request
from unittest import mock

from tests.util import trained_once

from config import Settings, settings
from remote_client import RemoteError
from routing.artifact import save_artifact
from webui import service
from webui.server import make_server
from webui.service import DemoConfig, config_from_payload


class SettingsCase(unittest.TestCase):
    _FIELDS = (
        "mock_mode", "tier_mode", "router_mode", "router_artifact_path",
        "local_model_name", "cheap_model_name", "strong_model_name",
        "remote_model_name", "confidence_threshold", "enable_escalation",
        "allowed_models", "usage_log_path",
    )

    def setUp(self):
        self._saved = {f: getattr(settings, f) for f in self._FIELDS}

    def tearDown(self):
        for f, v in self._saved.items():
            setattr(settings, f, v)


def _artifact_with_meta(meta_overrides):
    """A copy of the shared test artifact with controlled pair metadata."""
    import copy

    _, (artifact, _, _, _) = trained_once()
    artifact = copy.copy(artifact)
    artifact.dataset_meta = dict(artifact.dataset_meta or {})
    artifact.dataset_meta.pop("tier_mode", None)
    artifact.dataset_meta.update(meta_overrides)
    return artifact


class TestConfigValidation(SettingsCase):
    def test_defaults_are_valid(self):
        cfg, errors = config_from_payload({})
        self.assertEqual(errors, [])
        self.assertEqual(cfg.mode, "hybrid")
        self.assertEqual(cfg.hybrid_pair, "local_remote")
        self.assertEqual(Settings().tier_mode, "local_remote")
        self.assertTrue(cfg.mock)

    def test_bad_enum_values_rejected(self):
        _, errors = config_from_payload({"mode": "yolo"})
        self.assertTrue(any("mode" in e for e in errors))
        _, errors = config_from_payload({"hybrid_pair": "both"})
        self.assertTrue(any("hybrid_pair" in e for e in errors))
        _, errors = config_from_payload({"router_kind": "vibes"})
        self.assertTrue(any("router_kind" in e for e in errors))

    def test_threshold_range_checked(self):
        _, errors = config_from_payload({"confidence_threshold": 1.5})
        self.assertTrue(any("confidence_threshold" in e for e in errors))
        _, errors = config_from_payload({"confidence_threshold": "abc"})
        self.assertTrue(any("confidence_threshold" in e for e in errors))

    def test_allowed_models_rejects_unlisted_model_no_substitution(self):
        settings.allowed_models = "accounts/fireworks/models/gpt-oss-120b"
        cfg, errors = config_from_payload(
            {"mode": "hybrid", "hybrid_pair": "remote_pair"}
        )
        # default cheap model is not in the allow-list
        self.assertTrue(any("cheap_model" in e for e in errors))
        # the config still carries the USER's model — nothing substituted
        self.assertEqual(
            cfg.cheap_model, "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
        )

    def test_allowed_models_ignores_models_the_mode_never_uses(self):
        settings.allowed_models = "accounts/fireworks/models/gpt-oss-120b"
        _, errors = config_from_payload({"mode": "local_only"})
        self.assertEqual(errors, [])
        _, errors = config_from_payload(
            {"mode": "hybrid", "hybrid_pair": "local_remote"}
        )
        self.assertEqual(errors, [])  # only the strong model is remote — allowed


class TestLocalOnlyIsolation(SettingsCase):
    def test_local_only_never_invokes_fireworks(self):
        cfg = DemoConfig(mode="local_only", mock=True)
        with mock.patch(
            "remote_client.FireworksClient.generate",
            side_effect=AssertionError("Fireworks was contacted in local_only"),
        ) as fw:
            result = service.execute(cfg, "hello")
        self.assertTrue(result["ok"])
        self.assertEqual(fw.call_count, 0)
        self.assertEqual(result["provider"], "local")
        self.assertFalse(result["billable"])
        self.assertEqual(result["estimated_cost_usd"], 0.0)
        self.assertEqual(result["tokens"]["billable"], 0)

    def test_local_only_failure_does_not_fall_back_to_remote(self):
        cfg = DemoConfig(mode="local_only", mock=True)
        with mock.patch(
            "local_model.LocalModel.generate",
            side_effect=RuntimeError("local model exploded"),
        ), mock.patch(
            "remote_client.FireworksClient.generate",
            side_effect=AssertionError("Fireworks was contacted in local_only"),
        ) as fw:
            result = service.execute(cfg, "hello")
        self.assertFalse(result["ok"])
        self.assertIn("local model exploded", result["error"])
        self.assertEqual(fw.call_count, 0)


class TestRemoteOnlyIsolation(SettingsCase):
    def test_remote_only_never_invokes_local_backend(self):
        cfg = DemoConfig(mode="remote_only", mock=True)
        with mock.patch(
            "local_model.LocalModel.generate",
            side_effect=AssertionError("local backend used in remote_only"),
        ) as lm:
            result = service.execute(cfg, "hello")
        self.assertTrue(result["ok"])
        self.assertEqual(lm.call_count, 0)
        self.assertEqual(result["provider"], "fireworks")
        self.assertTrue(result["would_bill"])

    def test_remote_only_failure_is_an_error_not_a_local_fallback(self):
        cfg = DemoConfig(mode="remote_only", mock=True)
        with mock.patch(
            "remote_client.FireworksClient.generate",
            side_effect=RemoteError("simulated outage"),
        ), mock.patch(
            "local_model.LocalModel.generate",
            side_effect=AssertionError("local backend used in remote_only"),
        ) as lm:
            result = service.execute(cfg, "hello")
        self.assertFalse(result["ok"])
        self.assertIn("simulated outage", result["error"])
        self.assertEqual(lm.call_count, 0)
        self.assertNotIn("answer", result)

    def test_remote_only_mock_reports_hypothetical_cost(self):
        cfg = DemoConfig(mode="remote_only", mock=True)
        result = service.execute(cfg, "what is two plus two")
        self.assertTrue(result["ok"])
        self.assertEqual(result["estimated_cost_usd"], 0.0)  # mock spends nothing
        self.assertGreater(result["mock_estimated_cost_usd"], 0.0)
        self.assertGreater(result["tokens"]["billable"], 0)


class TestHybridModes(SettingsCase):
    def test_local_remote_cheap_answer_is_local_and_free(self):
        cfg = DemoConfig(
            mode="hybrid", hybrid_pair="local_remote", mock=True,
            confidence_threshold=0.0,  # heuristic keeps everything cheap
        )
        result = service.execute(cfg, "What is the capital of France?")
        self.assertTrue(result["ok"])
        self.assertEqual(result["hybrid_pair"], "local_remote")
        self.assertEqual(result["provider"], "local")
        self.assertEqual(result["route"], "cheap")
        self.assertFalse(result["escalated"])
        self.assertFalse(result["would_bill"])
        self.assertEqual(result["tokens"]["billable"], 0)

    def test_remote_pair_cheap_answer_is_fireworks_and_billable(self):
        cfg = DemoConfig(
            mode="hybrid", hybrid_pair="remote_pair", mock=True,
            confidence_threshold=0.0,
        )
        result = service.execute(cfg, "What is the capital of France?")
        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "fireworks")
        self.assertEqual(result["route"], "cheap")
        # BOTH tiers of remote_pair are billable — a cheap Fireworks answer
        # is never described as local or free.
        self.assertTrue(result["would_bill"])
        self.assertGreater(result["tokens"]["billable"], 0)
        self.assertGreater(result["mock_estimated_cost_usd"], 0.0)

    def test_hybrid_strong_route_uses_strong_model(self):
        cfg = DemoConfig(
            mode="hybrid", hybrid_pair="remote_pair", mock=True,
            confidence_threshold=1.0,  # nothing clears the bar -> strong
        )
        result = service.execute(cfg, "Prove that 17 is prime.")
        self.assertTrue(result["ok"])
        self.assertEqual(result["route"], "strong")
        self.assertEqual(result["provider"], "fireworks")

    def test_settings_are_restored_after_execution(self):
        before = (settings.tier_mode, settings.cheap_model_name,
                  settings.mock_mode, settings.confidence_threshold)
        cfg = DemoConfig(mode="hybrid", hybrid_pair="local_remote", mock=True,
                         cheap_model="accounts/fireworks/models/kimi-k2p6",
                         confidence_threshold=0.11)
        service.execute(cfg, "hello")
        after = (settings.tier_mode, settings.cheap_model_name,
                 settings.mock_mode, settings.confidence_threshold)
        self.assertEqual(before, after)


class TestArtifactPairHandling(SettingsCase):
    def _deploy(self, artifact):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "r.joblib")
        save_artifact(artifact, path)
        settings.router_artifact_path = path

    def test_matching_pair_is_valid_and_runs(self):
        cfg = DemoConfig(mode="hybrid", hybrid_pair="remote_pair",
                         router_kind="learned", mock=True)
        self._deploy(_artifact_with_meta({
            "tier_mode": "remote_pair",
            "local_model": cfg.cheap_model,
            "remote_model": cfg.strong_model,
        }))
        status = service.artifact_status(cfg)
        self.assertIs(status["pair_match"], True)
        result = service.execute(cfg, "What is the capital of France?")
        self.assertTrue(result["ok"])
        self.assertEqual(result["routing"]["kind"], "learned")

    def test_mismatched_pair_requires_retraining_and_blocks_learned(self):
        cfg = DemoConfig(mode="hybrid", hybrid_pair="remote_pair",
                         router_kind="learned", mock=True)
        self._deploy(_artifact_with_meta({
            "tier_mode": "remote_pair",
            "local_model": "accounts/fireworks/models/some-other-cheap",
            "remote_model": cfg.strong_model,
        }))
        status = service.artifact_status(cfg)
        self.assertIs(status["pair_match"], False)
        self.assertIn("retraining required", status["detail"])
        result = service.execute(cfg, "hello")
        self.assertFalse(result["ok"])
        self.assertIn("retraining required", result["error"])
        # ...but the heuristic router stays available for the untrained pair.
        cfg.router_kind = "heuristic"
        self.assertTrue(service.execute(cfg, "hello")["ok"])

    def test_pair_change_via_model_selection_invalidates_artifact(self):
        cfg = DemoConfig(mode="hybrid", hybrid_pair="remote_pair",
                         router_kind="learned", mock=True)
        self._deploy(_artifact_with_meta({
            "tier_mode": "remote_pair",
            "local_model": cfg.cheap_model,
            "remote_model": cfg.strong_model,
        }))
        self.assertIs(service.artifact_status(cfg)["pair_match"], True)
        cfg.cheap_model = "accounts/fireworks/models/kimi-k2p6"  # user changes pair
        self.assertIs(service.artifact_status(cfg)["pair_match"], False)

    def test_toy_artifact_is_demo_only_and_labelled(self):
        cfg = DemoConfig(mode="hybrid", hybrid_pair="remote_pair",
                         router_kind="learned", mock=True)
        self._deploy(_artifact_with_meta({}))  # no tier metadata = toy
        status = service.artifact_status(cfg)
        self.assertIsNone(status["pair_match"])
        self.assertIn("toy", status["detail"])
        result = service.execute(cfg, "What is the capital of France?")
        self.assertTrue(result["ok"])
        self.assertTrue(any("TOY" in w for w in result["warnings"]))
        # In real mode the toy artifact is NOT accepted as a learned router.
        cfg.mock = False
        result = service.execute(cfg, "hello")
        self.assertFalse(result["ok"])
        self.assertIn("mock demos", result["error"])

    def test_missing_artifact_blocks_learned_with_clear_error(self):
        settings.router_artifact_path = "/nonexistent/never.joblib"
        cfg = DemoConfig(mode="hybrid", router_kind="learned", mock=True)
        status = service.artifact_status(cfg)
        self.assertFalse(status["available"])
        result = service.execute(cfg, "hello")
        self.assertFalse(result["ok"])
        self.assertIn("unavailable", result["error"])


class TestDescribe(SettingsCase):
    def test_local_only_description_promises_no_api(self):
        d = service.describe(DemoConfig(mode="local_only", mock=False))
        text = " ".join(d["summary"])
        self.assertIn("never contacted", text)
        self.assertFalse(d["billable"])

    def test_remote_only_description_shows_billable(self):
        d = service.describe(DemoConfig(mode="remote_only", mock=False))
        self.assertIn("BILLABLE", " ".join(d["summary"]))
        self.assertTrue(d["billable"])

    def test_mock_mode_is_never_billable(self):
        d = service.describe(DemoConfig(mode="remote_only", mock=True))
        self.assertFalse(d["billable"])
        self.assertIn("MOCK", " ".join(d["summary"]))

    def test_remote_pair_cheap_tier_labelled_billable_not_local(self):
        d = service.describe(DemoConfig(mode="hybrid", hybrid_pair="remote_pair"))
        text = " ".join(d["summary"])
        self.assertIn("cheap tier, Fireworks, BILLABLE", text)


class TestHttpFlow(SettingsCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(port=0, allow_real=False)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as res:
                return res.status, json.loads(res.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    def test_config_endpoint_has_no_api_key(self):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/api/config"
        ) as res:
            payload = json.loads(res.read())
        self.assertIn("defaults", payload)
        self.assertFalse(payload["real_allowed"])
        blob = json.dumps(payload)
        self.assertNotIn("fireworks_api_key", blob)
        self.assertNotIn("api_key\":", blob.replace("api_key_present", ""))

    def test_index_served(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/") as res:
            body = res.read().decode()
        self.assertIn("banana", body)
        self.assertIn("Fully local", body)

    def test_welcome_served(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/welcome") as res:
            body = res.read().decode()
        self.assertIn("Try Nana", body)
        self.assertIn("How it works", body)

    def test_run_mock_hybrid_round_trip(self):
        status, result = self._post("/api/run", {
            "config": {"mode": "hybrid", "hybrid_pair": "remote_pair",
                       "mock": True, "confidence_threshold": 0.0},
            "prompt": "What is the capital of France?",
        })
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.assertEqual(result["route"], "cheap")
        self.assertTrue(result["answer"])

    def test_describe_round_trip(self):
        status, result = self._post("/api/describe", {
            "config": {"mode": "local_only"},
        })
        self.assertEqual(status, 200)
        self.assertIn("never contacted", " ".join(result["summary"]))

    def test_server_without_real_flag_refuses_real_runs(self):
        status, result = self._post("/api/run", {
            "config": {"mode": "remote_only", "mock": False},
            "prompt": "hello",
        })
        self.assertEqual(status, 400)
        self.assertIn("--real", result["error"])

    def test_invalid_config_is_a_400_not_a_crash(self):
        status, result = self._post("/api/run", {
            "config": {"mode": "nonsense"}, "prompt": "hi",
        })
        self.assertEqual(status, 400)
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
