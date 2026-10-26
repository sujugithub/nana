"""Collector records, failure handling, resume safety, accounting, UTF-8.

Everything runs against mock or patched backends: no models, no network.
"""
from __future__ import annotations

import ast
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

from tests.util import install_network_guard

install_network_guard()

from config import settings  # noqa: E402
from remote_client import CheapRemoteClient, RemoteError, StrongRemoteClient  # noqa: E402
from scripts import collect_outcomes as co  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIELDS = ("mock_mode", "tier_mode", "strong_model_name", "remote_model_name",
           "local_context_tokens", "fireworks_api_key", "local_backend",
           "ollama_base_url")


class SettingsCase(unittest.TestCase):
    def setUp(self):
        self._saved = {f: getattr(settings, f) for f in _FIELDS}
        settings.mock_mode = True
        settings.tier_mode = "remote_pair"
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        for f, v in self._saved.items():
            setattr(settings, f, v)
        self.tmp.cleanup()

    def write_tasks(self, tasks, name="tasks.json"):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(tasks, fh, ensure_ascii=False)
        return path

    def args(self, tasks_path, **extra):
        values = dict(
            tasks=tasks_path, out=os.path.join(self.dir, "out.json"), grades=None,
            quality_threshold=0.6, local_cost_per_second=2e-5,
            cheap_in_per_mtok=settings.cheap_input_per_mtok,
            cheap_out_per_mtok=settings.cheap_output_per_mtok,
            strong_in_per_mtok=settings.strong_input_per_mtok,
            strong_out_per_mtok=settings.strong_output_per_mtok,
            max_consecutive_failures=3, local_context_tokens=None,
        )
        values.update(extra)
        return SimpleNamespace(**values)

    def run_collect(self, args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            code = co.collect(args)
        return code, err.getvalue()

    def read(self, args, suffix):
        with open(args.out + suffix, encoding="utf-8") as fh:
            return json.load(fh)


def tasks(*ids):
    return [{"task_id": t, "prompt": f"Name a colour ({t}).", "reference": "answer to",
             "grader": "contains"} for t in ids]


def strong_failing_for(*bad_ids):
    real = StrongRemoteClient.generate

    def generate(self, prompt):
        if any(f"({b})" in prompt for b in bad_ids):
            raise RemoteError("simulated read timeout")
        return real(self, prompt)
    return mock.patch.object(StrongRemoteClient, "generate", generate)


class TestCompletionMetadata(unittest.TestCase):
    def setUp(self):
        self._saved = {f: getattr(settings, f) for f in _FIELDS}
        settings.mock_mode = False

    def tearDown(self):
        for f, v in self._saved.items():
            setattr(settings, f, v)

    def test_ollama_reports_finish_reason_usage_timings_and_raw_text(self):
        import ollama_client

        final = {"message": {"content": "  42\n"}, "done": True, "done_reason": "length",
                 "prompt_eval_count": 11, "eval_count": 7,
                 "total_duration": 2_500_000_000, "load_duration": 500_000_000,
                 "prompt_eval_duration": 250_000_000, "eval_duration": 1_000_000_000}
        with mock.patch.object(ollama_client, "request", return_value=final) as req:
            c = ollama_client.generate("fixture:1b", "What is 6*7?")
        self.assertEqual((c.text, c.raw_text), ("42", "  42\n"))
        self.assertEqual((c.finish_reason, c.usage_source), ("length", "provider"))
        self.assertEqual(c.timings, {"total_s": 2.5, "load_s": 0.5,
                                     "prompt_eval_s": 0.25, "eval_s": 1.0})
        self.assertNotIn("num_ctx", req.call_args[0][1]["options"])

    def test_ollama_explicit_context_setting_is_sent(self):
        import ollama_client

        settings.local_context_tokens = 8192
        final = {"message": {"content": "x"}, "done": True}
        with mock.patch.object(ollama_client, "request", return_value=final) as req:
            c = ollama_client.generate("fixture:1b", "hi")
        self.assertEqual(req.call_args[0][1]["options"]["num_ctx"], 8192)
        self.assertEqual(c.usage_source, "")  # counts absent: not claimed

    def test_ollama_model_identity_reads_runtime_facts(self):
        import ollama_client

        replies = {
            "/api/show": {"details": {"quantization_level": "Q4_K_M", "format": "gguf",
                                      "family": "qwen", "parameter_size": "27B"}},
            "/api/tags": {"models": [{"name": "fixture:27b", "digest": "abc123"}]},
            "/api/version": {"version": "0.0-test"},
        }
        with mock.patch.object(ollama_client, "request",
                               side_effect=lambda path, *a, **k: replies[path]):
            facts = ollama_client.model_identity("fixture:27b")
        self.assertEqual(facts["digest"], "abc123")
        self.assertEqual(facts["quantization"], "Q4_K_M")
        self.assertEqual(facts["runtime_version"], "0.0-test")

    def test_fireworks_reports_finish_reason_and_usage_source(self):
        settings.fireworks_api_key = "unit-test-placeholder"  # never sent: post is patched
        client = StrongRemoteClient()
        for usage, expected in (({"prompt_tokens": 5, "completion_tokens": 3}, "provider"),
                                ({}, "estimated")):
            body = {"choices": [{"message": {"content": " B \n"}, "finish_reason": "stop"}],
                    "usage": usage}
            response = mock.Mock(status_code=200, json=mock.Mock(return_value=body))
            with self.subTest(usage=usage), \
                    mock.patch("requests.post", return_value=response), \
                    redirect_stderr(io.StringIO()):
                c = client.generate("Pick one")
            self.assertEqual((c.text, c.raw_text, c.finish_reason), ("B", " B \n", "stop"))
            self.assertEqual(c.usage_source, expected)


class TestAttemptRecords(SettingsCase):
    def test_rows_carry_ids_metadata_and_manifest(self):
        args = self.args(self.write_tasks(tasks("a", "b")))
        code, _ = self.run_collect(args)
        self.assertEqual(code, 0)
        rows = co._load_attempts(args.out + ".work.jsonl")
        manifest = self.read(args, ".manifest.json")
        self.assertEqual(len({r["attempt_id"] for r in rows}), 2)
        self.assertEqual({r["run_id"] for r in rows}, {manifest["runs"][0]["run_id"]})
        row = rows[0]
        for key in ("status", "attempt_number", "local_answer_raw", "local_finish_reason",
                    "local_usage_source", "remote_timings", "local_grade"):
            self.assertIn(key, row)
        self.assertEqual((row["status"], row["local_usage_source"]), ("complete", "mock"))
        self.assertEqual(set(manifest), {"kind", "manifest_version", "compatibility",
                                         "provenance", "runs"})
        self.assertTrue(manifest["compatibility"]["pair_identity"]["synthetic"])
        self.assertIn("grader_version", manifest["compatibility"]["label_identity"])
        self.assertEqual(manifest["provenance"]["tasks_path"], "tasks.json")
        self.assertEqual(self.read(args, "")["meta"]["pair_identity"],
                         manifest["compatibility"]["pair_identity"])

    def test_strong_failure_recorded_separately_and_reported(self):
        args = self.args(self.write_tasks(tasks("a", "b", "c")))
        with strong_failing_for("b"):
            code, err = self.run_collect(args)
        self.assertEqual(code, 0)
        rows = {r["task_id"]: r for r in co._load_attempts(args.out + ".work.jsonl")}
        failed = rows["b"]
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["error"]["tier"], "strong")
        self.assertEqual(failed["error"]["type"], "RemoteError")
        self.assertTrue(failed["local_answer"])  # cheap answer preserved
        self.assertNotIn("remote_quality", failed)  # not scored as wrong
        self.assertIn("attempt failed (strong: RemoteError)", err)

        dataset_ids = [r["task_id"] for r in self.read(args, "")["records"]]
        self.assertEqual(dataset_ids, ["a", "c"])
        acc = self.read(args, ".summary.json")["accounting"]
        self.assertEqual((acc["tasks_in_file"], acc["tasks_attempted"],
                          acc["tasks_completed"], acc["tasks_failed_infrastructure"],
                          acc["tasks_graded"], acc["tasks_unresolved_review"]),
                         (3, 3, 2, 1, 2, 0))
        self.assertEqual(acc["failed_attempts_by_tier"], {"strong": 1})
        rate = acc["rates"]["infrastructure_failure_rate"]
        self.assertAlmostEqual(rate["value"], 1 / 3)
        self.assertEqual(rate["denominator"], "tasks_attempted")
        self.assertEqual(acc["quality"]["denominator"], "usable_paired_outcomes")
        self.assertEqual(acc["quality"]["n"], 2)

    def test_cheap_failure_skips_the_paid_strong_call(self):
        args = self.args(self.write_tasks(tasks("a")))
        with mock.patch.object(CheapRemoteClient, "generate",
                               side_effect=RuntimeError("local runtime down")), \
                mock.patch.object(StrongRemoteClient, "generate") as strong:
            self.run_collect(args)
        strong.assert_not_called()
        row = co._load_attempts(args.out + ".work.jsonl")[0]
        self.assertEqual((row["status"], row["error"]["tier"]), ("error", "cheap"))

    def test_consecutive_failures_stop_the_run(self):
        args = self.args(self.write_tasks(tasks("a", "b", "c", "d")),
                         max_consecutive_failures=2)
        with strong_failing_for("a", "b", "c", "d"):
            code, err = self.run_collect(args)
        self.assertEqual(code, 3)
        self.assertEqual(len(co._load_attempts(args.out + ".work.jsonl")), 2)
        self.assertIn("consecutive failed attempts", err)
        acc = self.read(args, ".summary.json")["accounting"]
        self.assertEqual((acc["tasks_attempted"], acc["tasks_not_attempted"]), (2, 2))

    def test_resume_retries_failed_tasks_and_keeps_history(self):
        args = self.args(self.write_tasks(tasks("a", "b")))
        with strong_failing_for("b"):
            self.run_collect(args)
        code, _ = self.run_collect(args)
        self.assertEqual(code, 0)
        rows = co._load_attempts(args.out + ".work.jsonl")
        b_rows = [r for r in rows if r["task_id"] == "b"]
        self.assertEqual([(r["attempt_number"], r["status"]) for r in b_rows],
                         [(1, "error"), (2, "complete")])
        self.assertEqual(len(rows), 3)  # "a" was not repeated
        self.assertEqual(len(self.read(args, ".manifest.json")["runs"]), 2)
        acc = self.read(args, ".summary.json")["accounting"]
        self.assertEqual((acc["tasks_completed"], acc["attempts_failed"]), (2, 1))


class TestAccountingSummary(SettingsCase):
    """S2 correction A: failures, skipped strong calls, unresolved reviews and
    usable paired outcomes are all reported; tasks vs attempts stay separate."""

    def _tasks(self):
        rows = tasks("a", "c", "d")
        # Mock answers contain no number -> both tiers need review.
        rows.insert(1, {"task_id": "b", "prompt": "Name a colour (b).",
                        "reference": "4", "grader": "numeric"})
        return rows

    def test_mixed_failures_and_reviews_then_resume(self):
        args = self.args(self.write_tasks(self._tasks()), max_consecutive_failures=0)
        real_cheap = CheapRemoteClient.generate

        def cheap_fails_for_d(self, prompt):
            if "(d)" in prompt:
                raise RuntimeError("local runtime down")
            return real_cheap(self, prompt)

        with strong_failing_for("c"), \
                mock.patch.object(CheapRemoteClient, "generate", cheap_fails_for_d):
            code, _ = self.run_collect(args)
        self.assertEqual(code, 0)
        acc = self.read(args, ".summary.json")["accounting"]
        self.assertEqual((acc["tasks_attempted"], acc["tasks_completed"],
                          acc["tasks_failed_infrastructure"], acc["usable_paired_outcomes"],
                          acc["tasks_unresolved_review"]), (4, 2, 2, 1, 1))
        self.assertEqual(acc["unresolved_review_by_tier"], {"cheap": 1, "strong": 1})
        self.assertEqual(acc["strong_calls"], {
            "completed": 2, "failed": 1, "skipped": 1,
            "skipped_by_reason": {"cheap_tier_failed": 1}, "unit": "attempts"})
        self.assertEqual(acc["failed_attempts_by_tier"], {"strong": 1, "cheap": 1})
        self.assertEqual(acc["quality"]["n"], 1)
        rows = {r["task_id"]: r for r in co._load_attempts(args.out + ".work.jsonl")}
        self.assertEqual((rows["d"]["strong_call"], rows["d"]["strong_skipped_reason"]),
                         ("skipped", "cheap_tier_failed"))
        self.assertEqual(rows["c"]["strong_call"], "failed")
        # The review task is reported but never becomes a training label.
        self.assertEqual([r["task_id"] for r in self.read(args, "")["records"]], ["a"])

        code, out = self.run_collect(args)  # resume: backends healthy
        self.assertEqual(code, 0)
        acc = self.read(args, ".summary.json")["accounting"]
        self.assertEqual((acc["tasks_attempted"], acc["attempts_total"],
                          acc["tasks_completed"], acc["tasks_failed_infrastructure"],
                          acc["usable_paired_outcomes"], acc["tasks_unresolved_review"]),
                         (4, 6, 4, 0, 3, 1))
        self.assertEqual(acc["attempts_failed"], 2)  # history kept after recovery
        self.assertEqual(acc["strong_calls"]["skipped"], 1)
        self.assertEqual(sorted(r["task_id"] for r in self.read(args, "")["records"]),
                         ["a", "c", "d"])

    def test_verdict_resolves_review_into_a_usable_outcome(self):
        args = self.args(self.write_tasks(self._tasks()[:2]))
        self.run_collect(args)
        grades = os.path.join(self.dir, "verdicts.json")
        with open(grades, "w", encoding="utf-8") as fh:
            json.dump({"b": {"cheap": 0.0}}, fh)  # strong side still unresolved
        args.grades = grades
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            co.finalize(args, co._load_existing(args.out + ".work.jsonl"))
        acc = self.read(args, ".summary.json")["accounting"]
        self.assertEqual(acc["unresolved_review_by_tier"], {"cheap": 0, "strong": 1})
        self.assertEqual(acc["usable_paired_outcomes"], 1)


class TestProvenanceRecorded(SettingsCase):
    """S2 correction H: the provenance fields already collected are present."""

    def test_manifest_provenance_fields(self):
        args = self.args(self.write_tasks(tasks("a")))
        self.run_collect(args)
        manifest = self.read(args, ".manifest.json")
        prov = manifest["provenance"]
        for key in ("created_at", "repository_commit", "working_tree_dirty", "tasks_path",
                    "tasks_file_sha256", "mock", "python", "platform", "machine",
                    "processor", "cpu_count", "dependencies", "local_runtime",
                    "strong_revision", "sampling", "split", "cost_assumptions",
                    "budget_usd"):
            self.assertIn(key, prov)
        import platform
        self.assertEqual(prov["python"], platform.python_version())
        self.assertEqual(prov["tasks_file_sha256"], co.file_sha256(args.tasks))
        self.assertEqual(set(prov["dependencies"]), set(co._DEPENDENCIES))
        self.assertEqual(set(prov["local_runtime"]),
                         {"runtime_version", "format", "family", "parameter_size"})
        run = manifest["runs"][0]
        self.assertEqual(set(run), {"run_id", "started_at", "repository_commit",
                                    "local_runtime_version"})
        self.assertIn("informational", manifest["compatibility"]["pair_identity"])


class TestResumeSafety(SettingsCase):
    def _first_run(self):
        args = self.args(self.write_tasks(tasks("a", "b")))
        with strong_failing_for("b"):
            self.run_collect(args)
        with open(args.out + ".work.jsonl", "rb") as fh:
            return args, fh.read()

    def assert_refused(self, args, before, expected):
        code, err = self.run_collect(args)
        self.assertEqual(code, 2)
        self.assertIn(expected, err)
        with open(args.out + ".work.jsonl", "rb") as fh:
            self.assertEqual(fh.read(), before)  # nothing appended

    def test_changed_strong_model_is_refused(self):
        args, before = self._first_run()
        settings.strong_model_name = "accounts/fireworks/models/another-model"
        self.assert_refused(args, before, "strong.model")

    def test_changed_task_file_is_refused(self):
        args, before = self._first_run()
        self.write_tasks(tasks("a", "b", "c"))
        self.assert_refused(args, before, "task file content differs")

    def test_changed_label_definition_is_refused(self):
        args, before = self._first_run()
        args.quality_threshold = 0.9
        self.assert_refused(args, before, "label identity differs")

    def test_work_file_without_manifest_is_refused(self):
        args, before = self._first_run()
        os.remove(args.out + ".manifest.json")
        self.assert_refused(args, before, "has no manifest")

    def test_informational_provenance_does_not_block_resume(self):
        args, _ = self._first_run()
        manifest = self.read(args, ".manifest.json")
        manifest["provenance"].update(processor="another machine", python="0.0",
                                      created_at="2000-01-01T00:00:00")
        co._write_json(args.out + ".manifest.json", manifest)
        code, _ = self.run_collect(args)
        self.assertEqual(code, 0)


class TestMergeIdentity(SettingsCase):
    def test_merge_refuses_different_pair_or_label_identity(self):
        from routing.dataset import OutcomeDataset, OutcomeRecord, save_dataset
        from scripts.merge_outcome_datasets import merge

        def dataset(name, **meta):
            path = os.path.join(self.dir, name)
            rec = OutcomeRecord(task_id=name, prompt=f"prompt {name}",
                                local_quality=1.0, remote_quality=1.0, local_ok=True)
            save_dataset(OutcomeDataset(meta={"local_model": "c", "remote_model": "s",
                                              "quality_threshold": 0.6, **meta},
                                        records=[rec]), path)
            return path

        same = {"pair_identity": {"identity_version": "1", "tier_mode": "remote_pair"},
                "label_identity": {"grader_version": "2", "quality_threshold": 0.6}}
        self.assertEqual(len(merge([dataset("a", **same), dataset("b", **same)]).records), 2)
        for key, value in (("pair_identity", {"identity_version": "1", "tier_mode": "other"}),
                           ("label_identity", {"grader_version": "1", "quality_threshold": 0.6})):
            with self.subTest(key=key), self.assertRaises(ValueError) as ctx:
                merge([dataset("c", **same), dataset(f"d-{key}", **dict(same, **{key: value}))])
            self.assertIn(key, str(ctx.exception))


class TestUtf8(SettingsCase):
    PROMPT = "Café — 東京 ✓ ({})"

    def test_non_ascii_round_trip_is_exact(self):
        from routing.dataset import load_dataset

        data = [{"task_id": "u1", "prompt": self.PROMPT.format("u1"),
                 "reference": "answer to", "grader": "contains"}]
        args = self.args(self.write_tasks(data))
        code, _ = self.run_collect(args)
        self.assertEqual(code, 0)
        dataset = load_dataset(args.out)
        self.assertEqual(dataset.records[0].prompt, self.PROMPT.format("u1"))
        row = co._load_attempts(args.out + ".work.jsonl")[0]
        self.assertEqual(row["prompt"], self.PROMPT.format("u1"))
        for suffix in ("", ".work.jsonl", ".manifest.json", ".summary.json"):
            with open(args.out + suffix, "rb") as fh:
                raw = fh.read()
            raw.decode("utf-8")  # strict
            self.assertNotIn(b"\r\n", raw, suffix)  # LF on every OS

    def test_pipeline_text_opens_declare_utf8(self):
        """Static guard: platform-default encodings broke Windows (S0)."""
        modules = ["routing/dataset.py", "routing/train.py", "routing/identity.py",
                   "evaluation/run.py", "scripts/collect_outcomes.py",
                   "scripts/build_real_pilot.py", "scripts/calibrate.py"]
        offenders = []
        for rel in modules:
            with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), rel)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open"):
                    continue
                mode = node.args[1].value if len(node.args) > 1 and isinstance(
                    node.args[1], ast.Constant) else "r"
                if "b" in str(mode):
                    continue
                if not any(k.arg == "encoding" for k in node.keywords):
                    offenders.append(f"{rel}:{node.lineno}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
