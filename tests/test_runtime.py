"""run_task cascade behaviour, logging fields, and batch mode — all offline."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

from tests.util import trained_once

import main as main_module
from config import ROUTE_LOCAL, ROUTE_REMOTE, settings
from local_model import LocalModel
from main import _gate_confidence, build_backends, run_task
from remote_client import CheapRemoteClient, RemoteError, StrongRemoteClient
from router import Router
from routing.learned_router import LearnedRouter
from schemas import Completion, Task
from token_tracker import TokenTracker

ALWAYS_LOCAL = 0.0  # heuristic scores are >= 0
ALWAYS_REMOTE = 1.01  # heuristic scores are <= 1


class StubLocal:
    def __init__(self, text="a perfectly fine answer", confidence=0.9,
                 min_token_prob=0.6, low_token_frac=0.1):
        self.completion = Completion(
            text=text,
            prompt_tokens=5,
            completion_tokens=5,
            source=ROUTE_LOCAL,
            confidence=confidence,
            min_token_prob=min_token_prob,
            low_token_frac=low_token_frac,
        )
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return self.completion


class StubRemote:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        if self.fail:
            raise RemoteError("stub remote is down")
        return Completion(
            text="remote answer",
            prompt_tokens=7,
            completion_tokens=9,
            source=ROUTE_REMOTE,
        )


class SettingsCase(unittest.TestCase):
    """Snapshot/restore every setting a test might mutate."""

    _FIELDS = (
        "mock_mode", "tier_mode", "router_mode", "router_artifact_path", "local_conf_stat",
        "logprob_confidence_threshold", "enable_escalation", "usage_log_path",
    )

    def setUp(self):
        self._saved = {f: getattr(settings, f) for f in self._FIELDS}
        settings.usage_log_path = ""  # tests never write logs/usage.jsonl

    def tearDown(self):
        for f, v in self._saved.items():
            setattr(settings, f, v)


class TestCascade(SettingsCase):
    def test_confident_local_answer_stands(self):
        local, remote = StubLocal(confidence=0.9), StubRemote()
        result = run_task(
            Task("t", "hi"), Router(threshold=ALWAYS_LOCAL), local, remote,
            TokenTracker(log_path=""),
        )
        self.assertEqual(result["route"], ROUTE_LOCAL)
        self.assertFalse(result["escalated"])
        self.assertEqual(remote.calls, 0)

    def test_low_mean_confidence_escalates(self):
        settings.local_conf_stat = "mean"
        settings.logprob_confidence_threshold = 0.4
        local, remote = StubLocal(confidence=0.1), StubRemote()
        result = run_task(
            Task("t", "hi"), Router(threshold=ALWAYS_LOCAL), local, remote,
            TokenTracker(log_path=""),
        )
        self.assertTrue(result["escalated"])
        self.assertEqual(result["route"], ROUTE_REMOTE)
        self.assertTrue(
            any(p.startswith("low_confidence:mean") for p in
                result["post_check_problems"])
        )

    def test_gate_stat_min_catches_what_mean_misses(self):
        settings.logprob_confidence_threshold = 0.4
        local = StubLocal(confidence=0.9, min_token_prob=0.05)

        settings.local_conf_stat = "mean"
        result = run_task(
            Task("t", "hi"), Router(threshold=ALWAYS_LOCAL), local,
            StubRemote(), TokenTracker(log_path=""),
        )
        self.assertFalse(result["escalated"])

        settings.local_conf_stat = "min"
        result = run_task(
            Task("t", "hi"), Router(threshold=ALWAYS_LOCAL), local,
            StubRemote(), TokenTracker(log_path=""),
        )
        self.assertTrue(result["escalated"])

    def test_gate_stat_none_disables_gate(self):
        settings.local_conf_stat = "none"
        local = StubLocal(confidence=0.01, min_token_prob=0.01,
                          low_token_frac=0.99)
        result = run_task(
            Task("t", "hi"), Router(threshold=ALWAYS_LOCAL), local,
            StubRemote(), TokenTracker(log_path=""),
        )
        self.assertFalse(result["escalated"])

    def test_gate_stat_low_frac_direction(self):
        settings.local_conf_stat = "low_frac"
        settings.logprob_confidence_threshold = 0.4
        # 80% of tokens low-confidence → gate confidence 0.2 → escalate.
        conf = _gate_confidence(Completion(
            text="x", prompt_tokens=1, completion_tokens=1,
            source=ROUTE_LOCAL, low_token_frac=0.8,
        ))
        self.assertAlmostEqual(conf, 0.2, places=6)

    def test_invalid_gate_stat_raises(self):
        settings.local_conf_stat = "bogus"
        with self.assertRaises(ValueError):
            _gate_confidence(Completion(
                text="x", prompt_tokens=1, completion_tokens=1,
                source=ROUTE_LOCAL,
            ))

    def test_escalation_failure_keeps_flagged_local_answer(self):
        settings.local_conf_stat = "mean"
        settings.logprob_confidence_threshold = 0.4
        local, remote = StubLocal(confidence=0.1), StubRemote(fail=True)
        result = run_task(
            Task("t", "hi"), Router(threshold=ALWAYS_LOCAL), local, remote,
            TokenTracker(log_path=""),
        )
        self.assertEqual(result["route"], ROUTE_LOCAL)
        self.assertTrue(
            any("escalation_failed" in p for p in result["post_check_problems"])
        )

    def test_remote_failure_falls_back_to_local(self):
        local, remote = StubLocal(), StubRemote(fail=True)
        result = run_task(
            Task("t", "hi"), Router(threshold=ALWAYS_REMOTE), local, remote,
            TokenTracker(log_path=""),
        )
        self.assertEqual(result["route"], ROUTE_LOCAL)
        self.assertTrue(
            any("strong_failed_cheap_fallback" in p
                for p in result["post_check_problems"])
        )
        self.assertEqual(local.calls, 1)


class TestLoggingFields(SettingsCase):
    def test_heuristic_decision_logged(self):
        tracker = TokenTracker(log_path="")
        run_task(Task("t", "hi"), Router(threshold=ALWAYS_LOCAL), StubLocal(),
                 StubRemote(), tracker)
        rec = tracker.records[0]
        self.assertEqual(rec.router, "heuristic")
        self.assertIsNone(rec.artifact_version)
        self.assertIsNone(rec.p_local)
        self.assertEqual(rec.local_min_token_prob, 0.6)
        self.assertEqual(rec.local_low_token_frac, 0.1)

    def test_learned_decision_logged_with_artifact_version(self):
        _, (artifact, _, _, _) = trained_once()
        router = LearnedRouter(artifact=artifact)
        tracker = TokenTracker(log_path="")
        run_task(Task("t", "What is the capital of France?"), router,
                 StubLocal(), StubRemote(), tracker)
        rec = tracker.records[0]
        self.assertEqual(rec.router, "learned")
        self.assertEqual(rec.artifact_version, artifact.version)
        self.assertIsNotNone(rec.p_local)
        self.assertEqual(rec.threshold, artifact.threshold)


class TestTierModes(SettingsCase):
    def test_remote_pair_builds_two_fireworks_clients(self):
        settings.tier_mode = "remote_pair"
        mode, cheap, strong = build_backends()
        self.assertEqual(mode, "remote_pair")
        self.assertIsInstance(cheap, CheapRemoteClient)
        self.assertIsInstance(strong, StrongRemoteClient)

    def test_local_remote_keeps_local_backend_available(self):
        settings.tier_mode = "local_remote"
        mode, cheap, strong = build_backends()
        self.assertEqual(mode, "local_remote")
        self.assertIsInstance(cheap, LocalModel)
        self.assertIsInstance(strong, StrongRemoteClient)

    def test_invalid_tier_mode_fails_loudly(self):
        settings.tier_mode = "not-a-mode"
        with self.assertRaises(ValueError):
            build_backends()

    def test_accounting_uses_provider_not_route_name(self):
        tracker = TokenTracker(log_path="")
        local = Completion(
            text="local", prompt_tokens=10, completion_tokens=5,
            source=ROUTE_LOCAL, provider="local", model_name="qwen",
        )
        strong = Completion(
            text="strong", prompt_tokens=20, completion_tokens=10,
            source=ROUTE_REMOTE, provider="fireworks", model_name="pro",
        )
        record = tracker.record("t", ROUTE_REMOTE, local=local, remote=strong)
        self.assertEqual(record.billable_tokens, 30)
        self.assertGreater(record.estimated_cost_usd, 0)

        cheap_remote = Completion(
            text="cheap", prompt_tokens=10, completion_tokens=5,
            source=ROUTE_LOCAL, provider="fireworks", model_name="flash",
        )
        record = tracker.record("u", ROUTE_LOCAL, local=cheap_remote)
        self.assertEqual(record.billable_tokens, 15)

    def test_collector_records_exact_remote_pair_without_network(self):
        from scripts.collect_outcomes import collect

        settings.mock_mode = True
        settings.tier_mode = "remote_pair"
        with tempfile.TemporaryDirectory() as tmp:
            tasks = os.path.join(tmp, "tasks.json")
            out = os.path.join(tmp, "out.json")
            with open(tasks, "w") as fh:
                json.dump([{
                    "task_id": "one",
                    "prompt": "What is 2 + 2?",
                    "reference": "answer to",
                    "grader": "contains",
                }], fh)
            args = SimpleNamespace(
                tasks=tasks,
                out=out,
                grades=None,
                quality_threshold=0.6,
                local_cost_per_second=2e-5,
                cheap_in_per_mtok=settings.cheap_input_per_mtok,
                cheap_out_per_mtok=settings.cheap_output_per_mtok,
                strong_in_per_mtok=settings.strong_input_per_mtok,
                strong_out_per_mtok=settings.strong_output_per_mtok,
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(collect(args), 0)
            with open(out) as fh:
                dataset = json.load(fh)
            with open(out + ".work.jsonl") as fh:
                row = json.loads(fh.readline())

        self.assertEqual(dataset["meta"]["tier_mode"], "remote_pair")
        self.assertEqual(dataset["meta"]["local_model"], settings.cheap_model_name)
        self.assertEqual(dataset["meta"]["remote_model"], settings.strong_model_name)
        self.assertEqual(row["cheap_provider"], "fireworks")
        self.assertEqual(row["strong_provider"], "fireworks")

    def test_numeric_grader_accepts_thousands_separators(self):
        from scripts.collect_outcomes import grade

        self.assertEqual(grade("numeric", "72000", "The answer is 72,000."), 1.0)
        self.assertEqual(grade("numeric", "72,000", "Final: 72000"), 1.0)

    def test_collection_cost_ceiling_counts_only_unfinished_remote_calls(self):
        from scripts.collect_outcomes import configured_api_cost_ceiling

        settings.tier_mode = "local_remote"
        args = SimpleNamespace(
            strong_in_per_mtok=1.0,
            strong_out_per_mtok=2.0,
            cheap_in_per_mtok=3.0,
            cheap_out_per_mtok=4.0,
        )
        tasks = [
            {"task_id": "done", "prompt": "a"},
            {"task_id": "left", "prompt": "b"},
        ]
        local_remote = configured_api_cost_ceiling(tasks, {"done": {}}, args)
        settings.tier_mode = "remote_pair"
        remote_pair = configured_api_cost_ceiling(tasks, {"done": {}}, args)
        self.assertGreater(local_remote, 0)
        self.assertGreater(remote_pair, local_remote)


class TestBatchMode(SettingsCase):
    def _run_batch(self, tasks):
        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "tasks.json")
            out = os.path.join(tmp, "results.json")
            with open(inp, "w") as fh:
                json.dump(tasks, fh)
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main_module.main(["--input", inp, "--output", out, "--mock"])
            with open(out) as fh:
                rows = json.load(fh)
        return code, rows, stderr.getvalue()

    def test_batch_mode_with_learned_router(self):
        _, (artifact, _, _, _) = trained_once()
        from routing.artifact import save_artifact

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.joblib")
            save_artifact(artifact, path)
            settings.router_mode = "learned"
            settings.router_artifact_path = path
            code, rows, stderr = self._run_batch(
                [{"task_id": "a", "prompt": "What is the capital of France?"},
                 {"task_id": "b", "prompt": "Prove that 2 is prime."}]
            )
        self.assertEqual(code, 0)
        self.assertEqual([r["task_id"] for r in rows], ["a", "b"])
        self.assertTrue(all(r["answer"] for r in rows))
        self.assertIn("router: learned", stderr)

    def test_batch_mode_heuristic_unchanged(self):
        settings.router_mode = "heuristic"
        code, rows, _ = self._run_batch(
            [{"task_id": "a", "prompt": "What is the capital of France?"}]
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
