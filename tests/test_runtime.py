"""run_task cascade behaviour, logging fields, and batch mode — all offline."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from tests.util import trained_once

import main as main_module
from config import ROUTE_LOCAL, ROUTE_REMOTE, settings
from main import _gate_confidence, run_task
from remote_client import RemoteError
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
        "mock_mode", "router_mode", "router_artifact_path", "local_conf_stat",
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
            any("remote_failed_local_fallback" in p
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
