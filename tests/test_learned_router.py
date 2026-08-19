"""LearnedRouter runtime behaviour + main.build_router mode handling."""
from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest import mock

from tests.util import trained_once

from config import ROUTE_LOCAL, ROUTE_REMOTE, settings
from routing.artifact import save_artifact
from routing.learned_router import LearnedRouter
from schemas import Task


class TestLearnedRouterDecisions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, (cls.artifact, _, _, _) = trained_once()
        cls.router = LearnedRouter(artifact=cls.artifact)

    def test_score_direction_higher_means_local(self):
        # An easy-family prompt must score HIGHER than a hard-family one —
        # the direction contract shared with the heuristic router.
        easy = self.router.decide(Task("e", "What is the capital of France? (case 2)"))
        hard = self.router.decide(
            Task("h", "Prove that the sum of two odd numbers is even, step by step.")
        )
        self.assertGreater(easy.confidence, hard.confidence)

    def test_decision_fields_for_logging(self):
        decision = self.router.decide(Task("x", "What is the capital of Peru?"))
        self.assertEqual(decision.router_kind, "learned")
        self.assertEqual(decision.artifact_version, self.artifact.version)
        self.assertIn("p_local", decision.signals)
        self.assertIn("threshold", decision.reason)

    def test_threshold_boundary_inclusive(self):
        task = Task("b", "Some prompt")
        with mock.patch.object(
            self.router.artifact, "predict_p_local"
        ) as predict:
            import numpy as np

            predict.return_value = np.array([self.router.threshold])
            self.assertEqual(self.router.decide(task).target, ROUTE_LOCAL)
            predict.return_value = np.array([self.router.threshold - 1e-6])
            self.assertEqual(self.router.decide(task).target, ROUTE_REMOTE)

    def test_post_check_inherited(self):
        ok, problems = self.router.post_check("What is 2+2?", "")
        self.assertFalse(ok)
        self.assertIn("empty_or_truncated", problems)

    def test_artifact_loaded_exactly_once(self):
        _, (artifact, _, _, _) = trained_once()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.joblib")
            save_artifact(artifact, path)
            with mock.patch(
                "routing.learned_router.load_artifact",
                side_effect=lambda p: artifact,
            ) as loader:
                router = LearnedRouter(artifact_path=path)
                for _ in range(3):
                    router.decide(Task("x", "hello"))
                self.assertEqual(loader.call_count, 1)


class TestBuildRouter(unittest.TestCase):
    def setUp(self):
        self.saved = (settings.router_mode, settings.router_artifact_path)

    def tearDown(self):
        settings.router_mode, settings.router_artifact_path = self.saved

    def test_heuristic_mode(self):
        import main

        settings.router_mode = "heuristic"
        router = main.build_router()
        self.assertNotIsInstance(router, LearnedRouter)

    def test_learned_mode_loads_artifact(self):
        import main

        _, (artifact, _, _, _) = trained_once()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.joblib")
            save_artifact(artifact, path)
            settings.router_mode = "learned"
            settings.router_artifact_path = path
            with redirect_stderr(io.StringIO()):
                router = main.build_router()
        self.assertIsInstance(router, LearnedRouter)
        self.assertEqual(router.threshold, artifact.threshold)

    def test_learned_mode_missing_artifact_is_fatal(self):
        import main

        settings.router_mode = "learned"
        settings.router_artifact_path = "/nonexistent/r.joblib"
        with self.assertRaises(RuntimeError) as ctx:
            main.build_router()
        self.assertIn("ROUTER_MODE=learned", str(ctx.exception))

    def test_auto_mode_falls_back_with_visible_warning(self):
        import main

        settings.router_mode = "auto"
        settings.router_artifact_path = "/nonexistent/r.joblib"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            router = main.build_router()
        self.assertNotIsInstance(router, LearnedRouter)
        self.assertIn("WARNING", stderr.getvalue())
        self.assertIn("falling back", stderr.getvalue())

    def test_invalid_mode_rejected(self):
        import main

        settings.router_mode = "sometimes"
        with self.assertRaises(ValueError):
            main.build_router()


if __name__ == "__main__":
    unittest.main()
