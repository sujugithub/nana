"""Metric implementations and the offline evaluator's guarantees."""
from __future__ import annotations

import unittest

import numpy as np

from tests.util import make_test_dataset, trained_once

from routing.metrics import (
    bootstrap_ci,
    classification_metrics,
    confusion_at_threshold,
    expected_calibration_error,
    paired_bootstrap_diff,
)
from routing.splits import make_splits
from evaluation.run import evaluate, mixture_report


class TestMetrics(unittest.TestCase):
    def test_perfect_classifier(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        m = classification_metrics(y, p)
        self.assertEqual(m["roc_auc"], 1.0)
        self.assertEqual(m["pr_auc_local_ok"], 1.0)
        self.assertEqual(m["pr_auc_local_fail"], 1.0)
        self.assertLess(m["brier"], 0.05)

    def test_single_class_reports_none_not_fake_numbers(self):
        m = classification_metrics(np.array([1, 1]), np.array([0.7, 0.8]))
        self.assertIsNone(m["roc_auc"])

    def test_ece_perfectly_calibrated_bins(self):
        # 100 items at p=0.7 with a 70% positive rate → ECE ~0.
        y = np.array([1] * 70 + [0] * 30)
        p = np.full(100, 0.7)
        self.assertAlmostEqual(expected_calibration_error(y, p), 0.0, places=9)

    def test_ece_maximally_miscalibrated(self):
        y = np.zeros(50)
        p = np.full(50, 0.99)
        self.assertGreater(expected_calibration_error(y, p), 0.9)

    def test_confusion_cells(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.4, 0.9, 0.1])
        c = confusion_at_threshold(y, p, 0.5)
        self.assertEqual(c, {"tp": 1, "fp": 1, "fn": 1, "tn": 1})

    def test_bootstrap_ci_brackets_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(5.0, 1.0, size=500)
        point, lo, hi = bootstrap_ci(values, seed=1)
        self.assertLess(lo, point)
        self.assertGreater(hi, point)
        self.assertAlmostEqual(point, 5.0, delta=0.2)

    def test_paired_diff_detects_constant_shift(self):
        rng = np.random.default_rng(0)
        b = rng.normal(0, 1, size=300)
        a = b + 0.5
        d = paired_bootstrap_diff(a, b, seed=1)
        self.assertAlmostEqual(d["mean_diff"], 0.5, places=9)
        self.assertGreater(d["ci_lo"], 0.49)


class TestEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset, (cls.artifact, _, cls.splits, cls.candidates) = trained_once()
        cls.test_records = [
            r for r in cls.dataset.records
            if cls.splits.assignment[r.task_id] == "test"
        ]
        cls.results = evaluate(
            cls.test_records,
            {cls.artifact.model_name: cls.artifact},
            selected_name=cls.artifact.model_name,
            heuristic_threshold=0.55,
            seed=5,
        )

    def test_all_required_systems_present(self):
        for name in ("all_local", "all_remote", "heuristic", "oracle",
                     self.artifact.model_name):
            self.assertIn(name, self.results["systems"])
        self.assertTrue(self.results["random_curve"])

    def test_oracle_never_unsafe_and_cheapest_safe(self):
        oracle = self.results["systems"]["oracle"]
        self.assertEqual(oracle["unsafe_local_rate"], 0.0)
        self.assertLessEqual(
            oracle["cost_total"], self.results["systems"]["all_remote"]["cost_total"]
        )

    def test_all_remote_has_zero_unsafe_and_full_remote_rate(self):
        ar = self.results["systems"]["all_remote"]
        self.assertEqual(ar["unsafe_local_rate"], 0.0)
        self.assertEqual(ar["remote_call_rate"], 1.0)
        self.assertEqual(ar["quality_regret_vs_all_remote"], 0.0)

    def test_all_local_never_calls_remote(self):
        al = self.results["systems"]["all_local"]
        self.assertEqual(al["remote_call_rate"], 0.0)
        self.assertEqual(al["local_utilisation"], 1.0)

    def test_random_curve_endpoints(self):
        curve = self.results["random_curve"]
        p0 = next(c for c in curve if c["p_remote_pre_route"] == 0.0)
        p1 = next(c for c in curve if c["p_remote_pre_route"] == 1.0)
        self.assertEqual(p1["remote_call_rate"], 1.0)
        self.assertLess(p0["remote_call_rate"], p1["remote_call_rate"])

    def test_learned_curve_dominates_random_line_mid_range(self):
        # The test dataset has a clean lexical signal by construction, so the
        # learned router's threshold sweep must sit ABOVE the random line
        # through the mid-range remote rates (at the extremes every router
        # converges to all-local/all-remote and there is nothing to win).
        from evaluation.run import _interp_random

        curve = self.results["random_curve"]
        sweep = self.results["sweeps"][self.artifact.model_name]
        mid = [p for p in sweep if 0.2 <= p["remote_call_rate"] <= 0.8]
        self.assertTrue(mid, "sweep has no mid-range operating points")
        advantages = [
            p["quality_mean"]
            - _interp_random(curve, p["remote_call_rate"], "quality_mean")
            for p in mid
        ]
        self.assertGreater(sum(advantages) / len(advantages), 0.0)

    def test_bootstrap_cis_present(self):
        sel = self.results["systems"][self.artifact.model_name]
        self.assertEqual(len(sel["quality_ci"]), 2)
        self.assertLessEqual(sel["quality_ci"][0], sel["quality_mean"])
        self.assertGreaterEqual(sel["quality_ci"][1], sel["quality_mean"])

    def test_per_category_and_mixtures(self):
        per_cat = self.results["per_category"][self.artifact.model_name]
        self.assertEqual(set(per_cat), {"easy", "hard"})
        mixtures = self.results["mixtures"][self.artifact.model_name]
        self.assertEqual(set(mixtures), {"balanced", "easy_heavy", "hard_heavy"})
        self.assertGreaterEqual(
            mixtures["easy_heavy"]["quality_mean"], 0.0
        )

    def test_mixture_report_weighting(self):
        per_cat = {
            "easy": {"quality_mean": 1.0, "cost_mean": 0.0,
                     "unsafe_local_rate": 0.0, "remote_call_rate": 0.0},
            "hard": {"quality_mean": 0.0, "cost_mean": 1.0,
                     "unsafe_local_rate": 1.0, "remote_call_rate": 1.0},
        }
        out = mixture_report(per_cat, hard_categories=["hard"])
        self.assertAlmostEqual(out["balanced"]["quality_mean"], 0.5)
        self.assertAlmostEqual(out["easy_heavy"]["quality_mean"], 0.8)
        self.assertAlmostEqual(out["hard_heavy"]["quality_mean"], 0.2)

    def test_empty_split_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([], {}, selected_name="x", heuristic_threshold=0.5)


if __name__ == "__main__":
    unittest.main()
