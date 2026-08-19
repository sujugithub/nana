"""Simulation semantics and threshold-selection policies."""
from __future__ import annotations

import unittest

import numpy as np

from tests.util import make_test_dataset  # noqa: F401

from routing.dataset import OutcomeRecord
from routing.policies import (
    ALL_REMOTE_THRESHOLD,
    PolicyConfig,
    select_threshold,
    sweep_thresholds,
)
from routing.simulate import SimulationConfig, gate_confidence, simulate, summarize


def _rec(tid, local_q, remote_q, conf=0.9, problems=None, local_cost=0.1,
         remote_cost=1.0):
    return OutcomeRecord(
        task_id=tid,
        prompt=f"prompt {tid}",
        group_id=f"g-{tid}",
        local_quality=local_q,
        remote_quality=remote_q,
        local_ok=local_q >= 0.6,
        local_cost=local_cost,
        remote_cost=remote_cost,
        local_latency_s=1.0,
        remote_latency_s=3.0,
        local_confidence=conf,
        local_min_token_prob=conf * 0.5,
        local_low_token_frac=1 - conf,
        post_check_problems=problems or [],
    )


class TestSimulate(unittest.TestCase):
    def test_remote_decision_uses_remote_outcome(self):
        sim = simulate([_rec("a", 0.2, 0.9)], [False])
        self.assertAlmostEqual(sim.quality[0], 0.9)
        self.assertAlmostEqual(sim.cost[0], 1.0)
        self.assertFalse(sim.unsafe_local[0])

    def test_local_decision_uses_local_outcome(self):
        sim = simulate([_rec("a", 0.8, 0.9)], [True])
        self.assertAlmostEqual(sim.quality[0], 0.8)
        self.assertAlmostEqual(sim.cost[0], 0.1)
        self.assertTrue(sim.final_local[0])

    def test_unsafe_local_flagged(self):
        sim = simulate([_rec("a", 0.2, 0.9)], [True])
        self.assertTrue(sim.unsafe_local[0])

    def test_low_confidence_escalates_and_pays_both(self):
        rec = _rec("a", 0.2, 0.9, conf=0.1)
        sim = simulate([rec], [True], SimulationConfig(gate_threshold=0.4))
        self.assertTrue(sim.escalated[0])
        self.assertAlmostEqual(sim.quality[0], 0.9)
        self.assertAlmostEqual(sim.cost[0], 1.1)  # discarded local attempt paid
        self.assertAlmostEqual(sim.latency[0], 4.0)
        self.assertFalse(sim.unsafe_local[0])

    def test_post_check_problems_escalate(self):
        rec = _rec("a", 0.2, 0.9, conf=0.99, problems=["hedging_or_refusal"])
        sim = simulate([rec], [True])
        self.assertTrue(sim.escalated[0])

    def test_escalation_disabled(self):
        rec = _rec("a", 0.2, 0.9, conf=0.1, problems=["hedging_or_refusal"])
        sim = simulate([rec], [True], SimulationConfig(escalation=False))
        self.assertFalse(sim.escalated[0])
        self.assertTrue(sim.unsafe_local[0])

    def test_missing_signal_never_treated_as_low(self):
        rec = _rec("a", 0.8, 0.9)
        rec.local_confidence = None
        sim = simulate([rec], [True], SimulationConfig(gate_stat="mean"))
        self.assertFalse(sim.escalated[0])

    def test_gate_stat_direction(self):
        rec = _rec("a", 0.8, 0.9, conf=0.9)
        # low_frac = 0.1 → gate confidence 0.9: higher = safer for ALL stats
        self.assertAlmostEqual(gate_confidence(rec, "mean"), 0.9)
        self.assertAlmostEqual(gate_confidence(rec, "min"), 0.45)
        self.assertAlmostEqual(gate_confidence(rec, "low_frac"), 0.9, places=6)
        self.assertIsNone(gate_confidence(rec, "none"))

    def test_unknown_gate_stat_raises(self):
        with self.assertRaises(ValueError):
            gate_confidence(_rec("a", 0.8, 0.9), "median")

    def test_misaligned_inputs_raise(self):
        with self.assertRaises(ValueError):
            simulate([_rec("a", 0.5, 0.9)], [True, False])

    def test_summary_rates(self):
        recs = [_rec("a", 0.8, 0.9), _rec("b", 0.2, 0.9), _rec("c", 0.9, 0.9)]
        s = summarize(simulate(recs, [True, False, True]))
        self.assertAlmostEqual(s["local_utilisation"], 2 / 3)
        self.assertAlmostEqual(s["remote_call_rate"], 1 / 3)
        self.assertEqual(s["unsafe_local_count"], 0)


class TestThresholdPolicies(unittest.TestCase):
    def setUp(self):
        # Four rows; p_local separates good from bad locals perfectly.
        self.records = [
            _rec("good1", 0.9, 0.92),
            _rec("good2", 0.85, 0.92),
            _rec("bad1", 0.2, 0.92),
            _rec("bad2", 0.1, 0.92),
        ]
        self.p = np.array([0.9, 0.8, 0.3, 0.2])

    def test_threshold_boundary_is_inclusive(self):
        # p == threshold must route LOCAL (>= semantics).
        sim = simulate(self.records, self.p >= 0.8)
        self.assertTrue(sim.route_local[1])
        self.assertFalse(sim.route_local[2])

    def test_sweep_covers_all_operating_points(self):
        points = sweep_thresholds(self.records, self.p)
        rates = {p["pre_route_local_rate"] for p in points}
        self.assertIn(0.0, rates)  # all-remote endpoint
        self.assertIn(1.0, rates)  # all-local endpoint

    def test_max_unsafe_policy(self):
        choice = select_threshold(
            self.records, self.p, PolicyConfig(name="max_unsafe", max_unsafe_rate=0.0)
        )
        # Best zero-unsafe point keeps both good locals: threshold <= 0.8, > 0.3.
        self.assertLessEqual(choice.threshold, 0.8)
        self.assertGreater(choice.threshold, 0.3)
        self.assertEqual(choice.operating_point["unsafe_local_rate"], 0.0)
        self.assertAlmostEqual(choice.operating_point["local_utilisation"], 0.5)

    def test_quality_floor_policy_prefers_cheapest_eligible(self):
        choice = select_threshold(
            self.records,
            self.p,
            PolicyConfig(name="quality_floor", min_quality_retention=0.9),
        )
        op = choice.operating_point
        floor = 0.9 * 0.92
        self.assertGreaterEqual(op["quality_mean"], floor)
        # Cheapest eligible point routes the two good rows local.
        self.assertAlmostEqual(op["local_utilisation"], 0.5)

    def test_quality_floor_at_full_retention_goes_all_remote(self):
        # Local answers here are strictly worse than remote, so retaining
        # 100% of all-remote quality leaves only the all-remote point.
        choice = select_threshold(
            self.records,
            self.p,
            PolicyConfig(name="quality_floor", min_quality_retention=1.0),
        )
        self.assertGreaterEqual(choice.threshold, ALL_REMOTE_THRESHOLD)
        self.assertEqual(choice.operating_point["remote_call_rate"], 1.0)

    def test_remote_rate_policy(self):
        choice = select_threshold(
            self.records,
            self.p,
            PolicyConfig(name="remote_rate", target_remote_rate=0.5),
        )
        self.assertAlmostEqual(
            choice.operating_point["remote_call_rate"], 0.5, delta=0.26
        )

    def test_utility_policy_zero_cost_weight_wants_quality(self):
        choice = select_threshold(
            self.records,
            self.p,
            PolicyConfig(name="utility", cost_weight=0.0),
        )
        best_quality = max(p["quality_mean"] for p in choice.sweep)
        self.assertAlmostEqual(
            choice.operating_point["quality_mean"], best_quality
        )

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            PolicyConfig(name="vibes")


if __name__ == "__main__":
    unittest.main()
