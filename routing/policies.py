"""Operating-threshold selection from validation data.

0.5 is not assumed to mean anything. The router outputs a calibrated
P(cheap-tier answer acceptable); where to cut it depends on the deployment
is optimising, so the cut is chosen by POLICY against the actual
quality/cost outcomes replayed on validation data (routing/simulate.py).

Policies:
    quality_floor   cheapest threshold whose routed quality retains at least
                    `min_quality_retention` of the all-strong quality.
    max_unsafe      highest local utilisation subject to
                    unsafe_local_rate <= `max_unsafe_rate`.
    utility         maximise  quality - cost_weight * cost_mean
                                      - latency_weight * latency_mean.
    remote_rate     hit `target_remote_rate` remote calls as closely as
                    possible (RouteLLM-style "calibrate to a spend budget").

Threshold semantics everywhere in this project: route CHEAP iff
p_local >= threshold (``p_local`` is the schema-1.0 compatibility name).
Candidate thresholds are the observed probabilities plus 0.0 (all cheap) and
slightly above 1.0 (all strong) — sweeping between
observed values cannot change any decision, so this sweep is exhaustive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from routing.dataset import OutcomeRecord
from routing.simulate import SimulationConfig, simulate, summarize

POLICY_NAMES = ("quality_floor", "max_unsafe", "utility", "remote_rate")

ALL_REMOTE_THRESHOLD = 1.000001  # p_local is a probability; nothing reaches this


@dataclass
class PolicyConfig:
    name: str = "quality_floor"
    min_quality_retention: float = 0.97  # quality_floor
    max_unsafe_rate: float = 0.05  # max_unsafe
    cost_weight: float = 1.0  # utility: quality points per unit mean cost
    latency_weight: float = 0.0  # utility: quality points per second
    target_remote_rate: float = 0.5  # remote_rate

    def __post_init__(self) -> None:
        if self.name not in POLICY_NAMES:
            raise ValueError(
                f"unknown policy {self.name!r}; expected one of {POLICY_NAMES}"
            )


@dataclass
class ThresholdChoice:
    threshold: float
    policy: str
    # summarize() dict of the chosen operating point on validation data
    operating_point: Dict[str, float]
    # every swept point, for the training report / Pareto plotting
    sweep: List[Dict[str, float]] = field(default_factory=list)
    note: str = ""


def sweep_thresholds(
    records: Sequence[OutcomeRecord],
    p_local: np.ndarray,
    sim_config: Optional[SimulationConfig] = None,
) -> List[Dict[str, float]]:
    """One summarize() dict (plus 'threshold') per distinct operating point."""
    candidates = sorted(set(np.round(p_local, 6).tolist()))
    candidates = [0.0] + candidates + [ALL_REMOTE_THRESHOLD]
    points = []
    for t in candidates:
        sim = simulate(records, p_local >= t, sim_config)
        point = summarize(sim)
        point["threshold"] = float(t)
        points.append(point)
    return points


def select_threshold(
    records: Sequence[OutcomeRecord],
    p_local: np.ndarray,
    policy: PolicyConfig,
    sim_config: Optional[SimulationConfig] = None,
) -> ThresholdChoice:
    if len(records) == 0:
        raise ValueError("cannot select a threshold from zero validation records")
    sweep = sweep_thresholds(records, p_local, sim_config)
    all_remote = next(p for p in sweep if p["threshold"] >= ALL_REMOTE_THRESHOLD)

    note = ""
    if policy.name == "quality_floor":
        floor = policy.min_quality_retention * all_remote["quality_mean"]
        eligible = [p for p in sweep if p["quality_mean"] >= floor]
        if not eligible:
            chosen, note = all_remote, (
                f"no threshold retains {policy.min_quality_retention:.0%} of "
                f"all-remote quality — falling back to all-remote"
            )
        else:
            chosen = min(eligible, key=lambda p: (p["cost_total"], -p["threshold"]))
    elif policy.name == "max_unsafe":
        eligible = [p for p in sweep if p["unsafe_local_rate"] <= policy.max_unsafe_rate]
        if not eligible:  # all-remote always has unsafe rate 0, so unreachable
            chosen, note = all_remote, "no eligible point (unexpected)"
        else:
            chosen = max(
                eligible, key=lambda p: (p["local_utilisation"], -p["cost_total"])
            )
    elif policy.name == "utility":
        chosen = max(
            sweep,
            key=lambda p: (
                p["quality_mean"]
                - policy.cost_weight * p["cost_mean"]
                - policy.latency_weight * p["latency_mean"]
            ),
        )
    else:  # remote_rate
        chosen = min(
            sweep,
            key=lambda p: (
                abs(p["remote_call_rate"] - policy.target_remote_rate),
                p["cost_total"],
            ),
        )

    operating = {k: v for k, v in chosen.items() if k != "threshold"}
    return ThresholdChoice(
        threshold=float(chosen["threshold"]),
        policy=policy.name,
        operating_point=operating,
        sweep=sweep,
        note=note,
    )
