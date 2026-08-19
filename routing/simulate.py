"""Replay routing decisions against recorded outcomes.

Given per-record routing decisions (local/remote) this module computes what
the SYSTEM would have delivered, using only quantities the dataset actually
recorded — no modelled guesses:

- remote decision:  quality = remote_quality, cost = remote_cost.
- local decision:   the runtime cascade still applies. If the recorded local
  answer would have tripped the post-generation gate (post_check problems, or
  the chosen confidence statistic below the gate threshold), the task
  ESCALATES: quality = remote_quality, cost = local_cost + remote_cost
  (the discarded local attempt is paid for — honest accounting), latency =
  local + remote. Otherwise the local answer stands.
- unsafe-local: the final answer came from the local model AND local_ok is
  False. This is the router's cardinal error and is reported separately.

The same simulation backs threshold selection (routing/policies.py) and the
offline evaluator (evaluation/), so a threshold picked on validation means
exactly what the evaluation later measures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from routing.dataset import OutcomeRecord

# Post-generation confidence statistics the gate can run on. Higher must
# always mean "safer to keep local" — low_token_frac counts LOW-confidence
# tokens, so it is inverted here to preserve that direction.
GATE_STATS = ("mean", "min", "low_frac", "none")


def gate_confidence(rec: OutcomeRecord, stat: str) -> Optional[float]:
    if stat == "mean":
        return rec.local_confidence
    if stat == "min":
        return rec.local_min_token_prob
    if stat == "low_frac":
        if rec.local_low_token_frac is None:
            return None
        return 1.0 - rec.local_low_token_frac
    if stat == "none":
        return None
    raise ValueError(f"unknown gate stat {stat!r}; expected one of {GATE_STATS}")


@dataclass
class SimulationConfig:
    # Mirror of the runtime cascade. escalation=False models a system where
    # local answers are final (used for ablations).
    escalation: bool = True
    gate_stat: str = "mean"
    gate_threshold: float = 0.4
    post_check: bool = True


@dataclass
class SimulationResult:
    """Per-record outcome arrays (aligned with the input records)."""

    route_local: np.ndarray  # bool: pre-router chose local
    final_local: np.ndarray  # bool: the DELIVERED answer is the local one
    escalated: np.ndarray  # bool: local ran, gate tripped, remote retried
    quality: np.ndarray  # float: quality of the delivered answer
    cost: np.ndarray  # float: total cost incl. discarded local attempts
    latency: np.ndarray  # float: end-to-end latency
    unsafe_local: np.ndarray  # bool: delivered local answer with local_ok=False

    @property
    def n(self) -> int:
        return len(self.route_local)


def would_escalate(rec: OutcomeRecord, config: SimulationConfig) -> bool:
    """Replay the runtime post-generation cascade on a recorded local answer."""
    if not config.escalation:
        return False
    if config.post_check and rec.post_check_problems:
        return True
    conf = gate_confidence(rec, config.gate_stat)
    # None = signal not recorded → "no signal", never treated as low
    # (same rule as main.run_task).
    return conf is not None and conf < config.gate_threshold


def simulate(
    records: Sequence[OutcomeRecord],
    route_local: Sequence[bool],
    config: Optional[SimulationConfig] = None,
) -> SimulationResult:
    if len(records) != len(route_local):
        raise ValueError(
            f"records ({len(records)}) and decisions ({len(route_local)}) "
            f"must align"
        )
    config = config or SimulationConfig()
    n = len(records)
    route = np.asarray(route_local, dtype=bool)
    escalated = np.zeros(n, dtype=bool)
    final_local = np.zeros(n, dtype=bool)
    quality = np.zeros(n, dtype=float)
    cost = np.zeros(n, dtype=float)
    latency = np.zeros(n, dtype=float)
    unsafe = np.zeros(n, dtype=bool)

    for i, rec in enumerate(records):
        if route[i]:
            if would_escalate(rec, config):
                escalated[i] = True
                quality[i] = rec.remote_quality
                cost[i] = rec.local_cost + rec.remote_cost
                latency[i] = rec.local_latency_s + rec.remote_latency_s
            else:
                final_local[i] = True
                quality[i] = rec.local_quality
                cost[i] = rec.local_cost
                latency[i] = rec.local_latency_s
                unsafe[i] = not rec.local_ok
        else:
            quality[i] = rec.remote_quality
            cost[i] = rec.remote_cost
            latency[i] = rec.remote_latency_s

    return SimulationResult(
        route_local=route,
        final_local=final_local,
        escalated=escalated,
        quality=quality,
        cost=cost,
        latency=latency,
        unsafe_local=unsafe,
    )


def summarize(sim: SimulationResult) -> dict:
    """Aggregate a simulation into the metric dict every report uses."""
    n = sim.n or 1
    remote_calls = int((~sim.route_local).sum() + sim.escalated.sum())
    return {
        "n": sim.n,
        "quality_mean": float(sim.quality.mean()) if sim.n else 0.0,
        "cost_total": float(sim.cost.sum()),
        "cost_mean": float(sim.cost.mean()) if sim.n else 0.0,
        "latency_mean": float(sim.latency.mean()) if sim.n else 0.0,
        "remote_call_rate": remote_calls / n,
        "local_utilisation": float(sim.final_local.sum()) / n,
        "pre_route_local_rate": float(sim.route_local.sum()) / n,
        "escalation_rate": float(sim.escalated.sum()) / n,
        "unsafe_local_rate": float(sim.unsafe_local.sum()) / n,
        "unsafe_local_count": int(sim.unsafe_local.sum()),
    }
