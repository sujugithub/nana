"""Token and estimated API-cost accounting for both routing tiers.

Rules this module encodes:
- In ``remote_pair`` mode both cheap and strong tokens are billable.
- In ``local_remote`` mode cheap-tier local tokens are recorded but do not
  incur API spend; strong-tier tokens remain billable.
- Every task appends one JSON line to logs/usage.jsonl, including the
  routing confidence, the active threshold, the per-signal breakdown, any
  post-check problems, and a per-run run_id.

What the log lets you do after a calibration run:
- separate sweep runs (group lines by run_id / threshold),
- see exactly which tasks would flip local<->remote at a candidate threshold
  (compare each line's confidence against it),
- compute the token savings of LOWERING the threshold — the remote tokens of
  tasks that would flip to local are already recorded.
Raising the threshold still needs a rerun: tasks that would flip to remote
never had their remote cost measured.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from config import ROUTE_ERROR, ROUTE_LOCAL, ROUTE_REMOTE, settings
from schemas import Completion


@dataclass
class UsageRecord:
    task_id: str
    route: str  # backend that produced the FINAL answer (or "error")
    escalated: bool  # local tried first, failed post_check, remote retried
    confidence: float  # router's pre-route confidence for this task
    threshold: float  # threshold active for this run (calibration sweeps)
    signals: Dict[str, float] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)  # post-check / fallbacks
    # The local model's OWN mean token probability for its answer (the
    # draft-and-judge signal) — distinct from `confidence`, which is the
    # router's pre-route heuristic. None when local never ran / mock mode.
    local_confidence: Optional[float] = None
    # Alternative post-gen statistics from the same logits (see schemas.py).
    local_min_token_prob: Optional[float] = None
    local_low_token_frac: Optional[float] = None
    # Which router made the pre-route decision ("heuristic" | "learned"),
    # the artifact version behind a learned decision, and the raw predicted
    # P(local ok) — so every logged task is traceable to its routing policy.
    router: str = "heuristic"
    artifact_version: Optional[str] = None
    p_cheap_ok: Optional[float] = None
    p_local: Optional[float] = None
    tier_mode: str = "remote_pair"
    cheap_model: str = ""
    cheap_provider: str = ""
    strong_model: str = ""
    strong_provider: str = ""
    cheap_prompt_tokens: int = 0
    cheap_completion_tokens: int = 0
    strong_prompt_tokens: int = 0
    strong_completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    # Legacy aliases retained so older calibration scripts and logs remain
    # readable. They mirror cheap_* and strong_* respectively.
    local_prompt_tokens: int = 0
    local_completion_tokens: int = 0
    remote_prompt_tokens: int = 0
    remote_completion_tokens: int = 0
    billable_tokens: int = 0  # all Fireworks-hosted tokens across both tiers
    latency_s: float = 0.0
    run_id: str = ""
    timestamp: float = 0.0


class TokenTracker:
    def __init__(self, log_path: Optional[str] = None):
        # Pass log_path="" to disable file logging (used by the test harness).
        self.log_path = settings.usage_log_path if log_path is None else log_path
        self.records: List[UsageRecord] = []
        # One id per process so sweep runs are separable in the shared file.
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        # record() is called from main.run_all's worker threads.
        self._lock = threading.Lock()

    def record(
        self,
        task_id: str,
        route: str,
        escalated: bool = False,
        local: Optional[Completion] = None,
        remote: Optional[Completion] = None,
        confidence: float = 0.0,
        threshold: float = 0.0,
        signals: Optional[Dict[str, float]] = None,
        problems: Optional[List[str]] = None,
        local_confidence: Optional[float] = None,
        latency_s: float = 0.0,
        local_min_token_prob: Optional[float] = None,
        local_low_token_frac: Optional[float] = None,
        router: str = "heuristic",
        artifact_version: Optional[str] = None,
        p_local: Optional[float] = None,
    ) -> UsageRecord:
        cheap = local
        strong = remote
        cheap_billable = bool(cheap and cheap.provider == "fireworks")
        strong_billable = bool(strong and strong.provider == "fireworks")
        estimated_cost = 0.0
        if cheap_billable:
            estimated_cost += (
                cheap.prompt_tokens * settings.cheap_input_per_mtok
                + cheap.completion_tokens * settings.cheap_output_per_mtok
            ) / 1e6
        if strong_billable:
            estimated_cost += (
                strong.prompt_tokens * settings.strong_input_per_mtok
                + strong.completion_tokens * settings.strong_output_per_mtok
            ) / 1e6
        rec = UsageRecord(
            task_id=task_id,
            route=route,
            escalated=escalated,
            confidence=confidence,
            threshold=threshold,
            signals=signals or {},
            problems=problems or [],
            local_confidence=local_confidence,
            local_min_token_prob=local_min_token_prob,
            local_low_token_frac=local_low_token_frac,
            router=router,
            artifact_version=artifact_version,
            p_cheap_ok=p_local,
            p_local=p_local,
            tier_mode=settings.tier_mode,
            cheap_model=cheap.model_name if cheap else "",
            cheap_provider=cheap.provider if cheap else "",
            strong_model=strong.model_name if strong else "",
            strong_provider=strong.provider if strong else "",
            cheap_prompt_tokens=cheap.prompt_tokens if cheap else 0,
            cheap_completion_tokens=cheap.completion_tokens if cheap else 0,
            strong_prompt_tokens=strong.prompt_tokens if strong else 0,
            strong_completion_tokens=strong.completion_tokens if strong else 0,
            estimated_cost_usd=round(estimated_cost, 9),
            local_prompt_tokens=cheap.prompt_tokens if cheap else 0,
            local_completion_tokens=cheap.completion_tokens if cheap else 0,
            remote_prompt_tokens=strong.prompt_tokens if strong else 0,
            remote_completion_tokens=strong.completion_tokens if strong else 0,
            billable_tokens=(
                (cheap.total_tokens if cheap_billable else 0)
                + (strong.total_tokens if strong_billable else 0)
            ),
            latency_s=round(latency_s, 3),
            run_id=self.run_id,
            timestamp=time.time(),
        )
        with self._lock:
            self.records.append(rec)
            self._append_jsonl(rec)
        return rec

    def _append_jsonl(self, rec: UsageRecord) -> None:
        if not self.log_path:
            return
        directory = os.path.dirname(self.log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(asdict(rec)) + "\n")

    def summary(self) -> dict:
        n = len(self.records)
        final_local = sum(1 for r in self.records if r.route == ROUTE_LOCAL)
        final_remote = sum(1 for r in self.records if r.route == ROUTE_REMOTE)
        errors = sum(1 for r in self.records if r.route == ROUTE_ERROR)
        return {
            "tasks": n,
            "final_local": final_local,
            "final_remote": final_remote,
            "errors": errors,
            "escalations": sum(1 for r in self.records if r.escalated),
            "billable_prompt_tokens": sum(
                (r.cheap_prompt_tokens if r.cheap_provider == "fireworks" else 0)
                + (r.strong_prompt_tokens if r.strong_provider == "fireworks" else 0)
                for r in self.records
            ),
            "billable_completion_tokens": sum(
                (r.cheap_completion_tokens if r.cheap_provider == "fireworks" else 0)
                + (r.strong_completion_tokens if r.strong_provider == "fireworks" else 0)
                for r in self.records
            ),
            "billable_total_tokens": sum(r.billable_tokens for r in self.records),
            "estimated_cost_usd": sum(r.estimated_cost_usd for r in self.records),
            "free_local_tokens": sum(
                r.cheap_prompt_tokens + r.cheap_completion_tokens
                for r in self.records if r.cheap_provider == "local"
            ),
            "local_share": round(final_local / n, 3) if n else 0.0,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("\n──── token usage summary ────")
        print(
            f"tasks: {s['tasks']}  |  cheap-tier answers: {s['final_local']} "
            f"({s['local_share']:.0%})  |  strong-tier: {s['final_remote']}  |  "
            f"escalations: {s['escalations']}  |  errors: {s['errors']}"
        )
        print(
            f"billable API tokens: {s['billable_prompt_tokens']} prompt "
            f"+ {s['billable_completion_tokens']} completion "
            f"= {s['billable_total_tokens']}"
        )
        print(f"estimated API cost:       ${s['estimated_cost_usd']:.6f}")
        if settings.tier_mode == "local_remote":
            print(f"local cheap-tier tokens:  {s['free_local_tokens']}")
