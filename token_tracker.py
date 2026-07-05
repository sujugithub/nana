"""Token accounting — the scoring function, made observable.

Rules this module encodes:
- LOCAL tokens are FREE (count 0 toward the score) but are still recorded, so
  you can see how much work the small model absorbed.
- REMOTE tokens (prompt + completion, from the Fireworks `usage` field) are
  the billable spend.
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
    local_prompt_tokens: int = 0
    local_completion_tokens: int = 0
    remote_prompt_tokens: int = 0
    remote_completion_tokens: int = 0
    billable_tokens: int = 0  # remote prompt + completion; local counts ZERO
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
    ) -> UsageRecord:
        rec = UsageRecord(
            task_id=task_id,
            route=route,
            escalated=escalated,
            confidence=confidence,
            threshold=threshold,
            signals=signals or {},
            problems=problems or [],
            local_confidence=local_confidence,
            local_prompt_tokens=local.prompt_tokens if local else 0,
            local_completion_tokens=local.completion_tokens if local else 0,
            remote_prompt_tokens=remote.prompt_tokens if remote else 0,
            remote_completion_tokens=remote.completion_tokens if remote else 0,
            billable_tokens=remote.total_tokens if remote else 0,
            latency_s=round(latency_s, 3),
            run_id=self.run_id,
            timestamp=time.time(),
        )
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
                r.remote_prompt_tokens for r in self.records
            ),
            "billable_completion_tokens": sum(
                r.remote_completion_tokens for r in self.records
            ),
            "billable_total_tokens": sum(r.billable_tokens for r in self.records),
            "free_local_tokens": sum(
                r.local_prompt_tokens + r.local_completion_tokens
                for r in self.records
            ),
            "local_share": round(final_local / n, 3) if n else 0.0,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("\n──── token usage summary ────")
        print(
            f"tasks: {s['tasks']}  |  answered locally: {s['final_local']} "
            f"({s['local_share']:.0%})  |  remote: {s['final_remote']}  |  "
            f"escalations: {s['escalations']}  |  errors: {s['errors']}"
        )
        print(
            f"billable (remote) tokens: {s['billable_prompt_tokens']} prompt "
            f"+ {s['billable_completion_tokens']} completion "
            f"= {s['billable_total_tokens']}"
        )
        print(f"free (local) tokens:      {s['free_local_tokens']}")
