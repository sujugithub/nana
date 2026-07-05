"""Orchestrator: load tasks → route → execute → account → report.

Flow per task (see README for the diagram):

    Router.decide ──▶ local? ──▶ LocalModel.generate ──▶ Router.post_check
                         │                                   ok │ bad
                         │                                      ▼
                         └─ remote? ─────────────▶ RemoteClient.generate
    every step ──▶ TokenTracker (logs/usage.jsonl + summary)

Failure policy — scoring = tokens + accuracy, so an ANSWER always beats no
answer, and one bad task must never kill the run:
- escalation's remote call fails → keep the flagged local answer;
- a remote-routed call fails → fall back to a local attempt (some chance of
  being right beats none);
- anything else per-task → record an error row and continue the run.

Usage:
    python3 main.py --tasks tasks/sample_tasks.json --mock   # offline wiring run
    python3 main.py --tasks tasks/sample_tasks.json          # real models
    python3 main.py --tasks real_tasks.json --threshold 0.7  # calibration sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional

from config import ROUTE_ERROR, ROUTE_LOCAL, settings
from local_model import LocalModel
from remote_client import RemoteClient, RemoteError
from router import Router
from schemas import Task
from token_tracker import TokenTracker


def run_task(
    task: Task,
    router: Router,
    local: LocalModel,
    remote: RemoteClient,
    tracker: TokenTracker,
) -> dict:
    """Run one task through decide → execute → post-check → account.

    Returns a plain dict so results are trivially JSON-serializable. This is
    the function to adapt if the competition hands tasks over a different
    interface (HTTP endpoint, stdin stream, ...): everything above it is I/O,
    everything below it is policy.
    """
    started = time.time()
    decision = router.decide(task)

    local_completion = None
    remote_completion = None
    escalated = False
    problems: List[str] = []

    if decision.target == ROUTE_LOCAL:
        local_completion = local.generate(task.prompt)
        ok, problems = router.post_check(task.prompt, local_completion.text)
        # Draft-and-judge: the local model's own mean token probability.
        # Low self-confidence catches the failure mode post_check's surface
        # rules can't — a fluent, well-formed, WRONG answer. None (mock mode,
        # zero-length output) means "no signal": never treated as low.
        low_confidence = (
            local_completion.confidence is not None
            and local_completion.confidence
            < settings.logprob_confidence_threshold
        )
        if low_confidence:
            problems.append(
                f"low_confidence:{local_completion.confidence:.2f}"
            )
        if (not ok or low_confidence) and settings.enable_escalation:
            # The free local attempt produced something that looks wrong.
            # Pay for a remote retry rather than risk the accuracy penalty —
            # the local attempt itself cost 0 tokens, only latency.
            escalated = True
            try:
                remote_completion = remote.generate(task.prompt)
            except RemoteError as err:
                # A flagged local answer still beats no answer.
                problems.append(f"escalation_failed: {err}")
    else:
        try:
            remote_completion = remote.generate(task.prompt)
        except RemoteError as err:
            # Last resort: a low-confidence local attempt has SOME chance of
            # scoring; an unanswered task has none.
            problems.append(f"remote_failed_local_fallback: {err}")
            local_completion = local.generate(task.prompt)

    final = remote_completion or local_completion
    record = tracker.record(
        task_id=task.task_id,
        route=final.source,
        escalated=escalated,
        local=local_completion,
        remote=remote_completion,
        confidence=decision.confidence,
        threshold=router.threshold,
        signals=decision.signals,
        problems=problems,
        local_confidence=(
            local_completion.confidence if local_completion else None
        ),
        latency_s=time.time() - started,
    )

    return {
        "task_id": task.task_id,
        "route": final.source,
        "escalated": escalated,
        "confidence": decision.confidence,
        "local_confidence": (
            local_completion.confidence if local_completion else None
        ),
        "signals": decision.signals,
        "reason": decision.reason,
        "post_check_problems": problems,
        "billable_tokens": record.billable_tokens,
        "answer": final.text,
    }


def load_tasks(path: str) -> List[Task]:
    """Expected file format: JSON list of {task_id, prompt, metadata?}.

    If kickoff reveals a different format, adapt ONLY this function.
    """
    with open(path) as fh:
        raw = json.load(fh)
    return [
        Task(
            task_id=str(item["task_id"]),
            prompt=item["prompt"],
            # `or {}` (not a .get default): guards explicit "metadata": null
            metadata=item.get("metadata") or {},
        )
        for item in raw
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid token-efficient routing agent"
    )
    parser.add_argument(
        "--tasks",
        default="tasks/sample_tasks.json",
        help="path to a JSON list of {task_id, prompt, metadata?}",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="run without model weights or network (wiring test)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override CONFIDENCE_THRESHOLD for this run (calibration sweeps)",
    )
    args = parser.parse_args(argv)

    # Apply overrides BEFORE constructing components — Router snapshots the
    # threshold at construction time.
    if args.mock:
        settings.mock_mode = True
    if args.threshold is not None:
        settings.confidence_threshold = args.threshold

    tasks = load_tasks(args.tasks)
    router = Router()
    local = LocalModel()
    remote = RemoteClient()
    tracker = TokenTracker()

    if not settings.mock_mode:
        local.load()  # pay the cold-start once, up front, not on task #1

    for task in tasks:
        try:
            result = run_task(task, router, local, remote, tracker)
        except Exception as err:  # one bad task must not kill the scoring run
            print(f"[{task.task_id}] ERROR: {err}", file=sys.stderr)
            tracker.record(task_id=task.task_id, route=ROUTE_ERROR)
            continue
        escalated = " (escalated)" if result["escalated"] else ""
        extra = (
            f" problems={result['post_check_problems']}"
            if result["post_check_problems"]
            else ""
        )
        if result.get("local_confidence") is not None:
            extra = f" local_conf={result['local_confidence']:.2f}" + extra
        preview = result["answer"].replace("\n", " ")[:100]
        print(
            f"[{result['task_id']}] route={result['route']}{escalated} "
            f"conf={result['confidence']:.2f} signals={result['signals']} "
            f"billable={result['billable_tokens']}{extra}\n"
            f"    {preview}"
        )

    tracker.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
