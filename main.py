"""Orchestrator: load tasks → route → execute → account → report.

Flow per task (see README for the diagram):

    Router.decide ──▶ local? ──▶ LocalModel.generate ──▶ Router.post_check
                         │                                   ok │ bad
                         │                                      ▼
                         └─ remote? ─────────────▶ RemoteClient.generate
    every step ──▶ TokenTracker (logs/usage.jsonl + summary)

Failure policy — an ANSWER always beats no answer, and one bad task must
never kill the run:
- escalation's remote call fails → keep the flagged local answer;
- a remote-routed call fails → fall back to a local attempt (some chance of
  being right beats none);
- anything else per-task → record an error row and continue the run.

Usage:
    python3 main.py --tasks tasks/sample_tasks.json --mock   # offline wiring run
    python3 main.py --tasks tasks/sample_tasks.json          # real models
    python3 main.py --tasks real_tasks.json --threshold 0.7  # calibration sweep
    python3 main.py --input /input/tasks.json --output /output/results.json
                                                             # batch mode

Batch contract: read [{task_id, prompt}] from --input, write [{task_id, answer}]
valid JSON to --output — ALWAYS, even on partial failure (malformed or missing
output is unusable; a blank answer loses only that task). All tasks run on a
thread pool (REMOTE_CONCURRENCY workers) with a global deadline
(RUN_DEADLINE_S) so results land before any wall-clock limit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Dict, List, Optional, Tuple

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
    the function to adapt if tasks arrive over a different
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
            # being right; an unanswered task has none.
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

    If the task format differs, adapt ONLY this function.
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


def _report(result: dict) -> None:
    """One progress line per finished task (stdout → container logs)."""
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


def run_all(
    tasks: List[Task],
    router: Router,
    local: LocalModel,
    remote: RemoteClient,
    tracker: TokenTracker,
    deadline: float,
) -> Tuple[Dict[str, dict], bool]:
    """Run every task through run_task on a thread pool.

    Remote calls (~tens of seconds each) overlap up to REMOTE_CONCURRENCY;
    local generation serializes on LocalModel's internal lock, so a worker
    doing local work never blocks the remote ones. Collection stops at
    `deadline` (time.monotonic() seconds): whatever finished is returned,
    the rest is abandoned so the caller can still write results in time.

    Returns ({task_id: result_dict}, deadline_hit).
    """
    results: Dict[str, dict] = {}

    def _guarded(task: Task) -> dict:
        try:
            return run_task(task, router, local, remote, tracker)
        except Exception as err:  # one bad task must not kill the whole run
            print(f"[{task.task_id}] ERROR: {err}", file=sys.stderr)
            tracker.record(task_id=task.task_id, route=ROUTE_ERROR)
            return {
                "task_id": task.task_id,
                "route": ROUTE_ERROR,
                "escalated": False,
                "confidence": 0.0,
                "local_confidence": None,
                "signals": {},
                "reason": "task crashed",
                "post_check_problems": [f"error: {err}"],
                "billable_tokens": 0,
                "answer": "",
            }

    pool = ThreadPoolExecutor(max_workers=max(1, settings.remote_concurrency))
    futures = {pool.submit(_guarded, task): task for task in tasks}
    pending = set(futures)
    try:
        for fut in as_completed(
            futures, timeout=max(0.0, deadline - time.monotonic())
        ):
            pending.discard(fut)
            result = fut.result()  # _guarded never raises
            results[result["task_id"]] = result
            _report(result)
    except FuturesTimeoutError:
        # Harvest tasks that finished but weren't yielded before the timeout
        # — a computed answer must never be dropped at the deadline boundary.
        for fut in [f for f in pending if f.done()]:
            pending.discard(fut)
            result = fut.result()
            results[result["task_id"]] = result
            _report(result)
        abandoned = sorted(futures[f].task_id for f in pending)
        print(
            f"DEADLINE ({settings.run_deadline_s:.0f}s): abandoning "
            f"{len(abandoned)} unfinished task(s): {abandoned}",
            file=sys.stderr,
        )
    # No waiting: a worker may sit in a remote read for up to
    # REQUEST_TIMEOUT_S, and the results file must be written NOW.
    pool.shutdown(wait=False, cancel_futures=True)
    return results, bool(pending)


def write_results(path: str, tasks: List[Task], results: Dict[str, dict]) -> None:
    """Write the harness contract file: [{task_id, answer}] for EVERY input
    task, in input order. Tasks that never finished get "" — a blank answer
    loses one task; a missing/malformed file scores zero for the whole run."""
    payload = [
        {
            "task_id": task.task_id,
            "answer": (results.get(task.task_id) or {}).get("answer") or "",
        }
        for task in tasks
    ]
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, path)  # atomic: never leave a half-written results.json


def main(argv: Optional[List[str]] = None) -> int:
    started = time.monotonic()  # the 10-min cap counts model load too
    parser = argparse.ArgumentParser(
        description="Hybrid token-efficient routing agent"
    )
    parser.add_argument(
        "--tasks",
        default="tasks/sample_tasks.json",
        help="path to a JSON list of {task_id, prompt, metadata?} (dev mode)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="harness mode: tasks file, e.g. /input/tasks.json (overrides --tasks)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="harness mode: write [{task_id, answer}] JSON here — always "
        "written, even on partial failure",
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

    harness = args.output is not None
    deadline = started + settings.run_deadline_s

    try:
        tasks = load_tasks(args.input or args.tasks)
    except Exception as err:
        print(f"FATAL: cannot read tasks: {err}", file=sys.stderr)
        if harness:
            try:  # still leave valid JSON behind — never a missing file
                write_results(args.output, [], {})
            except Exception:
                pass
        return 1

    exit_code = 0
    results: Dict[str, dict] = {}
    pending = False
    try:
        router = Router()
        local = LocalModel()
        remote = RemoteClient()
        tracker = TokenTracker()
        if not settings.mock_mode:
            local.load()  # pay the cold-start once, up front, not on task #1
        results, pending = run_all(tasks, router, local, remote, tracker, deadline)
        tracker.print_summary()
    except Exception as err:
        # Belt and braces: nothing above should raise (run_all guards each
        # task), but a valid results file must be written regardless.
        print(f"FATAL: run aborted: {err}", file=sys.stderr)
        exit_code = 1

    if harness:
        try:
            write_results(args.output, tasks, results)
            print(f"wrote {len(tasks)} answers to {args.output}")
        except Exception as err:
            print(f"FATAL: cannot write results: {err}", file=sys.stderr)
            exit_code = 1

    if pending:
        # Abandoned workers are non-daemon threads, possibly blocked in a
        # remote read; a normal exit would join them and blow the runtime
        # cap. Results are on disk — leave immediately.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
