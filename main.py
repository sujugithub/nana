"""Orchestrator: load tasks → route → execute → account → report.

Flow per task (see README for the diagram):

    Router.decide ──▶ cheap? ──▶ selected cheap backend ──▶ post_check
                         │                                ok │ bad
                         │                                   ▼
                         └─ strong? ─────────────▶ StrongRemoteClient

    every step ──▶ TokenTracker (logs/usage.jsonl + summary)

Failure policy — an ANSWER always beats no answer, and one bad task must
never kill the run:

- escalation's strong call fails → keep the flagged cheap-tier answer;
- a strong-routed call fails → fall back to a cheap-tier attempt
  (some chance of being right beats none);
- anything else per-task → record an error row and continue the run.

Usage:

    python3 main.py --tasks tasks/sample_tasks.json --mock
    python3 main.py --tasks tasks/sample_tasks.json
    python3 main.py --tasks real_tasks.json --threshold 0.7
    python3 main.py --input /input/tasks.json --output /output/results.json

Batch contract: read [{task_id, prompt}] from --input,
write [{task_id, answer}] valid JSON to --output — ALWAYS,
even on partial failure.

All tasks run on a thread pool (REMOTE_CONCURRENCY workers)
with a global deadline (RUN_DEADLINE_S).
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
from remote_client import (
    CheapRemoteClient,
    RemoteError,
    StrongRemoteClient,
)
from router import Router
from schemas import Completion, Task
from token_tracker import TokenTracker


def build_router(
    threshold: Optional[float] = None,
) -> Router:
    """
    Construct the pre-router for ROUTER_MODE.

    Modes:
    - heuristic
    - learned
    - auto
    """

    mode = settings.router_mode.strip().lower()

    if mode == "heuristic":
        return Router(threshold=threshold)

    if mode not in ("learned", "auto"):
        raise ValueError(
            f"ROUTER_MODE={settings.router_mode!r} invalid: expected "
            f"heuristic | learned | auto"
        )

    try:
        # Deferred import so heuristic mode works without
        # sklearn/joblib installed.
        from routing.learned_router import LearnedRouter

        router = LearnedRouter(
            artifact_path=settings.router_artifact_path
        )

        print(
            f"router: learned "
            f"(artifact {router.artifact.version}, "
            f"threshold {router.threshold:.3f})",
            file=sys.stderr,
        )

        return router

    except Exception as err:
        if mode == "learned":
            raise RuntimeError(
                "ROUTER_MODE=learned but the artifact "
                f"is unusable: {err}"
            ) from err

        print(
            "WARNING: ROUTER_MODE=auto — "
            f"learned router unavailable ({err}); "
            "falling back to heuristic rules",
            file=sys.stderr,
        )

        return Router(threshold=threshold)


def build_backends():
    """
    Build configured cheap and strong execution tiers.

    remote_pair:
        cheap Fireworks + strong Fireworks

    local_remote:
        LocalModel + strong Fireworks
    """

    tier_mode = settings.tier_mode.strip().lower()

    if tier_mode == "remote_pair":
        cheap = CheapRemoteClient()

    elif tier_mode == "local_remote":
        cheap = LocalModel()

    else:
        raise ValueError(
            f"TIER_MODE={settings.tier_mode!r} invalid: expected "
            "remote_pair | local_remote"
        )

    return (
        tier_mode,
        cheap,
        StrongRemoteClient(),
    )


def _gate_confidence(
    completion: Completion,
) -> Optional[float]:
    """
    Return the post-generation confidence value used
    by the escalation gate.
    """

    stat = settings.local_conf_stat.strip().lower()

    if stat == "mean":
        return completion.confidence

    if stat == "min":
        return completion.min_token_prob

    if stat == "low_frac":
        if completion.low_token_frac is None:
            return None

        return 1.0 - completion.low_token_frac

    if stat == "none":
        return None

    raise ValueError(
        f"LOCAL_CONF_STAT={settings.local_conf_stat!r} invalid: expected "
        "mean | min | low_frac | none"
    )


def run_task(
    task: Task,
    router: Router,
    cheap,
    strong,
    tracker: TokenTracker,
) -> dict:
    """
    Run one task through:

        decide
        → execute
        → post-check
        → account

    Returns a plain JSON-serializable dictionary.

    This function can also be reused by HTTP/chat interfaces.
    """

    started = time.time()

    decision = router.decide(task)

    cheap_completion = None
    strong_completion = None

    escalated = False
    problems: List[str] = []

    # ---------------------------------------------------------
    # Router selected cheap/local tier
    # ---------------------------------------------------------

    if decision.target == ROUTE_LOCAL:
        cheap_completion = cheap.generate(
            task.prompt
        )

        ok, problems = router.post_check(
            task.prompt,
            cheap_completion.text,
        )

        gate_conf = _gate_confidence(
            cheap_completion
        )

        low_confidence = (
            gate_conf is not None
            and gate_conf
            < settings.logprob_confidence_threshold
        )

        if low_confidence:
            problems.append(
                "low_confidence:"
                f"{settings.local_conf_stat}:"
                f"{gate_conf:.2f}"
            )

        if (
            not ok or low_confidence
        ) and settings.enable_escalation:

            escalated = True

            try:
                strong_completion = strong.generate(
                    task.prompt
                )

            except RemoteError as err:
                # Keep the cheap answer if the
                # escalation request fails.
                problems.append(
                    f"escalation_failed: {err}"
                )

    # ---------------------------------------------------------
    # Router selected strong tier
    # ---------------------------------------------------------

    else:
        try:
            strong_completion = strong.generate(
                task.prompt
            )

        except RemoteError as err:
            # If strong remote fails, attempt cheap tier
            # rather than returning no answer.
            problems.append(
                f"strong_failed_cheap_fallback: {err}"
            )

            cheap_completion = cheap.generate(
                task.prompt
            )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    final = (
        strong_completion
        or cheap_completion
    )

    if final is None:
        raise RuntimeError(
            "No model backend produced a completion"
        )

    # Calculate ONCE so the value recorded by TokenTracker
    # and returned to the API is consistent.
    latency_s = time.time() - started

    record = tracker.record(
        task_id=task.task_id,
        route=final.source,
        escalated=escalated,
        local=cheap_completion,
        remote=strong_completion,
        confidence=decision.confidence,
        threshold=router.threshold,
        signals=decision.signals,
        problems=problems,

        local_confidence=(
            cheap_completion.confidence
            if cheap_completion
            else None
        ),

        latency_s=latency_s,

        local_min_token_prob=(
            cheap_completion.min_token_prob
            if cheap_completion
            else None
        ),

        local_low_token_frac=(
            cheap_completion.low_token_frac
            if cheap_completion
            else None
        ),

        router=decision.router_kind,

        artifact_version=(
            decision.artifact_version
        ),

        p_local=(
            decision.confidence
            if decision.router_kind == "learned"
            else None
        ),
    )

    return {
        "task_id": task.task_id,

        "route": final.source,

        "escalated": escalated,

        "confidence": (
            decision.confidence
        ),

        "router": (
            decision.router_kind
        ),

        "artifact_version": (
            decision.artifact_version
        ),

        "local_confidence": (
            cheap_completion.confidence
            if cheap_completion
            else None
        ),

        "signals": (
            decision.signals
        ),

        "reason": (
            decision.reason
        ),

        "post_check_problems": (
            problems
        ),

        "billable_tokens": (
            record.billable_tokens
        ),

        "estimated_cost_usd": (
            record.estimated_cost_usd
        ),

        "model_name": (
            final.model_name
        ),

        "provider": (
            final.provider
        ),

        # Added for chat persistence and routing details UI.
        "latency_s": latency_s,

        "answer": (
            final.text
        ),
    }


def load_tasks(
    path: str,
) -> List[Task]:
    """
    Expected file format:

    [
        {
            "task_id": "...",
            "prompt": "...",
            "metadata": {}
        }
    ]
    """

    with open(
        path,
        encoding="utf-8",
    ) as fh:
        raw = json.load(fh)

    return [
        Task(
            task_id=str(
                item["task_id"]
            ),
            prompt=item["prompt"],
            metadata=(
                item.get("metadata")
                or {}
            ),
        )
        for item in raw
    ]


def _report(
    result: dict,
) -> None:
    """
    Print one progress line per finished task.
    """

    escalated = (
        " (escalated)"
        if result["escalated"]
        else ""
    )

    extra = (
        f" problems="
        f"{result['post_check_problems']}"
        if result["post_check_problems"]
        else ""
    )

    if (
        result.get(
            "local_confidence"
        )
        is not None
    ):
        extra = (
            f" local_conf="
            f"{result['local_confidence']:.2f}"
            + extra
        )

    preview = (
        result["answer"]
        .replace("\n", " ")[:100]
    )

    print(
        f"[{result['task_id']}] "
        f"route={result['route']}"
        f"{escalated} "
        f"conf={result['confidence']:.2f} "
        f"signals={result['signals']} "
        f"billable="
        f"{result['billable_tokens']}"
        f"{extra}\n"
        f"    {preview}"
    )


def run_all(
    tasks: List[Task],
    router: Router,
    cheap,
    strong,
    tracker: TokenTracker,
    deadline: float,
) -> Tuple[Dict[str, dict], bool]:
    """
    Run tasks concurrently.

    Returns:
        (
            results_by_task_id,
            deadline_hit,
        )
    """

    results: Dict[str, dict] = {}

    def _guarded(
        task: Task,
    ) -> dict:
        try:
            return run_task(
                task,
                router,
                cheap,
                strong,
                tracker,
            )

        except Exception as err:
            print(
                f"[{task.task_id}] "
                f"ERROR: {err}",
                file=sys.stderr,
            )

            tracker.record(
                task_id=task.task_id,
                route=ROUTE_ERROR,
            )

            # Keep same basic response shape as successful tasks.
            return {
                "task_id": task.task_id,

                "route": ROUTE_ERROR,

                "escalated": False,

                "confidence": 0.0,

                "router": "error",

                "artifact_version": None,

                "local_confidence": None,

                "signals": {},

                "reason": "task crashed",

                "post_check_problems": [
                    f"error: {err}"
                ],

                "billable_tokens": 0,

                "estimated_cost_usd": 0.0,

                "model_name": "",

                "provider": "",

                "latency_s": 0.0,

                "answer": "",
            }

    pool = ThreadPoolExecutor(
        max_workers=max(
            1,
            settings.remote_concurrency,
        )
    )

    futures = {
        pool.submit(
            _guarded,
            task,
        ): task
        for task in tasks
    }

    pending = set(
        futures
    )

    try:
        for fut in as_completed(
            futures,
            timeout=max(
                0.0,
                deadline
                - time.monotonic(),
            ),
        ):
            pending.discard(
                fut
            )

            result = fut.result()

            results[
                result["task_id"]
            ] = result

            _report(
                result
            )

    except FuturesTimeoutError:
        # Harvest any tasks that completed exactly
        # around the deadline boundary.
        for fut in [
            f
            for f in pending
            if f.done()
        ]:
            pending.discard(
                fut
            )

            result = fut.result()

            results[
                result["task_id"]
            ] = result

            _report(
                result
            )

        abandoned = sorted(
            futures[f].task_id
            for f in pending
        )

        print(
            "DEADLINE "
            f"({settings.run_deadline_s:.0f}s): "
            "abandoning "
            f"{len(abandoned)} "
            "unfinished task(s): "
            f"{abandoned}",
            file=sys.stderr,
        )

    # Do not wait for slow workers after the deadline.
    pool.shutdown(
        wait=False,
        cancel_futures=True,
    )

    return (
        results,
        bool(pending),
    )


def write_results(
    path: str,
    tasks: List[Task],
    results: Dict[str, dict],
) -> None:
    """
    Write:

    [
        {
            "task_id": "...",
            "answer": "..."
        }
    ]

    for every input task.
    """

    payload = [
        {
            "task_id": (
                task.task_id
            ),

            "answer": (
                (
                    results.get(
                        task.task_id
                    )
                    or {}
                ).get(
                    "answer"
                )
                or ""
            ),
        }

        for task in tasks
    ]

    directory = os.path.dirname(
        path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    tmp = (
        path
        + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            payload,
            fh,
            ensure_ascii=False,
        )

    # Atomic replacement:
    # never leave half-written JSON behind.
    os.replace(
        tmp,
        path,
    )


def main(
    argv: Optional[
        List[str]
    ] = None,
) -> int:
    """
    CLI entry point.
    """

    # Runtime cap includes model loading.
    started = time.monotonic()

    parser = argparse.ArgumentParser(
        description=(
            "Hybrid token-efficient "
            "routing agent"
        )
    )

    parser.add_argument(
        "--tasks",
        default=(
            "tasks/"
            "sample_tasks.json"
        ),
        help=(
            "path to a JSON list of "
            "{task_id, prompt, metadata?} "
            "(dev mode)"
        ),
    )

    parser.add_argument(
        "--input",
        default=None,
        help=(
            "harness mode: tasks file, "
            "e.g. /input/tasks.json "
            "(overrides --tasks)"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "harness mode: write "
            "[{task_id, answer}] "
            "JSON here — always written, "
            "even on partial failure"
        ),
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "run without model weights "
            "or network"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "override "
            "CONFIDENCE_THRESHOLD "
            "for this run"
        ),
    )

    parser.add_argument(
        "--tier-mode",
        choices=(
            "remote_pair",
            "local_remote",
        ),
        default=None,
        help=(
            "override TIER_MODE: "
            "two Fireworks models "
            "or local cheap "
            "+ strong remote"
        ),
    )

    args = parser.parse_args(
        argv
    )

    # Apply overrides BEFORE constructing components.
    if args.mock:
        settings.mock_mode = True

    if (
        args.tier_mode
        is not None
    ):
        settings.tier_mode = (
            args.tier_mode
        )

    if (
        args.threshold
        is not None
    ):
        settings.confidence_threshold = (
            args.threshold
        )

    harness = (
        args.output
        is not None
    )

    deadline = (
        started
        + settings.run_deadline_s
    )

    # ---------------------------------------------------------
    # Load tasks
    # ---------------------------------------------------------

    try:
        tasks = load_tasks(
            args.input
            or args.tasks
        )

    except Exception as err:
        print(
            "FATAL: cannot read tasks: "
            f"{err}",
            file=sys.stderr,
        )

        if harness:
            try:
                write_results(
                    args.output,
                    [],
                    {},
                )

            except Exception:
                pass

        return 1

    exit_code = 0

    results: Dict[
        str,
        dict,
    ] = {}

    pending = False

    # ---------------------------------------------------------
    # Build router + backends
    # ---------------------------------------------------------

    try:
        router = build_router()

        (
            tier_mode,
            cheap,
            strong,
        ) = build_backends()

        tracker = (
            TokenTracker()
        )

        if (
            tier_mode
            == "local_remote"
            and not settings.mock_mode
        ):
            # Pay local model cold-start once.
            cheap.load()

        print(
            f"tiers: {tier_mode} | "
            f"cheap="
            f"{getattr(cheap, 'model_name', settings.local_model_name)} "
            f"| strong={strong.model_name}",
            file=sys.stderr,
        )

        (
            results,
            pending,
        ) = run_all(
            tasks,
            router,
            cheap,
            strong,
            tracker,
            deadline,
        )

        tracker.print_summary()

    except Exception as err:
        print(
            "FATAL: run aborted: "
            f"{err}",
            file=sys.stderr,
        )

        exit_code = 1

    # ---------------------------------------------------------
    # Harness output
    # ---------------------------------------------------------

    if harness:
        try:
            write_results(
                args.output,
                tasks,
                results,
            )

            print(
                f"wrote {len(tasks)} "
                f"answers to "
                f"{args.output}"
            )

        except Exception as err:
            print(
                "FATAL: cannot write "
                f"results: {err}",
                file=sys.stderr,
            )

            exit_code = 1

    # ---------------------------------------------------------
    # Deadline handling
    # ---------------------------------------------------------

    if pending:
        # Some worker threads may still be blocked in
        # remote reads. Results are already written,
        # so terminate immediately.
        sys.stdout.flush()
        sys.stderr.flush()

        os._exit(
            exit_code
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(
        main()
    )