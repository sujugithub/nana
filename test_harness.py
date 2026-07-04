"""Offline end-to-end test of the routing pipeline.

Runs entirely in MOCK mode: no model download, no network, no third-party
deps — pure stdlib. This is the "is the wiring sane?" check to run after any
change, especially live during the competition:

    python3 test_harness.py

It asserts the routing DECISIONS (trivial → local, complex → remote), not
answer content — mock answers are canned.
"""
from __future__ import annotations

import os
import sys

# Must happen before importing config: settings reads env at import time.
os.environ["AGENT_MOCK"] = "1"
# Pin the knobs the assertions below are calibrated against, so an exported
# CONFIDENCE_THRESHOLD from a tuning session can't make `make test` fail
# spuriously.
os.environ["CONFIDENCE_THRESHOLD"] = "0.55"
os.environ["ENABLE_ESCALATION"] = "1"

from config import ROUTE_LOCAL, ROUTE_REMOTE, settings  # noqa: E402
from local_model import LocalModel  # noqa: E402
from main import run_task  # noqa: E402
from remote_client import RemoteClient  # noqa: E402
from router import Router  # noqa: E402
from schemas import Task  # noqa: E402
from token_tracker import TokenTracker  # noqa: E402

# (task, expected pre-route) — expected=None means "print, don't assert".
TASKS = [
    (
        Task("trivial-1", "What is the capital of France?"),
        ROUTE_LOCAL,  # short factual lookup: exactly what a 1-3B model nails
    ),
    (
        Task(
            "moderate-1",
            "Summarize the following announcement in one sentence: The team "
            "released version 2.0 of the toolkit on Tuesday. The release adds "
            "support for quantized models and cuts memory usage roughly in half.",
        ),
        None,  # borderline by design — watch which way the threshold sends it
    ),
    (
        Task(
            "complex-1",
            "Write a Python function that computes the median of a stream of "
            "integers, then explain step by step why your approach is "
            "efficient, and calculate the expected memory usage for 10 "
            "million values.",
        ),
        ROUTE_REMOTE,  # code + multi-step reasoning + math: beyond a small model
    ),
]


def main() -> int:
    assert settings.mock_mode, "harness must run in mock mode"

    router = Router()
    local = LocalModel()
    remote = RemoteClient()
    tracker = TokenTracker(log_path="")  # no file logging during tests

    failures = []
    for task, expected in TASKS:
        result = run_task(task, router, local, remote, tracker)
        status = "  "
        if expected is not None:
            if result["route"] == expected:
                status = "PASS "
            else:
                status = "FAIL "
                failures.append(
                    f"{task.task_id}: expected {expected}, got {result['route']} "
                    f"(confidence={result['confidence']}, "
                    f"signals={result['signals']})"
                )
        print(
            f"{status}[{task.task_id}] route={result['route']} "
            f"conf={result['confidence']:.2f} signals={result['signals']} "
            f"billable={result['billable_tokens']}"
        )

    tracker.print_summary()

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("\nAll routing assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
