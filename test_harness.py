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

import json  # noqa: E402
import tempfile  # noqa: E402
from typing import List  # noqa: E402

import main as main_module  # noqa: E402
from config import ROUTE_LOCAL, ROUTE_REMOTE, settings  # noqa: E402
from local_model import LocalModel  # noqa: E402
from main import run_task  # noqa: E402
from remote_client import RemoteClient, resolve_remote_model  # noqa: E402
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


def harness_io_check() -> List[str]:
    """End-to-end check of the scoring-harness adapter (--input/--output),
    which also exercises the concurrent run_all path. Mock mode, no deps."""
    failures: List[str] = []
    expected_ids = [task.task_id for task, _ in TASKS]

    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, "tasks.json")
        output_path = os.path.join(tmp, "out", "results.json")  # dir must be created
        with open(input_path, "w") as fh:
            json.dump(
                [{"task_id": t.task_id, "prompt": t.prompt} for t, _ in TASKS],
                fh,
            )

        # main() constructs its own TokenTracker — silence file logging so a
        # wiring test never pollutes logs/usage.jsonl (calibration data).
        saved_log_path = settings.usage_log_path
        settings.usage_log_path = ""
        try:
            code = main_module.main(
                ["--input", input_path, "--output", output_path]
            )
        finally:
            settings.usage_log_path = saved_log_path

        if code != 0:
            failures.append(f"harness mode: expected exit 0, got {code}")
            return failures
        try:
            with open(output_path) as fh:
                rows = json.load(fh)
        except Exception as err:
            failures.append(f"harness mode: results.json unreadable: {err}")
            return failures

        if [row.get("task_id") for row in rows] != expected_ids:
            failures.append(
                f"harness mode: task_ids {rows} != input order {expected_ids}"
            )
        if any(set(row) != {"task_id", "answer"} for row in rows):
            failures.append(
                "harness mode: rows must contain exactly task_id + answer"
            )
        if any(not isinstance(row.get("answer"), str) or not row["answer"]
               for row in rows):
            failures.append(f"harness mode: empty/non-string answer in {rows}")
    return failures


def allowed_models_check() -> List[str]:
    """resolve_remote_model must honor the ALLOWED_MODELS contract (kickoff
    #10): pure settings-driven logic, no network."""
    DEV_DEFAULT = "accounts/fireworks/models/deepseek-v4-pro"
    # (allowed_models, remote_model_name, preference, expected)
    cases = [
        # unset → dev fallback
        ("", DEV_DEFAULT, "deepseek-v4-pro", DEV_DEFAULT),
        # explicit REMOTE_MODEL_NAME matches an allowed entry (short vs full
        # path spellings) → the ALLOWED entry, verbatim
        (
            "m-a, accounts/fireworks/models/deepseek-v4-pro ,m-b",
            "deepseek-v4-pro",
            "",
            "accounts/fireworks/models/deepseek-v4-pro",
        ),
        # explicit not allowed, preference is → preference's allowed entry
        ("m-a,m-b,m-c", "not-allowed", "nope,M-B", "m-b"),
        # nothing matches → first of the list
        ("m-a,m-b", "not-allowed", "also-not", "m-a"),
        # set but unusable → None (remote disabled, local fallback per call)
        ("  ,  ", DEV_DEFAULT, "deepseek-v4-pro", None),
    ]
    failures: List[str] = []
    saved = (
        settings.allowed_models,
        settings.remote_model_name,
        settings.remote_model_preference,
    )
    try:
        for allowed, name, pref, expected in cases:
            settings.allowed_models = allowed
            settings.remote_model_name = name
            settings.remote_model_preference = pref
            got = resolve_remote_model()
            if got != expected:
                failures.append(
                    f"resolve_remote_model(allowed={allowed!r}, name={name!r}, "
                    f"pref={pref!r}): expected {expected!r}, got {got!r}"
                )
    finally:
        (
            settings.allowed_models,
            settings.remote_model_name,
            settings.remote_model_preference,
        ) = saved
    return failures


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

    io_failures = harness_io_check()
    if io_failures:
        failures.extend(io_failures)
    else:
        print("PASS harness I/O adapter (--input → --output, concurrent run)")

    model_failures = allowed_models_check()
    if model_failures:
        failures.extend(model_failures)
    else:
        print("PASS ALLOWED_MODELS resolution (kickoff #10 contract)")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("\nAll routing assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
