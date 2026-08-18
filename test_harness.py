"""Offline end-to-end test of the routing pipeline.

Runs entirely in MOCK mode: no model download, no network, no third-party
deps — pure stdlib. This is the "is the wiring sane?" check to run after any
change:

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
    # ── Capability categories ───────────────────────────────────────────
    # easy-local: the boost patterns must beat the length ramp
    (
        Task(
            "sentiment-1",
            "Classify the sentiment of the following review as positive, "
            "negative, or neutral: I had high hopes for this laptop after "
            "reading the glowing reviews online, and to be fair the screen "
            "is gorgeous and the keyboard feels great, but the battery "
            "barely lasts three hours, the fans spin up constantly even "
            "when idle, the customer support team took two weeks to respond "
            "to my ticket, and when they finally did they just sent me a "
            "generic troubleshooting checklist that solved nothing at all.",
        ),
        ROUTE_LOCAL,
    ),
    (
        Task(
            "ner-1",
            "Extract all named entities (people, organizations, locations) "
            "from this text: Tim Cook announced Apple's new partnership "
            "with OpenAI at a press event in San Francisco last Tuesday, "
            "alongside Microsoft representatives.",
        ),
        ROUTE_LOCAL,
    ),
    (
        Task(
            "summarize-1",
            "Summarize the following article in one sentence: The city "
            "council voted on Tuesday to approve the new transit plan after "
            "months of contentious debate. The plan allocates two hundred "
            "million dollars to expanding the light rail network, adds "
            "forty new electric buses, and creates protected bike lanes "
            "along the main downtown corridors. Opponents argued the "
            "funding should have gone to road repairs instead, while "
            "supporters said the investment would reduce congestion and "
            "emissions over the next decade.",
        ),
        ROUTE_LOCAL,
    ),
    # hard-remote: one decisive category signal must cross the bar even on
    # short prompts (0.75 penalties — see confidence.py)
    (
        Task(
            "logic-1",
            "If all bloops are razzies and all razzies are lazzies, are all "
            "bloops definitely lazzies? Alice says yes, Bob says no, and "
            "exactly one of them is telling the truth. Deduce step by step "
            "who is right and state which conclusion must be true.",
        ),
        ROUTE_REMOTE,
    ),
    (
        Task(
            "code-debug-1",
            "Debug this Python function — it crashes with an IndexError on "
            "empty lists and returns the wrong output for lists with "
            "duplicate values. Fix the bug: "
            "def second_largest(xs): return sorted(xs)[-2]",
        ),
        ROUTE_REMOTE,
    ),
    (
        Task(
            "math-word-1",
            "A bakery sells muffins for 3 dollars each and cookies for 2 "
            "dollars each. Maria bought 4 muffins and some cookies, "
            "spending 20 dollars in total. How many cookies did she buy?",
        ),
        ROUTE_REMOTE,
    ),
    (
        Task(
            "code-gen-1",
            "Write a Python function that merges two sorted linked lists "
            "into one sorted linked list without using extra memory.",
        ),
        ROUTE_REMOTE,  # tightest margin in the set (0.535 vs 0.55 threshold)
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
    """resolve_remote_model must honor the ALLOWED_MODELS contract:
    pure settings-driven logic, no network."""
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


def calibrate_logprob_check() -> List[str]:
    """The logprob-gate recommendation logic in scripts/calibrate.py (#8) —
    pure functions over synthetic records, no log file needed."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
    from calibrate import logprob_rows, recommend_logprob_threshold

    def rec(task_id, conf, route="local", escalated=False):
        return {
            "task_id": task_id,
            "route": route,
            "escalated": escalated,
            "local_confidence": conf,
        }

    failures: List[str] = []

    # Clean separation: wrong answers below 0.85, right ones at/above →
    # recommend 0.85, keeping 4 at accuracy 1.0, 0 wasted escalations.
    records = [
        rec("t1", 0.95), rec("t2", 0.90), rec("t3", 0.85),
        rec("t4", 0.80), rec("t5", 0.60), rec("t6", 0.92),
        rec("t7", None, route="remote"),           # never local: excluded
        rec("t8", 0.35, route="remote", escalated=True),  # escalated: excluded
    ]
    grades = {"t1": 1.0, "t2": 1.0, "t3": 1.0, "t4": 0.0, "t5": 0.0,
              "t6": 1.0, "t7": 1.0, "t8": 1.0}
    rows = logprob_rows(records, grades)
    if [r["task_id"] for r in rows] != ["t5", "t4", "t3", "t2", "t6", "t1"]:
        failures.append(f"logprob_rows: wrong selection/order: {rows}")
    got = recommend_logprob_threshold(rows, 0.9)
    if got != (0.85, 1.0, 4, 0):
        failures.append(f"clean separation: expected (0.85, 1.0, 4, 0), got {got}")

    # Confident-wrong on top: no gate separates → None (the mean→min cue).
    bad_rows = logprob_rows(
        [rec("w1", 0.99), rec("r1", 0.70), rec("r2", 0.75)],
        {"w1": 0.0, "r1": 1.0, "r2": 1.0},
    )
    if recommend_logprob_threshold(bad_rows, 0.9) is not None:
        failures.append("confident-wrong: expected None (no usable gate)")

    # Everything already right → 0.0 = gate off.
    ok_rows = logprob_rows(
        [rec("a", 0.5), rec("b", 0.9)], {"a": 1.0, "b": 1.0}
    )
    got = recommend_logprob_threshold(ok_rows, 0.9)
    if got is None or got[0] != 0.0:
        failures.append(f"all-right: expected gate 0.0 (off), got {got}")

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
        print("PASS ALLOWED_MODELS resolution")

    calibrate_failures = calibrate_logprob_check()
    if calibrate_failures:
        failures.extend(calibrate_failures)
    else:
        print("PASS logprob-gate recommendation (calibrate.py)")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("\nAll routing assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
