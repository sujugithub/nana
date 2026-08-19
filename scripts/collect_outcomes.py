"""Collect REAL routing outcomes: run every task through BOTH models, grade
both answers, and write an outcome dataset (routing/dataset.py format).

    # dry run of the whole pipeline, no models, no network, no cost:
    python3 scripts/collect_outcomes.py --tasks tasks/graded_tasks.json \
        --out data/collected.json --mock

    # REAL run — makes one PAID remote call per task, so it must be asked
    # for explicitly:
    python3 scripts/collect_outcomes.py --tasks tasks/graded_tasks.json \
        --out data/collected.json --run-paid-calls

Task file format (JSON list):
    {
      "task_id": "gsm8k-17",
      "prompt": "...",
      "category": "math",            # optional
      "source": "gsm8k",             # optional dataset/source group
      "group_id": "gsm8k-17",        # optional; related prompts share one
      "reference": "42",             # expected answer, for auto-grading
      "grader": "numeric"            # exact | numeric | contains | choice
    }

Grading: programmatic wherever possible (EVALUATION.md). Tasks without a
reference+grader are written with quality -1 sentinels into a SEPARATE
ungraded file for manual/LLM grading; merge verdicts back with --grades
(JSON {task_id: {"local": 0..1, "remote": 0..1}}).

Cost model: remote costs use the API's usage counts at the configured
$/Mtoken rates; local costs use measured wall-clock at an amortised
$/compute-second rate. Both rates are CLI flags so the cost model is
explicit, not buried.

SAFETY RAILS
- Never runs paid calls unless --run-paid-calls is given (or --mock).
- Refuses --run-paid-calls without FIREWORKS_API_KEY set.
- Appends after every task (crash-safe) and skips task_ids already collected,
  so an interrupted run resumes without re-paying for finished tasks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings  # noqa: E402
from routing.dataset import (  # noqa: E402
    OutcomeDataset,
    OutcomeRecord,
    save_dataset,
)

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def grade(grader: str, reference: str, answer: str) -> float:
    """Programmatic 0/1 grade. Deliberately strict-and-simple: these graders
    must be unarguable, per EVALUATION.md."""
    answer_norm = answer.strip().lower()
    ref_norm = reference.strip().lower()
    if grader == "exact":
        return 1.0 if answer_norm == ref_norm else 0.0
    if grader == "numeric":
        nums = _NUM_RE.findall(answer)
        if not nums:
            return 0.0
        try:
            return 1.0 if abs(float(nums[-1]) - float(reference)) < 1e-6 else 0.0
        except ValueError:
            return 0.0
    if grader == "contains":
        return 1.0 if ref_norm in answer_norm else 0.0
    if grader == "choice":
        # multiple-choice: the reference letter/word must appear as a whole
        # token near the start of the answer
        head = answer_norm[:80]
        return 1.0 if re.search(rf"\b{re.escape(ref_norm)}\b", head) else 0.0
    raise ValueError(f"unknown grader {grader!r}")


def _load_existing(path: str) -> Dict[str, dict]:
    """Rows already collected (JSONL work file) — resumability."""
    rows: Dict[str, dict] = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    rows[row["task_id"]] = row
    return rows


def collect(args: argparse.Namespace) -> int:
    with open(args.tasks) as fh:
        tasks = json.load(fh)

    work_path = args.out + ".work.jsonl"
    done = _load_existing(work_path)
    if done:
        print(f"resuming: {len(done)} task(s) already collected in {work_path}")

    from local_model import LocalModel
    from remote_client import RemoteClient
    from router import Router

    local = LocalModel()
    remote = RemoteClient()
    router = Router()
    if not settings.mock_mode:
        local.load()

    ungraded: List[str] = []
    with open(work_path, "a") as work:
        for i, task in enumerate(tasks):
            tid = str(task["task_id"])
            if tid in done:
                continue
            prompt = task["prompt"]
            print(f"[{i + 1}/{len(tasks)}] {tid}", flush=True)

            t0 = time.time()
            local_completion = local.generate(prompt)
            local_latency = time.time() - t0
            _, problems = router.post_check(prompt, local_completion.text)

            t0 = time.time()
            remote_completion = remote.generate(prompt)
            remote_latency = time.time() - t0

            grader = task.get("grader")
            reference = task.get("reference")
            if grader and reference is not None:
                local_q = grade(grader, str(reference), local_completion.text)
                remote_q = grade(grader, str(reference), remote_completion.text)
            else:
                local_q = remote_q = -1.0  # sentinel: needs external grading
                ungraded.append(tid)

            row = {
                "task_id": tid,
                "prompt": prompt,
                "category": task.get("category", "unknown"),
                "source": task.get("source", "unknown"),
                "group_id": task.get("group_id", ""),
                "metadata": task.get("metadata") or {},
                "local_answer": local_completion.text,
                "remote_answer": remote_completion.text,
                "local_quality": local_q,
                "remote_quality": remote_q,
                "local_prompt_tokens": local_completion.prompt_tokens,
                "local_completion_tokens": local_completion.completion_tokens,
                "remote_prompt_tokens": remote_completion.prompt_tokens,
                "remote_completion_tokens": remote_completion.completion_tokens,
                "local_latency_s": round(local_latency, 3),
                "remote_latency_s": round(remote_latency, 3),
                "local_confidence": local_completion.confidence,
                "local_min_token_prob": local_completion.min_token_prob,
                "local_low_token_frac": local_completion.low_token_frac,
                "post_check_problems": problems,
            }
            work.write(json.dumps(row, ensure_ascii=False) + "\n")
            work.flush()
            done[tid] = row

    if ungraded:
        print(
            f"{len(ungraded)} task(s) have no grader/reference — grade their "
            f"answers (they are in {work_path}) and rerun with "
            f"--grades verdicts.json"
        )

    return finalize(args, done)


def finalize(args: argparse.Namespace, done: Dict[str, dict]) -> int:
    """Apply external grades if given, compute costs, and write the dataset."""
    grades: Dict[str, dict] = {}
    if args.grades:
        with open(args.grades) as fh:
            grades = json.load(fh)

    records = []
    skipped = 0
    for tid, row in sorted(done.items()):
        local_q, remote_q = row["local_quality"], row["remote_quality"]
        if tid in grades:
            local_q = float(grades[tid]["local"])
            remote_q = float(grades[tid]["remote"])
        if local_q < 0 or remote_q < 0:
            skipped += 1
            continue
        local_cost = (
            row["local_latency_s"] * args.local_cost_per_second
        )
        remote_cost = (
            row["remote_prompt_tokens"] * args.remote_in_per_mtok / 1e6
            + row["remote_completion_tokens"] * args.remote_out_per_mtok / 1e6
        )
        records.append(
            OutcomeRecord(
                task_id=tid,
                prompt=row["prompt"],
                category=row["category"],
                source=row["source"],
                group_id=row.get("group_id", ""),
                metadata=row.get("metadata") or {},
                local_quality=round(local_q, 4),
                remote_quality=round(remote_q, 4),
                local_ok=local_q >= args.quality_threshold,
                local_prompt_tokens=row["local_prompt_tokens"],
                local_completion_tokens=row["local_completion_tokens"],
                remote_prompt_tokens=row["remote_prompt_tokens"],
                remote_completion_tokens=row["remote_completion_tokens"],
                local_cost=round(local_cost, 9),
                remote_cost=round(remote_cost, 9),
                local_latency_s=row["local_latency_s"],
                remote_latency_s=row["remote_latency_s"],
                local_confidence=row["local_confidence"],
                local_min_token_prob=row["local_min_token_prob"],
                local_low_token_frac=row["local_low_token_frac"],
                post_check_problems=row.get("post_check_problems") or [],
            )
        )

    if not records:
        print("no graded records to write — nothing finalized", file=sys.stderr)
        return 1

    dataset = OutcomeDataset(
        meta={
            "local_model": settings.local_model_name,
            "remote_model": settings.remote_model_name,
            "quality_threshold": args.quality_threshold,
            "cost_unit": "usd",
            "seed": None,
            "notes": (
                f"collected by scripts/collect_outcomes.py"
                f"{' in MOCK mode (answers are canned!)' if settings.mock_mode else ''}; "
                f"cost model: local {args.local_cost_per_second}/s amortised "
                f"compute, remote {args.remote_in_per_mtok}/Mtok in + "
                f"{args.remote_out_per_mtok}/Mtok out"
            ),
        },
        records=records,
    )
    save_dataset(dataset, args.out)
    print(
        f"wrote {len(records)} graded record(s) to {args.out}"
        + (f" ({skipped} ungraded skipped)" if skipped else "")
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect real both-model outcomes into a routing dataset."
    )
    parser.add_argument("--tasks", required=True, help="graded task file (JSON)")
    parser.add_argument("--out", required=True, help="output dataset path")
    parser.add_argument("--mock", action="store_true",
                        help="mock backends: pipeline test, no cost, fake answers")
    parser.add_argument(
        "--run-paid-calls", action="store_true",
        help="EXPLICIT consent to make one paid remote API call per task"
    )
    parser.add_argument(
        "--grades", default=None,
        help="external verdicts JSON {task_id: {local: q, remote: q}}"
    )
    parser.add_argument(
        "--finalize-only", action="store_true",
        help="skip collection; rebuild the dataset from the .work.jsonl file"
    )
    parser.add_argument("--quality-threshold", type=float, default=0.6)
    parser.add_argument("--local-cost-per-second", type=float, default=2e-5,
                        help="amortised local compute $/second (default 2e-5)")
    parser.add_argument("--remote-in-per-mtok", type=float, default=0.56)
    parser.add_argument("--remote-out-per-mtok", type=float, default=1.68)
    args = parser.parse_args()

    if args.mock:
        settings.mock_mode = True
    elif args.finalize_only:
        pass
    elif not args.run_paid_calls:
        print(
            "refusing to run: collection makes one PAID remote call per task.\n"
            "Pass --run-paid-calls to consent, or --mock for a free pipeline "
            "test, or --finalize-only to rebuild from already-collected rows.",
            file=sys.stderr,
        )
        return 2
    elif not settings.fireworks_api_key:
        print(
            "FIREWORKS_API_KEY is not set — cannot make remote calls. "
            "Export it (or use --mock).",
            file=sys.stderr,
        )
        return 2

    if args.finalize_only:
        done = _load_existing(args.out + ".work.jsonl")
        if not done:
            print(f"no collected rows at {args.out}.work.jsonl", file=sys.stderr)
            return 1
        return finalize(args, done)
    return collect(args)


if __name__ == "__main__":
    sys.exit(main())
