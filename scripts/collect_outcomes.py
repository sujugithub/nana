"""Collect REAL outcomes from BOTH configured tiers, grade both answers,
and write an outcome dataset in routing/dataset.py format.

    # dry run of the whole pipeline, no models, no network, no cost:
    python3 scripts/collect_outcomes.py --tasks tasks/graded_tasks.json \
        --out data/collected.json --mock

    # REAL run — makes paid Fireworks calls, so it must be explicit:
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
(JSON {task_id: {"cheap": 0..1, "strong": 0..1}}). Legacy ``local`` and
``remote`` verdict keys remain accepted.

Cost model: Fireworks tiers use API usage counts at configured $/Mtoken
rates. A truly local cheap tier uses wall-clock time at an amortised
$/compute-second rate. Every rate is an explicit flag.

SAFETY RAILS
- Never runs paid calls unless --run-paid-calls is given (or --mock).
- Refuses --run-paid-calls without FIREWORKS_API_KEY set. ``remote_pair``
  makes two paid calls per task; ``local_remote`` makes one.
- Appends after every task (crash-safe) and skips task_ids already collected,
  so an interrupted run resumes without re-paying for finished tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _load_dotenv() -> None:
    """Load repo .env without replacing explicitly exported variables."""
    try:
        with open(os.path.join(REPO, ".env")) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    except FileNotFoundError:
        pass


_load_dotenv()

from config import settings  # noqa: E402
from routing.dataset import (  # noqa: E402
    OutcomeDataset,
    OutcomeRecord,
    save_dataset,
)

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def configured_api_cost_ceiling(
    tasks: List[dict], done: Dict[str, dict], args: argparse.Namespace
) -> float:
    """Conservative configured API-cost ceiling for unfinished tasks.

    The bound assumes every Fireworks response consumes its full max-token
    cap and pessimistically treats each UTF-8 input byte as a token. It is a
    cost-control guard based on configured rates, not a provider guarantee.
    Local generation is excluded because it has no API charge.
    """
    remaining = [t for t in tasks if str(t["task_id"]) not in done]
    input_upper = sum(
        len((settings.system_prompt + "\n" + t["prompt"]).encode("utf-8")) + 64
        for t in remaining
    )
    strong = (
        input_upper * args.strong_in_per_mtok
        + len(remaining) * settings.strong_max_tokens * args.strong_out_per_mtok
    ) / 1e6
    if settings.tier_mode.strip().lower() != "remote_pair":
        return strong
    cheap = (
        input_upper * args.cheap_in_per_mtok
        + len(remaining) * settings.cheap_max_tokens * args.cheap_out_per_mtok
    ) / 1e6
    return strong + cheap


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
            predicted = float(nums[-1].replace(",", ""))
            expected = float(reference.replace(",", ""))
            return 1.0 if abs(predicted - expected) < 1e-6 else 0.0
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

    if not settings.mock_mode:
        ceiling = configured_api_cost_ceiling(tasks, done, args)
        budget = getattr(args, "max_api_cost_usd", None)
        if budget is None:
            print(
                f"refusing paid collection: configured worst-case API cost "
                f"for unfinished tasks is ${ceiling:.4f}. Pass "
                "--max-api-cost-usd with an explicit budget at least this "
                "large.",
                file=sys.stderr,
            )
            return 2
        if budget <= 0 or ceiling > budget + 1e-12:
            print(
                f"refusing paid collection: configured worst-case API cost "
                f"${ceiling:.4f} exceeds --max-api-cost-usd ${budget:.4f}.",
                file=sys.stderr,
            )
            return 2
        print(
            f"paid-collection guard: {len(tasks) - len(done)} unfinished "
            f"task(s), configured worst-case API cost ${ceiling:.4f} <= "
            f"budget ${budget:.4f}",
            file=sys.stderr,
        )

    from local_model import LocalModel
    from remote_client import CheapRemoteClient, StrongRemoteClient
    from router import Router

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
    strong = StrongRemoteClient()
    router = Router()
    if tier_mode == "local_remote" and not settings.mock_mode:
        cheap.load()

    expected_pair = (tier_mode, cheap.model_name, strong.model_name)
    for row in done.values():
        found = (
            row.get("tier_mode"),
            row.get("cheap_model"),
            row.get("strong_model"),
        )
        if not all(found):
            raise ValueError(
                f"cannot safely resume legacy rows in {work_path}: they do "
                "not record the exact tier mode/model pair. Finalize them "
                "with --finalize-only or collect this pair to a new --out path."
            )
        if found != expected_pair:
            raise ValueError(
                "refusing to mix rows from different tier pairs in "
                f"{work_path}: found {found}, expected {expected_pair}"
            )

    ungraded: List[str] = []
    with open(work_path, "a") as work:
        for i, task in enumerate(tasks):
            tid = str(task["task_id"])
            if tid in done:
                continue
            prompt = task["prompt"]
            print(f"[{i + 1}/{len(tasks)}] {tid}", flush=True)

            t0 = time.time()
            cheap_completion = cheap.generate(prompt)
            cheap_latency = time.time() - t0
            _, problems = router.post_check(prompt, cheap_completion.text)

            t0 = time.time()
            strong_completion = strong.generate(prompt)
            strong_latency = time.time() - t0

            grader = task.get("grader")
            reference = task.get("reference")
            if grader and reference is not None:
                cheap_q = grade(grader, str(reference), cheap_completion.text)
                strong_q = grade(grader, str(reference), strong_completion.text)
            else:
                cheap_q = strong_q = -1.0  # sentinel: needs external grading
                ungraded.append(tid)

            row = {
                "task_id": tid,
                "prompt": prompt,
                "category": task.get("category", "unknown"),
                "source": task.get("source", "unknown"),
                "group_id": task.get("group_id", ""),
                "metadata": task.get("metadata") or {},
                "tier_mode": tier_mode,
                "cheap_model": cheap_completion.model_name,
                "cheap_provider": cheap_completion.provider,
                "strong_model": strong_completion.model_name,
                "strong_provider": strong_completion.provider,
                # Schema 1.0 uses local/remote field names. For new data they
                # intentionally mean cheap/strong, preserving Fable's model.
                "local_answer": cheap_completion.text,
                "remote_answer": strong_completion.text,
                "local_quality": cheap_q,
                "remote_quality": strong_q,
                "local_prompt_tokens": cheap_completion.prompt_tokens,
                "local_completion_tokens": cheap_completion.completion_tokens,
                "remote_prompt_tokens": strong_completion.prompt_tokens,
                "remote_completion_tokens": strong_completion.completion_tokens,
                "local_latency_s": round(cheap_latency, 3),
                "remote_latency_s": round(strong_latency, 3),
                "local_confidence": cheap_completion.confidence,
                "local_min_token_prob": cheap_completion.min_token_prob,
                "local_low_token_frac": cheap_completion.low_token_frac,
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
        cheap_q, strong_q = row["local_quality"], row["remote_quality"]
        if tid in grades:
            verdict = grades[tid]
            cheap_q = float(verdict.get("cheap", verdict.get("local")))
            strong_q = float(verdict.get("strong", verdict.get("remote")))
        if cheap_q < 0 or strong_q < 0:
            skipped += 1
            continue
        if row.get("cheap_provider") == "fireworks":
            cheap_cost = (
                row["local_prompt_tokens"] * args.cheap_in_per_mtok / 1e6
                + row["local_completion_tokens"] * args.cheap_out_per_mtok / 1e6
            )
        else:
            cheap_cost = row["local_latency_s"] * args.local_cost_per_second
        strong_cost = (
            row["remote_prompt_tokens"] * args.strong_in_per_mtok / 1e6
            + row["remote_completion_tokens"] * args.strong_out_per_mtok / 1e6
        )
        records.append(
            OutcomeRecord(
                task_id=tid,
                prompt=row["prompt"],
                category=row["category"],
                source=row["source"],
                group_id=row.get("group_id", ""),
                metadata=row.get("metadata") or {},
                local_quality=round(cheap_q, 4),
                remote_quality=round(strong_q, 4),
                local_ok=cheap_q >= args.quality_threshold,
                local_prompt_tokens=row["local_prompt_tokens"],
                local_completion_tokens=row["local_completion_tokens"],
                remote_prompt_tokens=row["remote_prompt_tokens"],
                remote_completion_tokens=row["remote_completion_tokens"],
                local_cost=round(cheap_cost, 9),
                remote_cost=round(strong_cost, 9),
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

    first_row = next(iter(done.values()))
    dataset = OutcomeDataset(
        meta={
            # Required schema-1.0 names; semantically cheap and strong.
            "local_model": first_row.get("cheap_model", settings.local_model_name),
            "remote_model": first_row.get("strong_model", settings.strong_model_name),
            "tier_mode": first_row.get("tier_mode", settings.tier_mode),
            "tier_schema": "legacy local=cheap, remote=strong",
            "cheap_max_tokens": settings.local_max_new_tokens
            if first_row.get("tier_mode", settings.tier_mode) == "local_remote"
            else settings.cheap_max_tokens,
            "strong_max_tokens": settings.strong_max_tokens,
            "temperature": 0,
            "system_prompt_sha256": hashlib.sha256(
                settings.system_prompt.encode("utf-8")
            ).hexdigest(),
            "quality_threshold": args.quality_threshold,
            "cost_unit": "usd",
            "seed": None,
            "notes": (
                f"collected by scripts/collect_outcomes.py"
                f"{' in MOCK mode (answers are canned!)' if settings.mock_mode else ''}; "
                f"cost model: local {args.local_cost_per_second}/s amortised; "
                f"cheap Fireworks {args.cheap_in_per_mtok}/Mtok in + "
                f"{args.cheap_out_per_mtok}/Mtok out; strong Fireworks "
                f"{args.strong_in_per_mtok}/Mtok in + "
                f"{args.strong_out_per_mtok}/Mtok out"
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
        description="Collect real cheap/strong outcomes into a routing dataset."
    )
    parser.add_argument("--tasks", required=True, help="graded task file (JSON)")
    parser.add_argument("--out", required=True, help="output dataset path")
    parser.add_argument("--mock", action="store_true",
                        help="mock backends: pipeline test, no cost, fake answers")
    parser.add_argument(
        "--tier-mode",
        choices=("remote_pair", "local_remote"),
        default=None,
        help="override TIER_MODE for this collection run",
    )
    parser.add_argument(
        "--run-paid-calls", action="store_true",
        help="EXPLICIT consent to paid Fireworks calls (1 or 2 per task)"
    )
    parser.add_argument(
        "--max-api-cost-usd", type=float, default=None,
        help="required for paid collection; must cover the configured "
        "worst-case cost of unfinished tasks",
    )
    parser.add_argument(
        "--grades", default=None,
        help="external verdicts JSON {task_id: {cheap: q, strong: q}}"
    )
    parser.add_argument(
        "--finalize-only", action="store_true",
        help="skip collection; rebuild the dataset from the .work.jsonl file"
    )
    parser.add_argument("--quality-threshold", type=float, default=0.6)
    parser.add_argument("--local-cost-per-second", type=float, default=2e-5,
                        help="amortised local compute $/second (default 2e-5)")
    parser.add_argument(
        "--cheap-in-per-mtok", type=float,
        default=settings.cheap_input_per_mtok,
    )
    parser.add_argument(
        "--cheap-out-per-mtok", type=float,
        default=settings.cheap_output_per_mtok,
    )
    parser.add_argument(
        "--strong-in-per-mtok", "--remote-in-per-mtok", type=float,
        default=settings.strong_input_per_mtok,
    )
    parser.add_argument(
        "--strong-out-per-mtok", "--remote-out-per-mtok", type=float,
        default=settings.strong_output_per_mtok,
    )
    args = parser.parse_args()

    if args.tier_mode is not None:
        settings.tier_mode = args.tier_mode
    if args.mock:
        settings.mock_mode = True
    elif args.finalize_only:
        pass
    elif not args.run_paid_calls:
        print(
            "refusing to run: collection makes PAID Fireworks calls "
            f"({2 if settings.tier_mode.strip().lower() == 'remote_pair' else 1} "
            "per task in this tier mode).\n"
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
