#!/usr/bin/env python3
"""Build a deterministic, automatically graded router outcome task set.

The script does not download anything and never calls a model. Point it at
local copies of the official MIT-licensed GSM8K, MMLU, and BIG-Bench Hard
datasets. It writes the task format consumed by ``collect_outcomes.py`` plus
a provenance manifest suitable for the FYP report.

Example:
    python3 scripts/build_real_pilot.py \
      --gsm8k-test /tmp/grade-school-math/grade_school_math/data/test.jsonl \
      --mmlu-test-dir /tmp/mmlu-data/test \
      --bbh-dir /tmp/BIG-Bench-Hard/bbh \
      --out tasks/real_pilot_qwen_pro.json \
      --manifest tasks/real_pilot_qwen_pro.manifest.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter
from datetime import date
from typing import Dict, Iterable, List

SEED = 21
MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_computer_science",
    "college_mathematics",
    "conceptual_physics",
    "global_facts",
    "high_school_geography",
    "high_school_psychology",
    "high_school_world_history",
    "machine_learning",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "professional_law",
    "professional_medicine",
    "security_studies",
    "world_religions",
)
BBH_TASKS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "logical_deduction_five_objects",
)
LETTERS = "ABCD"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sample(
    rows: List, n: int, seed: int, excluded_indices: Iterable[int] = ()
) -> List[dict]:
    excluded = set(excluded_indices)
    eligible = [i for i in range(len(rows)) if i not in excluded]
    if len(eligible) < n:
        raise ValueError(
            f"need {n} rows but source has only {len(eligible)} after exclusions"
        )
    rng = random.Random(seed)
    picked = rng.sample(eligible, n)
    sampled = []
    for i in sorted(picked):
        row = dict(rows[i]) if isinstance(rows[i], dict) else {"row": list(rows[i])}
        row["source_index"] = i
        sampled.append(row)
    return sampled


def gsm8k_tasks(
    path: str,
    n: int = 40,
    excluded_indices: Iterable[int] = (),
    id_prefix: str = "",
    answer_only: bool = False,
) -> List[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            source = json.loads(line)
            match = re.search(r"####\s*([^\n]+)\s*$", source["answer"])
            if not match:
                continue
            reference = match.group(1).strip().replace(",", "")
            try:
                float(reference)
            except ValueError:
                continue
            rows.append({"question": source["question"], "reference": reference})

    out = []
    for i, row in enumerate(
        _stable_sample(rows, n, SEED + 1, excluded_indices)
    ):
        task_id = f"{id_prefix}gsm8k-{i:03d}"
        instruction = (
            "Solve this grade-school math problem silently and reply with "
            "only the final numeric answer."
            if answer_only else
            "Solve this grade-school math problem. Show only essential "
            "reasoning and finish with the final numeric answer."
        )
        out.append({
            "task_id": task_id,
            "prompt": instruction + "\n\n" + row["question"].strip(),
            "category": "math",
            "source": "gsm8k-test",
            "group_id": task_id,
            "reference": row["reference"],
            "grader": "numeric",
            "metadata": {
                "benchmark": "GSM8K",
                "split": "test",
                "source_index": row["source_index"],
            },
        })
    return out


def _mmlu_prompt(row: List[str], subject: str) -> str:
    question, options = row[0].strip(), row[1:5]
    rendered = "\n".join(
        f"({letter}) {option.strip()}" for letter, option in zip(LETTERS, options)
    )
    return (
        f"Answer this {subject.replace('_', ' ')} multiple-choice question. "
        "Reply with exactly one option letter: A, B, C, or D.\n\n"
        f"{question}\nOptions:\n{rendered}"
    )


def mmlu_tasks(
    directory: str,
    per_subject: int = 2,
    excluded_by_subject: Dict[str, Iterable[int]] = None,
    id_prefix: str = "",
) -> List[dict]:
    excluded_by_subject = excluded_by_subject or {}
    out = []
    serial = 0
    for subject_no, subject in enumerate(MMLU_SUBJECTS):
        path = os.path.join(directory, f"{subject}_test.csv")
        with open(path, newline="") as fh:
            rows = [row for row in csv.reader(fh) if len(row) >= 6]
        picked = _stable_sample(
            rows,
            per_subject,
            SEED + 100 + subject_no,
            excluded_by_subject.get(subject, ()),
        )
        for sampled in picked:
            row = sampled["row"]
            answer = row[5].strip().upper()
            if answer not in LETTERS:
                raise ValueError(f"invalid MMLU answer {answer!r} in {path}")
            task_id = f"{id_prefix}mmlu-{serial:03d}"
            out.append({
                "task_id": task_id,
                "prompt": _mmlu_prompt(row, subject),
                "category": "knowledge",
                "source": "mmlu-test",
                "group_id": task_id,
                "reference": answer,
                "grader": "choice",
                "metadata": {
                    "benchmark": "MMLU",
                    "split": "test",
                    "subject": subject,
                    "source_index": sampled["source_index"],
                },
            })
            serial += 1
    return out


def _bbh_reference(target: str) -> str:
    target = target.strip()
    match = re.fullmatch(r"\(([A-Z])\)", target)
    return match.group(1) if match else target


def bbh_tasks(
    directory: str,
    per_task: int = 10,
    excluded_by_task: Dict[str, Iterable[int]] = None,
    id_prefix: str = "",
) -> List[dict]:
    excluded_by_task = excluded_by_task or {}
    out = []
    serial = 0
    for task_no, task_name in enumerate(BBH_TASKS):
        path = os.path.join(directory, f"{task_name}.json")
        with open(path) as fh:
            examples = json.load(fh)["examples"]
        picked = _stable_sample(
            examples,
            per_task,
            SEED + 200 + task_no,
            excluded_by_task.get(task_name, ()),
        )
        for row in picked:
            reference = _bbh_reference(str(row["target"]))
            instruction = (
                "Solve this reasoning problem. Reply with exactly the option "
                "letter or truth-value requested; do not add another answer."
            )
            task_id = f"{id_prefix}bbh-{serial:03d}"
            out.append({
                "task_id": task_id,
                "prompt": instruction + "\n\n" + row["input"].strip(),
                "category": "reasoning",
                "source": f"bbh-{task_name}",
                "group_id": task_id,
                "reference": reference,
                "grader": "choice",
                "metadata": {
                    "benchmark": "BIG-Bench Hard",
                    "split": "published task set",
                    "subtask": task_name,
                    "source_index": row["source_index"],
                },
            })
            serial += 1
    return out


def _write_json(path: str, payload) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _validate(tasks: Iterable[dict]) -> List[dict]:
    tasks = list(tasks)
    ids = [row["task_id"] for row in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task_id in pilot")
    prompts = [" ".join(row["prompt"].lower().split()) for row in tasks]
    if len(prompts) != len(set(prompts)):
        raise ValueError("duplicate normalized prompt in pilot")
    for row in tasks:
        if row["grader"] not in ("numeric", "choice"):
            raise ValueError(f"unsupported grader in {row['task_id']}")
        if not row["reference"]:
            raise ValueError(f"empty reference in {row['task_id']}")
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm8k-test", required=True)
    parser.add_argument("--mmlu-test-dir", required=True)
    parser.add_argument("--bbh-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--tasks-per-category", type=int, default=40,
        help="balanced count for each of math, knowledge, and reasoning",
    )
    parser.add_argument(
        "--exclude-tasks",
        help="optional prior task JSON; its source rows will not be sampled",
    )
    parser.add_argument(
        "--task-id-prefix", default="",
        help="prefix for task/group IDs when building a disjoint extension",
    )
    parser.add_argument(
        "--math-id-prefix",
        help="optional separate ID prefix for math tasks",
    )
    parser.add_argument(
        "--math-answer-only", action="store_true",
        help="ask GSM8K models for only the final number",
    )
    args = parser.parse_args()

    if args.tasks_per_category <= 0:
        parser.error("--tasks-per-category must be positive")
    if args.tasks_per_category % len(MMLU_SUBJECTS):
        parser.error("--tasks-per-category must be divisible by 20 for MMLU")
    if args.tasks_per_category % len(BBH_TASKS):
        parser.error("--tasks-per-category must be divisible by 4 for BBH")

    excluded_gsm8k = set()
    excluded_mmlu: Dict[str, set] = {}
    excluded_bbh: Dict[str, set] = {}
    if args.exclude_tasks:
        with open(args.exclude_tasks) as fh:
            prior_tasks = json.load(fh)
        for row in prior_tasks:
            metadata = row.get("metadata", {})
            source_index = metadata.get("source_index")
            if not isinstance(source_index, int):
                continue
            benchmark = metadata.get("benchmark")
            if benchmark == "GSM8K":
                excluded_gsm8k.add(source_index)
            elif benchmark == "MMLU":
                excluded_mmlu.setdefault(metadata["subject"], set()).add(
                    source_index
                )
            elif benchmark == "BIG-Bench Hard":
                excluded_bbh.setdefault(metadata["subtask"], set()).add(
                    source_index
                )

    per_mmlu_subject = args.tasks_per_category // len(MMLU_SUBJECTS)
    per_bbh_task = args.tasks_per_category // len(BBH_TASKS)

    tasks = _validate(
        gsm8k_tasks(
            args.gsm8k_test,
            n=args.tasks_per_category,
            excluded_indices=excluded_gsm8k,
            id_prefix=args.math_id_prefix or args.task_id_prefix,
            answer_only=args.math_answer_only,
        )
        + mmlu_tasks(
            args.mmlu_test_dir,
            per_subject=per_mmlu_subject,
            excluded_by_subject=excluded_mmlu,
            id_prefix=args.task_id_prefix,
        )
        + bbh_tasks(
            args.bbh_dir,
            per_task=per_bbh_task,
            excluded_by_task=excluded_bbh,
            id_prefix=args.task_id_prefix,
        )
    )
    # A final seeded shuffle avoids category blocks while remaining reproducible.
    random.Random(SEED).shuffle(tasks)
    _write_json(args.out, tasks)

    manifest: Dict[str, object] = {
        "kind": "real-outcome-pilot-manifest",
        "created": date.today().isoformat(),
        "seed": SEED,
        "prompt_protocol": {
            "math_answer_only": args.math_answer_only,
        },
        "excluded_tasks": (
            {
                "path": args.exclude_tasks,
                "sha256": _sha256(args.exclude_tasks),
                "count": len(prior_tasks),
            }
            if args.exclude_tasks else None
        ),
        "task_count": len(tasks),
        "categories": dict(sorted(Counter(t["category"] for t in tasks).items())),
        "graders": dict(sorted(Counter(t["grader"] for t in tasks).items())),
        "intended_pair": {
            "tier_mode": "local_remote",
            "cheap": "Qwen/Qwen2.5-1.5B-Instruct",
            "strong": "accounts/fireworks/models/deepseek-v4-pro",
        },
        "sources": [
            {
                "name": "GSM8K",
                "url": "https://github.com/openai/grade-school-math",
                "license": "MIT",
                "input_sha256": _sha256(args.gsm8k_test),
            },
            {
                "name": "MMLU",
                "url": "https://github.com/hendrycks/test",
                "license": "MIT",
                "input_sha256": {
                    subject: _sha256(
                        os.path.join(args.mmlu_test_dir, f"{subject}_test.csv")
                    )
                    for subject in MMLU_SUBJECTS
                },
            },
            {
                "name": "BIG-Bench Hard",
                "url": "https://github.com/suzgunmirac/BIG-Bench-Hard",
                "license": "MIT",
                "input_sha256": {
                    task: _sha256(os.path.join(args.bbh_dir, f"{task}.json"))
                    for task in BBH_TASKS
                },
            },
        ],
        "limitations": [
            "Pilot scale: results are preliminary, not a final FYP claim.",
            "Public benchmark prompts may have appeared in model pretraining data.",
            "Only deterministic numeric/multiple-choice grading is included.",
            "A separate outcome collection and artifact are required for remote_pair.",
        ],
    }
    _write_json(args.manifest, manifest)
    print(f"wrote {len(tasks)} tasks to {args.out}")
    print(f"wrote provenance manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
