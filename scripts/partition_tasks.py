#!/usr/bin/env python3
"""Partition tracked graded tasks into pilot / development / reserved final test.

    python scripts/partition_tasks.py \
        --tasks tasks/real_pilot_qwen_pro.json \
        --tasks tasks/real_extension_fast_qwen_pro.json \
        --prior-dataset data/real_qwen_pro_600.json \
        --out-dir tasks/pair_v2

Offline and deterministic: no downloads, no model calls. Writes
    pilot.json                60 tasks for checking collection and grading
    development_remaining.json  development tasks not in the pilot
    final_test.json           RESERVED: used once, for the approved final run
    manifest.json             provenance, licences, exposure, fingerprints

Leakage rules:
- Groups are effective groups: declared group_id, merged with any
  near-duplicate prompts (routing/splits.py, same threshold as training), so
  no near-duplicate pair can straddle final test and development.
- Tasks in the previous pair's test split (reconstructed from
  --prior-dataset with the trainer's default seed and ratios) have had their
  results seen by the team; they never enter the final test.
- The final test takes whole groups per benchmark source; the pilot takes
  individual development tasks per source.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Set

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from routing.identity import canonical_file_sha256, canonical_json_sha256  # noqa: E402
from routing.splits import (  # noqa: E402
    TEST,
    LeakageError,
    _UnionFind,
    find_exact_duplicates,
    find_near_duplicates,
    make_splits,
)

PARTITION_VERSION = "1"
SEED = 26
NEAR_DUP_THRESHOLD = 0.85
# Pilot composition: per benchmark source, so every grader and BBH subtask
# is exercised.
PILOT_QUOTA = {
    "gsm8k-test": 20,
    "mmlu-test": 20,
    "bbh-boolean_expressions": 5,
    "bbh-causal_judgement": 5,
    "bbh-date_understanding": 5,
    "bbh-logical_deduction_five_objects": 5,
}
# Final test: whole effective groups only, quota per benchmark source (50 per
# category). All 50 bbh-boolean_expressions prompts share an instruction
# template and merge into ONE near-duplicate group, which cannot be split,
# so that subtask stays wholly in development.
FINAL_TEST_QUOTA = {
    "gsm8k-test": 50,
    "mmlu-test": 50,
    "bbh-causal_judgement": 17,
    "bbh-date_understanding": 17,
    "bbh-logical_deduction_five_objects": 16,
}

# Verified 2026-10-26 from each repository's published LICENSE file.
SOURCES = [
    {"name": "GSM8K", "url": "https://github.com/openai/grade-school-math",
     "license": "MIT", "copyright": "Copyright (c) 2021 OpenAI",
     "task_sources": ["gsm8k-test"]},
    {"name": "MMLU", "url": "https://github.com/hendrycks/test",
     "license": "MIT", "copyright": "Copyright (c) 2020 Dan Hendrycks",
     "task_sources": ["mmlu-test"]},
    {"name": "BIG-Bench Hard", "url": "https://github.com/suzgunmirac/BIG-Bench-Hard",
     "license": "MIT", "copyright": "Copyright (c) 2022 suzgunmirac",
     "upstream": {"name": "BIG-bench", "url": "https://github.com/google/BIG-bench",
                  "license": "Apache-2.0",
                  "note": "BBH tasks are derived from BIG-bench tasks; keep the "
                          "Apache-2.0 attribution and notice when redistributing"},
     "task_sources": sorted(s for s in PILOT_QUOTA if s.startswith("bbh-"))},
]


def load_tasks(paths: Sequence[str]) -> List[dict]:
    tasks: List[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            tasks.extend(json.load(fh))
    ids = Counter(str(t["task_id"]) for t in tasks)
    duplicates = sorted(t for t, n in ids.items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate task_id across inputs: {duplicates[:10]}")
    return tasks


def effective_groups(tasks: Sequence[dict]) -> Dict[str, str]:
    """task_id -> effective group (declared groups merged across
    near-duplicates); raises LeakageError on exact duplicates."""
    rows = [SimpleNamespace(task_id=str(t["task_id"]), prompt=t["prompt"],
                            group_id=str(t.get("group_id") or t["task_id"]))
            for t in tasks]
    exact = find_exact_duplicates(rows)
    if exact:
        raise LeakageError(f"exact-duplicate prompts in different groups: {exact[:5]}")
    uf = _UnionFind([r.group_id for r in rows])
    by_id = {r.task_id: r for r in rows}
    for a, b, _score in find_near_duplicates(rows, NEAR_DUP_THRESHOLD):
        uf.union(by_id[a].group_id, by_id[b].group_id)
    return {r.task_id: uf.find(r.group_id) for r in rows}


def prior_test_ids(dataset_path: Optional[str]) -> Set[str]:
    if not dataset_path:
        return set()
    from routing.dataset import load_dataset

    dataset = load_dataset(dataset_path)
    return set(make_splits(dataset.records).ids(TEST))


def _pick(groups: List[str], members: Dict[str, List[dict]], want: int,
          rng: random.Random) -> List[str]:
    """Shuffle whole groups and take them while they fit the quota exactly
    (groups larger than the remaining room are skipped, not split)."""
    order = sorted(groups)
    rng.shuffle(order)
    chosen, size = [], 0
    for g in order:
        n = len(members[g])
        if size + n <= want:
            chosen.append(g)
            size += n
        if size == want:
            break
    if size != want:
        raise ValueError(f"could not fill quota {want} with whole groups (got {size})")
    return chosen


def partition(tasks: Sequence[dict], exposed: Set[str], seed: int = SEED) -> Dict[str, List[dict]]:
    groups = effective_groups(tasks)
    members: Dict[str, List[dict]] = defaultdict(list)
    for t in tasks:
        members[groups[str(t["task_id"])]].append(t)
    rng = random.Random(seed)

    def single(g, key):
        values = {t.get(key, "unknown") for t in members[g]}
        return next(iter(values)) if len(values) == 1 else None

    # Final test: whole single-source groups with no exposed task.
    final_groups: Set[str] = set()
    for source, want in sorted(FINAL_TEST_QUOTA.items()):
        eligible = [g for g in members if single(g, "source") == source
                    and not any(str(t["task_id"]) in exposed for t in members[g])]
        final_groups.update(_pick(eligible, members, want, rng))

    def ordered(rows):
        return sorted(rows, key=lambda t: str(t["task_id"]))

    final = ordered(t for g in final_groups for t in members[g])
    development = ordered(t for g in members if g not in final_groups for t in members[g])

    # Pilot: individual development tasks per source. Pilot and the rest of
    # development are both development data (not an isolation boundary), so
    # a large group may contribute a few pilot tasks.
    pilot: List[dict] = []
    for source, want in sorted(PILOT_QUOTA.items()):
        pool = [t for t in development if t.get("source") == source]
        if len(pool) < want:
            raise ValueError(f"only {len(pool)} development tasks for {source}")
        pilot.extend(rng.sample(pool, want))
    pilot_ids = {str(t["task_id"]) for t in pilot}

    return {
        "pilot": ordered(pilot),
        "development_remaining": [t for t in development if str(t["task_id"]) not in pilot_ids],
        "final_test": final,
    }


def _summary(rows: Sequence[dict], exposed: Set[str]) -> Dict:
    return {
        "count": len(rows),
        "canonical_sha256": canonical_json_sha256(list(rows)),
        "categories": dict(sorted(Counter(t.get("category") for t in rows).items())),
        "sources": dict(sorted(Counter(t.get("source") for t in rows).items())),
        "graders": dict(sorted(Counter(t.get("grader") for t in rows).items())),
        "prior_test_exposed": sum(str(t["task_id"]) in exposed for t in rows),
    }


def build_manifest(parts: Dict[str, List[dict]], tasks: Sequence[dict],
                   exposed: Set[str], inputs: Sequence[str],
                   prior_dataset: Optional[str]) -> Dict:
    return {
        "kind": "nana-task-partition",
        "partition_version": PARTITION_VERSION,
        "seed": SEED,
        "inputs": [{"path": os.path.relpath(p, REPO).replace(os.sep, "/"),
                    "canonical_sha256": canonical_file_sha256(p)}
                   for p in inputs],
        "grouping": {
            "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
            "effective_groups": len(set(effective_groups(tasks).values())),
            "rule": "declared group_id merged across near-duplicate prompts; a group "
                    "never straddles final_test and development (pilot and "
                    "development_remaining are both development and may share a group)",
            "largest_groups": sorted(Counter(effective_groups(tasks).values()).values(),
                                     reverse=True)[:5],
            "note": "all 50 bbh-boolean_expressions prompts share an instruction "
                    "template and form one near-duplicate group; it cannot be split, "
                    "so that subtask is absent from final_test",
        },
        "final_test_quota": FINAL_TEST_QUOTA,
        "pilot_quota": PILOT_QUOTA,
        "prior_exposure": {
            "dataset": (os.path.relpath(prior_dataset, REPO).replace(os.sep, "/")
                        if prior_dataset else None),
            "exposed_test_tasks": len(exposed),
            "method": "test split of the previous pair reconstructed with "
                      "routing.splits.make_splits defaults (seed 7, 70/15/15); sizes "
                      "match the documented 420/90/90 but the original splits.json "
                      "was not available to confirm identity",
            "rule": "exposed tasks never enter final_test",
        },
        "parts": {name: _summary(rows, exposed) for name, rows in parts.items()},
        "use": {
            "pilot": "checking collection and grading only; not a statistical claim",
            "development_remaining": "with pilot outcomes (same identity), the "
                                     "pool for training, calibration, validation",
            "final_test": "RESERVED: collect and evaluate once, after the classifier, "
                          "threshold, grading, split and model pair are frozen and the "
                          "owner approves the final evaluation; do not inspect outcomes "
                          "or tune on it",
        },
        "sources": SOURCES,
        "contamination": [
            "GSM8K (2021), MMLU (2020) and BIG-Bench Hard (2022) are public and "
            "widely reproduced; both tiers may have seen them in training. Absolute "
            "accuracies may be inflated; the router study compares the two tiers "
            "on identical prompts. Not measurable here.",
            "All 600 prompts were used with the previous model pair "
            "(Qwen2.5-1.5B-Instruct -> DeepSeek V4 Pro); its reconstructed test "
            "split is excluded from final_test.",
        ],
        "limitations": [
            f"final_test has {sum(FINAL_TEST_QUOTA.values())} tasks (50 per "
            "category); confidence intervals will be wide.",
            "final_test contains no bbh-boolean_expressions tasks (one inseparable "
            "near-duplicate group).",
            "No coding, extraction, summarisation, open-ended or long-context tasks "
            "(deferred by owner decision).",
        ],
    }


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", action="append", required=True)
    parser.add_argument("--prior-dataset", default=None,
                        help="outcome dataset whose reconstructed test split is excluded")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing partition (refused by default)")
    args = parser.parse_args(argv)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    if os.path.exists(manifest_path) and not args.force:
        print(f"refusing to overwrite {manifest_path}; pass --force", file=sys.stderr)
        return 2
    tasks = load_tasks(args.tasks)
    exposed = prior_test_ids(args.prior_dataset)
    parts = partition(tasks, exposed)
    for name, rows in parts.items():
        write_json(os.path.join(args.out_dir, f"{name}.json"), rows)
    write_json(manifest_path, build_manifest(parts, tasks, exposed, args.tasks,
                                             args.prior_dataset))
    for name, rows in parts.items():
        print(f"{name}: {len(rows)} tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
