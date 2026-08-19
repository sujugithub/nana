"""Group-aware train/validation/test splits + duplicate/leakage detection.

Leakage rules enforced here:
- Related or paraphrased prompts share a group_id, and a GROUP is assigned to
  exactly one split — so a paraphrase of a training prompt can never sit in
  the test set inflating the score.
- Exact duplicates (same normalized prompt) in DIFFERENT groups are an error:
  the grouping is wrong and must be fixed at the source.
- Near-duplicates (character-shingle Jaccard above a threshold) in different
  groups get their groups MERGED before splitting, and the merge is reported.

The split itself is deterministic given (records, seed, ratios): groups are
shuffled with a seeded RNG and packed greedily by record count.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

from routing.dataset import OutcomeRecord, normalize_prompt

TRAIN, VAL, TEST = "train", "val", "test"


class LeakageError(ValueError):
    """The dataset's grouping contradicts its content."""


@dataclass
class SplitResult:
    assignment: Dict[str, str]  # task_id -> "train" | "val" | "test"
    seed: int
    ratios: Tuple[float, float, float]
    merged_groups: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def ids(self, split: str) -> List[str]:
        return [tid for tid, s in self.assignment.items() if s == split]

    def indices(self, records: Sequence[OutcomeRecord], split: str) -> List[int]:
        return [
            i for i, r in enumerate(records) if self.assignment[r.task_id] == split
        ]


def _shingles(text: str, k: int = 5) -> Set[str]:
    text = normalize_prompt(text)
    if len(text) <= k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


class _UnionFind:
    def __init__(self, items: Sequence[str]):
        self.parent = {item: item for item in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic direction: smaller root name wins.
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def find_exact_duplicates(
    records: Sequence[OutcomeRecord],
) -> List[Tuple[str, str]]:
    """(task_id_a, task_id_b) pairs whose normalized prompts are identical
    but whose group_ids differ — a grouping bug, not a mergeable nuisance."""
    by_norm: Dict[str, OutcomeRecord] = {}
    conflicts: List[Tuple[str, str]] = []
    for rec in records:
        norm = normalize_prompt(rec.prompt)
        prior = by_norm.get(norm)
        if prior is None:
            by_norm[norm] = rec
        elif prior.group_id != rec.group_id:
            conflicts.append((prior.task_id, rec.task_id))
    return conflicts


def find_near_duplicates(
    records: Sequence[OutcomeRecord],
    threshold: float = 0.85,
) -> List[Tuple[str, str, float]]:
    """(task_id_a, task_id_b, jaccard) for cross-group near-duplicate pairs.

    O(n^2) with a cheap length pre-filter — fine for the dataset sizes this
    project realistically collects (hundreds to a few thousand rows).
    """
    shingle_sets = [_shingles(r.prompt) for r in records]
    sizes = [len(s) for s in shingle_sets]
    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i].group_id == records[j].group_id:
                continue
            # Jaccard >= t requires the size ratio to be >= t.
            lo, hi = sorted((sizes[i], sizes[j]))
            if hi == 0 or lo / hi < threshold:
                continue
            score = _jaccard(shingle_sets[i], shingle_sets[j])
            if score >= threshold:
                pairs.append((records[i].task_id, records[j].task_id, score))
    return pairs


def make_splits(
    records: Sequence[OutcomeRecord],
    seed: int = 7,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    near_dup_threshold: float = 0.85,
) -> SplitResult:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")

    exact = find_exact_duplicates(records)
    if exact:
        shown = ", ".join(f"{a}/{b}" for a, b in exact[:10])
        raise LeakageError(
            f"{len(exact)} exact-duplicate prompt pair(s) with DIFFERENT "
            f"group_ids ({shown}...). Fix the grouping: identical prompts "
            f"must share a group."
        )

    warnings: List[str] = []
    by_id = {r.task_id: r for r in records}
    uf = _UnionFind([r.group_id for r in records])
    near = find_near_duplicates(records, near_dup_threshold)
    merged: List[Tuple[str, str]] = []
    for tid_a, tid_b, score in near:
        ga, gb = by_id[tid_a].group_id, by_id[tid_b].group_id
        if uf.find(ga) != uf.find(gb):
            merged.append((ga, gb))
            warnings.append(
                f"near-duplicate ({score:.2f}) across groups: {tid_a} / "
                f"{tid_b} — groups merged for splitting"
            )
        uf.union(ga, gb)

    effective_group = {r.task_id: uf.find(r.group_id) for r in records}
    group_sizes: Dict[str, int] = {}
    for g in effective_group.values():
        group_sizes[g] = group_sizes.get(g, 0) + 1

    groups = sorted(group_sizes)
    random.Random(seed).shuffle(groups)

    total = len(records)
    targets = {TRAIN: ratios[0] * total, VAL: ratios[1] * total, TEST: ratios[2] * total}
    filled = {TRAIN: 0, VAL: 0, TEST: 0}
    group_split: Dict[str, str] = {}
    # Greedy: each group goes to the split with the largest remaining deficit,
    # ties broken train > val > test so small datasets keep a usable train set.
    for g in groups:
        split = max(
            (TRAIN, VAL, TEST),
            key=lambda s: (targets[s] - filled[s], s == TRAIN, s == VAL),
        )
        group_split[g] = split
        filled[split] += group_sizes[g]

    assignment = {tid: group_split[g] for tid, g in effective_group.items()}

    for split in (TRAIN, VAL, TEST):
        if filled[split] == 0:
            warnings.append(
                f"split {split!r} is EMPTY — dataset has too few groups "
                f"({len(groups)}) for ratios {ratios}"
            )

    # Label-balance sanity: a big positive-rate gap between splits usually
    # means too few groups, and validation metrics will be noisy.
    rates = {}
    for split in (TRAIN, VAL, TEST):
        rows = [by_id[t] for t, s in assignment.items() if s == split]
        if rows:
            rates[split] = sum(r.local_ok for r in rows) / len(rows)
    if rates and (max(rates.values()) - min(rates.values())) > 0.15:
        warnings.append(
            "local_ok rate differs by more than 15 points across splits "
            f"({ {k: round(v, 2) for k, v in rates.items()} }) — validation "
            "metrics will be noisy; consider more data or a different seed"
        )

    return SplitResult(
        assignment=assignment,
        seed=seed,
        ratios=ratios,
        merged_groups=merged,
        warnings=warnings,
    )
