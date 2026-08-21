#!/usr/bin/env python3
"""Merge compatible real-outcome datasets with provenance and validation."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Dict, List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from routing.dataset import (
    OutcomeDataset,
    load_dataset,
    normalize_prompt,
    save_dataset,
)


COMPATIBLE_META = (
    "local_model",
    "remote_model",
    "quality_threshold",
    "tier_mode",
    "system_prompt_sha256",
    "temperature",
    "cheap_max_tokens",
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge(paths: List[str]) -> OutcomeDataset:
    if len(paths) < 2:
        raise ValueError("at least two input datasets are required")
    datasets = [load_dataset(path) for path in paths]
    first = datasets[0]
    for path, dataset in zip(paths[1:], datasets[1:]):
        for key in COMPATIBLE_META:
            if dataset.meta.get(key) != first.meta.get(key):
                raise ValueError(
                    f"incompatible meta.{key} in {path}: "
                    f"{dataset.meta.get(key)!r} != {first.meta.get(key)!r}"
                )

    records = [record for dataset in datasets for record in dataset.records]
    ids: Dict[str, str] = {}
    prompts: Dict[str, str] = {}
    for record in records:
        if record.task_id in ids:
            raise ValueError(f"duplicate task_id across inputs: {record.task_id}")
        ids[record.task_id] = record.task_id
        prompt = normalize_prompt(record.prompt)
        if prompt in prompts:
            raise ValueError(
                f"duplicate prompt across inputs: {prompts[prompt]} / "
                f"{record.task_id}"
            )
        prompts[prompt] = record.task_id

    meta = dict(first.meta)
    meta["strong_max_tokens"] = sorted(
        {int(dataset.meta.get("strong_max_tokens", 0)) for dataset in datasets}
    )
    meta["source_datasets"] = [
        {
            "path": path,
            "sha256": _sha256(path),
            "records": len(dataset.records),
            "strong_max_tokens": dataset.meta.get("strong_max_tokens"),
        }
        for path, dataset in zip(paths, datasets)
    ]
    meta["notes"] = (
        "Merged compatible real-outcome collections. Per-source generation "
        "caps and hashes are recorded in meta.source_datasets."
    )
    return OutcomeDataset(meta=meta, records=records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dataset = merge(args.input)
    save_dataset(dataset, args.out)
    print(
        f"wrote {len(dataset.records)} records to {args.out} "
        f"(hash {dataset.content_hash()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
