#!/usr/bin/env python3
"""Inspect a deployment identity, or verify a collection before transfer.

    # On the inference machine, BEFORE any collection: report the exact
    # deployment the collector would record. No generation, no Fireworks
    # call; with an Ollama tier it makes three local metadata requests.
    python scripts/check_collection.py identity --json-out identity.json

    # After a collection (sender) and after transfer (receiver):
    python scripts/check_collection.py verify --out data/pilot.json \
        --tasks tasks/pair_v2/pilot.json --write-checksums data/pilot.sha256

Never prints credentials: settings are reported by name/value only for
non-secret fields, and the Ollama URL by host only.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections import Counter
from typing import Any, Dict, List
from urllib.parse import urlsplit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from routing.identity import (  # noqa: E402
    canonical_file_sha256,
    canonical_json_sha256,
    file_sha256,
    runtime_identity,
)

_SUFFIXES = (".manifest.json", ".work.jsonl", "", ".summary.json")


def identity_report(probe: bool = True) -> Dict[str, Any]:
    from config import settings

    facts: Dict[str, Any] = {}
    error = None
    tier = settings.tier_mode.strip().lower()
    if probe and tier == "local_remote" and settings.local_backend.strip().lower() == "ollama":
        from ollama_client import model_identity
        try:
            facts = model_identity(settings.local_model_name)
        except Exception as err:  # report, never guess
            error = f"{type(err).__name__}: {err}"
    identity = runtime_identity(probe=lambda _m: facts)
    unknown = sorted(
        f"{tier_name}.{k}" for tier_name in ("cheap", "strong")
        for k, v in identity[tier_name].items() if v is None)
    return {
        "kind": "nana-deployment-identity",
        "pair_identity": identity,
        "unknown_fields": unknown,
        "probe_error": error,
        "local_runtime": {k: facts.get(k) for k in
                          ("runtime_version", "format", "family", "parameter_size")},
        "settings": {
            "TIER_MODE": settings.tier_mode,
            "LOCAL_BACKEND": settings.local_backend,
            "LOCAL_MODEL_NAME": settings.local_model_name,
            "OLLAMA_HOST": urlsplit(settings.ollama_base_url).hostname,
            "LOCAL_CONTEXT_TOKENS": settings.local_context_tokens,
            "LOCAL_MAX_NEW_TOKENS": settings.local_max_new_tokens,
            "CHEAP_MODEL_NAME": settings.cheap_model_name,
            "STRONG_MODEL_NAME": settings.strong_model_name,
            "STRONG_MAX_TOKENS": settings.strong_max_tokens,
            "FIREWORKS_API_KEY_SET": bool(settings.fireworks_api_key),
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_rows(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def verify(out: str, tasks_path: str = None) -> Dict[str, Any]:
    """Consistency of manifest, attempt rows, dataset and summary. Returns
    {"problems": [...], "files": {...}}; empty problems = consistent."""
    problems: List[str] = []
    files: Dict[str, Dict[str, Any]] = {}
    for suffix in _SUFFIXES:
        path = out + suffix
        if os.path.exists(path):
            entry = {"sha256": file_sha256(path)}
            if suffix != ".work.jsonl":
                entry["canonical_sha256"] = canonical_file_sha256(path)
            files[path] = entry
        elif suffix in (".manifest.json", ".work.jsonl"):
            problems.append(f"missing {path}")
    if problems:
        return {"problems": problems, "files": files}

    manifest = _read_json(out + ".manifest.json")
    if manifest.get("kind") != "nana-collection-manifest":
        problems.append("manifest kind is not nana-collection-manifest")
    compat = manifest.get("compatibility") or {}
    identity = compat.get("pair_identity") or {}
    run_ids = {r.get("run_id") for r in manifest.get("runs", [])}
    rows = _read_rows(out + ".work.jsonl")

    attempt_ids = Counter(r.get("attempt_id") for r in rows)
    repeated = [a for a, n in attempt_ids.items() if n > 1 or a is None]
    if repeated:
        problems.append(f"missing or repeated attempt_id: {repeated[:5]}")
    stray = sorted({r.get("run_id") for r in rows} - run_ids)
    if stray:
        problems.append(f"attempt rows from runs not in the manifest: {stray[:5]}")
    expected = (identity.get("tier_mode"), (identity.get("cheap") or {}).get("model"),
                (identity.get("strong") or {}).get("model"))
    mixed = sorted({(r.get("tier_mode"), r.get("cheap_model"), r.get("strong_model"))
                    for r in rows} - {expected})
    if mixed:
        problems.append(f"rows for a different pair than the manifest: {mixed[:3]}")

    if tasks_path:
        tasks = _read_json(tasks_path)
        if canonical_json_sha256(tasks) != (compat.get("tasks") or {}).get("canonical_sha256"):
            problems.append("task file content differs from the manifest")
        unknown_ids = {r["task_id"] for r in rows} - {str(t["task_id"]) for t in tasks}
        if unknown_ids:
            problems.append(f"rows for tasks not in the task file: {sorted(unknown_ids)[:5]}")

    completed = {r["task_id"] for r in rows if r.get("status", "complete") == "complete"}
    if os.path.exists(out):
        dataset = _read_json(out)
        meta = dataset.get("meta") or {}
        if meta.get("pair_identity") != identity:
            problems.append("dataset pair_identity differs from the manifest")
        if meta.get("label_identity") != compat.get("label_identity"):
            problems.append("dataset label_identity differs from the manifest")
        record_ids = {r["task_id"] for r in dataset.get("records", [])}
        if record_ids - completed:
            problems.append(f"dataset records without a completed attempt: "
                            f"{sorted(record_ids - completed)[:5]}")
        summary_path = out + ".summary.json"
        if os.path.exists(summary_path):
            acc = _read_json(summary_path).get("accounting") or {}
            if acc.get("usable_paired_outcomes") != len(record_ids):
                problems.append("summary usable_paired_outcomes != dataset records")
    if os.path.exists(out + ".summary.json"):
        acc = _read_json(out + ".summary.json").get("accounting") or {}
        checks = {"attempts_total": len(rows),
                  "tasks_attempted": len({r["task_id"] for r in rows}),
                  "tasks_completed": len(completed)}
        for key, value in checks.items():
            if acc.get(key) != value:
                problems.append(f"summary {key}={acc.get(key)} but work file gives {value}")
    else:
        problems.append(f"missing {out}.summary.json")
    return {"problems": problems, "files": files,
            "synthetic": bool(identity.get("synthetic")), "attempt_rows": len(rows)}


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ident = sub.add_parser("identity", help="report the deployment identity")
    ident.add_argument("--no-probe", action="store_true",
                       help="do not contact Ollama (fields stay unknown)")
    ident.add_argument("--json-out", default=None)
    ver = sub.add_parser("verify", help="check a collection's files")
    ver.add_argument("--out", required=True, help="the collector's --out path")
    ver.add_argument("--tasks", default=None, help="task file used for the collection")
    ver.add_argument("--write-checksums", default=None,
                     help="write exact SHA-256 lines (sha256sum -c format)")
    args = parser.parse_args(argv)

    if args.command == "identity":
        report = identity_report(probe=not args.no_probe)
        text = json.dumps(report, indent=1, ensure_ascii=False)
        print(text)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text + "\n")
        return 0 if not report["unknown_fields"] else 1

    result = verify(args.out, args.tasks)
    for path, entry in result["files"].items():
        print(f"{entry['sha256']}  {os.path.basename(path)}"
              + (f"  (canonical {entry['canonical_sha256']})" if "canonical_sha256" in entry else ""))
    if result.get("synthetic"):
        print("NOTE: synthetic (mock) collection — pipeline evidence only")
    if args.write_checksums:
        with open(args.write_checksums, "w", encoding="utf-8", newline="\n") as fh:
            for path, entry in result["files"].items():
                fh.write(f"{entry['sha256']}  {os.path.basename(path)}\n")
    for problem in result["problems"]:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    print("consistent" if not result["problems"] else f"{len(result['problems'])} problem(s)")
    return 0 if not result["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
