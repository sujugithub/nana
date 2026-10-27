"""Collect REAL outcomes from BOTH configured tiers, grade both answers,
and write an outcome dataset in routing/dataset.py format.

    # dry run of the whole pipeline, no models, no network, no cost:
    python3 scripts/collect_outcomes.py --tasks tasks/graded_tasks.json \
        --out data/collected.json --mock

    # REAL run — makes paid Fireworks calls, so it must be explicit:
    python3 scripts/collect_outcomes.py --tasks tasks/graded_tasks.json \
        --out data/collected.json --run-paid-calls --max-api-cost-usd 1.00 \
        --local-context-tokens 8192

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

Grading: programmatic wherever possible (EVALUATION.md), via
routing/grading.py. Tasks without a reference+grader, and answers the grader
flags as ambiguous or contradictory, get quality -1 sentinels and need
manual grading; merge verdicts back with --grades
(JSON {task_id: {"cheap": 0..1, "strong": 0..1}}; either key may be omitted
when only one tier needed review). Legacy ``local`` and ``remote`` verdict
keys remain accepted. Each work row keeps the raw answers unchanged plus
``local_grade``/``remote_grade`` status, flags and extracted answers.

Files written next to --out:
    <out>.manifest.json  compatibility identity (pair, labels, task set) and
                         informational provenance; one entry per run
    <out>.work.jsonl     one row per ATTEMPT (run_id, attempt_id, status,
                         error, raw answers, finish reasons, usage source,
                         backend timings, grades); failed attempts are kept
    <out>                the graded dataset (completed + graded tasks only)
    <out>.summary.json   accounting: attempted / completed / failed / graded
                         / unresolved, with the denominator of every rate

Cost model: Fireworks tiers use API usage counts at configured $/Mtoken
rates. A truly local cheap tier uses wall-clock time at an amortised
$/compute-second rate. Every rate is an explicit flag.

SAFETY RAILS
- Never runs paid calls unless --run-paid-calls is given (or --mock).
- Refuses --run-paid-calls without FIREWORKS_API_KEY set. ``remote_pair``
  makes two paid calls per task; ``local_remote`` makes one.
- Appends after every attempt (crash-safe). A resumed run skips completed
  tasks and retries failed ones, after checking that the deployment, label
  definition and task file are identical to the manifest's.
- Stops after --max-consecutive-failures failed attempts in a row, so an
  outage cannot keep spending.
- If the cheap tier fails, the strong tier is not called for that attempt.
- --max-api-cost-usd is a TOTAL budget across every run of the same --out.
  Each attempt row records its spend (``api_spend``); calls whose billing is
  unknown (failed after sending, or usage omitted) are charged at their full
  per-call ceiling. A run is refused if already-charged + the ceiling for
  unfinished tasks exceeds the budget, and stops (exit 4) before any task
  whose ceiling could cross it. Transport retries inside remote_client
  (429/5xx/connection errors) are assumed unbilled; that is not verified.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _load_dotenv() -> None:
    """Load repo .env without replacing explicitly exported variables."""
    try:
        with open(os.path.join(REPO, ".env"), encoding="utf-8") as fh:
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
from routing.grading import GRADER_VERSION, grade_answer  # noqa: E402
from routing.identity import (  # noqa: E402
    canonical_json_sha256,
    file_sha256,
    runtime_identity,
    sha256_text,
)

MANIFEST_VERSION = "1"
_DEPENDENCIES = ("requests", "numpy", "scikit-learn", "joblib", "torch", "transformers")
_ERROR_MESSAGE_CHARS = 500


def configured_api_cost_ceiling(
    tasks: List[dict], done: Dict[str, dict], args: argparse.Namespace
) -> float:
    """Configured API-cost ceiling for unfinished tasks.

    Assumes ONE paid attempt per tier per unfinished task, every Fireworks
    response consuming its full max-token cap, and each UTF-8 input byte
    (plus system prompt) as a token. It does NOT cover transport retries
    inside remote_client (429/5xx/connection errors) or repeat attempts on a
    later resume, so it is a budget guard on configured rates, not a
    worst-case bound or a provider guarantee. Local generation is excluded
    because it has no API charge.
    """
    remaining = [t for t in tasks if str(t["task_id"]) not in done]
    return sum(sum(_call_ceilings(t["prompt"], args)) for t in remaining)


def _call_ceilings(prompt: str, args: argparse.Namespace) -> Tuple[float, float]:
    """(cheap, strong) configured cost ceiling of ONE paid call for a prompt:
    full output cap, each UTF-8 input byte (plus system prompt) as a token.
    The cheap ceiling is 0 when the cheap tier is local (no API charge)."""
    input_upper = len((settings.system_prompt + "\n" + prompt).encode("utf-8")) + 64
    strong = (input_upper * args.strong_in_per_mtok
              + settings.strong_max_tokens * args.strong_out_per_mtok) / 1e6
    if settings.tier_mode.strip().lower() != "remote_pair":
        return 0.0, strong
    cheap = (input_upper * args.cheap_in_per_mtok
             + settings.cheap_max_tokens * args.cheap_out_per_mtok) / 1e6
    return cheap, strong


def row_spend(row: dict, args: argparse.Namespace) -> Dict[str, Any]:
    """API spend of one attempt row, for the cumulative budget.

    A paid call with provider-reported usage is charged at usage x configured
    rates. A paid call whose billing is unknown — it failed after being
    sent, or the provider omitted usage (estimated counts) — is charged at
    its full per-call ceiling. Calls never made (skipped strong call, local
    cheap tier) cost 0. Rates are the configured ones, not provider invoices.
    """
    cheap_cap, strong_cap = _call_ceilings(row.get("prompt", ""), args)
    error_tier = (row.get("error") or {}).get("tier")
    calls = []  # (made, usage_source, prompt_tokens, completion_tokens, in, out, cap)
    if row.get("cheap_provider") == "fireworks":
        calls.append((True, row.get("local_usage_source") if error_tier != "cheap" else None,
                      row.get("local_prompt_tokens"), row.get("local_completion_tokens"),
                      args.cheap_in_per_mtok, args.cheap_out_per_mtok, cheap_cap))
    strong_call = row.get("strong_call") or ("completed" if row.get("status", "complete")
                                             == "complete" else "failed")
    if strong_call != "skipped":
        calls.append((True, row.get("remote_usage_source") if strong_call == "completed" else None,
                      row.get("remote_prompt_tokens"), row.get("remote_completion_tokens"),
                      args.strong_in_per_mtok, args.strong_out_per_mtok, strong_cap))
    recorded = charged = 0.0
    unknown = 0
    for _made, source, p_tok, c_tok, rate_in, rate_out, cap in calls:
        if source in ("provider", "mock") and p_tok is not None and c_tok is not None:
            cost = (p_tok * rate_in + c_tok * rate_out) / 1e6
            recorded += cost
            charged += cost
        else:
            unknown += 1
            charged += cap
    return {"recorded_usd": round(recorded, 9), "charged_for_budget_usd": round(charged, 9),
            "unknown_billing_calls": unknown, "paid_calls": len(calls)}


def spend_summary(attempts: List[dict], args: argparse.Namespace,
                  budget: Optional[float]) -> Dict[str, Any]:
    rows = [row_spend(r, args) for r in attempts]
    return {
        "budget_usd": budget,
        "recorded_usd": round(sum(r["recorded_usd"] for r in rows), 9),
        "charged_for_budget_usd": round(sum(r["charged_for_budget_usd"] for r in rows), 9),
        "paid_calls": sum(r["paid_calls"] for r in rows),
        "unknown_billing_calls": sum(r["unknown_billing_calls"] for r in rows),
        "mock": bool(settings.mock_mode),
        "basis": ("provider-reported usage x configured rates; calls with unknown "
                  "billing (failed after sending, or usage omitted) charged at "
                  "their full per-call ceiling; cumulative over all runs of this "
                  "--out; not a provider invoice"),
    }


def grade(grader: str, reference: str, answer: str) -> float:
    """Legacy 0/1 grade. Delegates to routing.grading.grade_answer; answers
    that need human review score 0.0 here, so callers that must tell
    "wrong" from "uncertain" use grade_answer directly (collect does)."""
    return grade_answer(grader, reference, answer).score


# ── Files ────────────────────────────────────────────────────────────────

def _write_json(path: str, obj: Any) -> None:
    """Write via a temporary file so an interruption never leaves half a file."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _load_tasks(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        tasks = json.load(fh)
    ids = [str(t["task_id"]) for t in tasks]
    duplicates = sorted(i for i, n in Counter(ids).items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate task_id(s) in {path}: {duplicates[:10]}")
    return tasks


def _load_attempts(path: str) -> List[dict]:
    """Every attempt row in the JSONL work file, in order."""
    rows: List[dict] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _is_complete(row: dict) -> bool:
    # Rows written before S2 have no status and were only ever successes.
    return row.get("status", "complete") == "complete"


def _load_existing(path: str) -> Dict[str, dict]:
    """Completed rows by task_id (latest wins) — the resume skip-set."""
    return {row["task_id"]: row for row in _load_attempts(path) if _is_complete(row)}


# ── Manifest ─────────────────────────────────────────────────────────────

def _manifest_path(out: str) -> str:
    return out + ".manifest.json"


def _git(*args: str) -> Optional[str]:
    """Read-only git query; --no-optional-locks keeps git from touching the index."""
    try:
        done = subprocess.run(
            ["git", "--no-optional-locks", *args], cwd=REPO,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _shareable_path(path: str) -> str:
    """Repo-relative path, or the bare file name for files outside the repo,
    so manifests never record personal absolute paths."""
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, REPO)
    except ValueError:  # different drive on Windows
        return os.path.basename(absolute)
    if relative.startswith(".."):
        return os.path.basename(absolute)
    return relative.replace(os.sep, "/")


def _dependency_versions() -> Dict[str, Optional[str]]:
    from importlib import metadata

    versions: Dict[str, Optional[str]] = {}
    for name in _DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest(args: argparse.Namespace, tasks: List[dict],
                   identity: Dict[str, Any], facts: Dict[str, Any]) -> Dict[str, Any]:
    """``compatibility`` is compared on resume; ``provenance`` never is."""
    status = _git("status", "--porcelain")
    return {
        "kind": "nana-collection-manifest",
        "manifest_version": MANIFEST_VERSION,
        "compatibility": {
            "pair_identity": identity,
            "label_identity": {
                "grader_version": GRADER_VERSION,
                "quality_threshold": args.quality_threshold,
            },
            "tasks": {
                "canonical_sha256": canonical_json_sha256(tasks),
                "count": len(tasks),
            },
        },
        "provenance": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "repository_commit": _git("rev-parse", "HEAD"),
            "working_tree_dirty": None if status is None else bool(status),
            "tasks_path": _shareable_path(args.tasks),
            "tasks_file_sha256": file_sha256(args.tasks),
            "mock": bool(settings.mock_mode),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(),
            "dependencies": _dependency_versions(),
            "local_runtime": {k: facts.get(k) for k in
                              ("runtime_version", "format", "family", "parameter_size")},
            "strong_revision": "not exposed by the provider",
            "seed": None,
            "sampling": "temperature 0 (greedy); no sampling seed",
            "split": "not assigned at collection; routing.train assigns splits",
            "cost_assumptions": {
                "cheap_in_per_mtok": args.cheap_in_per_mtok,
                "cheap_out_per_mtok": args.cheap_out_per_mtok,
                "strong_in_per_mtok": args.strong_in_per_mtok,
                "strong_out_per_mtok": args.strong_out_per_mtok,
                "local_cost_per_second": args.local_cost_per_second,
            },
            "budget_usd": getattr(args, "max_api_cost_usd", None),
        },
        "runs": [],
    }


def _differing_paths(a: Any, b: Any, prefix: str = "") -> List[str]:
    """Dotted paths whose values differ between two nested dicts."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for key in sorted(set(a) | set(b)):
            out += _differing_paths(a.get(key), b.get(key),
                                    f"{prefix}.{key}" if prefix else str(key))
        return out
    return [] if a == b else [prefix or "(root)"]


def resume_problem(existing: Dict[str, Any], fresh: Dict[str, Any]) -> Optional[str]:
    """None when appending to ``existing`` is safe, else the reason. A resume
    must continue the SAME experiment: identical compatibility block."""
    if existing.get("manifest_version") != MANIFEST_VERSION:
        return (f"manifest_version {existing.get('manifest_version')!r} is not "
                f"{MANIFEST_VERSION!r}")
    old, new = existing.get("compatibility") or {}, fresh["compatibility"]
    problems = []
    # Informational fields (runtime version) never block; collect() warns.
    fields = [f for f in _differing_paths(old.get("pair_identity"), new["pair_identity"])
              if not f.startswith("informational")]
    if fields:
        problems.append(f"model-pair identity differs ({', '.join(fields)})")
    if old.get("label_identity") != new["label_identity"]:
        problems.append(f"label identity differs: {old.get('label_identity')} "
                        f"vs {new['label_identity']}")
    if old.get("tasks") != new["tasks"]:
        problems.append("task file content differs")
    return "; ".join(problems) or None


# ── Collection ───────────────────────────────────────────────────────────

def _tier_fields(prefix: str, completion: Any, latency: float) -> Dict[str, Any]:
    return {
        f"{prefix}_answer": completion.text,
        f"{prefix}_answer_raw": getattr(completion, "raw_text", None),
        f"{prefix}_prompt_tokens": completion.prompt_tokens,
        f"{prefix}_completion_tokens": completion.completion_tokens,
        f"{prefix}_latency_s": round(latency, 3),
        f"{prefix}_finish_reason": getattr(completion, "finish_reason", None),
        f"{prefix}_usage_source": getattr(completion, "usage_source", ""),
        f"{prefix}_timings": getattr(completion, "timings", {}) or {},
    }


def _error(tier: str, err: BaseException) -> Dict[str, str]:
    return {"tier": tier, "type": type(err).__name__,
            "message": str(err)[:_ERROR_MESSAGE_CHARS]}


def collect(args: argparse.Namespace) -> int:
    tasks = _load_tasks(args.tasks)
    work_path = args.out + ".work.jsonl"
    attempts = _load_attempts(work_path)
    done = {r["task_id"]: r for r in attempts if _is_complete(r)}
    if attempts:
        print(f"resuming: {len(done)} task(s) completed, "
              f"{len(attempts)} attempt row(s) in {work_path}")

    budget = getattr(args, "max_api_cost_usd", None)
    spent = spend_summary(attempts, args, budget)["charged_for_budget_usd"]
    if not settings.mock_mode:
        # --max-api-cost-usd is the TOTAL budget for this --out, across every
        # run and resume, not a per-run allowance.
        ceiling = configured_api_cost_ceiling(tasks, done, args)
        if budget is None:
            print(
                f"refusing paid collection: already charged ${spent:.4f}; the "
                f"configured API-cost ceiling for unfinished tasks is "
                f"${ceiling:.4f} (one paid attempt per tier at the full output "
                "cap; excludes transport retries). Pass --max-api-cost-usd with "
                "the TOTAL budget for this collection, at least the sum.",
                file=sys.stderr,
            )
            return 2
        if budget <= 0 or spent + ceiling > budget + 1e-12:
            print(
                f"refusing paid collection: already charged ${spent:.4f} + "
                f"ceiling ${ceiling:.4f} for unfinished tasks exceeds the total "
                f"budget --max-api-cost-usd ${budget:.4f}.",
                file=sys.stderr,
            )
            return 2
        print(
            f"paid-collection guard: charged so far ${spent:.4f} + ceiling "
            f"${ceiling:.4f} for {len(tasks) - len(done)} unfinished task(s) "
            f"<= total budget ${budget:.4f}",
            file=sys.stderr,
        )

    context = getattr(args, "local_context_tokens", None)
    if context:
        settings.local_context_tokens = context

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

    facts: Dict[str, Any] = {}
    if (tier_mode == "local_remote" and settings.local_backend == "ollama"
            and not settings.mock_mode):
        from ollama_client import model_identity
        try:
            facts = model_identity(cheap.model_name)
        except Exception as err:
            print(f"refusing to collect: cannot read the local model's "
                  f"identity from Ollama ({err})", file=sys.stderr)
            return 2
    identity = runtime_identity(probe=lambda _model: facts,
                                synthetic=bool(settings.mock_mode))
    manifest = build_manifest(args, tasks, identity, facts)
    manifest_path = _manifest_path(args.out)
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            existing = json.load(fh)
        problem = resume_problem(existing, manifest)
        if problem:
            print(f"refusing to resume {work_path}: {problem}. Collect this "
                  "configuration to a new --out path.", file=sys.stderr)
            return 2
        changed = [f for f in _differing_paths(
            existing["compatibility"].get("pair_identity"),
            manifest["compatibility"]["pair_identity"]) if f.startswith("informational")]
        if changed:
            print(f"WARNING: resuming with changed {', '.join(changed)}; outputs may "
                  "differ between runs — each run's value is recorded in the "
                  "manifest; review before using this dataset", file=sys.stderr)
        manifest = existing
    elif attempts:
        print(f"refusing to resume {work_path}: it has no manifest, so its "
              "deployment identity cannot be verified. Finalize it with "
              "--finalize-only or collect to a new --out path.", file=sys.stderr)
        return 2

    expected_pair = (tier_mode, cheap.model_name, strong.model_name)
    for row in attempts:
        found = (row.get("tier_mode"), row.get("cheap_model"), row.get("strong_model"))
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

    run_id = uuid.uuid4().hex
    manifest["runs"].append({
        "run_id": run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repository_commit": _git("rev-parse", "HEAD"),
        "local_runtime_version": facts.get("runtime_version"),
        "total_budget_usd": budget,
        "charged_before_run_usd": round(spent, 9),
        "rates_per_mtok": {
            "cheap_in": args.cheap_in_per_mtok, "cheap_out": args.cheap_out_per_mtok,
            "strong_in": args.strong_in_per_mtok, "strong_out": args.strong_out_per_mtok,
        },
    })
    _write_json(manifest_path, manifest)

    attempt_counts = Counter(r["task_id"] for r in attempts)
    max_failures = getattr(args, "max_consecutive_failures", 3) or 0
    consecutive_failures = 0
    stopped = False
    budget_stop = False
    ungraded: List[str] = []
    with open(work_path, "a", encoding="utf-8", newline="\n") as work:
        for i, task in enumerate(tasks):
            tid = str(task["task_id"])
            if tid in done:
                continue
            prompt = task["prompt"]
            if not settings.mock_mode and budget is not None:
                # Defence in depth: never start a task whose ceiling could
                # cross the total budget (unknown-billing failures in this run
                # are charged at their ceiling, which can exhaust it).
                task_cap = sum(_call_ceilings(prompt, args))
                if spent + task_cap > budget + 1e-12:
                    print(f"stopping: charged ${spent:.4f}; the next task's ceiling "
                          f"${task_cap:.4f} would exceed the total budget "
                          f"${budget:.4f}", file=sys.stderr)
                    budget_stop = True
                    break
            attempt_counts[tid] += 1
            print(f"[{i + 1}/{len(tasks)}] {tid} (attempt {attempt_counts[tid]})",
                  flush=True)
            row: Dict[str, Any] = {
                "task_id": tid,
                "run_id": run_id,
                "attempt_id": uuid.uuid4().hex,
                "attempt_number": attempt_counts[tid],
                "prompt": prompt,
                "category": task.get("category", "unknown"),
                "source": task.get("source", "unknown"),
                "group_id": task.get("group_id", ""),
                "metadata": task.get("metadata") or {},
                "tier_mode": tier_mode,
                "cheap_model": cheap.model_name,
                "cheap_provider": "fireworks" if tier_mode == "remote_pair" else "local",
                "strong_model": strong.model_name,
                "strong_provider": "fireworks",
                "status": "error",
                "error": None,
                # completed | failed | skipped (with strong_skipped_reason)
                "strong_call": "skipped",
                "strong_skipped_reason": "cheap_tier_failed",
            }
            try:
                t0 = time.time()
                cheap_completion = cheap.generate(prompt)
                cheap_latency = time.time() - t0
            except Exception as err:  # infrastructure failure, not a wrong answer
                row["error"] = _error("cheap", err)
                cheap_completion = None
            if cheap_completion is not None:
                # Schema 1.0 uses local/remote field names. For new data they
                # intentionally mean cheap/strong, preserving Fable's model.
                row.update(_tier_fields("local", cheap_completion, cheap_latency))
                row["cheap_model"] = cheap_completion.model_name or row["cheap_model"]
                row["cheap_provider"] = cheap_completion.provider or row["cheap_provider"]
                row["local_confidence"] = cheap_completion.confidence
                row["local_min_token_prob"] = cheap_completion.min_token_prob
                row["local_low_token_frac"] = cheap_completion.low_token_frac
                row["post_check_problems"] = router.post_check(
                    prompt, cheap_completion.text)[1]
                row["strong_call"], row["strong_skipped_reason"] = "failed", None
                try:
                    t0 = time.time()
                    strong_completion = strong.generate(prompt)
                    strong_latency = time.time() - t0
                except Exception as err:
                    row["error"] = _error("strong", err)
                    strong_completion = None
                if strong_completion is not None:
                    row.update(_tier_fields("remote", strong_completion, strong_latency))
                    row["strong_model"] = strong_completion.model_name or row["strong_model"]
                    row["status"], row["strong_call"] = "complete", "completed"

            if row["status"] == "complete":
                consecutive_failures = 0
                _grade_row(row, task, ungraded)
                done[tid] = row
            else:
                consecutive_failures += 1
                print(f"  attempt failed ({row['error']['tier']}: "
                      f"{row['error']['type']})", file=sys.stderr)
            row["api_spend"] = row_spend(row, args)
            spent += row["api_spend"]["charged_for_budget_usd"]
            work.write(json.dumps(row, ensure_ascii=False) + "\n")
            work.flush()
            if max_failures and consecutive_failures >= max_failures:
                print(f"stopping: {consecutive_failures} consecutive failed "
                      "attempts; check the backends, then rerun to resume",
                      file=sys.stderr)
                stopped = True
                break

    if ungraded:
        print(
            f"{len(ungraded)} task(s) need human grading (no grader/reference, "
            f"or an ambiguous/contradictory answer) — review their answers "
            f"(they are in {work_path}) and rerun with --grades verdicts.json"
        )

    code = finalize(args, done)
    if budget_stop:
        return 4
    return 3 if stopped else code


def _grade_row(row: Dict[str, Any], task: dict, ungraded: List[str]) -> None:
    grader = task.get("grader")
    reference = task.get("reference")
    cheap_grade = strong_grade = None
    if grader and reference is not None:
        # Grades are computed on a grading copy; the raw answers stored in
        # the row are exactly what the models returned.
        cheap_grade = grade_answer(grader, str(reference), row["local_answer"],
                                   row.get("local_finish_reason"))
        strong_grade = grade_answer(grader, str(reference), row["remote_answer"],
                                    row.get("remote_finish_reason"))
        # -1 sentinel = needs external (human) grading via --grades.
        cheap_q = -1.0 if cheap_grade.needs_review else cheap_grade.score
        strong_q = -1.0 if strong_grade.needs_review else strong_grade.score
        if cheap_grade.needs_review or strong_grade.needs_review:
            ungraded.append(row["task_id"])
    else:
        cheap_q = strong_q = -1.0  # sentinel: needs external grading
        ungraded.append(row["task_id"])
    row["local_quality"] = cheap_q
    row["remote_quality"] = strong_q
    # Grading status/flags (empty, refusal, truncated, review reasons)
    # recorded separately from correctness.
    row["local_grade"] = cheap_grade.to_dict() if cheap_grade else None
    row["remote_grade"] = strong_grade.to_dict() if strong_grade else None


# ── Finalize ─────────────────────────────────────────────────────────────

def _rate(numerator: int, denominator: int, num_name: str, den_name: str) -> Dict[str, Any]:
    return {"value": (numerator / denominator) if denominator else None,
            "numerator": num_name, "denominator": den_name}


def _strong_call(row: dict) -> Tuple[str, Optional[str]]:
    """(completed|failed|skipped, skip reason). Rows written before the
    strong_call field existed are inferred from status and error tier."""
    if row.get("strong_call"):
        return row["strong_call"], row.get("strong_skipped_reason")
    if _is_complete(row):
        return "completed", None
    if (row.get("error") or {}).get("tier") == "cheap":
        return "skipped", "cheap_tier_failed"
    return "failed", None


def account(task_ids: Optional[List[str]], attempts: List[dict],
            done: Dict[str, dict],
            final_quality: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """Every task is counted somewhere; every rate names its denominator.

    ``tasks_*`` count unique task IDs; ``attempts_*`` and ``strong_calls``
    count attempt rows, so a task retried on resume adds attempts but is
    still one task. ``final_quality`` holds the (cheap, strong) quality of
    every completed task after verdicts; -1 means awaiting human review.
    Only tasks with both tiers graded are usable paired outcomes; the rest
    stay counted here and never become training labels.
    """
    attempted = {r["task_id"] for r in attempts} | set(done)
    completed = set(done)
    failed = attempted - completed
    graded = {t for t in completed
              if t in final_quality and min(final_quality[t]) >= 0}
    unresolved_by_tier = {
        "cheap": sum(1 for t in completed - graded if final_quality.get(t, (-1, -1))[0] < 0),
        "strong": sum(1 for t in completed - graded if final_quality.get(t, (-1, -1))[1] < 0),
    }
    flags = {"cheap": Counter(), "strong": Counter()}
    finish = {"cheap": Counter(), "strong": Counter()}
    for tid in completed:
        row = done[tid]
        for tier, prefix in (("cheap", "local"), ("strong", "remote")):
            flags[tier].update((row.get(f"{prefix}_grade") or {}).get("flags", []))
            finish[tier][str(row.get(f"{prefix}_finish_reason"))] += 1
    errors = Counter((r.get("error") or {}).get("tier") for r in attempts
                     if r.get("status") == "error")
    calls = Counter(_strong_call(r)[0] for r in attempts)
    skipped_reasons = Counter(_strong_call(r)[1] for r in attempts
                              if _strong_call(r)[0] == "skipped")
    n = len(graded)
    return {
        "tasks_in_file": len(task_ids) if task_ids is not None else None,
        "tasks_not_attempted": (len(set(task_ids) - attempted)
                                if task_ids is not None else None),
        "tasks_attempted": len(attempted),
        "tasks_completed": len(completed),
        "tasks_failed_infrastructure": len(failed),
        "tasks_graded": n,
        "usable_paired_outcomes": n,
        "tasks_unresolved_review": len(completed - graded),
        "unresolved_review_by_tier": unresolved_by_tier,
        "attempts_total": len(attempts),
        "attempts_failed": sum(errors.values()),
        "failed_attempts_by_tier": dict(errors),
        "strong_calls": {
            "completed": calls["completed"],
            "failed": calls["failed"],
            "skipped": calls["skipped"],
            "skipped_by_reason": dict(skipped_reasons),
            "unit": "attempts",
        },
        "grade_flags": {tier: dict(c) for tier, c in flags.items()},
        # Finish status is reported separately from correctness. A reply cut
        # at the cap with no usable answer counts as unsuccessful under the
        # fixed generation budget, not as proof of missing knowledge.
        "finish_reasons": {tier: dict(c) for tier, c in finish.items()},
        "rates": {
            "completion_rate": _rate(len(completed), len(attempted),
                                     "tasks_completed", "tasks_attempted"),
            "infrastructure_failure_rate": _rate(len(failed), len(attempted),
                                                 "tasks_failed_infrastructure",
                                                 "tasks_attempted"),
            "grading_coverage": _rate(n, len(completed), "tasks_graded",
                                      "tasks_completed"),
        },
        "quality": {
            "cheap_mean": (sum(final_quality[t][0] for t in graded) / n) if n else None,
            "strong_mean": (sum(final_quality[t][1] for t in graded) / n) if n else None,
            "denominator": "usable_paired_outcomes",
            "n": n,
        },
    }


def finalize(args: argparse.Namespace, done: Dict[str, dict]) -> int:
    """Apply external grades if given, compute costs, write the dataset and
    the accounting summary."""
    grades: Dict[str, dict] = {}
    if args.grades:
        with open(args.grades, encoding="utf-8") as fh:
            grades = json.load(fh)

    records = []
    # (cheap, strong) after verdicts for EVERY completed task; -1 = awaiting
    # review. Only rows with both >= 0 become dataset records (labels).
    final_quality: Dict[str, Tuple[float, float]] = {}
    for tid, row in sorted(done.items()):
        cheap_q, strong_q = row["local_quality"], row["remote_quality"]
        if tid in grades:
            # A verdict may cover only the tier that needed review.
            verdict = grades[tid]
            cheap_v = verdict.get("cheap", verdict.get("local"))
            strong_v = verdict.get("strong", verdict.get("remote"))
            if cheap_v is not None:
                cheap_q = float(cheap_v)
            if strong_v is not None:
                strong_q = float(strong_v)
        final_quality[tid] = (cheap_q, strong_q)
        if cheap_q < 0 or strong_q < 0:
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
                local_confidence=row.get("local_confidence"),
                local_min_token_prob=row.get("local_min_token_prob"),
                local_low_token_frac=row.get("local_low_token_frac"),
                post_check_problems=row.get("post_check_problems") or [],
            )
        )

    work_path = args.out + ".work.jsonl"
    attempts = _load_attempts(work_path) or list(done.values())
    try:
        task_ids: Optional[List[str]] = [str(t["task_id"]) for t in _load_tasks(args.tasks)]
    except (OSError, ValueError, KeyError, TypeError):
        task_ids = None
    accounting = account(task_ids, attempts, done, final_quality)
    accounting["api_spend"] = spend_summary(attempts, args,
                                            getattr(args, "max_api_cost_usd", None))

    manifest: Dict[str, Any] = {}
    manifest_path = _manifest_path(args.out)
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    compatibility = manifest.get("compatibility") or {}
    collection = {
        "manifest_canonical_sha256": canonical_json_sha256(manifest) if manifest else None,
        "run_ids": sorted({r.get("run_id") for r in attempts if r.get("run_id")}),
        "accounting": accounting,
    }
    _write_json(args.out + ".summary.json", {
        "kind": "nana-collection-summary", **collection,
        "label_identity": compatibility.get("label_identity"),
    })
    _print_accounting(accounting)

    if not records:
        print("no graded records to write — nothing finalized", file=sys.stderr)
        return 1

    first_row = next(iter(done.values()))
    tier = first_row.get("tier_mode", settings.tier_mode)
    dataset = OutcomeDataset(
        meta={
            # Required schema-1.0 names; semantically cheap and strong.
            "local_model": first_row.get("cheap_model", settings.local_model_name),
            "remote_model": first_row.get("strong_model", settings.strong_model_name),
            "tier_mode": tier,
            "tier_schema": "legacy local=cheap, remote=strong",
            "cheap_max_tokens": settings.local_max_new_tokens
            if tier == "local_remote" else settings.cheap_max_tokens,
            "strong_max_tokens": settings.strong_max_tokens,
            "temperature": 0,
            "system_prompt_sha256": sha256_text(settings.system_prompt),
            "quality_threshold": args.quality_threshold,
            "cost_unit": "usd",
            "seed": None,
            # None for rows collected before manifests existed: such a
            # dataset trains an "unverifiable" router (routing/identity.py).
            "pair_identity": compatibility.get("pair_identity"),
            "label_identity": compatibility.get("label_identity"),
            "collection": collection,
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
    print(f"wrote {len(records)} graded record(s) to {args.out}")
    return 0


def _print_accounting(a: Dict[str, Any]) -> None:
    rates = a["rates"]

    def pct(rate: Dict[str, Any]) -> str:
        return "n/a" if rate["value"] is None else f"{rate['value']:.1%}"

    print(
        f"accounting: {a['tasks_attempted']} attempted"
        + (f" of {a['tasks_in_file']} in file" if a["tasks_in_file"] is not None else "")
        + f" | {a['tasks_completed']} completed ({pct(rates['completion_rate'])} of attempted)"
        f" | {a['tasks_failed_infrastructure']} infrastructure failures"
        f" | {a['usable_paired_outcomes']} usable paired outcomes"
        f" ({pct(rates['grading_coverage'])} of completed)"
        f" | {a['tasks_unresolved_review']} awaiting review"
        f" (cheap {a['unresolved_review_by_tier']['cheap']},"
        f" strong {a['unresolved_review_by_tier']['strong']})"
        f" | {a['attempts_failed']} failed attempt(s) of {a['attempts_total']}"
        f" | strong calls skipped: {a['strong_calls']['skipped']}"
        + (f" {a['strong_calls']['skipped_by_reason']}" if a['strong_calls']['skipped'] else "")
    )
    spend = a.get("api_spend")
    if spend:
        print(
            f"api spend{' (MOCK token counts)' if spend['mock'] else ''}: "
            f"recorded ${spend['recorded_usd']:.4f}, charged against budget "
            f"${spend['charged_for_budget_usd']:.4f}"
            + (f" of ${spend['budget_usd']:.4f}" if spend["budget_usd"] is not None else "")
            + f" | {spend['paid_calls']} paid call(s), "
            f"{spend['unknown_billing_calls']} with unknown billing"
        )


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
        help="required for paid collection: the TOTAL budget for this --out "
        "across all runs and resumes; must cover the amount already charged "
        "plus the configured ceiling of unfinished tasks",
    )
    parser.add_argument(
        "--local-context-tokens", type=int, default=None,
        help="explicit Ollama num_ctx for this collection (recorded in the "
        "manifest; overrides LOCAL_CONTEXT_TOKENS)",
    )
    parser.add_argument(
        "--max-consecutive-failures", type=int, default=3,
        help="stop after this many failed attempts in a row (0 = never)",
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
