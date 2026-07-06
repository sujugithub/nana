"""Threshold-calibration analysis over logs/usage.jsonl.

The workflow this tool completes (README "kickoff-day checklist" step 3):

    python3 main.py --tasks revealed_tasks.json --threshold 0.4
    python3 main.py --tasks revealed_tasks.json --threshold 0.55
    python3 main.py --tasks revealed_tasks.json --threshold 0.7
    python3 scripts/calibrate.py                       # tokens per threshold
    python3 scripts/calibrate.py --accuracy graded.json --min-accuracy 0.9

Each usage.jsonl line already carries run_id / threshold / confidence /
billable_tokens (see token_tracker.py), so the sweep is analyzed from the
log — no rerunning.

What it prints:
1. One row per run (grouped by run_id): threshold, routing mix, billable vs
   free tokens, and mean accuracy if a grades file is given.
2. A recommendation: the LOWEST threshold whose run clears the accuracy bar
   (lower threshold = more local = fewer billable tokens; ties break toward
   fewer billable tokens).
3. A lowering-threshold replay per run: tasks that went remote WITHOUT
   escalating already have their remote cost logged, so we can compute
   exactly how many billable tokens a lower threshold would have saved.
   (Escalated tasks are excluded — they routed local first and would again.)
   RAISING the threshold is not replayable: tasks that would flip to remote
   never had their remote cost measured. That direction needs a rerun.

4. With --accuracy, BOTH dials from the same graded sweep (ISSUES #8): the
   heuristic CONFIDENCE_THRESHOLD recommendation above, plus a
   LOGPROB_CONFIDENCE_THRESHOLD recommendation — graded LOCAL answers are
   ranked by the model's own local_confidence and the tool picks the lowest
   gate at which the answers KEPT local clear the accuracy bar (everything
   below the gate escalates and is paid for). If no gate separates right
   from wrong, it says so — that's the cue to switch the confidence
   statistic from mean to min-token-prob (see local_model.py).

Accuracy grades file (--accuracy): a JSON object mapping task_id to either
true/false or a 0..1 score, e.g. {"trivial-1": true, "complex-1": 0.5}.
Grading is task-set-specific, so it stays a manual/external step — this tool
only consumes the verdicts.

Stdlib only, like test_harness.py — runs anywhere, zero deps.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional

# Route names duplicated from config.py on purpose: config imports are wired
# for the repo root, and this script must also work on a bare copied-out log.
ROUTE_LOCAL = "local"
ROUTE_REMOTE = "remote"
ROUTE_ERROR = "error"


def load_records(path: str) -> List[dict]:
    records = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated line (killed run) shouldn't sink the analysis.
                print(f"warning: skipping malformed line {lineno}", file=sys.stderr)
    return records


def load_grades(path: str) -> Dict[str, float]:
    """Normalize true/false grades to 1.0/0.0; pass numeric scores through."""
    with open(path) as fh:
        raw = json.load(fh)
    return {str(k): (1.0 if v is True else 0.0 if v is False else float(v))
            for k, v in raw.items()}


def group_runs(records: List[dict]) -> Dict[str, List[dict]]:
    runs: Dict[str, List[dict]] = {}
    for rec in records:
        runs.setdefault(rec.get("run_id", "?"), []).append(rec)
    return runs


def run_stats(rows: List[dict], grades: Optional[Dict[str, float]]) -> dict:
    graded = [grades[r["task_id"]] for r in rows if grades and r["task_id"] in grades]
    return {
        # Error rows log threshold=0.0 (no decision was made), so take the
        # threshold from any non-error row; all rows in a run share one.
        "threshold": next(
            (r["threshold"] for r in rows if r.get("route") != ROUTE_ERROR), 0.0
        ),
        "tasks": len(rows),
        "local": sum(1 for r in rows if r.get("route") == ROUTE_LOCAL),
        "remote": sum(1 for r in rows if r.get("route") == ROUTE_REMOTE),
        "escalations": sum(1 for r in rows if r.get("escalated")),
        "errors": sum(1 for r in rows if r.get("route") == ROUTE_ERROR),
        "billable": sum(r.get("billable_tokens", 0) for r in rows),
        "free": sum(r.get("local_prompt_tokens", 0)
                    + r.get("local_completion_tokens", 0) for r in rows),
        # None = no grades supplied / none matched; distinguishable from 0.0.
        "accuracy": (sum(graded) / len(graded)) if graded else None,
        "graded": len(graded),
    }


def lowering_replay(rows: List[dict], candidates: List[float]) -> List[tuple]:
    """(candidate_threshold, tasks_flipped, billable_tokens_saved) rows.

    A task flips local at candidate t if it was pre-routed remote (confidence
    below the run threshold, not escalated) but has confidence >= t. Its
    logged billable_tokens are the exact savings. Accuracy of the flipped
    tasks is UNKNOWN (the local answer never ran) — that's the gamble the
    post-check cascade is there to bound.
    """
    remote_pre_routed = [
        r for r in rows
        if r.get("route") == ROUTE_REMOTE and not r.get("escalated")
    ]
    out = []
    for t in candidates:
        flipped = [r for r in remote_pre_routed if r.get("confidence", 0.0) >= t]
        if flipped:
            out.append((t, len(flipped), sum(r["billable_tokens"] for r in flipped)))
    return out


def logprob_rows(records: List[dict], grades: Dict[str, float]) -> List[dict]:
    """One {task_id, conf, grade} per graded task whose FINAL answer was
    local — those are the answers the logprob gate decides over. Escalated
    tasks are excluded (their grade judges the remote answer). Greedy
    decoding makes repeats identical across runs, so keep first-seen."""
    rows: Dict[str, dict] = {}
    for rec in records:
        task_id = rec.get("task_id")
        if (
            rec.get("route") == ROUTE_LOCAL
            and rec.get("local_confidence") is not None
            and task_id in grades
            and task_id not in rows
        ):
            rows[task_id] = {
                "task_id": task_id,
                "conf": float(rec["local_confidence"]),
                "grade": grades[task_id],
            }
    return sorted(rows.values(), key=lambda r: r["conf"])


def recommend_logprob_threshold(
    rows: List[dict], min_accuracy: float
) -> Optional[tuple]:
    """(threshold, kept_accuracy, kept_count, wasted_escalations) or None.

    At gate t, answers with conf < t escalate (paid remote retries); the
    answers KEPT local must clear the accuracy bar. Candidates are 0.0 (gate
    disabled) plus every observed confidence, ascending — the first that
    clears is the cheapest (fewest escalations). wasted_escalations counts
    escalated answers that were graded right (>= 0.5): pure token cost.
    """
    for t in [0.0] + sorted({r["conf"] for r in rows}):
        kept = [r for r in rows if r["conf"] >= t]
        if not kept:
            break
        accuracy = sum(r["grade"] for r in kept) / len(kept)
        if accuracy >= min_accuracy:
            wasted = sum(1 for r in rows if r["conf"] < t and r["grade"] >= 0.5)
            return (t, accuracy, len(kept), wasted)
    return None


def print_logprob_analysis(
    records: List[dict], grades: Dict[str, float], min_accuracy: float
) -> None:
    print("\n──── logprob gate (LOGPROB_CONFIDENCE_THRESHOLD) ────")
    rows = logprob_rows(records, grades)
    if not rows:
        print("no graded LOCAL answers with local_confidence in the log —")
        print("run real (non-mock) tasks and grade them first.")
        return

    print(f"{'task_id':<24} {'local_conf':>10} {'grade':>6}")
    for r in rows:
        print(f"{r['task_id']:<24} {r['conf']:>10.3f} {r['grade']:>6.2f}")

    rec = recommend_logprob_threshold(rows, min_accuracy)
    if rec is None:
        print(f"\nNO gate clears accuracy >= {min_accuracy}: wrong answers "
              f"score as high as right ones (confident-wrong). Switch the "
              f"confidence statistic from mean to min-token-prob "
              f"(local_model.py, noted in its comments) and rerun the sweep.")
        return
    t, accuracy, kept, wasted = rec
    if t == 0.0:
        print(f"\nRECOMMENDED: LOGPROB_CONFIDENCE_THRESHOLD = 0 (gate off) — "
              f"all {kept} graded local answers already clear "
              f"{min_accuracy} (accuracy {accuracy:.2f}).")
        return
    escalated = len(rows) - kept
    print(f"\nRECOMMENDED: LOGPROB_CONFIDENCE_THRESHOLD = {t:.3f} — keeps "
          f"{kept} local answer(s) at accuracy {accuracy:.2f} >= "
          f"{min_accuracy}; {escalated} escalate ({wasted} of them graded "
          f"right = wasted paid retries).")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tabulate billable tokens vs threshold from usage.jsonl "
                    "and recommend the cheapest threshold that clears the "
                    "accuracy bar."
    )
    parser.add_argument("--log", default="logs/usage.jsonl",
                        help="path to the usage log (default: logs/usage.jsonl)")
    parser.add_argument("--accuracy", default=None,
                        help="JSON file {task_id: true/false or 0..1 score}")
    parser.add_argument("--min-accuracy", type=float, default=0.9,
                        help="accuracy bar a run must clear (default: 0.9)")
    parser.add_argument("--candidates", default="0.3,0.4,0.5,0.55,0.6,0.7",
                        help="comma-separated thresholds for the lowering replay")
    args = parser.parse_args(argv)

    try:
        records = load_records(args.log)
    except FileNotFoundError:
        print(f"no log at {args.log} — run main.py first (see module docstring)")
        return 1
    if not records:
        print(f"{args.log} is empty — run main.py first (see module docstring)")
        return 1

    grades = load_grades(args.accuracy) if args.accuracy else None
    candidates = sorted(float(c) for c in args.candidates.split(","))

    runs = {rid: run_stats(rows, grades) for rid, rows in group_runs(records).items()}
    ordered = sorted(runs.items(), key=lambda kv: (kv[1]["threshold"], kv[0]))

    print("──── tokens vs threshold (one row per run) ────")
    header = (f"{'run_id':<17} {'thresh':>6} {'tasks':>5} {'local':>5} "
              f"{'remote':>6} {'esc':>3} {'err':>3} {'billable':>8} "
              f"{'free':>7} {'accuracy':>8}")
    print(header)
    for rid, s in ordered:
        acc = "n/a" if s["accuracy"] is None else f"{s['accuracy']:.2f}"
        print(f"{rid:<17} {s['threshold']:>6.2f} {s['tasks']:>5} {s['local']:>5} "
              f"{s['remote']:>6} {s['escalations']:>3} {s['errors']:>3} "
              f"{s['billable']:>8} {s['free']:>7} {acc:>8}")

    # ── Recommendation ───────────────────────────────────────────────────
    print()
    if grades is None:
        print("no --accuracy file given → no recommendation. Grade the runs'")
        print("answers, write {task_id: true/false} JSON, and rerun with")
        print("--accuracy grades.json --min-accuracy <bar>.")
    else:
        passing = [(rid, s) for rid, s in ordered
                   if s["accuracy"] is not None and s["accuracy"] >= args.min_accuracy]
        if not passing:
            print(f"NO run clears accuracy >= {args.min_accuracy} — raise the "
                  f"threshold (more remote) and rerun the sweep.")
        else:
            rid, s = min(passing, key=lambda kv: (kv[1]["threshold"], kv[1]["billable"]))
            print(f"RECOMMENDED: threshold {s['threshold']} "
                  f"(run {rid}: accuracy {s['accuracy']:.2f} >= "
                  f"{args.min_accuracy}, {s['billable']} billable tokens) — "
                  f"the lowest threshold that clears the bar.")
            partial = [rid for rid, s in ordered if grades and s["graded"] < s["tasks"]]
            if partial:
                print(f"note: runs with ungraded tasks (accuracy is partial): "
                      f"{', '.join(partial)}")

    # ── Logprob gate (the SECOND dial, from the same graded sweep) ───────
    if grades is not None:
        print_logprob_analysis(records, grades, args.min_accuracy)

    # ── Lowering-threshold replay ────────────────────────────────────────
    printed_header = False
    for rid, rows in sorted(group_runs(records).items()):
        replay = lowering_replay(rows, [c for c in candidates
                                        if c < runs[rid]["threshold"]])
        if not replay:
            continue
        if not printed_header:
            print("\n──── if the threshold had been lower (replayed from the log) ────")
            printed_header = True
        for t, n, saved in replay:
            print(f"run {rid} (thresh {runs[rid]['threshold']}): at {t} → "
                  f"{n} task(s) flip local, saving {saved} billable tokens "
                  f"(local accuracy on them: unmeasured)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
