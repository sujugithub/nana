"""Offline routing evaluation on the held-out TEST split.

    python3 -m evaluation.run --dataset data/toy_dataset.json \
        --artifact artifacts/router.joblib --train-report reports/train \
        --report reports/eval

Compares, on identical tasks:
  all_local          every task on the small model (no cascade — the floor)
  all_remote         every task on the frontier model (the ceiling)
  random@p           random routing swept across remote rates, repeated
                     seeded trials — the line a useful router must beat
  heuristic          the existing keyword rules at their deployed threshold
  <candidates>       every trained candidate at its policy threshold
  selected           the artifact actually deployed
  oracle             route remote exactly when local would fail (ground
                     truth; the ceiling on what any pre-router can do)

All routed systems (heuristic, learned, random) are simulated WITH the
runtime escalation cascade, because that is the system that actually ships;
all_local and oracle run without it (all_local is the no-remote floor, and
the oracle already knows the truth). Costs count discarded local attempts.

The TEST split comes from the training run's splits.json and is meant to be
spent ONCE — evaluate here only after training and threshold selection are
finished.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from routing.artifact import RouterArtifact, load_artifact
from routing.dataset import OutcomeRecord, load_dataset
from routing.metrics import (
    bootstrap_ci,
    classification_metrics,
    confusion_at_threshold,
    paired_bootstrap_diff,
)
from routing.simulate import SimulationConfig, simulate, summarize
from routing.train import _heuristic_scores

RANDOM_GRID = [i / 20 for i in range(21)]  # pre-route remote rates 0..1
RANDOM_TRIALS = 50
BOOT_SEED = 17


def _system_report(
    records: Sequence[OutcomeRecord],
    route_local: np.ndarray,
    sim_config: Optional[SimulationConfig],
    all_remote_quality: float,
    all_remote_cost: float,
) -> Dict[str, Any]:
    sim = simulate(records, route_local, sim_config)
    out = summarize(sim)
    q_point, q_lo, q_hi = bootstrap_ci(sim.quality, seed=BOOT_SEED)
    c_point, c_lo, c_hi = bootstrap_ci(sim.cost, seed=BOOT_SEED)
    out["quality_ci"] = [q_lo, q_hi]
    out["cost_mean_ci"] = [c_lo, c_hi]
    out["cost_saved_vs_all_remote"] = all_remote_cost - out["cost_total"]
    out["cost_saved_vs_all_remote_pct"] = (
        (all_remote_cost - out["cost_total"]) / all_remote_cost
        if all_remote_cost
        else 0.0
    )
    out["quality_regret_vs_all_remote"] = all_remote_quality - out["quality_mean"]
    out["_sim"] = sim  # stripped before serialisation; used for paired tests
    return out


def _per_category(
    records: Sequence[OutcomeRecord],
    route_local: np.ndarray,
    sim_config: Optional[SimulationConfig],
) -> Dict[str, Dict[str, float]]:
    cats = sorted({r.category for r in records})
    out = {}
    for cat in cats:
        idx = [i for i, r in enumerate(records) if r.category == cat]
        sub = simulate([records[i] for i in idx], route_local[idx], sim_config)
        out[cat] = summarize(sub)
    return out


def mixture_report(
    per_cat_system: Dict[str, Dict[str, float]],
    hard_categories: List[str],
) -> Dict[str, Dict[str, float]]:
    cats = sorted(per_cat_system)
    if not cats:
        return {}
    easy = [c for c in cats if c not in hard_categories] or cats
    hard = [c for c in cats if c in hard_categories] or cats
    mixtures = {
        "balanced": {c: 1 / len(cats) for c in cats},
        "easy_heavy": {
            **{c: 0.8 / len(easy) for c in easy},
            **{c: 0.2 / len(hard) for c in hard},
        },
        "hard_heavy": {
            **{c: 0.2 / len(easy) for c in easy},
            **{c: 0.8 / len(hard) for c in hard},
        },
    }
    out = {}
    for name, weights in mixtures.items():
        total_w = sum(weights.values())
        out[name] = {
            metric: sum(
                weights[c] * per_cat_system[c][metric] for c in cats
            )
            / total_w
            for metric in ("quality_mean", "cost_mean", "unsafe_local_rate",
                           "remote_call_rate")
        }
    return out


def _random_curve(
    records: Sequence[OutcomeRecord],
    sim_config: Optional[SimulationConfig],
    seed: int,
) -> List[Dict[str, float]]:
    """Mean outcomes of random routing at each pre-route remote rate, over
    RANDOM_TRIALS seeded trials each."""
    n = len(records)
    curve = []
    for p_remote in RANDOM_GRID:
        qualities, costs, remote_rates, unsafe = [], [], [], []
        for trial in range(RANDOM_TRIALS):
            rng = np.random.default_rng(seed * 100003 + int(p_remote * 100) * 101 + trial)
            route_local = rng.random(n) >= p_remote
            s = summarize(simulate(records, route_local, sim_config))
            qualities.append(s["quality_mean"])
            costs.append(s["cost_mean"])
            remote_rates.append(s["remote_call_rate"])
            unsafe.append(s["unsafe_local_rate"])
        curve.append(
            {
                "p_remote_pre_route": p_remote,
                "quality_mean": float(np.mean(qualities)),
                "quality_std": float(np.std(qualities)),
                "quality_ci": [
                    float(np.percentile(qualities, 2.5)),
                    float(np.percentile(qualities, 97.5)),
                ],
                "cost_mean": float(np.mean(costs)),
                "remote_call_rate": float(np.mean(remote_rates)),
                "unsafe_local_rate": float(np.mean(unsafe)),
                "trials": RANDOM_TRIALS,
            }
        )
    return curve


def _interp_random(curve: List[Dict[str, float]], remote_rate: float, key: str) -> float:
    """Linear interpolation of the random curve at a realized remote rate."""
    pts = sorted(curve, key=lambda p: p["remote_call_rate"])
    xs = [p["remote_call_rate"] for p in pts]
    ys = [p[key] for p in pts]
    return float(np.interp(remote_rate, xs, ys))


def _threshold_curve(
    records: Sequence[OutcomeRecord],
    p_local: np.ndarray,
    sim_config: Optional[SimulationConfig],
) -> List[Dict[str, float]]:
    thresholds = sorted(set(np.round(p_local, 4).tolist()))
    thresholds = [0.0] + thresholds + [1.000001]
    out = []
    for t in thresholds:
        s = summarize(simulate(records, p_local >= t, sim_config))
        s["threshold"] = float(t)
        out.append(s)
    return out


def evaluate(
    records: Sequence[OutcomeRecord],
    artifacts: Dict[str, RouterArtifact],
    selected_name: str,
    heuristic_threshold: float,
    sim_config: Optional[SimulationConfig] = None,
    seed: int = 17,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("no records in the evaluation split")
    sim_config = sim_config or SimulationConfig()
    n = len(records)
    y = np.array([1 if r.local_ok else 0 for r in records])
    prompts = [r.prompt for r in records]

    all_true = np.ones(n, dtype=bool)
    all_false = np.zeros(n, dtype=bool)
    no_cascade = SimulationConfig(escalation=False)

    all_remote_sum = summarize(simulate(records, all_false, sim_config))
    ar_quality = all_remote_sum["quality_mean"]
    ar_cost = all_remote_sum["cost_total"]

    systems: Dict[str, Dict[str, Any]] = {}
    systems["all_local"] = _system_report(
        records, all_true, no_cascade, ar_quality, ar_cost
    )
    systems["all_remote"] = _system_report(
        records, all_false, sim_config, ar_quality, ar_cost
    )
    oracle_route = np.array([r.local_ok for r in records])
    systems["oracle"] = _system_report(
        records, oracle_route, no_cascade, ar_quality, ar_cost
    )

    heur_scores = _heuristic_scores(prompts)
    systems["heuristic"] = _system_report(
        records, heur_scores >= heuristic_threshold, sim_config, ar_quality, ar_cost
    )
    systems["heuristic"]["threshold"] = heuristic_threshold
    systems["heuristic"]["classification"] = classification_metrics(y, heur_scores)
    systems["heuristic"]["confusion"] = confusion_at_threshold(
        y, heur_scores, heuristic_threshold
    )

    scores_by_system: Dict[str, np.ndarray] = {"heuristic": heur_scores}
    for name, artifact in artifacts.items():
        p = artifact.predict_p_local(prompts)
        scores_by_system[name] = p
        rep = _system_report(
            records, p >= artifact.threshold, sim_config, ar_quality, ar_cost
        )
        rep["threshold"] = artifact.threshold
        rep["artifact_version"] = artifact.version
        rep["classification"] = classification_metrics(y, p)
        rep["confusion"] = confusion_at_threshold(y, p, artifact.threshold)
        systems[name] = rep

    random_curve = _random_curve(records, sim_config, seed)

    # Per-category + mixtures for the headline systems.
    per_category: Dict[str, Any] = {}
    for name in ["all_local", "all_remote", "heuristic", selected_name, "oracle"]:
        if name in ("all_local", "oracle"):
            cfg, route = no_cascade, (
                all_true if name == "all_local" else oracle_route
            )
        elif name == "all_remote":
            cfg, route = sim_config, all_false
        else:
            t = systems[name]["threshold"]
            cfg, route = sim_config, scores_by_system[name] >= t
        per_category[name] = _per_category(records, np.asarray(route), cfg)

    # Data-driven easy/hard partition: categories whose ALL-LOCAL quality is
    # below the median are "hard".
    cat_quality = {
        c: per_category["all_local"][c]["quality_mean"]
        for c in per_category["all_local"]
    }
    median_q = float(np.median(list(cat_quality.values())))
    hard_categories = [c for c, q in cat_quality.items() if q < median_q]
    mixtures = {
        name: mixture_report(per_category[name], hard_categories)
        for name in per_category
    }

    # Threshold sweeps (the Pareto raw material).
    sweeps = {
        name: _threshold_curve(records, scores, sim_config)
        for name, scores in scores_by_system.items()
    }

    # ── Verdicts: the honest bottom line ─────────────────────────────────
    sel = systems[selected_name]
    rand_q_at_rate = _interp_random(
        random_curve, sel["remote_call_rate"], "quality_mean"
    )
    rand_c_at_rate = _interp_random(random_curve, sel["remote_call_rate"], "cost_mean")
    sel_sim = sel["_sim"]
    heur_sim = systems["heuristic"]["_sim"]
    verdicts = {
        "selected": selected_name,
        "vs_random_at_matched_remote_rate": {
            "remote_call_rate": sel["remote_call_rate"],
            "selected_quality": sel["quality_mean"],
            "selected_quality_ci": sel["quality_ci"],
            "random_quality_interpolated": rand_q_at_rate,
            "random_cost_interpolated": rand_c_at_rate,
            "quality_advantage": sel["quality_mean"] - rand_q_at_rate,
            "beats_random": bool(sel["quality_ci"][0] > rand_q_at_rate),
        },
        "vs_heuristic": {
            "quality_diff_paired": paired_bootstrap_diff(
                sel_sim.quality, heur_sim.quality, seed=BOOT_SEED
            ),
            "cost_diff_paired": paired_bootstrap_diff(
                sel_sim.cost, heur_sim.cost, seed=BOOT_SEED
            ),
            "selected_unsafe_rate": sel["unsafe_local_rate"],
            "heuristic_unsafe_rate": systems["heuristic"]["unsafe_local_rate"],
        },
        "oracle_gap_quality": systems["oracle"]["quality_mean"] - sel["quality_mean"],
        "oracle_gap_cost": sel["cost_total"] - systems["oracle"]["cost_total"],
    }

    for rep in systems.values():
        rep.pop("_sim", None)

    return {
        "kind": "routing-eval-report",
        "n_test": n,
        "seed": seed,
        "sim_config": vars(sim_config),
        "systems": systems,
        "random_curve": random_curve,
        "sweeps": sweeps,
        "per_category": per_category,
        "hard_categories": hard_categories,
        "mixtures": mixtures,
        "verdicts": verdicts,
    }


# ── Pareto plot: self-contained HTML + inline SVG, zero plotting deps ────

def write_pareto_html(results: Dict[str, Any], path: str) -> None:
    W, H, PAD = 760, 480, 60
    series = []  # (label, color, points, is_line)
    for name, sweep in results["sweeps"].items():
        pts = sorted(
            {(p["cost_mean"], p["quality_mean"]) for p in sweep}
        )
        color = {"heuristic": "#e08b2d"}.get(name, "#3d7bd9" if "logreg" in name else "#7a44c0")
        series.append((name + " (threshold sweep)", color, pts, True))
    rand_pts = sorted(
        (p["cost_mean"], p["quality_mean"]) for p in results["random_curve"]
    )
    series.append(("random routing", "#999999", rand_pts, True))

    marks = []
    for name, color in (
        ("all_local", "#c0392b"),
        ("all_remote", "#1e8449"),
        ("oracle", "#111111"),
        ("heuristic", "#e08b2d"),
        (results["verdicts"]["selected"], "#7a44c0"),
    ):
        s = results["systems"][name]
        marks.append((name, color, s["cost_mean"], s["quality_mean"]))

    xs = [x for _, _, pts, _ in series for x, _ in pts] + [m[2] for m in marks]
    ys = [y for _, _, pts, _ in series for _, y in pts] + [m[3] for m in marks]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    y0, y1 = y0 - 0.02, min(1.0, y1 + 0.02)

    def sx(x):
        return PAD + (x - x0) / (x1 - x0 + 1e-12) * (W - 2 * PAD)

    def sy(y):
        return H - PAD - (y - y0) / (y1 - y0 + 1e-12) * (H - 2 * PAD)

    parts = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="max-width:100%;background:#fff;font-family:system-ui">',
        f'<text x="{W/2}" y="24" text-anchor="middle" font-size="15" '
        f'font-weight="600">Cost vs quality — test split (n={results["n_test"]})</text>',
        f'<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#444"/>',
        f'<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#444"/>',
        f'<text x="{W/2}" y="{H-14}" text-anchor="middle" font-size="12">'
        f"mean cost per task</text>",
        f'<text x="16" y="{H/2}" font-size="12" transform="rotate(-90 16 {H/2})" '
        f'text-anchor="middle">mean routed quality</text>',
    ]
    for i in range(5):
        xv = x0 + (x1 - x0) * i / 4
        yv = y0 + (y1 - y0) * i / 4
        parts.append(
            f'<text x="{sx(xv):.0f}" y="{H-PAD+16}" text-anchor="middle" '
            f'font-size="10">{xv:.2e}</text>'
        )
        parts.append(
            f'<text x="{PAD-6}" y="{sy(yv):.0f}" text-anchor="end" '
            f'font-size="10">{yv:.2f}</text>'
        )
    for label, color, pts, _ in series:
        d = " ".join(f"{sx(px):.1f},{sy(py):.1f}" for px, py in pts)
        parts.append(
            f'<polyline points="{d}" fill="none" stroke="{color}" '
            f'stroke-width="1.6" opacity="0.85"/>'
        )
    for name, color, cx, cy in marks:
        parts.append(
            f'<circle cx="{sx(cx):.1f}" cy="{sy(cy):.1f}" r="5" fill="{color}"/>'
            f'<text x="{sx(cx)+8:.1f}" y="{sy(cy)-6:.1f}" font-size="11" '
            f'fill="{color}">{name}</text>'
        )
    ly = PAD + 4
    for label, color, _, _ in series:
        parts.append(
            f'<rect x="{W-250}" y="{ly-9}" width="14" height="3" fill="{color}"/>'
            f'<text x="{W-232}" y="{ly-4}" font-size="11">{label}</text>'
        )
        ly += 16
    parts.append("</svg>")
    html = (
        "<!-- generated by evaluation/run.py -->\n"
        "<title>Routing Pareto</title>\n" + "\n".join(parts) + "\n"
    )
    with open(path, "w") as fh:
        fh.write(html)


def _print_summary(results: Dict[str, Any], out_dir: str) -> None:
    lines = [
        "routing evaluation — TEST split",
        "=" * 66,
        f"{'system':<22} {'quality':>8} {'cost':>10} {'remote%':>8} "
        f"{'local%':>7} {'esc%':>6} {'unsafe%':>8}",
    ]
    order = ["all_local", "random", "heuristic"] + [
        n for n in results["systems"] if n not in (
            "all_local", "all_remote", "heuristic", "oracle")
    ] + ["all_remote", "oracle"]
    for name in order:
        if name == "random":
            v = results["verdicts"]["vs_random_at_matched_remote_rate"]
            lines.append(
                f"{'random@matched':<22} {v['random_quality_interpolated']:>8.3f} "
                f"{v['random_cost_interpolated'] * results['n_test']:>10.4f} "
                f"{v['remote_call_rate']:>8.1%} {'—':>7} {'—':>6} {'—':>8}"
            )
            continue
        s = results["systems"][name]
        lines.append(
            f"{name:<22} {s['quality_mean']:>8.3f} {s['cost_total']:>10.4f} "
            f"{s['remote_call_rate']:>8.1%} {s['local_utilisation']:>7.1%} "
            f"{s['escalation_rate']:>6.1%} {s['unsafe_local_rate']:>8.2%}"
        )
    v = results["verdicts"]
    vr = v["vs_random_at_matched_remote_rate"]
    vh = v["vs_heuristic"]
    sel = results["systems"][v["selected"]]
    cls = sel.get("classification", {})
    lines += [
        "",
        f"selected router: {v['selected']} "
        f"(threshold {sel.get('threshold'):.3f}, "
        f"artifact {sel.get('artifact_version', '?')})",
        f"  roc_auc {cls.get('roc_auc'):.3f}  pr_auc(fail) "
        f"{cls.get('pr_auc_local_fail'):.3f}  brier {cls.get('brier'):.3f}  "
        f"ece {cls.get('ece'):.3f}" if cls.get("roc_auc") is not None else "",
        "",
        f"vs random at matched remote rate ({vr['remote_call_rate']:.1%}): "
        f"quality {vr['selected_quality']:.3f} vs {vr['random_quality_interpolated']:.3f} "
        f"(advantage {vr['quality_advantage']:+.3f}; "
        f"CI-clears-random: {vr['beats_random']})",
        f"vs heuristic (paired): quality diff "
        f"{vh['quality_diff_paired']['mean_diff']:+.4f} "
        f"[{vh['quality_diff_paired']['ci_lo']:+.4f}, "
        f"{vh['quality_diff_paired']['ci_hi']:+.4f}], cost diff "
        f"{vh['cost_diff_paired']['mean_diff']:+.2e} "
        f"[{vh['cost_diff_paired']['ci_lo']:+.2e}, "
        f"{vh['cost_diff_paired']['ci_hi']:+.2e}]",
        f"oracle gap: quality {v['oracle_gap_quality']:+.3f}, "
        f"cost {v['oracle_gap_cost']:+.4f}",
        "",
        f"full results: {os.path.join(out_dir, 'results.json')}",
        f"pareto chart: {os.path.join(out_dir, 'pareto.html')}",
    ]
    summary = "\n".join(line for line in lines if line is not None)
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(summary + "\n")
    print(summary)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Offline routing evaluation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--artifact", default="artifacts/router.joblib")
    parser.add_argument(
        "--train-report", default="reports/train",
        help="training report dir (splits.json + candidate_*.joblib live here)"
    )
    parser.add_argument("--report", default="reports/eval")
    parser.add_argument(
        "--split", default="test", choices=["test", "val", "train"],
        help="which split to evaluate (test = the one-shot final number)"
    )
    parser.add_argument("--heuristic-threshold", type=float, default=None,
                        help="default: CONFIDENCE_THRESHOLD from config")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gate-stat", default="mean")
    parser.add_argument("--gate-threshold", type=float, default=0.4)
    parser.add_argument("--no-escalation-sim", action="store_true")
    args = parser.parse_args(argv)

    from config import settings

    heuristic_threshold = (
        settings.confidence_threshold
        if args.heuristic_threshold is None
        else args.heuristic_threshold
    )

    try:
        dataset = load_dataset(args.dataset)
        selected = load_artifact(args.artifact)
        splits_path = os.path.join(args.train_report, "splits.json")
        with open(splits_path) as fh:
            assignment = json.load(fh)["assignment"]
    except Exception as err:
        print(f"evaluation setup failed: {err}", file=sys.stderr)
        return 1

    if selected.dataset_hash and selected.dataset_hash != dataset.content_hash():
        print(
            f"WARNING: artifact was trained on dataset {selected.dataset_hash} "
            f"but this file hashes to {dataset.content_hash()} — the split "
            f"assignment may not match and test data may be contaminated",
            file=sys.stderr,
        )

    missing = [r.task_id for r in dataset.records if r.task_id not in assignment]
    if missing:
        print(
            f"evaluation failed: {len(missing)} dataset records missing from "
            f"{splits_path} (e.g. {missing[:3]}) — retrain on this dataset",
            file=sys.stderr,
        )
        return 1

    records = [r for r in dataset.records if assignment[r.task_id] == args.split]
    if args.split != "test":
        print(
            f"NOTE: evaluating the {args.split!r} split — a dry run, not the "
            f"final number",
            file=sys.stderr,
        )

    artifacts: Dict[str, RouterArtifact] = {}
    for path in sorted(glob.glob(os.path.join(args.train_report, "candidate_*.joblib"))):
        cand = load_artifact(path)
        artifacts[cand.model_name] = cand
    # The deployed artifact wins on name collision (it is the same model).
    artifacts[selected.model_name] = selected

    sim_config = SimulationConfig(
        escalation=not args.no_escalation_sim,
        gate_stat=args.gate_stat,
        gate_threshold=args.gate_threshold,
    )
    results = evaluate(
        records,
        artifacts,
        selected_name=selected.model_name,
        heuristic_threshold=heuristic_threshold,
        sim_config=sim_config,
        seed=args.seed,
    )
    results["split"] = args.split
    results["dataset_hash"] = dataset.content_hash()
    results["artifact_version"] = selected.version

    os.makedirs(args.report, exist_ok=True)
    with open(os.path.join(args.report, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    write_pareto_html(results, os.path.join(args.report, "pareto.html"))
    _print_summary(results, args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
