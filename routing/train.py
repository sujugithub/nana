"""Reproducible training CLI for the learned pre-router.

    python3 -m routing.train --dataset data/toy_dataset.json \
        --out artifacts/router.joblib --report reports/train

Pipeline (every step group-aware and seeded):
 1. validate the dataset (routing/dataset.py — hard fail on any problem)
 2. duplicate + leakage detection, group-aware 70/15/15 split (routing/splits.py)
 3. train candidate routers on TRAIN only:
      logreg_tfidf  TF-IDF (word+char) + engineered numerics -> logistic
                    regression (the simple baseline that must always exist)
      gbdt_svd      same features -> TruncatedSVD(64) -> histogram gradient
                    boosting (the "stronger" candidate)
    hyperparameters tuned with GroupKFold grid search inside TRAIN
 4. calibrate each candidate's probabilities (sigmoid + isotonic fitted with
    group-aware CV on TRAIN; the method with the better VALIDATION Brier wins)
 5. pick the operating threshold on VALIDATION via the chosen policy
    (routing/policies.py) — 0.5 is never assumed
 6. select the candidate whose VALIDATION operating point best satisfies the
    policy; the heuristic router is scored alongside as a reference
 7. analyse post-generation signals (mean/min/low-frac token prob) on
    TRAIN+VAL rows and record which best predicts local failure
 8. save the artifact (routing/artifact.py) + machine-readable report
    (report.json) + splits.json + human summary (summary.txt)

The TEST split is written to splits.json and never touched here — the offline
evaluator (python3 -m evaluation.run) spends it exactly once.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from routing import ARTIFACT_FORMAT_VERSION, FEATURE_VERSION, SCHEMA_VERSION
from routing.artifact import RouterArtifact, save_artifact
from routing.dataset import OutcomeDataset, OutcomeRecord, load_dataset
from routing.features import build_text_engineered_union
from routing.metrics import classification_metrics, confusion_at_threshold
from routing.policies import (
    PolicyConfig,
    ThresholdChoice,
    select_threshold,
)
from routing.simulate import GATE_STATS, SimulationConfig, gate_confidence
from routing.splits import TEST, TRAIN, VAL, SplitResult, make_splits

CANDIDATE_NAMES = ("logreg_tfidf", "gbdt_svd")


def _group_cv_splits(
    prompts: Sequence[str], groups: Sequence[str], n_splits: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Materialised group-aware CV folds (list form so CalibratedClassifierCV
    and GridSearchCV can consume the same splits)."""
    from sklearn.model_selection import GroupKFold

    n_groups = len(set(groups))
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(prompts)), groups=np.asarray(groups)))


def _build_candidate(name: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    features = build_text_engineered_union()
    if name == "logreg_tfidf":
        estimator = Pipeline(
            [
                ("features", features),
                (
                    "clf",
                    LogisticRegression(max_iter=3000, random_state=seed),
                ),
            ]
        )
        grid = {"clf__C": [0.25, 1.0, 4.0]}
    elif name == "gbdt_svd":
        from sklearn.decomposition import TruncatedSVD

        estimator = Pipeline(
            [
                ("features", features),
                ("svd", TruncatedSVD(n_components=64, random_state=seed)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=200, random_state=seed
                    ),
                ),
            ]
        )
        grid = {"clf__max_depth": [3, None], "clf__learning_rate": [0.1]}
    else:
        raise ValueError(f"unknown candidate {name!r}; expected {CANDIDATE_NAMES}")
    return estimator, grid


def _fit_candidate(
    name: str,
    prompts: List[str],
    y: np.ndarray,
    groups: List[str],
    seed: int,
) -> Tuple[Any, Dict[str, Any]]:
    """Grid-search hyperparameters with group-aware CV on TRAIN only."""
    from sklearn.model_selection import GridSearchCV

    estimator, grid = _build_candidate(name, seed)
    cv = _group_cv_splits(prompts, groups, n_splits=3, seed=seed)
    search = GridSearchCV(
        estimator, grid, cv=cv, scoring="roc_auc", n_jobs=1, refit=True
    )
    search.fit(prompts, y)
    return search.best_estimator_, dict(search.best_params_)


def _calibrate(
    estimator,
    prompts: List[str],
    y: np.ndarray,
    groups: List[str],
    val_prompts: List[str],
    y_val: np.ndarray,
    seed: int,
    method: str,
):
    """Fit CV-calibrated versions on TRAIN; pick the method by VALIDATION
    Brier when method='auto'. Isotonic is only attempted with enough data —
    it badly overfits small calibration folds."""
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss

    methods = [method]
    if method == "auto":
        methods = ["sigmoid"] + (["isotonic"] if len(prompts) >= 500 else [])

    cv = _group_cv_splits(prompts, groups, n_splits=3, seed=seed)
    best = None
    for m in methods:
        calibrated = CalibratedClassifierCV(clone(estimator), method=m, cv=cv)
        calibrated.fit(prompts, y)
        if len(val_prompts) and len(np.unique(y_val)) > 0:
            p_val = _predict_p_local(calibrated, val_prompts)
            score = brier_score_loss(y_val, p_val)
        else:
            score = 0.0
        if best is None or score < best[1]:
            best = (calibrated, score, m)
    return best[0], best[2]


def _predict_p_local(model, prompts: Sequence[str]) -> np.ndarray:
    proba = model.predict_proba(list(prompts))
    classes = list(getattr(model, "classes_", [0, 1]))
    col = classes.index(1) if 1 in classes else len(classes) - 1
    return np.asarray(proba[:, col], dtype=float)


def _heuristic_scores(prompts: Sequence[str]) -> np.ndarray:
    """The existing keyword router's confidence, scored like a probability so
    it can sit in the same sweeps. It is NOT calibrated — that is part of
    what the learned router is supposed to fix."""
    from confidence import ConfidenceEstimator

    estimator = ConfidenceEstimator()
    return np.asarray(
        [estimator.estimate(p).score for p in prompts], dtype=float
    )


def _policy_rank(choice: ThresholdChoice, policy: PolicyConfig) -> Tuple:
    """Comparable tuple, HIGHER = better, encoding what the policy wants."""
    op = choice.operating_point
    penalised = 1 if choice.note else 0  # fell back / infeasible
    if policy.name == "quality_floor":
        return (-penalised, -op["cost_total"], op["quality_mean"])
    if policy.name == "max_unsafe":
        return (-penalised, op["local_utilisation"], -op["cost_total"])
    if policy.name == "utility":
        utility = (
            op["quality_mean"]
            - policy.cost_weight * op["cost_mean"]
            - policy.latency_weight * op["latency_mean"]
        )
        return (-penalised, utility)
    return (  # remote_rate
        -penalised,
        -abs(op["remote_call_rate"] - policy.target_remote_rate),
        op["quality_mean"],
    )


def _postgen_analysis(records: Sequence[OutcomeRecord]) -> Dict[str, Any]:
    """Which post-generation confidence statistic best predicts local failure?
    Computed on TRAIN+VAL rows that carry the signals. Informative only —
    the runtime gate statistic is a config choice (LOCAL_CONF_STAT), and this
    analysis is the evidence for making it."""
    from sklearn.metrics import roc_auc_score

    out: Dict[str, Any] = {"n_with_signals": 0, "auc_by_stat": {}, "recommended": None}
    stats = [s for s in GATE_STATS if s != "none"]
    rows = [
        r for r in records if all(gate_confidence(r, s) is not None for s in stats)
    ]
    out["n_with_signals"] = len(rows)
    if len(rows) < 20:
        out["note"] = "too few rows with post-gen signals for a reliable AUC"
        return out
    y = np.array([1 if r.local_ok else 0 for r in rows])
    if len(np.unique(y)) < 2:
        out["note"] = "post-gen rows are single-class; AUC undefined"
        return out
    for stat in stats:
        scores = np.array([gate_confidence(r, stat) for r in rows], dtype=float)
        out["auc_by_stat"][stat] = float(roc_auc_score(y, scores))
    out["recommended"] = max(out["auc_by_stat"], key=out["auc_by_stat"].get)
    return out


def train(
    dataset: OutcomeDataset,
    seed: int = 7,
    candidates: Sequence[str] = CANDIDATE_NAMES,
    policy: Optional[PolicyConfig] = None,
    calibration: str = "auto",
    sim_config: Optional[SimulationConfig] = None,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[RouterArtifact, Dict[str, Any], SplitResult, Dict[str, RouterArtifact]]:
    """Programmatic entry point; the CLI is a thin wrapper. Returns
    (artifact, report_dict, splits, all_candidate_artifacts) — the last one
    so the evaluator can score every candidate, not just the winner."""
    policy = policy or PolicyConfig()
    sim_config = sim_config or SimulationConfig()
    records = dataset.records

    splits = make_splits(records, seed=seed, ratios=ratios)
    for warning in splits.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    idx = {s: splits.indices(records, s) for s in (TRAIN, VAL, TEST)}
    train_rows = [records[i] for i in idx[TRAIN]]
    val_rows = [records[i] for i in idx[VAL]]
    if len(train_rows) < 40:
        raise ValueError(
            f"only {len(train_rows)} training rows — too few to train a "
            f"router; collect more data"
        )
    if len(val_rows) < 10:
        raise ValueError(
            f"only {len(val_rows)} validation rows — cannot calibrate or "
            f"select a threshold; collect more data"
        )

    y_train = np.array([1 if r.local_ok else 0 for r in train_rows])
    y_val = np.array([1 if r.local_ok else 0 for r in val_rows])
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "training split has a single local_ok class — the router has "
            "nothing to learn; check the quality threshold and the data"
        )

    train_prompts = [r.prompt for r in train_rows]
    val_prompts = [r.prompt for r in val_rows]
    train_groups = [r.group_id for r in train_rows]

    report: Dict[str, Any] = {
        "kind": "routing-train-report",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "dataset_hash": dataset.content_hash(),
        "dataset_meta": dataset.meta,
        "versions": {
            "schema": SCHEMA_VERSION,
            "features": FEATURE_VERSION,
            "artifact_format": ARTIFACT_FORMAT_VERSION,
        },
        "split_strategy": {
            "kind": "group-shuffle-greedy",
            "ratios": list(ratios),
            "seed": seed,
            "sizes": {s: len(idx[s]) for s in (TRAIN, VAL, TEST)},
            "merged_near_duplicate_groups": len(splits.merged_groups),
            "warnings": splits.warnings,
        },
        "policy": vars(policy),
        "sim_config": vars(sim_config),
        "candidates": {},
    }

    fitted: Dict[str, Any] = {}
    choices: Dict[str, ThresholdChoice] = {}
    for name in candidates:
        model, best_params = _fit_candidate(
            name, train_prompts, y_train, train_groups, seed
        )
        calibrated, method = _calibrate(
            model,
            train_prompts,
            y_train,
            train_groups,
            val_prompts,
            y_val,
            seed,
            calibration,
        )
        p_val = _predict_p_local(calibrated, val_prompts)
        choice = select_threshold(val_rows, p_val, policy, sim_config)
        fitted[name] = calibrated
        choices[name] = choice
        report["candidates"][name] = {
            "best_params": best_params,
            "calibration_method": method,
            "val_classification": classification_metrics(y_val, p_val),
            "val_confusion_at_threshold": confusion_at_threshold(
                y_val, p_val, choice.threshold
            ),
            "threshold": choice.threshold,
            "threshold_note": choice.note,
            "val_operating_point": choice.operating_point,
            "val_sweep": choice.sweep,
        }

    # Heuristic reference, same sweep, never selectable.
    heur_val = _heuristic_scores(val_prompts)
    heur_choice = select_threshold(val_rows, heur_val, policy, sim_config)
    report["candidates"]["heuristic_reference"] = {
        "best_params": {},
        "calibration_method": "none",
        "val_classification": classification_metrics(y_val, heur_val),
        "val_confusion_at_threshold": confusion_at_threshold(
            y_val, heur_val, heur_choice.threshold
        ),
        "threshold": heur_choice.threshold,
        "threshold_note": heur_choice.note,
        "val_operating_point": heur_choice.operating_point,
        "val_sweep": heur_choice.sweep,
    }

    selected = max(candidates, key=lambda n: _policy_rank(choices[n], policy))
    report["selected_candidate"] = selected
    report["postgen"] = _postgen_analysis(train_rows + val_rows)

    def _as_artifact(name: str) -> RouterArtifact:
        return RouterArtifact(
            model=fitted[name],
            threshold=choices[name].threshold,
            model_name=name,
            version=f"{name}-{dataset.content_hash()}-s{seed}",
            policy=vars(policy),
            calibration_method=report["candidates"][name]["calibration_method"],
            dataset_meta=dataset.meta,
            dataset_hash=dataset.content_hash(),
            seed=seed,
            metrics={
                "val_classification": report["candidates"][name][
                    "val_classification"
                ],
                "val_operating_point": choices[name].operating_point,
            },
            postgen=report["postgen"],
        )

    candidate_artifacts = {name: _as_artifact(name) for name in candidates}
    return candidate_artifacts[selected], report, splits, candidate_artifacts


def _write_report(
    report: Dict[str, Any], splits: SplitResult, artifact: RouterArtifact, out_dir: str
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    with open(os.path.join(out_dir, "splits.json"), "w") as fh:
        json.dump(
            {
                "seed": splits.seed,
                "ratios": list(splits.ratios),
                "assignment": splits.assignment,
            },
            fh,
            indent=1,
        )
    lines = [
        "learned router — training summary",
        "=" * 50,
        f"dataset hash:  {report['dataset_hash']}",
        f"seed:          {report['seed']}",
        f"splits:        {report['split_strategy']['sizes']}",
        f"policy:        {report['policy']}",
        "",
        f"{'candidate':<20} {'roc_auc':>8} {'brier':>7} {'ece':>6} "
        f"{'thresh':>7} {'quality':>8} {'cost':>9} {'unsafe%':>8}",
    ]
    for name, cand in report["candidates"].items():
        cls = cand["val_classification"]
        op = cand["val_operating_point"]
        auc = "n/a" if cls["roc_auc"] is None else f"{cls['roc_auc']:.3f}"
        lines.append(
            f"{name:<20} {auc:>8} {cls['brier']:>7.3f} {cls['ece']:>6.3f} "
            f"{cand['threshold']:>7.3f} {op['quality_mean']:>8.3f} "
            f"{op['cost_total']:>9.4f} {op['unsafe_local_rate']:>8.2%}"
        )
    lines += [
        "",
        f"SELECTED: {report['selected_candidate']} "
        f"(artifact version {artifact.version})",
        f"post-gen signal AUCs: {report['postgen'].get('auc_by_stat', {})}",
        f"post-gen recommended stat: {report['postgen'].get('recommended')}",
        "",
        "Validation numbers above guide selection only — run "
        "`python3 -m evaluation.run` for the one-shot TEST evaluation.",
    ]
    summary = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(summary + "\n")
    print(summary)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the learned pre-router from an outcome dataset."
    )
    parser.add_argument("--dataset", required=True, help="outcome dataset JSON")
    parser.add_argument(
        "--out", default="artifacts/router.joblib", help="artifact output path"
    )
    parser.add_argument(
        "--report", default="reports/train", help="report output directory"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--candidates",
        default=",".join(CANDIDATE_NAMES),
        help=f"comma-separated subset of {CANDIDATE_NAMES}",
    )
    parser.add_argument(
        "--policy", default="quality_floor", help="threshold policy: "
        "quality_floor | max_unsafe | utility | remote_rate"
    )
    parser.add_argument("--min-quality-retention", type=float, default=0.97)
    parser.add_argument("--max-unsafe-rate", type=float, default=0.05)
    parser.add_argument("--cost-weight", type=float, default=1.0)
    parser.add_argument("--latency-weight", type=float, default=0.0)
    parser.add_argument("--target-remote-rate", type=float, default=0.5)
    parser.add_argument(
        "--calibration", default="auto", choices=["auto", "sigmoid", "isotonic"]
    )
    parser.add_argument(
        "--gate-stat", default="mean", choices=list(GATE_STATS),
        help="post-gen confidence statistic used in escalation simulation"
    )
    parser.add_argument("--gate-threshold", type=float, default=0.4)
    parser.add_argument(
        "--no-escalation-sim", action="store_true",
        help="simulate WITHOUT the post-generation escalation cascade"
    )
    args = parser.parse_args(argv)

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    for c in candidates:
        if c not in CANDIDATE_NAMES:
            print(
                f"unknown candidate {c!r}; expected {CANDIDATE_NAMES}",
                file=sys.stderr,
            )
            return 2

    policy = PolicyConfig(
        name=args.policy,
        min_quality_retention=args.min_quality_retention,
        max_unsafe_rate=args.max_unsafe_rate,
        cost_weight=args.cost_weight,
        latency_weight=args.latency_weight,
        target_remote_rate=args.target_remote_rate,
    )
    sim_config = SimulationConfig(
        escalation=not args.no_escalation_sim,
        gate_stat=args.gate_stat,
        gate_threshold=args.gate_threshold,
    )

    try:
        dataset = load_dataset(args.dataset)
        artifact, report, splits, candidate_artifacts = train(
            dataset,
            seed=args.seed,
            candidates=candidates,
            policy=policy,
            calibration=args.calibration,
            sim_config=sim_config,
        )
    except Exception as err:
        print(f"training failed: {err}", file=sys.stderr)
        return 1

    save_artifact(artifact, args.out)
    print(f"artifact written: {args.out} (version {artifact.version})")
    # Every candidate is saved next to the report so the evaluator can score
    # all of them on the test split, not just the winner.
    for name, cand in candidate_artifacts.items():
        save_artifact(cand, os.path.join(args.report, f"candidate_{name}.joblib"))
    _write_report(report, splits, artifact, args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
