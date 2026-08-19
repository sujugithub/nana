"""Classification/calibration metrics + bootstrap confidence intervals.

Thin, explicit wrappers so every report in the project computes these the
same way. Positive class throughout: local_ok == True ("local is safe").
PR-AUC is reported for BOTH classes because the rare class is usually the
failure class, and that is the one the router exists to catch.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np


def classification_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    out: Dict[str, float] = {
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
    }
    if len(np.unique(y)) < 2:
        # AUC family is undefined on a single-class sample; report None
        # rather than a fake number.
        out["roc_auc"] = None
        out["pr_auc_local_ok"] = None
        out["pr_auc_local_fail"] = None
    else:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc_local_ok"] = float(average_precision_score(y, p))
        out["pr_auc_local_fail"] = float(average_precision_score(1 - y, 1 - p))
    return out


def expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> float:
    """Standard binned ECE: sum over bins of |accuracy - confidence| weighted
    by bin occupancy."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-inclusive last bin so p=1.0 lands in a bin
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def confusion_at_threshold(
    y_true: np.ndarray, p: np.ndarray, threshold: float
) -> Dict[str, int]:
    """Confusion counts for the routing decision (predict local iff
    p >= threshold). 'fp' = routed local but local failed — the unsafe cell."""
    y = np.asarray(y_true, dtype=bool)
    pred_local = np.asarray(p, dtype=float) >= threshold
    return {
        "tp": int((pred_local & y).sum()),
        "fp": int((pred_local & ~y).sum()),
        "fn": int((~pred_local & y).sum()),
        "tn": int((~pred_local & ~y).sum()),
    }


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """(point, lo, hi) percentile bootstrap over per-task values."""
    values = np.asarray(values, dtype=float)
    point = float(statistic(values))
    if len(values) < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    stats = np.array([statistic(values[row]) for row in idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap CI on mean(a - b) over the SAME tasks (paired — every system
    sees identical queries, which is where the statistical power comes from)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired comparison requires aligned arrays")
    diff = a - b
    point, lo, hi = bootstrap_ci(diff, np.mean, n_boot=n_boot, seed=seed, alpha=alpha)
    return {"mean_diff": point, "ci_lo": lo, "ci_hi": hi}
