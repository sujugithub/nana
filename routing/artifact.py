"""Trained-router artifact: everything inference needs in one file.

The bundle (joblib) contains the FULL preprocessing + model pipeline, the
calibrator wrapping it, the selected operating threshold, and every version
string needed to detect drift — so inference can never silently recompute
features differently from training.

SECURITY — read before pointing ROUTER_ARTIFACT at a file:
joblib artifacts are pickle-based. Loading one executes arbitrary code
embedded in the file. Treat artifacts exactly like executables: load ONLY
files you (or your pipeline) produced. Never load an artifact downloaded
from an untrusted source. This project never auto-downloads artifacts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from routing import ARTIFACT_FORMAT_VERSION, FEATURE_VERSION, SCHEMA_VERSION


class ArtifactError(RuntimeError):
    """Artifact missing, unreadable, or incompatible with this code."""


@dataclass
class RouterArtifact:
    """In-memory form of a loaded artifact."""

    model: Any  # fitted sklearn estimator: predict_proba(prompts) -> [:, 2]
    threshold: float
    model_name: str  # candidate id, e.g. "logreg_tfidf"
    version: str  # human-readable artifact version tag (timestamp-based)
    policy: Dict[str, Any] = field(default_factory=dict)
    calibration_method: str = ""
    dataset_meta: Dict[str, Any] = field(default_factory=dict)
    dataset_hash: str = ""
    seed: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    postgen: Dict[str, Any] = field(default_factory=dict)
    format_version: str = ARTIFACT_FORMAT_VERSION
    feature_version: str = FEATURE_VERSION
    schema_version: str = SCHEMA_VERSION
    sklearn_version: str = ""

    def predict_p_local(self, prompts) -> "np.ndarray":  # noqa: F821
        """P(local answer acceptable) per prompt. Higher = safer to run local
        — the same direction as every other confidence in this codebase."""
        import numpy as np

        proba = self.model.predict_proba(list(prompts))
        classes = list(getattr(self.model, "classes_", [0, 1]))
        col = classes.index(1) if 1 in classes else len(classes) - 1
        return np.asarray(proba[:, col], dtype=float)


def save_artifact(artifact: RouterArtifact, path: str) -> None:
    import joblib
    import sklearn

    artifact.sklearn_version = sklearn.__version__
    payload = {
        "format_version": artifact.format_version,
        "feature_version": artifact.feature_version,
        "schema_version": artifact.schema_version,
        "sklearn_version": artifact.sklearn_version,
        "model": artifact.model,
        "threshold": artifact.threshold,
        "model_name": artifact.model_name,
        "version": artifact.version,
        "policy": artifact.policy,
        "calibration_method": artifact.calibration_method,
        "dataset_meta": artifact.dataset_meta,
        "dataset_hash": artifact.dataset_hash,
        "seed": artifact.seed,
        "metrics": artifact.metrics,
        "postgen": artifact.postgen,
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(payload, path)


def load_artifact(path: str) -> RouterArtifact:
    """Load + compatibility-check. Raises ArtifactError with a actionable
    message on every failure mode (missing, corrupt, incompatible)."""
    if not os.path.exists(path):
        raise ArtifactError(
            f"router artifact not found: {path!r}. Train one first:\n"
            f"  python3 -m routing.train --dataset <data.json> --out {path}"
        )
    try:
        import joblib

        payload = joblib.load(path)
    except ArtifactError:
        raise
    except Exception as err:  # joblib raises a zoo of types on corrupt files
        raise ArtifactError(f"cannot load router artifact {path!r}: {err}") from err

    if not isinstance(payload, dict) or "model" not in payload:
        raise ArtifactError(
            f"{path!r} is not a router artifact (missing bundle fields)"
        )

    fmt = payload.get("format_version")
    if fmt != ARTIFACT_FORMAT_VERSION:
        raise ArtifactError(
            f"artifact format {fmt!r} incompatible with this code "
            f"({ARTIFACT_FORMAT_VERSION!r}) — retrain: python3 -m routing.train"
        )
    feat = payload.get("feature_version")
    if feat != FEATURE_VERSION:
        raise ArtifactError(
            f"artifact was trained with feature_version {feat!r} but this "
            f"code extracts {FEATURE_VERSION!r} — features would drift; "
            f"retrain the router"
        )

    import sklearn

    stored_skl = payload.get("sklearn_version", "")
    if stored_skl and stored_skl != sklearn.__version__:
        import sys

        print(
            f"WARNING: artifact trained with scikit-learn {stored_skl}, "
            f"running {sklearn.__version__} — verify predictions if behaviour "
            f"looks off",
            file=sys.stderr,
        )

    threshold = payload.get("threshold")
    if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.01):
        raise ArtifactError(f"artifact has invalid threshold {threshold!r}")

    return RouterArtifact(
        model=payload["model"],
        threshold=float(threshold),
        model_name=str(payload.get("model_name", "?")),
        version=str(payload.get("version", "?")),
        policy=payload.get("policy") or {},
        calibration_method=str(payload.get("calibration_method", "")),
        dataset_meta=payload.get("dataset_meta") or {},
        dataset_hash=str(payload.get("dataset_hash", "")),
        seed=int(payload.get("seed", 0)),
        metrics=payload.get("metrics") or {},
        postgen=payload.get("postgen") or {},
        format_version=str(fmt),
        feature_version=str(feat),
        schema_version=str(payload.get("schema_version", "")),
        sklearn_version=stored_skl,
    )
