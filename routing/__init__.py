"""Learned pre-routing: predict P(local model answers acceptably | prompt).

Package layout:
    dataset.py        versioned outcome-dataset format + validation
    features.py       pre-route feature extraction (prompt + safe metadata ONLY)
    splits.py         group-aware train/val/test splits, duplicate + leakage checks
    simulate.py       replay routing decisions against recorded outcomes
    policies.py       operating-threshold selection from validation data
    metrics.py        classification + calibration metrics, bootstrap CIs
    artifact.py       trained-router artifact save/load with version checks
    train.py          reproducible training CLI (python3 -m routing.train)
    learned_router.py runtime router backed by a trained artifact

The hard boundary this package enforces: everything the PRE-router consumes
must be computable before any model generates a token. Post-generation
signals (local answer text, token probabilities, post-check results) exist in
the dataset for *simulation and evaluation*, never as pre-route features.
"""

SCHEMA_VERSION = "1.0"   # outcome-dataset format (routing/dataset.py)
FEATURE_VERSION = "1.0"  # pre-route feature extraction (routing/features.py)
ARTIFACT_FORMAT_VERSION = "1.0"  # artifact bundle layout (routing/artifact.py)
