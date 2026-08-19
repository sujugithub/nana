"""Versioned outcome-dataset format for training and evaluating the router.

A dataset is one JSON file:

    {
      "schema_version": "1.0",
      "meta": {
        "local_model":  "Qwen/Qwen2.5-1.5B-Instruct",
        "remote_model": "accounts/fireworks/models/deepseek-v4-pro",
        "quality_threshold": 0.6,      # local_ok := local_quality >= this
        "cost_unit": "usd",
        "seed": 7,                     # generation/collection seed if any
        "created_at": "...", "notes": "..."
      },
      "records": [ {<record>}, ... ]
    }

Each record captures what BOTH models actually did on one prompt:

    task_id            unique string
    prompt             the exact prompt text
    category           task family ("math", "sentiment", ...; "unknown" ok)
    source             dataset/source group the prompt came from ("gsm8k", ...)
    group_id           leakage group: related/paraphrased prompts share one.
                       Defaults to a normalized-prompt hash when omitted.
    metadata           safe PRE-ROUTE metadata only (dict, may be empty)
    local_quality      graded quality of the local answer, 0..1
    remote_quality     graded quality of the remote answer, 0..1
    local_ok           bool — MUST equal local_quality >= meta.quality_threshold
    local_prompt_tokens / local_completion_tokens     ints
    remote_prompt_tokens / remote_completion_tokens   ints
    local_cost / remote_cost         measured or estimated cost per answer
    local_latency_s / remote_latency_s               floats
    local_confidence       mean token prob of the local answer (post-gen; optional)
    local_min_token_prob   min token prob (post-gen; optional)
    local_low_token_frac   fraction of tokens with prob < 0.5 (post-gen; optional)
    post_check_problems    router.post_check problems on the local answer (optional)

POST_GEN_FIELDS marks which fields exist only after local generation ran.
They feed escalation *simulation* and post-gen-signal evaluation; the feature
extractor (routing/features.py) is forbidden from reading them, and
tests/test_features.py enforces that.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from routing import SCHEMA_VERSION

# Fields that only exist once the local model has generated an answer.
# Pre-route features must never consume these.
POST_GEN_FIELDS = (
    "local_quality",
    "remote_quality",
    "local_ok",
    "local_confidence",
    "local_min_token_prob",
    "local_low_token_frac",
    "post_check_problems",
    "local_completion_tokens",
    "remote_completion_tokens",
    "local_cost",
    "remote_cost",
    "local_latency_s",
    "remote_latency_s",
)

_REQUIRED_META = ("local_model", "remote_model", "quality_threshold")
_REQUIRED_RECORD = (
    "task_id",
    "prompt",
    "local_quality",
    "remote_quality",
    "local_ok",
)

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_prompt(prompt: str) -> str:
    """Canonical form used for duplicate detection and default group ids:
    lowercase, punctuation stripped, whitespace collapsed."""
    lowered = _PUNCT_RE.sub(" ", prompt.lower())
    return _WS_RE.sub(" ", lowered).strip()


def default_group_id(prompt: str) -> str:
    return "g-" + hashlib.sha256(normalize_prompt(prompt).encode()).hexdigest()[:16]


@dataclass
class OutcomeRecord:
    task_id: str
    prompt: str
    local_quality: float
    remote_quality: float
    local_ok: bool
    category: str = "unknown"
    source: str = "unknown"
    group_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    local_prompt_tokens: int = 0
    local_completion_tokens: int = 0
    remote_prompt_tokens: int = 0
    remote_completion_tokens: int = 0
    local_cost: float = 0.0
    remote_cost: float = 0.0
    local_latency_s: float = 0.0
    remote_latency_s: float = 0.0
    local_confidence: Optional[float] = None
    local_min_token_prob: Optional[float] = None
    local_low_token_frac: Optional[float] = None
    post_check_problems: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.group_id:
            self.group_id = default_group_id(self.prompt)


@dataclass
class OutcomeDataset:
    meta: Dict[str, Any]
    records: List[OutcomeRecord]
    schema_version: str = SCHEMA_VERSION

    @property
    def quality_threshold(self) -> float:
        return float(self.meta["quality_threshold"])

    def content_hash(self) -> str:
        """Stable fingerprint of the dataset content, recorded in training
        reports so results can be traced to the exact data that produced them."""
        blob = json.dumps(
            {
                "schema_version": self.schema_version,
                "meta": self.meta,
                "records": [_record_to_dict(r) for r in self.records],
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class DatasetError(ValueError):
    """The dataset file is malformed, inconsistent, or version-incompatible."""


def _record_to_dict(rec: OutcomeRecord) -> Dict[str, Any]:
    return {
        "task_id": rec.task_id,
        "prompt": rec.prompt,
        "category": rec.category,
        "source": rec.source,
        "group_id": rec.group_id,
        "metadata": rec.metadata,
        "local_quality": rec.local_quality,
        "remote_quality": rec.remote_quality,
        "local_ok": rec.local_ok,
        "local_prompt_tokens": rec.local_prompt_tokens,
        "local_completion_tokens": rec.local_completion_tokens,
        "remote_prompt_tokens": rec.remote_prompt_tokens,
        "remote_completion_tokens": rec.remote_completion_tokens,
        "local_cost": rec.local_cost,
        "remote_cost": rec.remote_cost,
        "local_latency_s": rec.local_latency_s,
        "remote_latency_s": rec.remote_latency_s,
        "local_confidence": rec.local_confidence,
        "local_min_token_prob": rec.local_min_token_prob,
        "local_low_token_frac": rec.local_low_token_frac,
        "post_check_problems": rec.post_check_problems,
    }


def _check_finite(name: str, value: float, task_id: str, errors: List[str]) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{task_id}: {name} must be a number, got {value!r}")
    elif not math.isfinite(float(value)):
        errors.append(f"{task_id}: {name} must be finite, got {value!r}")


def _check_unit(name: str, value: float, task_id: str, errors: List[str]) -> None:
    _check_finite(name, value, task_id, errors)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not (0.0 <= float(value) <= 1.0):
            errors.append(f"{task_id}: {name} must be in [0, 1], got {value!r}")


def validate_dataset(raw: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems; empty list = valid."""
    errors: List[str] = []
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        errors.append(
            f"schema_version {version!r} unsupported (this code reads "
            f"{SCHEMA_VERSION!r}); convert the dataset or update the code"
        )
        return errors  # nothing below is trustworthy on a version mismatch

    meta = raw.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta must be an object")
        return errors
    for key in _REQUIRED_META:
        if key not in meta:
            errors.append(f"meta.{key} is required")
    qt = meta.get("quality_threshold")
    if qt is not None:
        _check_unit("meta.quality_threshold", qt, "meta", errors)
    if errors:
        return errors

    records = raw.get("records")
    if not isinstance(records, list) or not records:
        errors.append("records must be a non-empty list")
        return errors

    seen_ids: Dict[str, int] = {}
    threshold = float(qt)
    for i, rec in enumerate(records):
        rid = str(rec.get("task_id", f"records[{i}]"))
        for key in _REQUIRED_RECORD:
            if key not in rec:
                errors.append(f"{rid}: missing required field {key!r}")
        if "task_id" in rec:
            if rid in seen_ids:
                errors.append(
                    f"{rid}: duplicate task_id (also records[{seen_ids[rid]}])"
                )
            seen_ids[rid] = i
        prompt = rec.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"{rid}: prompt must be a non-empty string")
        for name in ("local_quality", "remote_quality"):
            if name in rec:
                _check_unit(name, rec[name], rid, errors)
        for name in (
            "local_confidence",
            "local_min_token_prob",
            "local_low_token_frac",
        ):
            if rec.get(name) is not None:
                _check_unit(name, rec[name], rid, errors)
        for name in ("local_cost", "remote_cost", "local_latency_s", "remote_latency_s"):
            if name in rec:
                _check_finite(name, rec[name], rid, errors)
                if isinstance(rec[name], (int, float)) and float(rec[name]) < 0:
                    errors.append(f"{rid}: {name} must be >= 0")
        # The label must be DERIVABLE from the explicit quality threshold —
        # an inconsistent local_ok means the labelling pipeline is broken.
        if (
            isinstance(rec.get("local_ok"), bool)
            and isinstance(rec.get("local_quality"), (int, float))
            and not isinstance(rec.get("local_quality"), bool)
        ):
            derived = float(rec["local_quality"]) >= threshold
            if rec["local_ok"] != derived:
                errors.append(
                    f"{rid}: local_ok={rec['local_ok']} contradicts "
                    f"local_quality={rec['local_quality']} vs "
                    f"quality_threshold={threshold}"
                )
        if "local_ok" in rec and not isinstance(rec["local_ok"], bool):
            errors.append(f"{rid}: local_ok must be a bool")
    return errors


def load_dataset(path: str) -> OutcomeDataset:
    """Load + validate; raises DatasetError listing every problem found."""
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        raise DatasetError(f"dataset file not found: {path}")
    except json.JSONDecodeError as err:
        raise DatasetError(f"dataset is not valid JSON ({path}): {err}")

    if not isinstance(raw, dict):
        raise DatasetError(f"dataset root must be an object, got {type(raw).__name__}")

    problems = validate_dataset(raw)
    if problems:
        shown = "\n  - ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise DatasetError(f"invalid dataset {path}:\n  - {shown}{more}")

    records = [
        OutcomeRecord(
            task_id=str(rec["task_id"]),
            prompt=rec["prompt"],
            category=str(rec.get("category", "unknown")),
            source=str(rec.get("source", "unknown")),
            group_id=str(rec.get("group_id", "")),
            metadata=rec.get("metadata") or {},
            local_quality=float(rec["local_quality"]),
            remote_quality=float(rec["remote_quality"]),
            local_ok=bool(rec["local_ok"]),
            local_prompt_tokens=int(rec.get("local_prompt_tokens", 0)),
            local_completion_tokens=int(rec.get("local_completion_tokens", 0)),
            remote_prompt_tokens=int(rec.get("remote_prompt_tokens", 0)),
            remote_completion_tokens=int(rec.get("remote_completion_tokens", 0)),
            local_cost=float(rec.get("local_cost", 0.0)),
            remote_cost=float(rec.get("remote_cost", 0.0)),
            local_latency_s=float(rec.get("local_latency_s", 0.0)),
            remote_latency_s=float(rec.get("remote_latency_s", 0.0)),
            local_confidence=(
                None if rec.get("local_confidence") is None
                else float(rec["local_confidence"])
            ),
            local_min_token_prob=(
                None if rec.get("local_min_token_prob") is None
                else float(rec["local_min_token_prob"])
            ),
            local_low_token_frac=(
                None if rec.get("local_low_token_frac") is None
                else float(rec["local_low_token_frac"])
            ),
            post_check_problems=list(rec.get("post_check_problems") or []),
        )
        for rec in raw["records"]
    ]
    return OutcomeDataset(
        meta=raw["meta"], records=records, schema_version=raw["schema_version"]
    )


def save_dataset(dataset: OutcomeDataset, path: str) -> None:
    payload = {
        "schema_version": dataset.schema_version,
        "meta": dataset.meta,
        "records": [_record_to_dict(r) for r in dataset.records],
    }
    problems = validate_dataset(payload)
    if problems:
        raise DatasetError(
            "refusing to save an invalid dataset:\n  - " + "\n  - ".join(problems[:20])
        )
    import os

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
