"""Framework-free core of the demo UI: config, validation, describe, execute.

Three top-level modes, with runtime semantics the tests pin down:

    hybrid       the pre-router (heuristic or learned) picks cheap vs strong;
                 the existing run_task cascade (post-check, confidence gate,
                 escalation, safe fallbacks) is reused unchanged.
                 Pair types: remote_pair  (Fireworks cheap -> Fireworks strong)
                             local_remote (local cheap    -> Fireworks strong)
    remote_only  exactly one chosen Fireworks model; routing bypassed; a
                 failure is an error — the local backend is NEVER invoked.
    local_only   exactly one local model; routing bypassed; Fireworks is
                 NEVER contacted, not even as a fallback.

Session config is applied to the process `settings` singleton under a lock
and restored afterwards — per-request, never persisted, never written to
.env. The Fireworks API key is never read into any response.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import ROUTE_CHEAP, ROUTE_STRONG, settings
from remote_client import (
    FireworksClient,
    RemoteError,
    _allowed_models,
    _model_key,
)

MODES = ("hybrid", "remote_only", "local_only")
HYBRID_PAIRS = ("remote_pair", "local_remote")
ROUTER_KINDS = ("heuristic", "learned")

# Curated dropdown options. The UI also accepts free-text entries; whatever
# is chosen is validated against ALLOWED_MODELS when that is configured —
# never silently substituted.
KNOWN_LOCAL_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]
KNOWN_FIREWORKS_MODELS = [
    "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
    "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/deepseek-v4p1-flash",
    "accounts/fireworks/models/glm-5p2",
]

# One request at a time: execution mutates the process-wide settings
# singleton (snapshot/restore), so concurrent runs must serialize.
_EXEC_LOCK = threading.RLock()


@dataclass
class DemoConfig:
    mode: str = "hybrid"
    hybrid_pair: str = "local_remote"
    router_kind: str = "heuristic"
    local_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    cheap_model: str = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
    strong_model: str = "accounts/fireworks/models/gpt-oss-120b"
    remote_model: str = "accounts/fireworks/models/gpt-oss-120b"  # remote_only
    confidence_threshold: float = 0.55
    enable_escalation: bool = True
    mock: bool = True


def config_from_payload(payload: Dict[str, Any]) -> Tuple[DemoConfig, List[str]]:
    """Build + validate a DemoConfig from a request body. Returns
    (config, errors); a non-empty error list means DO NOT execute."""
    cfg = DemoConfig()
    errors: List[str] = []

    def _str(key: str, default: str) -> str:
        value = payload.get(key, default)
        return value.strip() if isinstance(value, str) else default

    cfg.mode = _str("mode", cfg.mode).lower()
    cfg.hybrid_pair = _str("hybrid_pair", cfg.hybrid_pair).lower()
    cfg.router_kind = _str("router_kind", cfg.router_kind).lower()
    cfg.local_model = _str("local_model", cfg.local_model)
    cfg.cheap_model = _str("cheap_model", cfg.cheap_model)
    cfg.strong_model = _str("strong_model", cfg.strong_model)
    cfg.remote_model = _str("remote_model", cfg.remote_model)
    cfg.enable_escalation = bool(payload.get("enable_escalation", True))
    cfg.mock = bool(payload.get("mock", True))

    if cfg.mode not in MODES:
        errors.append(f"mode must be one of {MODES}, got {cfg.mode!r}")
    if cfg.hybrid_pair not in HYBRID_PAIRS:
        errors.append(f"hybrid_pair must be one of {HYBRID_PAIRS}")
    if cfg.router_kind not in ROUTER_KINDS:
        errors.append(f"router_kind must be one of {ROUTER_KINDS}")

    try:
        cfg.confidence_threshold = float(
            payload.get("confidence_threshold", cfg.confidence_threshold)
        )
        if not (0.0 <= cfg.confidence_threshold <= 1.0):
            errors.append("confidence_threshold must be in [0, 1]")
    except (TypeError, ValueError):
        errors.append("confidence_threshold must be a number")

    for name, value in (
        ("local_model", cfg.local_model),
        ("cheap_model", cfg.cheap_model),
        ("strong_model", cfg.strong_model),
        ("remote_model", cfg.remote_model),
    ):
        if not value:
            errors.append(f"{name} must not be empty")

    # Validate every Fireworks model this mode would actually use against
    # ALLOWED_MODELS (when configured). No silent substitution, ever.
    for name, model in _fireworks_models_in_use(cfg):
        problem = validate_fireworks_model(model)
        if problem:
            errors.append(f"{name}: {problem}")

    return cfg, errors


def _fireworks_models_in_use(cfg: DemoConfig) -> List[Tuple[str, str]]:
    if cfg.mode == "local_only":
        return []
    if cfg.mode == "remote_only":
        return [("remote_model", cfg.remote_model)]
    pairs = [("strong_model", cfg.strong_model)]
    if cfg.hybrid_pair == "remote_pair":
        pairs.insert(0, ("cheap_model", cfg.cheap_model))
    return pairs


def validate_fireworks_model(model: str) -> Optional[str]:
    """None if usable; otherwise a human-readable rejection. Mirrors
    remote_client.resolve_tier_model but returns the message instead of
    printing, so the UI can display it."""
    allowed = _allowed_models()
    if allowed is None:
        return None
    if not allowed:
        return "ALLOWED_MODELS is set but empty — no Fireworks model is usable"
    if _model_key(model) not in {_model_key(m) for m in allowed}:
        return (
            f"{model!r} is not in ALLOWED_MODELS; pick one of the allowed "
            f"models — the demo never substitutes a different model silently"
        )
    return None


# ── Artifact status: is the learned router valid for the SELECTED pair? ──

def _selected_pair(cfg: DemoConfig) -> Tuple[str, str, str]:
    """(tier_mode, cheap_model, strong_model) the hybrid config selects."""
    if cfg.hybrid_pair == "remote_pair":
        return ("remote_pair", cfg.cheap_model, cfg.strong_model)
    return ("local_remote", cfg.local_model, cfg.strong_model)


def artifact_status(cfg: DemoConfig) -> Dict[str, Any]:
    """Load-check the configured artifact against the selected model pair.

    States:
      available=False                artifact missing/corrupt/incompatible
      pair_match=True                trained for exactly this pair — valid
      pair_match=False               trained for another pair — "router
                                     retraining required", learned is blocked
      pair_match=None (no metadata)  pre-pair-metadata artifact (the toy one):
                                     usable for MOCK demos only, always
                                     labelled; never real ML evidence
    """
    out: Dict[str, Any] = {
        "path": settings.router_artifact_path,
        "available": False,
        "pair_match": None,
        "version": None,
        "trained_pair": None,
        "detail": "",
    }
    try:
        from routing.artifact import load_artifact

        artifact = load_artifact(settings.router_artifact_path)
    except Exception as err:
        out["detail"] = str(err)
        return out

    out["available"] = True
    out["version"] = artifact.version
    meta = artifact.dataset_meta or {}
    trained_mode = meta.get("tier_mode")
    if not trained_mode:
        out["pair_match"] = None
        out["detail"] = (
            "artifact has no model-pair metadata (trained on synthetic toy "
            "data) — usable for mock demos only; router retraining required "
            "before any real claim"
        )
        return out

    trained = (
        str(trained_mode).strip().lower(),
        str(meta.get("local_model", "")),
        str(meta.get("remote_model", "")),
    )
    out["trained_pair"] = {
        "tier_mode": trained[0], "cheap": trained[1], "strong": trained[2],
    }
    selected = _selected_pair(cfg)
    if trained == selected:
        out["pair_match"] = True
        out["detail"] = "artifact was trained for exactly this model pair"
    else:
        out["pair_match"] = False
        out["detail"] = (
            f"router retraining required: artifact was trained for "
            f"{trained}, but the selected pair is {selected}. The heuristic "
            f"router remains available for this (untrained) pair."
        )
    return out


def _learned_router_usable(
    cfg: DemoConfig, status: Dict[str, Any]
) -> Tuple[bool, str]:
    """(usable, reason-if-not) for router_kind=learned under this config."""
    if not status["available"]:
        return False, f"learned router unavailable: {status['detail']}"
    if status["pair_match"] is False:
        return False, status["detail"]
    if status["pair_match"] is None and not cfg.mock:
        return False, (
            "the loaded artifact has no verified model pair (toy/synthetic "
            "training); learned routing is limited to mock demos until a "
            "router is trained for the selected pair"
        )
    return True, ""


# ── Plain-language description of the active configuration ───────────────

def describe(cfg: DemoConfig) -> Dict[str, Any]:
    lines: List[str] = []
    warnings: List[str] = []
    status = artifact_status(cfg)
    billable = False

    if cfg.mode == "hybrid":
        if cfg.hybrid_pair == "remote_pair":
            lines.append(
                f"Hybrid — cheap remote + strong remote: the "
                f"{cfg.router_kind} router sends each prompt to "
                f"{_short(cfg.cheap_model)} (cheap tier, Fireworks, BILLABLE) "
                f"or {_short(cfg.strong_model)} (strong tier, Fireworks, "
                f"BILLABLE)."
            )
            billable = True
        else:
            lines.append(
                f"Hybrid — local + remote: the {cfg.router_kind} router sends "
                f"each prompt to {_short(cfg.local_model)} (cheap tier, runs "
                f"on this machine, no API billing — compute and latency only) "
                f"or {_short(cfg.strong_model)} (strong tier, Fireworks, "
                f"BILLABLE)."
            )
            billable = True  # the strong tier can always be reached
        if cfg.router_kind == "learned":
            usable, reason = _learned_router_usable(cfg, status)
            if not usable:
                warnings.append(reason)
            elif status["pair_match"] is None:
                warnings.append(
                    "learned router is running a TOY artifact "
                    f"({status['version']}): trained on synthetic data, not "
                    "this model pair — demo only, not real ML evidence"
                )
            lines.append(
                "Routing threshold comes from the trained artifact"
                + (f" ({status['version']})" if status["available"] else "")
                + ", not the slider."
            )
        else:
            lines.append(
                f"Prompts scoring ≥ {cfg.confidence_threshold:.2f} on the "
                f"keyword heuristics stay on the cheap tier."
            )
            if status["pair_match"] is False:
                warnings.append(
                    "heuristic rules on an untrained pair: " + status["detail"]
                )
        lines.append(
            "Escalation is "
            + (
                "ON: a cheap answer that fails the post-check (or, for a "
                "local cheap tier, the confidence gate) is retried on the "
                "strong tier."
                if cfg.enable_escalation
                else "OFF: cheap answers are final."
            )
        )
    elif cfg.mode == "remote_only":
        lines.append(
            f"Remote only: every prompt goes to {_short(cfg.remote_model)} "
            f"on Fireworks. Routing is bypassed. Each request is a BILLABLE "
            f"API call. If the call fails, the demo reports the error — it "
            f"never falls back to a local model."
        )
        billable = True
    else:
        lines.append(
            f"Fully local: every prompt runs on {_short(cfg.local_model)} on "
            f"this machine. No API calls — Fireworks is never contacted, not "
            f"even as a fallback."
        )

    if cfg.mock:
        lines.append(
            "MOCK MODE: canned answers, no model weights, no network, no "
            "cost. Uncheck to use real backends (requires --real server "
            "flag)."
        )

    return {
        "summary": lines,
        "warnings": warnings,
        "billable": billable and not cfg.mock,
        "artifact": status,
    }


def _short(model: str) -> str:
    return model.rsplit("/", 1)[-1]


# ── Execution ────────────────────────────────────────────────────────────

_SETTINGS_FIELDS = (
    "mock_mode",
    "tier_mode",
    "router_mode",
    "local_model_name",
    "cheap_model_name",
    "strong_model_name",
    "remote_model_name",
    "confidence_threshold",
    "enable_escalation",
    "usage_log_path",
)


@contextmanager
def _session(cfg: DemoConfig):
    """Apply the session config to the settings singleton, restore after.
    Serialized: settings is process-global state."""
    with _EXEC_LOCK:
        saved = {f: getattr(settings, f) for f in _SETTINGS_FIELDS}
        try:
            settings.mock_mode = cfg.mock
            settings.tier_mode = cfg.hybrid_pair
            settings.router_mode = cfg.router_kind
            settings.local_model_name = cfg.local_model
            settings.cheap_model_name = cfg.cheap_model
            settings.strong_model_name = cfg.strong_model
            settings.remote_model_name = cfg.strong_model
            settings.confidence_threshold = cfg.confidence_threshold
            settings.enable_escalation = cfg.enable_escalation
            settings.usage_log_path = ""  # demo runs never pollute usage.jsonl
            yield
        finally:
            for f, v in saved.items():
                setattr(settings, f, v)


def _error(cfg: DemoConfig, message: str) -> Dict[str, Any]:
    return {"ok": False, "error": message, "mode": cfg.mode, "mock": cfg.mock}


def _fireworks_rates(model: str) -> Tuple[float, float, str]:
    """(input $/Mtok, output $/Mtok, rate-table label) for a Fireworks model.
    The cheap table applies only to the configured cheap model; anything else
    is estimated at the strong-tier table."""
    if _model_key(model) == _model_key(settings.cheap_model_name):
        return (
            settings.cheap_input_per_mtok,
            settings.cheap_output_per_mtok,
            "cheap-tier rate table",
        )
    return (
        settings.strong_input_per_mtok,
        settings.strong_output_per_mtok,
        "strong-tier rate table",
    )


def execute(cfg: DemoConfig, prompt: str) -> Dict[str, Any]:
    """Run one prompt under the session config. Never raises: every failure
    comes back as {"ok": False, "error": ...} for the UI to display."""
    prompt = (prompt or "").strip()
    if not prompt:
        return _error(cfg, "prompt is empty")
    with _session(cfg):
        try:
            if cfg.mode == "hybrid":
                return _run_hybrid(cfg, prompt)
            if cfg.mode == "remote_only":
                return _run_remote_only(cfg, prompt)
            return _run_local_only(cfg, prompt)
        except RemoteError as err:
            return _error(cfg, f"remote call failed: {err}")
        except Exception as err:  # surface, never crash the server
            return _error(cfg, f"{type(err).__name__}: {err}")


def _run_hybrid(cfg: DemoConfig, prompt: str) -> Dict[str, Any]:
    # Late imports keep service importable without torch/sklearn until used.
    from main import build_backends, build_router, run_task
    from schemas import Task
    from token_tracker import TokenTracker

    status = artifact_status(cfg)
    if cfg.router_kind == "learned":
        usable, reason = _learned_router_usable(cfg, status)
        if not usable:
            return _error(cfg, reason)

    try:
        router = build_router(threshold=cfg.confidence_threshold)
    except (RuntimeError, ValueError) as err:
        return _error(cfg, str(err))
    tier_mode, cheap, strong = build_backends()
    tracker = TokenTracker(log_path="")

    result = run_task(Task("webui", prompt), router, cheap, strong, tracker)
    rec = tracker.records[0]

    warnings = []
    if cfg.router_kind == "learned" and status["pair_match"] is None:
        warnings.append(
            "learned decision came from a TOY artifact (synthetic training "
            "data) — demo only, not real ML evidence"
        )

    billable_now = rec.billable_tokens > 0 and not cfg.mock
    return {
        "ok": True,
        "mode": "hybrid",
        "hybrid_pair": tier_mode,
        "mock": cfg.mock,
        "answer": result["answer"],
        "final_model": result["model_name"],
        "provider": result["provider"],
        "route": result["route"],
        "escalated": result["escalated"],
        "routing": {
            "kind": result["router"],
            "confidence": result["confidence"],
            "threshold": router.threshold,
            "artifact_version": result["artifact_version"],
            "reason": result["reason"],
            "signals": result["signals"],
            "local_confidence": result["local_confidence"],
        },
        "billable": billable_now,
        "would_bill": rec.billable_tokens > 0,
        "estimated_cost_usd": rec.estimated_cost_usd if not cfg.mock else 0.0,
        "mock_estimated_cost_usd": rec.estimated_cost_usd,
        "tokens": {
            "cheap_prompt": rec.cheap_prompt_tokens,
            "cheap_completion": rec.cheap_completion_tokens,
            "strong_prompt": rec.strong_prompt_tokens,
            "strong_completion": rec.strong_completion_tokens,
            "billable": rec.billable_tokens,
        },
        "latency_s": rec.latency_s,
        "problems": result["post_check_problems"],
        "warnings": warnings,
    }


def _run_remote_only(cfg: DemoConfig, prompt: str) -> Dict[str, Any]:
    # Exactly one Fireworks call. No router, no local backend, no fallback:
    # a RemoteError propagates to execute()'s handler and becomes an error
    # result — by design, this function never imports or touches LocalModel.
    client = FireworksClient(
        tier="single",
        route=ROUTE_STRONG,
        model_name=cfg.remote_model,
        max_tokens=settings.strong_max_tokens,
    )
    started = time.time()
    completion = client.generate(prompt)
    in_rate, out_rate, rate_label = _fireworks_rates(cfg.remote_model)
    cost = (
        completion.prompt_tokens * in_rate
        + completion.completion_tokens * out_rate
    ) / 1e6
    return {
        "ok": True,
        "mode": "remote_only",
        "mock": cfg.mock,
        "answer": completion.text,
        "final_model": completion.model_name,
        "provider": completion.provider,
        "route": "single",
        "escalated": False,
        "routing": {"kind": "bypassed", "reason": "remote-only mode"},
        "billable": not cfg.mock,
        "would_bill": True,
        "estimated_cost_usd": cost if not cfg.mock else 0.0,
        "mock_estimated_cost_usd": cost,
        "cost_rate_table": rate_label,
        "tokens": {
            "prompt": completion.prompt_tokens,
            "completion": completion.completion_tokens,
            "billable": completion.total_tokens,
        },
        "latency_s": round(time.time() - started, 3),
        "problems": [],
        "warnings": [],
    }


def _run_local_only(cfg: DemoConfig, prompt: str) -> Dict[str, Any]:
    # Exactly one local generation. No router, and Fireworks is never
    # contacted — this function never constructs a remote client, so there
    # is nothing to fall back TO.
    from local_model import LocalModel

    model = LocalModel(model_name=cfg.local_model)
    started = time.time()
    completion = model.generate(prompt)
    return {
        "ok": True,
        "mode": "local_only",
        "mock": cfg.mock,
        "answer": completion.text,
        "final_model": completion.model_name or cfg.local_model,
        "provider": completion.provider or "local",
        "route": "local",
        "escalated": False,
        "routing": {"kind": "bypassed", "reason": "fully-local mode"},
        "billable": False,
        "would_bill": False,
        "estimated_cost_usd": 0.0,
        "mock_estimated_cost_usd": 0.0,
        "tokens": {
            "prompt": completion.prompt_tokens,
            "completion": completion.completion_tokens,
            "billable": 0,
        },
        "local_confidence": completion.confidence,
        "latency_s": round(time.time() - started, 3),
        "problems": [],
        "warnings": [],
    }


def frontend_config() -> Dict[str, Any]:
    """Static data the UI needs at load. NEVER includes the API key."""
    allowed = _allowed_models()
    return {
        "defaults": DemoConfig().__dict__,
        "known_local_models": KNOWN_LOCAL_MODELS,
        "known_fireworks_models": KNOWN_FIREWORKS_MODELS,
        "allowed_models": allowed,  # None = unrestricted
        "api_key_present": bool(settings.fireworks_api_key),
        "artifact_path": settings.router_artifact_path,
    }
