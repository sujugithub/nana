"""Central configuration for the hybrid routing agent.

Everything you should need to tune lives HERE (plus, maybe,
keyword patterns in confidence.py). Every value can also be overridden with an
environment variable, so you can tune the router inside the container without
rebuilding the image:

    docker run --rm -e CONFIDENCE_THRESHOLD=0.7 -e LOCAL_MODEL_NAME=... agent
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Route names used across the codebase — import these, never hardcode strings.
# Route labels describe policy tiers, not hosting. The provider/model fields in
# each Completion and usage record say where the answer actually ran.
ROUTE_CHEAP = "cheap"
ROUTE_STRONG = "strong"
ROUTE_LOCAL = ROUTE_CHEAP       # compatibility alias for ML/runtime internals
ROUTE_REMOTE = ROUTE_STRONG     # compatibility alias for ML/runtime internals
ROUTE_ERROR = "error"  # tracker-only sentinel: task crashed, no answer produced


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value in (None, "") else value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # cheap tier backend:
    #   remote_pair  -> Nemotron Lightning on Fireworks (demo, no local load)
    #   local_remote -> local_model.py / LOCAL_MODEL_NAME (original FYP mode)
    # The strong tier is Fireworks-hosted in both modes.
    tier_mode: str = "local_remote"

    # ── Local cheap-tier backend ─────────────────────────────────────────
    # Used when TIER_MODE=local_remote. It is deliberately a first-class
    # runtime option, not merely training-data compatibility.
    local_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_backend: str = "transformers"
    ollama_base_url: str = "http://127.0.0.1:11434"
    local_max_new_tokens: int = 512

    # ── Two Fireworks-hosted routing tiers ───────────────────────────────
    # In remote_pair these are both remote; the strong model is also used by
    # local_remote. Keep that distinction honest in logs and reports.
    cheap_model_name: str = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
    strong_model_name: str = "accounts/fireworks/models/gpt-oss-120b"
    cheap_max_tokens: int = 1024
    strong_max_tokens: int = 4096
    # Explicit USD / 1M-token assumptions for demo accounting. Override when
    # Fireworks pricing changes; actual token counts still come from `usage`.
    cheap_input_per_mtok: float = 0.05
    cheap_output_per_mtok: float = 0.20
    strong_input_per_mtok: float = 0.15
    strong_output_per_mtok: float = 0.60

    # Backward-compatible strong-tier settings used by the old harness and
    # archived deployment environment. New code should use strong_model_name.
    remote_model_name: str = "accounts/fireworks/models/gpt-oss-120b"
    # Optional allow-list: when ALLOWED_MODELS is set (comma-separated model
    # IDs), the remote model MUST come from it — useful for cost control or
    # when a deployment restricts which models may be called. Empty = use
    # remote_model_name. Selection: remote_client.resolve_remote_model().
    allowed_models: str = ""
    # Legacy single-remote tie-breaker, retained for compatibility helpers.
    remote_model_preference: str = "gpt-oss-120b"
    fireworks_api_key: str = ""  # set FIREWORKS_API_KEY; never commit a key
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    # Legacy alias for strong_max_tokens.
    remote_max_tokens: int = 4096
    connect_timeout_s: float = 10.0  # slow handshakes fail fast; safe to retry
    # READ timeout — must cover a full remote generation at remote_max_tokens.
    # Read timeouts are NOT retried (the server may have billed the tokens).
    request_timeout_s: float = 120.0
    max_retries: int = 3  # retries for connect errors / 429 / 5xx only

    # ── Answer style (accuracy gate + token rank) ────────────────────────
    # System prompt for BOTH backends. The judge scores the ANSWER and the
    # rank counts remote completion tokens, so demand directness. The
    # "exactly one of the options" clause targets an observed live failure
    # (2026-07-07: local answered "mixed" to a positive/negative/neutral
    # sentiment question). Empty disables. Costs ~30 prompt tokens per
    # remote call; buys back far more in completions.
    system_prompt: str = (
        "Answer directly and concisely. If the question offers answer "
        "options, reply with exactly one of them. Give the final answer "
        "with only the essential reasoning - no filler, no restating the "
        "question, no extra background."
    )

    # ── Router selection ─────────────────────────────────────────────────
    # heuristic: keyword rules in confidence.py (always available, no deps).
    # learned:   trained artifact (ROUTER_ARTIFACT) — FAILS LOUDLY at startup
    #            if the artifact is missing or incompatible.
    # auto:      learned if the artifact loads, otherwise heuristic with a
    #            visible warning on stderr.
    router_mode: str = "heuristic"
    # Path to the trained pre-router artifact (see routing/train.py).
    # SECURITY: joblib artifacts execute code on load — point this only at
    # files you trained yourself (routing/artifact.py docstring).
    router_artifact_path: str = "artifacts/router.joblib"

    # ── Routing: the dials that decide the score ─────────────────────────
    # Queries whose confidence >= threshold go to the CHEAP tier.
    # Lower threshold  = more cheap-tier use, lower cost, more quality risk.
    # Higher threshold = safer, more expensive.
    # THE single most important number to calibrate (see EVALUATION.md).
    confidence_threshold: float = 0.55
    # If a cheap-tier answer fails router.post_check, retry on the strong tier.
    enable_escalation: bool = True
    # Only truly-EMPTY local output counts as a failure by default: a 1-char
    # answer ("B", "7") can be exactly right on multiple-choice/short-answer
    # sets, and those route local — flagging them would force a paid
    # escalation on every correct answer. Raise only if the real task set
    # never has short answers.
    post_check_min_chars: int = 1
    # Draft-and-judge gate: the local model's OWN mean token probability for
    # its answer (Completion.confidence). Below this → escalate to remote
    # even if post_check passed — a fluent-but-unsure answer is the failure
    # mode regexes can't see. 0.4 is a placeholder; CALIBRATE it by
    # comparing logged local_confidence against graded answers
    # (scripts/calibrate.py). Set to 0 to disable the gate.
    logprob_confidence_threshold: float = 0.4
    # Which statistic of the local model's token probabilities the gate
    # compares against the threshold above. All are computed from logits the
    # forward pass already produced (zero extra compute):
    #   mean      mean token probability (default; flatters short answers)
    #   min       minimum token probability (harshest single-token view)
    #   low_frac  1 - fraction of tokens below 0.5 probability
    # Higher ALWAYS means safer-to-keep-local. Pick with evidence: the
    # training report's post-gen AUC comparison (routing/train.py) says
    # which statistic actually separates right from wrong local answers.
    local_conf_stat: str = "mean"

    # ── Concurrency and deadlines ────────────────────────────────────────
    # Worker threads for the task pool. Remote calls (~27 s each observed)
    # parallelize up to this; local generation serializes on the model lock
    # regardless, so this dial only bounds concurrent Fireworks requests.
    remote_concurrency: int = 8
    # Seconds from process start until we stop waiting on unfinished tasks
    # and write results.json with what we have. The scoring cap is 600 s
    # TOTAL (container start → exit); 540 leaves slack for model load,
    # result writing, and container overhead.
    run_deadline_s: float = 540.0

    # ── Infra ────────────────────────────────────────────────────────────
    mock_mode: bool = False  # AGENT_MOCK=1 → no weights, no network (wiring tests)
    usage_log_path: str = "logs/usage.jsonl"

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.tier_mode = _env_str("TIER_MODE", s.tier_mode)
        s.local_model_name = _env_str("LOCAL_MODEL_NAME", s.local_model_name)
        s.local_backend = _env_str("LOCAL_BACKEND", s.local_backend)
        s.ollama_base_url = _env_str("OLLAMA_BASE_URL", s.ollama_base_url)
        s.local_max_new_tokens = _env_int("LOCAL_MAX_NEW_TOKENS", s.local_max_new_tokens)
        s.cheap_model_name = _env_str("CHEAP_MODEL_NAME", s.cheap_model_name)
        # REMOTE_MODEL_NAME remains a supported fallback for existing .env
        # files, while STRONG_MODEL_NAME is the clear two-tier spelling.
        legacy_remote = _env_str("REMOTE_MODEL_NAME", s.remote_model_name)
        s.strong_model_name = _env_str("STRONG_MODEL_NAME", legacy_remote)
        s.remote_model_name = s.strong_model_name
        s.allowed_models = _env_str("ALLOWED_MODELS", s.allowed_models)
        s.remote_model_preference = _env_str(
            "REMOTE_MODEL_PREFERENCE", s.remote_model_preference
        )
        s.fireworks_api_key = _env_str("FIREWORKS_API_KEY", s.fireworks_api_key)
        s.fireworks_base_url = _env_str("FIREWORKS_BASE_URL", s.fireworks_base_url)
        s.cheap_max_tokens = _env_int("CHEAP_MAX_TOKENS", s.cheap_max_tokens)
        legacy_max = _env_int("REMOTE_MAX_TOKENS", s.remote_max_tokens)
        s.strong_max_tokens = _env_int("STRONG_MAX_TOKENS", legacy_max)
        s.remote_max_tokens = s.strong_max_tokens
        s.cheap_input_per_mtok = _env_float(
            "CHEAP_INPUT_PER_MTOK", s.cheap_input_per_mtok
        )
        s.cheap_output_per_mtok = _env_float(
            "CHEAP_OUTPUT_PER_MTOK", s.cheap_output_per_mtok
        )
        s.strong_input_per_mtok = _env_float(
            "STRONG_INPUT_PER_MTOK", s.strong_input_per_mtok
        )
        s.strong_output_per_mtok = _env_float(
            "STRONG_OUTPUT_PER_MTOK", s.strong_output_per_mtok
        )
        s.connect_timeout_s = _env_float("CONNECT_TIMEOUT_S", s.connect_timeout_s)
        s.request_timeout_s = _env_float("REQUEST_TIMEOUT_S", s.request_timeout_s)
        s.max_retries = _env_int("MAX_RETRIES", s.max_retries)
        s.system_prompt = _env_str("SYSTEM_PROMPT", s.system_prompt)
        s.router_mode = _env_str("ROUTER_MODE", s.router_mode)
        s.router_artifact_path = _env_str("ROUTER_ARTIFACT", s.router_artifact_path)
        s.local_conf_stat = _env_str("LOCAL_CONF_STAT", s.local_conf_stat)
        s.confidence_threshold = _env_float("CONFIDENCE_THRESHOLD", s.confidence_threshold)
        s.enable_escalation = _env_bool("ENABLE_ESCALATION", s.enable_escalation)
        s.post_check_min_chars = _env_int("POST_CHECK_MIN_CHARS", s.post_check_min_chars)
        s.logprob_confidence_threshold = _env_float(
            "LOGPROB_CONFIDENCE_THRESHOLD", s.logprob_confidence_threshold
        )
        s.remote_concurrency = _env_int("REMOTE_CONCURRENCY", s.remote_concurrency)
        s.run_deadline_s = _env_float("RUN_DEADLINE_S", s.run_deadline_s)
        s.mock_mode = _env_bool("AGENT_MOCK", s.mock_mode)
        s.usage_log_path = _env_str("USAGE_LOG_PATH", s.usage_log_path)
        return s


# Singleton read by every module. main.py mutates it for CLI overrides
# (--mock, --threshold) BEFORE constructing the router/models, so construct
# components after applying overrides.
settings = Settings.from_env()
