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
ROUTE_LOCAL = "local"
ROUTE_REMOTE = "remote"
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
    # ── Local model ──────────────────────────────────────────────────────
    # Placeholder: a 1.5B instruct model that runs on CPU. Swap via
    # LOCAL_MODEL_NAME to swap in a different local model.
    local_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_max_new_tokens: int = 512

    # ── Remote model (Fireworks AI) ──────────────────────────────────────
    # llama-v3p3-70b-instruct (the original placeholder) was retired from
    # Fireworks serverless — the API 404s on it (verified live 2026-07-04).
    # deepseek-v4-pro won the available-models bake-off: flagship accuracy
    # and the FEWEST completion tokens on a trivial prompt (41 vs 48–85 for
    # gpt-oss-120b / glm-5p1 / glm-5p2 / kimi-k2p6 — every serverless chat
    # model now bills hidden reasoning tokens into completion usage).
    remote_model_name: str = "accounts/fireworks/models/deepseek-v4-pro"
    # Optional allow-list: when ALLOWED_MODELS is set (comma-separated model
    # IDs), the remote model MUST come from it — useful for cost control or
    # when a deployment restricts which models may be called. Empty = use
    # remote_model_name. Selection: remote_client.resolve_remote_model().
    allowed_models: str = ""
    # Tie-breaker when ALLOWED_MODELS has several entries: first name here
    # (comma-separated, matched on the ID's last path segment) that appears
    # in the allow-list wins; otherwise the list's first entry. deepseek won
    # the 2026-07-04 bake-off (flagship quality, fewest completion tokens).
    remote_model_preference: str = "deepseek-v4-pro"
    fireworks_api_key: str = ""  # set FIREWORKS_API_KEY; never commit a key
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    # 4096, not 1024: reasoning models spend completion budget on thinking
    # first — at 1024 the hard sample task came back truncated mid-thought
    # (billed but useless). Truncation fails accuracy; the router already
    # sends only hard tasks remote, so the bigger cap only pays when needed.
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

    # ── Routing: the dials that decide the score ─────────────────────────
    # Queries whose confidence >= threshold go LOCAL (free tokens).
    # Lower threshold  = more local = fewer billable tokens, more accuracy risk.
    # Higher threshold = safer, more expensive.
    # THE single most important number to calibrate (see EVALUATION.md).
    confidence_threshold: float = 0.55
    # If a local answer fails router.post_check, retry the task remotely
    # instead of submitting a probably-wrong answer. Costs remote tokens only
    # when the local gamble actually failed — cheap insurance for accuracy.
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
        s.local_model_name = _env_str("LOCAL_MODEL_NAME", s.local_model_name)
        s.local_max_new_tokens = _env_int("LOCAL_MAX_NEW_TOKENS", s.local_max_new_tokens)
        s.remote_model_name = _env_str("REMOTE_MODEL_NAME", s.remote_model_name)
        s.allowed_models = _env_str("ALLOWED_MODELS", s.allowed_models)
        s.remote_model_preference = _env_str(
            "REMOTE_MODEL_PREFERENCE", s.remote_model_preference
        )
        s.fireworks_api_key = _env_str("FIREWORKS_API_KEY", s.fireworks_api_key)
        s.fireworks_base_url = _env_str("FIREWORKS_BASE_URL", s.fireworks_base_url)
        s.remote_max_tokens = _env_int("REMOTE_MAX_TOKENS", s.remote_max_tokens)
        s.connect_timeout_s = _env_float("CONNECT_TIMEOUT_S", s.connect_timeout_s)
        s.request_timeout_s = _env_float("REQUEST_TIMEOUT_S", s.request_timeout_s)
        s.max_retries = _env_int("MAX_RETRIES", s.max_retries)
        s.system_prompt = _env_str("SYSTEM_PROMPT", s.system_prompt)
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
