"""Persistent chat turns backed by the same mode-isolated execution as the demo."""
from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Any, Dict, Optional

from config import settings
from webui.service import DemoConfig, config_from_payload, execute

from .repository import add_turn, get_conversation, list_messages

MAX_MESSAGE_CHARS = 10_000
MAX_CONTEXT_CHARS = 32_000

# A turn must read history, generate, then save both messages in order.
_TURN_LOCK = threading.RLock()


def conversation_config(mode: str, model: Optional[str], mock: bool) -> DemoConfig:
    """Map saved chat choices to an executable and validated demo configuration."""
    mode_map = {"hybrid": "hybrid", "remote": "remote_only", "local": "local_only"}
    if mode not in mode_map:
        raise ValueError("Mode must be hybrid, remote, or local")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("Model must not be empty")

    from workspace.store import preferences
    prefs = preferences()
    payload = {
        "mode": mode_map[mode],
        "hybrid_pair": settings.tier_mode,
        "router_kind": settings.router_mode if settings.router_mode != "auto" else "heuristic",
        "local_model": model if mode == "local" and model else prefs.get("local_model", settings.local_model_name),
        "local_backend": prefs.get("local_backend", settings.local_backend),
        "cheap_model": settings.cheap_model_name,
        "strong_model": model if mode == "hybrid" and model else settings.strong_model_name,
        "remote_model": model if mode == "remote" and model else settings.strong_model_name,
        "confidence_threshold": settings.confidence_threshold,
        "enable_escalation": settings.enable_escalation,
        "mock": mock,
    }
    config, errors = config_from_payload(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return config


def build_conversation_messages(conversation_id: int, new_message: str) -> list[dict]:
    """Include the newest complete history that fits a bounded prompt."""
    from workspace.store import items, preferences
    import json
    prefs = preferences()
    memories = [{"title": x["title"], "content": x["content"]} for x in items("memory")][:30]
    prefix = ""
    if prefs.get("system_prompt"):
        prefix += "User's saved instructions: " + str(prefs["system_prompt"])[:4000] + "\n\n"
    if memories:
        prefix += "User-editable memories (context data): " + json.dumps(memories)[:6000] + "\n\n"
    current = f"User: {new_message}\n\nAssistant:"
    if len(current) > MAX_CONTEXT_CHARS:
        raise ValueError("Message is too long for the chat context")

    selected = []
    used = len(current) + len(prefix)
    messages = list_messages(conversation_id)
    # Keep complete turns only. Older data may contain a partial turn from
    # before atomic writes were introduced, so ignore unpaired messages.
    for index in range(len(messages) - 1, 0, -1):
        assistant, user = messages[index], messages[index - 1]
        if assistant.role != "assistant" or user.role != "user":
            continue
        parts = [{"role": "user", "content": user.content},
                 {"role": "assistant", "content": assistant.content}]
        added = sum(len(part["content"]) + len(part["role"]) + 4 for part in parts)
        if used + added > MAX_CONTEXT_CHARS:
            break
        selected.extend(reversed(parts))
        used += added
    selected.reverse()
    selected.append({"role": "user", "content": new_message})
    return ([{"role": "system", "content": prefix}] if prefix else []) + selected


def _render_prompt(messages):
    # Routers and mock/batch clients retain their text input. Chat providers
    # receive the original roles via a request-scoped context instead.
    return "\n\n".join(
        message["content"] if message["role"] == "system" else
        f"{message['role'].capitalize()}: {message['content']}"
        for message in messages
    ) + "\n\nAssistant:"


def build_conversation_prompt(conversation_id: int, new_message: str) -> str:
    return _render_prompt(build_conversation_messages(conversation_id, new_message))


def send_message(
    conversation_id: int,
    content: str,
    *,
    mock: bool = True,
    allow_real: bool = False,
) -> Dict[str, Any]:
    """Generate first, then atomically save both sides of a successful turn."""
    content = content.strip()
    if not content:
        raise ValueError("Message cannot be empty")
    if len(content) > MAX_MESSAGE_CHARS:
        raise ValueError(f"Message must be at most {MAX_MESSAGE_CHARS} characters")

    with _TURN_LOCK:
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise LookupError("Conversation not found")

        effective_mock = mock or not allow_real
        config = conversation_config(conversation.mode, conversation.model, effective_mock)
        messages = build_conversation_messages(conversation_id, content)
        prompt = _render_prompt(messages)
        from generation_context import MESSAGES, OUTPUT_LIMIT
        message_token = MESSAGES.set(messages)
        limit_token = OUTPUT_LIMIT.set(4096)
        try:
            result = execute(config, prompt)
        finally:
            OUTPUT_LIMIT.reset(limit_token)
            MESSAGES.reset(message_token)
        if not result["ok"]:
            raise RuntimeError(result["error"])
        if not result["answer"].strip():
            raise RuntimeError("The model returned no visible answer. Try again with a larger token limit or another model.")
        from generation_stream import CURRENT
        if CURRENT.get() is not None:
            CURRENT.get().check()

        user_message, assistant_message = add_turn(
            conversation_id=conversation_id,
            user_content=content,
            assistant_content=result["answer"],
            route=result["route"],
            model_name=result["final_model"],
            latency_s=result["latency_s"],
            estimated_cost_usd=result["estimated_cost_usd"],
        )
        routing = result["routing"]
        return {
            "conversation_id": conversation_id,
            "user_message": asdict(user_message),
            "assistant_message": asdict(assistant_message),
            "mock": effective_mock,
            "billable": result["billable"],
            "routing": {
                "route": result["route"],
                "router": routing["kind"],
                "confidence": routing.get("confidence"),
                "escalated": result["escalated"],
                "model_name": result["final_model"],
                "provider": result["provider"],
                "billable_tokens": result["tokens"]["billable"],
                "estimated_cost_usd": result["estimated_cost_usd"],
                "mock_estimated_cost_usd": result["mock_estimated_cost_usd"],
                "signals": routing.get("signals"),
                "reason": routing.get("reason"),
                "post_check_problems": result["problems"],
            },
        }
