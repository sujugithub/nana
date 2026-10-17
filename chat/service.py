"""Persistent chat turns backed by the same mode-isolated execution as the demo."""
from __future__ import annotations

import threading
import time
from contextlib import nullcontext
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


def build_conversation_messages(conversation_id: int, new_message: str, *, context_chars=None, attachment_config=None, replay_images=True) -> list[dict]:
    """Include the newest complete history that fits a bounded prompt."""
    from workspace.store import items, preferences
    import json
    budget = MAX_CONTEXT_CHARS if context_chars is None else min(MAX_CONTEXT_CHARS, context_chars)
    prefs = preferences()
    from workspace.context import relevant
    memories = relevant("memory", new_message, scope=f"chat:{conversation_id}")
    prefix = ""
    if prefs.get("system_prompt"):
        prefix += "User's saved instructions: " + str(prefs["system_prompt"])[:4000] + "\n\n"
    while memories and len(prefix) + len(json.dumps(memories)) + len(new_message) + 100 > budget:
        memories.pop()
    if memories:
        prefix += "User-editable memories (context data): " + json.dumps(memories) + "\n\n"
    current = f"User: {new_message}\n\nAssistant:"
    if len(current) + len(prefix) > budget:
        raise ValueError("Message is too long for the chat context")

    selected = []
    used = len(current) + len(prefix)
    messages = list_messages(conversation_id)
    historical_images = 0
    # Keep complete turns only. Older data may contain a partial turn from
    # before atomic writes were introduced, so ignore unpaired messages.
    for index in range(len(messages) - 1, 0, -1):
        assistant, user = messages[index], messages[index - 1]
        if assistant.role != "assistant" or user.role != "user":
            continue
        user_context = user.content
        turn_images = []
        if user.attachments:
            from .attachments import resolve
            docs = []
            for meta in user.attachments:
                try:
                    item = resolve([meta["id"]])[0]
                    if replay_images and attachment_config and item["meta"].get("images") and historical_images + item["meta"]["images"] <= 3 and attachment_config.mode == "local_only":
                        from .attachments import require_vision
                        from workspace.tools import file_path
                        require_vision("local", attachment_config.local_model)
                        turn_images += json.loads(file_path(item["meta"]["filename"] + ".images.json").read_text())
                    docs.append({"name": item["title"], "id": item["id"], "content": item["content"][:3000], "total_chars": len(item["content"]), "images": item["meta"].get("images", 0)})
                except (LookupError, ValueError):
                    docs.append({"name": meta["name"], "unavailable": True})
            user_context += "\nPrior attachment data (excerpts, images may be omitted when unavailable or outside the selected model/context; reattach if needed): " + json.dumps(docs)
        parts = [{"role": "user", "content": user_context},
                 {"role": "assistant", "content": assistant.content}]
        if turn_images:
            parts[0]["images"] = turn_images
        added = 3000 * len(turn_images) + sum(len(part["content"]) + len(part["role"]) + 4 for part in parts)
        if used + added > budget:
            break
        historical_images += len(turn_images)
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
    web: bool = False,
    attachments=None,
) -> Dict[str, Any]:
    """Generate first, then atomically save both sides of a successful turn."""
    if not isinstance(content, str):
        raise ValueError("Message must be text")
    content = content.strip()
    if not isinstance(web, bool):
        raise ValueError("Web must be true or false")
    if not content:
        raise ValueError("Message cannot be empty")
    if len(content) > MAX_MESSAGE_CHARS:
        raise ValueError(f"Message must be at most {MAX_MESSAGE_CHARS} characters")

    with _TURN_LOCK:
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise LookupError("Conversation not found")

        from .tool_turns import pending
        if pending(conversation_id):
            raise ValueError("Finish or cancel this chat's workspace run before sending another message")
        effective_mock = mock or not allow_real
        config = conversation_config(conversation.mode, conversation.model, effective_mock)
        from . import web as web_chat
        use_web = web and not effective_mock
        started = time.monotonic()
        with web_chat.bounded_stream() if use_web else nullcontext():
            web_context = web_chat.collect(content, config, execute) if use_web else None
            messages = build_conversation_messages(
                conversation_id, content,
                context_chars=web_chat.message_budget(config, web_context) if web_context else 8500 if attachments else None,
                attachment_config=config, replay_images=not attachments and not web)
            from .attachments import ground
            messages, attachment_meta = ground(messages, attachments or [], config)
            if attachments:
                # Keep a conservative allowance for document content and history.
                messages = [messages[0]] + build_conversation_messages(conversation_id, messages[-1]["content"], context_chars=8500, attachment_config=config, replay_images=False)
                if attachment_meta and any(x["images"] for x in attachment_meta):
                    grounded, _ = ground([{"role": "user", "content": content}], attachments, config)
                    messages[-1]["images"] = grounded[-1]["images"]
            if web_context:
                if attachments:
                    raise ValueError("Use attachments or Web sources in a turn; workspace tools can read both explicitly")
                messages = web_chat.ground(messages, web_context)
            prompt = _render_prompt(messages)
            from generation_context import MESSAGES, OUTPUT_LIMIT, LOCAL_CONTEXT, TOOLS
            tokens = [(MESSAGES, MESSAGES.set(messages)), (OUTPUT_LIMIT, OUTPUT_LIMIT.set(4096))]
            if attachments:
                tokens += [(LOCAL_CONTEXT, LOCAL_CONTEXT.set(4096 if "27b" in config.local_model.lower() else 8192))]
            if web_context:
                tokens += [(LOCAL_CONTEXT, LOCAL_CONTEXT.set(web_chat.CONTEXT)), (TOOLS, TOOLS.set(None))]
            try:
                result = execute(config, prompt)
            finally:
                for variable, token in reversed(tokens):
                    variable.reset(token)
            if not result["ok"]:
                raise RuntimeError(result["error"])
            if not result["answer"].strip():
                raise RuntimeError("The model returned no visible answer. Try again with a larger token limit or another model.")
            if web_context:
                result = web_chat.finish(result, web_context)
                result["latency_s"] = round(time.monotonic() - started, 3)
                from generation_stream import CURRENT
                CURRENT.get().check()
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
            attachments=attachment_meta,
        )
        routing = result["routing"]
        return {
            "conversation_id": conversation_id,
            "user_message": asdict(user_message),
            "assistant_message": asdict(assistant_message),
            "mock": effective_mock,
            "billable": result["billable"],
            "web": result.get("web", {"enabled": False}),
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
