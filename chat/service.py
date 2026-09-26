from typing import Dict

from schemas import Task
from main import build_backends, build_router, run_task
from token_tracker import TokenTracker

from .repository import (
    add_message,
    get_conversation,
    list_messages,
)


# Build these once when the service module loads.
# That avoids rebuilding/loading the router and models for every message.
router = build_router()
tier_mode, cheap_backend, strong_backend = build_backends()
tracker = TokenTracker()


def build_conversation_prompt(conversation_id: int, new_message: str) -> str:
    """
    Build a multi-turn prompt using the previous messages in the conversation.

    Example:
        User: What is binary search?
        Assistant: Binary search is...
        User: Explain the middle step again.
    """
    messages = list_messages(conversation_id)

    parts = []

    for message in messages:
        if message.role == "user":
            parts.append(f"User: {message.content}")
        elif message.role == "assistant":
            parts.append(f"Assistant: {message.content}")

    parts.append(f"User: {new_message}")
    parts.append("Assistant:")

    return "\n\n".join(parts)


def send_message(
    conversation_id: int,
    content: str,
) -> Dict:
    """
    Save the user message, run it through Transit,
    save the assistant response, and return the result.
    """

    conversation = get_conversation(conversation_id)

    if conversation is None:
        raise ValueError(
            f"Conversation with id {conversation_id} does not exist"
        )

    if not content.strip():
        raise ValueError("Message cannot be empty")

    # Important:
    # build the prompt BEFORE saving the new user message,
    # otherwise the newest message would appear twice.
    prompt = build_conversation_prompt(
        conversation_id=conversation_id,
        new_message=content,
    )

    # Save the actual user message in SQLite.
    user_message = add_message(
        conversation_id=conversation_id,
        role="user",
        content=content,
    )

    # Convert the chat request into the existing routing Task format.
    task = Task(
        task_id=f"conversation-{conversation_id}-message-{user_message.id}",
        prompt=prompt,
        metadata={
            "conversation_id": conversation_id,
            "mode": conversation.mode,
            "model": conversation.model,
        },
    )

    # Reuse your teammate's existing router/model pipeline.
    result = run_task(
        task=task,
        router=router,
        cheap=cheap_backend,
        strong=strong_backend,
        tracker=tracker,
    )

    answer = result.get("answer", "")

    # Save assistant response and routing metadata.
    assistant_message = add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        route=result.get("route"),
        model_name=result.get("model_name"),
        latency_s=result.get("latency_s"),
        estimated_cost_usd=result.get("estimated_cost_usd"),
    )

    return {
        "conversation_id": conversation_id,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "routing": {
            "route": result.get("route"),
            "router": result.get("router"),
            "confidence": result.get("confidence"),
            "escalated": result.get("escalated"),
            "model_name": result.get("model_name"),
            "provider": result.get("provider"),
            "billable_tokens": result.get("billable_tokens"),
            "estimated_cost_usd": result.get("estimated_cost_usd"),
            "signals": result.get("signals"),
            "reason": result.get("reason"),
            "post_check_problems": result.get("post_check_problems"),
        },
    }