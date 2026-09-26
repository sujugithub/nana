import time
from typing import Dict, Optional

from local_model import LocalModel
from main import build_backends, build_router, run_task
from remote_client import StrongRemoteClient
from schemas import Task
from token_tracker import TokenTracker

from .repository import (
    add_message,
    get_conversation,
    list_messages,
)


# ---------------------------------------------------------
# Shared Transit components
# ---------------------------------------------------------

router = build_router()

tier_mode, hybrid_cheap_backend, hybrid_strong_backend = (
    build_backends()
)

tracker = TokenTracker()


# Forced-mode backends are lazy-loaded so starting FastAPI
# does not automatically load the local model.
_forced_local_backend: Optional[LocalModel] = None
_forced_remote_backend: Optional[StrongRemoteClient] = None


def get_local_backend() -> LocalModel:
    global _forced_local_backend

    if _forced_local_backend is None:
        _forced_local_backend = LocalModel()
        _forced_local_backend.load()

    return _forced_local_backend


def get_remote_backend() -> StrongRemoteClient:
    global _forced_remote_backend

    if _forced_remote_backend is None:
        _forced_remote_backend = StrongRemoteClient()

    return _forced_remote_backend


# ---------------------------------------------------------
# Conversation context
# ---------------------------------------------------------

def build_conversation_prompt(
    conversation_id: int,
    new_message: str,
) -> str:
    """
    Build a multi-turn prompt using the existing conversation history.
    """

    messages = list_messages(conversation_id)

    parts = []

    for message in messages:
        if message.role == "user":
            parts.append(
                f"User: {message.content}"
            )

        elif message.role == "assistant":
            parts.append(
                f"Assistant: {message.content}"
            )

    parts.append(
        f"User: {new_message}"
    )

    parts.append(
        "Assistant:"
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------
# Forced remote mode
# ---------------------------------------------------------

def run_remote_only(
    task: Task,
) -> Dict:
    """
    Bypass Transit routing completely and always use the
    strong remote backend.
    """

    started = time.time()

    backend = get_remote_backend()

    completion = backend.generate(
        task.prompt
    )

    latency_s = (
        time.time()
        - started
    )

    record = tracker.record(
        task_id=task.task_id,

        route=completion.source,

        escalated=False,

        local=None,

        remote=completion,

        confidence=0.0,

        threshold=0.0,

        signals={},

        problems=[],

        local_confidence=None,

        latency_s=latency_s,

        local_min_token_prob=None,

        local_low_token_frac=None,

        router="forced_remote",

        artifact_version=None,

        p_local=None,
    )

    return {
        "task_id": task.task_id,

        "route": completion.source,

        "escalated": False,

        "confidence": 0.0,

        "router": "forced_remote",

        "artifact_version": None,

        "local_confidence": None,

        "signals": {},

        "reason": (
            "Remote mode selected by user"
        ),

        "post_check_problems": [],

        "billable_tokens": (
            record.billable_tokens
        ),

        "estimated_cost_usd": (
            record.estimated_cost_usd
        ),

        "model_name": (
            completion.model_name
        ),

        "provider": (
            completion.provider
        ),

        "latency_s": (
            latency_s
        ),

        "answer": (
            completion.text
        ),
    }


# ---------------------------------------------------------
# Forced local mode
# ---------------------------------------------------------

def run_local_only(
    task: Task,
) -> Dict:
    """
    Bypass Transit routing completely and always use the
    local model.

    Post-checks can still run for diagnostics, but this mode
    NEVER escalates to Fireworks.
    """

    started = time.time()

    backend = get_local_backend()

    completion = backend.generate(
        task.prompt
    )

    latency_s = (
        time.time()
        - started
    )

    ok, problems = router.post_check(
        task.prompt,
        completion.text,
    )

    if ok:
        problems = []

    record = tracker.record(
        task_id=task.task_id,

        route=completion.source,

        escalated=False,

        local=completion,

        remote=None,

        confidence=0.0,

        threshold=0.0,

        signals={},

        problems=problems,

        local_confidence=(
            completion.confidence
        ),

        latency_s=latency_s,

        local_min_token_prob=(
            completion.min_token_prob
        ),

        local_low_token_frac=(
            completion.low_token_frac
        ),

        router="forced_local",

        artifact_version=None,

        p_local=None,
    )

    return {
        "task_id": task.task_id,

        "route": (
            completion.source
        ),

        "escalated": False,

        "confidence": 0.0,

        "router": "forced_local",

        "artifact_version": None,

        "local_confidence": (
            completion.confidence
        ),

        "signals": {},

        "reason": (
            "Local mode selected by user"
        ),

        "post_check_problems": (
            problems
        ),

        "billable_tokens": (
            record.billable_tokens
        ),

        "estimated_cost_usd": (
            record.estimated_cost_usd
        ),

        "model_name": (
            completion.model_name
        ),

        "provider": (
            completion.provider
        ),

        "latency_s": (
            latency_s
        ),

        "answer": (
            completion.text
        ),
    }


# ---------------------------------------------------------
# Main chat service
# ---------------------------------------------------------

def send_message(
    conversation_id: int,
    content: str,
) -> Dict:
    """
    Save user message, build multi-turn context, run Transit,
    save assistant response, and return routing metadata.
    """

    conversation = get_conversation(
        conversation_id
    )

    if conversation is None:
        raise ValueError(
            f"Conversation with id "
            f"{conversation_id} "
            "does not exist"
        )

    content = content.strip()

    if not content:
        raise ValueError(
            "Message cannot be empty"
        )

    # Build the prompt BEFORE saving the newest message,
    # otherwise the message appears twice.
    prompt = build_conversation_prompt(
        conversation_id=conversation_id,
        new_message=content,
    )

    user_message = add_message(
        conversation_id=conversation_id,
        role="user",
        content=content,
    )

    task = Task(
        task_id=(
            f"conversation-"
            f"{conversation_id}-"
            f"message-"
            f"{user_message.id}"
        ),

        prompt=prompt,

        metadata={
            "conversation_id": (
                conversation_id
            ),

            "mode": (
                conversation.mode
            ),

            "model": (
                conversation.model
            ),
        },
    )

    mode = (
        conversation.mode
        .strip()
        .lower()
    )

    # -----------------------------------------------------
    # AUTO / HYBRID
    # -----------------------------------------------------

    if mode == "hybrid":
        result = run_task(
            task=task,
            router=router,
            cheap=hybrid_cheap_backend,
            strong=hybrid_strong_backend,
            tracker=tracker,
        )

    # -----------------------------------------------------
    # FORCE REMOTE
    # -----------------------------------------------------

    elif mode == "remote":
        result = run_remote_only(
            task
        )

    # -----------------------------------------------------
    # FORCE LOCAL
    # -----------------------------------------------------

    elif mode == "local":
        result = run_local_only(
            task
        )

    else:
        raise ValueError(
            "Unsupported conversation mode: "
            f"{conversation.mode}"
        )

    answer = (
        result.get("answer")
        or ""
    )

    assistant_message = add_message(
        conversation_id=conversation_id,

        role="assistant",

        content=answer,

        route=(
            result.get("route")
        ),

        model_name=(
            result.get("model_name")
        ),

        latency_s=(
            result.get("latency_s")
        ),

        estimated_cost_usd=(
            result.get(
                "estimated_cost_usd"
            )
        ),
    )

    return {
        "conversation_id": (
            conversation_id
        ),

        "user_message": (
            user_message
        ),

        "assistant_message": (
            assistant_message
        ),

        "routing": {
            "mode": mode,

            "route": (
                result.get("route")
            ),

            "router": (
                result.get("router")
            ),

            "confidence": (
                result.get("confidence")
            ),

            "local_confidence": (
                result.get(
                    "local_confidence"
                )
            ),

            "escalated": (
                result.get(
                    "escalated"
                )
            ),

            "model_name": (
                result.get(
                    "model_name"
                )
            ),

            "provider": (
                result.get(
                    "provider"
                )
            ),

            "latency_s": (
                result.get(
                    "latency_s"
                )
            ),

            "billable_tokens": (
                result.get(
                    "billable_tokens"
                )
            ),

            "estimated_cost_usd": (
                result.get(
                    "estimated_cost_usd"
                )
            ),

            "signals": (
                result.get(
                    "signals"
                )
            ),

            "reason": (
                result.get(
                    "reason"
                )
            ),

            "post_check_problems": (
                result.get(
                    "post_check_problems"
                )
            ),
        },
    }