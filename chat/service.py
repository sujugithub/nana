import time
from typing import Dict, Optional

from config import ROUTE_LOCAL, settings
from local_model import LocalModel
from main import (
    _gate_confidence,
    build_backends,
    build_router,
    run_task,
)
from remote_client import RemoteError, StrongRemoteClient
from schemas import Completion, Task
from token_tracker import TokenTracker

from .repository import (
    add_message,
    get_conversation,
    list_messages,
)


# =========================================================
# SHARED TRANSIT COMPONENTS
# =========================================================

router = build_router()

tier_mode, hybrid_cheap_backend, hybrid_strong_backend = (
    build_backends()
)

tracker = TokenTracker()


# Forced-mode backends are lazy loaded.
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


# =========================================================
# CONVERSATION CONTEXT
# =========================================================

def build_conversation_prompt(
    conversation_id: int,
    new_message: str,
) -> str:
    """
    Build the complete multi-turn prompt.

    Example:

        User: What is binary search?
        Assistant: ...
        User: Explain that again
        Assistant:
    """

    messages = list_messages(
        conversation_id
    )

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

    return "\n\n".join(
        parts
    )


def _prepare_task(
    conversation_id: int,
    content: str,
):
    """
    Validate the conversation, construct context,
    save the user message and create the routing Task.
    """

    conversation = get_conversation(
        conversation_id
    )

    if conversation is None:
        raise ValueError(
            f"Conversation with id "
            f"{conversation_id} does not exist"
        )

    content = content.strip()

    if not content:
        raise ValueError(
            "Message cannot be empty"
        )

    # Build prompt BEFORE saving current message,
    # otherwise it would appear twice.
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

    return (
        conversation,
        user_message,
        task,
    )


# =========================================================
# NORMAL NON-STREAMING MODES
# =========================================================

def run_remote_only(
    task: Task,
) -> Dict:
    started = time.time()

    backend = get_remote_backend()

    completion = backend.generate(
        task.prompt
    )

    latency_s = (
        time.time() - started
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
        router="forced_remote",
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
        "reason": "Remote mode selected by user",
        "post_check_problems": [],
        "billable_tokens": record.billable_tokens,
        "estimated_cost_usd": record.estimated_cost_usd,
        "model_name": completion.model_name,
        "provider": completion.provider,
        "latency_s": latency_s,
        "answer": completion.text,
    }


def run_local_only(
    task: Task,
) -> Dict:
    started = time.time()

    backend = get_local_backend()

    completion = backend.generate(
        task.prompt
    )

    latency_s = (
        time.time() - started
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
        local_confidence=completion.confidence,
        latency_s=latency_s,
        local_min_token_prob=completion.min_token_prob,
        local_low_token_frac=completion.low_token_frac,
        router="forced_local",
    )

    return {
        "task_id": task.task_id,
        "route": completion.source,
        "escalated": False,
        "confidence": 0.0,
        "router": "forced_local",
        "artifact_version": None,
        "local_confidence": completion.confidence,
        "signals": {},
        "reason": "Local mode selected by user",
        "post_check_problems": problems,
        "billable_tokens": record.billable_tokens,
        "estimated_cost_usd": record.estimated_cost_usd,
        "model_name": completion.model_name,
        "provider": completion.provider,
        "latency_s": latency_s,
        "answer": completion.text,
    }


# =========================================================
# NORMAL NON-STREAMING CHAT
# =========================================================

def send_message(
    conversation_id: int,
    content: str,
) -> Dict:

    (
        conversation,
        user_message,
        task,
    ) = _prepare_task(
        conversation_id,
        content,
    )

    mode = (
        conversation.mode
        .strip()
        .lower()
    )

    # Hybrid routing.
    if mode == "hybrid":
        result = run_task(
            task=task,
            router=router,
            cheap=hybrid_cheap_backend,
            strong=hybrid_strong_backend,
            tracker=tracker,
        )

    # Forced remote.
    elif mode == "remote":
        result = run_remote_only(
            task
        )

    # Forced local.
    elif mode == "local":
        result = run_local_only(
            task
        )

    else:
        raise ValueError(
            f"Unsupported conversation mode: "
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
        route=result.get("route"),
        model_name=result.get("model_name"),
        latency_s=result.get("latency_s"),
        estimated_cost_usd=result.get(
            "estimated_cost_usd"
        ),
    )

    return {
        "conversation_id": conversation_id,

        "user_message": user_message,

        "assistant_message": assistant_message,

        "routing": {
            "mode": mode,
            "route": result.get("route"),
            "router": result.get("router"),
            "confidence": result.get("confidence"),
            "local_confidence": result.get(
                "local_confidence"
            ),
            "escalated": result.get("escalated"),
            "model_name": result.get("model_name"),
            "provider": result.get("provider"),
            "latency_s": result.get("latency_s"),
            "billable_tokens": result.get(
                "billable_tokens"
            ),
            "estimated_cost_usd": result.get(
                "estimated_cost_usd"
            ),
            "signals": result.get("signals"),
            "reason": result.get("reason"),
            "post_check_problems": result.get(
                "post_check_problems"
            ),
        },
    }


# =========================================================
# STREAMING HELPER
# =========================================================

def _stream_backend(
    backend,
    prompt: str,
    stop_event,
    stage: str,
):
    """
    Stream one model backend.

    The backend itself yields:

        (text_chunk, None)

    followed by:

        (None, Completion)

    This helper converts the chunks into events for api.py.

    The final Completion is returned through `yield from`.
    """

    final_completion = None

    for chunk, completion in backend.stream_generate(
        prompt,
        stop_event=stop_event,
    ):

        if chunk is not None:
            yield {
                "type": "token",

                "content": chunk,

                # Useful to the frontend if an escalation occurs.
                "stage": stage,
            }

        if completion is not None:
            final_completion = completion

    return final_completion


# =========================================================
# STREAMING CHAT
# =========================================================

def stream_message(
    conversation_id: int,
    content: str,
    stop_event,
):
    """
    Stream one Transit chat response.

    Events yielded to api.py include:

        token
        escalation
        fallback
        routing
        stopped
        done

    The final assistant response is persisted in SQLite.
    """

    started = time.time()

    (
        conversation,
        user_message,
        task,
    ) = _prepare_task(
        conversation_id,
        content,
    )

    mode = (
        conversation.mode
        .strip()
        .lower()
    )

    cheap_completion = None
    strong_completion = None

    escalated = False

    problems = []

    decision = None

    router_name = ""
    router_confidence = 0.0
    router_signals = {}
    router_reason = ""
    artifact_version = None

    # =====================================================
    # FORCED REMOTE
    # =====================================================

    if mode == "remote":

        backend = get_remote_backend()

        strong_completion = yield from _stream_backend(
            backend,
            task.prompt,
            stop_event,
            stage="remote",
        )

        router_name = "forced_remote"
        router_reason = (
            "Remote mode selected by user"
        )

    # =====================================================
    # FORCED LOCAL
    # =====================================================

    elif mode == "local":

        backend = get_local_backend()

        cheap_completion = yield from _stream_backend(
            backend,
            task.prompt,
            stop_event,
            stage="local",
        )

        router_name = "forced_local"
        router_reason = (
            "Local mode selected by user"
        )

        if cheap_completion is not None:

            ok, problems = router.post_check(
                task.prompt,
                cheap_completion.text,
            )

            if ok:
                problems = []

    # =====================================================
    # HYBRID
    # =====================================================

    elif mode == "hybrid":

        decision = router.decide(
            task
        )

        router_name = (
            decision.router_kind
        )

        router_confidence = (
            decision.confidence
        )

        router_signals = (
            decision.signals
        )

        router_reason = (
            decision.reason
        )

        artifact_version = (
            decision.artifact_version
        )

        # -------------------------------------------------
        # Router selected cheap/local tier
        # -------------------------------------------------

        if decision.target == ROUTE_LOCAL:

            cheap_completion = yield from _stream_backend(
                hybrid_cheap_backend,
                task.prompt,
                stop_event,
                stage="draft",
            )

            if cheap_completion is None:
                raise RuntimeError(
                    "Cheap backend finished without "
                    "producing a completion"
                )

            ok, problems = router.post_check(
                task.prompt,
                cheap_completion.text,
            )

            gate_conf = _gate_confidence(
                cheap_completion
            )

            low_confidence = (
                gate_conf is not None
                and gate_conf
                < settings.logprob_confidence_threshold
            )

            if low_confidence:
                problems.append(
                    "low_confidence:"
                    f"{settings.local_conf_stat}:"
                    f"{gate_conf:.2f}"
                )

            # Do NOT escalate if the user pressed Stop.
            should_escalate = (
                not stop_event.is_set()
                and settings.enable_escalation
                and (
                    not ok
                    or low_confidence
                )
            )

            if should_escalate:

                escalated = True

                # The frontend should clear the streamed draft
                # when it receives this event.
                yield {
                    "type": "escalation",

                    "clear_previous": True,

                    "reason": problems,
                }

                try:
                    strong_completion = (
                        yield from _stream_backend(
                            hybrid_strong_backend,
                            task.prompt,
                            stop_event,
                            stage="final",
                        )
                    )

                except RemoteError as error:

                    # Existing Transit policy:
                    # cheap answer beats no answer.
                    problems.append(
                        "escalation_failed: "
                        f"{error}"
                    )

        # -------------------------------------------------
        # Router selected strong tier
        # -------------------------------------------------

        else:

            try:
                strong_completion = (
                    yield from _stream_backend(
                        hybrid_strong_backend,
                        task.prompt,
                        stop_event,
                        stage="remote",
                    )
                )

            except RemoteError as error:

                problems.append(
                    "strong_failed_cheap_fallback: "
                    f"{error}"
                )

                # Tell the frontend that the previous attempt
                # should be replaced by fallback output.
                yield {
                    "type": "fallback",

                    "clear_previous": True,

                    "reason": str(error),
                }

                cheap_completion = (
                    yield from _stream_backend(
                        hybrid_cheap_backend,
                        task.prompt,
                        stop_event,
                        stage="fallback",
                    )
                )

    else:
        raise ValueError(
            f"Unsupported conversation mode: "
            f"{conversation.mode}"
        )

    # =====================================================
    # DETERMINE FINAL ANSWER
    # =====================================================

    final = (
        strong_completion
        or cheap_completion
    )

    if final is None:
        raise RuntimeError(
            "No model produced a completion"
        )

    latency_s = (
        time.time()
        - started
    )

    # =====================================================
    # TOKEN / COST TRACKING
    # =====================================================

    record = tracker.record(
        task_id=task.task_id,

        route=final.source,

        escalated=escalated,

        local=cheap_completion,

        remote=strong_completion,

        confidence=router_confidence,

        threshold=(
            router.threshold
            if mode == "hybrid"
            else 0.0
        ),

        signals=router_signals,

        problems=problems,

        local_confidence=(
            cheap_completion.confidence
            if cheap_completion
            else None
        ),

        latency_s=latency_s,

        local_min_token_prob=(
            cheap_completion.min_token_prob
            if cheap_completion
            else None
        ),

        local_low_token_frac=(
            cheap_completion.low_token_frac
            if cheap_completion
            else None
        ),

        router=router_name,

        artifact_version=(
            artifact_version
        ),

        p_local=(
            router_confidence
            if router_name == "learned"
            else None
        ),
    )

    # =====================================================
    # SAVE FINAL ASSISTANT MESSAGE
    # =====================================================

    assistant_message = None

    # Save partial output too if Stop was pressed,
    # as long as something was actually generated.
    if final.text:

        assistant_message = add_message(
            conversation_id=conversation_id,

            role="assistant",

            content=final.text,

            route=final.source,

            model_name=final.model_name,

            latency_s=latency_s,

            estimated_cost_usd=(
                record.estimated_cost_usd
            ),
        )

    # =====================================================
    # SEND ROUTING DETAILS
    # =====================================================

    yield {
        "type": "routing",

        "routing": {
            "mode": mode,

            "route": final.source,

            "router": router_name,

            "confidence": (
                router_confidence
            ),

            "local_confidence": (
                cheap_completion.confidence
                if cheap_completion
                else None
            ),

            "escalated": escalated,

            "model_name": (
                final.model_name
            ),

            "provider": (
                final.provider
            ),

            "latency_s": (
                latency_s
            ),

            "billable_tokens": (
                record.billable_tokens
            ),

            "estimated_cost_usd": (
                record.estimated_cost_usd
            ),

            "signals": (
                router_signals
            ),

            "reason": (
                router_reason
            ),

            "post_check_problems": (
                problems
            ),
        },
    }

    # =====================================================
    # FINAL EVENT
    # =====================================================

    if stop_event.is_set():

        yield {
            "type": "stopped",

            "conversation_id": (
                conversation_id
            ),

            "assistant_message_id": (
                assistant_message.id
                if assistant_message
                else None
            ),
        }

    else:

        yield {
            "type": "done",

            "conversation_id": (
                conversation_id
            ),

            "assistant_message_id": (
                assistant_message.id
                if assistant_message
                else None
            ),
        }