from dataclasses import asdict
import json
from typing import Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from chat.database import initialise_database

from chat.generation import (
    finish_generation,
    start_generation,
    stop_generation,
)

from chat.repository import (
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    list_messages,
    rename_conversation,
    search_conversations,
    update_conversation_settings,
)

from chat.service import (
    send_message,
    stream_message,
)


app = FastAPI(
    title="Transit API",
    description=(
        "Backend API for the Transit "
        "hybrid AI router"
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

@app.on_event("startup")
def startup():
    initialise_database()


# =========================================================
# REQUEST SCHEMAS
# =========================================================

class CreateConversationRequest(
    BaseModel
):
    title: str = Field(
        default="New Chat",
        min_length=1,
        max_length=200,
    )

    mode: str = "hybrid"

    model: Optional[str] = None


class RenameConversationRequest(
    BaseModel
):
    title: str = Field(
        min_length=1,
        max_length=200,
    )


class ConversationSettingsRequest(
    BaseModel
):
    mode: str

    model: Optional[str] = None


class SendMessageRequest(
    BaseModel
):
    content: str = Field(
        min_length=1,
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Transit API",
    }


# =========================================================
# CONVERSATIONS
# =========================================================

@app.post("/conversations")
def create_chat(
    request: CreateConversationRequest,
):
    try:
        mode = (
            request.mode
            .strip()
            .lower()
        )

        allowed_modes = {
            "hybrid",
            "remote",
            "local",
        }

        if mode not in allowed_modes:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Mode must be hybrid, "
                    "remote, or local"
                ),
            )

        conversation = (
            create_conversation(
                title=(
                    request.title.strip()
                ),
                mode=mode,
                model=request.model,
            )
        )

        return asdict(
            conversation
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not create "
                f"conversation: {error}"
            ),
        )


@app.get("/conversations")
def get_chats():
    conversations = (
        list_conversations()
    )

    return [
        asdict(conversation)
        for conversation
        in conversations
    ]


# Keep search before /{conversation_id}.
@app.get("/conversations/search")
def search_chats(
    q: str = Query(
        ...,
        min_length=1,
    ),
):
    conversations = (
        search_conversations(q)
    )

    return [
        asdict(conversation)
        for conversation
        in conversations
    ]


@app.get(
    "/conversations/{conversation_id}"
)
def get_chat(
    conversation_id: int,
):
    conversation = (
        get_conversation(
            conversation_id
        )
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found"
            ),
        )

    messages = list_messages(
        conversation_id
    )

    return {
        "conversation": (
            asdict(conversation)
        ),

        "messages": [
            asdict(message)
            for message in messages
        ],
    }


@app.patch(
    "/conversations/"
    "{conversation_id}/title"
)
def rename_chat(
    conversation_id: int,
    request: RenameConversationRequest,
):
    conversation = (
        rename_conversation(
            conversation_id=(
                conversation_id
            ),
            title=(
                request.title.strip()
            ),
        )
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found"
            ),
        )

    return asdict(
        conversation
    )


@app.patch(
    "/conversations/"
    "{conversation_id}/settings"
)
def update_chat_settings(
    conversation_id: int,
    request: ConversationSettingsRequest,
):
    allowed_modes = {
        "hybrid",
        "remote",
        "local",
    }

    mode = (
        request.mode
        .strip()
        .lower()
    )

    if mode not in allowed_modes:
        raise HTTPException(
            status_code=400,
            detail=(
                "Mode must be hybrid, "
                "remote, or local"
            ),
        )

    conversation = (
        update_conversation_settings(
            conversation_id=(
                conversation_id
            ),
            mode=mode,
            model=request.model,
        )
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found"
            ),
        )

    return asdict(
        conversation
    )


@app.delete(
    "/conversations/{conversation_id}"
)
def delete_chat(
    conversation_id: int,
):
    # Stop generation before deleting
    # the conversation if necessary.
    stop_generation(
        conversation_id
    )

    deleted = (
        delete_conversation(
            conversation_id
        )
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found"
            ),
        )

    return {
        "deleted": True,
        "conversation_id": (
            conversation_id
        ),
    }


# =========================================================
# NORMAL NON-STREAMING MESSAGES
# =========================================================

@app.post(
    "/conversations/"
    "{conversation_id}/messages"
)
def create_message(
    conversation_id: int,
    request: SendMessageRequest,
):
    try:
        return send_message(
            conversation_id=(
                conversation_id
            ),

            content=(
                request.content.strip()
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Transit failed to "
                "generate a response: "
                f"{error}"
            ),
        )


# =========================================================
# SSE STREAMING
# =========================================================

@app.post(
    "/conversations/"
    "{conversation_id}/stream"
)
def stream_chat_message(
    conversation_id: int,
    request: SendMessageRequest,
):
    """
    Stream the assistant response using
    Server-Sent Events (SSE).
    """

    conversation = (
        get_conversation(
            conversation_id
        )
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found"
            ),
        )

    content = (
        request.content.strip()
    )

    if not content:
        raise HTTPException(
            status_code=400,
            detail=(
                "Message cannot be empty"
            ),
        )

    stop_event = (
        start_generation(
            conversation_id
        )
    )

    def event_stream():
        try:
            # Tell frontend generation started.
            yield _sse(
                {
                    "type": "start",

                    "conversation_id": (
                        conversation_id
                    ),
                }
            )

            for event in stream_message(
                conversation_id=(
                    conversation_id
                ),

                content=content,

                stop_event=(
                    stop_event
                ),
            ):
                yield _sse(
                    event
                )

        except Exception as error:
            yield _sse(
                {
                    "type": "error",
                    "message": str(
                        error
                    ),
                }
            )

        finally:
            finish_generation(
                conversation_id
            )

    return StreamingResponse(
        event_stream(),

        media_type=(
            "text/event-stream"
        ),

        headers={
            # Prevent proxies/browser caches
            # from buffering SSE output.
            "Cache-Control": (
                "no-cache"
            ),

            "Connection": (
                "keep-alive"
            ),

            "X-Accel-Buffering": (
                "no"
            ),
        },
    )


# =========================================================
# STOP GENERATION
# =========================================================

@app.post(
    "/conversations/"
    "{conversation_id}/stop"
)
def stop_chat_generation(
    conversation_id: int,
):
    stopped = (
        stop_generation(
            conversation_id
        )
    )

    if not stopped:
        return {
            "stopped": False,

            "message": (
                "No active generation "
                "for this conversation"
            ),
        }

    return {
        "stopped": True,

        "conversation_id": (
            conversation_id
        ),
    }


# =========================================================
# SSE HELPER
# =========================================================

def _sse(
    payload: dict,
) -> str:
    """
    Convert a dictionary to one SSE event.

    Example:

        data: {"type":"token","content":"hello"}

    SSE events are separated by TWO newlines.
    """

    return (
        "data: "
        + json.dumps(
            payload,
            ensure_ascii=False,
        )
        + "\n\n"
    )