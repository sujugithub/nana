import hmac
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

# Match the demo server: load local defaults before importing config.py.
_REPO = Path(__file__).resolve().parent
try:
    for line in (_REPO / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
except FileNotFoundError:
    pass

from chat.database import initialise_database
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
from chat.service import MAX_MESSAGE_CHARS, conversation_config, send_message
from config import settings
from webui.service import KNOWN_FIREWORKS_MODELS, KNOWN_LOCAL_MODELS


def real_calls_allowed() -> bool:
    return os.environ.get("NANA_CHAT_ALLOW_REAL", "").strip().lower() in {"1", "true", "yes"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialise_database()
    yield


app = FastAPI(
    title="Nana Chat API",
    description="Persistent chat for the Nana hybrid router",
    version="1.0.0",
    lifespan=lifespan,
)


# React frontend will need this later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def local_or_token_access(request: Request, call_next):
    """Private by default, including conversation reads and billable sends."""
    from workspace import auth, store
    session = auth.session(request.headers)
    if auth.read_users() and session is None:
        return JSONResponse({"detail": "Sign in to Nana"}, status_code=401)
    context = store.USER.set(session["user_id"] if session else "local")
    if request.url.path.startswith("/conversations"):
        token = os.environ.get("NANA_API_TOKEN", "")
        supplied = request.headers.get("authorization", "")
        local = bool(request.client and request.client.host in {"127.0.0.1", "::1", "localhost", "testclient"})
        authorized = bool(token and hmac.compare_digest(supplied, f"Bearer {token}"))
        if not local and not authorized:
            store.USER.reset(context)
            return JSONResponse({"detail": "Local access only, or provide NANA_API_TOKEN"}, status_code=403)
    try:
        return await call_next(request)
    finally:
        store.USER.reset(context)


# -------------------------------------------------------------------
# Request schemas
# -------------------------------------------------------------------

class CreateConversationRequest(BaseModel):
    title: str = Field(default="New Chat", min_length=1, max_length=200)
    mode: Literal["hybrid", "remote", "local"] = "hybrid"
    model: Optional[str] = None


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationSettingsRequest(BaseModel):
    mode: Literal["hybrid", "remote", "local"]
    model: Optional[str] = None


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    mock: bool = True
    web: bool = False
    attachments: list[str] = Field(default_factory=list, max_length=4)


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Nana Chat API",
    }


@app.get("/chat")
def chat_page():
    return FileResponse(Path(__file__).resolve().parent / "webui" / "static" / "chat.html")


@app.get("/chat-response.js")
def chat_response_script():
    return FileResponse(_REPO / "webui" / "static" / "chat-response.js", media_type="text/javascript")


@app.get("/chat-response.css")
def chat_response_style():
    return FileResponse(_REPO / "webui" / "static" / "chat-response.css", media_type="text/css")


@app.get("/conversations/{conversation_id}/messages/{message_id}/artifacts/{index}")
def chat_artifact(conversation_id: int, message_id: int, index: int):
    from chat.artifacts import download
    try:
        body, mime, name = download(conversation_id, message_id, index)
        return Response(body, media_type=mime, headers={
            "Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox"})
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/chat/config")
def chat_config():
    from workspace.store import preferences
    from workspace.search import provider
    prefs = preferences()
    local_model = prefs.get("local_model", settings.local_model_name)
    if prefs.get("local_backend", settings.local_backend) == "ollama":
        from ollama_client import request
        try:
            local_models = [m["name"] for m in request("/api/tags", timeout=2).get("models", [])
                            if isinstance(m, dict) and isinstance(m.get("name"), str)]
        except Exception:
            local_models = []
        local_models = list(dict.fromkeys(local_models)) or [local_model]
        if local_model not in local_models:
            local_model = local_models[0]
    else:
        local_models = list(dict.fromkeys([local_model, *KNOWN_LOCAL_MODELS]))
    return {
        "real_allowed": real_calls_allowed(),
        "web": {"model": "qwen3.5:4b", "local_backend": prefs.get("local_backend", settings.local_backend),
                "search_provider": provider(prefs)},
        "known_local_models": local_models,
        "known_fireworks_models": KNOWN_FIREWORKS_MODELS,
        "default_mode": "hybrid",
        "default_models": {
            "hybrid": settings.strong_model_name,
            "remote": settings.strong_model_name,
            "local": local_model,
        },
    }


# -------------------------------------------------------------------
# Conversations
# -------------------------------------------------------------------

@app.post("/conversations")
def create_chat(request: CreateConversationRequest):
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be blank")
    try:
        conversation_config(request.mode, request.model, mock=True)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        conversation = create_conversation(
            title=title,
            mode=request.mode,
            model=request.model,
        )

        return asdict(conversation)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Could not create conversation",
        )


@app.get("/conversations")
def get_chats():
    conversations = list_conversations()

    return [
        asdict(conversation)
        for conversation in conversations
    ]


# IMPORTANT:
# Keep /search above /{conversation_id}
# so FastAPI does not try to interpret "search" as an integer ID.
@app.get("/conversations/search")
def search_chats(
    q: str = Query(..., min_length=1),
):
    conversations = search_conversations(q)

    return [
        asdict(conversation)
        for conversation in conversations
    ]


@app.get("/conversations/{conversation_id}")
def get_chat(conversation_id: int):
    conversation = get_conversation(conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    messages = list_messages(conversation_id)

    return {
        "conversation": asdict(conversation),
        "messages": [
            asdict(message)
            for message in messages
        ],
    }


@app.patch("/conversations/{conversation_id}/title")
def rename_chat(
    conversation_id: int,
    request: RenameConversationRequest,
):
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be blank")
    conversation = rename_conversation(
        conversation_id=conversation_id,
        title=title,
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    return asdict(conversation)


@app.patch("/conversations/{conversation_id}/settings")
def update_chat_settings(
    conversation_id: int,
    request: ConversationSettingsRequest,
):
    try:
        conversation_config(request.mode, request.model, mock=True)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    conversation = update_conversation_settings(
        conversation_id=conversation_id,
        mode=request.mode,
        model=request.model,
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    return asdict(conversation)


@app.delete("/conversations/{conversation_id}")
def delete_chat(conversation_id: int):
    deleted = delete_conversation(conversation_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    return {
        "deleted": True,
        "conversation_id": conversation_id,
    }


# -------------------------------------------------------------------
# Messages
# -------------------------------------------------------------------

@app.post("/conversations/{conversation_id}/messages")
def create_message(
    conversation_id: int,
    request: SendMessageRequest,
):
    try:
        return send_message(
            conversation_id=conversation_id,
            content=request.content.strip(),
            mock=request.mock,
            allow_real=real_calls_allowed(),
            web=request.web,
            attachments=request.attachments,
        )

    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Nana failed to save the chat turn",
        )


@app.post("/api/workspace/chat/attachments")
def upload_chat_attachment(payload: dict):
    from chat.attachments import upload
    try:
        return upload(payload)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/workspace/chat/capabilities")
def chat_capabilities(mode: Literal["local", "remote", "hybrid"] = "local", model: Optional[str] = None):
    from chat.attachments import capabilities
    from workspace.tools import TOOLS
    try:
        return {**capabilities(mode, model), "tools": TOOLS, "max_bytes": 10*1024*1024}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/conversations/{conversation_id}/tool-run")
def start_chat_tools(conversation_id: int, payload: dict):
    from chat.tool_turns import start
    try:
        return start(conversation_id, payload, real_calls_allowed())
    except (ValueError, LookupError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/conversations/{conversation_id}/tool-run")
def poll_chat_tools(conversation_id: int):
    from chat.tool_turns import poll
    try:
        return poll(conversation_id)
    except (ValueError, LookupError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/workspace/runs/{run_id}/review")
def review_chat_tools(run_id: str, payload: dict):
    from workspace.runs import review
    try:
        if not isinstance(payload.get("approve"), bool) or not isinstance(payload.get("action_id"), str):
            raise ValueError("Review requires approve and the exact action_id")
        return review(run_id, payload["approve"], real_calls_allowed(), payload["action_id"])
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/workspace/runs/{run_id}/cancel")
def cancel_chat_tools(run_id: str):
    from workspace.runs import cancel
    try:
        return cancel(run_id)
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
