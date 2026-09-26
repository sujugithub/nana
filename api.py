from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
from chat.service import send_message


app = FastAPI(
    title="Transit API",
    description="Backend API for the Transit hybrid AI router",
    version="1.0.0",
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


@app.on_event("startup")
def startup():
    initialise_database()


# -------------------------------------------------------------------
# Request schemas
# -------------------------------------------------------------------

class CreateConversationRequest(BaseModel):
    title: str = Field(default="New Chat", min_length=1, max_length=200)
    mode: str = "hybrid"
    model: Optional[str] = None


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationSettingsRequest(BaseModel):
    mode: str
    model: Optional[str] = None


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1)


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Transit API",
    }


# -------------------------------------------------------------------
# Conversations
# -------------------------------------------------------------------

@app.post("/conversations")
def create_chat(request: CreateConversationRequest):
    try:
        conversation = create_conversation(
            title=request.title.strip(),
            mode=request.mode,
            model=request.model,
        )

        return asdict(conversation)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not create conversation: {error}",
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
    conversation = rename_conversation(
        conversation_id=conversation_id,
        title=request.title.strip(),
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
    allowed_modes = {
        "hybrid",
        "remote",
        "local",
    }

    mode = request.mode.strip().lower()

    if mode not in allowed_modes:
        raise HTTPException(
            status_code=400,
            detail="Mode must be hybrid, remote, or local",
        )

    conversation = update_conversation_settings(
        conversation_id=conversation_id,
        mode=mode,
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
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Transit failed to generate a response: {error}",
        )