from typing import List, Optional

from .database import get_connection
from .models import Conversation, Message


def create_conversation(
    title: str = "New Chat",
    mode: str = "hybrid",
    model: Optional[str] = None,
) -> Conversation:
    conn = get_connection()

    try:
        cursor = conn.execute(
            """
            INSERT INTO conversations (title, mode, model)
            VALUES (?, ?, ?)
            """,
            (title, mode, model),
        )

        conversation_id = cursor.lastrowid
        conn.commit()

        row = conn.execute(
            """
            SELECT *
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

        return Conversation(**dict(row))

    finally:
        conn.close()


def list_conversations() -> List[Conversation]:
    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT *
            FROM conversations
            ORDER BY updated_at DESC, id DESC
            """
        ).fetchall()

        return [Conversation(**dict(row)) for row in rows]

    finally:
        conn.close()


def get_conversation(conversation_id: int) -> Optional[Conversation]:
    conn = get_connection()

    try:
        row = conn.execute(
            """
            SELECT *
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if row is None:
            return None

        return Conversation(**dict(row))

    finally:
        conn.close()


def rename_conversation(
    conversation_id: int,
    title: str,
) -> Optional[Conversation]:
    conn = get_connection()

    try:
        cursor = conn.execute(
            """
            UPDATE conversations
            SET title = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (title, conversation_id),
        )

        if cursor.rowcount == 0:
            return None

        conn.commit()

        row = conn.execute(
            """
            SELECT *
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

        return Conversation(**dict(row))

    finally:
        conn.close()


def update_conversation_settings(
    conversation_id: int,
    mode: str,
    model: Optional[str] = None,
) -> Optional[Conversation]:
    conn = get_connection()

    try:
        cursor = conn.execute(
            """
            UPDATE conversations
            SET mode = ?,
                model = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (mode, model, conversation_id),
        )

        if cursor.rowcount == 0:
            return None

        conn.commit()

        row = conn.execute(
            """
            SELECT *
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

        return Conversation(**dict(row))

    finally:
        conn.close()


def delete_conversation(conversation_id: int) -> bool:
    conn = get_connection()

    try:
        cursor = conn.execute(
            """
            DELETE FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        )

        conn.commit()

        return cursor.rowcount > 0

    finally:
        conn.close()


def add_message(
    conversation_id: int,
    role: str,
    content: str,
    route: Optional[str] = None,
    model_name: Optional[str] = None,
    latency_s: Optional[float] = None,
    estimated_cost_usd: Optional[float] = None,
) -> Message:
    if role not in ("user", "assistant"):
        raise ValueError("role must be either 'user' or 'assistant'")

    if not content.strip():
        raise ValueError("message content cannot be empty")

    conn = get_connection()

    try:
        conversation = conn.execute(
            """
            SELECT id
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if conversation is None:
            raise ValueError(
                f"Conversation with id {conversation_id} does not exist"
            )

        cursor = conn.execute(
            """
            INSERT INTO messages (
                conversation_id,
                role,
                content,
                route,
                model_name,
                latency_s,
                estimated_cost_usd
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                route,
                model_name,
                latency_s,
                estimated_cost_usd,
            ),
        )

        message_id = cursor.lastrowid

        conn.execute(
            """
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (conversation_id,),
        )

        conn.commit()

        row = conn.execute(
            """
            SELECT *
            FROM messages
            WHERE id = ?
            """,
            (message_id,),
        ).fetchone()

        return Message(**dict(row))

    finally:
        conn.close()


def list_messages(conversation_id: int) -> List[Message]:
    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT *
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

        return [Message(**dict(row)) for row in rows]

    finally:
        conn.close()


def search_conversations(query: str) -> List[Conversation]:
    conn = get_connection()

    try:
        query = query.strip()

        if not query:
            return list_conversations()

        pattern = f"%{query}%"

        rows = conn.execute(
            """
            SELECT DISTINCT conversations.*
            FROM conversations
            LEFT JOIN messages
                ON conversations.id = messages.conversation_id
            WHERE conversations.title LIKE ?
               OR messages.content LIKE ?
            ORDER BY conversations.updated_at DESC,
                     conversations.id DESC
            """,
            (pattern, pattern),
        ).fetchall()

        return [Conversation(**dict(row)) for row in rows]

    finally:
        conn.close()