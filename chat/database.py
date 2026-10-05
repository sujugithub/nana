import os
import sqlite3
from pathlib import Path


DB_PATH = Path(os.environ.get("NANA_CHAT_DB", Path(__file__).resolve().parents[1] / "data" / "nana-chat.db"))


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialise_database():
    conn = get_connection()

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL DEFAULT 'New Chat',
            mode TEXT NOT NULL DEFAULT 'hybrid',
            model TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,

            route TEXT,
            model_name TEXT,
            latency_s REAL,
            estimated_cost_usd REAL,

            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY(conversation_id)
                REFERENCES conversations(id)
                ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    conn.close()
