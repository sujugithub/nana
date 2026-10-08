"""Versioned persistent workspace records, revisions, jobs and preferences."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import uuid

ROOT = Path(os.environ.get("NANA_WORKSPACE_DIR", Path(__file__).resolve().parents[1] / "data" / "workspace"))
USER = ContextVar("workspace_user", default="local")
KINDS = {"note", "task", "document", "memory", "event", "preset", "draft", "asset", "notification"}


def now():
    return datetime.now(timezone.utc).isoformat()


def home():
    path = ROOT / USER.get()
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


@contextmanager
def db():
    path = home() / "workspace.db"
    conn = sqlite3.connect(path, timeout=20)
    path.chmod(0o600)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY);
        INSERT OR IGNORE INTO schema_version VALUES(1);
        CREATE TABLE IF NOT EXISTS records(
          id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL,
          content TEXT NOT NULL, meta TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS revisions(
          id INTEGER PRIMARY KEY, record_id TEXT NOT NULL, content TEXT NOT NULL,
          title TEXT NOT NULL, meta TEXT NOT NULL, created TEXT NOT NULL,
          FOREIGN KEY(record_id) REFERENCES records(id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS preferences(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(
          id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
          payload TEXT NOT NULL, result TEXT NOT NULL, steps TEXT NOT NULL,
          created TEXT NOT NULL, updated TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(
          id TEXT PRIMARY KEY, title TEXT NOT NULL, due TEXT NOT NULL,
          payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', run_id TEXT,
          updated TEXT NOT NULL);
    """)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def record(row):
    if row is None:
        raise LookupError("Item not found")
    data = dict(row)
    data["meta"] = json.loads(data["meta"])
    return data


def get(item_id):
    with db() as conn:
        return record(conn.execute("SELECT * FROM records WHERE id=?", (item_id,)).fetchone())


def items(kind=None, query=""):
    if kind is not None and kind not in KINDS:
        raise ValueError("Unknown item type")
    with db() as conn:
        rows = conn.execute("SELECT * FROM records WHERE (? IS NULL OR kind=?) "
                            "AND (title LIKE ? OR content LIKE ?) ORDER BY updated DESC LIMIT 500",
                            (kind, kind, f"%{query}%", f"%{query}%")).fetchall()
        return [record(row) for row in rows]


def save(payload, item_id=None):
    if not isinstance(payload, dict):
        raise ValueError("Item must be an object")
    old = get(item_id) if item_id else None
    kind = old["kind"] if old else payload.get("kind", "note")
    title = payload.get("title", old["title"] if old else "Untitled")
    content = payload.get("content", old["content"] if old else "")
    meta = payload.get("meta", old["meta"] if old else {})
    if kind not in KINDS or not isinstance(title, str) or not title.strip() or len(title) > 300:
        raise ValueError("Provide a valid kind and a title of 1–300 characters")
    if not isinstance(content, str) or len(content) > 500_000 or not isinstance(meta, dict):
        raise ValueError("Invalid content or metadata")
    if len(json.dumps(meta)) > 50_000:
        raise ValueError("Metadata is too large")
    item_id = item_id or uuid.uuid4().hex
    timestamp = now()
    with db() as conn:
        if old:
            expected = payload.get("expected_updated")
            current = conn.execute("SELECT updated FROM records WHERE id=?", (item_id,)).fetchone()
            if current is None:
                raise LookupError("Item not found")
            if current[0] != old["updated"] or (expected is not None and current[0] != expected):
                raise ValueError("Item changed; reload before saving")
            conn.execute("INSERT INTO revisions(record_id,content,title,meta,created) VALUES(?,?,?,?,?)",
                         (item_id, old["content"], old["title"], json.dumps(old["meta"]), timestamp))
            conn.execute("UPDATE records SET title=?,content=?,meta=?,updated=? WHERE id=?",
                         (title.strip(), content, json.dumps(meta), timestamp, item_id))
        else:
            conn.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?)",
                         (item_id, kind, title.strip(), content, json.dumps(meta), timestamp, timestamp))
    return get(item_id)


def delete(item_id):
    with db() as conn:
        if conn.execute("DELETE FROM records WHERE id=?", (item_id,)).rowcount == 0:
            raise LookupError("Item not found")


def revisions(item_id):
    get(item_id)
    with db() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM revisions WHERE record_id=? ORDER BY id DESC", (item_id,))]


def preferences(payload=None):
    with db() as conn:
        if payload is not None:
            if not isinstance(payload, dict) or len(json.dumps(payload)) > 50_000:
                raise ValueError("Invalid preferences")
            for key, value in payload.items():
                if key not in {"theme", "agent_mode", "agent_model", "system_prompt", "searxng_url", "mcp_servers", "local_model", "local_backend"}:
                    raise ValueError("Unknown preference")
                conn.execute("INSERT OR REPLACE INTO preferences VALUES(?,?)", (key, json.dumps(value)))
        return {row["key"]: json.loads(row["value"]) for row in conn.execute("SELECT * FROM preferences")}


def run_get(run_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise LookupError("Run not found")
        data = dict(row)
        for name in ("payload", "result", "steps"):
            data[name] = json.loads(data[name])
        return data


def run_save(run_id, **updates):
    with db() as conn:
        for key, value in updates.items():
            if key not in {"status", "result", "steps"}:
                raise ValueError("Invalid run update")
            if key != "status":
                value = json.dumps(value)
            conn.execute(f"UPDATE runs SET {key}=?,updated=? WHERE id=?", (value, now(), run_id))


def run_create(kind, payload):
    run_id = uuid.uuid4().hex
    with db() as conn:
        conn.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?)",
                     (run_id, kind, "queued", json.dumps(payload), "{}", "[]", now(), now()))
    return run_id


def run_list():
    with db() as conn:
        ids = [row[0] for row in conn.execute("SELECT id FROM runs ORDER BY created DESC LIMIT 100")]
    return [run_get(item) for item in ids]


def job_save(payload):
    due = datetime.fromisoformat(str(payload.get("due", "")).replace("Z", "+00:00"))
    if due.tzinfo is None:
        raise ValueError("Schedule time must include a timezone")
    title = str(payload.get("title", "Reminder")).strip()[:300]
    if payload.get("type", "reminder") not in {"reminder", "agent"}:
        raise ValueError("Job type must be reminder or agent")
    if payload.get("type") == "agent" and not str(payload.get("prompt", "")).strip():
        raise ValueError("Scheduled agent needs a prompt")
    if not isinstance(payload.get("repeat_minutes", 0), int) or not 0 <= payload.get("repeat_minutes", 0) <= 525600:
        raise ValueError("Repeat minutes must be an integer between 0 and 525600")
    job_id = uuid.uuid4().hex
    with db() as conn:
        conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", (job_id, title,
                     due.astimezone(timezone.utc).isoformat(), json.dumps(payload), "pending", None, now()))
    return job_id


def jobs():
    with db() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM jobs ORDER BY due")]
