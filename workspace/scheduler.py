"""Durable local reminders and scheduled routed agents; no outbound messages."""
from datetime import datetime, timedelta, timezone
import json
import threading
from . import store, runs


def tick(allow_real):
    with store.db() as conn:
        candidates = conn.execute("SELECT * FROM jobs WHERE status='pending' AND due<=?", (store.now(),)).fetchall()
        due = []
        for job in candidates:
            claimed = conn.execute("UPDATE jobs SET status='running',updated=? WHERE id=? AND status='pending'", (store.now(), job["id"]))
            if claimed.rowcount:
                due.append(job)
    for job in due:
        payload = json.loads(job["payload"])
        try:
            if payload.get("type", "reminder") == "agent":
                # Scheduled jobs only get read-only tools. User reviews writes
                # interactively, rather than granting a blanket send policy.
                run = runs.start("agent", {"prompt": payload["prompt"], "mode": payload.get("mode", "remote"),
                                          "tools": ["list_items", "read_item", "calendar_list"]}, allow_real)
                run_id = run["id"]
            else:
                store.save({"kind": "notification", "title": job["title"], "content": str(payload.get("content", "")), "meta": {"job_id": job["id"]}})
                run_id = None
            repeat = int(payload.get("repeat_minutes", 0))
            with store.db() as conn:
                if repeat > 0:
                    next_due = (datetime.now(timezone.utc) + timedelta(minutes=max(1, repeat))).isoformat()
                    conn.execute("UPDATE jobs SET status='pending',due=?,run_id=?,updated=? WHERE id=?", (next_due, run_id, store.now(), job["id"]))
                else:
                    conn.execute("UPDATE jobs SET status='completed',run_id=?,updated=? WHERE id=?", (run_id, store.now(), job["id"]))
        except Exception:
            with store.db() as conn:
                conn.execute("UPDATE jobs SET status='failed',updated=? WHERE id=?", (store.now(), job["id"]))


class Scheduler:
    def __init__(self, allow_real):
        self.stop = threading.Event()
        self.allow_real = allow_real
        # Recover only pre-existing data, before any new requests are accepted.
        for home in list(store.ROOT.iterdir()) if store.ROOT.exists() else []:
            if home.is_dir() and (home / "workspace.db").exists():
                token = store.USER.set(home.name)
                try:
                    runs.recover()
                    with store.db() as conn:
                        conn.execute("UPDATE jobs SET status='interrupted',updated=? WHERE status='running'", (store.now(),))
                finally:
                    store.USER.reset(token)
        self.thread = threading.Thread(target=self.loop, daemon=True, name="nana-scheduler")
        self.thread.start()

    def loop(self):
        while not self.stop.is_set():
            for home in list(store.ROOT.iterdir()) if store.ROOT.exists() else []:
                if not home.is_dir() or not (home / "workspace.db").exists():
                    continue
                token = store.USER.set(home.name)
                try:
                    tick(self.allow_real)
                except Exception:
                    pass  # Failed job state is visible; loop stays alive.
                finally:
                    store.USER.reset(token)
            self.stop.wait(10)
