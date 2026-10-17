"""One durable occurrence per due time, with no overlap and explicit recovery."""
from datetime import datetime, timedelta, timezone
import json
import threading
import uuid
from zoneinfo import ZoneInfo
from . import store, runs

TERMINAL = {"completed", "failed", "cancelled", "interrupted", "completed_with_errors"}


def next_due(due, payload, now):
    """Coalesce missed intervals; daily wall time uses fold 0, gaps roll forward."""
    if payload.get("recurrence") == "daily":
        zone = ZoneInfo(payload.get("timezone", "UTC"))
        anchor = datetime.fromisoformat(str(payload.get("due", due))).astimezone(zone)
        local = now.astimezone(zone)
        candidate = datetime.combine(local.date(), anchor.timetz().replace(tzinfo=None), zone).replace(fold=0)
        # UTC round trip maps a nonexistent spring-forward time to after the gap.
        candidate = candidate.astimezone(timezone.utc).astimezone(zone)
        if candidate <= local:
            candidate = datetime.combine(local.date()+timedelta(days=1), anchor.timetz().replace(tzinfo=None), zone).replace(fold=0)
        return candidate.astimezone(timezone.utc).isoformat()
    minutes = payload.get("repeat_minutes", 0)
    if not minutes:
        return None
    anchor = datetime.fromisoformat(due)
    interval = timedelta(minutes=minutes)
    count = max(1, (now-anchor)//interval + 1)
    return (anchor + count*interval).isoformat()


def reconcile():
    with store.db() as conn:
        rows = conn.execute("SELECT o.*,r.status AS run_status FROM occurrences o LEFT JOIN runs r ON r.id=o.run_id WHERE o.status IN ('queued','running')").fetchall()
        for row in rows:
            status = row["run_status"]
            if status is None:
                status = "interrupted"
            if status in TERMINAL:
                conn.execute("UPDATE occurrences SET status=?,updated=? WHERE id=?", (status, store.now(), row["id"]))
                job = conn.execute("SELECT * FROM jobs WHERE id=?", (row["job_id"],)).fetchone()
                due = next_due(job["due"], json.loads(job["payload"]), datetime.now(timezone.utc)) if status == "completed" else None
                conn.execute("UPDATE jobs SET status=?,due=?,updated=? WHERE id=?", ("pending" if due else status, due or job["due"], store.now(), job["id"]))
            elif status:
                conn.execute("UPDATE occurrences SET status=?,updated=? WHERE id=?", ("running" if status == "running" else "queued", store.now(), row["id"]))


def tick(allow_real):
    reconcile()
    dispatch = []
    with store.db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        candidates = conn.execute("SELECT * FROM jobs WHERE status='pending' AND due<=?", (store.now(),)).fetchall()
        for job in candidates:
            payload = json.loads(job["payload"])
            occurrence = uuid.uuid4().hex
            # Run creation, occurrence claim and linking are one transaction.
            run_id = uuid.uuid4().hex if payload.get("type") == "agent" else None
            added = conn.execute("INSERT OR IGNORE INTO occurrences VALUES(?,?,?,?,?,?,?)", (occurrence, job["id"], job["due"], "queued", run_id, None, store.now()))
            if not added.rowcount:
                continue
            if run_id:
                run_payload = {"prompt": payload["prompt"], "mode": payload.get("mode", "local"),
                               "model": payload.get("model"), "mock": payload.get("mock", False),
                               "tools": ["list_items", "read_item", "calendar_list"], "real_allowed": allow_real,
                               "occurrence_id": occurrence}
                conn.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?)", (run_id, "agent", "queued", json.dumps(run_payload), "{}", "[]", store.now(), store.now()))
                conn.execute("UPDATE jobs SET status='running',run_id=?,updated=? WHERE id=?", (run_id, store.now(), job["id"]))
                dispatch.append(run_id)
            else:
                # Local reminder and completion share a transaction; restart
                # cannot duplicate a notification between these writes.
                meta = json.dumps({"job_id": job["id"], "occurrence_id": occurrence})
                conn.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?)", (uuid.uuid4().hex, "notification", job["title"], str(payload.get("content", "")), meta, store.now(), store.now()))
                conn.execute("UPDATE occurrences SET status='completed',updated=? WHERE id=?", (store.now(), occurrence))
                due = next_due(job["due"], payload, datetime.now(timezone.utc))
                conn.execute("UPDATE jobs SET status=?,due=?,updated=? WHERE id=?", ("pending" if due else "completed", due or job["due"], store.now(), job["id"]))
    for run_id in dispatch:
        try:
            runs.resume(run_id, allow_real)
        except Exception as exc:
            store.run_save(run_id, status="failed", result={"error": str(exc)[:500]})


def recover(allow_real):
    """Only safely queued scheduled reads restart automatically, once on startup."""
    with store.db() as conn:
        queued = [row[0] for row in conn.execute("SELECT r.id FROM runs r JOIN occurrences o ON o.run_id=r.id WHERE r.status='queued' AND o.status='queued'")]
    runs.recover()
    # In-flight reads may have billed a remote provider. Leave interrupted,
    # requiring a conscious user resume; never silently repeat that generation.
    for run_id in queued:
        store.run_save(run_id, status="queued")
        runs.resume(run_id, allow_real)
    reconcile()
    with store.db() as conn:
        conn.execute("UPDATE jobs SET status='interrupted',updated=? WHERE status='running' AND id NOT IN (SELECT job_id FROM occurrences)", (store.now(),))


class Scheduler:
    def __init__(self, allow_real):
        import fcntl
        store.ROOT.mkdir(parents=True, exist_ok=True)
        self.lease = (store.ROOT/".scheduler.lock").open("a")
        try:
            fcntl.flock(self.lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lease.close()
            raise RuntimeError("Another Nana scheduler owns this workspace; stop it before starting another") from None
        self.stop = threading.Event()
        self.allow_real = allow_real
        for home in list(store.ROOT.iterdir()) if store.ROOT.exists() else []:
            if home.is_dir() and (home / "workspace.db").exists():
                token = store.USER.set(home.name)
                try:
                    recover(allow_real)
                finally:
                    store.USER.reset(token)
        self.thread = threading.Thread(target=self.loop, daemon=True, name="nana-scheduler")
        self.thread.start()

    def loop(self):
        try:
            self._loop()
        finally:
            self.lease.close()

    def _loop(self):
        while not self.stop.is_set():
            for home in list(store.ROOT.iterdir()) if store.ROOT.exists() else []:
                if not home.is_dir() or not (home / "workspace.db").exists():
                    continue
                token = store.USER.set(home.name)
                try:
                    tick(self.allow_real)
                except Exception:
                    # Failed claim transactions roll back. Durable occurrences
                    # remain inspectable, and the next tick can retry safely.
                    pass
                finally:
                    store.USER.reset(token)
            self.stop.wait(10)
