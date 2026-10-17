"""Persist intent before effects. An uncertain effect is never blindly replayed."""
import hashlib
import json
import re
from . import store, contracts

READ_ONLY = {"search", "search_chats", "read_url", "list_items", "read_item", "read_file", "list_files", "gmail_search", "gmail_read", "calendar_list", "read_skill", "read_attachment"}
REVIEW_REQUIRED = {"shell", "mcp", "browser"}


def policy(name):
    if name in READ_ONLY:
        return "read"
    if name == "save_item":
        return "local_write"
    if name in REVIEW_REQUIRED:
        return "reviewed_execution"
    return "external_mutation"


def publish(name, item_id, expected_updated, operation):
    item = store.get(item_id)
    if not expected_updated or item["updated"] != expected_updated:
        raise ValueError("Draft changed; reload and review its current contents before publishing")
    request = {"tool": name, "arguments": {"item": item}}
    pending = {**request, "action_id": identity(item_id, expected_updated, request)}
    result = execute_action("item:" + item_id, pending, lambda: operation(item))
    if not result["ok"]:
        raise RuntimeError(result["error"])
    return result["data"]


def submit(name, arguments, operation_id, operation):
    if not isinstance(operation_id, str) or not re.fullmatch(r"[a-f0-9-]{36}", operation_id):
        raise ValueError("A unique operation ID is required")
    pending = {"tool": name, "arguments": arguments, "action_id": hashlib.sha256(operation_id.encode()).hexdigest()}
    result = execute_action("operation:" + operation_id, pending, operation)
    if not result["ok"]:
        raise RuntimeError(result["error"])
    return result["data"]


def identity(run_id, sequence, decision):
    if decision.get("tool") == "save_item":
        sequence = "identical-local-write"
    return hashlib.sha256(json.dumps([run_id, sequence, decision], sort_keys=True).encode()).hexdigest()


def execute_action(run_id, pending, operation):
    contracts.arguments(pending["tool"], pending["arguments"])
    key = pending["action_id"]
    body = json.dumps({k: pending[k] for k in ("tool", "arguments")}, sort_keys=True)
    with store.db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        old = conn.execute("SELECT * FROM actions WHERE id=?", (key,)).fetchone()
        if old:
            if old["request"] != body:
                raise ValueError("Action identity does not match its approved arguments")
            if old["status"] == "completed":
                return json.loads(old["result"])
            raise RuntimeError("Action outcome is uncertain or failed; inspect and reconcile before retrying")
        if run_id.startswith("item:") and conn.execute("SELECT 1 FROM actions WHERE run_id=? AND status IN ('started','uncertain')", (run_id,)).fetchone():
            raise RuntimeError("An earlier publication of this item is uncertain; reconcile it before a new version is submitted")
        conn.execute("INSERT INTO actions VALUES(?,?,?,?,?,?,?)", (key, run_id, body, "started", "{}", store.now(), store.now()))
    try:
        output = operation()
        result = contracts.result_envelope(output)
        status = "completed" if result["ok"] else "failed"
    except Exception as exc:
        result = contracts.result_envelope(error=str(exc)[:1000])
        status = "failed" if pending["tool"] in READ_ONLY else "uncertain"
    with store.db() as conn:
        conn.execute("UPDATE actions SET status=?,result=?,updated=? WHERE id=?", (status, json.dumps(result), store.now(), key))
    if status == "uncertain":
        raise RuntimeError("Tool outcome is uncertain; inspect action history before any retry")
    return result


def reconcile(action_id, outcome, evidence):
    """Explicit user attestation, clearly distinct from a provider confirmation."""
    if outcome not in {"confirmed_completed", "confirmed_not_applied"} or not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 2000:
        raise ValueError("Provide the observed outcome and how you verified it")
    with store.db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM actions WHERE id=?", (action_id,)).fetchone()
        if not row or row["status"] != "uncertain":
            raise ValueError("Choose an uncertain action")
        result = json.loads(row["result"])
        result["reconciliation"] = {"outcome": outcome, "evidence": evidence, "source": "user_attestation", "at": store.now()}
        conn.execute("UPDATE actions SET status=?,result=?,updated=? WHERE id=? AND status='uncertain'", (outcome, json.dumps(result), store.now(), action_id))
        if row["run_id"].startswith("item:"):
            item_id = row["run_id"].removeprefix("item:")
            current = conn.execute("SELECT * FROM records WHERE id=?", (item_id,)).fetchone()
            if current:
                request = json.loads(row["request"])
                meta = json.loads(current["meta"])
                meta.update({"sending": False, "publishing": False, "outcome_source": "user_attestation"})
                if outcome == "confirmed_completed":
                    if request["tool"] == "mail_send":
                        meta["sent"] = True
                    elif request["tool"] == "calendar_publish":
                        original = request["arguments"]["item"]
                        meta["google_id"] = original["meta"].get("google_id") or hashlib.sha256((item_id + original["created"]).encode()).hexdigest()
                conn.execute("INSERT INTO revisions(record_id,content,title,meta,created) VALUES(?,?,?,?,?)", (item_id, current["content"], current["title"], current["meta"], store.now()))
                conn.execute("UPDATE records SET meta=?,updated=? WHERE id=?", (json.dumps(meta), store.now(), item_id))
    # The original receipt is retained. Reconciliation never executes an action.
    return {"action_id": action_id, "status": outcome, "source": "user_attestation"}
