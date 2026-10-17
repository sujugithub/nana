"""Background comparison and routed agent runs with durable progress."""
from concurrent.futures import ThreadPoolExecutor
import json
import random
import re
import threading
import time

from chat.service import conversation_config
from webui.service import execute
from . import store, tools, contracts, context
from generation_stream import Stream, CURRENT, Cancelled
from generation_context import TOOLS, OUTPUT_LIMIT, LOCAL_CONTEXT, MESSAGES

_STREAMS = {}

POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nana-workspace")
_ACTIVE = set()
_LOCK = threading.Lock()


def generate(payload, prompt, allow_real):
    mode = payload.get("mode", store.preferences().get("agent_mode", "local"))
    model = payload.get("model") or None
    mock = payload.get("mock", False) is True or not allow_real
    cfg = conversation_config(mode, model, mock)
    native = (mode == "remote" or mode == "local" and cfg.local_backend == "ollama") and not mock
    catalog = contracts.catalog(payload.get("_selected", []), tools.TOOLS) if native and "_selected" in payload else None
    t = TOOLS.set(catalog)
    limit = OUTPUT_LIMIT.set(512 if "_selected" in payload else None)
    ctx = LOCAL_CONTEXT.set(2048 if "27b" in str(cfg.local_model).lower() else 4096)
    message_token = MESSAGES.set(None)
    try:
        if catalog and payload.get("_run"):
            from ollama_client import local_system_prompt
            profile = LOCAL_CONTEXT.get() if mode == "local" else 8192
            messages = context.native_messages(payload["_run"], payload["_selected"], payload["_steps"], tools.TOOLS,
                                                profile, local_system_prompt(cfg.local_model) if mode == "local" else "")
            MESSAGES.set(messages)
        if not mock and mode != "local" and "_steps" in payload:
            from webui.service import _fireworks_rates
            # Reserve a conservative full prompt/output allowance for each
            # possible remote generation before dispatching a paid request.
            models = [cfg.remote_model] if mode == "remote" else [cfg.cheap_model, cfg.strong_model]
            from config import settings
            input_bound = len(json.dumps([MESSAGES.get() or prompt, catalog, settings.system_prompt]).encode()) + 512
            reserve = sum((input_bound * _fireworks_rates(m)[0] + 512 * _fireworks_rates(m)[1])/1e6 for m in models)
            if usage(payload["_steps"])["cost_usd"] + reserve > .10:
                raise RuntimeError("Insufficient cost budget for the next request's reserved allowance")
        result = execute(cfg, prompt)
    finally:
        TOOLS.reset(t)
        OUTPUT_LIMIT.reset(limit)
        LOCAL_CONTEXT.reset(ctx)
        MESSAGES.reset(message_token)
    if not result["ok"]:
        raise RuntimeError(result["error"])
    result["protocol"] = "native" if catalog else "json"
    if mock and "_selected" in payload:
        result["answer"] = json.dumps({"answer": "Preview completed. No tools were executed and no changes were made."})
    return result


def start(kind, payload, allow_real):
    if kind not in {"agent", "research", "compare", "edit"}:
        raise ValueError("Unknown run type")
    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 10_000:
        raise ValueError("Provide a prompt of 1–10,000 characters")
    if kind == "compare":
        choices = payload.get("choices", [])
        if len(choices) != 2:
            raise ValueError("Comparison requires two model choices")
        for choice in choices:
            conversation_config(choice.get("mode", "remote"), choice.get("model"), not allow_real)
    selected = payload.get("tools", [])
    if not isinstance(selected, list) or any(x not in tools.TOOLS for x in selected):
        raise ValueError("Choose tools from the available registry")
    if kind in {"agent", "research"}:
        mode = payload.get("mode", store.preferences().get("agent_mode", "local"))
        cfg = conversation_config(mode, payload.get("model") or None, not allow_real or payload.get("mock") is True)
        model = cfg.local_model if mode == "local" else cfg.remote_model if mode == "remote" else cfg.strong_model
        payload = {**payload, "mode": mode, "model": model}
    run_id = store.run_create(kind, {**payload, "real_allowed": allow_real})
    resume(run_id, allow_real)
    return store.run_get(run_id)


def resume(run_id, allow_real):
    run = store.run_get(run_id)
    if run["status"] != "queued":
        raise ValueError("Only a queued run can execute")
    user = store.USER.get()
    with _LOCK:
        key = (user, run_id)
        if key in _ACTIVE:
            raise ValueError("Run is already executing")
        _ACTIVE.add(key)
    POOL.submit(worker, user, run_id, allow_real)


def cancel(run_id):
    run = store.run_get(run_id)
    if run["status"] in {"queued", "running", "awaiting_review"}:
        store.run_save(run_id, status="cancelled" if run["status"] == "awaiting_review" else "cancelling")
        with _LOCK:
            stream = _STREAMS.get((store.USER.get(), run_id))
            if stream:
                stream.cancel.set()
    return store.run_get(run_id)


def review(run_id, approve, allow_real, action_id=None):
    # A conditional transition prevents double approval from two browser tabs.
    with store.db() as conn:
        row = conn.execute("SELECT result,status FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row or row["status"] != "awaiting_review":
            raise ValueError("Run is not waiting for review")
        saved = json.loads(row["result"])
        pending = saved.get("pending_tool")
        if not pending:
            raise ValueError("Missing pending tool")
        if action_id is not None and action_id != pending.get("action_id"):
            raise ValueError("Tool request changed; review the current action")
        saved.pop("pending_tool", None)
        saved["approved_tool" if approve else "declined_tool"] = pending
        changed = conn.execute("UPDATE runs SET status=?,result=?,updated=? WHERE id=? AND status='awaiting_review'",
                               ("queued" if approve else "cancelled", json.dumps(saved), store.now(), run_id))
        if not changed.rowcount:
            raise ValueError("Review was already handled")
    if approve:
        resume(run_id, allow_real)
    return store.run_get(run_id)


def worker(user, run_id, allow_real):
    token = store.USER.set(user)
    try:
        run = store.run_get(run_id)
        if run["status"] == "cancelling":
            store.run_save(run_id, status="cancelled")
            return
        store.run_save(run_id, status="running")
        if run["kind"] == "compare":
            compare(run, allow_real)
        elif run["kind"] == "edit":
            edit(run, allow_real)
        else:
            agent(run, allow_real)
    except Exception as exc:
        current = store.run_get(run_id)
        stopped = current["status"] in {"cancelling", "cancelled"} or isinstance(exc, Cancelled)
        store.run_save(run_id, status="cancelled" if stopped else "failed",
                       result={**current["result"], "error": str(exc)[:2000]})
    finally:
        with _LOCK:
            _ACTIVE.discard((user, run_id))
        store.USER.reset(token)


def is_cancelled(run_id):
    if store.run_get(run_id)["status"] in {"cancelling", "cancelled"}:
        store.run_save(run_id, status="cancelled")
        return True
    return False


def compare(run, allow_real):
    steps = []
    choices = list(run["payload"]["choices"])
    random.SystemRandom().shuffle(choices)
    for index, choice in enumerate(choices):
        if is_cancelled(run["id"]):
            return
        result = generate({**choice, "mock": run["payload"].get("mock", False)}, run["payload"]["prompt"], allow_real)
        steps.append({"label": "AB"[index], "answer": result["answer"], "model": result["final_model"],
                      "provider": result["provider"], "latency_s": result["latency_s"],
                      "cost": result["estimated_cost_usd"], "mock": result["mock"]})
        store.run_save(run["id"], steps=steps)
    if not is_cancelled(run["id"]):
        store.run_save(run["id"], status="completed", result={"answer": "Comparison ready", "revealed": False})


def edit(run, allow_real):
    item = store.get(run["payload"].get("item_id", ""))
    result = generate(run["payload"], "Rewrite the following document according to the request. "
                      "Return only the complete revised document.\nRequest: " + run["payload"]["prompt"] +
                      "\nDocument:\n" + item["content"][:30000], allow_real)
    if not is_cancelled(run["id"]):
        store.run_save(run["id"], status="completed", steps=[result], result={
            "answer": result["answer"], "item_id": item["id"], "base_updated": item["updated"]})


def usage(steps):
    models = [s for s in steps if s.get("type") == "model"]
    return {"requests": sum(s.get("attempts", 1) for s in models), "model_steps": len(models),
            "tokens": sum(sum(v for k, v in s.get("tokens", {}).items() if k != "billable" and isinstance(v, (int, float))) for s in models),
            "cost_usd": sum(s.get("cost", 0) for s in models),
            "active_seconds": sum(s.get("duration_s", 0) for s in steps),
            "validation_errors": sum(s.get("type") == "validation_error" for s in steps)}


def check_budget(steps):
    used = usage(steps)
    if used["requests"] >= 8 or used["tokens"] >= 20000 or used["cost_usd"] >= .10 or used["active_seconds"] >= 180:
        raise RuntimeError("Cumulative run budget reached; inspect usage and completed steps")
    if used["validation_errors"] >= 3:
        raise RuntimeError("Model decision validation failed three times; no successful final answer")


def perform(run_id, pending, steps, approved=False):
    from .effects import execute_action
    if is_cancelled(run_id):
        return False
    check_budget(steps)
    started = time.monotonic()
    name, args = pending["tool"], pending["arguments"]
    stream = Stream(lambda event: None, deadline=time.monotonic()+max(0, 180-usage(steps)["active_seconds"]))
    stream_token = CURRENT.set(stream)
    with _LOCK:
        _STREAMS[(store.USER.get(), run_id)] = stream
    try:
        if is_cancelled(run_id):
            return False
        store.run_save(run_id, result={"phase": "tool", "tool": name, "usage": usage(steps)})
        envelope = execute_action(run_id, pending, lambda: tools.call(name, args, approved=approved))
    finally:
        CURRENT.reset(stream_token)
        with _LOCK:
            _STREAMS.pop((store.USER.get(), run_id), None)
    raw = envelope["data"]
    output = context.offload(run_id, raw)
    steps.append({"type": "tool", **pending, "output": output, "ok": envelope["ok"],
                  "error": envelope["error"], "duration_s": time.monotonic()-started})
    store.run_save(run_id, steps=steps, result={"usage": usage(steps)})
    return True


def agent(run, allow_real):
    from .effects import identity
    payload = run["payload"]
    selected = list(payload.get("tools", ["list_items", "read_item"]))
    if run["kind"] == "research" and "read_url" not in selected:
        selected.append("read_url")
    steps = list(run["steps"])
    pending = run["result"].get("approved_tool")
    if pending and not perform(run["id"], pending, steps, approved=True):
        return
    while True:
        if is_cancelled(run["id"]):
            return
        check_budget(steps)
        prompt = context.build(run, selected, steps, tools.TOOLS)
        partial, last = [], [0.0]
        def event(data):
            if data["type"] == "delta":
                partial.append(data["text"])
            if time.monotonic()-last[0] > .3:
                last[0] = time.monotonic()
                store.run_save(run["id"], result={"usage": usage(steps), "progress": "".join(partial)[-4000:],
                                                "phase": "generating", "usage_complete": False})
        stream = Stream(event, deadline=time.monotonic()+max(0, 180-usage(steps)["active_seconds"]))
        with _LOCK:
            _STREAMS[(store.USER.get(), run["id"])] = stream
        stream_token = CURRENT.set(stream)
        started = time.monotonic()
        try:
            if is_cancelled(run["id"]):
                return
            result = generate({**payload, "_selected": selected, "_run": run, "_steps": steps}, prompt, allow_real)
        finally:
            CURRENT.reset(stream_token)
            with _LOCK:
                _STREAMS.pop((store.USER.get(), run["id"]), None)
        steps.append({"type": "model", "answer": result["answer"], "tool_calls": result.get("tool_calls", []), "protocol": result.get("protocol", "json"),
                      "model": result["final_model"], "provider": result["provider"], "route": result["route"],
                      "tokens": result["tokens"], "cost": result["estimated_cost_usd"], "mock": result["mock"],
                      "duration_s": time.monotonic()-started, "attempts": max(1, len(stream.attempts))})
        store.run_save(run["id"], steps=steps, result={"usage": usage(steps), "usage_complete": True})
        if is_cancelled(run["id"]):
            return
        try:
            decision = contracts.decision(result, selected)
            if run["kind"] == "research":
                known = {r["url"] for r in context.receipts(steps) if r["ok"] and r.get("url")}
                content = decision.get("answer", "")
                if decision.get("tool") == "save_item":
                    content = decision["arguments"].get("content", "")
                cited = {url.rstrip(".,;:!?)\"]}") for url in re.findall(r"https?://[^\s<>]+", content)}
                if cited - known:
                    raise ValueError("Research citations must point to sources successfully read in this run")
                if decision.get("tool") == "save_item" and decision["arguments"].get("kind") in {"note", "document"} and known and not cited:
                    raise ValueError("Include a read source URL in the research note")
        except (ValueError, TypeError, KeyError) as exc:
            steps.append({"type": "validation_error", "error": str(exc)[:500],
                          "instruction": "Correct the decision format or arguments; no action was executed."})
            store.run_save(run["id"], steps=steps, result={"usage": usage(steps)})
            continue
        if "answer" in decision:
            sources = [s["output"]["url"] for s in steps if s.get("ok") and s.get("tool") in {"read_url", "browser"} and isinstance(s.get("output"), dict) and s["output"].get("url")]
            failed_tools = sum(s.get("type") == "tool" and not s.get("ok") for s in steps)
            store.run_save(run["id"], status="completed_with_errors" if failed_tools else "completed", result={"answer": decision["answer"], "failed_tools": failed_tools,
                           "sources": list(dict.fromkeys(sources)), "usage": usage(steps), "usage_complete": True})
            return
        pending = {**decision, "action_id": identity(run["id"], len(steps), decision)}
        if decision["tool"] in tools.REVIEW_TOOLS:
            store.run_save(run["id"], status="awaiting_review", result={"pending_tool": pending, "usage": usage(steps)})
            return
        if not perform(run["id"], pending, steps):
            return


def recover():
    # A crashed generation can have billed the provider: surface interruption,
    # do not repeat the request or a previously reviewed external action.
    with store.db() as conn:
        conn.execute("UPDATE runs SET status='interrupted',updated=? WHERE status IN ('queued','running','cancelling')", (store.now(),))
        conn.execute("UPDATE actions SET status='uncertain',updated=? WHERE status='started'", (store.now(),))


def resume_interrupted(run_id, allow_real):
    from .effects import READ_ONLY
    with store.db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row or row["status"] != "interrupted" or row["kind"] not in {"agent", "research"}:
            raise ValueError("Choose an interrupted agent run")
        payload = json.loads(row["payload"])
        saved = json.loads(row["result"])
        if saved.get("usage_complete") is False and payload.get("mode", "remote") != "local":
            raise ValueError("Interrupted remote usage is unknown; reconcile billing and create a new budgeted task")
        if set(payload.get("tools", ["list_items", "read_item"])) - READ_ONLY:
            raise ValueError("Only read-only runs can resume; inspect effects and create a fresh task")
        if conn.execute("SELECT 1 FROM actions WHERE run_id=? AND status='uncertain'", (run_id,)).fetchone():
            raise ValueError("Resolve uncertain action outcomes before resuming")
        conn.execute("UPDATE runs SET status='queued',updated=? WHERE id=?", (store.now(), run_id))
        conn.execute("UPDATE occurrences SET status='queued',updated=? WHERE run_id=?", (store.now(), run_id))
        conn.execute("UPDATE jobs SET status='running',updated=? WHERE run_id=?", (store.now(), run_id))
    resume(run_id, allow_real)
    return store.run_get(run_id)
