"""HTTP endpoints for the native workspace, shared by its static UI."""
import base64
import json
import mimetypes
import os
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

from . import store, runs, tools, connections, auth


def safe_run(run):
    if run["kind"] == "compare" and not run["result"].get("revealed"):
        run = {**run, "payload": {"prompt": run["payload"]["prompt"]},
               "steps": [{"label": step["label"], "answer": step["answer"]} for step in run["steps"]]}
    return run


def handle(handler, method):
    parsed = urlsplit(handler.path)
    path = parsed.path.removeprefix("/api/workspace")
    query = parse_qs(parsed.query)
    data = {}
    try:
        if method in {"POST", "PATCH", "PUT"}:
            data = handler._read_json()
            if data is None:
                raise ValueError("Invalid JSON body")
        if method == "GET" and path == "/state":
            from config import settings
            from ollama_client import request
            try:
                local = request("/api/tags", timeout=2).get("models", [])
                local_error = None
            except Exception as exc:
                local, local_error = [], str(exc)
            result = {"items": store.items(), "runs": [safe_run(r) for r in store.run_list()],
                      "jobs": store.jobs(), "preferences": store.preferences(),
                      "tools": tools.TOOLS, "connections": connections.status(),
                      "local_models": local, "local_error": local_error,
                      "local_backend": store.preferences().get("local_backend", settings.local_backend),
                      "default_model": settings.strong_model_name,
                      "local_model": store.preferences().get("local_model", settings.local_model_name),
                      "real_allowed": handler.server.allow_real,
                      "user": handler.account.get("username") if handler.account else None}
        elif path == "/items" and method == "GET":
            result = store.items((query.get("kind") or [None])[0], (query.get("q") or [""])[0])
        elif path == "/items" and method == "POST":
            result = store.save(data)
        elif match := re.fullmatch(r"/items/([a-f0-9]+)(?:/(revisions|restore|export))?", path):
            item_id, suffix = match.groups()
            if method == "GET" and suffix == "revisions":
                result = store.revisions(item_id)
            elif method == "POST" and suffix == "restore":
                revision = next((r for r in store.revisions(item_id) if r["id"] == data.get("revision_id")), None)
                if revision is None:
                    raise LookupError("Revision not found")
                result = store.save({"title": revision["title"], "content": revision["content"], "meta": json.loads(revision["meta"])}, item_id)
            elif method == "GET" and suffix == "export":
                item = store.get(item_id)
                body = item["content"].encode()
                format_name = (query.get("format") or ["md"])[0]
                types = {"md": "text/markdown", "txt": "text/plain", "csv": "text/csv", "html": "text/html"}
                if format_name not in types:
                    raise ValueError("Export format must be md, txt, csv or html")
                if format_name == "html":
                    import html
                    body = ("<!doctype html><html><meta charset=utf-8><title>" + html.escape(item["title"]) +
                            "</title><body><h1>" + html.escape(item["title"]) + "</h1><pre>" +
                            html.escape(item["content"]) + "</pre></body></html>").encode()
                filename = re.sub(r"[^A-Za-z0-9_.-]", "_", item["title"])[:80] + "." + format_name
                return download(handler, body, types[format_name] + "; charset=utf-8", filename)
            elif method == "GET" and suffix is None:
                result = store.get(item_id)
            elif method == "PATCH" and suffix is None:
                result = store.save(data, item_id)
            elif method == "DELETE" and suffix is None:
                store.delete(item_id)
                result = {"deleted": True}
            else:
                raise LookupError("Route not found")
        elif path == "/preferences" and method in {"GET", "POST"}:
            result = store.preferences(data if method == "POST" else None)
        elif path == "/local/pull" and method == "POST":
            from ollama_client import request
            from config import settings
            import threading
            if not handler.server.allow_real:
                raise ValueError("Model downloads require real mode")
            model = data.get("model", "")
            if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_./:-]{1,150}", model):
                raise ValueError("Invalid Ollama model tag")
            request("/api/tags", timeout=2)
            run_id = store.run_create("download", {"prompt": "Download " + model})
            user = store.USER.get()
            def pull():
                token = store.USER.set(user)
                try:
                    store.run_save(run_id, status="running")
                    request("/api/pull", {"model": model, "stream": False}, timeout=1800)
                    store.run_save(run_id, status="completed", result={"answer": model + " installed. Select it in Local models."})
                except Exception as exc:
                    store.run_save(run_id, status="failed", result={"error": str(exc)})
                finally:
                    store.USER.reset(token)
            threading.Thread(target=pull, daemon=True).start()
            result = store.run_get(run_id)
        elif path == "/local/select" and method == "POST":
            from ollama_client import request
            model = data.get("model", "")
            request("/api/show", {"model": model})
            # Keep backend preferences user-scoped; no browser writes to .env.
            result = store.preferences({"local_model": model, "local_backend": "ollama"})
        elif path == "/connections" and method == "POST":
            result = connections.configure(data)
        elif path == "/google/connect" and method == "POST":
            result = connections.google_start(handler.server.server_address[1])
        elif path == "/google/disconnect" and method == "POST":
            result = connections.disconnect()
        elif path == "/mail" and method == "GET":
            result = connections.mail_list((query.get("q") or ["in:inbox"])[0])
        elif path.startswith("/mail/") and method == "GET":
            result = connections.mail_read(path.split("/")[-1])
        elif path == "/mail/send" and method == "POST":
            if data.get("confirm") is not True:
                raise ValueError("Review the recipient and message before sending")
            result = connections.mail_send(str(data.get("draft_id", "")))
        elif path == "/mail/labels" and method == "POST":
            result = connections.google("gmail/v1/users/me/messages/" + str(data["id"]) + "/modify",
                                        {"addLabelIds": data.get("add", []), "removeLabelIds": data.get("remove", [])})
        elif path == "/calendar" and method == "GET":
            result = connections.calendar_list()
        elif path == "/calendar/publish" and method == "POST":
            if data.get("confirm") is not True:
                raise ValueError("Review the event before publishing")
            result = connections.calendar_publish(str(data.get("id", "")))
        elif path == "/calendar/delete" and method == "POST":
            if data.get("confirm") is not True:
                raise ValueError("Confirm event deletion")
            from urllib.parse import quote
            result = connections.google("calendar/v3/calendars/primary/events/" + quote(str(data["google_id"]), safe=""), method="DELETE")
        elif path == "/runs" and method == "POST":
            result = safe_run(runs.start(data.get("kind", "agent"), data, handler.server.allow_real))
        elif path == "/runs" and method == "GET":
            result = [safe_run(r) for r in store.run_list()]
        elif match := re.fullmatch(r"/runs/([a-f0-9]+)(?:/(cancel|review|reveal|apply))?", path):
            run_id, action = match.groups()
            if method == "GET" and action is None:
                result = safe_run(store.run_get(run_id))
            elif method == "POST" and action == "cancel":
                result = safe_run(runs.cancel(run_id))
            elif method == "POST" and action == "review":
                result = safe_run(runs.review(run_id, data.get("approve") is True, handler.server.allow_real))
            elif method == "POST" and action == "reveal":
                run = store.run_get(run_id)
                if run["kind"] != "compare" or run["status"] != "completed":
                    raise ValueError("Wait for the comparison to complete")
                vote = data.get("vote")
                if vote not in {"A", "B", "tie"}:
                    raise ValueError("Choose A, B or tie")
                store.run_save(run_id, result={**run["result"], "revealed": True, "vote": vote})
                result = store.run_get(run_id)
            elif method == "POST" and action == "apply":
                run = store.run_get(run_id)
                if run["kind"] != "edit" or run["status"] != "completed":
                    raise ValueError("Choose a completed document edit")
                item = store.get(run["result"]["item_id"])
                if item["updated"] != run["result"]["base_updated"]:
                    raise ValueError("Document changed since the suggestion; request a fresh edit")
                result = store.save({"content": run["result"]["answer"]}, item["id"])
            else:
                raise LookupError("Route not found")
        elif path == "/jobs" and method == "POST":
            result = {"id": store.job_save(data)}
        elif path == "/jobs" and method == "GET":
            result = store.jobs()
        elif path.startswith("/jobs/") and method == "DELETE":
            with store.db() as conn:
                conn.execute("UPDATE jobs SET status='cancelled',updated=? WHERE id=? AND status='pending'", (store.now(), path.split("/")[-1]))
            result = {"cancelled": True}
        elif path == "/files" and method == "POST":
            name = str(data.get("name", ""))
            raw = base64.b64decode(data.get("data", ""), validate=True)
            if len(raw) > 600_000:
                raise ValueError("Uploads must be at most 600 KB")
            target = tools.file_path(name)
            if target.exists():
                raise ValueError("A file with that name already exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
            content = raw.decode("utf-8", "replace") if mime.startswith("text/") else ""
            result = store.save({"kind": "asset", "title": Path(name).name,
                                 "content": content, "meta": {"filename": name, "mime": mime, "size": len(raw)}})
        elif path.startswith("/files/") and method == "GET":
            item = store.get(path.split("/")[-1])
            target = tools.file_path(item["meta"]["filename"])
            return download(handler, target.read_bytes(), "application/octet-stream", target.name)
        elif path == "/images" and method == "POST":
            if not handler.server.allow_real:
                raise ValueError("Image generation requires real mode")
            config = connections.secrets_read()
            if not config.get("image_url") or not config.get("image_model"):
                raise ValueError("Configure an image provider in Connections")
            result = connections.http_json(config["image_url"], {"model": config["image_model"], "prompt": str(data.get("prompt", "")), "response_format": "b64_json", "n": 1},
                                           {"Authorization": "Bearer " + config.get("image_key", "")})
            encoded = result.get("data", [{}])[0].get("b64_json")
            if not encoded:
                raise ValueError("Image provider must return b64_json; check provider compatibility")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > 5_000_000:
                raise ValueError("Generated image exceeds storage limit")
            import uuid
            name = uuid.uuid4().hex + ".png"
            tools.file_path(name).write_bytes(raw)
            result = store.save({"kind": "asset", "title": str(data.get("prompt", "Generated image"))[:200],
                                 "meta": {"filename": name, "mime": "image/png", "size": len(raw)}})
        elif path == "/mcp/discover" and method == "POST":
            if store.USER.get() != "local":
                raise PermissionError("Only the owner can start MCP host processes")
            from .mcp import Session
            config = store.preferences().get("mcp_servers", {}).get(data.get("server"))
            if not config:
                raise ValueError("Configure this MCP server first")
            with Session(config, store.home()) as session:
                result = session.call("tools/list")
        else:
            raise LookupError("Workspace route not found")
        handler._send_json(result)
    except PermissionError as exc:
        handler._send_json({"error": str(exc)}, 403)
    except LookupError as exc:
        handler._send_json({"error": str(exc)}, 404)
    except (ValueError, KeyError, TypeError) as exc:
        handler._send_json({"error": str(exc)}, 400)
    except Exception as exc:
        handler._send_json({"error": str(exc)[:1000]}, 502)


def download(handler, body, mime, name):
    handler.send_response(200)
    handler.send_header("Content-Type", mime)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Content-Disposition", 'attachment; filename="' + re.sub(r"[^A-Za-z0-9_.-]", "_", name) + '"')
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(body)
