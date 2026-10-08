"""Stdlib HTTP wrapper for the demo UI.

    python3 -m webui.server                 # mock-only, http://127.0.0.1:8642
    python3 -m webui.server --real          # allow real (billable) calls
    python3 -m webui.server --port 9000

Safety rails:
- Binds 127.0.0.1 only — the demo is never reachable from another machine.
- Without --real, every request is FORCED into mock mode server-side, no
  matter what the browser sends; the UI shows the lock.
- The Fireworks API key is never serialized into any response.
- Session settings are applied per-request and restored (webui/service.py);
  nothing from the browser is persisted or written to .env.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

# Allow `python3 webui/server.py` from anywhere, like scripts/banana.py does.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _load_dotenv() -> None:
    """Load local demo settings before importing the config singleton."""
    try:
        with open(os.path.join(_REPO, ".env")) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    except FileNotFoundError:
        pass


_load_dotenv()

from webui import service  # noqa: E402

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_MAX_BODY = 1 << 20  # 1 MiB — a demo prompt, not an upload endpoint


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr: Tuple[str, int], allow_real: bool):
        super().__init__(addr, DemoHandler)
        self.allow_real = allow_real


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    # ── plumbing ─────────────────────────────────────────────────────────
    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > _MAX_BODY:
            return None
        try:
            data = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _google_error(self, message: str) -> None:
        import html
        body = ("<!doctype html><html lang=en><meta charset=utf-8>"
                "<meta name=viewport content='width=device-width,initial-scale=1'>"
                "<title>Nana · Google connection</title>"
                "<style>body{font:18px system-ui;margin:0;background:#e6e5fa;color:#232538}"
                "main{max-width:560px;margin:12vh auto;padding:32px;background:#fff;border-radius:24px}"
                "a{display:inline-block;padding:12px 20px;background:#232538;color:white;border-radius:12px;text-decoration:none}"
                "p{line-height:1.6}</style><main><h1>Reconnect Google</h1><p>" + html.escape(message) +
                "</p><p>Return to Connections and select Connect Google to start a fresh sign-in. "
                "Your saved client credentials are still available.</p>"
                "<a href='/workspace#connections'>Return to Connections</a></main></html>").encode()
        self.send_response(400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # quiet default logging
        if "google/callback" in self.path:
            sys.stderr.write("webui: Google authorization callback\n")
            return
        sys.stderr.write("webui: " + fmt % args + "\n")

    def _guard(self) -> bool:
        """Local origin checks and optional accounts for all workspace/chat routes."""
        from workspace import auth, store, connections
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        if host not in hosts:
            self._send_json({"error": "Invalid local host"}, 403)
            return False
        if self.command != "GET" and self.headers.get("Origin") not in {None, *("http://" + h for h in hosts)}:
            self._send_json({"error": "Cross-origin action rejected"}, 403)
            return False
        self.account = auth.session(self.headers)
        store.USER.set(self.account["user_id"] if self.account else "local")
        parsed = urlsplit(self.path)
        if parsed.path == "/api/workspace/google/callback":
            query = parse_qs(parsed.query)
            state = (query.get("state") or [""])[0]
            # OAuth callbacks can arrive in a different browser; the random,
            # one-use state binds the response to the initiating Nana user.
            found = False
            for home in store.ROOT.iterdir() if store.ROOT.exists() else []:
                if not home.is_dir():
                    continue
                store.USER.set(home.name)
                if connections.google_has_pending(state):
                    found = True
                    break
            try:
                if not found:
                    raise ValueError("This Google sign-in link has already been used or is no longer valid.")
                connections.google_callback(query)
                self.send_response(303)
                self.send_header("Location", "/workspace#connections")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
            except Exception as exc:
                self._google_error(str(exc))
            return False
        if parsed.path.startswith("/api/auth"):
            try:
                data = self._read_json() if self.command == "POST" else {}
                if data is None:
                    raise ValueError("Invalid JSON body")
                name = self.account["username"] if self.account else None
                if parsed.path == "/api/auth/status" and self.command == "GET":
                    self._send_json({"enabled": bool(auth.read_users()), "username": name})
                elif parsed.path == "/api/auth/create" and self.command == "POST":
                    self._send_json(auth.create(data.get("username", ""), data.get("password", ""), name))
                elif parsed.path == "/api/auth/login" and self.command == "POST":
                    token = auth.login(data)
                    self.send_response(200)
                    self.send_header("Set-Cookie", f"nana_session={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age=43200")
                    self.send_header("Content-Length", "2")
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b"{}")
                elif parsed.path == "/api/auth/logout" and self.command == "POST":
                    from http.cookies import SimpleCookie
                    cookie = SimpleCookie(self.headers.get("Cookie", ""))
                    if "nana_session" in cookie:
                        auth.SESSIONS.pop(cookie["nana_session"].value, None)
                    self.send_response(200)
                    self.send_header("Set-Cookie", "nana_session=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0")
                    self.send_header("Content-Length", "2")
                    self.end_headers()
                    self.wfile.write(b"{}")
                elif parsed.path == "/api/auth/totp/setup" and self.command == "POST":
                    self._send_json(auth.setup_totp(name))
                elif parsed.path == "/api/auth/totp/confirm" and self.command == "POST" and name:
                    self._send_json(auth.confirm_totp(name, data.get("code", "")))
                else:
                    self._send_json({"error": "Route not found"}, 404)
            except PermissionError as exc:
                self._send_json({"error": str(exc)}, 403)
            except (ValueError, TypeError) as exc:
                self._send_json({"error": str(exc)}, 400)
            return False
        if parsed.path == "/auth":
            self._send_static("auth.html", "text/html; charset=utf-8")
            return False
        if auth.read_users() and not self.account:
            if self.command == "GET" and not parsed.path.startswith(("/api/", "/conversations")):
                self.send_response(303)
                self.send_header("Location", "/auth")
                self.end_headers()
            else:
                self._send_json({"error": "Sign in to Nana"}, 401)
            return False
        if parsed.path.startswith("/api/workspace"):
            from workspace.api import handle
            handle(self, self.command)
            return False
        return True

    # ── routes ───────────────────────────────────────────────────────────
    def do_GET(self) -> None:
        if not self._guard():
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._send_static("index.html", "text/html; charset=utf-8")
        elif path == "/chat":
            self._send_static("chat.html", "text/html; charset=utf-8")
        elif path == "/workspace":
            self._send_static("workspace.html", "text/html; charset=utf-8")
        elif path == "/api/config":
            payload = service.frontend_config()
            payload["real_allowed"] = self.server.allow_real
            self._send_json(payload)
        elif path == "/api/chat/config":
            from config import settings
            from workspace.store import preferences
            prefs = preferences()
            self._send_json({
                "real_allowed": self.server.allow_real,
                "known_local_models": list(dict.fromkeys([prefs.get("local_model", settings.local_model_name), *service.KNOWN_LOCAL_MODELS])),
                "known_fireworks_models": service.KNOWN_FIREWORKS_MODELS,
                "default_mode": "hybrid",
                "default_models": {
                    "hybrid": settings.strong_model_name,
                    "remote": settings.strong_model_name,
                    "local": prefs.get("local_model", settings.local_model_name),
                },
            })
        elif path.startswith("/conversations"):
            self._chat_request("GET", path, parse_qs(parsed.query))
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if not self._guard():
            return
        if urlsplit(self.path).path.startswith("/conversations"):
            self._chat_request("POST", urlsplit(self.path).path)
            return
        payload = self._read_json()
        if payload is None:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return
        cfg, errors = service.config_from_payload(payload.get("config") or {})
        if not self.server.allow_real and not cfg.mock:
            # Server-side enforcement: browsers cannot opt into real spend.
            cfg.mock = True
            errors = list(errors)  # keep validation errors, add the notice
            if self.path == "/api/run":
                errors.append(
                    "real mode is disabled: restart the server with --real "
                    "to allow billable calls"
                )
        if self.path == "/api/describe":
            description = service.describe(cfg)
            description["errors"] = errors
            self._send_json(description)
        elif self.path == "/api/run":
            if errors:
                self._send_json(
                    {"ok": False, "error": "; ".join(errors)}, status=400
                )
                return
            result = service.execute(cfg, payload.get("prompt", ""))
            self._send_json(result, status=200 if result.get("ok") else 400)
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_PATCH(self) -> None:
        if not self._guard():
            return
        self._chat_request("PATCH", urlsplit(self.path).path)

    def do_DELETE(self) -> None:
        if not self._guard():
            return
        self._chat_request("DELETE", urlsplit(self.path).path)

    def _chat_request(self, method: str, path: str, query=None) -> None:
        """Expose the persistent chat API on the existing localhost demo."""
        from chat.database import initialise_database
        from chat.repository import (
            create_conversation, delete_conversation, get_conversation,
            list_conversations, list_messages, rename_conversation,
            search_conversations, update_conversation_settings,
        )
        from chat.service import MAX_MESSAGE_CHARS, conversation_config, send_message

        stream_cancel = re.fullmatch(r"/conversations/streams/([a-f0-9-]+)/cancel", path)
        if method == "POST" and stream_cancel:
            from chat.streaming import cancel
            self._send_json(cancel(stream_cancel.group(1)))
            return
        stream_match = re.fullmatch(r"/conversations/(\d+)/stream", path)
        if method == "POST" and stream_match:
            from chat.streaming import serve
            payload = self._read_json()
            try:
                if payload is None:
                    raise ValueError("Invalid JSON body")
                if get_conversation(int(stream_match.group(1))) is None:
                    raise LookupError("Conversation not found")
                serve(self, int(stream_match.group(1)), payload)
            except (ValueError, LookupError) as exc:
                self._send_json({"error": str(exc)}, 400)
            return

        if not path.startswith("/conversations"):
            self._send_json({"error": "not found"}, status=404)
            return
        initialise_database()
        payload = None
        if method in ("POST", "PATCH"):
            payload = self._read_json()
            if payload is None:
                self._send_json({"error": "invalid JSON body"}, status=400)
                return
        query = query or {}
        match = re.fullmatch(r"/conversations/(\d+)(?:/(messages|title|settings))?", path)
        try:
            if method == "GET" and path == "/conversations":
                result = [asdict(item) for item in list_conversations()]
            elif method == "GET" and path == "/conversations/search":
                term = (query.get("q") or [""])[0].strip()
                if not term:
                    raise ValueError("Search query is required")
                result = [asdict(item) for item in search_conversations(term)]
            elif method == "POST" and path == "/conversations":
                title = str(payload.get("title", "New Chat")).strip()
                mode = payload.get("mode", "hybrid")
                model = payload.get("model")
                if not title or len(title) > 200:
                    raise ValueError("Title must be 1–200 characters")
                conversation_config(mode, model, mock=True)
                result = asdict(create_conversation(title, mode, model))
            elif match:
                conversation_id = int(match.group(1))
                suffix = match.group(2)
                if method == "GET" and suffix is None:
                    conversation = get_conversation(conversation_id)
                    if conversation is None:
                        raise LookupError("Conversation not found")
                    result = {
                        "conversation": asdict(conversation),
                        "messages": [asdict(item) for item in list_messages(conversation_id)],
                    }
                elif method == "DELETE" and suffix is None:
                    if not delete_conversation(conversation_id):
                        raise LookupError("Conversation not found")
                    result = {"deleted": True, "conversation_id": conversation_id}
                elif method == "POST" and suffix == "messages":
                    content = payload.get("content", "")
                    if not isinstance(content, str) or len(content) > MAX_MESSAGE_CHARS:
                        raise ValueError(f"Message must be at most {MAX_MESSAGE_CHARS} characters")
                    if payload.get("mock") is False and not self.server.allow_real:
                        raise ValueError("Real calls are disabled; restart with --real")
                    result = send_message(
                        conversation_id, content,
                        mock=payload.get("mock", True) is not False,
                        allow_real=self.server.allow_real,
                    )
                elif method == "PATCH" and suffix == "title":
                    title = str(payload.get("title", "")).strip()
                    if not title or len(title) > 200:
                        raise ValueError("Title must be 1–200 characters")
                    conversation = rename_conversation(conversation_id, title)
                    if conversation is None:
                        raise LookupError("Conversation not found")
                    result = asdict(conversation)
                elif method == "PATCH" and suffix == "settings":
                    mode = payload.get("mode")
                    model = payload.get("model")
                    conversation_config(mode, model, mock=True)
                    conversation = update_conversation_settings(conversation_id, mode, model)
                    if conversation is None:
                        raise LookupError("Conversation not found")
                    result = asdict(conversation)
                else:
                    raise LookupError("Route not found")
            else:
                raise LookupError("Route not found")
            self._send_json(result)
        except LookupError as err:
            self._send_json({"error": str(err)}, status=404)
        except ValueError as err:
            self._send_json({"error": str(err)}, status=400)
        except RuntimeError as err:
            self._send_json({"error": str(err)}, status=502)
        except Exception:
            self._send_json({"error": "Chat request failed"}, status=500)

    def _send_static(self, name: str, content_type: str) -> None:
        path = os.path.join(_STATIC_DIR, name)
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError:
            self._send_json({"error": f"missing static file {name}"}, 500)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(port: int = 0, allow_real: bool = False) -> DemoServer:
    """Bound but not serving — tests use port 0 and serve_forever in a thread."""
    return DemoServer(("127.0.0.1", port), allow_real=allow_real)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="banana demo UI server")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument(
        "--real",
        action="store_true",
        help="allow REAL model calls (billable Fireworks requests, local "
        "model weight loading). Default: mock only.",
    )
    args = parser.parse_args(argv)

    server = make_server(port=args.port, allow_real=args.real)
    from workspace.scheduler import Scheduler
    scheduler = Scheduler(args.real)
    mode = "REAL CALLS ALLOWED (billable)" if args.real else "mock-only"
    print(
        f"banana demo UI: http://127.0.0.1:{server.server_address[1]}  [{mode}]"
    )
    if args.real:
        print(
            "  --real is set: remote requests will bill your Fireworks "
            "account; local mode may download model weights on first use.",
            file=sys.stderr,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        scheduler.stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
