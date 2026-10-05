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

    def log_message(self, fmt: str, *args) -> None:  # quiet default logging
        sys.stderr.write("webui: " + fmt % args + "\n")

    # ── routes ───────────────────────────────────────────────────────────
    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._send_static("index.html", "text/html; charset=utf-8")
        elif path == "/chat":
            self._send_static("chat.html", "text/html; charset=utf-8")
        elif path == "/api/config":
            payload = service.frontend_config()
            payload["real_allowed"] = self.server.allow_real
            self._send_json(payload)
        elif path == "/api/chat/config":
            from config import settings
            self._send_json({
                "real_allowed": self.server.allow_real,
                "known_local_models": service.KNOWN_LOCAL_MODELS,
                "known_fireworks_models": service.KNOWN_FIREWORKS_MODELS,
                "default_mode": "hybrid",
                "default_models": {
                    "hybrid": settings.strong_model_name,
                    "remote": settings.strong_model_name,
                    "local": settings.local_model_name,
                },
            })
        elif path.startswith("/conversations"):
            self._chat_request("GET", path, parse_qs(parsed.query))
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
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
        self._chat_request("PATCH", urlsplit(self.path).path)

    def do_DELETE(self) -> None:
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
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
