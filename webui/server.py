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
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

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
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_static("index.html", "text/html; charset=utf-8")
        elif path == "/api/config":
            payload = service.frontend_config()
            payload["real_allowed"] = self.server.allow_real
            self._send_json(payload)
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
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
