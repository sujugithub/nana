"""Bounded, newline-delimited MCP stdio sessions with explicit server configs."""
import json
import os
import selectors
import subprocess
import time


class Session:
    def __new__(cls, config, cwd):
        if config.get("url"):
            return HTTPSession(config)
        return super().__new__(cls)
    def __init__(self, config, cwd):
        command = config.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError("MCP command must be a non-empty list of arguments")
        env = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME", "LANG", "TMPDIR"}}
        self.process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.sequence = 0
        self.buffer = b""
        try:
            self.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                     "clientInfo": {"name": "nana", "version": "0.1"}})
            self.write({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except Exception:
            self.close()
            raise

    def write(self, data):
        self.process.stdin.write(json.dumps(data).encode() + b"\n")
        self.process.stdin.flush()

    def call(self, method, params=None):
        self.sequence += 1
        request_id = self.sequence
        self.write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        deadline = time.monotonic() + 20
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                while b"\n" in self.buffer:
                    line, self.buffer = self.buffer.split(b"\n", 1)
                    data = json.loads(line)
                    if data.get("id") == request_id and ("result" in data or "error" in data):
                        if "error" in data:
                            raise RuntimeError("MCP server rejected the request")
                        return data["result"]
                    if "method" in data and "id" in data:
                        self.write({"jsonrpc": "2.0", "id": data["id"],
                                    "error": {"code": -32601, "message": "Client method unsupported"}})
                if not selector.select(max(0, deadline - time.monotonic())):
                    break
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("MCP server closed its output")
                self.buffer += chunk
                if len(self.buffer) > 2_000_000:
                    raise RuntimeError("MCP response too large")
        raise TimeoutError("MCP server timed out")

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout):
            pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class HTTPSession:
    """Streamable HTTP JSON/SSE responses; no browser or stdio process."""
    def __init__(self, config):
        from urllib.parse import urlsplit
        self.url = config["url"]
        if urlsplit(self.url).scheme not in {"http", "https"}:
            raise ValueError("MCP URL must be HTTP(S)")
        self.session_id = None
        self.sequence = 0
        self.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                 "clientInfo": {"name": "nana", "version": "0.1"}})
        self.call("notifications/initialized", notification=True)

    def call(self, method, params=None, notification=False):
        import urllib.request
        self.sequence += 1
        request_id = self.sequence
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notification:
            payload["id"] = request_id
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": "2025-06-18"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            self.session_id = response.headers.get("Mcp-Session-Id", self.session_id)
            if notification or response.status == 202:
                return {}
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                total = 0
                for line in response:
                    total += len(line)
                    if total > 2_000_000:
                        raise ValueError("MCP response too large")
                    if not line.startswith(b"data:"):
                        continue
                    data = json.loads(line[5:].strip())
                    if data.get("id") == request_id:
                        break
                else:
                    raise RuntimeError("MCP response ended without a result")
            else:
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ValueError("MCP response too large")
                data = json.loads(raw)
        if "error" in data:
            raise RuntimeError("MCP server rejected the request")
        if data.get("id") != request_id:
            raise RuntimeError("Unexpected MCP response ID")
        return data.get("result", {})

    def close(self):
        if self.session_id:
            import urllib.request
            try:
                urllib.request.urlopen(urllib.request.Request(self.url, method="DELETE", headers={"Mcp-Session-Id": self.session_id}), timeout=5).close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
