"""Workspace tool registry. External actions are kept out of automatic runs."""
import html
import http.client
from html.parser import HTMLParser
import ipaddress
import json
from pathlib import Path
import re
import socket
import subprocess
import urllib.request
from urllib.parse import urlencode, urlsplit, urljoin

from . import store, connections


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.skip:
            self.skip -= 1
        if tag in {"p", "div", "br", "li", "h1", "h2"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def public_url(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Use a public HTTP(S) URL")
    for result in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)):
        if not ipaddress.ip_address(result[4][0]).is_global:
            raise ValueError("Web reader accepts public addresses only")
    return url


def read_url(url):
    for _ in range(5):
        public_url(url)
        parsed = urlsplit(url)
        connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        connection = connection_cls(parsed.hostname, parsed.port, timeout=20)
        try:
            connection.connect()
            # Validate the connected peer before transmitting the request,
            # including after DNS changes between validation and connection.
            if not ipaddress.ip_address(connection.sock.getpeername()[0]).is_global:
                raise ValueError("Web reader cannot connect to private addresses")
            target = (parsed.path or "/") + ("?" + parsed.query if parsed.query else "")
            connection.request("GET", target, headers={"User-Agent": "NanaResearch/0.1"})
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308} and response.getheader("Location"):
                url = urljoin(url, response.getheader("Location"))
                continue
            if response.status != 200:
                raise RuntimeError(f"Source returned HTTP {response.status}")
            mime = response.getheader("Content-Type", "")
            if not any(x in mime for x in ("text/", "json", "xml")):
                raise ValueError("Web reader supports text/HTML sources; upload other documents separately")
            raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                raise ValueError("Source exceeds the 1 MB reading limit")
            parser = Text()
            parser.feed(raw.decode("utf-8", "replace"))
            return {"url": url, "content": re.sub(r"[ \t]+", " ", "".join(parser.parts))[:24000]}
        finally:
            connection.close()
    raise ValueError("Too many source redirects")


def search_web(query):
    endpoint = store.preferences().get("searxng_url", "")
    if not endpoint:
        raise ValueError("Configure a SearXNG search endpoint in Connections")
    if urlsplit(endpoint).scheme not in {"http", "https"}:
        raise ValueError("Invalid search endpoint")
    result = connections.http_json(endpoint.rstrip("/") + "/search?" + urlencode({"q": query, "format": "json"}))
    return [{"title": row.get("title"), "url": row.get("url"), "snippet": row.get("content", "")}
            for row in result.get("results", [])[:8]]


def file_path(name):
    root = (store.home() / "files").resolve()
    root.mkdir(exist_ok=True)
    target = (root / name).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("File must be inside Nana's workspace files directory")
    return target


TOOLS = {
    "search": "Search the web. arguments: {query: string}",
    "read_url": "Read a public web page. arguments: {url: string}",
    "list_items": "Search notes, tasks, documents, memories or drafts. arguments: {kind?: string, query?: string}",
    "read_item": "Read a workspace item. arguments: {id: string}",
    "save_item": "Create a note, task, document, memory or email draft. arguments: {kind, title, content, meta?: object}",
    "read_file": "Read a UTF-8 workspace file. arguments: {path: string}",
    "list_files": "List workspace files. arguments: {}",
    "gmail_search": "Read Gmail message summaries. arguments: {query: string}",
    "gmail_read": "Read a Gmail message. arguments: {id: string}",
    "calendar_list": "Read upcoming Google Calendar events. arguments: {}",
    "shell": "Run a command after the user reviews it. arguments: {command: string[]}",
    "mcp": "Call an explicitly configured MCP tool after user review. arguments: {server: string, tool: string, arguments: object}",
    "browser": "Read or interact with a public website in an isolated browser after user review. arguments: {url: string, action: read|click|fill, selector?: string, value?: string}",
}
REVIEW_TOOLS = {"shell", "mcp", "browser"}


def call(name, args, *, approved=False):
    if name not in TOOLS or not isinstance(args, dict):
        raise ValueError("Invalid tool call")
    if name in REVIEW_TOOLS and not approved:
        raise PermissionError("This tool needs user review")
    if name in {"shell", "mcp"} and store.USER.get() != "local":
        raise PermissionError("Host command and MCP execution are restricted to the owner account")
    if name == "search":
        return search_web(str(args["query"]))
    if name == "read_url":
        return read_url(str(args["url"]))
    if name == "list_items":
        return store.items(args.get("kind"), str(args.get("query", "")))
    if name == "read_item":
        return store.get(str(args["id"]))
    if name == "save_item":
        if args.get("kind") not in {"note", "task", "document", "memory", "draft"}:
            raise ValueError("Agent can create notes, tasks, documents, memories or drafts")
        return store.save(args)
    if name == "read_file":
        target = file_path(str(args["path"]))
        if target.stat().st_size > 500_000:
            raise ValueError("File too large")
        return {"path": args["path"], "content": target.read_text()}
    if name == "list_files":
        root = store.home() / "files"
        root.mkdir(exist_ok=True)
        return [p.name for p in root.iterdir() if p.is_file() and not p.is_symlink()]
    if name == "gmail_search":
        return connections.mail_list(str(args.get("query", "in:inbox")))
    if name == "gmail_read":
        return connections.mail_read(str(args["id"]))
    if name == "calendar_list":
        return connections.calendar_list()
    if name == "shell":
        cmd = args.get("command")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
            raise ValueError("Shell command must be a list of arguments")
        root = store.home() / "files"
        root.mkdir(exist_ok=True)
        import os
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "LANG", "TMPDIR"}}
        # cwd is a convenience, not a sandbox. UI makes this explicit at review.
        result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, timeout=20)
        return {"exit_code": result.returncode, "stdout": result.stdout[:32000].decode("utf-8", "replace"),
                "stderr": result.stderr[:32000].decode("utf-8", "replace")}
    if name == "mcp":
        from .mcp import Session
        servers = store.preferences().get("mcp_servers", {})
        config = servers.get(args["server"])
        if not config:
            raise ValueError("Unknown MCP server")
        with Session(config, store.home()) as session:
            return session.call("tools/call", {"name": args["tool"], "arguments": args.get("arguments", {})})
    if name == "browser":
        from .browser import action
        return action(args)
