"""Explicit, account-scoped web search providers with bounded responses."""
import json
import re
import ssl
import urllib.error
import urllib.request
from urllib.parse import urlencode, urlsplit

import certifi

from . import store, connections

EXA_URL = "https://mcp.exa.ai/mcp"


def provider(prefs=None):
    prefs = store.preferences() if prefs is None else prefs
    return prefs.get("search_provider", "searxng" if prefs.get("searxng_url") else "off")


def public_link(url):
    parsed = urlsplit(url)
    return (parsed.scheme in {"http", "https"} and bool(parsed.hostname)
            and not parsed.username and not parsed.password and len(url) <= 2048)


def parse_exa(body):
    """Exa's read-only MCP call can return JSON or SSE-wrapped JSON."""
    packets = [json.loads(body)] if body.lstrip().startswith("{") else [
        json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")
    ]
    packet = next((p for p in packets if isinstance(p, dict) and p.get("id") == 1), None)
    if not packet or packet.get("error") or packet.get("result", {}).get("isError"):
        raise RuntimeError("Exa search failed; no verified search result was returned")
    text = "\n".join(p.get("text", "") for p in packet.get("result", {}).get("content", [])
                     if isinstance(p, dict) and p.get("type") == "text")
    results = []
    for block in re.split(r"(?m)^Title: ", text)[1:]:
        title, _, rest = block.partition("\n")
        match = re.search(r"(?m)^URL: (https?://\S+)\s*$", rest)
        if not match or not public_link(match[1]):
            continue
        snippet = re.split(r"(?m)^(?:Highlights|Text|Content):\s*", rest, maxsplit=1)
        results.append({"title": title[:300], "url": match[1],
                        "snippet": snippet[-1][:2000] if len(snippet) > 1 else rest[:2000]})
        if len(results) == 3:
            break
    if not results:
        raise RuntimeError("Exa search returned no usable source links; try a more specific query")
    return results


def exa_search(query):
    from generation_stream import CURRENT
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "web_search_exa", "arguments": {"query": query, "numResults": 3,
            "objective": "Find relevant primary sources and extract factual excerpts that answer the query."}}}
    req = urllib.request.Request(EXA_URL, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
        "User-Agent": "NanaResearch/0.1"})
    try:
        timeout = CURRENT.get().remaining(30) if CURRENT.get() else 30
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=ssl.create_default_context(cafile=certifi.where())) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise RuntimeError("Exa search response exceeded the reading limit")
        return parse_exa(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        if code == 429:
            raise RuntimeError("Exa search is rate limited. Try later or choose another search provider in Connections.") from None
        raise RuntimeError(f"Exa search returned HTTP {code}; check the search connection") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Exa search connection failed or timed out; check your network") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise RuntimeError("Exa search returned an unreadable response") from None


def search_web(query):
    if not isinstance(query, str) or not query.strip() or len(query) > 300:
        raise ValueError("Use a web search query of 1–300 characters")
    prefs = store.preferences()
    chosen = provider(prefs)
    if chosen == "exa":
        return exa_search(query.strip())
    if chosen != "searxng":
        raise ValueError("Choose a web search provider in Connections, or provide a public source URL")
    endpoint = prefs.get("searxng_url", "")
    if not endpoint or urlsplit(endpoint).scheme not in {"http", "https"}:
        raise ValueError("Configure a SearXNG URL with its JSON API enabled in Connections")
    result = connections.http_json(endpoint.rstrip("/") + "/search?" + urlencode({"q": query, "format": "json"}))
    return [{"title": str(row.get("title", ""))[:300], "url": row["url"],
             "snippet": str(row.get("content", ""))[:2000]}
            for row in result.get("results", [])[:8]
            if isinstance(row, dict) and isinstance(row.get("url"), str) and public_link(row["url"])]
