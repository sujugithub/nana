"""Experimental local intent router; no downloads and no remote router calls.

Prompt structure follows katanemo/Arch-Router-1.5B's model card. Route names
are Nana policy, not model-quality predictions. See docs/ARCH_ROUTER_NOTICE.md.
"""
import ast
import json
import sys
import time
from urllib.parse import urlsplit

from config import ROUTE_LOCAL, ROUTE_REMOTE, settings
from generation_context import MESSAGES
from generation_stream import CURRENT
from router import Router, RoutingDecision


ROUTES = [
    {"name": "local", "description": "Greetings, casual conversation, simple facts, translation, short writing, and summaries of supplied text."},
    {"name": "strong", "description": "Complex coding or debugging, difficult maths, multi-step reasoning, or specialist analysis."},
]


def _conversation(task):
    """Use recent intent with a hard character bound; never replay all history."""
    turns = MESSAGES.get()
    if not turns:
        return [{"role": "user", "content": task.prompt[-2000:]}]
    recent, remaining = [], 2000
    for turn in reversed(turns):
        if turn.get("role") not in {"user", "assistant"}:
            continue
        content = str(turn.get("content", ""))[-remaining:]
        recent.insert(0, {"role": turn["role"], "content": content})
        remaining -= len(content)
        if len(recent) == 3 or remaining <= 0:
            break
    return recent or [{"role": "user", "content": task.prompt[-2000:]}]


class ArchRouter(Router):
    def decide(self, task):
        stream = CURRENT.get()
        if stream:
            stream.check()
        if settings.mock_mode:
            decision = super().decide(task)
            decision.router_kind = "arch-preview"
            decision.reason = "Preview: heuristic simulation; Arch-Router was not called. " + decision.reason
            return decision
        started = time.monotonic()
        try:
            if urlsplit(settings.ollama_base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("Arch-Router requires a loopback Ollama address")
            prompt = (
                "Find the route matching the latest user intent. Route descriptions:\n<routes>\n"
                + json.dumps(ROUTES) + "\n</routes>\n<conversation>\n"
                + json.dumps(_conversation(task)) + "\n</conversation>\n"
                'Return only {"route": "local"} or {"route": "strong"}. '
                'If neither route fits, return {"route": "other"}.'
            )
            from ollama_client import request
            result = request("/api/chat", {
                "model": settings.arch_router_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # Release router residency before loading the answer model.
                "keep_alive": 0,
                "options": {"temperature": 0, "num_predict": 32, "num_ctx": 2048},
            }, timeout=stream.remaining(15) if stream else 15)
            if not isinstance(result, dict) or not isinstance(result.get("message"), dict):
                raise ValueError("Invalid router response")
            raw = result["message"].get("content")
            if not isinstance(raw, str) or result.get("done") is False:
                raise ValueError("Incomplete router response")
            raw = raw.strip()
            if raw.startswith("```") and raw.endswith("```"):
                raw = "\n".join(raw.splitlines()[1:-1]).strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(raw)
            route = parsed.get("route") if isinstance(parsed, dict) else None
            if route not in {"local", "strong"}:
                raise ValueError("Arch-Router returned no supported route")
        except (RuntimeError, ValueError, SyntaxError, TypeError, KeyError, OSError):
            # Keep user/provider content out of warning strings and logs.
            decision = super().decide(task)
            decision.router_kind = "heuristic-fallback"
            decision.reason = "Arch-Router unavailable or invalid; used heuristic fallback. " + decision.reason
            decision.signals["router_latency_s"] = time.monotonic() - started
            print("WARNING: Arch-Router unavailable or invalid; used heuristic fallback.", file=sys.stderr)
            if stream:
                stream.check()
            return decision
        if stream:
            stream.check()
        return RoutingDecision(
            target=ROUTE_LOCAL if route == "local" else ROUTE_REMOTE,
            confidence=1.0 if route == "local" else 0.0,
            signals={"route": route, "router_latency_s": time.monotonic() - started,
                     "router_prompt_tokens": result.get("prompt_eval_count"),
                     "router_completion_tokens": result.get("eval_count")},
            reason=f"Arch-Router intent route: {route}; experimental mapping, not a calibrated quality probability.",
            router_kind="arch",
        )
