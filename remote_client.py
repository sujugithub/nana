"""Fireworks AI clients for the cheap and strong remotely hosted tiers.

Both clients use the same OpenAI-compatible endpoint and billing-aware retry
policy. The module keeps ``RemoteClient`` and ``resolve_remote_model``
compatibility shims for the older single-remote harness; new runtime code
uses ``CheapRemoteClient`` and ``StrongRemoteClient``.

No network call is made in mock mode. Read timeouts are deliberately not
retried because a completed generation may already have been billed.
"""
from __future__ import annotations

import sys
import time
from typing import Optional

from config import ROUTE_CHEAP, ROUTE_STRONG, settings
from schemas import Completion
from generation_context import completion_messages, output_limit


def _tool_call_text(calls):
    """Preserve tool requests for a caller explicitly using fenced transport."""
    import json
    import re
    from generation_stream import TEXT_TOOL_PROTOCOL
    if not calls or not TEXT_TOOL_PROTOCOL.get():
        return ""
    blocks = []
    for call in calls:
        function = call.get("function") or {}
        name = function.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,99}", name):
            raise RemoteError("Provider returned an invalid tool name")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, ValueError):
            raise RemoteError("Provider returned incomplete tool arguments") from None
        if not isinstance(arguments, dict):
            raise RemoteError("Provider returned non-object tool arguments")
        encoded = json.dumps(arguments, ensure_ascii=False).replace("`", "\\u0060")
        blocks.append(f"```{name}\n{encoded}\n```")
    return "\n\n".join(blocks)


class RemoteError(RuntimeError):
    """A Fireworks call failed in a way this client cannot recover from."""


def _model_key(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].strip().lower()


def _allowed_models() -> Optional[list[str]]:
    """Return the deployment allow-list, or None when it is not configured."""
    raw = settings.allowed_models.strip()
    if not raw:
        return None
    allowed = [model.strip() for model in raw.split(",") if model.strip()]
    if not allowed:
        print(
            "ERROR: ALLOWED_MODELS is set but contains no model IDs",
            file=sys.stderr,
        )
        return []
    return allowed


def resolve_tier_model(configured: str, tier: str) -> Optional[str]:
    """Resolve one configured tier against ALLOWED_MODELS.

    Each requested tier must appear in a configured allow-list. Silently
    substituting another model would invalidate an artifact trained for a
    specific model pair.
    """
    allowed = _allowed_models()
    if allowed is None:
        return configured
    if not allowed:
        return None
    by_key = {_model_key(model): model for model in allowed}
    resolved = by_key.get(_model_key(configured))
    if resolved is None:
        print(
            f"ERROR: configured {tier} model {configured!r} is not present "
            "in ALLOWED_MODELS",
            file=sys.stderr,
        )
    return resolved


def resolve_remote_model() -> Optional[str]:
    """Legacy single-remote resolver retained for the old harness."""
    allowed = _allowed_models()
    if allowed is None:
        return settings.remote_model_name
    if not allowed:
        return None
    by_key = {}
    for model in allowed:
        by_key.setdefault(_model_key(model), model)
    candidates = [settings.remote_model_name]
    candidates.extend(settings.remote_model_preference.split(","))
    for candidate in candidates:
        chosen = by_key.get(_model_key(candidate))
        if chosen:
            print(f"remote model: {chosen!r} (from ALLOWED_MODELS)", file=sys.stderr)
            return chosen
    print(
        f"remote model: {allowed[0]!r} (first of ALLOWED_MODELS; no "
        "preference matched)",
        file=sys.stderr,
    )
    return allowed[0]


class _Transient(Exception):
    def __init__(self, message: str, retry_after: Optional[str] = None):
        super().__init__(message)
        self.retry_after = retry_after


class FireworksClient:
    """One configured Fireworks model tier."""

    def __init__(
        self,
        *,
        tier: str,
        route: str,
        model_name: str,
        max_tokens: int,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.tier = tier
        self.route = route
        self.model_name = resolve_tier_model(model_name, tier)
        self.max_tokens = max_tokens
        self.api_key = api_key or settings.fireworks_api_key
        self.base_url = (base_url or settings.fireworks_base_url).rstrip("/")

    def generate(self, prompt: str) -> Completion:
        started = time.time()
        from generation_stream import CURRENT
        if CURRENT.get() is not None and not settings.mock_mode:
            return self._stream(prompt, CURRENT.get())
        if settings.mock_mode:
            adjective = "concise" if self.route == ROUTE_CHEAP else "detailed"
            completion_tokens = 12 if self.route == ROUTE_CHEAP else 24
            return Completion(
                text=f"[mock-{self.tier}] {adjective} answer to: {prompt[:60]}",
                prompt_tokens=len(prompt.split()),
                completion_tokens=completion_tokens,
                source=self.route,
                latency_s=time.time() - started,
                model_name=self.model_name or "mock",
                provider="fireworks",
            )

        if not self.model_name:
            raise RemoteError(
                f"no usable {self.tier} model; check the tier model setting "
                "and ALLOWED_MODELS"
            )
        if not self.api_key:
            raise RemoteError(
                "FIREWORKS_API_KEY is not set. Export it or use --mock for "
                "offline tests."
            )

        import requests  # lazy: mock mode remains stdlib-only

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = completion_messages(prompt, settings.system_prompt)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": output_limit(self.max_tokens),
            "temperature": 0,
        }

        data = None
        for attempt in range(settings.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=(settings.connect_timeout_s, settings.request_timeout_s),
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise _Transient(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        retry_after=response.headers.get("Retry-After"),
                    )
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError as err:
                    raise _Transient(f"unparseable response body: {err}")
                break
            except requests.HTTPError as err:
                raise RemoteError(
                    f"non-retryable HTTP error for {self.tier} model "
                    f"{self.model_name!r}: {err}. Body: {response.text[:300]}"
                ) from err
            except requests.exceptions.ReadTimeout as err:
                raise RemoteError(
                    f"{self.tier} read timeout after {settings.request_timeout_s}s "
                    f"(not retried to avoid double billing): {err}"
                ) from err
            except (_Transient, requests.RequestException) as err:
                if attempt == settings.max_retries:
                    raise RemoteError(
                        f"{self.tier} call failed after {attempt + 1} attempts: {err}"
                    ) from err
                backoff = min(30.0, 2.0 ** attempt)
                retry_after = getattr(err, "retry_after", None)
                if retry_after:
                    try:
                        backoff = max(backoff, min(30.0, float(retry_after)))
                    except ValueError:
                        pass
                time.sleep(backoff)

        choice = data["choices"][0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        tool_text = _tool_call_text((choice.get("message") or {}).get("tool_calls"))
        if tool_text:
            text = "\n\n".join(part for part in (text, tool_text) if part)
        usage = data.get("usage") or {}
        if not usage:
            print(
                f"WARNING: {self.tier} response omitted usage; token counts "
                "are estimated",
                file=sys.stderr,
            )
        prompt_tokens = int(usage.get("prompt_tokens", max(1, len(prompt) // 4)))
        completion_tokens = int(
            usage.get("completion_tokens", max(1, len(text) // 4))
        )
        return Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            source=self.route,
            latency_s=time.time() - started,
            model_name=self.model_name,
            provider="fireworks",
        )

    def _stream(self, prompt, stream):
        import json
        import requests
        if not self.model_name or not self.api_key:
            raise RemoteError("Configure a permitted Fireworks model and API key")
        started = time.monotonic()
        messages = completion_messages(prompt, settings.system_prompt)
        payload = {"model": self.model_name, "messages": messages, "max_tokens": output_limit(self.max_tokens),
                   "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        stream.start("fireworks", self.model_name)
        pieces, usage, tool_calls = [], {}, {}
        try:
            with requests.post(self.base_url + "/chat/completions", json=payload,
                               headers={"Authorization": "Bearer " + self.api_key}, stream=True,
                               timeout=(settings.connect_timeout_s, settings.request_timeout_s)) as response:
                if response.status_code != 200:
                    raise RemoteError(f"Fireworks streaming returned HTTP {response.status_code}")
                for line in response.iter_lines(chunk_size=1):
                    stream.check()
                    if not line.startswith(b"data: "):
                        continue
                    raw = line[6:]
                    if raw == b"[DONE]":
                        break
                    data = json.loads(raw)
                    if data.get("error"):
                        raise RemoteError("Fireworks reported a streaming error")
                    if data.get("usage"):
                        usage = data["usage"]
                        stream.usage(usage)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        for call in delta.get("tool_calls") or []:
                            index = call.get("index", 0)
                            target = tool_calls.setdefault(index, {"function": {"name": "", "arguments": ""}})
                            function = call.get("function") or {}
                            for key in ("name", "arguments"):
                                target["function"][key] += function.get(key) or ""
                        text = delta.get("content") or ""
                        if text:
                            pieces.append(text)
                            stream.delta(text)
        except requests.RequestException:
            raise RemoteError("Fireworks streaming connection failed; not retried to avoid duplicate billing") from None
        tool_text = _tool_call_text([tool_calls[index] for index in sorted(tool_calls)])
        if tool_text:
            suffix = ("\n\n" if pieces else "") + tool_text
            pieces.append(suffix)
            stream.delta(suffix)
            stream.attempts[-1]["native_tool_calls_converted"] = len(tool_calls)
        text = "".join(pieces).strip()
        return Completion(text=text, prompt_tokens=int(usage.get("prompt_tokens", max(1, len(prompt)//4))),
                          completion_tokens=int(usage.get("completion_tokens", max(1, len(text)//4))),
                          source=self.route, latency_s=time.monotonic()-started,
                          model_name=self.model_name, provider="fireworks")


class CheapRemoteClient(FireworksClient):
    def __init__(self, **kwargs):
        super().__init__(
            tier="cheap",
            route=ROUTE_CHEAP,
            model_name=kwargs.pop("model_name", settings.cheap_model_name),
            max_tokens=kwargs.pop("max_tokens", settings.cheap_max_tokens),
            **kwargs,
        )


class StrongRemoteClient(FireworksClient):
    def __init__(self, **kwargs):
        super().__init__(
            tier="strong",
            route=ROUTE_STRONG,
            model_name=kwargs.pop("model_name", settings.strong_model_name),
            max_tokens=kwargs.pop("max_tokens", settings.strong_max_tokens),
            **kwargs,
        )


class RemoteClient(StrongRemoteClient):
    """Backward-compatible name for the strong remote tier."""
