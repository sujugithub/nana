"""Fireworks AI client (OpenAI-compatible /chat/completions endpoint).

Design notes:
- Plain `requests` instead of an SDK: one fewer dependency, the exact payload
  is visible in this file (easy to debug live), and the endpoint is
  OpenAI-compatible anyway. Import is lazy so mock mode needs no deps at all.
- Token counts come from the API's `usage` field — the authoritative number
  for the billable side of the score. If a proxy/gateway strips `usage`, we
  warn loudly and estimate rather than silently logging 0.
- Retry policy is billing-aware:
    * connect errors, 429, 5xx, dropped/garbled bodies → retry with backoff
      (the generation was never completed, so retrying cannot double-bill);
      429 honors the Retry-After header.
    * READ timeouts are NOT retried: the server may have finished and billed
      the generation, so a retry would pay for the same answer twice. The
      caller (main.run_task) falls back to a local answer instead. Size
      REQUEST_TIMEOUT_S to comfortably cover remote_max_tokens of generation.
    * other 4xx (bad key, bad model name) fail fast — retrying a config
      error just burns clock during the scoring run.
- temperature=0: deterministic → reproducible accuracy, no flaky reruns.
"""
from __future__ import annotations

import sys
import time
from typing import Optional

from config import ROUTE_REMOTE, settings
from schemas import Completion


class RemoteError(RuntimeError):
    """Remote call failed in a way this client cannot recover from."""


class _Transient(Exception):
    """Internal marker for retryable HTTP statuses."""

    def __init__(self, message: str, retry_after: Optional[str] = None):
        super().__init__(message)
        self.retry_after = retry_after


class RemoteClient:
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name or settings.remote_model_name
        self.api_key = api_key or settings.fireworks_api_key
        self.base_url = (base_url or settings.fireworks_base_url).rstrip("/")

    def generate(self, prompt: str) -> Completion:
        started = time.time()

        if settings.mock_mode:
            text = f"[mock-remote] detailed answer to: {prompt[:60]}"
            return Completion(
                text=text,
                prompt_tokens=len(prompt.split()),  # fake but deterministic
                completion_tokens=24,
                source=ROUTE_REMOTE,
                latency_s=time.time() - started,
            )

        if not self.api_key:
            raise RemoteError(
                "FIREWORKS_API_KEY is not set. Export it (or use docker run "
                "--env-file .env). For offline wiring tests use AGENT_MOCK=1."
            )

        import requests  # lazy: mock mode must run on stdlib alone

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": settings.remote_max_tokens,
            "temperature": 0,
        }

        data = None
        for attempt in range(settings.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    # (connect, read): a slow handshake fails fast and is safe
                    # to retry; the read timeout must cover full generation.
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
                    # 200 with truncated/garbled body (proxy or LB reset):
                    # generation state unknown but response unusable — retry.
                    raise _Transient(f"unparseable response body: {err}")
                break
            except requests.HTTPError as err:
                # Non-retryable 4xx: almost always a wrong model name or key.
                raise RemoteError(
                    f"non-retryable HTTP error: {err} — check REMOTE_MODEL_NAME "
                    f"and FIREWORKS_API_KEY. Body: {response.text[:300]}"
                ) from err
            except requests.exceptions.ReadTimeout as err:
                # Do NOT retry: the server may have completed and billed the
                # generation. Fail the call; run_task falls back locally.
                raise RemoteError(
                    f"read timeout after {settings.request_timeout_s}s (not "
                    f"retried to avoid double-billing a completed generation; "
                    f"raise REQUEST_TIMEOUT_S if this recurs): {err}"
                ) from err
            except (_Transient, requests.RequestException) as err:
                if attempt == settings.max_retries:
                    raise RemoteError(
                        f"remote call failed after {attempt + 1} attempts: {err}"
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

        usage = data.get("usage") or {}
        if not usage:
            # Never silently record 0 for real spend — it would poison the
            # threshold calibration. Estimate at ~4 chars/token and say so.
            print(
                "WARNING: response contained no 'usage' field; billable "
                "tokens are ESTIMATED for this call",
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
            source=ROUTE_REMOTE,
            latency_s=time.time() - started,
        )
