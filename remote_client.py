"""Fireworks AI clients for the cheap and strong remotely hosted tiers.

Both clients use the same OpenAI-compatible endpoint and billing-aware retry
policy. The module keeps ``RemoteClient`` and ``resolve_remote_model``
compatibility shims for the older single-remote harness; new runtime code
uses ``CheapRemoteClient`` and ``StrongRemoteClient``.

No network call is made in mock mode. Read timeouts are deliberately not
retried because a completed generation may already have been billed.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Optional

from config import ROUTE_CHEAP, ROUTE_STRONG, settings
from schemas import Completion


class RemoteError(RuntimeError):
    """A Fireworks call failed in a way this client cannot recover from."""


def _model_key(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].strip().lower()


def _allowed_models() -> Optional[list[str]]:
    """Return the deployment allow-list, or None when it is not configured."""

    raw = settings.allowed_models.strip()

    if not raw:
        return None

    allowed = [
        model.strip()
        for model in raw.split(",")
        if model.strip()
    ]

    if not allowed:
        print(
            "ERROR: ALLOWED_MODELS is set but contains no model IDs",
            file=sys.stderr,
        )

        return []

    return allowed


def resolve_tier_model(
    configured: str,
    tier: str,
) -> Optional[str]:
    """Resolve one configured tier against ALLOWED_MODELS."""

    allowed = _allowed_models()

    if allowed is None:
        return configured

    if not allowed:
        return None

    by_key = {
        _model_key(model): model
        for model in allowed
    }

    resolved = by_key.get(
        _model_key(configured)
    )

    if resolved is None:
        print(
            f"ERROR: configured {tier} model "
            f"{configured!r} is not present "
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
        by_key.setdefault(
            _model_key(model),
            model,
        )

    candidates = [
        settings.remote_model_name
    ]

    candidates.extend(
        settings.remote_model_preference.split(",")
    )

    for candidate in candidates:
        chosen = by_key.get(
            _model_key(candidate)
        )

        if chosen:
            print(
                f"remote model: {chosen!r} "
                "(from ALLOWED_MODELS)",
                file=sys.stderr,
            )

            return chosen

    print(
        f"remote model: {allowed[0]!r} "
        "(first of ALLOWED_MODELS; "
        "no preference matched)",
        file=sys.stderr,
    )

    return allowed[0]


class _Transient(Exception):
    def __init__(
        self,
        message: str,
        retry_after: Optional[str] = None,
    ):
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

        self.model_name = resolve_tier_model(
            model_name,
            tier,
        )

        self.max_tokens = max_tokens

        self.api_key = (
            api_key
            or settings.fireworks_api_key
        )

        self.base_url = (
            base_url
            or settings.fireworks_base_url
        ).rstrip("/")

    # =========================================================
    # NORMAL NON-STREAMING GENERATION
    # =========================================================

    def generate(
        self,
        prompt: str,
    ) -> Completion:
        started = time.time()

        if settings.mock_mode:
            adjective = (
                "concise"
                if self.route == ROUTE_CHEAP
                else "detailed"
            )

            completion_tokens = (
                12
                if self.route == ROUTE_CHEAP
                else 24
            )

            return Completion(
                text=(
                    f"[mock-{self.tier}] "
                    f"{adjective} answer to: "
                    f"{prompt[:60]}"
                ),
                prompt_tokens=len(
                    prompt.split()
                ),
                completion_tokens=(
                    completion_tokens
                ),
                source=self.route,
                latency_s=(
                    time.time()
                    - started
                ),
                model_name=(
                    self.model_name
                    or "mock"
                ),
                provider="fireworks",
            )

        if not self.model_name:
            raise RemoteError(
                f"no usable {self.tier} model; "
                "check the tier model setting "
                "and ALLOWED_MODELS"
            )

        if not self.api_key:
            raise RemoteError(
                "FIREWORKS_API_KEY is not set. "
                "Export it or use --mock "
                "for offline tests."
            )

        import requests

        url = (
            f"{self.base_url}"
            "/chat/completions"
        )

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
        }

        messages = []

        if settings.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        settings.system_prompt
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }

        data = None

        for attempt in range(
            settings.max_retries + 1
        ):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=(
                        settings.connect_timeout_s,
                        settings.request_timeout_s,
                    ),
                )

                if (
                    response.status_code == 429
                    or response.status_code >= 500
                ):
                    raise _Transient(
                        f"HTTP "
                        f"{response.status_code}: "
                        f"{response.text[:200]}",
                        retry_after=(
                            response.headers.get(
                                "Retry-After"
                            )
                        ),
                    )

                response.raise_for_status()

                try:
                    data = response.json()

                except ValueError as err:
                    raise _Transient(
                        "unparseable response "
                        f"body: {err}"
                    )

                break

            except requests.HTTPError as err:
                raise RemoteError(
                    "non-retryable HTTP error "
                    f"for {self.tier} model "
                    f"{self.model_name!r}: "
                    f"{err}. Body: "
                    f"{response.text[:300]}"
                ) from err

            except (
                requests.exceptions.ReadTimeout
            ) as err:
                raise RemoteError(
                    f"{self.tier} read timeout "
                    f"after "
                    f"{settings.request_timeout_s}s "
                    "(not retried to avoid "
                    f"double billing): {err}"
                ) from err

            except (
                _Transient,
                requests.RequestException,
            ) as err:

                if (
                    attempt
                    == settings.max_retries
                ):
                    raise RemoteError(
                        f"{self.tier} call failed "
                        f"after "
                        f"{attempt + 1} attempts: "
                        f"{err}"
                    ) from err

                backoff = min(
                    30.0,
                    2.0 ** attempt,
                )

                retry_after = getattr(
                    err,
                    "retry_after",
                    None,
                )

                if retry_after:
                    try:
                        backoff = max(
                            backoff,
                            min(
                                30.0,
                                float(
                                    retry_after
                                ),
                            ),
                        )

                    except ValueError:
                        pass

                time.sleep(
                    backoff
                )

        choice = (
            data["choices"][0]
        )

        text = (
            (
                choice.get(
                    "message"
                )
                or {}
            ).get(
                "content"
            )
            or ""
        ).strip()

        usage = (
            data.get("usage")
            or {}
        )

        if not usage:
            print(
                f"WARNING: {self.tier} "
                "response omitted usage; "
                "token counts are estimated",
                file=sys.stderr,
            )

        prompt_tokens = int(
            usage.get(
                "prompt_tokens",
                max(
                    1,
                    len(prompt) // 4,
                ),
            )
        )

        completion_tokens = int(
            usage.get(
                "completion_tokens",
                max(
                    1,
                    len(text) // 4,
                ),
            )
        )

        return Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=(
                completion_tokens
            ),
            source=self.route,
            latency_s=(
                time.time()
                - started
            ),
            model_name=self.model_name,
            provider="fireworks",
        )

    # =========================================================
    # STREAMING GENERATION
    # =========================================================

    def stream_generate(
        self,
        prompt: str,
        stop_event=None,
    ):
        """
        Stream text from Fireworks.

        Yields:
            (chunk, None)
                while text is being generated

            (None, Completion)
                when generation finishes

        stop_event may be a threading.Event.
        """

        started = time.time()

        # -----------------------------------------------------
        # Mock mode
        # -----------------------------------------------------

        if settings.mock_mode:
            adjective = (
                "concise"
                if self.route == ROUTE_CHEAP
                else "detailed"
            )

            text = (
                f"[mock-{self.tier}] "
                f"{adjective} answer to: "
                f"{prompt[:60]}"
            )

            generated_parts = []

            for word in text.split():
                if (
                    stop_event is not None
                    and stop_event.is_set()
                ):
                    break

                chunk = (
                    word + " "
                )

                generated_parts.append(
                    chunk
                )

                yield (
                    chunk,
                    None,
                )

            final_text = "".join(
                generated_parts
            ).strip()

            completion = Completion(
                text=final_text,
                prompt_tokens=len(
                    prompt.split()
                ),
                completion_tokens=len(
                    final_text.split()
                ),
                source=self.route,
                latency_s=(
                    time.time()
                    - started
                ),
                model_name=(
                    self.model_name
                    or "mock"
                ),
                provider="fireworks",
            )

            yield (
                None,
                completion,
            )

            return

        # -----------------------------------------------------
        # Validate config
        # -----------------------------------------------------

        if not self.model_name:
            raise RemoteError(
                f"no usable "
                f"{self.tier} model; "
                "check model configuration"
            )

        if not self.api_key:
            raise RemoteError(
                "FIREWORKS_API_KEY "
                "is not set"
            )

        import requests

        url = (
            f"{self.base_url}"
            "/chat/completions"
        )

        headers = {
            "Authorization": (
                f"Bearer "
                f"{self.api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
        }

        messages = []

        if settings.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        settings.system_prompt
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "model": (
                self.model_name
            ),

            "messages": messages,

            "max_tokens": (
                self.max_tokens
            ),

            "temperature": 0,

            # Tell Fireworks to stream.
            "stream": True,

            # Ask for token usage in final event.
            "stream_options": {
                "include_usage": True,
            },
        }

        response = None

        # -----------------------------------------------------
        # Establish streaming connection
        # -----------------------------------------------------

        for attempt in range(
            settings.max_retries + 1
        ):
            try:
                response = requests.post(
                    url,

                    json=payload,

                    headers=headers,

                    stream=True,

                    timeout=(
                        settings
                        .connect_timeout_s,

                        settings
                        .request_timeout_s,
                    ),
                )

                if (
                    response.status_code
                    == 429
                    or response.status_code
                    >= 500
                ):
                    raise _Transient(
                        f"HTTP "
                        f"{response.status_code}: "
                        f"{response.text[:200]}",

                        retry_after=(
                            response.headers.get(
                                "Retry-After"
                            )
                        ),
                    )

                response.raise_for_status()

                break

            except requests.HTTPError as err:
                raise RemoteError(
                    "non-retryable HTTP "
                    f"error for "
                    f"{self.tier}: "
                    f"{err}"
                ) from err

            except (
                requests.exceptions
                .ReadTimeout
            ) as err:
                raise RemoteError(
                    f"{self.tier} "
                    "read timeout: "
                    f"{err}"
                ) from err

            except (
                _Transient,
                requests.RequestException,
            ) as err:

                if (
                    attempt
                    == settings.max_retries
                ):
                    raise RemoteError(
                        f"{self.tier} "
                        "stream failed after "
                        f"{attempt + 1} "
                        f"attempts: {err}"
                    ) from err

                backoff = min(
                    30.0,
                    2.0 ** attempt,
                )

                retry_after = getattr(
                    err,
                    "retry_after",
                    None,
                )

                if retry_after:
                    try:
                        backoff = max(
                            backoff,
                            min(
                                30.0,
                                float(
                                    retry_after
                                ),
                            ),
                        )

                    except ValueError:
                        pass

                time.sleep(
                    backoff
                )

        # -----------------------------------------------------
        # Read streamed chunks
        # -----------------------------------------------------

        text_parts = []
        usage = {}

        try:
            for raw_line in (
                response.iter_lines(
                    decode_unicode=True
                )
            ):
                # Stop button support.
                if (
                    stop_event is not None
                    and stop_event.is_set()
                ):
                    response.close()
                    break

                if not raw_line:
                    continue

                line = (
                    raw_line.strip()
                )

                if line.startswith(
                    "data:"
                ):
                    line = (
                        line[5:]
                        .strip()
                    )

                if not line:
                    continue

                if line == "[DONE]":
                    break

                try:
                    data = json.loads(
                        line
                    )

                except json.JSONDecodeError:
                    continue

                # Fireworks may include token usage
                # in the final stream object.
                if data.get("usage"):
                    usage = (
                        data["usage"]
                    )

                choices = (
                    data.get("choices")
                    or []
                )

                if not choices:
                    continue

                delta = (
                    choices[0]
                    .get(
                        "delta",
                        {}
                    )
                    .get(
                        "content"
                    )
                    or ""
                )

                if not delta:
                    continue

                text_parts.append(
                    delta
                )

                # Immediately return this piece
                # to the FastAPI layer.
                yield (
                    delta,
                    None,
                )

        except (
            requests.exceptions
            .ReadTimeout
        ) as err:
            raise RemoteError(
                f"{self.tier} "
                "streaming timeout: "
                f"{err}"
            ) from err

        finally:
            if response is not None:
                response.close()

        # -----------------------------------------------------
        # Build final Completion
        # -----------------------------------------------------

        text = "".join(
            text_parts
        ).strip()

        prompt_tokens = int(
            usage.get(
                "prompt_tokens",
                max(
                    1,
                    len(prompt) // 4,
                ),
            )
        )

        completion_tokens = int(
            usage.get(
                "completion_tokens",
                (
                    max(
                        1,
                        len(text) // 4,
                    )
                    if text
                    else 0
                ),
            )
        )

        completion = Completion(
            text=text,

            prompt_tokens=(
                prompt_tokens
            ),

            completion_tokens=(
                completion_tokens
            ),

            source=self.route,

            latency_s=(
                time.time()
                - started
            ),

            model_name=(
                self.model_name
            ),

            provider="fireworks",
        )

        # Signal that generation has finished.
        yield (
            None,
            completion,
        )


class CheapRemoteClient(
    FireworksClient
):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            tier="cheap",
            route=ROUTE_CHEAP,

            model_name=kwargs.pop(
                "model_name",
                settings
                .cheap_model_name,
            ),

            max_tokens=kwargs.pop(
                "max_tokens",
                settings
                .cheap_max_tokens,
            ),

            **kwargs,
        )


class StrongRemoteClient(
    FireworksClient
):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            tier="strong",
            route=ROUTE_STRONG,

            model_name=kwargs.pop(
                "model_name",
                settings
                .strong_model_name,
            ),

            max_tokens=kwargs.pop(
                "max_tokens",
                settings
                .strong_max_tokens,
            ),

            **kwargs,
        )


class RemoteClient(
    StrongRemoteClient
):
    """Backward-compatible name for the strong remote tier."""