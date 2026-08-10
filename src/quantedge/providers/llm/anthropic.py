"""Anthropic Messages API reviewer over a real HTTP call.

The `anthropic` SDK is not a dependency of this project, so the call is made
directly against the documented HTTP surface with httpx -- the same client every
market-data provider already uses. One less package to pin, and the request is
visible in the file rather than behind a wrapper.

Kept alongside the AgentRouter adapter rather than folded into it: the two wire
formats differ in three ways that matter (``x-api-key`` instead of Bearer, a
top-level ``system`` parameter instead of a system message, and a content-block
array instead of ``choices[0].message.content``). A single adapter with branches
on the provider name would hide those differences behind a flag.
"""

from __future__ import annotations

from typing import Any

import httpx

from quantedge.config import get_settings
from quantedge.contracts import (
    HealthStatus,
    LLMSignalResponse,
    ProviderHealth,
    SignalContext,
)
from quantedge.errors import (
    ProviderAuthError,
    ProviderBadResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from quantedge.logging import get_logger
from quantedge.providers.llm.base import BaseLLMProvider
from quantedge.providers.llm.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    extract_json_object,
)

__all__ = ["AnthropicLLMProvider"]

log = get_logger(__name__)

_BASE_URL = "https://api.anthropic.com/v1"
_API_VERSION = "2023-06-01"
_TIMEOUT_SECONDS = 45.0
_TEMPERATURE = 0.2
# Headroom above the ~700-token JSON reply. A reasoning model spends most of its
# budget before the first visible character and that spend is charged against
# this same ceiling, so a tight limit does not produce a short answer -- it
# produces no answer at all, returned as stop_reason="max_tokens" with an empty
# text block. That is exactly what 1200 did on the AgentRouter path, where it
# failed every review while a one-token health probe kept reporting "ok".
_MAX_TOKENS = 8000


class AnthropicLLMProvider(BaseLLMProvider):
    """Reviews a :class:`SignalContext` via the Anthropic Messages API."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        super().__init__("anthropic", model or settings.anthropic_model)
        self._api_key = api_key or settings.secret(settings.anthropic_api_key)

    @property
    def credentials_present(self) -> bool:
        return bool(self._api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key or "",
            "anthropic-version": _API_VERSION,
            "Content-Type": "application/json",
        }

    def health(self) -> ProviderHealth:
        """Probe with a one-token message rather than trusting the key's presence.

        There is no unauthenticated ping on this API, so the cheapest honest
        check is the smallest possible real request. Claiming OK because a string
        is set in ``.env`` would report a rejected key as healthy.
        """
        if not self._api_key:
            return ProviderHealth(
                provider=self.provider_name,
                kind="llm",
                status=HealthStatus.DISABLED,
                enabled=False,
                credentials_present=False,
                missing_env=["ANTHROPIC_API_KEY"],
                message="ANTHROPIC_API_KEY is not configured; reviews are skipped",
            )

        probe = {
            "model": self.model_name,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(f"{_BASE_URL}/messages", headers=self._headers(), json=probe)
        except httpx.HTTPError as exc:
            return ProviderHealth(
                provider=self.provider_name,
                kind="llm",
                status=HealthStatus.ERROR,
                enabled=True,
                credentials_present=True,
                message=f"api.anthropic.com unreachable: {type(exc).__name__}",
            )

        ok = response.status_code == httpx.codes.OK
        return ProviderHealth(
            provider=self.provider_name,
            kind="llm",
            status=HealthStatus.OK if ok else HealthStatus.ERROR,
            enabled=True,
            credentials_present=True,
            message=(
                f"Messages API reachable, model {self.model_name}"
                if ok
                else f"Messages API returned HTTP {response.status_code}"
            ),
        )

    def evaluate_signal_context(self, context: SignalContext) -> LLMSignalResponse:
        """Send the context for review and return the validated response."""
        if not self._api_key:
            raise ProviderUnavailableError(
                self.provider_name, "ANTHROPIC_API_KEY is not configured"
            )

        body = {
            "model": self.model_name,
            "max_tokens": _MAX_TOKENS,
            "temperature": _TEMPERATURE,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_user_prompt(context)}],
        }

        text = self._post_messages(body)
        payload = extract_json_object(text, provider=self.provider_name)
        payload["asset"] = context.symbol
        payload["horizon"] = context.horizon
        payload["calibrated_probability"] = None
        payload.setdefault("heuristic_score", context.heuristic_score)

        from quantedge.services.llm_review import validate_llm_response

        return validate_llm_response(payload, context)

    def _post_messages(self, body: dict[str, Any]) -> str:
        """POST to /v1/messages and return the concatenated text blocks."""
        try:
            with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
                response = client.post(f"{_BASE_URL}/messages", headers=self._headers(), json=body)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self.provider_name, _TIMEOUT_SECONDS) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                self.provider_name, f"api.anthropic.com unreachable: {type(exc).__name__}"
            ) from exc

        if response.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            raise ProviderAuthError(
                self.provider_name,
                f"the Messages API rejected the configured credential "
                f"(HTTP {response.status_code})",
            )
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            retry_after = response.headers.get("retry-after")
            raise ProviderRateLimitError(
                self.provider_name,
                retry_after_seconds=float(retry_after) if retry_after else None,
            )
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise ProviderUnavailableError(
                self.provider_name,
                f"the Messages API returned HTTP {response.status_code}",
            )

        try:
            data = response.json()
            blocks = data["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderBadResponseError(
                self.provider_name,
                f"unexpected messages payload: {type(exc).__name__}",
                sample=response.text[:200],
            ) from exc

        text = "".join(
            str(b.get("text", ""))
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        if not text.strip():
            # Name stop_reason and the block types present. Without them this
            # error is indistinguishable from a dozen unrelated causes, and the
            # one that actually occurred -- the whole budget spent on reasoning
            # before any text, reported as "max_tokens" -- is the one a raised
            # ceiling fixes in seconds once it is named.
            kinds = sorted({b.get("type", "?") for b in blocks if isinstance(b, dict)})
            raise ProviderBadResponseError(
                self.provider_name,
                f"model returned no text content (stop_reason="
                f"{data.get('stop_reason')!r}, blocks={kinds or 'none'})",
            )

        usage = data.get("usage") or {}
        log.info(
            "llm review completed",
            extra={
                "provider": self.provider_name,
                "model": self.model_name,
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
            },
        )
        return text
