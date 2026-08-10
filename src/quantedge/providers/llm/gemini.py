"""Gemini generateContent API reviewer over a raw HTTP call.

No Gemini SDK is used. Every request is a direct POST to the
``v1beta/models/{model}:generateContent`` endpoint via ``httpx``, consistent
with the project's policy of avoiding heavy LLM SDKs and keeping the wire
format visible in the source.

The request body uses ``responseMimeType: "application/json"`` inside
``generationConfig`` so the model is instructed to emit a bare JSON object
rather than prose-wrapped output. The ``extract_json_object`` parser is still
applied as a safety net, because even with the MIME hint some models emit a
markdown fence.

HTTP status mapping
-------------------
- 400  Bad request / invalid API key format  → ProviderBadResponseError
- 401  Invalid or expired API key            → ProviderAuthError
- 403  Key lacks permission for this model   → ProviderAuthError
- 429  Quota or rate limit exceeded          → ProviderRateLimitError
- 503  Service temporarily unavailable       → ProviderUnavailableError
- Any other 4xx / 5xx                        → ProviderUnavailableError
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

__all__ = ["GeminiLLMProvider"]

log = get_logger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_TIMEOUT_SECONDS = 45.0
_TEMPERATURE = 0.2
_MAX_OUTPUT_TOKENS = 1200


class GeminiLLMProvider(BaseLLMProvider):
    """Reviews a :class:`SignalContext` via the Google Gemini generateContent API.

    Uses a raw ``httpx`` POST — no Google AI SDK dependency. The API key is
    supplied as a query parameter (``?key=…``) as documented; it is redacted
    from all log output by :func:`quantedge.logging.register_secret` at
    settings load time.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        settings = get_settings()
        super().__init__("gemini", model or settings.gemini_model)
        self._api_key = api_key or settings.secret(settings.gemini_api_key)

    @property
    def credentials_present(self) -> bool:
        return bool(self._api_key)

    def _endpoint(self, action: str = "generateContent") -> str:
        """Build the full endpoint URL with the API key as a query parameter."""
        return f"{_BASE_URL}/{self.model_name}:{action}?key={self._api_key}"

    def health(self) -> ProviderHealth:
        """Probe with a minimal 1-token request to verify API key validity.

        There is no unauthenticated ping on this API, so the cheapest honest
        check is a real request with ``maxOutputTokens: 1``. Claiming OK
        because a string is set in ``.env`` would silently mask a rejected key.
        """
        if not self._api_key:
            return ProviderHealth(
                provider=self.provider_name,
                kind="llm",
                status=HealthStatus.DISABLED,
                enabled=False,
                credentials_present=False,
                missing_env=["GEMINI_API_KEY"],
                message="GEMINI_API_KEY is not configured; reviews are skipped",
            )

        probe_body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
            "generationConfig": {"maxOutputTokens": 1},
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(self._endpoint(), json=probe_body)
        except httpx.HTTPError as exc:
            return ProviderHealth(
                provider=self.provider_name,
                kind="llm",
                status=HealthStatus.ERROR,
                enabled=True,
                credentials_present=True,
                message=(
                    f"generativelanguage.googleapis.com unreachable: "
                    f"{type(exc).__name__}"
                ),
            )

        ok = response.status_code == httpx.codes.OK
        return ProviderHealth(
            provider=self.provider_name,
            kind="llm",
            status=HealthStatus.OK if ok else HealthStatus.ERROR,
            enabled=True,
            credentials_present=True,
            message=(
                f"generateContent API reachable, model {self.model_name}"
                if ok
                else f"generateContent API returned HTTP {response.status_code}"
            ),
        )

    def evaluate_signal_context(self, context: SignalContext) -> LLMSignalResponse:
        """Send the context for review and return the validated response.

        Raises rather than degrading: a caller that receives an exception knows
        no review happened, whereas a synthesised result would be silently
        trusted. :mod:`quantedge.services.signal` catches this and reports the
        deterministic candidate on its own.
        """
        if not self._api_key:
            raise ProviderUnavailableError(
                self.provider_name, "GEMINI_API_KEY is not configured"
            )

        text = self._post_generate_content(context)
        payload = extract_json_object(text, provider=self.provider_name)

        # The pipeline knows these four values better than the model does; a
        # model that echoes them back wrongly must not be able to change them.
        payload["asset"] = context.symbol
        payload["horizon"] = context.horizon
        payload["calibrated_probability"] = None
        payload.setdefault("heuristic_score", context.heuristic_score)

        from quantedge.services.llm_review import validate_llm_response

        return validate_llm_response(payload, context)

    def _post_generate_content(self, context: SignalContext) -> str:
        """POST to generateContent and return the first candidate's text.

        Builds the Gemini-native request body:
        - ``systemInstruction`` carries the immutable review prompt.
        - ``contents`` carries the serialised :class:`SignalContext`.
        - ``generationConfig`` pins temperature, token budget, and forces
          ``application/json`` as the response MIME type so the model is
          explicitly instructed to output a bare JSON object.
        """
        body: dict[str, Any] = {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": build_user_prompt(context)}],
                }
            ],
            "generationConfig": {
                "temperature": _TEMPERATURE,
                "maxOutputTokens": _MAX_OUTPUT_TOKENS,
                "responseMimeType": "application/json",
            },
        }

        try:
            with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
                response = client.post(self._endpoint(), json=body)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self.provider_name, _TIMEOUT_SECONDS) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                self.provider_name,
                f"generativelanguage.googleapis.com unreachable: {type(exc).__name__}",
            ) from exc

        self._raise_for_status(response)

        # Extract text from the first candidate's first part.
        try:
            data = response.json()
            text: str = data["candidates"][0]["content"]["parts"][0]["text"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderBadResponseError(
                self.provider_name,
                f"unexpected generateContent payload: {type(exc).__name__}",
                sample=response.text[:200],
            ) from exc

        if not text.strip():
            raise ProviderBadResponseError(
                self.provider_name,
                "model returned no text content",
            )

        # Log token usage from usageMetadata (present on successful responses).
        usage = data.get("usageMetadata") or {}
        log.info(
            "llm review completed",
            extra={
                "provider": self.provider_name,
                "model": self.model_name,
                "prompt_tokens": usage.get("promptTokenCount"),
                "completion_tokens": usage.get("candidatesTokenCount"),
            },
        )
        return text

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map Gemini HTTP status codes onto the provider error hierarchy.

        Called after every non-network response. Only the listed codes get a
        typed error; anything else above 400 becomes an unavailability error so
        future Gemini status codes degrade gracefully without a code change.
        """
        code = response.status_code

        if code == httpx.codes.OK:
            return

        if code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            raise ProviderAuthError(
                self.provider_name,
                f"the Gemini API rejected the configured credential "
                f"(HTTP {code})",
            )

        if code == httpx.codes.TOO_MANY_REQUESTS:
            retry_after = response.headers.get("retry-after")
            raise ProviderRateLimitError(
                self.provider_name,
                retry_after_seconds=float(retry_after) if retry_after else None,
            )

        if code == httpx.codes.BAD_REQUEST:
            raise ProviderBadResponseError(
                self.provider_name,
                "the Gemini API returned HTTP 400 (bad request); "
                "check model name and request body",
                sample=response.text[:200],
            )

        if code == httpx.codes.SERVICE_UNAVAILABLE:
            raise ProviderUnavailableError(
                self.provider_name,
                "the Gemini API is temporarily unavailable (HTTP 503)",
            )

        # Catch-all for any other 4xx / 5xx.
        if code >= httpx.codes.BAD_REQUEST:
            raise ProviderUnavailableError(
                self.provider_name,
                f"the Gemini API returned HTTP {code}",
            )
