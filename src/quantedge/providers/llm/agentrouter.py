"""AgentRouter reviewer using the official Anthropic SDK.

AgentRouter serves an Anthropic-compatible Messages API behind a Web Application
Firewall (WAF) that fingerprints HTTP clients.  Raw ``httpx`` calls are rejected
with HTTP 401 "unauthorized client detected" regardless of API key validity.
The official ``anthropic`` Python SDK carries the TLS fingerprint and headers the
WAF expects, so routing through it is the only reliable path.

Endpoint and model both come from configuration
-----------------------------------------------
``AGENTROUTER_BASE_URL`` and ``AGENTROUTER_MODEL`` override the defaults, so
pointing this at a different Anthropic-compatible gateway is a config change
rather than a code change.
"""

from __future__ import annotations

from typing import Any

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

__all__ = ["AgentRouterLLMProvider"]

log = get_logger(__name__)

# AgentRouter base URL is the *root* for the Anthropic SDK (it appends /v1/messages
# internally).  When the user sets AGENTROUTER_BASE_URL to "https://agentrouter.org/v1"
# we strip the trailing "/v1" so the SDK doesn't double it.
_DEFAULT_BASE_URL = "https://agentrouter.org"
_DEFAULT_MODEL = "claude-sonnet-4-5-20250514"
_TIMEOUT_SECONDS = 45.0

# Low but non-zero: the review is a judgement over fixed evidence, so there is
# no benefit in sampling widely, and a deterministic-leaning setting keeps two
# runs over the same context close enough to compare.
_TEMPERATURE = 0.2
_MAX_TOKENS = 1200


def _normalise_base_url(url: str) -> str:
    """Strip a trailing ``/v1`` so the Anthropic SDK can append its own."""
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


class AgentRouterLLMProvider(BaseLLMProvider):
    """Reviews a :class:`SignalContext` via the Anthropic SDK routed through AgentRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        settings = get_settings()
        resolved_model = model or settings.agentrouter_model or _DEFAULT_MODEL
        super().__init__("agentrouter", resolved_model)
        self._api_key = api_key or settings.secret(settings.agentrouter_api_key)
        raw_url = base_url or settings.agentrouter_base_url or _DEFAULT_BASE_URL
        self._base_url = _normalise_base_url(raw_url)

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def credentials_present(self) -> bool:
        return bool(self._api_key)

    def _get_client(self, timeout: float = _TIMEOUT_SECONDS) -> Any:
        """Create an Anthropic client pointed at AgentRouter."""
        import anthropic

        return anthropic.Anthropic(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=timeout,
        )

    def health(self) -> ProviderHealth:
        """Probe the endpoint with a minimal request.

        A configured key that the gateway rejects is a *worse* state than no key
        at all, because it looks configured.  ``DISABLED`` means nothing to try;
        ``ERROR`` means we tried and it did not work, with the reason attached.
        """
        if not self._api_key:
            return ProviderHealth(
                provider=self.provider_name,
                kind="llm",
                status=HealthStatus.DISABLED,
                enabled=False,
                credentials_present=False,
                missing_env=["AGENTROUTER_API_KEY"],
                message="AGENTROUTER_API_KEY is not configured; reviews are skipped",
            )

        try:
            client = self._get_client(timeout=15.0)
            response = client.messages.create(
                model=self.model_name,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            _ = response.content  # confirm we got a valid response
        except Exception as exc:  # noqa: BLE001 - a health probe reports any failure, not just ours
            err_msg = str(exc)
            # Truncate long error messages for the health report
            if len(err_msg) > 200:
                err_msg = err_msg[:200] + "..."
            return ProviderHealth(
                provider=self.provider_name,
                kind="llm",
                status=HealthStatus.ERROR,
                enabled=True,
                credentials_present=True,
                message=f"{self._base_url}: {type(exc).__name__} — {err_msg}",
            )

        return ProviderHealth(
            provider=self.provider_name,
            kind="llm",
            status=HealthStatus.OK,
            enabled=True,
            credentials_present=True,
            message=f"{self._base_url} reachable, model {self.model_name}",
        )

    def evaluate_signal_context(self, context: SignalContext) -> LLMSignalResponse:
        """Send the context for review and return the validated response.

        Raises rather than degrading: a caller that receives an exception knows
        no review happened, whereas a synthesised "review" would be silently
        trusted.  :mod:`quantedge.services.signal` catches this and reports the
        deterministic candidate on its own.
        """
        if not self._api_key:
            raise ProviderUnavailableError(
                self.provider_name, "AGENTROUTER_API_KEY is not configured"
            )

        text = self._call_model(context)
        payload = extract_json_object(text, provider=self.provider_name)

        # The pipeline knows these four better than the model does, and a model
        # that echoes them back wrongly should not be able to change them.
        payload["asset"] = context.symbol
        payload["horizon"] = context.horizon
        payload["calibrated_probability"] = None
        payload.setdefault("heuristic_score", context.heuristic_score)

        from quantedge.services.llm_review import validate_llm_response

        return validate_llm_response(payload, context)

    def _call_model(self, context: SignalContext) -> str:
        """Call the model via the Anthropic SDK and return the text response.

        Exceptions from the SDK are mapped onto the provider error hierarchy so
        the signal service can tell an outage (retry / skip) from a rejected
        credential (do not retry).
        """
        import anthropic

        client = self._get_client()

        try:
            response = client.messages.create(
                model=self.model_name,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_prompt(context)}],
            )
        except anthropic.AuthenticationError as exc:
            raise ProviderAuthError(
                self.provider_name,
                f"{self._base_url} rejected the configured credential: {exc}",
            ) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimitError(
                self.provider_name,
                retry_after_seconds=None,
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(
                self.provider_name, _TIMEOUT_SECONDS
            ) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderUnavailableError(
                self.provider_name,
                f"{self._base_url} returned HTTP {exc.status_code}: {exc.message}",
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderUnavailableError(
                self.provider_name,
                f"{self._base_url} unreachable: {type(exc).__name__}",
            ) from exc

        # Extract text content from the response
        if isinstance(response, str):
            text = response
        elif isinstance(response, dict):
            # If the response is a dictionary, extract the text content from the message
            content = response.get("content", [])
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text_parts = [
                    block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                    for block in content
                ]
                text = "".join(text_parts)
            else:
                text = str(response)
        else:
            text_parts = [
                block.text
                for block in getattr(response, "content", [])
                if hasattr(block, "text")
            ]
            text = "".join(text_parts)

        if not text.strip():
            raise ProviderBadResponseError(
                self.provider_name, "model returned an empty message"
            )

        log.info(
            "llm review completed",
            extra={
                "provider": self.provider_name,
                "model": self.model_name,
                "prompt_tokens": getattr(response.usage, "input_tokens", None),
                "completion_tokens": getattr(response.usage, "output_tokens", None),
            },
        )
        return text
