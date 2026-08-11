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
_DEFAULT_MODEL = "claude-opus-5"

# Generous, because a reasoning model spends most of its budget before the first
# visible character and the whole call is one round trip we either complete or
# discard. A short timeout here does not fail fast, it fails silently: the review
# is dropped and the deterministic candidate ships unreviewed.
#
# This is the default for a long-lived process. Under a host that enforces its
# own wall clock -- serverless -- the host wins, and losing to the host is worse:
# it kills the request mid-flight and returns nothing, where expiring here still
# returns the deterministic answer. Such deployments set ``LLM_TIMEOUT_SECONDS``
# below the host ceiling, less the time the scan already spent.
_DEFAULT_TIMEOUT_SECONDS = 180.0

# Low but non-zero: the review is a judgement over fixed evidence, so there is
# no benefit in sampling widely, and a deterministic-leaning setting keeps two
# runs over the same context close enough to compare.
_TEMPERATURE = 0.2

# The reply itself is ~700 tokens of JSON. The rest of this budget is headroom
# for reasoning: models in this family emit a `thinking` block before any text,
# and that block is charged against the same ceiling. At 1200 the entire budget
# went to reasoning, the response came back `stop_reason="max_tokens"` carrying a
# thinking block and no text at all, and every review failed as "empty message"
# while the health probe -- one token, no system prompt, no reasoning -- passed.
# The gateway was never the problem; the ceiling was.
_MAX_TOKENS = 8000

# Enough for a reasoning preamble plus one word. A single-token probe is what
# made the board lie: it could not reach the failure mode it was there to catch.
_HEALTH_MAX_TOKENS = 512


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
        self._timeout = settings.llm_timeout_seconds or _DEFAULT_TIMEOUT_SECONDS

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def credentials_present(self) -> bool:
        return bool(self._api_key)

    def _get_client(self, timeout: float | None = None) -> Any:
        """Create an Anthropic client pointed at AgentRouter."""
        import anthropic

        return anthropic.Anthropic(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=timeout if timeout is not None else self._timeout,
        )

    def health(self) -> ProviderHealth:
        """Probe the endpoint with a minimal request.

        A configured key that the gateway rejects is a *worse* state than no key
        at all, because it looks configured.  ``DISABLED`` means nothing to try;
        ``ERROR`` means we tried and it did not work, with the reason attached.

        The probe asks for a real, if tiny, completion and insists on getting
        text back. It used to request ``max_tokens=1``, which proved only that
        the socket opened -- so the board read ``ok`` during a period when every
        actual review was failing, because the failure lived in the response
        budget rather than in the connection. A probe that cannot fail the way
        production fails is not measuring production.
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
            client = self._get_client(timeout=30.0)
            response = client.messages.create(
                model=self.model_name,
                max_tokens=_HEALTH_MAX_TOKENS,
                messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            )
            # Touch the payload so a malformed body fails here rather than at the
            # first real call. A str or dict body is what some gateways return in
            # place of a message object; either is enough to prove reachability.
            if isinstance(response, str | dict):
                returned_text = bool(str(response).strip())
            else:
                returned_text = any(
                    getattr(block, "type", None) == "text"
                    and getattr(block, "text", "").strip()
                    for block in (getattr(response, "content", None) or [])
                )
            if not returned_text:
                return ProviderHealth(
                    provider=self.provider_name,
                    kind="llm",
                    status=HealthStatus.ERROR,
                    enabled=True,
                    credentials_present=True,
                    message=(
                        f"{self._base_url} answered but produced no text "
                        f"(stop_reason={getattr(response, 'stop_reason', 'unknown')}); "
                        "reviews would fail"
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - a health probe reports any failure
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
            message=f"{self._base_url} reachable, model {self.model_name} answering",
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
            raise ProviderTimeoutError(self.provider_name, self._timeout) from exc
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
            text = response.get("content", "")
        else:
            content = getattr(response, "content", [])
            # Filter on the block's declared type rather than on the presence of
            # a `.text` attribute: a reasoning block carries its own payload and
            # must not be concatenated into the JSON we are about to parse.
            text_parts = [
                block.text
                for block in content
                if getattr(block, "type", None) == "text" and hasattr(block, "text")
            ]
            text = "".join(text_parts)

        if not text.strip():
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == "max_tokens":
                # Distinguishable on purpose. "Empty message" reads as a gateway
                # fault and sends the next reader to check the endpoint; the
                # actual cause is our own ceiling being spent on reasoning before
                # the model reached the answer, which is a config fix here.
                raise ProviderBadResponseError(
                    self.provider_name,
                    f"response hit the {_MAX_TOKENS}-token ceiling before emitting "
                    "any text; raise AGENTROUTER max_tokens for this model",
                )
            raise ProviderBadResponseError(
                self.provider_name, "model returned an empty message"
            )

        usage = getattr(response, "usage", None)
        log.info(
            "llm review completed",
            extra={
                "provider": self.provider_name,
                "model": self.model_name,
                "prompt_tokens": getattr(usage, "input_tokens", None),
                "completion_tokens": getattr(usage, "output_tokens", None),
            },
        )
        return text
