"""Error model.

Every error that can reach a client (MCP tool result or HTTP response) is a
:class:`QuantEdgeError`. Its :meth:`~QuantEdgeError.to_dict` payload is
deliberately narrow: a stable machine code, a human message, and a small
``details`` mapping that is scrubbed of anything secret-looking.

Two rules matter here:

1. **Never leak credentials.** Provider errors routinely echo the request URL,
   which may contain an ``apikey=`` query parameter. Everything passing through
   this module goes through :func:`quantedge.logging.redact`.
2. **Distinguish "no answer" from "bad answer".** ``INSUFFICIENT_DATA`` means we
   could not obtain trustworthy inputs. It is not an internal failure and it is
   never silently converted into a neutral or optimistic result.
"""

from __future__ import annotations

from typing import Any

from quantedge.logging import redact

__all__ = [
    "AllProvidersFailedError",
    "CircuitOpenError",
    "ConfigurationError",
    "DataQualityError",
    "ErrorCode",
    "InsufficientDataError",
    "LLMContractError",
    "PersistenceError",
    "ProviderAuthError",
    "ProviderDisabledError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "QuantEdgeError",
    "UnsupportedSymbolError",
    "UnsupportedTimeframeError",
    "ValidationError",
]


class ErrorCode:
    """Stable machine-readable error codes.

    These are part of the public contract for MCP tools and HTTP routes.
    Values must not be renamed without a version bump.
    """

    CONFIGURATION = "CONFIGURATION_ERROR"
    VALIDATION = "VALIDATION_ERROR"
    UNSUPPORTED_SYMBOL = "UNSUPPORTED_SYMBOL"
    UNSUPPORTED_TIMEFRAME = "UNSUPPORTED_TIMEFRAME"
    PROVIDER_DISABLED = "PROVIDER_DISABLED"
    PROVIDER_AUTH = "PROVIDER_AUTH_ERROR"
    PROVIDER_RATE_LIMIT = "PROVIDER_RATE_LIMIT"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PROVIDER_BAD_RESPONSE = "PROVIDER_BAD_RESPONSE"
    ALL_PROVIDERS_FAILED = "ALL_PROVIDERS_FAILED"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    DATA_QUALITY_FAILED = "DATA_QUALITY_FAILED"
    LLM_CONTRACT = "LLM_CONTRACT_VIOLATION"
    PERSISTENCE = "PERSISTENCE_ERROR"
    INTERNAL = "INTERNAL_ERROR"


class QuantEdgeError(Exception):
    """Base class for all gateway errors.

    Parameters
    ----------
    message:
        Human-readable description. Redacted before it is ever surfaced.
    code:
        A :class:`ErrorCode` value.
    details:
        Optional structured context. Keys are preserved; values are redacted.
    retryable:
        Whether an identical retry could plausibly succeed later. Drives the
        circuit breaker and the worker backoff logic.
    provider:
        Which provider the failure originated from, when applicable.
    """

    code: str = ErrorCode.INTERNAL
    http_status: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
        provider: str | None = None,
    ) -> None:
        self.message = redact(message)
        if code is not None:
            self.code = code
        self.details = {k: redact(str(v)) for k, v in (details or {}).items()}
        self.retryable = retryable
        self.provider = provider
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Return a safe, client-facing representation.

        Contains no stack trace, no file path and no credential material.
        """
        payload: dict[str, Any] = {
            "error": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.provider:
            payload["provider"] = self.provider
        if self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.message}"


# --------------------------------------------------------------------------- #
# Configuration / input                                                        #
# --------------------------------------------------------------------------- #


class ConfigurationError(QuantEdgeError):
    """Invalid or unsafe configuration (e.g. wildcard CORS in production)."""

    code = ErrorCode.CONFIGURATION
    http_status = 500


class ValidationError(QuantEdgeError):
    """Caller supplied input that violates the contract."""

    code = ErrorCode.VALIDATION
    http_status = 422


class UnsupportedSymbolError(ValidationError):
    """Symbol is not on the allowlist in ``config/symbols.yaml``."""

    code = ErrorCode.UNSUPPORTED_SYMBOL
    http_status = 400


class UnsupportedTimeframeError(ValidationError):
    """Timeframe is not one of the supported enumerated values."""

    code = ErrorCode.UNSUPPORTED_TIMEFRAME
    http_status = 400


# --------------------------------------------------------------------------- #
# Providers                                                                    #
# --------------------------------------------------------------------------- #


class ProviderError(QuantEdgeError):
    """Base class for provider-originated failures."""

    code = ErrorCode.PROVIDER_UNAVAILABLE
    http_status = 502


class ProviderDisabledError(ProviderError):
    """Provider is switched off or missing credentials.

    This is an expected, non-alarming state. Missing keys must fail *safely*:
    the provider reports ``disabled`` and routing moves on to the next
    candidate, or the caller receives a structured failure.
    """

    code = ErrorCode.PROVIDER_DISABLED
    http_status = 503

    def __init__(
        self,
        provider: str,
        reason: str = "provider is not configured",
        *,
        missing_env: list[str] | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if missing_env:
            # Names only -- never values.
            details["missing_env"] = ", ".join(sorted(missing_env))
        super().__init__(
            f"{provider}: {reason}",
            details=details,
            retryable=False,
            provider=provider,
        )


class ProviderAuthError(ProviderError):
    """Credentials were rejected (401/403)."""

    code = ErrorCode.PROVIDER_AUTH
    http_status = 502

    def __init__(self, provider: str, message: str = "credentials rejected") -> None:
        super().__init__(message, retryable=False, provider=provider)


class ProviderRateLimitError(ProviderError):
    """Provider rate limit hit (429/418), or our own limiter refused."""

    code = ErrorCode.PROVIDER_RATE_LIMIT
    http_status = 429

    def __init__(
        self,
        provider: str,
        message: str = "rate limit exceeded",
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if retry_after_seconds is not None:
            details["retry_after_seconds"] = retry_after_seconds
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message, details=details, retryable=True, provider=provider)


class ProviderTimeoutError(ProviderError):
    """Request exceeded the configured timeout."""

    code = ErrorCode.PROVIDER_TIMEOUT
    http_status = 504

    def __init__(self, provider: str, timeout_seconds: float) -> None:
        super().__init__(
            f"request timed out after {timeout_seconds:g}s",
            details={"timeout_seconds": timeout_seconds},
            retryable=True,
            provider=provider,
        )


class ProviderUnavailableError(ProviderError):
    """Network failure or 5xx from the provider."""

    code = ErrorCode.PROVIDER_UNAVAILABLE
    http_status = 502

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(message, retryable=True, provider=provider)


class ProviderGeoBlockedError(ProviderError):
    """Provider refuses this egress region (HTTP 451).

    Not retryable: the request is well-formed and the credential is fine, so
    repeating it from the same IP produces the same refusal. Only a different
    host or a different region changes the answer, which is why this is its own
    class rather than an unavailability the retry loop would keep hammering.
    """

    code = ErrorCode.PROVIDER_UNAVAILABLE
    http_status = 502

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(message, retryable=False, provider=provider)


class ProviderBadResponseError(ProviderError):
    """Provider returned a payload we cannot trust or parse.

    Raised on malformed JSON, missing required fields, or values that violate
    basic market invariants. We never "repair" such a payload by guessing.
    """

    code = ErrorCode.PROVIDER_BAD_RESPONSE
    http_status = 502

    def __init__(self, provider: str, message: str, *, sample: str | None = None) -> None:
        details: dict[str, Any] = {}
        if sample:
            details["sample"] = sample[:200]
        super().__init__(message, details=details, retryable=False, provider=provider)


class AllProvidersFailedError(ProviderError):
    """Every provider in a routing chain was skipped or failed.

    Distinct from :class:`ProviderUnavailableError`, which reports that *one*
    vendor is down and another may still answer. This says there is no path to
    the data at all, so the caller must surface ``INSUFFICIENT_DATA`` rather
    than an empty result: "nobody could answer" and "the answer is nothing" are
    different claims, and only one of them is safe to act on.

    The per-provider attempt log is kept as the structured :attr:`attempts`
    attribute. ``details`` values are flattened to redacted strings for the
    client-facing payload, which would destroy a nested list.
    """

    code = ErrorCode.ALL_PROVIDERS_FAILED
    http_status = 503

    def __init__(
        self,
        chain: list[str],
        message: str,
        *,
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        self.attempts = [dict(a) for a in (attempts or [])]
        self.chain = list(chain)
        summary = (
            "; ".join(
                f"{a.get('provider')}: {a.get('outcome')} ({a.get('reason')})"
                for a in self.attempts
            )
            or "none attempted"
        )
        # A chain skipped entirely for missing credentials will be skipped again
        # a second later; one that timed out may well succeed. Only the latter
        # should drive worker retries.
        super().__init__(
            message,
            details={"chain": ", ".join(chain) or "empty", "attempts": summary},
            retryable=any(a.get("outcome") == "failed" for a in self.attempts),
        )


class CircuitOpenError(ProviderError):
    """Circuit breaker is open; we are deliberately not calling the provider."""

    code = ErrorCode.CIRCUIT_OPEN
    http_status = 503

    def __init__(self, provider: str, recovery_in_seconds: float) -> None:
        super().__init__(
            f"circuit breaker open; retrying in {recovery_in_seconds:.1f}s",
            details={"recovery_in_seconds": round(recovery_in_seconds, 2)},
            retryable=True,
            provider=provider,
        )


# --------------------------------------------------------------------------- #
# Data quality / analysis                                                      #
# --------------------------------------------------------------------------- #


class InsufficientDataError(QuantEdgeError):
    """Required data is unavailable, stale, or too short to analyse.

    This is the canonical response when we *cannot know*. It must never be
    downgraded into a neutral-sounding success.
    """

    code = ErrorCode.INSUFFICIENT_DATA
    http_status = 422

    def __init__(
        self,
        message: str,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        missing: list[str] | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if symbol:
            details["symbol"] = symbol
        if timeframe:
            details["timeframe"] = timeframe
        if missing:
            details["missing"] = ", ".join(missing)
        super().__init__(message, details=details, retryable=True)


class DataQualityError(QuantEdgeError):
    """Data quality engine returned FAIL; downstream output is blocked."""

    code = ErrorCode.DATA_QUALITY_FAILED
    http_status = 422

    def __init__(self, message: str, *, blocking_reasons: list[str] | None = None) -> None:
        details: dict[str, Any] = {}
        if blocking_reasons:
            details["blocking_reasons"] = "; ".join(blocking_reasons)
        super().__init__(message, details=details, retryable=True)


class LLMContractError(QuantEdgeError):
    """LLM output violated the strict response contract.

    Covers malformed JSON, unknown fields, fabricated values that contradict the
    supplied context, and any attempt to emit a ``calibrated_probability``
    without a registered calibration model.
    """

    code = ErrorCode.LLM_CONTRACT
    http_status = 502

    def __init__(self, message: str, *, violations: list[str] | None = None) -> None:
        details: dict[str, Any] = {}
        if violations:
            details["violations"] = "; ".join(violations)
        super().__init__(message, details=details, retryable=False)


class PersistenceError(QuantEdgeError):
    """Storage layer failure."""

    code = ErrorCode.PERSISTENCE
    http_status = 500
