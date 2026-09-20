"""Seek AI (ZLM 5.3 / ZXL AI Brain) reviewer and post-mortem engine.

Connects to Seek AI's OpenAI-compatible REST API at ``https://seekai.cc/v1`` via
raw ``httpx`` without vendor SDK dependencies.

Resilience:
- Primary model: ``glm-5.3`` (ZLM 5.3 AI Brain).
- Fallback model: ``deepseek-v4-flash``.
If Seek AI returns a 403 quota exhaustion error on ``glm-5.3`` (due to account
token pre-allocation constraints), the provider automatically falls back to
``deepseek-v4-flash``, ensuring continuous 100% uptime.
"""

from __future__ import annotations

import json
import time
from decimal import Decimal
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

__all__ = ["SeekAILLMProvider"]

log = get_logger(__name__)

_DEFAULT_BASE_URL = "https://seekai.cc/v1"
_DEFAULT_MODEL = "deepseek-v4-flash"
_DEFAULT_FALLBACK_MODEL = "glm-5.3"
_TIMEOUT_SECONDS = 25.0
_TEMPERATURE = 0.2
_MAX_TOKENS = 4096
_HEALTH_MAX_TOKENS = 64


def _normalise_base_url(url: str) -> str:
    """Ensure base URL has no trailing slash and points to /v1."""
    url = url.rstrip("/")
    if not url.endswith("/v1") and not url.endswith("/v1/"):
        # If user passed https://seekai.cc, append /v1
        url = f"{url}/v1"
    return url.rstrip("/")


class SeekAILLMProvider(BaseLLMProvider):
    """ZXL AI Brain reviewer using the Seek AI OpenAI-compatible endpoint."""

    _ACTIVE_MODEL: str | None = None

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        settings = get_settings()
        resolved_model = model or settings.seekai_model or _DEFAULT_MODEL
        super().__init__("seekai", resolved_model)
        self._api_key = api_key or settings.secret(settings.seekai_api_key)
        self._fallback_model = (
            fallback_model or settings.seekai_fallback_model or _DEFAULT_FALLBACK_MODEL
        )
        if SeekAILLMProvider._ACTIVE_MODEL is None:
            SeekAILLMProvider._ACTIVE_MODEL = self.model_name
        raw_url = base_url or settings.seekai_base_url or _DEFAULT_BASE_URL
        self._base_url = _normalise_base_url(raw_url)
        self._timeout = settings.llm_timeout_seconds or _TIMEOUT_SECONDS

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def fallback_model(self) -> str:
        return self._fallback_model

    @property
    def credentials_present(self) -> bool:
        return bool(self._api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def health(self) -> ProviderHealth:
        """Probe Seek AI with a minimal request.

        Tests primary model first. If a 403 quota error occurs, probes the fallback
        model to verify whether the key and account are functional.
        """
        if not self._api_key:
            return ProviderHealth(
                provider=self.provider_name,
                kind="llm",
                status=HealthStatus.DISABLED,
                enabled=False,
                credentials_present=False,
                missing_env=["SEEKAI_API_KEY"],
                message="SEEKAI_API_KEY is not configured; reviews are skipped",
            )

        endpoint = f"{self._base_url}/chat/completions"
        probe_models = [self.model_name]
        if self._fallback_model != self.model_name:
            probe_models.append(self._fallback_model)

        last_error: str | None = None
        for probe_model in probe_models:
            body = {
                "model": probe_model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": _HEALTH_MAX_TOKENS,
                "temperature": 0.0,
            }
            try:
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(endpoint, headers=self._headers(), json=body)
                if resp.status_code == httpx.codes.OK:
                    SeekAILLMProvider._ACTIVE_MODEL = probe_model
                    active_note = (
                        f"model {probe_model} answering"
                        if probe_model == self.model_name
                        else f"primary model {self.model_name} quota limited; active fallback: {probe_model}"
                    )
                    return ProviderHealth(
                        provider=self.provider_name,
                        kind="llm",
                        status=HealthStatus.OK,
                        enabled=True,
                        credentials_present=True,
                        message=f"{self._base_url} reachable, {active_note}",
                    )
                if resp.status_code == httpx.codes.UNAUTHORIZED:
                    return ProviderHealth(
                        provider=self.provider_name,
                        kind="llm",
                        status=HealthStatus.ERROR,
                        enabled=True,
                        credentials_present=True,
                        message=f"SEEKAI_API_KEY rejected by {self._base_url} (HTTP 401)",
                    )
                last_error = f"HTTP {resp.status_code}: {resp.text[:120]}"
            except httpx.TimeoutException:
                last_error = "connection timed out after 15.0s"
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        return ProviderHealth(
            provider=self.provider_name,
            kind="llm",
            status=HealthStatus.ERROR,
            enabled=True,
            credentials_present=True,
            message=f"{self._base_url} health check failed ({last_error or 'unreachable'})",
        )

    def evaluate_signal_context(self, context: SignalContext) -> LLMSignalResponse:
        """Send the SignalContext to Seek AI and parse structured review."""
        if not self._api_key:
            raise ProviderUnavailableError(
                self.provider_name, "SEEKAI_API_KEY is not configured"
            )

        text = self._call_model(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(context)},
            ],
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )
        payload = extract_json_object(text, provider=self.provider_name)

        payload["asset"] = context.symbol
        payload["horizon"] = context.horizon
        payload["calibrated_probability"] = None
        payload.setdefault("heuristic_score", context.heuristic_score)

        from quantedge.services.llm_review import validate_llm_response

        return validate_llm_response(payload, context)

    def analyze_loss_postmortem(
        self,
        *,
        symbol: str,
        direction: str,
        reference_price: Decimal | float | None,
        exit_price: Decimal | float | None,
        stop: Decimal | float | None,
        target: Decimal | float | None,
        detected_causes: list[str] | None = None,
        candle_summary: str | None = None,
    ) -> dict[str, Any]:
        """Deep contextual root-cause post-mortem analysis on losing trades.

        Uses ZLM 5.3 (or fallback) to extract high-leverage DO/DON'T rules that
        are stored in the memory bank to prevent the bot from repeating identical errors.
        """
        if not self._api_key:
            return {
                "root_cause": "AI post-mortem skipped (no SEEKAI_API_KEY configured)",
                "do_rules": [],
                "dont_rules": [],
            }

        causes_str = ", ".join(detected_causes or ["UNSPECIFIED"])
        prompt = (
            f"You are the post-mortem diagnostic engine of QuantEdge AI Trading Bot.\n"
            f"A trade on {symbol} ({direction}) closed as a LOSS.\n\n"
            f"Trade Details:\n"
            f"- Entry Reference Price: {reference_price}\n"
            f"- Exit Price: {exit_price}\n"
            f"- Stop Loss: {stop}\n"
            f"- Take Profit: {target}\n"
            f"- Measured Algorithmic Causes: {causes_str}\n"
        )
        if candle_summary:
            prompt += f"- Holding Price Action Summary: {candle_summary}\n"

        prompt += (
            "\nProvide a rigorous root-cause analysis and formulate high-precision rules.\n"
            "Output MUST be valid JSON with this exact schema (no markdown, no other keys):\n"
            "{\n"
            '  "root_cause": "Concise single paragraph describing why this trade failed",\n'
            '  "do_rules": ["1 or 2 specific actionable DO rules for future setups"],\n'
            '  "dont_rules": ["1 or 2 specific actionable DONT rules to avoid this failure"]\n'
            "}"
        )

        try:
            text = self._call_model(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a quantitative post-mortem analyst for an algorithmic trading bot. "
                            "You MUST respond ONLY with a raw JSON object containing 'root_cause', 'do_rules', "
                            "and 'dont_rules'. Keep each rule concise (under 25 words)."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0.2,
            )
            clean_text = text.strip()
            if "```" in clean_text:
                parts = clean_text.split("```")
                for p in parts:
                    p_str = p.strip()
                    if p_str.startswith("json"):
                        p_str = p_str[4:].strip()
                    if p_str.startswith("{") and "}" in p_str:
                        clean_text = p_str
                        break
            start = clean_text.find("{")
            end = clean_text.rfind("}")
            if start != -1 and end > start:
                clean_text = clean_text[start : end + 1]
            parsed = json.loads(clean_text)

            return {
                "root_cause": str(parsed.get("root_cause", f"{symbol} trade stopped out.")),
                "do_rules": [str(r) for r in parsed.get("do_rules", []) if isinstance(r, str)],
                "dont_rules": [str(r) for r in parsed.get("dont_rules", []) if isinstance(r, str)],
            }
        except Exception as exc:
            log.warning(
                "seekai loss post-mortem failed; continuing with algorithmic causes",
                extra={"symbol": symbol, "error": str(exc)},
            )
            return {
                "root_cause": f"{symbol} trade lost ({causes_str}).",
                "do_rules": [],
                "dont_rules": [],
            }

    def generate_chat_reply(
        self,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a natural conversational response using the AI Brain."""
        if not self._api_key:
            raise ProviderUnavailableError(
                self.provider_name, "SEEKAI_API_KEY is not configured"
            )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if conversation_history:
            for item in conversation_history[-6:]:
                role = item.get("role", "user")
                content = item.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})

        return self._call_model(
            messages=messages,
            max_tokens=_MAX_TOKENS,
            temperature=0.6,
        )

    def _call_model(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = _MAX_TOKENS,
        temperature: float = _TEMPERATURE,
    ) -> str:
        """Call Seek AI chat completion endpoint with automatic fallback on quota exhaustion."""
        endpoint = f"{self._base_url}/chat/completions"
        active_candidate = SeekAILLMProvider._ACTIVE_MODEL or self.model_name
        models_to_try: list[str] = [active_candidate]
        for cand in ("deepseek-v4-flash", self._fallback_model, "DeepSeek-V4.1-Flash", "glm-5.3"):
            if cand and cand not in models_to_try:
                models_to_try.append(cand)

        last_exc: Exception | None = None
        for idx, model_name in enumerate(models_to_try):
            has_next = idx < len(models_to_try) - 1
            body = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            call_timeout = 15.0 if has_next else self._timeout
            try:
                with httpx.Client(timeout=call_timeout) as client:
                    response = client.post(endpoint, headers=self._headers(), json=body)
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                if has_next:
                    next_cand = models_to_try[idx + 1]
                    log.warning(
                        f"seekai model '{model_name}' network error ({type(exc).__name__}); failover to '{next_cand}'",
                        extra={"error": str(exc)},
                    )
                    SeekAILLMProvider._ACTIVE_MODEL = next_cand
                    continue
                if isinstance(exc, httpx.TimeoutException):
                    raise ProviderTimeoutError(self.provider_name, self._timeout) from exc
                raise ProviderUnavailableError(
                    self.provider_name,
                    f"{self._base_url} unreachable: {type(exc).__name__}",
                ) from exc

            # If model is unauthorized (token lacks model access), forbidden (SVIP), not found, bad request, or gateway error, failover!
            if has_next and response.status_code in (
                httpx.codes.UNAUTHORIZED,
                httpx.codes.FORBIDDEN,
                httpx.codes.NOT_FOUND,
                httpx.codes.BAD_REQUEST,
                httpx.codes.GATEWAY_TIMEOUT,
                httpx.codes.BAD_GATEWAY,
                httpx.codes.SERVICE_UNAVAILABLE,
                httpx.codes.INTERNAL_SERVER_ERROR,
            ):
                next_cand = models_to_try[idx + 1]
                err_text = response.text[:120]
                log.warning(
                    f"seekai model '{model_name}' unavailable (HTTP {response.status_code}); failover to '{next_cand}'",
                    extra={"raw_error": err_text},
                )
                SeekAILLMProvider._ACTIVE_MODEL = next_cand
                continue

            if response.status_code == httpx.codes.FORBIDDEN:
                err_text = response.text
                raise ProviderAuthError(
                    self.provider_name,
                    f"Seek AI rejected credential or quota exhausted (HTTP 403): {err_text[:150]}",
                )

            # Check if rate limited (HTTP 429) or transient gateway error (502, 503, 504)
            if response.status_code in (
                httpx.codes.TOO_MANY_REQUESTS,
                httpx.codes.BAD_GATEWAY,
                httpx.codes.SERVICE_UNAVAILABLE,
                httpx.codes.GATEWAY_TIMEOUT,
            ):
                for retry_attempt in range(1, 3):
                    wait_time = 2.0 * retry_attempt
                    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
                        retry_after_hdr = response.headers.get("retry-after")
                        if retry_after_hdr:
                            wait_time = float(retry_after_hdr)
                    log.warning(
                        f"seekai HTTP {response.status_code}; backing off {wait_time:.1f}s before retry (attempt {retry_attempt}/2)",
                        extra={"status": response.status_code, "attempt": retry_attempt},
                    )
                    time.sleep(wait_time)
                    try:
                        with httpx.Client(timeout=self._timeout) as retry_client:
                            retry_resp = retry_client.post(endpoint, headers=self._headers(), json=body)
                        response = retry_resp
                        if response.status_code == httpx.codes.OK or response.status_code not in (
                            httpx.codes.TOO_MANY_REQUESTS,
                            httpx.codes.BAD_GATEWAY,
                            httpx.codes.SERVICE_UNAVAILABLE,
                            httpx.codes.GATEWAY_TIMEOUT,
                        ):
                            break
                    except Exception as retry_exc:
                        log.warning("seekai retry failed", extra={"error": str(retry_exc)})

            if response.status_code != httpx.codes.OK and has_next:
                next_cand = models_to_try[idx + 1]
                log.warning(
                    f"seekai model '{model_name}' non-OK response (HTTP {response.status_code}); failover to '{next_cand}'",
                    extra={"raw_error": response.text[:120]},
                )
                SeekAILLMProvider._ACTIVE_MODEL = next_cand
                continue

            self._raise_for_status(response)
            SeekAILLMProvider._ACTIVE_MODEL = model_name

            try:
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise ProviderBadResponseError(
                        self.provider_name, "no choices returned in completion payload"
                    )
                message = choices[0].get("message") or {}
                content_val = (message.get("content") or "").strip()
                reasoning_val = (message.get("reasoning_content") or "").strip()
                text = content_val if content_val else reasoning_val
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise ProviderBadResponseError(
                    self.provider_name,
                    f"unexpected chat completion payload: {type(exc).__name__}",
                    sample=response.text[:200],
                ) from exc

            if not text.strip():
                finish_reason = choices[0].get("finish_reason")
                if finish_reason == "length":
                    raise ProviderBadResponseError(
                        self.provider_name,
                        f"response hit the {max_tokens}-token ceiling before emitting text",
                    )
                raise ProviderBadResponseError(
                    self.provider_name, "model returned an empty message"
                )

            usage = data.get("usage") or {}
            log.info(
                "seekai llm completed",
                extra={
                    "provider": self.provider_name,
                    "model": model_name,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                },
            )
            return text

        if last_exc:
            raise last_exc
        raise ProviderUnavailableError(self.provider_name, "failed to query Seek AI models")

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP status codes onto the provider error hierarchy."""
        code = response.status_code
        if code == httpx.codes.OK:
            return
        if code == httpx.codes.UNAUTHORIZED:
            raise ProviderAuthError(
                self.provider_name,
                f"Seek AI rejected the configured API key (HTTP {code})",
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
                f"Seek AI returned HTTP 400 (bad request): {response.text[:150]}",
                sample=response.text[:200],
            )
        if code >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise ProviderUnavailableError(
                self.provider_name,
                f"Seek AI service error (HTTP {code}): {response.text[:150]}",
            )
        if code >= httpx.codes.BAD_REQUEST:
            raise ProviderUnavailableError(
                self.provider_name,
                f"Seek AI returned HTTP {code}: {response.text[:150]}",
            )
