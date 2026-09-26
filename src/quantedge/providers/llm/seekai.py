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
from urllib.parse import urlsplit

import httpx

from quantedge.config import get_settings
from quantedge.contracts import (
    HealthStatus,
    LLMSignalResponse,
    ProviderHealth,
    SignalContext,
)
from quantedge.deadline import (
    DEADLINE_MARGIN_SECONDS,
    MIN_LLM_CALL_SECONDS,
    SERVERLESS_BUDGET_SECONDS,
    clamp_to_deadline,
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
_DEFAULT_MODEL = "glm-5.3-flash"
_DEFAULT_FALLBACK_MODEL = "deepseek-v4.1-flash"
_TIMEOUT_SECONDS = 25.0
_TEMPERATURE = 0.2
_MAX_TOKENS = 4096
_HEALTH_MAX_TOKENS = 64


def _normalise_base_url(url: str) -> str:
    """Trim a trailing slash; append ``/v1`` only for a bare host with no path.

    seekai.cc's OpenAI-compatible API lives under ``/v1``, so a bare
    ``https://seekai.cc`` is completed to ``https://seekai.cc/v1``. Other
    OpenAI-compatible GLM endpoints version their path differently -- Z.ai's real
    GLM API is ``https://api.z.ai/api/paas/v4`` -- so any URL that already carries
    a path is left exactly as given. The old logic force-appended ``/v1`` to every
    URL, which turned the Z.ai base into ``.../v4/v1`` and made pointing at a
    genuine GLM endpoint impossible.
    """
    url = url.rstrip("/")
    path = urlsplit(url).path
    if path in ("", "/"):
        # Bare host (no path): complete it to the seekai.cc default of /v1.
        url = f"{url}/v1"
    return url.rstrip("/")


def _leading_alpha_token(model_id: str) -> str:
    """The leading alphabetic run of a model id, vendor prefix stripped.

    ``MiniMaxAI/MiniMax-M2.7`` -> ``minimax``; ``glm-5.3-flash`` -> ``glm``;
    ``deepseek-v4.1-flash`` -> ``deepseek``. This is the coarse family token used
    to tell whether the model that answered is the family we asked for.
    """
    tail = model_id.lower().rsplit("/", 1)[-1]
    out: list[str] = []
    for ch in tail:
        if ch.isalpha():
            out.append(ch)
        else:
            break
    return "".join(out)


def _same_model_family(requested: str | None, served: str | None) -> bool | None:
    """Whether the served model is plausibly the model we requested.

    Returns ``None`` when either id is missing (nothing to compare). Providers
    that honour the request echo a model id from the same family; an aggregator
    that silently routes elsewhere -- e.g. seekai.cc answering a ``glm-5.3-flash``
    request with ``MiniMaxAI/MiniMax-M2.7`` -- is caught here, so a decision is
    never attributed to a model that did not actually make it. A config/env match
    on the requested name is NOT proof of identity; this compares against the
    server's own reported model id.
    """
    if not requested or not served:
        return None
    return _leading_alpha_token(requested) == _leading_alpha_token(served)


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
        raw_timeout = float(settings.llm_timeout_seconds or _TIMEOUT_SECONDS)
        # The per-call ceiling is the whole serverless budget less the response
        # margin: off a serverless host (CLI/tests) it bounds a runaway call, and
        # on one it is the largest value the request-deadline clamp in _call_model
        # can hand back. A configured llm_timeout_seconds larger than the host's
        # wall clock (the old 180) can no longer be honoured, so it is capped here
        # rather than silently promising time the host will never give -- this is
        # what dissolves the "180-second problem". The real bound at run time is
        # min(this, time actually left before the 60s kill).
        serverless_ceiling = max(MIN_LLM_CALL_SECONDS, SERVERLESS_BUDGET_SECONDS - DEADLINE_MARGIN_SECONDS)
        self._timeout = min(raw_timeout, serverless_ceiling)
        # Telemetry of the most recent successful completion, so a caller can prove
        # WHICH model actually answered rather than trusting config. Populated only
        # on the success path of ``_call_model`` and read immediately by the caller
        # in the same synchronous request, so there is no cross-request staleness:
        #   requested_model -- the model we asked for first (the active candidate)
        #   resolved_model  -- the candidate that actually returned (after any failover)
        #   response_model  -- the ``model`` field the server put in its own payload
        #   latency_ms      -- wall-clock of the HTTP round trip that answered
        self._last_call_meta: dict[str, Any] = {}

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
        """Probe Seek AI via the /models endpoint without consuming chat quota.

        Verifies API key validity and confirms configured models are available.
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

        endpoint = f"{self._base_url}/models"
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(endpoint, headers=self._headers())

            if resp.status_code == httpx.codes.OK:
                data = resp.json().get("data", [])
                available_ids = {m.get("id") for m in data if isinstance(m, dict)}
                if self.model_name in available_ids:
                    SeekAILLMProvider._ACTIVE_MODEL = self.model_name
                    msg = f"{self._base_url} reachable, model {self.model_name} ready"
                elif self._fallback_model in available_ids:
                    SeekAILLMProvider._ACTIVE_MODEL = self._fallback_model
                    msg = f"{self._base_url} reachable, fallback {self._fallback_model} ready"
                else:
                    msg = f"{self._base_url} reachable ({len(available_ids)} models available)"
                return ProviderHealth(
                    provider=self.provider_name,
                    kind="llm",
                    status=HealthStatus.OK,
                    enabled=True,
                    credentials_present=True,
                    message=msg,
                )
            if resp.status_code == httpx.codes.TOO_MANY_REQUESTS:
                return ProviderHealth(
                    provider=self.provider_name,
                    kind="llm",
                    status=HealthStatus.OK,
                    enabled=True,
                    credentials_present=True,
                    message=f"{self._base_url} reachable (rate-limit window active)",
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
            last_error = "connection timed out after 8.0s"
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        return ProviderHealth(
            provider=self.provider_name,
            kind="llm",
            status=HealthStatus.ERROR,
            enabled=True,
            credentials_present=True,
            message=f"{self._base_url} health check failed ({last_error})",
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
            timeout=min(self._timeout, 6.0),
        )
        payload = extract_json_object(text, provider=self.provider_name)

        payload["asset"] = context.symbol
        payload["horizon"] = context.horizon
        payload["calibrated_probability"] = None
        payload.setdefault("heuristic_score", context.heuristic_score)

        from quantedge.services.llm_review import validate_llm_response

        return validate_llm_response(payload, context)

    def decide_trade(
        self,
        evidence: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Let the AI brain DECIDE a trade from real, pre-verified market evidence.

        This is the decision authority (not merely a reviewer): it receives real
        indicators already computed from live TradingView data across two
        timeframes plus the deterministic candidate, and returns its own verdict.

        Returns a validated dict::

            {"decision": "UP"|"DOWN"|"NO_TRADE", "conviction": 0.0-1.0,
             "reason": str, "invalidation": str, "brain": <model id>}

        Raises a provider error on timeout, rate limit, or unparseable output so
        the caller can fall back to the deterministic gated decision. NO_TRADE is
        a first-class, expected outcome -- abstention is never treated as failure.
        """
        if not self._api_key:
            raise ProviderUnavailableError(
                self.provider_name, "SEEKAI_API_KEY is not configured"
            )

        system = (
            "You are the DECISION BRAIN of a quantitative trading bot -- you make the "
            "call, you are not a reviewer. You are given REAL, already-verified market "
            "evidence: indicators computed from live TradingView data across an "
            "execution timeframe and a higher confirmation timeframe. Decide the trade "
            "for the stated holding window.\n"
            "Default to committing to a DIRECTION (UP or DOWN) and expressing your "
            "honest confidence in the `conviction` field: a weak-but-real edge is a "
            "LOW-conviction lean (roughly 0.15-0.35), a clean setup with both "
            "timeframes aligned is high conviction (0.7+). A low conviction is NOT a "
            "reason to abstain -- it is how you tell the trader the edge is thin, and "
            "the bot sizes the position down accordingly. Reserve NO_TRADE for when "
            "there is genuinely no basis for a directional call: the two timeframes "
            "hard-conflict with no tie-breaker, the evidence is absent or "
            "self-contradictory, or extreme event risk makes either direction a coin "
            "toss. Do NOT invent numbers, prices, or levels beyond those provided. "
            "Trade only WITH the higher timeframe, never against a strong one.\n"
            "Decide directly and answer IMMEDIATELY. Do NOT emit any reasoning, "
            "analysis, planning, deliberation, or <think> blocks before or after the "
            "answer -- reason internally and keep it brief. Reply with exactly ONE raw "
            "JSON object and nothing else -- no markdown, no prose, no code fence:\n"
            '{"decision":"UP|DOWN|NO_TRADE","conviction":0.0-1.0,'
            '"reason":"one sentence citing the specific evidence",'
            '"invalidation":"the level or condition that would prove this wrong"}'
        )
        user = (
            "REAL market evidence (do not fabricate anything beyond this):\n"
            "```json\n"
            + json.dumps(evidence, indent=2, sort_keys=True, default=str)
            + "\n```"
        )

        budget = timeout if timeout is not None else min(self._timeout, 40.0)
        text = self._call_model(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # A bounded UP/DOWN/NO_TRADE verdict plus a one-sentence reason needs
            # few output tokens. The old 1200 let a reasoning model spend the whole
            # serverless window emitting <think> tokens -- measured >55s locally and
            # a Cloudflare 504 at ~61s upstream, i.e. it could never return inside
            # the 60s host wall. Capping output (with the no-reasoning directive
            # above) brings the same verdict back in ~25-35s, which fits. Full
            # deliberation, if wanted, belongs on the decoupled precompute path,
            # not on the synchronous request.
            max_tokens=500,
            temperature=0.1,
            timeout=budget,
            primary_only=True,
        )
        payload = extract_json_object(text, provider=self.provider_name)

        decision = str(payload.get("decision", "")).strip().upper()
        if decision not in ("UP", "DOWN", "NO_TRADE"):
            raise ProviderBadResponseError(
                self.provider_name,
                f"decision '{decision}' is not one of UP/DOWN/NO_TRADE",
                sample=text[:200],
            )
        try:
            conviction = float(payload.get("conviction"))
        except (TypeError, ValueError):
            conviction = 0.0
        conviction = max(0.0, min(1.0, conviction))

        # Merge the completion telemetry recorded by _call_model on this same call.
        # response_model is the server-declared model id -- the one piece of proof
        # that the model we intended is the model that actually answered, which an
        # env-var comparison alone cannot establish.
        meta = dict(self._last_call_meta)
        return {
            "decision": decision,
            "conviction": conviction,
            "reason": str(payload.get("reason") or "").strip()[:400],
            "invalidation": str(payload.get("invalidation") or "").strip()[:400],
            "brain": SeekAILLMProvider._ACTIVE_MODEL or self.model_name,
            "requested_model": meta.get("requested_model"),
            "resolved_model": meta.get("resolved_model"),
            "response_model": meta.get("response_model"),
            "latency_ms": meta.get("latency_ms"),
            # True/False when both ids are known: does the server's own model id
            # belong to the family we asked for? False here means the endpoint
            # substituted a different model (e.g. a glm-5.3-flash request answered
            # by MiniMax) and the decision must NOT be attributed to GLM.
            "model_verified": _same_model_family(
                meta.get("requested_model"), meta.get("response_model")
            ),
        }

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
                timeout=min(self._timeout, 10.0),
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

        # Chat has no preceding scan eating into the request budget, so it may use
        # almost the whole window. The old 12s cap was the prime cause of the bot
        # answering with a canned greeting instead of a real reply: the reasoning
        # model rarely finishes a conversational answer in 12s, timed out, and the
        # caller fell back. The request-deadline clamp in _call_model still trims
        # this to whatever wall clock actually remains, so it is safe on-host.
        return self._call_model(
            messages=messages,
            max_tokens=2048,
            temperature=0.6,
            timeout=min(self._timeout, 50.0),
        )

    def _call_model(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = _MAX_TOKENS,
        temperature: float = _TEMPERATURE,
        timeout: float | None = None,
        primary_only: bool = False,
    ) -> str:
        """Call Seek AI chat completion endpoint with automatic fallback on quota exhaustion.

        When ``primary_only`` is set the entire budget goes to a single model with
        no failover halving -- used for trade DECISIONS, where the slow reasoning
        model needs the full window rather than budget/2 split across candidates.
        """
        endpoint = f"{self._base_url}/chat/completions"
        active_candidate = SeekAILLMProvider._ACTIVE_MODEL or self.model_name
        models_to_try: list[str] = [active_candidate]
        for cand in ("glm-5.3-flash", "deepseek-v4.1-flash", "claude-sonnet-4-6", "claude-sonnet"):
            if cand and cand not in models_to_try:
                models_to_try.append(cand)
        # Bounded candidates: at most 3 real models
        effective_budget = timeout or self._timeout
        # A single-model decision, or a tight budget (<= 12s), gives the whole
        # budget to the primary model instead of splitting it across failovers.
        if primary_only or effective_budget <= 12.0:
            models_to_try = [active_candidate]
        else:
            models_to_try = models_to_try[:2]

        last_exc: Exception | None = None
        for idx, model_name in enumerate(models_to_try):
            has_next = idx < len(models_to_try) - 1
            # Re-clamp to the request deadline on every attempt: after a slow first
            # model, the wall clock has shrunk, so a failover gets only the time
            # that is genuinely left -- the cumulative work across failovers can
            # never push the request past the 60s host kill. Off a serverless host
            # (no deadline) this returns the full budget and behaviour is unchanged.
            budget_left = clamp_to_deadline(effective_budget)
            if budget_left < MIN_LLM_CALL_SECONDS:
                # Too little wall clock remains to finish a call. Fail fast into the
                # deterministic / grounded fallback rather than dial a doomed
                # sub-second timeout or let the host kill us mid-flight.
                if last_exc is not None:
                    raise last_exc
                raise ProviderTimeoutError(self.provider_name, max(budget_left, 0.0))
            # Reserve half for a possible failover, but only while that still leaves
            # each attempt enough time to actually return; otherwise spend the whole
            # remainder on this attempt (the next iteration will fail fast above).
            if has_next and (budget_left / 2.0) >= MIN_LLM_CALL_SECONDS:
                call_timeout = budget_left / 2.0
            else:
                call_timeout = budget_left
            body = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            try:
                call_started = time.monotonic()
                with httpx.Client(timeout=call_timeout) as client:
                    response = client.post(endpoint, headers=self._headers(), json=body)
                call_latency_ms = round((time.monotonic() - call_started) * 1000.0, 1)
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
                    raise ProviderTimeoutError(self.provider_name, call_timeout) from exc
                raise ProviderUnavailableError(
                    self.provider_name,
                    f"{self._base_url} unreachable: {type(exc).__name__}",
                ) from exc

            # Immediate rate limit exit: SeekAI limits the whole account (5 req/min) so retrying other models burns quota
            if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
                from quantedge.errors import ProviderRateLimitError

                raise ProviderRateLimitError(
                    self.provider_name,
                    "Seek AI 1-minute rate limit active (max 5 req/min)",
                )

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
                reasoning_val = (
                    message.get("reasoning")
                    or message.get("reasoning_content")
                    or message.get("thought")
                    or ""
                ).strip()

                import re
                cleaned_content = re.sub(r"<think>.*?</think>", "", content_val, flags=re.DOTALL).strip()
                if not cleaned_content and "<think>" in content_val:
                    if "</think>" in content_val:
                        cleaned_content = content_val.split("</think>")[-1].strip()
                    else:
                        cleaned_content = re.sub(r"</?think>", "", content_val).strip()

                text = cleaned_content if cleaned_content else (content_val if content_val else reasoning_val)
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
            # Record which model actually answered, straight from the server's own
            # payload -- this is the proof an env-var match cannot give: the caller
            # reads it back in the same request to attribute the decision honestly.
            self._last_call_meta = {
                "requested_model": models_to_try[0],
                "resolved_model": model_name,
                "response_model": (data.get("model") if isinstance(data, dict) else None),
                "latency_ms": call_latency_ms,
            }
            log.info(
                "seekai llm completed",
                extra={
                    "provider": self.provider_name,
                    "model": model_name,
                    "response_model": self._last_call_meta["response_model"],
                    "latency_ms": call_latency_ms,
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
