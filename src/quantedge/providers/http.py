"""Shared resilient HTTP client for every provider adapter.

One implementation, used by all adapters, so resilience behaviour is uniform
and testable in one place:

* **Timeouts** -- separate connect and total budgets.
* **Retry with exponential backoff and jitter** -- jitter matters; without it,
  N adapters that failed together retry together and re-create the outage.
* **Rate limiting** -- token-bucket per minute plus an optional daily cap, which
  free plans (Twelve Data: 8/min, 800/day) actually enforce.
* **Circuit breaker** -- after repeated failures we stop calling the provider
  entirely for a cooldown, rather than hammering a service that is down.
* **Request de-duplication** -- concurrent identical in-flight GETs share one
  response, so a burst of scanner tasks does not multiply paid API spend.

Only ``GET``/``POST`` of JSON is supported; nothing here can execute a trade.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from quantedge.errors import (
    CircuitOpenError,
    ProviderAuthError,
    ProviderBadResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from quantedge.logging import get_logger

__all__ = ["CircuitBreaker", "HttpClientConfig", "RateLimiter", "ResilientHttpClient"]

log = get_logger(__name__)


@dataclass(slots=True)
class HttpClientConfig:
    """Per-provider resilience policy, normally loaded from providers.yaml."""

    timeout_seconds: float = 10.0
    connect_timeout_seconds: float = 5.0
    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    backoff_jitter: bool = True
    requests_per_minute: int | None = None
    requests_per_day: int | None = None
    min_interval_seconds: float = 0.0
    """Minimum spacing between two requests to this provider.

    Distinct from ``requests_per_minute``: a 5/min budget still permits five
    calls in the same second, and providers that police a per-second burst
    (Alpha Vantage's free tier allows 1 request/second) reject that even though
    the per-minute total is legal.
    """
    failure_threshold: int = 5
    recovery_seconds: float = 30.0
    user_agent: str = "QuantEdge-Gateway/0.1 (+market-data-only)"


class RateLimiter:
    """Sliding-window limiter over a minute and (optionally) a day.

    A sliding window is used rather than a fixed bucket because providers
    measure "requests in the last 60 seconds", not "requests since :00".

    ``min_interval`` adds burst spacing on top of the windows, for providers
    that additionally cap requests per second.
    """

    def __init__(
        self,
        per_minute: int | None = None,
        per_day: int | None = None,
        min_interval: float = 0.0,
    ) -> None:
        self._per_minute = per_minute
        self._per_day = per_day
        self._min_interval = max(0.0, min_interval)
        self._minute: deque[float] = deque()
        self._day: deque[float] = deque()
        self._last_request: float | None = None
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        while self._minute and now - self._minute[0] > 60.0:
            self._minute.popleft()
        while self._day and now - self._day[0] > 86_400.0:
            self._day.popleft()

    async def acquire(self, provider: str) -> None:
        """Wait until a request slot is available.

        The daily cap is *not* waited out -- sleeping for hours would hang the
        caller, so we raise and let the caller degrade gracefully.
        """
        async with self._lock:
            now = time.monotonic()
            self._prune(now)

            if self._per_day is not None and len(self._day) >= self._per_day:
                raise ProviderRateLimitError(
                    provider,
                    f"daily request cap of {self._per_day} reached",
                    retry_after_seconds=86_400.0 - (now - self._day[0]),
                )

            if self._per_minute is not None and len(self._minute) >= self._per_minute:
                wait = 60.0 - (now - self._minute[0]) + 0.01
                log.debug(
                    "rate limit: waiting for a slot",
                    extra={"provider": provider, "wait_seconds": round(wait, 2)},
                )
                await asyncio.sleep(max(0.0, wait))
                now = time.monotonic()
                self._prune(now)

            # Burst spacing, applied last so it also spaces the request that
            # just finished waiting out a full window.
            if self._min_interval and self._last_request is not None:
                gap = now - self._last_request
                if gap < self._min_interval:
                    await asyncio.sleep(self._min_interval - gap)
                    now = time.monotonic()
                    self._prune(now)

            self._last_request = now
            self._minute.append(now)
            self._day.append(now)

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        self._prune(now)
        return {
            "used_last_minute": len(self._minute),
            "limit_per_minute": self._per_minute,
            "used_last_day": len(self._day),
            "limit_per_day": self._per_day,
            "min_interval_seconds": self._min_interval or None,
        }


@dataclass
class CircuitBreaker:
    """Classic three-state breaker: closed -> open -> half-open -> closed."""

    failure_threshold: int = 5
    recovery_seconds: float = 30.0
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _half_open: bool = field(default=False, init=False)

    @property
    def state(self) -> Literal["closed", "open", "half_open"]:
        if self._opened_at is None:
            return "closed"
        if self._half_open:
            return "half_open"
        return "open"

    def check(self, provider: str) -> None:
        """Raise if the circuit is open and the cooldown has not elapsed."""
        if self._opened_at is None:
            return
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self.recovery_seconds:
            # Let exactly one trial request through.
            self._half_open = True
            return
        raise CircuitOpenError(provider, self.recovery_seconds - elapsed)

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._half_open = False

    def record_failure(self) -> None:
        self._failures += 1
        if self._half_open:
            # The trial request failed: re-open for another full cooldown.
            self._opened_at = time.monotonic()
            self._half_open = False
            return
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


class ResilientHttpClient:
    """Async JSON HTTP client with retry, rate limiting and a circuit breaker."""

    def __init__(
        self,
        provider: str,
        base_url: str,
        config: HttpClientConfig | None = None,
        *,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.config = config or HttpClientConfig()
        self._headers = {"User-Agent": self.config.user_agent, **(default_headers or {})}
        self._client: httpx.AsyncClient | None = None
        self._limiter = RateLimiter(
            per_minute=self.config.requests_per_minute,
            per_day=self.config.requests_per_day,
            min_interval=self.config.min_interval_seconds,
        )
        self._breaker = CircuitBreaker(
            failure_threshold=self.config.failure_threshold,
            recovery_seconds=self.config.recovery_seconds,
        )
        self._inflight: dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers,
                timeout=httpx.Timeout(
                    self.config.timeout_seconds,
                    connect=self.config.connect_timeout_seconds,
                ),
                follow_redirects=False,  # a redirect off our allowlisted host is a red flag
            )
        return self._client

    async def aclose(self) -> None:
        for task in list(self._inflight.values()):
            task.cancel()
        self._inflight.clear()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def __aenter__(self) -> ResilientHttpClient:
        await self._ensure_client()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ #
    # request path                                                       #
    # ------------------------------------------------------------------ #

    @property
    def circuit_state(self) -> str:
        return self._breaker.state

    def rate_limit_snapshot(self) -> dict[str, Any]:
        return self._limiter.snapshot()

    def _backoff(self, attempt: int) -> float:
        delay = min(
            self.config.backoff_base_seconds * (2**attempt),
            self.config.backoff_max_seconds,
        )
        if self.config.backoff_jitter:
            # Full jitter: uniform in [0, delay]. Prevents synchronized retries.
            delay = random.uniform(0.0, delay)  # noqa: S311 - jitter, not cryptography
        return delay

    async def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        dedupe: bool = True,
    ) -> Any:
        """GET and parse JSON, de-duplicating concurrent identical requests."""
        if not dedupe:
            return await self._request_json("GET", path, params=params, headers=headers)

        key = f"GET {path}?{sorted((params or {}).items())}"
        async with self._lock:
            existing = self._inflight.get(key)
            if existing is not None and not existing.done():
                task = existing
            else:
                task = asyncio.create_task(
                    self._request_json("GET", path, params=params, headers=headers)
                )
                self._inflight[key] = task

        try:
            # shield=False: awaiting a shared task must not let one canceled
            # caller cancel the request for the others.
            return await asyncio.shield(task)
        finally:
            async with self._lock:
                if self._inflight.get(key) is task and task.done():
                    self._inflight.pop(key, None)

    async def post_json(
        self,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return await self._request_json("POST", path, json_body=json_body, headers=headers)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        self._breaker.check(self.provider)
        client = await self._ensure_client()
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            await self._limiter.acquire(self.provider)
            try:
                response = await client.request(
                    method, path, params=params, json=json_body, headers=headers
                )
            except httpx.TimeoutException:
                last_error = ProviderTimeoutError(self.provider, self.config.timeout_seconds)
                self._breaker.record_failure()
            except httpx.HTTPError as exc:
                # Message may embed the URL (and therefore an apikey param);
                # ProviderError redacts on construction.
                last_error = ProviderUnavailableError(self.provider, f"network error: {exc}")
                self._breaker.record_failure()
            else:
                outcome = self._classify(response)
                if outcome is None:
                    self._breaker.record_success()
                    return self._parse_json(response)
                last_error = outcome
                if not outcome.retryable:
                    self._breaker.record_failure()
                    raise outcome
                self._breaker.record_failure()
                if isinstance(outcome, ProviderRateLimitError) and outcome.retry_after_seconds:
                    await asyncio.sleep(min(outcome.retry_after_seconds, 30.0))
                    continue

            if attempt < self.config.max_retries:
                delay = self._backoff(attempt)
                log.warning(
                    "request failed; retrying",
                    extra={
                        "provider": self.provider,
                        "path": path,
                        "attempt": attempt + 1,
                        "max_attempts": self.config.max_retries + 1,
                        "delay_seconds": round(delay, 3),
                    },
                )
                await asyncio.sleep(delay)

        raise last_error or ProviderUnavailableError(self.provider, "request failed")

    def _classify(self, response: httpx.Response) -> Exception | None:
        """Map an HTTP status to an error, or ``None`` when it is a success."""
        status = response.status_code
        if 200 <= status < 300:
            return None
        if status in (401, 403):
            return ProviderAuthError(self.provider, f"authentication failed (HTTP {status})")
        if status in (429, 418):
            retry_after: float | None = None
            raw = response.headers.get("Retry-After")
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None
            return ProviderRateLimitError(
                self.provider,
                f"rate limited by provider (HTTP {status})",
                retry_after_seconds=retry_after,
            )
        if status == 402:
            return ProviderAuthError(
                self.provider,
                "endpoint requires a paid plan (HTTP 402)",
            )
        if 500 <= status < 600:
            return ProviderUnavailableError(self.provider, f"provider error HTTP {status}")
        return ProviderBadResponseError(
            self.provider,
            f"unexpected HTTP {status}",
            sample=response.text[:200],
        )

    def _parse_json(self, response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderBadResponseError(
                self.provider,
                f"response was not valid JSON: {exc}",
                sample=response.text[:200],
            ) from exc
