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
from datetime import datetime, timedelta
from typing import Any, Literal, Protocol

import httpx

from quantedge.contracts import utc_now
from quantedge.errors import (
    CircuitOpenError,
    ProviderAuthError,
    ProviderBadResponseError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from quantedge.logging import get_logger

__all__ = [
    "CircuitBreaker",
    "HttpClientConfig",
    "QuotaStore",
    "RateLimiter",
    "ResilientHttpClient",
]

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


class QuotaStore(Protocol):
    """The slice of the repository this module needs, named structurally.

    Declared as a Protocol rather than importing ``SqlRepository`` so that
    ``http.py`` -- which every adapter imports -- does not drag SQLAlchemy in
    behind it, and so a test can pass a hand-written double. Both real
    repositories satisfy it without knowing this Protocol exists.
    """

    def consume_quota(
        self,
        provider: str,
        *,
        window_kind: str = ...,
        limit: int | None = ...,
        now: datetime | None = ...,
    ) -> dict[str, Any]: ...

    def quota_state(
        self, provider: str, *, window_kind: str = ..., now: datetime | None = ...
    ) -> dict[str, Any] | None: ...


def _seconds_until_utc_midnight(now: datetime) -> float:
    """How long until the persisted daily window rolls over.

    The durable counter is keyed by calendar day, so this is the honest
    ``retry_after`` for a spent daily budget -- not a flat 86400, which would
    tell a caller at 23:58 to come back tomorrow evening.
    """
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(0.0, (tomorrow - now).total_seconds())


class RateLimiter:
    """Sliding-window limiter over a minute and (optionally) a day.

    A sliding window is used rather than a fixed bucket because providers
    measure "requests in the last 60 seconds", not "requests since :00".

    ``min_interval`` adds burst spacing on top of the windows, for providers
    that additionally cap requests per second.

    The daily cap is enforced twice, deliberately. ``quota_store`` is the
    authority when one is attached: it is a row in ``provider_quota``, so a
    25-request daily budget stays spent across restarts. The in-process ``_day``
    deque remains as a backstop for when the database is unreachable -- it
    cannot be the authority, because it holds :func:`time.monotonic` values,
    which have no meaning outside the process that produced them.

    The two windows differ, and the difference is intentional. The deque slides
    over the last 24 hours; the persisted counter is keyed to the calendar day
    in UTC, because that is when providers actually reset a daily allowance. The
    persisted one is therefore the more accurate model, which is why it wins
    when both are present.
    """

    def __init__(
        self,
        per_minute: int | None = None,
        per_day: int | None = None,
        min_interval: float = 0.0,
        quota_store: QuotaStore | None = None,
        *,
        quota_durable: bool = False,
    ) -> None:
        self._per_minute = per_minute
        self._per_day = per_day
        self._min_interval = max(0.0, min_interval)
        self._minute: deque[float] = deque()
        self._day: deque[float] = deque()
        self._last_request: float | None = None
        self._lock = asyncio.Lock()
        self._quota_store = quota_store
        self._quota_durable = quota_durable
        # Latched so a database that goes down mid-run produces one warning, not
        # one per request -- but the flag is what `snapshot` reports, so the
        # degradation stays visible instead of only being in the log.
        self._quota_store_failed = False

    def _prune(self, now: float) -> None:
        while self._minute and now - self._minute[0] > 60.0:
            self._minute.popleft()
        while self._day and now - self._day[0] > 86_400.0:
            self._day.popleft()

    async def _consume_durable(self, provider: str) -> bool:
        """Claim one request against the persisted daily budget.

        Returns True when the store granted it, False when it is spent, and
        True when there is no store or the store is unreachable -- in which case
        the caller's in-process backstop is the only remaining limit. Failing
        open on an infrastructure fault is the lesser evil: refusing every
        request because the quota table is briefly unavailable turns a database
        blip into a total outage of a read-only market data gateway.
        """
        if self._quota_store is None or self._per_day is None:
            return True

        store = self._quota_store
        stamp = utc_now()
        try:
            # to_thread because the repositories are synchronous SQLAlchemy; a
            # direct call would block the event loop for every other provider.
            state = await asyncio.to_thread(
                lambda: store.consume_quota(
                    provider, window_kind="day", limit=self._per_day, now=stamp
                )
            )
        except Exception as exc:  # noqa: BLE001 - any store fault must degrade, not propagate
            if not self._quota_store_failed:
                self._quota_store_failed = True
                log.warning(
                    "durable quota unavailable; falling back to the in-process daily "
                    "counter, which resets on restart",
                    extra={"provider": provider, "error": type(exc).__name__},
                )
            return True

        if self._quota_store_failed:
            self._quota_store_failed = False
            log.info("durable quota recovered", extra={"provider": provider})

        if not state.get("allowed", True):
            raise ProviderRateLimitError(
                provider,
                f"daily request cap of {state.get('requests_allowed')} reached "
                f"({state.get('requests_made')} used); resets at 00:00 UTC",
                retry_after_seconds=_seconds_until_utc_midnight(stamp),
            )
        return True

    async def acquire(self, provider: str) -> None:
        """Wait until a request slot is available.

        The daily cap is *not* waited out -- sleeping for hours would hang the
        caller, so we raise and let the caller degrade gracefully.
        """
        # Outside the lock: this is I/O, and consume_quota is already atomic in
        # the store. Holding the asyncio lock across a database round-trip would
        # serialise every request to this provider behind it.
        await self._consume_durable(provider)

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

    def snapshot(self, provider: str | None = None) -> dict[str, Any]:
        """Current usage. ``daily_cap_durable`` says whether it survives a restart.

        Reported rather than assumed: a caller reading ``used_last_day`` off a
        process that started a minute ago is reading a number that says nothing
        about what the provider thinks has been spent today.
        """
        now = time.monotonic()
        self._prune(now)
        snap: dict[str, Any] = {
            "used_last_minute": len(self._minute),
            "limit_per_minute": self._per_minute,
            "used_last_day": len(self._day),
            "limit_per_day": self._per_day,
            "min_interval_seconds": self._min_interval or None,
            "daily_cap_durable": bool(
                self._quota_durable
                and self._quota_store is not None
                and not self._quota_store_failed
            ),
        }
        if self._quota_store is not None and provider is not None:
            try:
                state = self._quota_store.quota_state(provider, window_kind="day")
            except Exception:  # noqa: BLE001 - a health snapshot must not raise
                state = None
            if state is not None:
                snap["persisted_used_today"] = state["requests_made"]
                snap["persisted_remaining_today"] = state["remaining"]
                snap["persisted_window_start_utc"] = state["window_start_utc"]
        return snap


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
        quota_store: QuotaStore | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.config = config or HttpClientConfig()
        self._headers = {"User-Agent": self.config.user_agent, **(default_headers or {})}
        self._client: httpx.AsyncClient | None = None
        store, durable = self._resolve_quota_store(quota_store)
        self._limiter = RateLimiter(
            per_minute=self.config.requests_per_minute,
            per_day=self.config.requests_per_day,
            min_interval=self.config.min_interval_seconds,
            quota_store=store,
            quota_durable=durable,
        )
        self._breaker = CircuitBreaker(
            failure_threshold=self.config.failure_threshold,
            recovery_seconds=self.config.recovery_seconds,
        )
        self._inflight: dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()

    def _resolve_quota_store(
        self, override: QuotaStore | None
    ) -> tuple[QuotaStore | None, bool]:
        """Find the store backing the daily cap, and say whether it is durable.

        Only wired when this provider actually has a daily cap: Binance has none,
        and attaching a repository to it would put a database round-trip in front
        of every candle fetch to buy nothing.

        The import is local and the whole thing is guarded, because a provider
        adapter has to keep working when the database does not -- the fallback
        for market data is a slightly stale read, not a dead gateway. Whether the
        counter actually survives a restart is read from the repository rather
        than assumed, so a run on the in-memory backend reports
        ``daily_cap_durable: false`` instead of quietly implying otherwise.
        """
        if override is not None:
            return override, False
        if self.config.requests_per_day is None:
            return None, False
        try:
            # Local, not module-scope: importing the repository package at the top
            # of http.py would make every provider adapter import SQLAlchemy, and
            # a failed database import would then break plain market-data fetches
            # that never needed a quota row.
            from quantedge.repositories import (  # noqa: PLC0415
                describe_persistence,
                get_repository,
            )

            return get_repository(), bool(describe_persistence().get("durable"))
        except Exception as exc:  # noqa: BLE001 - never let storage break a fetch
            log.warning(
                "no durable quota store; the daily cap will reset when this process does",
                extra={"provider": self.provider, "error": type(exc).__name__},
            )
            return None, False

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
        return self._limiter.snapshot(self.provider)

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

    def _classify(self, response: httpx.Response) -> ProviderError | None:
        """Map an HTTP status to an error, or ``None`` when it is a success.

        Narrower than ``Exception`` because the retry loop reads ``.retryable``
        off the result, and that attribute exists on :class:`ProviderError`
        alone -- typing this as ``Exception`` made the retry decision unchecked.
        """
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
