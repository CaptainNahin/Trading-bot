"""Financial Modeling Prep economic-calendar and news adapter.

Availability, stated plainly
---------------------------
No ``FMP_API_KEY`` is configured in this deployment, so this provider reports
``disabled`` and every call raises :class:`ProviderDisabledError`. Nothing here
has been exercised against the live API.

Even with a key, ``/economic_calendar`` is **not** in FMP's free tier -- it
answers 402/403. The adapter detects that and reports ``degraded`` with the
reason, because a paywalled endpoint is a known limitation rather than an
outage, and because silently returning an empty event list would read as
"no events scheduled" and let a scan proceed straight into an NFP release.

Impact mapping
--------------
FMP labels events ``Low``/``Medium``/``High``. Those labels are carried across
verbatim. An unrecognised or absent label becomes
:attr:`EventImpact.UNKNOWN` -- never ``LOW``, because "we could not tell" and
"this release is harmless" are different claims.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from quantedge.config import get_settings, providers_config
from quantedge.contracts import (
    EconomicEvent,
    EventImpact,
    HealthStatus,
    NewsItem,
    ProviderHealth,
    utc_now,
)
from quantedge.errors import (
    ProviderAuthError,
    ProviderBadResponseError,
    ProviderDisabledError,
    QuantEdgeError,
)
from quantedge.logging import get_logger
from quantedge.providers.base import EconomicCalendarProvider
from quantedge.providers.http import HttpClientConfig, ResilientHttpClient
from quantedge.symbols import normalize_symbol

__all__ = ["FMPCalendarProvider"]

log = get_logger(__name__)

PROVIDER = "fmp"
_CREDENTIAL_ENV = "FMP_API_KEY"

# FMP's own impact vocabulary. Anything outside this map stays UNKNOWN.
_IMPACT_MAP: dict[str, EventImpact] = {
    "low": EventImpact.LOW,
    "medium": EventImpact.MEDIUM,
    "moderate": EventImpact.MEDIUM,
    "high": EventImpact.HIGH,
}


class FMPCalendarProvider(EconomicCalendarProvider):
    """Economic calendar and news from Financial Modeling Prep."""

    name = PROVIDER

    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self._api_key = settings.secret(settings.fmp_api_key)
        self._base_url = (base_url or "https://financialmodelingprep.com").rstrip("/")

        cfg = providers_config()
        defaults = cfg.get("defaults", {})
        provider_cfg = cfg.get("providers", {}).get(PROVIDER, {})
        rate = provider_cfg.get("rate_limit", {})

        self._config_enabled = bool(provider_cfg.get("enabled", True))
        self._capabilities = provider_cfg.get(
            "capabilities",
            {"economic_calendar": True, "market_news": True, "symbol_news": True},
        )
        # Set once a 402/403 is actually observed, so health stops claiming the
        # calendar is available on this plan.
        self._calendar_paywalled = False

        self._client = ResilientHttpClient(
            provider=PROVIDER,
            base_url=self._base_url,
            config=HttpClientConfig(
                timeout_seconds=float(defaults.get("timeout_seconds", 10.0)),
                connect_timeout_seconds=float(defaults.get("connect_timeout_seconds", 5.0)),
                max_retries=int(defaults.get("max_retries", 3)),
                backoff_base_seconds=float(defaults.get("backoff_base_seconds", 0.5)),
                backoff_max_seconds=float(defaults.get("backoff_max_seconds", 8.0)),
                requests_per_minute=rate.get("requests_per_minute", 10),
                requests_per_day=rate.get("requests_per_day", 250),
                min_interval_seconds=float(rate.get("min_interval_seconds", 0.0)),
            ),
        )

    # ------------------------------------------------------------------ #
    # identity / health                                                  #
    # ------------------------------------------------------------------ #

    @property
    def enabled(self) -> bool:
        return self._config_enabled and self.credentials_present

    @property
    def credentials_present(self) -> bool:
        return bool(self._api_key)

    def capabilities(self) -> dict[str, bool]:
        caps = dict(self._capabilities)
        if self._calendar_paywalled:
            # Report what this plan can actually do, not what the vendor sells.
            caps["economic_calendar"] = False
        return caps

    def missing_env(self) -> list[str]:
        return [] if self._api_key else [_CREDENTIAL_ENV]

    def _require_credentials(self) -> str:
        if not self._api_key:
            raise ProviderDisabledError(PROVIDER, [_CREDENTIAL_ENV])
        return self._api_key

    async def health_check(self) -> ProviderHealth:
        """Probe the calendar endpoint. Never raises."""
        if not self._config_enabled:
            return self._health(HealthStatus.DISABLED, "disabled in config/providers.yaml")
        if not self.credentials_present:
            return self._health(
                HealthStatus.DISABLED,
                f"credential not configured: {_CREDENTIAL_ENV}. No economic calendar "
                "is available from this provider.",
            )

        started = time.perf_counter()
        now = utc_now()
        try:
            await self.get_events(now, now)
            return self._health(
                HealthStatus.OK,
                "economic calendar reachable",
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )
        except ProviderAuthError as exc:
            # Either a bad key or the free-plan paywall; both mean no calendar.
            return self._health(HealthStatus.DEGRADED, exc.message)
        except QuantEdgeError as exc:
            return self._health(HealthStatus.ERROR, exc.message)
        except Exception as exc:  # noqa: BLE001 - health must never propagate
            return self._health(HealthStatus.ERROR, f"unexpected error: {type(exc).__name__}")

    def _health(
        self,
        status: HealthStatus,
        message: str,
        *,
        latency_ms: float | None = None,
    ) -> ProviderHealth:
        limitations = ["free plan: 250 requests/day"]
        if self._calendar_paywalled:
            limitations.insert(
                0, "economic calendar is PAYWALLED on this plan; event risk is UNKNOWN"
            )
        elif not self.credentials_present:
            limitations.insert(0, "no credential: economic calendar unavailable")
        return ProviderHealth(
            provider=PROVIDER,
            kind=self.kind,
            status=status,
            enabled=self._config_enabled,
            credentials_present=self.credentials_present,
            latency_ms=latency_ms,
            capabilities=self.capabilities(),
            message=message,
            limitations=limitations,
            missing_env=self.missing_env(),
            circuit_state=self._client.circuit_state,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------ #
    # calendar                                                           #
    # ------------------------------------------------------------------ #

    async def get_events(self, start_utc: datetime, end_utc: datetime) -> list[EconomicEvent]:
        """Scheduled releases in a UTC window, ascending by time."""
        if start_utc.tzinfo is None or end_utc.tzinfo is None:
            raise ValueError("start_utc and end_utc must be timezone-aware")
        if end_utc < start_utc:
            raise ValueError("end_utc must not precede start_utc")

        payload = await self._get(
            "/api/v3/economic_calendar",
            {
                "from": start_utc.date().isoformat(),
                "to": end_utc.date().isoformat(),
            },
        )
        if not isinstance(payload, list):
            raise ProviderBadResponseError(
                PROVIDER, f"economic_calendar returned {type(payload).__name__}, expected a list"
            )

        events: list[EconomicEvent] = []
        skipped = 0
        for raw in payload:
            event = self._parse_event(raw)
            if event is None:
                skipped += 1
                continue
            # FMP's day granularity overshoots the requested window; trim to it
            # so a caller asking about the next hour does not see tomorrow.
            if start_utc <= event.scheduled_utc <= end_utc:
                events.append(event)
        if skipped:
            log.warning("dropped unusable calendar rows", extra={"count": skipped})

        events.sort(key=lambda e: e.scheduled_utc)
        return events

    def _parse_event(self, raw: Any) -> EconomicEvent | None:
        """One calendar row, or ``None`` when it cannot be trusted."""
        if not isinstance(raw, dict):
            return None
        title = str(raw.get("event") or "").strip()
        stamp = raw.get("date")
        if not title or not stamp:
            # An event with no name or no time cannot inform a risk decision.
            return None
        try:
            scheduled = self._parse_utc(stamp)
        except ProviderBadResponseError:
            return None

        currency = raw.get("currency")
        return EconomicEvent(
            provider=PROVIDER,
            event_id=None,
            title=title,
            country=str(raw["country"]) if raw.get("country") else None,
            currency=str(currency).upper() if currency else None,
            impact=self._map_impact(raw.get("impact")),
            scheduled_utc=scheduled,
            actual=self._as_text(raw.get("actual")),
            forecast=self._as_text(raw.get("estimate")),
            previous=self._as_text(raw.get("previous")),
        )

    @staticmethod
    def _map_impact(value: Any) -> EventImpact:
        """Provider label -> our enum. Unrecognised means UNKNOWN, not LOW."""
        if not value:
            return EventImpact.UNKNOWN
        return _IMPACT_MAP.get(str(value).strip().lower(), EventImpact.UNKNOWN)

    @staticmethod
    def _as_text(value: Any) -> str | None:
        """Keep provider figures as the strings they were published as.

        Releases carry units and qualifiers ("3.2%", "-0.1M", "1.2K"). Parsing
        them to numbers would either lose the unit or invent a scale, so the
        published text is preserved verbatim.
        """
        if value is None or value == "":
            return None
        return str(value)

    @staticmethod
    def _parse_utc(value: Any) -> datetime:
        """Parse an FMP timestamp, which is UTC but written without an offset."""
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ProviderBadResponseError(
                PROVIDER, f"unparseable event timestamp: {value!r}"
            ) from exc
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)

    # ------------------------------------------------------------------ #
    # news                                                               #
    # ------------------------------------------------------------------ #

    async def get_market_news(self, limit: int = 20) -> list[NewsItem]:
        """General market headlines, newest first."""
        limit = max(1, min(limit, 250))
        payload = await self._get("/api/v3/fmp/articles", {"page": "0", "size": str(limit)})
        rows = payload.get("content") if isinstance(payload, dict) else payload
        return self._parse_news(rows, limit=limit)

    async def get_symbol_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """Headlines for one ticker, newest first."""
        limit = max(1, min(limit, 250))
        payload = await self._get(
            "/api/v3/stock_news",
            {"tickers": normalize_symbol(symbol), "limit": str(limit)},
        )
        return self._parse_news(payload, limit=limit)

    def _parse_news(self, rows: Any, *, limit: int) -> list[NewsItem]:
        if not isinstance(rows, list):
            raise ProviderBadResponseError(
                PROVIDER, f"news payload was {type(rows).__name__}, expected a list"
            )
        items: list[NewsItem] = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            headline = str(raw.get("title") or "").strip()
            stamp = raw.get("publishedDate") or raw.get("date")
            if not headline or not stamp:
                continue
            try:
                published = self._parse_utc(stamp)
            except ProviderBadResponseError:
                continue
            tickers = raw.get("symbol") or raw.get("tickers") or ""
            items.append(
                NewsItem(
                    provider=PROVIDER,
                    headline=headline,
                    summary=str(raw["text"]).strip() if raw.get("text") else None,
                    url=str(raw["url"]) if raw.get("url") else None,
                    source=str(raw["site"]) if raw.get("site") else None,
                    symbols=[t.strip().upper() for t in str(tickers).split(",") if t.strip()],
                    published_at_utc=published,
                )
            )
        items.sort(key=lambda i: i.published_at_utc, reverse=True)
        return items[:limit]

    # ------------------------------------------------------------------ #
    # transport                                                          #
    # ------------------------------------------------------------------ #

    async def _get(self, path: str, params: dict[str, str]) -> Any:
        """One authenticated GET, with the paywall recognised as such."""
        params = {**params, "apikey": self._require_credentials()}
        try:
            return await self._client.get_json(path, params=params)
        except ProviderAuthError:
            # 401/402/403 on a premium path is the plan, not a transient fault.
            if "economic_calendar" in path:
                self._calendar_paywalled = True
                log.warning(
                    "FMP economic calendar is not available on this plan; "
                    "event risk will be reported as UNKNOWN",
                    extra={"provider": PROVIDER},
                )
            raise

    def rate_limit_snapshot(self) -> dict[str, Any]:
        return self._client.rate_limit_snapshot()

    async def aclose(self) -> None:
        await self._client.aclose()
