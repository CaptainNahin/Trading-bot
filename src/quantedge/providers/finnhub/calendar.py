"""Finnhub economic-calendar and news adapter.

Availability, stated plainly
----------------------------
No ``FINNHUB_API_KEY`` is configured in this deployment, so this provider
reports ``disabled`` and every call raises :class:`ProviderDisabledError`.
Nothing here has been exercised against the live API.

Even with a key, ``/calendar/economic`` requires a **premium** plan and answers
403 on free keys; general and company news are free. The adapter therefore
tracks calendar availability separately from news availability, so a working
news feed never gets misread as a working calendar.

Impact mapping
--------------
Finnhub reports impact as ``low``/``medium``/``high``, sometimes numerically
(1-3). Both forms are mapped; anything else becomes
:attr:`EventImpact.UNKNOWN` rather than defaulting to ``LOW``.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
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

__all__ = ["FinnhubProvider"]

log = get_logger(__name__)

PROVIDER = "finnhub"
_CREDENTIAL_ENV = "FINNHUB_API_KEY"

_IMPACT_MAP: dict[str, EventImpact] = {
    "low": EventImpact.LOW,
    "1": EventImpact.LOW,
    "medium": EventImpact.MEDIUM,
    "2": EventImpact.MEDIUM,
    "high": EventImpact.HIGH,
    "3": EventImpact.HIGH,
}


class FinnhubProvider(EconomicCalendarProvider):
    """Economic calendar (premium) and news (free) from Finnhub."""

    name = PROVIDER

    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self._api_key = settings.secret(settings.finnhub_api_key)
        self._base_url = (base_url or "https://finnhub.io/api/v1").rstrip("/")

        cfg = providers_config()
        defaults = cfg.get("defaults", {})
        provider_cfg = cfg.get("providers", {}).get(PROVIDER, {})
        rate = provider_cfg.get("rate_limit", {})

        self._config_enabled = bool(provider_cfg.get("enabled", True))
        self._capabilities = provider_cfg.get(
            "capabilities",
            {"economic_calendar": True, "market_news": True, "symbol_news": True},
        )
        self._calendar_premium_gated = False

        self._client = ResilientHttpClient(
            provider=PROVIDER,
            base_url=self._base_url,
            config=HttpClientConfig(
                timeout_seconds=float(defaults.get("timeout_seconds", 10.0)),
                connect_timeout_seconds=float(defaults.get("connect_timeout_seconds", 5.0)),
                max_retries=int(defaults.get("max_retries", 3)),
                backoff_base_seconds=float(defaults.get("backoff_base_seconds", 0.5)),
                backoff_max_seconds=float(defaults.get("backoff_max_seconds", 8.0)),
                requests_per_minute=rate.get("requests_per_minute", 60),
                requests_per_day=rate.get("requests_per_day"),
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
        if self._calendar_premium_gated:
            caps["economic_calendar"] = False
        return caps

    def missing_env(self) -> list[str]:
        return [] if self._api_key else [_CREDENTIAL_ENV]

    def _require_credentials(self) -> str:
        if not self._api_key:
            raise ProviderDisabledError(PROVIDER, missing_env=[_CREDENTIAL_ENV])
        return self._api_key

    async def health_check(self) -> ProviderHealth:
        """Probe the calendar, then fall back to probing news. Never raises.

        The two are probed separately on purpose: on a free key the calendar is
        403 while news is fine, and reporting the whole provider as broken would
        lose a news source we can actually use.
        """
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
        except ProviderAuthError as calendar_exc:
            # Calendar is gated. Check whether news still works before judging.
            try:
                await self.get_market_news(limit=1)
            except QuantEdgeError:
                return self._health(HealthStatus.ERROR, calendar_exc.message)
            return self._health(
                HealthStatus.DEGRADED,
                "news available; economic calendar requires a premium plan",
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )
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
        limitations: list[str] = []
        if self._calendar_premium_gated:
            limitations.append(
                "economic calendar is PREMIUM-gated on this key; event risk is UNKNOWN"
            )
        elif not self.credentials_present:
            limitations.append("no credential: economic calendar unavailable")
        limitations.append("free plan: 60 requests/minute")
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
            "/calendar/economic",
            {"from": start_utc.date().isoformat(), "to": end_utc.date().isoformat()},
        )
        rows = payload.get("economicCalendar") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ProviderBadResponseError(
                PROVIDER, f"calendar payload was {type(rows).__name__}, expected a list"
            )

        events: list[EconomicEvent] = []
        skipped = 0
        for raw in rows:
            event = self._parse_event(raw)
            if event is None:
                skipped += 1
                continue
            if start_utc <= event.scheduled_utc <= end_utc:
                events.append(event)
        if skipped:
            log.warning("dropped unusable calendar rows", extra={"count": skipped})

        events.sort(key=lambda e: e.scheduled_utc)
        return events

    def _parse_event(self, raw: Any) -> EconomicEvent | None:
        if not isinstance(raw, dict):
            return None
        title = str(raw.get("event") or "").strip()
        stamp = raw.get("time")
        if not title or not stamp:
            return None
        try:
            scheduled = self._parse_utc(stamp)
        except ProviderBadResponseError:
            return None

        country = raw.get("country")
        return EconomicEvent(
            provider=PROVIDER,
            event_id=None,
            title=title,
            country=str(country) if country else None,
            # Finnhub keys events by country code, not currency. Deriving a
            # currency from a country would be a guess (the euro area alone
            # spans twenty of them), so it is left absent and the event-risk
            # service matches on country instead.
            currency=None,
            impact=self._map_impact(raw.get("impact")),
            scheduled_utc=scheduled,
            actual=self._as_text(raw.get("actual")),
            forecast=self._as_text(raw.get("estimate")),
            previous=self._as_text(raw.get("prev")),
        )

    @staticmethod
    def _map_impact(value: Any) -> EventImpact:
        if value is None or value == "":
            return EventImpact.UNKNOWN
        return _IMPACT_MAP.get(str(value).strip().lower(), EventImpact.UNKNOWN)

    @staticmethod
    def _as_text(value: Any) -> str | None:
        """Preserve the published figure verbatim, units and all."""
        if value is None or value == "":
            return None
        return str(value)

    @staticmethod
    def _parse_utc(value: Any) -> datetime:
        """Parse a Finnhub timestamp (epoch seconds or ``YYYY-MM-DD HH:MM:SS`` UTC)."""
        if isinstance(value, int | float):
            return datetime.fromtimestamp(float(value), tz=UTC)
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
        payload = await self._get("/news", {"category": "general"})
        return self._parse_news(payload, limit=limit)

    async def get_symbol_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """Company news for one ticker over the last week, newest first."""
        limit = max(1, min(limit, 250))
        now = utc_now()
        payload = await self._get(
            "/company-news",
            {
                "symbol": normalize_symbol(symbol),
                "from": (now - timedelta(days=7)).date().isoformat(),
                "to": now.date().isoformat(),
            },
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
            headline = str(raw.get("headline") or "").strip()
            stamp = raw.get("datetime")
            if not headline or not stamp:
                continue
            try:
                published = self._parse_utc(stamp)
            except ProviderBadResponseError:
                continue
            related = raw.get("related") or ""
            items.append(
                NewsItem(
                    provider=PROVIDER,
                    headline=headline,
                    summary=str(raw["summary"]).strip() if raw.get("summary") else None,
                    url=str(raw["url"]) if raw.get("url") else None,
                    source=str(raw["source"]) if raw.get("source") else None,
                    symbols=[s.strip().upper() for s in str(related).split(",") if s.strip()],
                    published_at_utc=published,
                )
            )
        items.sort(key=lambda i: i.published_at_utc, reverse=True)
        return items[:limit]

    # ------------------------------------------------------------------ #
    # transport                                                          #
    # ------------------------------------------------------------------ #

    async def _get(self, path: str, params: dict[str, str]) -> Any:
        """One authenticated GET, recognising the premium gate as such."""
        params = {**params, "token": self._require_credentials()}
        try:
            return await self._client.get_json(path, params=params)
        except ProviderAuthError:
            if "calendar" in path:
                self._calendar_premium_gated = True
                log.warning(
                    "Finnhub economic calendar requires a premium plan; "
                    "event risk will be reported as UNKNOWN",
                    extra={"provider": PROVIDER},
                )
            raise

    def rate_limit_snapshot(self) -> dict[str, Any]:
        return self._client.rate_limit_snapshot()

    async def aclose(self) -> None:
        await self._client.aclose()
