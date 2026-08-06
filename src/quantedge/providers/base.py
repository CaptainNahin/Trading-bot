"""Provider interfaces.

Every adapter implements one of these. The services layer depends only on these
abstractions, never on a concrete vendor, which is what makes provider
fallback, testing with fakes, and swapping AgentRouter for Anthropic possible
without touching business logic.

Contract obligations for every implementation
---------------------------------------------
1. ``enabled`` / ``credentials_present`` are honest, and a provider without
   credentials returns ``HealthStatus.DISABLED`` -- it never raises on import
   or at container start.
2. Symbols are normalized on the way in and canonical on the way out.
3. Timestamps are timezone-aware UTC.
4. Errors are :class:`~quantedge.errors.QuantEdgeError` subclasses, never raw
   vendor exceptions, so no URL or credential escapes.
5. Missing capability is reported, never faked. If a vendor has no economic
   calendar, ``supports("economic_calendar")`` is ``False`` and the event-risk
   service reports ``UNKNOWN``.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from quantedge.contracts import (
    AIDecision,
    AssetClass,
    Candle,
    CandleSeries,
    DataQualityReport,
    EconomicEvent,
    NewsItem,
    OrderBook,
    PerformanceSummary,
    ProviderHealth,
    Quote,
    ScanCandidate,
    SettledSignal,
    SymbolInfo,
    Timeframe,
    Trade,
)

__all__ = [
    "BaseProvider",
    "EconomicCalendarProvider",
    "MarketDataProvider",
    "NewsProvider",
    "PersistenceProvider",
]


class BaseProvider(abc.ABC):
    """Common lifecycle, identity and health contract."""

    name: str
    kind: str

    @property
    @abc.abstractmethod
    def enabled(self) -> bool:
        """Operator intent: should this provider be considered at all?"""

    @property
    @abc.abstractmethod
    def credentials_present(self) -> bool:
        """Whether required credentials are configured. Never exposes values."""

    @abc.abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Probe the provider and report status.

        Must not raise. Failures are returned as
        :class:`~quantedge.contracts.ProviderHealth` with an ``error`` or
        ``disabled`` status, because health checks are called from monitoring
        loops that must not crash.
        """

    def supports(self, capability: str) -> bool:
        """Whether a named capability is actually implemented and permitted."""
        return bool(self.capabilities().get(capability, False))

    def capabilities(self) -> dict[str, bool]:
        return {}

    def missing_env(self) -> list[str]:
        """Names of missing credential env vars. **Names only.**"""
        return []

    async def aclose(self) -> None:
        """Release network resources. Safe to call repeatedly."""
        return None


class MarketDataProvider(BaseProvider):
    """Quotes, candles, books, trades and (optionally) live streams."""

    kind = "market_data"
    asset_classes: tuple[AssetClass, ...] = ()

    @abc.abstractmethod
    async def list_symbols(
        self, asset_class: AssetClass | None = None
    ) -> list[SymbolInfo]:
        """Instruments this provider can serve, in canonical form."""

    @abc.abstractmethod
    async def get_quote(self, symbol: str) -> Quote:
        """Latest price observation for one symbol."""

    @abc.abstractmethod
    async def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 200,
        *,
        include_forming: bool = False,
    ) -> CandleSeries:
        """Historical candles, oldest first.

        When ``include_forming`` is ``False`` the returned series contains only
        closed bars. When ``True``, the forming bar may be appended and is
        flagged ``is_closed=False``; callers must exclude it from analysis.
        """

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        """Depth snapshot. Providers without a book must raise."""
        raise NotImplementedError(f"{self.name} does not provide order book data")

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Recent executed trades. Providers without a tape must raise."""
        raise NotImplementedError(f"{self.name} does not provide trade data")

    def stream_market_data(
        self, symbols: list[str], intervals: list[str] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield normalized live market events. Providers without streams raise."""
        raise NotImplementedError(f"{self.name} does not provide streaming data")


class EconomicCalendarProvider(BaseProvider):
    """Scheduled macroeconomic releases."""

    kind = "economic_calendar"

    @abc.abstractmethod
    async def get_events(
        self, start_utc: datetime, end_utc: datetime
    ) -> list[EconomicEvent]:
        """All events in a UTC window."""

    async def get_events_for_currencies(
        self, currencies: list[str], start_utc: datetime, end_utc: datetime
    ) -> list[EconomicEvent]:
        """Events filtered to the given currency codes.

        Default implementation filters client-side; adapters whose API supports
        server-side filtering should override to save quota.
        """
        wanted = {c.upper() for c in currencies}
        events = await self.get_events(start_utc, end_utc)
        return [e for e in events if e.currency and e.currency.upper() in wanted]

    async def get_events_near_timestamp(
        self,
        timestamp_utc: datetime,
        currencies: list[str],
        *,
        window_minutes: int = 60,
    ) -> list[EconomicEvent]:
        """Events within +/- ``window_minutes`` of a timestamp."""
        from datetime import timedelta

        delta = timedelta(minutes=window_minutes)
        return await self.get_events_for_currencies(
            currencies, timestamp_utc - delta, timestamp_utc + delta
        )


class NewsProvider(BaseProvider):
    """Market and symbol news headlines."""

    kind = "news"

    @abc.abstractmethod
    async def get_market_news(self, limit: int = 20) -> list[NewsItem]:
        """General market headlines."""

    @abc.abstractmethod
    async def get_symbol_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """Headlines for one symbol."""


class PersistenceProvider(abc.ABC):
    """Storage contract.

    Implemented twice: a SQLAlchemy repository and an in-memory fallback. The
    in-memory implementation reports ``available=False`` so no caller can
    mistake development mode for durable storage.
    """

    kind = "persistence"

    @property
    @abc.abstractmethod
    def available(self) -> bool:
        """Whether durable persistence is genuinely available."""

    @abc.abstractmethod
    async def save_snapshot(self, snapshot: dict[str, Any]) -> str | None:
        """Persist a market snapshot; returns its id."""

    @abc.abstractmethod
    async def save_candles(self, candles: list[Candle]) -> int:
        """Persist closed candles idempotently; returns the number newly stored.

        Implementations must upsert on ``(provider, symbol, timeframe,
        open_time_utc)`` so a WebSocket replay cannot duplicate history, and
        must reject candles with ``is_closed=False``.
        """

    @abc.abstractmethod
    async def save_event(self, event: EconomicEvent) -> str | None:
        """Persist an economic event."""

    @abc.abstractmethod
    async def save_candidate(self, candidate: ScanCandidate) -> str | None:
        """Persist a scanner candidate."""

    @abc.abstractmethod
    async def save_ai_decision(self, decision: AIDecision) -> str:
        """Persist an LLM decision. Append-only: never updated in place."""

    @abc.abstractmethod
    async def settle_signal(
        self, signal_id: str, *, settlement_price: Any = None
    ) -> SettledSignal:
        """Settle a previously issued decision against realized price."""

    @abc.abstractmethod
    async def get_performance(
        self, symbol: str | None = None, horizon: str | None = None
    ) -> PerformanceSummary:
        """Observed outcome statistics. Not a prediction."""

    @abc.abstractmethod
    async def find_similar_setups(
        self, feature_summary: dict[str, Any], limit: int = 5
    ) -> list[dict[str, Any]]:
        """Historically similar feature snapshots, nearest first."""

    async def save_quality_report(self, report: DataQualityReport) -> None:
        """Optional: persist a quality report for auditing."""
        return None

    async def aclose(self) -> None:
        return None
