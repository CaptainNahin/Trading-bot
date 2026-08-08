"""Provider registry: construction, routing and fallback.

This is the seam between "which vendor answered" and everything above it. The
services layer asks for *forex candles*; it never asks for *Twelve Data*.

Routing rules, all enforced here
--------------------------------
* Order comes from ``routing:`` in ``config/providers.yaml`` -- explicit, not
  inferred from priority numbers scattered across the file.
* A provider is skipped when disabled, uncredentialled, or its circuit is open.
  Skipping is recorded with a reason; a silent skip is indistinguishable from a
  provider that was never configured.
* Fallback moves to the next provider only on *infrastructure* failure --
  timeout, outage, rate limit, bad response. It does **not** fall back when the
  request itself was invalid: an unknown symbol is unknown everywhere, and
  retrying it across four vendors just burns four quotas to produce the same
  error.
* When every provider is exhausted the caller gets a structured
  :class:`AllProvidersFailedError` naming each attempt and why it failed --
  never an empty list, which would read as "no data exists".

What this deliberately does not do
----------------------------------
It never blends two providers into one series (``allow_cross_provider_series``
is ``false``), and it never substitutes a different asset class or timeframe
than the one requested. A series stitched from two vendors has two different
notions of a bar boundary and two different clocks; the resulting indicator
values would be reproducible by nobody.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, TypeVar

from quantedge.asyncbridge import run_sync
from quantedge.config import providers_config
from quantedge.contracts import (
    AssetClass,
    CandleSeries,
    HealthStatus,
    OrderBook,
    ProviderHealth,
    Quote,
    Timeframe,
    Trade,
)
from quantedge.errors import (
    AllProvidersFailedError,
    ConfigurationError,
    ProviderError,
    QuantEdgeError,
    ValidationError,
)
from quantedge.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from quantedge.providers.base import (
        BaseProvider,
        EconomicCalendarProvider,
        MarketDataProvider,
        NewsProvider,
    )

__all__ = [
    "ProviderRegistry",
    "RouteAttempt",
    "get_registry",
    "reset_registry",
]

log = get_logger(__name__)

T = TypeVar("T")


class RouteAttempt:
    """One provider's turn at a request, and how it ended.

    Kept as a first-class record because "the scan returned nothing" and "every
    provider was rate-limited" look identical from the outside otherwise.
    """

    __slots__ = ("error_code", "outcome", "provider", "reason")

    def __init__(
        self, provider: str, outcome: str, reason: str, error_code: str | None = None
    ) -> None:
        self.provider = provider
        self.outcome = outcome  # "served" | "skipped" | "failed"
        self.reason = reason
        self.error_code = error_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "outcome": self.outcome,
            "reason": self.reason,
            "error_code": self.error_code,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<RouteAttempt {self.provider} {self.outcome}: {self.reason}>"


class ProviderRegistry:
    """Builds providers lazily and routes calls across them."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config if config is not None else providers_config()
        self._routing = self._config.get("routing", {})
        self._instances: dict[str, BaseProvider] = {}
        self._lock = asyncio.Lock()

        if not self._routing:
            raise ConfigurationError(
                "config/providers.yaml has no 'routing' section; "
                "without it there is no defined provider order"
            )

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #

    def _build(self, name: str) -> BaseProvider:
        """Instantiate one provider by name.

        Imports are local to keep a broken or heavy adapter from taking down
        the whole registry at import time -- one unavailable vendor should not
        prevent the others from serving.
        """
        match name:
            case "binance":
                from quantedge.providers.binance.rest import BinanceRestProvider

                return BinanceRestProvider()
            case "twelvedata":
                from quantedge.providers.twelvedata.rest import TwelveDataProvider

                return TwelveDataProvider()
            case "oanda":
                from quantedge.providers.oanda.rest import OandaProvider

                return OandaProvider()
            case "alphavantage":
                from quantedge.providers.alphavantage.news import AlphaVantageNewsProvider

                return AlphaVantageNewsProvider()
            case "fmp":
                from quantedge.providers.fmp.calendar import FMPCalendarProvider

                return FMPCalendarProvider()
            case "finnhub":
                from quantedge.providers.finnhub.calendar import FinnhubProvider

                return FinnhubProvider()
            case _:
                raise ConfigurationError(
                    f"unknown provider '{name}' in routing configuration",
                    details={"known": "binance, twelvedata, oanda, alphavantage, fmp, finnhub"},
                )

    def get(self, name: str) -> BaseProvider:
        """Return a provider by name, constructing it once."""
        provider = self._instances.get(name)
        if provider is None:
            provider = self._instances[name] = self._build(name)
        return provider

    # ------------------------------------------------------------------ #
    # routing tables                                                     #
    # ------------------------------------------------------------------ #

    def market_data_chain(self, asset_class: AssetClass) -> list[str]:
        """Ordered provider names for an asset class."""
        table = self._routing.get("market_data", {})
        chain = table.get(asset_class.value, [])
        if not chain:
            raise ConfigurationError(
                f"no market-data provider is configured for {asset_class.value}",
                details={"configured": ", ".join(sorted(table)) or "none"},
            )
        return list(chain)

    def calendar_chain(self) -> list[str]:
        return list(self._routing.get("economic_calendar", []))

    def news_chain(self) -> list[str]:
        return list(self._routing.get("news", []))

    # ------------------------------------------------------------------ #
    # execution                                                          #
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        chain: list[str],
        operation: Callable[[Any], Awaitable[T]],
        *,
        description: str,
        capability: str | None = None,
    ) -> tuple[T, str, list[RouteAttempt]]:
        """Try each provider in turn; return ``(result, provider, attempts)``.

        ``operation`` receives the provider instance. It must be idempotent --
        it may be invoked once per provider in the chain.
        """
        attempts: list[RouteAttempt] = []

        for name in chain:
            try:
                provider = self.get(name)
            except ConfigurationError as exc:
                attempts.append(RouteAttempt(name, "skipped", exc.message, exc.code))
                continue

            skip = self._skip_reason(provider, capability)
            if skip is not None:
                attempts.append(RouteAttempt(name, "skipped", skip))
                continue

            try:
                result = await operation(provider)
            except ValidationError as exc:
                # The request is wrong, not the provider. An unknown symbol or
                # unsupported timeframe will be just as unknown at the next
                # vendor, so failing over would waste quota to reach the same
                # answer. Surface it immediately.
                attempts.append(RouteAttempt(name, "failed", exc.message, exc.code))
                log.info(
                    "request rejected; not failing over",
                    extra={"provider": name, "operation": description, "code": exc.code},
                )
                raise
            except (ProviderError, TimeoutError) as exc:
                # Infrastructure failure: the next provider may well succeed.
                code = getattr(exc, "code", type(exc).__name__)
                message = getattr(exc, "message", str(exc))
                attempts.append(RouteAttempt(name, "failed", message, code))
                log.warning(
                    "provider failed; trying the next one",
                    extra={"provider": name, "operation": description, "code": code},
                )
                continue

            attempts.append(RouteAttempt(name, "served", "ok"))
            return result, name, attempts

        raise AllProvidersFailedError(
            chain,
            f"no provider could serve {description}",
            attempts=[a.to_dict() for a in attempts],
        )

    @staticmethod
    def _skip_reason(provider: BaseProvider, capability: str | None) -> str | None:
        """Why this provider cannot be tried, or ``None`` if it can."""
        if not provider.enabled:
            missing = provider.missing_env()
            if missing:
                return f"credentials not configured: {', '.join(missing)}"
            return "disabled in config/providers.yaml"
        if capability is not None and not provider.supports(capability):
            return f"does not support '{capability}'"
        if provider.circuit_state == "open":
            # The breaker is open precisely so we stop sending it traffic.
            return "circuit breaker is open"
        return None

    # ------------------------------------------------------------------ #
    # typed convenience wrappers                                          #
    # ------------------------------------------------------------------ #

    async def market_data(
        self,
        asset_class: AssetClass,
        operation: Callable[[MarketDataProvider], Awaitable[T]],
        *,
        description: str,
        capability: str | None = None,
    ) -> tuple[T, str, list[RouteAttempt]]:
        """Route a market-data call across the chain for an asset class."""
        return await self.execute(
            self.market_data_chain(asset_class),
            operation,  # type: ignore[arg-type]
            description=description,
            capability=capability,
        )

    async def calendar(
        self,
        operation: Callable[[EconomicCalendarProvider], Awaitable[T]],
        *,
        description: str,
    ) -> tuple[T, str, list[RouteAttempt]]:
        """Route an economic-calendar call.

        Callers must be prepared for :class:`AllProvidersFailedError`: in this
        deployment no calendar credential is configured, so every provider in
        the chain is skipped and event risk is reported ``UNKNOWN`` rather than
        assumed benign.
        """
        return await self.execute(
            self.calendar_chain(),
            operation,  # type: ignore[arg-type]
            description=description,
            capability="economic_calendar",
        )

    async def news(
        self,
        operation: Callable[[NewsProvider], Awaitable[T]],
        *,
        description: str,
        capability: str = "market_news",
    ) -> tuple[T, str, list[RouteAttempt]]:
        """Route a news call across the chain."""
        return await self.execute(
            self.news_chain(),
            operation,  # type: ignore[arg-type]
            description=description,
            capability=capability,
        )

    # ------------------------------------------------------------------ #
    # health                                                             #
    # ------------------------------------------------------------------ #

    def configured_providers(self) -> list[str]:
        """Every provider named anywhere in the routing tables, deduplicated."""
        names: list[str] = []
        for chain in self._routing.get("market_data", {}).values():
            names.extend(chain)
        names.extend(self.calendar_chain())
        names.extend(self.news_chain())
        # dict preserves insertion order, so this dedups while keeping the
        # routing order -- which is fallback priority, not an arbitrary listing.
        return list(dict.fromkeys(names))

    async def health(self) -> list[ProviderHealth]:
        """Health of every configured provider, probed concurrently.

        A provider whose ``health_check`` raises despite the contract is
        reported as an error rather than omitted -- a missing row would read as
        "not configured", which is a different and more reassuring claim.
        """
        names = self.configured_providers()

        async def probe(name: str) -> ProviderHealth:
            try:
                return await self.get(name).health_check()
            except QuantEdgeError as exc:
                return self._error_health(name, exc.message)
            except Exception as exc:  # noqa: BLE001 - one provider must not break the report
                return self._error_health(name, f"unexpected error: {type(exc).__name__}")

        return list(await asyncio.gather(*(probe(n) for n in names)))

    def _error_health(self, name: str, message: str) -> ProviderHealth:
        kind = "unknown"
        with contextlib.suppress(QuantEdgeError):
            kind = self.get(name).kind
        return ProviderHealth(
            provider=name,
            kind=kind,
            status=HealthStatus.ERROR,
            enabled=False,
            credentials_present=False,
            message=message,
        )

    def health_check_all(self) -> list[ProviderHealth]:
        """Synchronous wrapper for health across all providers."""
        return run_sync(self.health())

    def get_quote(self, symbol: str, provider_name: str = "binance") -> Quote:
        """Fetch one quote synchronously, with fallback across the chain.

        Routed through :meth:`market_data` rather than pinned to a single
        provider, so a vendor outage falls over instead of returning ``None``.
        A ``None`` return would have been indistinguishable from "this symbol
        has no price", which is a different and much stronger claim.
        """

        async def fetch() -> Quote:
            quote, _, _ = await self.market_data(
                self._asset_class_for(symbol),
                lambda p: p.get_quote(symbol),
                description=f"quote for {symbol}",
                capability="quote",
            )
            return quote

        return run_sync(fetch())

    def get_candles(
        self,
        symbol: str,
        timeframe: Any,
        limit: int = 100,
        provider_name: str | None = None,
    ) -> CandleSeries:
        """Fetch a candle series synchronously.

        Returns the :class:`CandleSeries`, not a bare list: the series carries
        the provider, symbol, timeframe and forming-bar flag that the quality
        engine needs to do its job. Handing back a list would strip exactly the
        metadata the Rule 9 and single-provider checks are built on.

        ``provider_name`` pins one vendor when supplied; otherwise the routing
        chain for the symbol's asset class is used with normal fallback.
        """
        tf = timeframe if isinstance(timeframe, Timeframe) else Timeframe(str(timeframe))

        async def fetch() -> CandleSeries:
            if provider_name is not None:
                provider = self.get(provider_name)
                return await provider.get_candles(symbol, tf, count=limit)  # type: ignore[attr-defined]
            series, _, _ = await self.market_data(
                self._asset_class_for(symbol),
                lambda p: p.get_candles(symbol, tf, count=limit),
                description=f"{limit} {tf.value} candles for {symbol}",
                capability="candles",
            )
            return series

        return run_sync(fetch())

    def get_order_book(self, symbol: str, limit: int = 20) -> OrderBook:
        """Fetch one L2 order book snapshot synchronously.

        Only providers advertising the ``order_book`` capability are tried; a
        vendor that cannot serve depth is skipped rather than asked and failed.
        """

        async def fetch() -> OrderBook:
            book, _, _ = await self.market_data(
                self._asset_class_for(symbol),
                lambda p: p.get_order_book(symbol, depth=limit),
                description=f"order book for {symbol}",
                capability="order_book",
            )
            return book

        return run_sync(fetch())

    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Fetch recent public trades synchronously."""

        async def fetch() -> list[Trade]:
            trades, _, _ = await self.market_data(
                self._asset_class_for(symbol),
                lambda p: p.get_recent_trades(symbol, limit=limit),
                description=f"recent trades for {symbol}",
                capability="recent_trades",
            )
            return trades

        return run_sync(fetch())

    @staticmethod
    def _asset_class_for(symbol: str) -> AssetClass:
        """Resolve a symbol's asset class from the symbol registry.

        Consulted rather than inferred from the string: guessing "anything
        ending USDT is crypto" would route a mislabelled forex pair to a crypto
        exchange and report the resulting 400 as a missing symbol.
        """
        from quantedge.symbols import asset_class_for

        return asset_class_for(symbol)

    async def aclose(self) -> None:
        """Close every constructed provider's HTTP client."""
        for provider in list(self._instances.values()):
            try:
                await provider.aclose()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                log.warning(
                    "provider failed to close cleanly",
                    extra={"provider": provider.name, "error": type(exc).__name__},
                )
        self._instances.clear()


_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Process-wide registry singleton shared by MCP, API and workers."""
    global _registry  # noqa: PLW0603 - deliberate process-level singleton
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def reset_registry() -> None:
    """Drop the singleton. For tests and configuration reloads."""
    global _registry  # noqa: PLW0603 - deliberate process-level singleton
    _registry = None
