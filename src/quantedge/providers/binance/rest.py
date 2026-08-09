"""Binance public REST market-data adapter.

Security boundary
-----------------
This module talks only to Binance's public market-data hosts: the configured
one, plus the mirrors it fails over to when a host refuses the deployment's
region with HTTP 451. It has no signing code, no ``recvWindow`` parameter and
no path under ``/api/v3/order``, ``/account``, ``/sapi``, ``/fapi`` or
``/margin``.

An ``X-MBX-APIKEY`` header is sent when a key is configured. On these endpoints
that header buys a higher public rate limit and nothing more: every private
Binance route *additionally* requires an HMAC ``signature`` parameter, which
this module never computes. There is deliberately no way to place an order,
read a balance or move funds through this class, and
``tests/unit/test_binance_no_private_endpoints.py`` asserts that by scanning
the source.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from quantedge.config import get_settings, providers_config
from quantedge.contracts import (
    AssetClass,
    Candle,
    CandleSeries,
    HealthStatus,
    OrderBook,
    ProviderHealth,
    Quote,
    SymbolInfo,
    Timeframe,
    Trade,
    utc_now,
)
from quantedge.errors import (
    ProviderBadResponseError,
    ProviderGeoBlockedError,
    QuantEdgeError,
    UnsupportedSymbolError,
)
from quantedge.logging import get_logger
from quantedge.providers.base import MarketDataProvider
from quantedge.providers.binance import mapping
from quantedge.providers.http import HttpClientConfig, ResilientHttpClient
from quantedge.symbols import enforce_limit, normalize_symbol

__all__ = ["BinanceRestProvider"]

log = get_logger(__name__)

# Paths that must never appear here. Enforced by test, listed for reviewers.
FORBIDDEN_PATH_FRAGMENTS = (
    "/order",
    "/account",
    "/myTrades",
    "/sapi",
    "/fapi",
    "/dapi",
    "/margin",
    "/capital",
    "/withdraw",
    "/userDataStream",
)


class BinanceRestProvider(MarketDataProvider):
    """Public spot market data from Binance. No credentials required."""

    name = "binance"
    asset_classes = (AssetClass.CRYPTO,)

    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.binance_rest_base_url).rstrip("/")

        cfg = providers_config()
        defaults = cfg.get("defaults", {})
        provider_cfg = cfg.get("providers", {}).get("binance", {})
        rate = provider_cfg.get("rate_limit", {})
        breaker = defaults.get("circuit_breaker", {})

        self._capabilities = provider_cfg.get(
            "capabilities",
            {
                "quote": True,
                "candles": True,
                "order_book": True,
                "recent_trades": True,
                "stream": True,
                "symbols": True,
            },
        )
        api_key = settings.secret(settings.binance_api_key)
        # Sent for the higher public rate-limit tier only. The key buys quota on
        # the same market-data endpoints; it is never combined with a signature,
        # so it cannot reach an order, account or wallet route even by accident.
        self._headers = {"X-MBX-APIKEY": api_key} if api_key else None

        # Alternates for HTTP 451, primary first and duplicates dropped while
        # keeping order. Only used when a host refuses the region.
        alternates = [
            h.strip().rstrip("/")
            for h in settings.binance_rest_fallback_urls.split(",")
            if h.strip()
        ]
        self._hosts = list(dict.fromkeys([self._base_url, *alternates]))

        self._client = self._build_client(self._base_url, api_key, defaults, rate)
        self._breaker_cfg = breaker
        self._symbol_cache: dict[str, SymbolInfo] = {}
        self._symbol_cache_at: float = 0.0
        self._symbol_cache_ttl = float(cfg.get("cache", {}).get("symbols_ttl_seconds", 3600))

    def _build_client(
        self,
        base_url: str,
        api_key: str | None,
        defaults: dict[str, Any],
        rate: dict[str, Any],
    ) -> ResilientHttpClient:
        """One client bound to one host, so a host swap is a fresh client."""
        return ResilientHttpClient(
            provider=self.name,
            base_url=base_url,
            default_headers={"X-MBX-APIKEY": api_key} if api_key else None,
            config=HttpClientConfig(
                timeout_seconds=float(defaults.get("timeout_seconds", 10.0)),
                connect_timeout_seconds=float(defaults.get("connect_timeout_seconds", 5.0)),
                max_retries=int(defaults.get("max_retries", 3)),
                backoff_base_seconds=float(defaults.get("backoff_base_seconds", 0.5)),
                backoff_max_seconds=float(defaults.get("backoff_max_seconds", 8.0)),
                backoff_jitter=bool(defaults.get("backoff_jitter", True)),
                requests_per_minute=rate.get("requests_per_minute"),
            ),
        )

    async def _get_json(self, path: str, **kw: Any) -> Any:
        """GET with host failover on a regional refusal.

        Every call routes through here so a geo-blocked deployment recovers on
        the first request rather than needing a redeploy with a different
        ``BINANCE_REST_BASE_URL``. Only 451 triggers a swap: a timeout or a 5xx
        is the host's problem and the retry loop already owns it, whereas a
        refusal is permanent for this region and no amount of retrying helps.
        The working host is kept so the cost is paid once, not per request.
        """
        settings = get_settings()
        api_key = settings.secret(settings.binance_api_key)
        cfg = providers_config()
        defaults = cfg.get("defaults", {})
        rate = cfg.get("providers", {}).get("binance", {}).get("rate_limit", {})

        last: ProviderGeoBlockedError | None = None
        for host in self._hosts:
            if host != self._base_url:
                log.warning(
                    "Binance host refused this region; trying %s",
                    host,
                    extra={"previous_host": self._base_url},
                )
                await self._client.aclose()
                self._client = self._build_client(host, api_key, defaults, rate)
                self._base_url = host
            try:
                return await self._client.get_json(path, **kw)
            except ProviderGeoBlockedError as exc:
                last = exc
        raise last or ProviderGeoBlockedError(self.name, "every configured host refused")

    # ------------------------------------------------------------------ #
    # identity / health                                                  #
    # ------------------------------------------------------------------ #

    @property
    def enabled(self) -> bool:
        cfg = providers_config().get("providers", {}).get("binance", {})
        return bool(cfg.get("enabled", True))

    @property
    def credentials_present(self) -> bool:
        """Always ``True``: public market data needs no credentials."""
        return True

    def capabilities(self) -> dict[str, bool]:
        return dict(self._capabilities)

    async def health_check(self) -> ProviderHealth:
        """Ping ``/api/v3/ping`` and report. Never raises."""
        if not self.enabled:
            return ProviderHealth(
                provider=self.name,
                kind=self.kind,
                status=HealthStatus.DISABLED,
                enabled=False,
                credentials_present=True,
                asset_classes=list(self.asset_classes),
                capabilities=self.capabilities(),
                message="disabled in config/providers.yaml",
            )

        started = time.perf_counter()
        try:
            await self._get_json("/api/v3/ping", dedupe=False)
            latency_ms = (time.perf_counter() - started) * 1000.0
            return ProviderHealth(
                provider=self.name,
                kind=self.kind,
                status=HealthStatus.OK,
                enabled=True,
                credentials_present=True,
                latency_ms=round(latency_ms, 2),
                asset_classes=list(self.asset_classes),
                capabilities=self.capabilities(),
                message=f"reachable at {self._base_url}",
                limitations=[
                    "public market data only; no account, order or wallet access",
                    "10m timeframe is resampled from 5m (Binance has no native 10m kline)",
                ],
                circuit_state=self._client.circuit_state,  # type: ignore[arg-type]
            )
        except QuantEdgeError as exc:
            return ProviderHealth(
                provider=self.name,
                kind=self.kind,
                status=HealthStatus.ERROR,
                enabled=True,
                credentials_present=True,
                asset_classes=list(self.asset_classes),
                capabilities=self.capabilities(),
                message=exc.message,
                circuit_state=self._client.circuit_state,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001 - health must never propagate
            return ProviderHealth(
                provider=self.name,
                kind=self.kind,
                status=HealthStatus.ERROR,
                enabled=True,
                credentials_present=True,
                asset_classes=list(self.asset_classes),
                capabilities=self.capabilities(),
                message=f"unexpected error: {type(exc).__name__}",
            )

    async def get_binance_health(self) -> ProviderHealth:
        """Alias required by the project specification."""
        return await self.health_check()

    # ------------------------------------------------------------------ #
    # symbols                                                            #
    # ------------------------------------------------------------------ #

    async def list_symbols(self, asset_class: AssetClass | None = None) -> list[SymbolInfo]:
        if asset_class is not None and asset_class is not AssetClass.CRYPTO:
            return []
        await self._refresh_symbols()
        return sorted(self._symbol_cache.values(), key=lambda s: s.symbol)

    async def list_spot_symbols(self) -> list[SymbolInfo]:
        """Alias required by the project specification."""
        return await self.list_symbols()

    async def _refresh_symbols(self, *, force: bool = False) -> None:
        """Cache ``exchangeInfo``. It is a large payload; refetch rarely."""
        age = time.monotonic() - self._symbol_cache_at
        if not force and self._symbol_cache and age < self._symbol_cache_ttl:
            return
        payload = await self._get_json("/api/v3/exchangeInfo")
        if not isinstance(payload, dict) or "symbols" not in payload:
            raise ProviderBadResponseError(self.name, "exchangeInfo missing 'symbols'")
        cache: dict[str, SymbolInfo] = {}
        for raw in payload["symbols"]:
            try:
                info = mapping.normalize_symbol_info(raw)
            except ProviderBadResponseError:
                continue  # skip one malformed entry rather than lose the catalogue
            cache[info.symbol] = info
        self._symbol_cache = cache
        self._symbol_cache_at = time.monotonic()
        log.info("binance symbol catalogue refreshed", extra={"count": len(cache)})

    async def validate_symbol(self, symbol: str) -> SymbolInfo:
        """Confirm a symbol exists and is trading on Binance."""
        canonical = normalize_symbol(symbol)
        await self._refresh_symbols()
        info = self._symbol_cache.get(canonical)
        if info is None:
            raise UnsupportedSymbolError(
                f"'{canonical}' is not listed on Binance spot",
                details={"provider": self.name},
            )
        return info

    # ------------------------------------------------------------------ #
    # quotes                                                             #
    # ------------------------------------------------------------------ #

    async def get_quote(self, symbol: str) -> Quote:
        """24h ticker enriched with best bid/ask from bookTicker."""
        canonical = normalize_symbol(symbol)
        ticker = await self._get_json("/api/v3/ticker/24hr", params={"symbol": canonical})
        book: dict[str, Any] | None = None
        try:
            book_raw = await self._get_json(
                "/api/v3/ticker/bookTicker", params={"symbol": canonical}
            )
            if isinstance(book_raw, dict):
                book = book_raw
        except QuantEdgeError as exc:
            # Degrade to a quote without bid/ask rather than failing outright.
            # spread stays None -- we never estimate one.
            log.debug("bookTicker unavailable", extra={"symbol": canonical, "reason": exc.code})
        return mapping.normalize_ticker(ticker, book=book)

    async def get_spot_quote(self, symbol: str) -> Quote:
        """Alias required by the project specification."""
        return await self.get_quote(symbol)

    async def get_price(self, symbol: str) -> Quote:
        """Lightweight ``/ticker/price`` lookup (last trade price only)."""
        canonical = normalize_symbol(symbol)
        raw = await self._get_json("/api/v3/ticker/price", params={"symbol": canonical})
        if not isinstance(raw, dict) or "price" not in raw:
            raise ProviderBadResponseError(self.name, "ticker/price missing 'price'")
        return Quote(
            provider=self.name,
            symbol=canonical,
            asset_class=AssetClass.CRYPTO,
            last=Decimal(str(raw["price"])),
            provider_time_utc=utc_now(),
        )

    async def get_book_ticker(self, symbol: str) -> Quote:
        """Best bid/ask with a real spread."""
        canonical = normalize_symbol(symbol)
        raw = await self._get_json("/api/v3/ticker/bookTicker", params={"symbol": canonical})
        if not isinstance(raw, dict):
            raise ProviderBadResponseError(self.name, "bookTicker returned a non-object")
        return mapping.normalize_book_ticker_event({**raw, "s": canonical})

    # ------------------------------------------------------------------ #
    # candles                                                            #
    # ------------------------------------------------------------------ #

    async def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 200,
        *,
        include_forming: bool = False,
    ) -> CandleSeries:
        """Fetch klines, oldest first.

        Binance has no native 10m interval. Instead of silently returning a
        different interval, we fetch 5m and aggregate two bars into one, and the
        series records that it came from ``binance`` with the requested
        timeframe. Aggregation is exact (open of first, close of last,
        max/min/sum), not interpolated.
        """
        canonical = normalize_symbol(symbol)
        count = enforce_limit(count, "max_candles_per_request", "count")

        if timeframe is Timeframe.M10:
            return await self._get_resampled_10m(canonical, count, include_forming)

        interval = mapping.to_binance_interval(timeframe)
        # Request one extra bar because the newest one is typically still forming.
        limit = min(count + 1, 1000)
        raw = await self._get_json(
            "/api/v3/klines",
            params={"symbol": canonical, "interval": interval, "limit": limit},
        )
        if not isinstance(raw, list):
            raise ProviderBadResponseError(self.name, "klines did not return a list")

        now = utc_now()
        candles = [mapping.normalize_kline(row, canonical, timeframe, now=now) for row in raw]
        return self._finalize_series(canonical, timeframe, candles, count, include_forming)

    async def get_spot_candles(self, symbol: str, interval: str, count: int = 200) -> CandleSeries:
        """Alias required by the project specification."""
        return await self.get_candles(symbol, Timeframe(interval), count)

    async def _get_resampled_10m(
        self, symbol: str, count: int, include_forming: bool
    ) -> CandleSeries:
        """Build 10m candles by aggregating pairs of 5m candles.

        Bars are aligned to even 10-minute boundaries; a leading odd bar is
        dropped rather than producing a misaligned candle.
        """
        needed_5m = min((count + 1) * 2, 1000)
        raw = await self._get_json(
            "/api/v3/klines",
            params={"symbol": symbol, "interval": "5m", "limit": needed_5m},
        )
        if not isinstance(raw, list):
            raise ProviderBadResponseError(self.name, "klines did not return a list")

        now = utc_now()
        five = [mapping.normalize_kline(row, symbol, Timeframe.M5, now=now) for row in raw]
        # Align: the first bar of a 10m group starts on an even 10-minute mark.
        while five and (five[0].open_time_utc.minute % 10) != 0:
            five.pop(0)

        merged: list[Candle] = []
        for i in range(0, len(five) - 1, 2):
            a, b = five[i], five[i + 1]
            merged.append(
                Candle(
                    provider=self.name,
                    symbol=symbol,
                    asset_class=AssetClass.CRYPTO,
                    timeframe=Timeframe.M10,
                    open_time_utc=a.open_time_utc,
                    close_time_utc=b.close_time_utc,
                    open=a.open,
                    high=max(a.high, b.high),
                    low=min(a.low, b.low),
                    close=b.close,
                    volume=(a.volume or 0) + (b.volume or 0),
                    quote_volume=(
                        (a.quote_volume or 0) + (b.quote_volume or 0)
                        if a.quote_volume is not None or b.quote_volume is not None
                        else None
                    ),
                    trade_count=(
                        (a.trade_count or 0) + (b.trade_count or 0)
                        if a.trade_count is not None or b.trade_count is not None
                        else None
                    ),
                    is_closed=a.is_closed and b.is_closed,
                )
            )
        return self._finalize_series(symbol, Timeframe.M10, merged, count, include_forming)

    def _finalize_series(
        self,
        symbol: str,
        timeframe: Timeframe,
        candles: list[Candle],
        count: int,
        include_forming: bool,
    ) -> CandleSeries:
        """Trim, drop or keep the forming bar, and wrap in a CandleSeries."""
        if not include_forming:
            candles = [c for c in candles if c.is_closed]
        candles = candles[-count:] if count < len(candles) else candles
        has_forming = bool(candles) and not candles[-1].is_closed
        return CandleSeries(
            provider=self.name,
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            timeframe=timeframe,
            candles=candles,
            includes_forming_candle=has_forming,
            source="resampled" if timeframe is Timeframe.M10 else "rest",
        )

    # ------------------------------------------------------------------ #
    # depth and trades                                                   #
    # ------------------------------------------------------------------ #

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        canonical = normalize_symbol(symbol)
        depth = enforce_limit(depth, "max_order_book_depth", "depth")
        # Binance only accepts specific limit values; round up to the next valid one.
        valid = (5, 10, 20, 50, 100, 500, 1000)
        limit = next((v for v in valid if v >= depth), 100)
        raw = await self._get_json(
            "/api/v3/depth", params={"symbol": canonical, "limit": limit}
        )
        book = mapping.normalize_depth(raw, canonical)
        if len(book.bids) > depth or len(book.asks) > depth:
            book = book.model_copy(update={"bids": book.bids[:depth], "asks": book.asks[:depth]})
        return book

    async def get_spot_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        """Alias required by the project specification."""
        return await self.get_order_book(symbol, depth)

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        canonical = normalize_symbol(symbol)
        limit = enforce_limit(limit, "max_recent_trades", "limit")
        raw = await self._get_json(
            "/api/v3/trades", params={"symbol": canonical, "limit": min(limit, 1000)}
        )
        if not isinstance(raw, list):
            raise ProviderBadResponseError(self.name, "trades did not return a list")
        return [mapping.normalize_trade(row, canonical) for row in raw]

    # ------------------------------------------------------------------ #
    # server time / lifecycle                                            #
    # ------------------------------------------------------------------ #

    async def get_server_time(self) -> datetime:
        """Binance server clock, used to measure our own clock skew."""
        raw = await self._get_json("/api/v3/time", dedupe=False)
        if not isinstance(raw, dict) or "serverTime" not in raw:
            raise ProviderBadResponseError(self.name, "time payload missing 'serverTime'")
        return datetime.fromtimestamp(int(raw["serverTime"]) / 1000.0, tz=UTC)

    def rate_limit_snapshot(self) -> dict[str, Any]:
        return self._client.rate_limit_snapshot()

    async def aclose(self) -> None:
        await self._client.aclose()
