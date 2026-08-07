"""Twelve Data REST adapter -- forex, stocks and indices.

Covers the asset classes Binance cannot: spot forex, equities and index CFDs.

Free-plan reality
-----------------
The free tier allows **8 requests/minute and 800/day**. That budget is small
enough that it shapes the design:

* The shared :class:`~quantedge.providers.http.RateLimiter` is configured from
  ``providers.yaml`` so requests queue rather than getting rejected.
* Concurrent identical requests are de-duplicated by the HTTP layer, so a
  scanner sweeping ten symbols does not spend ten calls on the same series.
* Symbol catalogues are cached aggressively; ``/forex_pairs`` changes rarely
  and is not worth a call per scan.

When the daily cap is hit the adapter raises
:class:`~quantedge.errors.ProviderRateLimitError`. It does not fall back to
stale data pretending to be live, and it does not silently return fewer bars
than requested without saying so.
"""

from __future__ import annotations

import time
from typing import Any

from quantedge.config import get_settings, providers_config
from quantedge.contracts import (
    AssetClass,
    CandleSeries,
    HealthStatus,
    ProviderHealth,
    Quote,
    SymbolInfo,
    Timeframe,
    utc_now,
)
from quantedge.errors import (
    ProviderBadResponseError,
    ProviderDisabledError,
    QuantEdgeError,
    UnsupportedSymbolError,
)
from quantedge.logging import get_logger
from quantedge.providers.base import MarketDataProvider
from quantedge.providers.http import HttpClientConfig, ResilientHttpClient
from quantedge.providers.twelvedata import mapping
from quantedge.symbols import asset_class_for, enforce_limit, normalize_symbol

__all__ = ["TwelveDataProvider"]

log = get_logger(__name__)

_CREDENTIAL_ENV = "TWELVE_DATA_API_KEY"

# Which listing endpoint serves each asset class.
_CATALOGUE_ENDPOINTS: dict[AssetClass, str] = {
    AssetClass.FOREX: "/forex_pairs",
    AssetClass.STOCK: "/stocks",
    AssetClass.INDEX: "/indices",
    AssetClass.COMMODITY: "/commodities",
}


class TwelveDataProvider(MarketDataProvider):
    """Forex, stock and index market data from Twelve Data."""

    name = "twelvedata"
    asset_classes = (
        AssetClass.FOREX,
        AssetClass.STOCK,
        AssetClass.INDEX,
        AssetClass.COMMODITY,
    )

    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self._api_key = settings.secret(settings.twelve_data_api_key)
        self._base_url = (base_url or "https://api.twelvedata.com").rstrip("/")

        cfg = providers_config()
        defaults = cfg.get("defaults", {})
        provider_cfg = cfg.get("providers", {}).get("twelvedata", {})
        rate = provider_cfg.get("rate_limit", {})

        self._config_enabled = bool(provider_cfg.get("enabled", True))
        self._capabilities = provider_cfg.get(
            "capabilities",
            {
                "quote": True,
                "candles": True,
                "symbols": True,
                "order_book": False,
                "recent_trades": False,
                "stream": False,
            },
        )
        self._client = ResilientHttpClient(
            provider=self.name,
            base_url=self._base_url,
            config=HttpClientConfig(
                timeout_seconds=float(defaults.get("timeout_seconds", 10.0)),
                connect_timeout_seconds=float(defaults.get("connect_timeout_seconds", 5.0)),
                max_retries=int(defaults.get("max_retries", 3)),
                backoff_base_seconds=float(defaults.get("backoff_base_seconds", 0.5)),
                backoff_max_seconds=float(defaults.get("backoff_max_seconds", 8.0)),
                requests_per_minute=rate.get("requests_per_minute"),
                requests_per_day=rate.get("requests_per_day"),
            ),
        )
        self._catalogue: dict[AssetClass, dict[str, SymbolInfo]] = {}
        self._catalogue_at: dict[AssetClass, float] = {}
        self._catalogue_ttl = float(cfg.get("cache", {}).get("symbols_ttl_seconds", 3600))

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
        return dict(self._capabilities)

    def missing_env(self) -> list[str]:
        return [] if self._api_key else [_CREDENTIAL_ENV]

    def _require_credentials(self) -> str:
        if not self._api_key:
            raise ProviderDisabledError(self.name, [_CREDENTIAL_ENV])
        return self._api_key

    def _params(self, **extra: Any) -> dict[str, Any]:
        """Build query params with the key and an explicit UTC pin.

        ``timezone=UTC`` is non-negotiable: without it Twelve Data returns
        exchange-local timestamps, and a series silently shifted by hours would
        corrupt every session and event-risk calculation downstream.
        """
        params: dict[str, Any] = {"apikey": self._require_credentials(), "timezone": "UTC"}
        params.update({k: v for k, v in extra.items() if v is not None})
        return params

    async def health_check(self) -> ProviderHealth:
        """Probe with a cheap quote. Never raises."""
        if not self._config_enabled:
            return self._health(HealthStatus.DISABLED, "disabled in config/providers.yaml")
        if not self.credentials_present:
            return self._health(
                HealthStatus.DISABLED, f"credential not configured: {_CREDENTIAL_ENV}"
            )

        started = time.perf_counter()
        try:
            payload = await self._client.get_json(
                "/quote", params=self._params(symbol="EUR/USD"), dedupe=False
            )
            mapping.raise_for_body_error(payload)
            latency_ms = (time.perf_counter() - started) * 1000.0
            return self._health(
                HealthStatus.OK,
                "reachable; free-tier quota applies",
                latency_ms=round(latency_ms, 2),
                limitations=[
                    "free plan: 8 requests/minute, 800/day",
                    "no order book or trade tape",
                    "no 3m or 10m interval",
                    "/quote carries no bid/ask, so spread is unavailable",
                ],
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
        limitations: list[str] | None = None,
    ) -> ProviderHealth:
        return ProviderHealth(
            provider=self.name,
            kind=self.kind,
            status=status,
            enabled=self._config_enabled,
            credentials_present=self.credentials_present,
            latency_ms=latency_ms,
            asset_classes=list(self.asset_classes),
            capabilities=self.capabilities(),
            message=message,
            limitations=limitations or [],
            missing_env=self.missing_env(),
            circuit_state=self._client.circuit_state,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------ #
    # symbols                                                            #
    # ------------------------------------------------------------------ #

    async def list_symbols(self, asset_class: AssetClass | None = None) -> list[SymbolInfo]:
        """Instrument catalogue for one asset class, or all supported ones."""
        wanted = (
            [asset_class]
            if asset_class is not None
            else [ac for ac in self.asset_classes if ac in _CATALOGUE_ENDPOINTS]
        )
        out: list[SymbolInfo] = []
        for ac in wanted:
            if ac not in _CATALOGUE_ENDPOINTS:
                continue
            try:
                await self._refresh_catalogue(ac)
            except QuantEdgeError as exc:
                # One unavailable catalogue must not empty the others.
                log.warning(
                    "catalogue unavailable",
                    extra={"provider": self.name, "asset_class": ac.value, "reason": exc.code},
                )
                continue
            out.extend(self._catalogue.get(ac, {}).values())
        return sorted(out, key=lambda s: s.symbol)

    async def _refresh_catalogue(self, asset_class: AssetClass) -> None:
        age = time.monotonic() - self._catalogue_at.get(asset_class, 0.0)
        if self._catalogue.get(asset_class) and age < self._catalogue_ttl:
            return

        endpoint = _CATALOGUE_ENDPOINTS[asset_class]
        payload = await self._client.get_json(endpoint, params=self._params())
        mapping.raise_for_body_error(payload)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ProviderBadResponseError(self.name, f"{endpoint} missing 'data' array")

        catalogue: dict[str, SymbolInfo] = {}
        for raw in payload["data"]:
            try:
                info = mapping.normalize_symbol_info(raw, asset_class)
            except ProviderBadResponseError:
                continue  # skip one bad entry rather than lose the catalogue
            catalogue[info.symbol] = info

        self._catalogue[asset_class] = catalogue
        self._catalogue_at[asset_class] = time.monotonic()
        log.info(
            "twelvedata catalogue refreshed",
            extra={"asset_class": asset_class.value, "count": len(catalogue)},
        )

    def _resolve(self, symbol: str) -> tuple[str, AssetClass, str]:
        """Return ``(canonical, asset_class, provider_symbol)``."""
        canonical = normalize_symbol(symbol)
        asset_class = asset_class_for(canonical)
        if asset_class is AssetClass.CRYPTO:
            raise UnsupportedSymbolError(
                f"'{canonical}' is crypto; route it to Binance instead",
                details={"provider": self.name},
            )
        return canonical, asset_class, mapping.to_provider_symbol(canonical, asset_class)

    # ------------------------------------------------------------------ #
    # market data                                                        #
    # ------------------------------------------------------------------ #

    async def get_quote(self, symbol: str) -> Quote:
        """Latest price. No bid/ask is available on this endpoint."""
        canonical, asset_class, provider_symbol = self._resolve(symbol)
        payload = await self._client.get_json("/quote", params=self._params(symbol=provider_symbol))
        return mapping.normalize_quote(payload, canonical, asset_class)

    async def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 200,
        *,
        include_forming: bool = False,
    ) -> CandleSeries:
        """Historical candles, oldest first, closed bars only by default."""
        canonical, asset_class, provider_symbol = self._resolve(symbol)
        count = enforce_limit(count, "max_candles_per_request", "count")
        interval = mapping.to_twelve_interval(timeframe)

        # Request one extra bar: the newest intraday bar is usually forming.
        payload = await self._client.get_json(
            "/time_series",
            params=self._params(
                symbol=provider_symbol,
                interval=interval,
                outputsize=min(count + 1, 5000),
                order="DESC",
            ),
        )
        candles = mapping.normalize_candles(
            payload, canonical, asset_class, timeframe, now=utc_now()
        )

        if not include_forming:
            candles = [c for c in candles if c.is_closed]
        if count < len(candles):
            candles = candles[-count:]
        has_forming = bool(candles) and not candles[-1].is_closed

        return CandleSeries(
            provider=self.name,
            symbol=canonical,
            asset_class=asset_class,
            timeframe=timeframe,
            candles=candles,
            includes_forming_candle=has_forming,
            source="rest",
        )

    async def validate_symbol(self, symbol: str) -> SymbolInfo:
        """Confirm Twelve Data lists a symbol."""
        canonical, asset_class, _provider_symbol = self._resolve(symbol)
        await self._refresh_catalogue(asset_class)
        info = self._catalogue.get(asset_class, {}).get(canonical)
        if info is None:
            raise UnsupportedSymbolError(
                f"'{canonical}' is not listed by Twelve Data",
                details={"provider": self.name, "asset_class": asset_class.value},
            )
        return info

    def rate_limit_snapshot(self) -> dict[str, Any]:
        return self._client.rate_limit_snapshot()

    async def aclose(self) -> None:
        await self._client.aclose()
