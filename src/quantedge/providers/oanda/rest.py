"""OANDA v20 read-only pricing adapter.

Availability, stated plainly
----------------------------
Neither ``OANDA_API_TOKEN`` nor ``OANDA_ACCOUNT_ID`` is configured in this
deployment, so this provider reports ``disabled`` and every call raises
:class:`ProviderDisabledError`. Nothing here has been exercised against the
live API; the request shapes follow OANDA's published v20 pricing spec.

Deliberate scope limit
----------------------
Only two endpoint families are implemented:

* ``GET /accounts/{id}/pricing``   -- current bid/ask
* ``GET /instruments/{ins}/candles`` -- historical candles
* ``GET /accounts/{id}/instruments`` -- tradable instrument catalogue

Order placement, trade management, position closing and transaction endpoints
are **not implemented and must not be added**. This service analyses markets; it
does not touch an account. The base URL defaults to the practice host so a
mistake cannot reach a live account.

Why OANDA is the preferred forex source when configured
-------------------------------------------------------
It quotes a real bid and ask, so the spread is *observed* rather than absent.
Twelve Data's ``/quote`` carries neither, and a spread that has to be guessed is
one that must be reported as unavailable -- which blocks any spread-sensitive
decision downstream.

Candle closure
--------------
OANDA marks each candle with a ``complete`` boolean. That flag is used directly
rather than being re-derived from the clock, because the venue knows its own
session boundaries and weekend gaps better than a timer does.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from quantedge.config import get_settings, providers_config
from quantedge.contracts import (
    AssetClass,
    Candle,
    CandleSeries,
    HealthStatus,
    ProviderHealth,
    Quote,
    SymbolInfo,
    Timeframe,
    timeframe_seconds,
)
from quantedge.errors import (
    ProviderBadResponseError,
    ProviderDisabledError,
    QuantEdgeError,
    UnsupportedSymbolError,
    UnsupportedTimeframeError,
)
from quantedge.logging import get_logger
from quantedge.providers.base import MarketDataProvider
from quantedge.providers.http import HttpClientConfig, ResilientHttpClient
from quantedge.symbols import asset_class_for, enforce_limit, normalize_symbol

__all__ = ["OandaProvider"]

log = get_logger(__name__)

PROVIDER = "oanda"
_CREDENTIAL_ENV = ["OANDA_API_TOKEN", "OANDA_ACCOUNT_ID"]

_HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

# OANDA granularity codes. There is no 3m and no 10m.
_GRANULARITY: dict[Timeframe, str] = {
    Timeframe.M1: "M1",
    Timeframe.M5: "M5",
    Timeframe.M15: "M15",
    Timeframe.M30: "M30",
    Timeframe.H1: "H1",
    Timeframe.H4: "H4",
    Timeframe.D1: "D",
}


class OandaProvider(MarketDataProvider):
    """Read-only forex, index and commodity pricing from OANDA v20."""

    name = PROVIDER
    asset_classes = (AssetClass.FOREX, AssetClass.COMMODITY, AssetClass.INDEX)

    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self._token = settings.secret(settings.oanda_api_token)
        self._account_id = settings.secret(settings.oanda_account_id)

        # Validated as Literal["practice", "live"] by Settings, so no fallback
        # branch is needed here. Practice is the default: a misconfiguration
        # must not be able to point market-data reads at a live account.
        self._environment = settings.oanda_environment
        self._base_url = (base_url or _HOSTS[self._environment]).rstrip("/")

        cfg = providers_config()
        defaults = cfg.get("defaults", {})
        provider_cfg = cfg.get("providers", {}).get(PROVIDER, {})
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
            provider=PROVIDER,
            base_url=self._base_url,
            config=HttpClientConfig(
                timeout_seconds=float(defaults.get("timeout_seconds", 10.0)),
                connect_timeout_seconds=float(defaults.get("connect_timeout_seconds", 5.0)),
                max_retries=int(defaults.get("max_retries", 3)),
                backoff_base_seconds=float(defaults.get("backoff_base_seconds", 0.5)),
                backoff_max_seconds=float(defaults.get("backoff_max_seconds", 8.0)),
                requests_per_minute=rate.get("requests_per_minute", 100),
                min_interval_seconds=float(rate.get("min_interval_seconds", 0.0)),
            ),
            default_headers={"Accept-Datetime-Format": "RFC3339"},
        )
        self._catalogue: dict[str, SymbolInfo] = {}
        self._catalogue_at: float = 0.0
        self._catalogue_ttl = float(cfg.get("cache", {}).get("symbols_ttl_seconds", 3600))

    # ------------------------------------------------------------------ #
    # identity / health                                                  #
    # ------------------------------------------------------------------ #

    @property
    def enabled(self) -> bool:
        return self._config_enabled and self.credentials_present

    @property
    def credentials_present(self) -> bool:
        return bool(self._token and self._account_id)

    def capabilities(self) -> dict[str, bool]:
        return dict(self._capabilities)

    def missing_env(self) -> list[str]:
        missing = []
        if not self._token:
            missing.append("OANDA_API_TOKEN")
        if not self._account_id:
            missing.append("OANDA_ACCOUNT_ID")
        return missing

    def _require_credentials(self) -> tuple[str, str]:
        if not self.credentials_present:
            raise ProviderDisabledError(PROVIDER, self.missing_env())
        assert self._token is not None and self._account_id is not None
        return self._token, self._account_id

    async def health_check(self) -> ProviderHealth:
        """Probe the instrument catalogue. Never raises."""
        if not self._config_enabled:
            return self._health(HealthStatus.DISABLED, "disabled in config/providers.yaml")
        if not self.credentials_present:
            return self._health(
                HealthStatus.DISABLED,
                f"credentials not configured: {', '.join(self.missing_env())}",
            )

        started = time.perf_counter()
        try:
            await self._refresh_catalogue(force=True)
            return self._health(
                HealthStatus.OK,
                f"reachable ({self._environment}); {len(self._catalogue)} instruments",
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
        return ProviderHealth(
            provider=PROVIDER,
            kind=self.kind,
            status=status,
            enabled=self._config_enabled,
            credentials_present=self.credentials_present,
            latency_ms=latency_ms,
            asset_classes=list(self.asset_classes),
            capabilities=self.capabilities(),
            message=message,
            limitations=[
                f"environment: {self._environment}",
                "READ-ONLY: no order, trade, position or transaction endpoint is implemented",
                "no order book or trade tape (OANDA is not an order-book venue)",
                "no 3m or 10m granularity",
            ],
            missing_env=self.missing_env(),
            circuit_state=self._client.circuit_state,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------ #
    # transport                                                          #
    # ------------------------------------------------------------------ #

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        token, _account = self._require_credentials()
        return await self._client.get_json(
            path, params=params, headers={"Authorization": f"Bearer {token}"}
        )

    def _resolve(self, symbol: str) -> tuple[str, AssetClass, str]:
        """Return ``(canonical, asset_class, oanda_instrument)``."""
        canonical = normalize_symbol(symbol)
        asset_class = asset_class_for(canonical)
        if asset_class is AssetClass.CRYPTO:
            raise UnsupportedSymbolError(
                f"'{canonical}' is crypto; route it to Binance instead",
                details={"provider": PROVIDER},
            )
        return canonical, asset_class, self._to_instrument(canonical)

    @staticmethod
    def _to_instrument(canonical: str) -> str:
        """``EURUSD`` -> ``EUR_USD``. Non-pair symbols pass through unchanged."""
        upper = canonical.upper()
        if "_" in upper:
            return upper
        if len(upper) == 6 and upper.isalpha():
            return f"{upper[:3]}_{upper[3:]}"
        return upper

    # ------------------------------------------------------------------ #
    # symbols                                                            #
    # ------------------------------------------------------------------ #

    async def list_symbols(self, asset_class: AssetClass | None = None) -> list[SymbolInfo]:
        await self._refresh_catalogue()
        out = list(self._catalogue.values())
        if asset_class is not None:
            out = [s for s in out if s.asset_class is asset_class]
        return sorted(out, key=lambda s: s.symbol)

    async def _refresh_catalogue(self, *, force: bool = False) -> None:
        if (
            not force
            and self._catalogue
            and time.monotonic() - self._catalogue_at < self._catalogue_ttl
        ):
            return
        _token, account_id = self._require_credentials()
        payload = await self._get(f"/v3/accounts/{account_id}/instruments")
        instruments = payload.get("instruments") if isinstance(payload, dict) else None
        if not isinstance(instruments, list):
            raise ProviderBadResponseError(PROVIDER, "instruments payload missing 'instruments'")

        catalogue: dict[str, SymbolInfo] = {}
        for raw in instruments:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            instrument = str(raw["name"])
            canonical = instrument.replace("_", "")
            base, _, quote = instrument.partition("_")
            catalogue[canonical] = SymbolInfo(
                provider=PROVIDER,
                symbol=canonical,
                provider_symbol=instrument,
                asset_class=self._map_asset_class(raw.get("type")),
                base_asset=base or None,
                quote_asset=quote or None,
                price_precision=self._as_int(raw.get("displayPrecision")),
                tick_size=self._as_decimal(raw.get("minimumTrailingStopDistance")),
                min_quantity=self._as_decimal(raw.get("minimumTradeSize")),
                description=str(raw["displayName"]) if raw.get("displayName") else None,
                is_tradable=True,
            )
        self._catalogue = catalogue
        self._catalogue_at = time.monotonic()
        log.info("oanda catalogue refreshed", extra={"count": len(catalogue)})

    @staticmethod
    def _map_asset_class(oanda_type: Any) -> AssetClass:
        """OANDA groups everything as CURRENCY, CFD or METAL.

        CFD covers both indices and commodities with no way to tell them apart
        from this field alone, so it maps to INDEX and the canonical symbol
        registry -- which does know -- has the final say elsewhere.
        """
        match str(oanda_type or "").upper():
            case "CURRENCY":
                return AssetClass.FOREX
            case "METAL":
                return AssetClass.COMMODITY
            case _:
                return AssetClass.INDEX

    # ------------------------------------------------------------------ #
    # market data                                                        #
    # ------------------------------------------------------------------ #

    async def get_quote(self, symbol: str) -> Quote:
        """Current bid/ask. OANDA quotes a real spread, unlike Twelve Data."""
        canonical, asset_class, instrument = self._resolve(symbol)
        _token, account_id = self._require_credentials()
        payload = await self._get(f"/v3/accounts/{account_id}/pricing", {"instruments": instrument})
        prices = payload.get("prices") if isinstance(payload, dict) else None
        if not isinstance(prices, list) or not prices:
            raise ProviderBadResponseError(PROVIDER, f"no price returned for {instrument}")

        price = prices[0]
        bid = self._best(price.get("bids"))
        ask = self._best(price.get("asks"))
        if bid is None or ask is None:
            raise ProviderBadResponseError(
                PROVIDER, f"pricing for {instrument} carried no bid/ask ladder"
            )

        tradeable = price.get("tradeable")
        return Quote(
            provider=PROVIDER,
            symbol=canonical,
            asset_class=asset_class,
            bid=bid,
            ask=ask,
            # mid and spread are derived by the contract from bid+ask, which is
            # arithmetic on observed values, not fabrication.
            is_market_open=bool(tradeable) if isinstance(tradeable, bool) else None,
            provider_time_utc=self._parse_utc(price.get("time")),
        )

    @staticmethod
    def _best(ladder: Any) -> Decimal | None:
        """Top of a price ladder, or ``None`` when the ladder is empty."""
        if not isinstance(ladder, list) or not ladder:
            return None
        top = ladder[0]
        if not isinstance(top, dict) or top.get("price") in (None, ""):
            return None
        try:
            return Decimal(str(top["price"]))
        except InvalidOperation:
            return None

    async def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 200,
        *,
        include_forming: bool = False,
    ) -> CandleSeries:
        """Historical candles, oldest first, closed bars only by default."""
        canonical, asset_class, instrument = self._resolve(symbol)
        count = enforce_limit(count, "max_candles_per_request", "count")
        granularity = _GRANULARITY.get(timeframe)
        if granularity is None:
            raise UnsupportedTimeframeError(
                f"OANDA does not offer a '{timeframe.value}' granularity",
                details={"available": ", ".join(_GRANULARITY.values())},
            )

        payload = await self._get(
            f"/v3/instruments/{instrument}/candles",
            {
                "granularity": granularity,
                "count": min(count + 1, 5000),
                # Mid prices: the bid/ask series would need two calls and the
                # indicator engine is defined on mid throughout.
                "price": "M",
            },
        )
        rows = payload.get("candles") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ProviderBadResponseError(PROVIDER, "candles payload missing 'candles'")

        candles: list[Candle] = []
        duration = timedelta(seconds=timeframe_seconds(timeframe))
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            mid = raw.get("mid")
            if not isinstance(mid, dict):
                continue
            open_time = self._parse_utc(raw.get("time"))
            # OANDA states completeness explicitly; trust the venue over a timer.
            is_closed = bool(raw.get("complete"))
            candles.append(
                Candle(
                    provider=PROVIDER,
                    symbol=canonical,
                    asset_class=asset_class,
                    timeframe=timeframe,
                    open_time_utc=open_time,
                    close_time_utc=open_time + duration,
                    open=self._decimal(mid.get("o"), "o"),
                    high=self._decimal(mid.get("h"), "h"),
                    low=self._decimal(mid.get("l"), "l"),
                    close=self._decimal(mid.get("c"), "c"),
                    # OANDA reports tick count, not traded size. It is recorded
                    # as trade_count; using it as volume would misstate units.
                    trade_count=self._as_int(raw.get("volume")),
                    is_closed=is_closed,
                )
            )

        candles.sort(key=lambda c: c.open_time_utc)
        if not include_forming:
            candles = [c for c in candles if c.is_closed]
        if count < len(candles):
            candles = candles[-count:]

        return CandleSeries(
            provider=PROVIDER,
            symbol=canonical,
            asset_class=asset_class,
            timeframe=timeframe,
            candles=candles,
            includes_forming_candle=bool(candles) and not candles[-1].is_closed,
            source="rest",
        )

    async def validate_symbol(self, symbol: str) -> SymbolInfo:
        """Confirm OANDA lists an instrument."""
        canonical, _asset_class, _instrument = self._resolve(symbol)
        await self._refresh_catalogue()
        info = self._catalogue.get(canonical)
        if info is None:
            raise UnsupportedSymbolError(
                f"'{canonical}' is not listed by OANDA",
                details={"provider": PROVIDER},
            )
        return info

    # ------------------------------------------------------------------ #
    # parsing helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_utc(value: Any) -> datetime:
        """Parse an RFC3339 timestamp. OANDA sends nanosecond precision."""
        if value in (None, ""):
            raise ProviderBadResponseError(PROVIDER, "missing timestamp")
        text = str(value).strip().replace("Z", "+00:00")
        # Python parses at most 6 fractional digits; OANDA sends 9.
        if "." in text:
            head, _, tail = text.partition(".")
            digits = "".join(c for c in tail if c.isdigit())
            offset = tail[len(digits) :]
            text = f"{head}.{digits[:6]}{offset}"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ProviderBadResponseError(PROVIDER, f"unparseable timestamp: {value!r}") from exc
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)

    @staticmethod
    def _decimal(value: Any, field: str) -> Decimal:
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ProviderBadResponseError(
                PROVIDER, f"field '{field}' is not numeric: {value!r}"
            ) from exc

    @staticmethod
    def _as_decimal(value: Any) -> Decimal | None:
        if value in (None, ""):
            return None
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None

    @staticmethod
    def _as_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def rate_limit_snapshot(self) -> dict[str, Any]:
        return self._client.rate_limit_snapshot()

    async def aclose(self) -> None:
        await self._client.aclose()
