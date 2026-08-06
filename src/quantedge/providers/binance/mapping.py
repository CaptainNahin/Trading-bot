"""Binance payload normalization.

Kept separate from transport so it can be unit-tested against recorded
fixtures with no network access -- that is what the contract test suite does.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from quantedge.contracts import (
    AssetClass,
    Candle,
    OrderBook,
    OrderBookLevel,
    Quote,
    SymbolInfo,
    Timeframe,
    Trade,
)
from quantedge.errors import ProviderBadResponseError, UnsupportedTimeframeError

__all__ = [
    "BINANCE_INTERVALS",
    "normalize_book_ticker_event",
    "normalize_depth",
    "normalize_kline",
    "normalize_kline_event",
    "normalize_symbol_info",
    "normalize_ticker",
    "normalize_trade",
    "to_binance_interval",
]

PROVIDER = "binance"

# Binance kline intervals. Note the absence of 10m -- Binance does not offer it.
# Rather than silently substituting 15m, the adapter resamples from 5m and says
# so. See BinanceRestProvider.get_candles.
BINANCE_INTERVALS: dict[Timeframe, str] = {
    Timeframe.M1: "1m",
    Timeframe.M3: "3m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.M30: "30m",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1d",
}


def to_binance_interval(timeframe: Timeframe) -> str:
    """Map a canonical timeframe to a Binance interval string."""
    try:
        return BINANCE_INTERVALS[timeframe]
    except KeyError as exc:
        raise UnsupportedTimeframeError(
            f"Binance does not offer a native '{timeframe.value}' kline interval",
            details={"native_intervals": ", ".join(BINANCE_INTERVALS.values())},
        ) from exc


def _decimal(value: Any, field: str) -> Decimal:
    """Parse a Binance numeric string into Decimal.

    Binance sends numbers as strings precisely to avoid float rounding; we keep
    that guarantee by never routing through ``float``.
    """
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ProviderBadResponseError(
            PROVIDER, f"field '{field}' is not numeric: {value!r}"
        ) from exc


def _ms_to_utc(value: Any, field: str) -> datetime:
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=UTC)
    except (TypeError, ValueError, OverflowError, OSError) as exc:
        raise ProviderBadResponseError(
            PROVIDER, f"field '{field}' is not a millisecond timestamp: {value!r}"
        ) from exc


def normalize_symbol_info(raw: dict[str, Any]) -> SymbolInfo:
    """Normalize one entry of ``/api/v3/exchangeInfo``'s ``symbols`` array."""
    if not isinstance(raw, dict) or "symbol" not in raw:
        raise ProviderBadResponseError(PROVIDER, "exchangeInfo entry missing 'symbol'")

    tick_size: Decimal | None = None
    min_qty: Decimal | None = None
    for flt in raw.get("filters", []) or []:
        if not isinstance(flt, dict):
            continue
        if flt.get("filterType") == "PRICE_FILTER" and "tickSize" in flt:
            tick_size = _decimal(flt["tickSize"], "tickSize")
        elif flt.get("filterType") == "LOT_SIZE" and "minQty" in flt:
            min_qty = _decimal(flt["minQty"], "minQty")

    return SymbolInfo(
        provider=PROVIDER,
        symbol=str(raw["symbol"]).upper(),
        provider_symbol=str(raw["symbol"]).upper(),
        asset_class=AssetClass.CRYPTO,
        base_asset=raw.get("baseAsset"),
        quote_asset=raw.get("quoteAsset"),
        tick_size=tick_size,
        min_quantity=min_qty,
        price_precision=raw.get("quotePrecision"),
        is_tradable=raw.get("status") == "TRADING",
    )


def normalize_kline(
    raw: list[Any], symbol: str, timeframe: Timeframe, *, now: datetime | None = None
) -> Candle:
    """Normalize one row of ``/api/v3/klines``.

    Binance returns a 12-element array:
    ``[openTime, open, high, low, close, volume, closeTime, quoteVolume,
    trades, takerBuyBase, takerBuyQuote, ignore]``

    Closure is derived from the clock, not asserted by the payload: REST klines
    carry no "is final" flag, so the most recent row is still forming whenever
    its ``closeTime`` is in the future.
    """
    if not isinstance(raw, list | tuple) or len(raw) < 9:
        raise ProviderBadResponseError(
            PROVIDER, f"kline row must have >=9 elements, got {len(raw) if raw else 0}"
        )

    open_time = _ms_to_utc(raw[0], "openTime")
    close_time = _ms_to_utc(raw[6], "closeTime")
    reference = now or datetime.now(UTC)

    return Candle(
        provider=PROVIDER,
        symbol=symbol.upper(),
        asset_class=AssetClass.CRYPTO,
        timeframe=timeframe,
        open_time_utc=open_time,
        close_time_utc=close_time,
        open=_decimal(raw[1], "open"),
        high=_decimal(raw[2], "high"),
        low=_decimal(raw[3], "low"),
        close=_decimal(raw[4], "close"),
        volume=_decimal(raw[5], "volume"),
        quote_volume=_decimal(raw[7], "quoteVolume"),
        trade_count=int(raw[8]) if raw[8] is not None else None,
        is_closed=close_time <= reference,
    )


def normalize_ticker(
    raw: dict[str, Any], *, book: dict[str, Any] | None = None
) -> Quote:
    """Normalize ``/api/v3/ticker/24hr``, optionally enriched with bookTicker.

    ``/api/v3/ticker/24hr`` carries no bid/ask, so ``spread`` stays ``None``
    unless a bookTicker payload is supplied. We never estimate a spread.
    """
    if not isinstance(raw, dict) or "symbol" not in raw:
        raise ProviderBadResponseError(PROVIDER, "ticker payload missing 'symbol'")

    symbol = str(raw["symbol"]).upper()
    bid = ask = None
    if book:
        if book.get("bidPrice") is not None:
            bid = _decimal(book["bidPrice"], "bidPrice")
        if book.get("askPrice") is not None:
            ask = _decimal(book["askPrice"], "askPrice")

    # closeTime is the ticker window end; it is the provider's own clock.
    provider_time = (
        _ms_to_utc(raw["closeTime"], "closeTime")
        if raw.get("closeTime") is not None
        else datetime.now(UTC)
    )

    return Quote(
        provider=PROVIDER,
        symbol=symbol,
        asset_class=AssetClass.CRYPTO,
        bid=bid,
        ask=ask,
        last=_decimal(raw["lastPrice"], "lastPrice") if raw.get("lastPrice") else None,
        volume_24h=_decimal(raw["volume"], "volume") if raw.get("volume") else None,
        change_24h_percent=(
            _decimal(raw["priceChangePercent"], "priceChangePercent")
            if raw.get("priceChangePercent") is not None
            else None
        ),
        provider_time_utc=provider_time,
    )


def normalize_depth(raw: dict[str, Any], symbol: str) -> OrderBook:
    """Normalize ``/api/v3/depth``."""
    if not isinstance(raw, dict) or "bids" not in raw or "asks" not in raw:
        raise ProviderBadResponseError(PROVIDER, "depth payload missing bids/asks")

    def levels(rows: Any, side: str) -> list[OrderBookLevel]:
        out: list[OrderBookLevel] = []
        for row in rows or []:
            if not isinstance(row, list | tuple) or len(row) < 2:
                raise ProviderBadResponseError(PROVIDER, f"malformed {side} level: {row!r}")
            out.append(
                OrderBookLevel(
                    price=_decimal(row[0], f"{side}.price"),
                    quantity=_decimal(row[1], f"{side}.quantity"),
                )
            )
        return out

    return OrderBook(
        provider=PROVIDER,
        symbol=symbol.upper(),
        asset_class=AssetClass.CRYPTO,
        bids=levels(raw.get("bids"), "bid"),
        asks=levels(raw.get("asks"), "ask"),
        last_update_id=raw.get("lastUpdateId"),
    )


def normalize_trade(raw: dict[str, Any], symbol: str) -> Trade:
    """Normalize one entry of ``/api/v3/trades``."""
    if not isinstance(raw, dict) or "price" not in raw:
        raise ProviderBadResponseError(PROVIDER, "trade payload missing 'price'")
    return Trade(
        provider=PROVIDER,
        symbol=symbol.upper(),
        asset_class=AssetClass.CRYPTO,
        trade_id=str(raw["id"]) if raw.get("id") is not None else None,
        price=_decimal(raw["price"], "price"),
        quantity=_decimal(raw["qty"], "qty"),
        is_buyer_maker=raw.get("isBuyerMaker"),
        trade_time_utc=_ms_to_utc(raw["time"], "time"),
    )


def normalize_kline_event(payload: dict[str, Any]) -> Candle:
    """Normalize a ``<symbol>@kline_<interval>`` WebSocket event.

    The stream payload has an explicit ``k.x`` boolean marking bar closure, so
    unlike REST we trust the provider's own flag. Event time ``E`` and the
    kline's own ``t``/``T`` are preserved exactly as sent.
    """
    kline = payload.get("k")
    if not isinstance(kline, dict):
        raise ProviderBadResponseError(PROVIDER, "kline event missing 'k' object")

    interval = str(kline.get("i", ""))
    timeframe = next(
        (tf for tf, native in BINANCE_INTERVALS.items() if native == interval), None
    )
    if timeframe is None:
        raise ProviderBadResponseError(PROVIDER, f"unknown kline interval '{interval}'")

    return Candle(
        provider=PROVIDER,
        symbol=str(kline.get("s", payload.get("s", ""))).upper(),
        asset_class=AssetClass.CRYPTO,
        timeframe=timeframe,
        open_time_utc=_ms_to_utc(kline["t"], "k.t"),
        close_time_utc=_ms_to_utc(kline["T"], "k.T"),
        open=_decimal(kline["o"], "k.o"),
        high=_decimal(kline["h"], "k.h"),
        low=_decimal(kline["l"], "k.l"),
        close=_decimal(kline["c"], "k.c"),
        volume=_decimal(kline["v"], "k.v"),
        quote_volume=_decimal(kline["q"], "k.q") if kline.get("q") is not None else None,
        trade_count=int(kline["n"]) if kline.get("n") is not None else None,
        is_closed=bool(kline.get("x", False)),
    )


def normalize_book_ticker_event(payload: dict[str, Any]) -> Quote:
    """Normalize a ``<symbol>@bookTicker`` WebSocket event.

    bookTicker has no event-time field on the combined stream, so the receipt
    time is used as the provider time and freshness is reported as ~0. That is
    a documented limitation, not an invented timestamp.
    """
    if "s" not in payload or "b" not in payload or "a" not in payload:
        raise ProviderBadResponseError(PROVIDER, "bookTicker event missing s/b/a fields")

    now = datetime.now(UTC)
    return Quote(
        provider=PROVIDER,
        symbol=str(payload["s"]).upper(),
        asset_class=AssetClass.CRYPTO,
        bid=_decimal(payload["b"], "b"),
        ask=_decimal(payload["a"], "a"),
        provider_time_utc=_ms_to_utc(payload["E"], "E") if payload.get("E") else now,
        received_at_utc=now,
    )
