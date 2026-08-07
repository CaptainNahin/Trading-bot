"""Twelve Data payload normalization.

Separated from transport so it can be tested against recorded fixtures with no
network access and no API quota.

Twelve Data quirks that matter here
-----------------------------------
* Errors arrive as **HTTP 200** with ``{"status": "error", "code": ..., ...}``
  in the body. Status codes alone are not a reliability signal.
* Timestamps are returned in the *exchange's* timezone unless ``timezone=UTC``
  is requested. Every request this adapter makes pins ``timezone=UTC``, and
  parsing still attaches UTC explicitly rather than trusting the server.
* ``time_series`` returns **newest first**. Our contract is oldest-first, so
  the series is reversed on the way in.
* The most recent bar of an intraday series is the *forming* bar. Twelve Data
  does not flag it, so closure is derived from the clock, exactly as for
  Binance REST.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from quantedge.contracts import (
    AssetClass,
    Candle,
    Quote,
    SymbolInfo,
    Timeframe,
    timeframe_seconds,
)
from quantedge.errors import (
    ProviderAuthError,
    ProviderBadResponseError,
    ProviderRateLimitError,
    UnsupportedTimeframeError,
)

__all__ = [
    "TWELVE_DATA_INTERVALS",
    "normalize_candles",
    "normalize_quote",
    "normalize_symbol_info",
    "raise_for_body_error",
    "to_provider_symbol",
    "to_twelve_interval",
]

PROVIDER = "twelvedata"

# Twelve Data interval strings. Note there is no 3m and no 10m.
TWELVE_DATA_INTERVALS: dict[Timeframe, str] = {
    Timeframe.M1: "1min",
    Timeframe.M5: "5min",
    Timeframe.M15: "15min",
    Timeframe.M30: "30min",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1day",
}


def to_twelve_interval(timeframe: Timeframe) -> str:
    """Map a canonical timeframe to a Twelve Data interval string."""
    try:
        return TWELVE_DATA_INTERVALS[timeframe]
    except KeyError as exc:
        raise UnsupportedTimeframeError(
            f"Twelve Data does not offer a '{timeframe.value}' interval",
            details={"available": ", ".join(TWELVE_DATA_INTERVALS.values())},
        ) from exc


def to_provider_symbol(symbol: str, asset_class: AssetClass) -> str:
    """Convert a canonical symbol to Twelve Data's format.

    Forex pairs are slash-separated there (``EUR/USD``), while our canonical
    form is ``EURUSD``. Stocks and indices pass through unchanged.
    """
    upper = symbol.upper()
    if asset_class is AssetClass.FOREX and "/" not in upper and len(upper) == 6:
        return f"{upper[:3]}/{upper[3:]}"
    if asset_class is AssetClass.CRYPTO and "/" not in upper:
        for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
            if upper.endswith(quote) and len(upper) > len(quote):
                return f"{upper[: -len(quote)]}/{quote}"
    return upper


def raise_for_body_error(payload: Any) -> None:
    """Raise when Twelve Data reports an error inside an HTTP 200 body.

    Called before any parsing. Without it, an error body would fall through to
    the field parser and surface as a confusing "missing field" error rather
    than the real cause (bad key, exhausted quota, unknown symbol).
    """
    if not isinstance(payload, dict):
        return
    if payload.get("status") != "error":
        return

    code = payload.get("code")
    message = str(payload.get("message", "unspecified error"))

    if code in (401, 403) or "api key" in message.lower():
        raise ProviderAuthError(PROVIDER, f"authentication rejected: {message}")
    if code == 429 or "limit" in message.lower():
        raise ProviderRateLimitError(PROVIDER, f"quota exhausted: {message}")
    if code == 404:
        raise ProviderBadResponseError(PROVIDER, f"not found: {message}")
    raise ProviderBadResponseError(PROVIDER, f"provider error (code {code}): {message}")


def _decimal(value: Any, field: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ProviderBadResponseError(
            PROVIDER, f"field '{field}' is not numeric: {value!r}"
        ) from exc


def _optional_decimal(value: Any, field: str) -> Decimal | None:
    if value is None or value == "":
        return None
    return _decimal(value, field)


def _parse_datetime(value: Any, field: str) -> datetime:
    """Parse a Twelve Data timestamp as UTC.

    Requests pin ``timezone=UTC``, so a naive timestamp is UTC by construction.
    A value carrying an offset is converted rather than trusted blindly.
    """
    if value is None:
        raise ProviderBadResponseError(PROVIDER, f"field '{field}' is missing")
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                # Naive by construction: these formats carry no offset, and the
                # request pinned timezone=UTC. The return below attaches UTC
                # explicitly rather than letting a naive value escape.
                parsed = datetime.strptime(text, fmt)  # noqa: DTZ007 - timezone=UTC is pinned on the request
                break
            except ValueError:
                continue
        else:
            raise ProviderBadResponseError(
                PROVIDER, f"field '{field}' is not a parseable timestamp: {value!r}"
            ) from None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def normalize_candles(
    payload: dict[str, Any],
    symbol: str,
    asset_class: AssetClass,
    timeframe: Timeframe,
    *,
    now: datetime | None = None,
) -> list[Candle]:
    """Normalize a ``/time_series`` response into oldest-first candles.

    Twelve Data gives an open time and no close time, so the close time is
    derived from the interval length. Closure is then derived from the clock:
    a bar whose computed close time is still in the future is forming.
    """
    raise_for_body_error(payload)

    values = payload.get("values")
    if not isinstance(values, list):
        raise ProviderBadResponseError(PROVIDER, "time_series payload missing 'values'")

    reference = now or datetime.now(UTC)
    duration = timedelta(seconds=timeframe_seconds(timeframe))
    candles: list[Candle] = []

    for row in values:
        if not isinstance(row, dict):
            raise ProviderBadResponseError(PROVIDER, f"malformed candle row: {row!r}")
        open_time = _parse_datetime(row.get("datetime"), "datetime")
        close_time = open_time + duration
        candles.append(
            Candle(
                provider=PROVIDER,
                symbol=symbol.upper(),
                asset_class=asset_class,
                timeframe=timeframe,
                open_time_utc=open_time,
                close_time_utc=close_time,
                open=_decimal(row.get("open"), "open"),
                high=_decimal(row.get("high"), "high"),
                low=_decimal(row.get("low"), "low"),
                close=_decimal(row.get("close"), "close"),
                # Forex has no consolidated tape, so volume is often absent.
                # Absent stays None; it is never defaulted to zero, which would
                # read as "no trading occurred".
                volume=_optional_decimal(row.get("volume"), "volume"),
                is_closed=close_time <= reference,
            )
        )

    # Twelve Data returns newest first; our contract is oldest first.
    candles.sort(key=lambda c: c.open_time_utc)
    return candles


def normalize_quote(payload: dict[str, Any], symbol: str, asset_class: AssetClass) -> Quote:
    """Normalize a ``/quote`` response.

    ``/quote`` carries no bid or ask, so ``spread`` stays ``None``. A spread is
    never derived from the last price.

    Timestamp selection matters more than it looks. ``/quote`` describes the
    current *daily* bar, so its ``timestamp`` field is that bar's **open** time
    -- for forex, the 21:00 UTC rollover, which can be twenty hours in the past
    while the quoted price is a second old. ``last_quote_at`` is the moment the
    price itself last moved, so that is preferred; ``timestamp`` is only a
    fallback. Getting this wrong would make every live forex quote look stale to
    the quality engine and suppress otherwise valid scans.
    """
    raise_for_body_error(payload)
    if not isinstance(payload, dict) or "close" not in payload:
        raise ProviderBadResponseError(PROVIDER, "quote payload missing 'close'")

    provider_time = _quote_timestamp(payload)

    # ``percent_change`` is measured against the previous *daily* close. For
    # continuously-traded markets that bar spans a full 24 hours, so the field
    # is what it claims to be. For session-traded instruments it is a session
    # change, not a 24-hour change, and reporting it as one would misstate the
    # observation -- so it is left absent rather than relabelled.
    change_percent = payload.get("percent_change")
    is_continuous = asset_class in (AssetClass.FOREX, AssetClass.CRYPTO)
    change_24h = (
        _optional_decimal(change_percent, "percent_change")
        if is_continuous and change_percent not in (None, "")
        else None
    )

    market_open = payload.get("is_market_open")

    return Quote(
        provider=PROVIDER,
        symbol=symbol.upper(),
        asset_class=asset_class,
        last=_decimal(payload["close"], "close"),
        volume_24h=_optional_decimal(payload.get("volume"), "volume"),
        change_24h_percent=change_24h,
        is_market_open=market_open if isinstance(market_open, bool) else None,
        provider_time_utc=provider_time,
    )


def _quote_timestamp(payload: dict[str, Any]) -> datetime:
    """Best available "when was this price true" moment, most precise first."""
    for field in ("last_quote_at", "timestamp"):
        value = payload.get(field)
        if value in (None, ""):
            continue
        try:
            return datetime.fromtimestamp(int(value), tz=UTC)
        except (TypeError, ValueError, OSError) as exc:
            raise ProviderBadResponseError(
                PROVIDER, f"field '{field}' is not a valid epoch: {value!r}"
            ) from exc
    if payload.get("datetime"):
        return _parse_datetime(payload["datetime"], "datetime")
    # No provider timestamp at all. Stamping "now" would assert a freshness the
    # provider never claimed, so this is an error rather than a default.
    raise ProviderBadResponseError(
        PROVIDER, "quote payload carries no timestamp; freshness cannot be established"
    )


def normalize_symbol_info(raw: dict[str, Any], asset_class: AssetClass) -> SymbolInfo:
    """Normalize one entry from ``/forex_pairs``, ``/stocks`` or ``/indices``."""
    if not isinstance(raw, dict) or "symbol" not in raw:
        raise ProviderBadResponseError(PROVIDER, "symbol entry missing 'symbol'")

    provider_symbol = str(raw["symbol"]).upper()
    canonical = provider_symbol.replace("/", "")

    return SymbolInfo(
        provider=PROVIDER,
        symbol=canonical,
        provider_symbol=provider_symbol,
        asset_class=asset_class,
        base_asset=raw.get("currency_base"),
        quote_asset=raw.get("currency_quote") or raw.get("currency"),
        exchange=raw.get("exchange"),
        description=raw.get("name"),
        is_tradable=True,
    )
