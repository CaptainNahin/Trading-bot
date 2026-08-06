"""Canonical market data contracts.

Invariants enforced here rather than trusted downstream:

* All timestamps are timezone-aware UTC. A naive datetime is rejected, not
  coerced silently -- silent coercion is how off-by-one-timezone bugs enter
  time-series analysis.
* Prices are :class:`~decimal.Decimal`. Floats are fine for indicator math on
  arrays, but a price that round-trips through storage and JSON must not drift.
* OHLC relationships are validated at construction: ``high >= max(open, close)``,
  ``low <= min(open, close)``, ``high >= low``, and no negative prices.
* ``is_closed`` distinguishes a completed bar from the forming one. Nothing
  downstream may treat a forming bar as history.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from quantedge.contracts.enums import AssetClass, Timeframe

__all__ = [
    "Candle",
    "CandleSeries",
    "OrderBook",
    "OrderBookLevel",
    "Quote",
    "SymbolInfo",
    "Trade",
    "utc_now",
]


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def _require_utc(value: datetime | str | int | float, field_name: str) -> datetime:
    """Coerce to timezone-aware UTC, rejecting ambiguous naive input."""
    if isinstance(value, int | float):
        # Heuristic: values above 1e11 are milliseconds (year 5138 in seconds).
        seconds = value / 1000.0 if value > 1e11 else float(value)
        return datetime.fromtimestamp(seconds, tz=UTC)
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            raise ValueError(
                f"{field_name} must carry a timezone; got naive '{value}'. "
                "Providers must state their timezone explicitly."
            )
        return parsed.astimezone(UTC)
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC; got naive {value!r}")
    return value.astimezone(UTC)


NonNegativeDecimal = Annotated[Decimal, Field(ge=0)]


class _Base(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        ser_json_timedelta="iso8601",
    )


class SymbolInfo(_Base):
    """Normalized instrument metadata."""

    provider: str
    symbol: str = Field(description="Canonical uppercase symbol, e.g. BTCUSDT or EURUSD")
    provider_symbol: str = Field(description="Symbol as the provider spells it, e.g. EUR/USD")
    asset_class: AssetClass
    base_asset: str | None = None
    quote_asset: str | None = None
    tick_size: Decimal | None = None
    min_quantity: Decimal | None = None
    price_precision: int | None = None
    is_tradable: bool = True
    retrieved_at_utc: datetime = Field(default_factory=utc_now)

    @field_validator("symbol", "provider_symbol", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @field_validator("retrieved_at_utc", mode="before")
    @classmethod
    def _utc(cls, v: Any) -> datetime:
        return _require_utc(v, "retrieved_at_utc")


class Candle(_Base):
    """A single OHLCV bar.

    ``is_closed=False`` marks the currently forming bar. Such a bar may be
    displayed, but it must never enter indicator calculations or be stored as
    completed history.
    """

    provider: str
    symbol: str
    asset_class: AssetClass
    timeframe: Timeframe

    open_time_utc: datetime
    close_time_utc: datetime

    open: NonNegativeDecimal
    high: NonNegativeDecimal
    low: NonNegativeDecimal
    close: NonNegativeDecimal

    volume: Decimal | None = None
    quote_volume: Decimal | None = None
    trade_count: int | None = Field(default=None, ge=0)

    bid: Decimal | None = None
    ask: Decimal | None = None
    spread: Decimal | None = None

    is_closed: bool
    received_at_utc: datetime = Field(default_factory=utc_now)

    @field_validator("open_time_utc", "close_time_utc", "received_at_utc", mode="before")
    @classmethod
    def _utc(cls, v: Any, info: Any) -> datetime:
        return _require_utc(v, info.field_name)

    @model_validator(mode="after")
    def _validate_ohlc(self) -> Self:
        """Reject impossible bars outright.

        A provider that emits ``high < low`` is malfunctioning. Accepting the
        bar and "fixing" it would silently corrupt every downstream indicator,
        so we refuse it and let the quality engine record a provider fault.
        """
        if self.high < self.low:
            raise ValueError(f"impossible OHLC: high {self.high} < low {self.low}")
        if self.high < max(self.open, self.close):
            raise ValueError(
                f"impossible OHLC: high {self.high} below max(open,close) "
                f"{max(self.open, self.close)}"
            )
        if self.low > min(self.open, self.close):
            raise ValueError(
                f"impossible OHLC: low {self.low} above min(open,close) "
                f"{min(self.open, self.close)}"
            )
        if self.close_time_utc <= self.open_time_utc:
            raise ValueError(
                f"close_time_utc {self.close_time_utc.isoformat()} must be after "
                f"open_time_utc {self.open_time_utc.isoformat()}"
            )
        if self.bid is not None and self.ask is not None and self.ask < self.bid:
            raise ValueError(f"crossed market: ask {self.ask} < bid {self.bid}")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def range(self) -> Decimal:
        """High minus low."""
        return self.high - self.low

    @computed_field  # type: ignore[prop-decorator]
    @property
    def body(self) -> Decimal:
        """Absolute open-to-close distance."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    def age_seconds(self, *, now: datetime | None = None) -> float:
        """Seconds elapsed since this bar closed."""
        reference = now or utc_now()
        return (reference - self.close_time_utc).total_seconds()


class CandleSeries(_Base):
    """An ordered series of candles from exactly one provider.

    ``allow_cross_provider_series`` in ``config/providers.yaml`` is ``false``:
    a series never mixes sources, so a single ``provider`` field is accurate.
    """

    provider: str
    symbol: str
    asset_class: AssetClass
    timeframe: Timeframe
    candles: list[Candle]
    retrieved_at_utc: datetime = Field(default_factory=utc_now)
    includes_forming_candle: bool = False
    source: Literal["rest", "websocket_cache", "database", "resampled"] = "rest"
    """Where these bars came from.

    The quality engine treats provenance as evidence: a websocket cache can have
    a gap the venue's REST history does not, and a resampled series inherits the
    limitations of the interval it was built from.
    """

    @field_validator("retrieved_at_utc", mode="before")
    @classmethod
    def _utc(cls, v: Any) -> datetime:
        return _require_utc(v, "retrieved_at_utc")

    @model_validator(mode="after")
    def _validate_series(self) -> Self:
        if not self.candles:
            return self
        previous = self.candles[0].open_time_utc
        for candle in self.candles[1:]:
            if candle.open_time_utc <= previous:
                raise ValueError(
                    "candles must be strictly ascending by open_time_utc; "
                    f"{candle.open_time_utc.isoformat()} follows {previous.isoformat()}"
                )
            previous = candle.open_time_utc
        forming = [c for c in self.candles if not c.is_closed]
        if len(forming) > 1:
            raise ValueError(f"at most one forming candle is permitted; found {len(forming)}")
        if forming and self.candles[-1].is_closed:
            raise ValueError("the forming candle must be the last element of the series")
        return self

    @property
    def closed(self) -> list[Candle]:
        """Only completed bars -- the correct input for every indicator."""
        return [c for c in self.candles if c.is_closed]

    @property
    def forming(self) -> Candle | None:
        """The currently forming bar, if the provider supplied one."""
        if self.candles and not self.candles[-1].is_closed:
            return self.candles[-1]
        return None

    def __len__(self) -> int:
        return len(self.candles)


class Quote(_Base):
    """A point-in-time price observation."""

    provider: str
    symbol: str
    asset_class: AssetClass
    bid: Decimal | None = None
    ask: Decimal | None = None
    mid: Decimal | None = None
    last: Decimal | None = None
    spread: Decimal | None = None
    volume_24h: Decimal | None = None
    change_24h_percent: Decimal | None = None
    provider_time_utc: datetime
    received_at_utc: datetime = Field(default_factory=utc_now)

    @field_validator("provider_time_utc", "received_at_utc", mode="before")
    @classmethod
    def _utc(cls, v: Any, info: Any) -> datetime:
        return _require_utc(v, info.field_name)

    @model_validator(mode="after")
    def _derive(self) -> Self:
        """Fill mid/spread from bid+ask when the provider omitted them.

        This is arithmetic on values the provider actually sent, not
        fabrication. When only ``last`` is available (common for crypto REST
        tickers) bid/ask/spread stay ``None`` -- we do not invent a spread.
        """
        if self.bid is not None and self.ask is not None:
            if self.ask < self.bid:
                raise ValueError(f"crossed market: ask {self.ask} < bid {self.bid}")
            if self.mid is None:
                object.__setattr__(self, "mid", (self.bid + self.ask) / Decimal(2))
            if self.spread is None:
                object.__setattr__(self, "spread", self.ask - self.bid)
        for name in ("bid", "ask", "mid", "last"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"negative price: {name}={value}")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def freshness_ms(self) -> int:
        """Milliseconds between the provider's timestamp and our receipt.

        Can be slightly negative when a provider's clock runs ahead of ours;
        clamped at zero so downstream staleness checks stay monotonic.
        """
        delta = (self.received_at_utc - self.provider_time_utc).total_seconds() * 1000.0
        return max(0, int(delta))

    @property
    def reference_price(self) -> Decimal | None:
        """Best available single price: mid, else last."""
        return self.mid if self.mid is not None else self.last


class OrderBookLevel(_Base):
    price: NonNegativeDecimal
    quantity: NonNegativeDecimal


class OrderBook(_Base):
    """Depth snapshot. Bids descending, asks ascending."""

    provider: str
    symbol: str
    asset_class: AssetClass
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    last_update_id: int | None = None
    provider_time_utc: datetime | None = None
    received_at_utc: datetime = Field(default_factory=utc_now)

    @field_validator("provider_time_utc", "received_at_utc", mode="before")
    @classmethod
    def _utc(cls, v: Any, info: Any) -> datetime | None:
        return None if v is None else _require_utc(v, info.field_name)

    @model_validator(mode="after")
    def _validate_book(self) -> Self:
        if self.bids and self.asks and self.asks[0].price < self.bids[0].price:
            raise ValueError(
                f"crossed book: best ask {self.asks[0].price} < best bid {self.bids[0].price}"
            )
        return self

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid


class Trade(_Base):
    """A single executed trade as reported by the venue."""

    provider: str
    symbol: str
    asset_class: AssetClass
    trade_id: str | None = None
    price: NonNegativeDecimal
    quantity: NonNegativeDecimal
    is_buyer_maker: bool | None = None
    trade_time_utc: datetime
    received_at_utc: datetime = Field(default_factory=utc_now)

    @field_validator("trade_time_utc", "received_at_utc", mode="before")
    @classmethod
    def _utc(cls, v: Any, info: Any) -> datetime:
        return _require_utc(v, info.field_name)
