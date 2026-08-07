"""Enumerations shared across the whole system.

Every enum here is part of the public contract exposed via MCP and HTTP.
Values are lowercase/uppercase deliberately and must not be renamed casually.
"""

from __future__ import annotations

from enum import StrEnum

from quantedge.errors import UnsupportedTimeframeError

__all__ = [
    "TIMEFRAME_SECONDS",
    "AssetClass",
    "EventImpact",
    "EventRiskStatus",
    "HealthStatus",
    "MarketRegime",
    "QualityStatus",
    "SignalDirection",
    "SignalStatus",
    "Timeframe",
    "timeframe_seconds",
]


class AssetClass(StrEnum):
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    INDEX = "index"
    STOCK = "stock"


class Timeframe(StrEnum):
    """Supported analysis timeframes.

    ``M10`` is not natively offered by every venue (Binance has no 10m kline).
    Providers that lack it must resample from a lower timeframe and say so;
    they must never silently substitute a different interval.
    """

    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M10 = "10m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


TIMEFRAME_SECONDS: dict[Timeframe, int] = {
    Timeframe.M1: 60,
    Timeframe.M3: 180,
    Timeframe.M5: 300,
    Timeframe.M10: 600,
    Timeframe.M15: 900,
    Timeframe.M30: 1800,
    Timeframe.H1: 3600,
    Timeframe.H4: 14400,
    Timeframe.D1: 86400,
}


def timeframe_seconds(timeframe: Timeframe | str) -> int:
    """Duration of one bar in seconds.

    Raises
    ------
    UnsupportedTimeframeError
        If the value is not an allowlisted timeframe.
    """
    try:
        tf = Timeframe(str(timeframe).lower())
    except ValueError as exc:
        raise UnsupportedTimeframeError(
            f"unsupported timeframe '{timeframe}'",
            details={"supported": ", ".join(t.value for t in Timeframe)},
        ) from exc
    return TIMEFRAME_SECONDS[tf]


class QualityStatus(StrEnum):
    """Outcome of the data quality engine.

    ``FAIL`` is a hard gate: no candidate and no LLM analysis may be released.
    """

    PASS = "PASS"  # noqa: S105 - a quality verdict, not a credential
    DEGRADED = "DEGRADED"
    FAIL = "FAIL"


class HealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    DISABLED = "disabled"
    ERROR = "error"
    UNKNOWN = "unknown"


class MarketRegime(StrEnum):
    STRONG_UPTREND = "STRONG_UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    LOW_VOLATILITY_RANGE = "LOW_VOLATILITY_RANGE"
    HIGH_VOLATILITY_RANGE = "HIGH_VOLATILITY_RANGE"
    BREAKOUT = "BREAKOUT"
    VOLATILITY_SHOCK = "VOLATILITY_SHOCK"
    UNCERTAIN = "UNCERTAIN"


class EventRiskStatus(StrEnum):
    """Event-risk level.

    ``UNKNOWN`` is returned whenever no calendar provider could answer. It is
    never downgraded to ``LOW`` by assumption -- absence of evidence is not
    evidence of absence.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class EventImpact(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class SignalStatus(StrEnum):
    """Terminal status of an analysis request."""

    SIGNAL = "SIGNAL"
    NO_TRADE = "NO_TRADE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class SignalDirection(StrEnum):
    UP = "UP"
    DOWN = "DOWN"


class SettlementOutcome(StrEnum):
    WIN = "WIN"
    LOSS = "LOSS"
    FLAT = "FLAT"
    VOID = "VOID"
    PENDING = "PENDING"
