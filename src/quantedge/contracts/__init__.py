"""Canonical data contracts for the QuantEdge gateway.

These Pydantic models are the single source of truth for every value that
crosses a boundary -- provider adapter, service, MCP tool, HTTP route or
database row. Anything that cannot be expressed as one of these models does not
leave the system.
"""

from __future__ import annotations

from quantedge.contracts.analysis import (
    STANDARD_WARNING,
    AIDecision,
    DataQualityReport,
    EconomicEvent,
    EventRiskReport,
    FeatureSnapshot,
    LLMSignalResponse,
    MultiTimeframeSnapshot,
    NewsItem,
    PerformanceSummary,
    ProviderHealth,
    RegimeReport,
    ScanCandidate,
    ScanRejection,
    ScanResult,
    SessionState,
    SettledSignal,
    SignalContext,
    StructureReport,
    TimeframeView,
)
from quantedge.contracts.enums import (
    TIMEFRAME_SECONDS,
    AssetClass,
    EventImpact,
    EventRiskStatus,
    HealthStatus,
    MarketRegime,
    QualityStatus,
    SettlementOutcome,
    SignalDirection,
    SignalStatus,
    Timeframe,
    timeframe_seconds,
)
from quantedge.contracts.market import (
    Candle,
    CandleSeries,
    OrderBook,
    OrderBookLevel,
    Quote,
    SymbolInfo,
    Trade,
    utc_now,
)

__all__ = [
    "STANDARD_WARNING",
    "TIMEFRAME_SECONDS",
    "AIDecision",
    "AssetClass",
    "Candle",
    "CandleSeries",
    "DataQualityReport",
    "EconomicEvent",
    "EventImpact",
    "EventRiskReport",
    "EventRiskStatus",
    "FeatureSnapshot",
    "HealthStatus",
    "LLMSignalResponse",
    "MarketRegime",
    "MultiTimeframeSnapshot",
    "NewsItem",
    "OrderBook",
    "OrderBookLevel",
    "PerformanceSummary",
    "ProviderHealth",
    "QualityStatus",
    "Quote",
    "RegimeReport",
    "ScanCandidate",
    "ScanRejection",
    "ScanResult",
    "SessionState",
    "SettledSignal",
    "SettlementOutcome",
    "SignalContext",
    "SignalDirection",
    "SignalStatus",
    "StructureReport",
    "SymbolInfo",
    "Timeframe",
    "TimeframeView",
    "Trade",
    "timeframe_seconds",
    "utc_now",
]
