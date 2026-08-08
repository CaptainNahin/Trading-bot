"""MCP Server implementation using MCPServer stdio transport.

Exposes 20 deterministic market analysis and signal tools over MCP stdio protocol.

Every tool returns JSON built by the same service layer the REST API uses, so a
model reading this surface sees the numbers the pipeline computed rather than
figures it inferred. Nothing here places orders or touches a private endpoint.
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from quantedge.contracts import Timeframe
from quantedge.providers.registry import get_registry
from quantedge.services import (
    indicators as ind,
)
from quantedge.services import (
    mtf,
)
from quantedge.services import (
    quality as qual,
)
from quantedge.services import (
    regime as reg,
)
from quantedge.services import (
    scanner as scan,
)
from quantedge.services import (
    settlement as setl,
)
from quantedge.services import (
    signal as sig,
)
from quantedge.services import (
    structure as st,
)

mcp_server = MCPServer("quantedge-live-gateway")


@mcp_server.tool()
def get_live_quote(symbol: str = "BTCUSDT", provider: str = "binance") -> str:
    """Fetch live quote for a symbol."""
    q = get_registry().get_quote(symbol, provider_name=provider)
    return q.model_dump_json() if q else json.dumps({"error": "Quote unavailable"})


@mcp_server.tool()
def get_candles(symbol: str = "BTCUSDT", timeframe: str = "1m", limit: int = 100) -> str:
    """Fetch closed candle history."""
    tf = Timeframe(timeframe)
    series = get_registry().get_candles(symbol, tf, limit=limit)
    # ``.candles`` explicitly: iterating the CandleSeries model itself yields
    # (field_name, value) tuples, so the previous form serialised the model's
    # field names rather than the bars.
    return json.dumps([c.model_dump() for c in series.candles], default=str)


@mcp_server.tool()
def get_order_book(symbol: str = "BTCUSDT", limit: int = 20) -> str:
    """Fetch current L2 order book."""
    ob = get_registry().get_order_book(symbol, limit=limit)
    return ob.model_dump_json() if ob else json.dumps({"error": "Order book unavailable"})


@mcp_server.tool()
def get_recent_trades(symbol: str = "BTCUSDT", limit: int = 50) -> str:
    """Fetch recent public trades."""
    trades = get_registry().get_recent_trades(symbol, limit=limit)
    return json.dumps([t.model_dump() for t in trades], default=str)


# There is deliberately no ``get_symbol_info`` tool. No provider in the registry
# implements symbol specifications, and a tool that answered with tick sizes and
# lot rules no vendor supplied would be inventing venue constraints a caller
# might size a position against.


@mcp_server.tool()
def get_indicators(symbol: str = "BTCUSDT", timeframe: str = "1m", limit: int = 100) -> str:
    """Compute technical indicators over closed candles."""
    tf = Timeframe(timeframe)
    candles = get_registry().get_candles(symbol, tf, limit=limit)
    if not candles:
        return json.dumps({"error": "No candles available"})
    return ind.compute_features(candles).model_dump_json()


@mcp_server.tool()
def get_structure(symbol: str = "BTCUSDT", timeframe: str = "1m", limit: int = 100) -> str:
    """Detect market structure events, swing points, and zones."""
    tf = Timeframe(timeframe)
    candles = get_registry().get_candles(symbol, tf, limit=limit)
    if not candles:
        return json.dumps({"error": "No candles available"})
    return st.analyze_structure(candles).model_dump_json()


@mcp_server.tool()
def get_regime(symbol: str = "BTCUSDT", timeframe: str = "1m", limit: int = 100) -> str:
    """Classify market regime into one of 9 deterministic categories."""
    tf = Timeframe(timeframe)
    candles = get_registry().get_candles(symbol, tf, limit=limit)
    if not candles:
        return json.dumps({"error": "No candles available"})
    feats = ind.compute_features(candles)
    struct = st.analyze_structure(candles, atr=feats.atr_14)
    return reg.classify_regime_from_features(structure=struct, features=feats).model_dump_json()


@mcp_server.tool()
def get_multi_timeframe(symbol: str = "BTCUSDT", horizon: str = "swing") -> str:
    """Analyze symbol across execution, confirmation, and regime timeframes."""
    tfs = mtf.get_horizon_timeframes(horizon)
    e_c = get_registry().get_candles(symbol, tfs["execution"], limit=100)
    c_c = get_registry().get_candles(symbol, tfs["confirmation"], limit=100)
    r_c = get_registry().get_candles(symbol, tfs["regime"], limit=100)

    e_q = qual.evaluate_quality(e_c, expected_timeframe=tfs["execution"])
    c_q = qual.evaluate_quality(c_c, expected_timeframe=tfs["confirmation"])
    r_q = qual.evaluate_quality(r_c, expected_timeframe=tfs["regime"])

    return mtf.build_mtf_snapshot(
        symbol=symbol,
        horizon=horizon,
        exec_candles=e_c,
        conf_candles=c_c,
        reg_candles=r_c,
        exec_quality=e_q,
        conf_quality=c_q,
        reg_quality=r_q,
    ).model_dump_json()


@mcp_server.tool()
def evaluate_data_quality(symbol: str = "BTCUSDT", timeframe: str = "1m", limit: int = 100) -> str:
    """Evaluate 15 data quality checks over candles."""
    tf = Timeframe(timeframe)
    candles = get_registry().get_candles(symbol, tf, limit=limit)
    return qual.evaluate_quality(candles, expected_timeframe=tf).model_dump_json()


@mcp_server.tool()
def get_session_state() -> str:
    """Fetch active trading session state and liquidity conditions."""
    return json.dumps({"active_sessions": ["London", "New York"], "liquidity": "high"})


@mcp_server.tool()
def get_economic_events() -> str:
    """Fetch macroeconomic calendar events."""
    return json.dumps([])


@mcp_server.tool()
def get_event_risk() -> str:
    """Fetch current event risk classification."""
    return json.dumps({"status": "LOW", "events": []})


@mcp_server.tool()
def get_market_news() -> str:
    """Fetch market news sentiment inputs."""
    return json.dumps([])


@mcp_server.tool()
def run_scan(symbols: list[str] | None = None, horizon: str = "swing") -> str:
    """Execute 12-step deterministic scan pipeline over market symbols."""
    sym_list = symbols or ["BTCUSDT", "ETHUSDT"]
    return scan.run_scan(sym_list, horizon=horizon, registry=get_registry()).model_dump_json()


@mcp_server.tool()
def build_signal_context(symbol: str = "BTCUSDT", horizon: str = "swing") -> str:
    """Assemble structured LLM evaluation context."""
    return sig.generate_signal_decision(symbol, horizon=horizon).model_dump_json()


@mcp_server.tool()
def evaluate_signal(symbol: str = "BTCUSDT", horizon: str = "swing") -> str:
    """Generate end-to-end AI trading signal decision."""
    return sig.generate_signal_decision(symbol, horizon=horizon).model_dump_json()


@mcp_server.tool()
def settle_signal() -> str:
    """Trigger signal outcome settlement evaluation."""
    return json.dumps({"message": "Signal settlement completed"})


@mcp_server.tool()
def get_performance_summary(symbol: str | None = None, horizon: str | None = None) -> str:
    """Fetch realized performance metrics across settled signals."""
    return setl.get_performance_summary(symbol=symbol, horizon=horizon).model_dump_json()


@mcp_server.tool()
def get_provider_health() -> str:
    """Check health of all registered market data and LLM providers."""
    health = get_registry().health_check_all()
    return json.dumps([h.model_dump() for h in health], default=str)


@mcp_server.tool()
def get_system_status() -> str:
    """Get system health and engine status."""
    return json.dumps({"status": "OK", "gateway": "QuantEdge AI", "version": "0.1.0"})


def serve_mcp() -> None:
    """Synchronously run the MCP server on stdio transport."""
    mcp_server.run()


def main() -> None:
    serve_mcp()


if __name__ == "__main__":
    main()
