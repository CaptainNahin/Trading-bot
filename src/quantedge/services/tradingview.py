"""TradingView MCP Connector & Market Intelligence Service.

Connects QuantEdge directly to the TradingView MCP engine (via tradingview-mcp)
to provide institutional technical analysis, pivot points, Bollinger squeeze
detection, multi-timeframe structure, and global market scanners across
Crypto (Binance), Commodities & Forex (OANDA / FX_IDC), and Equities (NASDAQ / NYSE).
"""

from __future__ import annotations

from typing import Any

from quantedge.logging import get_logger

log = get_logger(__name__)

# Common asset symbol to venue/exchange routing mapping
_EXCHANGE_MAP: dict[str, str] = {
    # Crypto
    "BTCUSDT": "BINANCE",
    "ETHUSDT": "BINANCE",
    "SOLUSDT": "BINANCE",
    "BNBUSDT": "BINANCE",
    "XRPUSDT": "BINANCE",
    "DOGEUSDT": "BINANCE",
    "ADAUSDT": "BINANCE",
    "AVAXUSDT": "BINANCE",
    "DOTUSDT": "BINANCE",
    "NEARUSDT": "BINANCE",
    "LINKUSDT": "BINANCE",
    "MATICUSDT": "BINANCE",
    "SUIUSDT": "BINANCE",
    "PEPEUSDT": "BINANCE",
    # Commodities & Precious Metals
    "XAUUSD": "OANDA",
    "GOLD": "OANDA",
    "XAGUSD": "OANDA",
    "SILVER": "OANDA",
    "BRENT": "TVC",
    "WTI": "TVC",
    # Major Forex Pairs
    "EURUSD": "FX_IDC",
    "GBPUSD": "FX_IDC",
    "USDJPY": "FX_IDC",
    "AUDUSD": "FX_IDC",
    "USDCAD": "FX_IDC",
    "USDCHF": "FX_IDC",
    "NZDUSD": "FX_IDC",
    # Equities & Indices
    "SPY": "AMEX",
    "QQQ": "NASDAQ",
    "AAPL": "NASDAQ",
    "NVDA": "NASDAQ",
    "TSLA": "NASDAQ",
    "MSFT": "NASDAQ",
    "AMZN": "NASDAQ",
    "GOOGL": "NASDAQ",
    "META": "NASDAQ",
}


def resolve_tradingview_target(symbol: str, exchange: str | None = None) -> tuple[str, str]:
    """Resolve symbol and exchange for TradingView queries.

    Parameters
    ----------
    symbol : str
        Input symbol (e.g. 'BTC', 'BTCUSDT', 'XAUUSD', 'GOLD', 'NVDA').
    exchange : str | None
        Optional explicit exchange override.

    Returns
    -------
    tuple[str, str]
        Clean TradingView symbol and exchange (e.g. ('BTCUSDT', 'BINANCE')).
    """
    clean_sym = symbol.strip().upper().replace("/", "").replace("-", "")
    if clean_sym in ("BTC", "BITCOIN"):
        clean_sym = "BTCUSDT"
    elif clean_sym in ("ETH", "ETHEREUM"):
        clean_sym = "ETHUSDT"
    elif clean_sym in ("SOL", "SOLANA"):
        clean_sym = "SOLUSDT"
    elif clean_sym in ("GOLD", "XAU"):
        clean_sym = "XAUUSD"
    elif clean_sym in ("SILVER", "XAG"):
        clean_sym = "XAGUSD"

    if exchange:
        return clean_sym, exchange.upper()

    if clean_sym in _EXCHANGE_MAP:
        return clean_sym, _EXCHANGE_MAP[clean_sym]

    if clean_sym.endswith("USDT") or clean_sym.endswith("BUSD"):
        return clean_sym, "BINANCE"

    if len(clean_sym) == 6 and clean_sym[3:] in ("USD", "EUR", "GBP", "JPY", "CHF", "CAD"):
        return clean_sym, "FX_IDC"

    # Default fallback for equity tickers
    return clean_sym, "NASDAQ"


def normalize_timeframe(tf: str) -> str:
    """Normalize timeframe string to TradingView supported intervals."""
    tf_clean = tf.strip().lower()
    if "15" in tf_clean or "quarter" in tf_clean:
        return "15m"
    if "30" in tf_clean:
        return "30m"
    if "5" in tf_clean and "m" in tf_clean:
        return "5m"
    if "1" in tf_clean and "m" in tf_clean and "1h" not in tf_clean:
        return "1m"
    if "4" in tf_clean and "h" in tf_clean:
        return "4h"
    if "2" in tf_clean and "h" in tf_clean:
        return "2h"
    if "60" in tf_clean or ("1" in tf_clean and "h" in tf_clean):
        return "1h"
    if "d" in tf_clean or "day" in tf_clean:
        return "1D"
    if "w" in tf_clean or "week" in tf_clean:
        return "1W"
    return "15m"


def get_tradingview_analysis(
    symbol: str,
    timeframe: str = "15m",
    exchange: str | None = None,
) -> dict[str, Any]:
    """Retrieve comprehensive TradingView technical analysis and indicators.

    Includes RSI, Bollinger Bands (squeeze/position), Pivot Points (R1-R3, S1-S3),
    200 EMA structure, momentum, and aggregate market sentiment.
    """
    sym, venue = resolve_tradingview_target(symbol, exchange)
    tf = normalize_timeframe(timeframe)

    try:
        from tradingview_mcp.core.services.screener_service import analyze_coin

        raw = analyze_coin(sym, venue, tf)
        if not raw or not isinstance(raw, dict):
            return {"status": "error", "error": f"No data returned for {sym} on {venue}"}

        price_data = raw.get("price_data") or {}
        rsi_data = raw.get("rsi") or {}
        macd_data = raw.get("macd") or {}
        bb_data = raw.get("bollinger_bands") or {}
        sr_data = raw.get("support_resistance") or {}
        struct_data = raw.get("market_structure") or {}
        sent_data = raw.get("market_sentiment") or {}

        return {
            "status": "ok",
            "symbol": sym,
            "exchange": venue,
            "timeframe": tf,
            "price": price_data.get("close"),
            "rsi": {
                "value": rsi_data.get("value"),
                "signal": rsi_data.get("signal"),
                "direction": rsi_data.get("direction"),
            },
            "macd": {
                "macd": macd_data.get("macd"),
                "signal": macd_data.get("signal"),
                "histogram": macd_data.get("histogram"),
                "cross": macd_data.get("cross"),
            },
            "bollinger_bands": {
                "upper": bb_data.get("upper"),
                "middle": bb_data.get("middle"),
                "lower": bb_data.get("lower"),
                "squeeze": bb_data.get("squeeze", False),
                "position": bb_data.get("position"),
            },
            "pivots": {
                "pivot": sr_data.get("pivot"),
                "r1": sr_data.get("resistance_1"),
                "r2": sr_data.get("resistance_2"),
                "r3": sr_data.get("resistance_3"),
                "s1": sr_data.get("support_1"),
                "s2": sr_data.get("support_2"),
                "s3": sr_data.get("support_3"),
                "nearest_resistance": sr_data.get("nearest_resistance"),
                "nearest_support": sr_data.get("nearest_support"),
                "distance_to_resistance_pct": sr_data.get("distance_to_resistance_pct"),
                "distance_to_support_pct": sr_data.get("distance_to_support_pct"),
            },
            "market_structure": {
                "trend": struct_data.get("trend"),
                "trend_score": struct_data.get("trend_score"),
                "trend_strength": struct_data.get("trend_strength"),
                "signals": struct_data.get("trend_signals", []),
                "candle": struct_data.get("candle"),
            },
            "sentiment": {
                "rating": sent_data.get("overall_rating"),
                "signal": sent_data.get("buy_sell_signal"),
                "volatility": sent_data.get("volatility"),
                "momentum": sent_data.get("momentum"),
            },
        }
    except Exception as exc:
        log.warning("TradingView analysis lookup failed", extra={"symbol": sym, "error": str(exc)})
        return {"status": "error", "symbol": sym, "exchange": venue, "error": str(exc)}


def get_tradingview_multi_timeframe(
    symbol: str,
    exchange: str | None = None,
) -> dict[str, Any]:
    """Retrieve multi-timeframe consensus (1W, 1D, 4h, 1h, 15m) from TradingView."""
    sym, venue = resolve_tradingview_target(symbol, exchange)
    try:
        from tradingview_mcp.core.services.screener_service import run_multi_timeframe_analysis

        raw = run_multi_timeframe_analysis(sym, venue)
        if not raw or not isinstance(raw, dict):
            return {"status": "error", "error": f"No MTF data returned for {sym}"}

        return {
            "status": "ok",
            "symbol": sym,
            "exchange": venue,
            "alignment": raw.get("alignment"),
            "recommendation": raw.get("recommendation"),
            "timeframes": raw.get("timeframes", {}),
        }
    except Exception as exc:
        log.warning("TradingView MTF analysis failed", extra={"symbol": sym, "error": str(exc)})
        return {"status": "error", "symbol": sym, "error": str(exc)}


def scan_tradingview_gainers(exchange: str = "BINANCE") -> list[dict[str, Any]]:
    """Scan top momentum gainers on an exchange using TradingView's screener."""
    try:
        from tradingview_mcp.core.services.scanner_service import volume_breakout_scan

        res = volume_breakout_scan(exchange=exchange.upper())
        if isinstance(res, list):
            return res[:10]
        if isinstance(res, dict) and "results" in res:
            return res["results"][:10]
        return []
    except Exception as exc:
        log.warning(
            "TradingView gainer scan failed",
            extra={"exchange": exchange, "error": str(exc)},
        )
        return []


def format_tradingview_summary(analysis: dict[str, Any]) -> str:
    """Format TradingView analysis into an executive textual summary."""
    if analysis.get("status") != "ok":
        return f"TradingView analysis unavailable: {analysis.get('error', 'unknown')}"

    sym = analysis.get("symbol")
    venue = analysis.get("exchange")
    tf = analysis.get("timeframe")
    price = analysis.get("price")
    rsi = analysis.get("rsi", {})
    bb = analysis.get("bollinger_bands", {})
    pivots = analysis.get("pivots", {})
    struct = analysis.get("market_structure", {})
    sent = analysis.get("sentiment", {})

    squeeze_str = " (SQUEEZE ACTIVE - Breakout imminent)" if bb.get("squeeze") else ""

    lines = [
        f"**TradingView Intelligence [{sym} on {venue} ({tf})]:**",
        f"- **Price:** {price} | **Trend:** {struct.get('trend')} "
        f"({struct.get('trend_strength', 'N/A')})",
        f"- **Sentiment:** {sent.get('signal')} (Rating: {sent.get('rating')}) | "
        f"**Momentum:** {sent.get('momentum')}",
        f"- **RSI ({tf}):** {rsi.get('value')} ({rsi.get('signal')}, {rsi.get('direction')})",
        f"- **Bollinger Bands:** Pos: {bb.get('position')}{squeeze_str}",
        f"- **Key Pivots:** Pivot: {pivots.get('pivot')} | S1: {pivots.get('s1')} | "
        f"R1: {pivots.get('r1')}",
    ]
    if pivots.get("nearest_resistance"):
        dist_res = pivots.get("distance_to_resistance_pct")
        dist_sup = pivots.get("distance_to_support_pct")
        lines.append(
            f"- **Levels:** Nearest Resistance {pivots.get('nearest_resistance')} (+{dist_res}%), "
            f"Nearest Support {pivots.get('nearest_support')} (-{dist_sup}%)"
        )
    return "\n".join(lines)
