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
    "POLUSDT": "BINANCE",
    "SUIUSDT": "BINANCE",
    "PEPEUSDT": "BINANCE",
    "SHIBUSDT": "BINANCE",
    "TRXUSDT": "BINANCE",
    "LTCUSDT": "BINANCE",
    # Commodities & Precious Metals
    "XAUUSD": "OANDA",
    "GOLD": "OANDA",
    "XAGUSD": "OANDA",
    "SILVER": "OANDA",
    "BRENT": "TVC",
    "UKOIL": "TVC",
    "WTI": "TVC",
    "USOIL": "TVC",
    "WTICOUSD": "TVC",
    "NATGAS": "TVC",
    "COPPER": "TVC",
    "PLATINUM": "TVC",
    # Indices
    "SPX": "TVC",
    "SP500": "TVC",
    "NDX": "TVC",
    "NAS100": "TVC",
    "DJI": "TVC",
    "US30": "TVC",
    "DAX": "TVC",
    "FTSE": "TVC",
    "DXY": "TVC",
    # Major & Exotic Forex Pairs
    "EURUSD": "FX_IDC",
    "GBPUSD": "FX_IDC",
    "USDJPY": "FX_IDC",
    "AUDUSD": "FX_IDC",
    "USDCAD": "FX_IDC",
    "USDCHF": "FX_IDC",
    "NZDUSD": "FX_IDC",
    "EURGBP": "FX_IDC",
    "EURJPY": "FX_IDC",
    "GBPJPY": "FX_IDC",
    "USDARS": "FX_IDC",
    "USDTRY": "FX_IDC",
    "USDBRL": "FX_IDC",
    "USDMXN": "FX_IDC",
    "USDINR": "FX_IDC",
    "USDZAR": "FX_IDC",
    "EURTRY": "FX_IDC",
    "USDSGD": "FX_IDC",
    "USDHKD": "FX_IDC",
    "USDCNH": "FX_IDC",
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
    "AMD": "NASDAQ",
    "COIN": "NASDAQ",
    "PLTR": "NASDAQ",
}

_ALL_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "ARS", "TRY", "BRL", "MXN",
    "INR", "ZAR", "SGD", "HKD", "NOK", "SEK", "DKK", "PLN", "CZK", "HUF", "ILS", "THB",
    "IDR", "MYR", "PHP", "KRW", "CNY", "CNH", "RUB", "CLP", "COP", "PEN", "TWD", "AED", "SAR",
}


def resolve_tradingview_target(symbol: str, exchange: str | None = None) -> tuple[str, str]:
    """Resolve symbol and exchange for TradingView queries.

    Parameters
    ----------
    symbol : str
        Input symbol (e.g. 'BTC', 'BTCUSDT', 'USDARS', 'XAUUSD', 'GOLD', 'NVDA').
    exchange : str | None
        Optional explicit exchange override.

    Returns
    -------
    tuple[str, str]
        Clean TradingView symbol and exchange (e.g. ('USDARS', 'FX_IDC')).
    """
    clean_sym = symbol.strip().upper().replace("/", "").replace("-", "").replace(":", "")
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
    elif clean_sym in ("CRUDE", "OIL", "WTI", "USOIL", "WTICOUSD"):
        clean_sym = "USOIL"

    if exchange:
        return clean_sym, exchange.upper()

    if clean_sym in _EXCHANGE_MAP:
        return clean_sym, _EXCHANGE_MAP[clean_sym]

    if clean_sym.endswith("USDT") or clean_sym.endswith("BUSD") or clean_sym.endswith("USDC"):
        return clean_sym, "BINANCE"

    # Any 6-character currency pair containing known global currency codes
    if len(clean_sym) == 6 and (clean_sym[:3] in _ALL_CURRENCIES or clean_sym[3:] in _ALL_CURRENCIES):
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
    200 EMA structure, momentum, and aggregate market sentiment. Automatically
    falls back to live market quote feeds if TradingView screener experiences rate limits.
    """
    sym, venue = resolve_tradingview_target(symbol, exchange)
    tf = normalize_timeframe(timeframe)

    try:
        from tradingview_mcp.core.services.screener_service import analyze_coin

        raw = analyze_coin(sym, venue, tf)
        if raw and isinstance(raw, dict) and not raw.get("error"):
            price_data = raw.get("price_data") or {}
            price = price_data.get("close") or price_data.get("current_price") or price_data.get("open")
            if price is not None:
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
                    "price": price,
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
        log.debug("TradingView screener query encountered transient error", extra={"symbol": sym, "error": str(exc)})

    # Screener unavailable. Fall back to a REAL price from Yahoo -- but never
    # fabricate indicators. A price with no computed RSI/MACD/structure is not a
    # tradeable setup, so we return it as "price_only" with indicators explicitly
    # null. The recommendation layer treats that as INSUFFICIENT_DATA and abstains
    # rather than inventing a direction from a made-up RSI (the old behaviour that
    # drove the accuracy regression).
    try:
        from tradingview_mcp.core.services.yahoo_finance_service import get_price

        # Map symbol to Yahoo format
        if len(sym) == 6 and (sym[:3] in _ALL_CURRENCIES or sym[3:] in _ALL_CURRENCIES):
            yf_sym = f"{sym}=X"
        elif sym in ("XAUUSD", "GOLD"):
            yf_sym = "GC=F"
        elif sym in ("XAGUSD", "SILVER"):
            yf_sym = "SI=F"
        elif sym in ("USOIL", "WTICOUSD", "WTI", "CRUDE"):
            yf_sym = "CL=F"
        elif sym.endswith("USDT"):
            yf_sym = f"{sym[:-4]}-USD"
        else:
            yf_sym = sym

        yf_data = get_price(yf_sym)
        if yf_data and yf_data.get("price"):
            return {
                "status": "price_only",
                "symbol": sym,
                "exchange": venue,
                "timeframe": tf,
                "price": float(yf_data["price"]),
                "change_pct": float(yf_data.get("change_pct") or 0.0),
                "indicators_available": False,
                "note": "Live price only; technical indicators unavailable from screener.",
            }
    except Exception as yf_exc:
        log.warning("Yahoo Finance fallback quote failed", extra={"symbol": sym, "error": str(yf_exc)})

    return {"status": "error", "symbol": sym, "exchange": venue, "error": f"No live market data reachable for {sym}"}


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


_CONFIRM_TF: dict[str, str] = {"1m": "15m", "5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h"}


def _tf_directional_vote(analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    """Net directional vote (-4..+4) from REAL indicators only; None if unavailable.

    Four equally-weighted sub-votes -- market-structure trend, aggregate sentiment,
    RSI (tightened 55/45 bands; mid-range counts as neutral), and MACD histogram
    sign. Nothing is inferred from price alone.
    """
    if not analysis or analysis.get("status") != "ok":
        return None
    struct = analysis.get("market_structure") or {}
    sent = analysis.get("sentiment") or {}
    rsi = analysis.get("rsi") or {}
    macd = analysis.get("macd") or {}
    votes: dict[str, int] = {}

    trend = str(struct.get("trend") or "").lower()
    votes["trend"] = 1 if ("bull" in trend or "up" in trend) else (-1 if ("bear" in trend or "down" in trend) else 0)

    sig = str(sent.get("signal") or "").upper()
    votes["sentiment"] = 1 if sig in ("BUY", "STRONG_BUY") else (-1 if sig in ("SELL", "STRONG_SELL") else 0)

    rv = rsi.get("value")
    votes["rsi"] = (1 if rv >= 55 else (-1 if rv <= 45 else 0)) if isinstance(rv, (int, float)) else 0

    hist = macd.get("histogram")
    votes["macd"] = (1 if hist > 0 else (-1 if hist < 0 else 0)) if isinstance(hist, (int, float)) else 0

    return {"net": sum(votes.values()), "votes": votes}


def _tv_risk_levels(is_up: bool, price: Any, pivots: dict[str, Any]) -> tuple[Any, Any, Any, str] | None:
    """(stop, target, rr, basis) from REAL pivots only. None if no usable structure.

    A long needs a real support below price to stop under and a real resistance
    above to aim at; a short is the mirror. No volatility-percent stop and no
    synthetic multiple target -- absent structure means the caller declines.
    """
    from decimal import Decimal

    def _d(x: Any) -> Any:
        try:
            return Decimal(str(x)) if x is not None else None
        except Exception:
            return None

    if is_up:
        stop = next((v for k in ("s1", "nearest_support", "pivot") if (v := _d(pivots.get(k))) is not None and v < price), None)
        target = next((v for k in ("r1", "nearest_resistance", "r2") if (v := _d(pivots.get(k))) is not None and v > price), None)
        if stop is None or target is None:
            return None
        risk, reward, basis = price - stop, target - price, "long: pivot support stop -> resistance target"
    else:
        stop = next((v for k in ("r1", "nearest_resistance", "pivot") if (v := _d(pivots.get(k))) is not None and v > price), None)
        target = next((v for k in ("s1", "nearest_support", "s2") if (v := _d(pivots.get(k))) is not None and v < price), None)
        if stop is None or target is None:
            return None
        risk, reward, basis = stop - price, price - target, "short: pivot resistance stop -> support target"
    if risk <= 0 or reward <= 0:
        return None
    return stop, target, reward / risk, basis


def _evidence_slice(analysis: dict[str, Any] | None, vote: dict[str, Any] | None) -> dict[str, Any]:
    """Compact, real-only indicator snapshot handed to the AI brain."""
    if not analysis:
        return {"available": False}
    rsi = analysis.get("rsi") or {}
    macd = analysis.get("macd") or {}
    struct = analysis.get("market_structure") or {}
    sent = analysis.get("sentiment") or {}
    bb = analysis.get("bollinger_bands") or {}
    return {
        "available": True,
        "trend": struct.get("trend"),
        "trend_strength": struct.get("trend_strength"),
        "rsi": rsi.get("value"),
        "macd_histogram": macd.get("histogram"),
        "sentiment_signal": sent.get("signal"),
        "bollinger_squeeze": bb.get("squeeze"),
        "net_vote": (vote or {}).get("net"),
        "votes": (vote or {}).get("votes"),
    }


def _tradingview_decision_core(symbol: str, minutes: int) -> dict[str, Any]:
    """Decide a trade from LIVE TradingView evidence across two timeframes.

    Shared decision core for both the recommendation and the signal-decision entry
    points. The AI brain (GLM via Seek AI) is the decision authority when reachable
    within budget; a deterministic multi-timeframe alignment gate is the pre-filter
    and the fallback. Every input is real (never fabricated) and NO_TRADE is
    first-class -- abstention is raised as ``NoTradeReason``.

    Returns a dict of the fully resolved decision (direction, real pivot-based risk
    levels, honest agreement-based confidence, brain attribution, rationale) for a
    caller to render as a ``TradeRecommendation`` or an ``AIDecision``.
    """
    import uuid
    from datetime import timedelta
    from decimal import Decimal
    from quantedge.contracts import SignalDirection, SignalStatus, utc_now
    from quantedge.services.horizons import horizon_for_minutes
    from quantedge.services.risk import MIN_ACCEPTABLE_RR
    from quantedge.services.signal import NoTradeReason

    sym, venue = resolve_tradingview_target(symbol)
    exec_tf = "1m" if minutes <= 1 else ("5m" if minutes <= 5 else ("15m" if minutes <= 15 else ("30m" if minutes <= 30 else "1h")))
    confirm_tf = _CONFIRM_TF.get(exec_tf, "1h")

    exec_a = get_tradingview_analysis(sym, timeframe=exec_tf)
    if exec_a.get("status") != "ok":
        detail = exec_a.get("note") or exec_a.get("error") or "screener returned no indicators"
        raise NoTradeReason(SignalStatus.INSUFFICIENT_DATA, f"No verified indicators for {sym} on {exec_tf}: {detail}")
    price_val = exec_a.get("price")
    if not price_val:
        raise NoTradeReason(SignalStatus.INSUFFICIENT_DATA, f"No live price returned for {sym}")

    confirm_a = get_tradingview_analysis(sym, timeframe=confirm_tf)
    exec_vote = _tf_directional_vote(exec_a)
    confirm_vote = _tf_directional_vote(confirm_a)
    if exec_vote is None:
        raise NoTradeReason(SignalStatus.INSUFFICIENT_DATA, f"Execution indicators unavailable for {sym}")
    if confirm_vote is None:
        raise NoTradeReason(SignalStatus.INSUFFICIENT_DATA, f"No higher-timeframe ({confirm_tf}) confirmation for {sym}; standing aside")

    e_net = exec_vote["net"]
    c_net = confirm_vote["net"]
    e_sign = 1 if e_net > 0 else (-1 if e_net < 0 else 0)
    c_sign = 1 if c_net > 0 else (-1 if c_net < 0 else 0)
    # Multi-timeframe alignment gate: both timeframes lean the same way and the
    # execution timeframe shows >= 2/4 real indicators agreeing. Otherwise skip.
    if e_sign == 0 or c_sign == 0 or e_sign != c_sign:
        raise NoTradeReason(
            SignalStatus.NO_TRADE,
            f"{sym}: {exec_tf}/{confirm_tf} not aligned (no directional edge)",
            detail=f"exec net {e_net}, confirm net {c_net} from real indicators",
        )
    if abs(e_net) < 2:
        raise NoTradeReason(
            SignalStatus.NO_TRADE,
            f"{sym}: execution indicators do not agree strongly enough (net {e_net}/4)",
        )
    det_direction = SignalDirection.UP if e_sign > 0 else SignalDirection.DOWN

    price = Decimal(str(price_val))
    pivots = exec_a.get("pivots") or {}
    bb = exec_a.get("bollinger_bands") or {}

    # Hand the REAL evidence to the AI brain to DECIDE. GLM leads when reachable;
    # the deterministic gate above is the guardrail and the fallback.
    evidence = {
        "symbol": sym,
        "venue": venue,
        "hold_minutes": minutes,
        "execution_timeframe": exec_tf,
        "confirmation_timeframe": confirm_tf,
        "price": price_val,
        "execution": _evidence_slice(exec_a, exec_vote),
        "confirmation": _evidence_slice(confirm_a, confirm_vote),
        "deterministic_candidate": det_direction.value,
        "deterministic_agreement": f"exec {abs(e_net)}/4, confirm {abs(c_net)}/4",
    }

    brain, brain_model = "deterministic", "tv_gate_v2"
    conviction: float | None = None
    glm_reason = glm_invalidation = ""
    direction = det_direction

    try:
        from quantedge.providers.llm import default_llm_provider
        provider = default_llm_provider()
    except Exception:
        provider = None
    decide = getattr(provider, "decide_trade", None) if provider is not None else None
    if callable(decide):
        try:
            verdict = decide(evidence)
            gd = verdict.get("decision")
            if gd == "NO_TRADE":
                raise NoTradeReason(
                    SignalStatus.NO_TRADE,
                    f"{sym}: AI brain ({verdict.get('brain', 'glm')}) declined despite alignment",
                    detail=verdict.get("reason") or "",
                )
            gdir = SignalDirection.UP if gd == "UP" else SignalDirection.DOWN
            if gdir != det_direction:
                raise NoTradeReason(
                    SignalStatus.NO_TRADE,
                    f"{sym}: AI brain disagrees with aligned multi-timeframe evidence; standing aside",
                    detail=verdict.get("reason") or "",
                )
            direction = gdir
            brain, brain_model = "seekai", str(verdict.get("brain") or "glm-5.3-flash")
            conviction = verdict.get("conviction")
            glm_reason = verdict.get("reason") or ""
            glm_invalidation = verdict.get("invalidation") or ""
        except NoTradeReason:
            raise
        except Exception as brain_exc:
            log.info("AI brain decide_trade unavailable for %s (%s); deterministic decision stands", sym, type(brain_exc).__name__)

    now = utc_now()
    exp = now + timedelta(minutes=minutes)

    # Stop/target from REAL pivots only -- no volatility-percent stop, no synthetic
    # 2R target. If the structure needed to place a stop and a reachable target
    # isn't there, the setup is declined rather than invented.
    levels = _tv_risk_levels(direction == SignalDirection.UP, price, pivots)
    if levels is None:
        raise NoTradeReason(
            SignalStatus.NO_TRADE,
            f"{sym}: no real pivot structure to place a stop and a reachable target",
        )
    stop, target, rr, basis = levels
    if rr < MIN_ACCEPTABLE_RR:
        raise NoTradeReason(
            SignalStatus.NO_TRADE,
            f"{sym}: reward:risk {rr:.2f} is below the {MIN_ACCEPTABLE_RR} minimum",
            detail=f"stop {stop} / target {target} from real pivots ({basis})",
        )

    # 3. Asset class
    sym_upper = sym.upper()
    if any(k in sym_upper for k in ("JPY", "EUR", "GBP", "CHF", "CAD", "AUD", "NZD")):
        ast = "forex"
    elif any(k in sym_upper for k in ("XAU", "XAG", "GOLD", "SILVER", "WTI", "BRENT", "OIL")):
        ast = "commodity"
    elif any(k in sym_upper for k in ("SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL")):
        ast = "stock"
    else:
        ast = "crypto"

    # Honest confidence: multi-timeframe agreement (0..1) mapped to 50..90, blended
    # with the brain's conviction when the brain decided. An agreement measure, NOT
    # a calibrated win probability, and never presented as one.
    agreement = (abs(e_net) + abs(c_net)) / 8.0
    base_conf = 50 + round(agreement * 40)
    confidence = int(round((base_conf + conviction * 100) / 2)) if conviction is not None else int(base_conf)
    confidence = max(50, min(90, confidence))

    trend = (exec_a.get("market_structure") or {}).get("trend") or direction.value
    squeeze_note = " Bollinger squeeze active (breakout pending)." if bb.get("squeeze") else ""
    if brain == "seekai":
        rationale = (
            f"AI brain ({brain_model}) decided {direction.value}, confirmed by {exec_tf}/{confirm_tf} "
            f"alignment ({abs(e_net)}/4 and {abs(c_net)}/4 real indicators agree). {glm_reason}{squeeze_note}"
        )
    else:
        rationale = (
            f"Multi-timeframe gate: {exec_tf} and {confirm_tf} both {direction.value} "
            f"({abs(e_net)}/4 and {abs(c_net)}/4 real indicators agree); levels from real pivots ({basis})."
            f"{squeeze_note} AI brain unavailable within budget."
        )

    return {
        "decision_id": f"rec-tv-{uuid.uuid4().hex[:10]}",
        "symbol": sym,
        "venue": venue,
        "asset_class": ast,
        "horizon": horizon_for_minutes(minutes),
        "direction": direction,
        "now": now,
        "expiry": exp,
        "reference_price": price,
        "stop": stop.quantize(price),
        "target": target.quantize(price),
        "risk_reward_ratio": round(rr, 2),
        "agreement": agreement,
        "confidence": confidence,
        "regime": trend,
        "rationale": rationale,
        "brain": brain,
        "brain_model": brain_model,
        "glm_invalidation": glm_invalidation,
    }


def _tv_build_decision(d: dict[str, Any]) -> Any:
    """Render a decision-core dict as a SIGNAL AIDecision (no persistence)."""
    from quantedge.contracts import AIDecision, SignalStatus

    return AIDecision(
        decision_id=d["decision_id"],
        symbol=d["symbol"],
        horizon=d["horizon"],
        status=SignalStatus.SIGNAL,
        direction=d["direction"],
        reference_price=d["reference_price"],
        expiry_utc=d["expiry"],
        regime=d["regime"],
        heuristic_score=d["confidence"] / 100.0,
        calibrated_probability=None,
        supporting_evidence=[d["rationale"]],
        contradictory_evidence=[],
        invalidation_conditions=[d["glm_invalidation"]] if d["glm_invalidation"] else [],
        missing_information=[],
        llm_provider=d["brain"],
        llm_model=d["brain_model"],
        scanner_version="tv_gate_v2",
        data_quality_status=None,
        created_at_utc=d["now"],
    )
def generate_tradingview_recommendation(symbol: str, minutes: int = 15) -> Any:
    """Decide a trade from LIVE TradingView evidence; return a TradeRecommendation.

    Thin renderer over :func:`_tradingview_decision_core`. Raises ``NoTradeReason``
    when the core abstains (first-class NO_TRADE / INSUFFICIENT_DATA), and records
    the real brain + model that decided into the lifecycle store.
    """
    from quantedge.contracts import TradeRecommendation

    d = _tradingview_decision_core(symbol, minutes)
    rec = TradeRecommendation(
        recommendation_id=d["decision_id"],
        symbol=d["symbol"],
        asset_class=d["asset_class"],
        horizon=d["horizon"],
        direction=d["direction"],
        valid_from_utc=d["now"],
        valid_until_utc=d["expiry"],
        reference_price=d["reference_price"],
        stop_loss=d["stop"],
        take_profit=d["target"],
        risk_reward_ratio=d["risk_reward_ratio"],
        risk_level="HIGH_CONVICTION" if d["agreement"] >= 0.75 else "MODERATE_CONVICTION",
        recommended_venue=f"TradingView ({d['venue']})",
        regime=d["regime"],
        memory_consulted_count=0,
        key_lessons_applied=[],
        memory_rules_applied=[],
        heuristic_score=d["confidence"] / 100.0,
        confidence_pct=d["confidence"],
        rationale=d["rationale"],
        warnings=[],
        generated_at_utc=d["now"],
    )
    try:
        from quantedge.repositories import get_repository
        from quantedge.services.signal import _persist

        _persist(get_repository(), _tv_build_decision(d))
    except Exception as exc:
        log.debug("could not persist tradingview decision into lifecycle", extra={"error": str(exc)})
    return rec
def generate_tradingview_decision(symbol: str, *, minutes: int = 15) -> Any:
    """Decide a trade from LIVE TradingView evidence; return an AIDecision.

    The signal-decision entry point for non-crypto symbols (forex, metals, indices,
    equities), which have no Binance candle feed and cannot be served by the
    deterministic candle scan. Same real-evidence core and GLM brain as the
    recommendation path, rendered as an ``AIDecision`` and persisted. Raises
    ``NoTradeReason`` when the core abstains, so callers surface an honest
    NO_TRADE / INSUFFICIENT_DATA rather than a manufactured setup.
    """
    d = _tradingview_decision_core(symbol, minutes)
    decision = _tv_build_decision(d)
    try:
        from quantedge.repositories import get_repository
        from quantedge.services.signal import _persist

        _persist(get_repository(), decision)
    except Exception as exc:
        log.debug("could not persist tradingview decision", extra={"error": str(exc)})
    return decision

