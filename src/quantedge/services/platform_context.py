"""Platform Knowledge and Live State Context Provider for the AI Brain.

Provides comprehensive platform metadata, capabilities, supported markets,
quantitative rules, TradingView tools, and live operational status (active trades,
provider health, memory bank rules) to ground conversational AI responses.
"""

from __future__ import annotations

from typing import Any

from quantedge.logging import get_logger

log = get_logger(__name__)


def get_platform_knowledge() -> dict[str, Any]:
    """Static architectural specifications and capabilities of QuantEdge AI."""
    return {
        "platform_name": "QuantEdge AI",
        "tagline": "Dual-Brain Institutional Quantitative Trading Intelligence Gateway",
        "architecture": {
            "mathematical_quant_engine": (
                "Deterministic pipeline analyzing 200 EMA trend alignment, ATR volatility-based "
                "dynamic stop-loss and take-profit levels, Multi-Timeframe Consensus (15m, 1H, 4H, 1D), "
                "Market Structure (CHoCH, BOS, Swing Highs/Lows), and Order Book volume delta."
            ),
            "zxl_ai_brain": (
                "ZXL AI Brain (ZLM 5.3 / DeepSeek reasoning engine via Seek AI): Evaluates candidate setups, "
                "filters out weak or trap signals, provides macroeconomic context, and performs deep "
                "post-mortem root-cause diagnostics on closed trades."
            ),
            "tradingview_mcp": (
                "FastMCP server with 24 registered institutional tools: Real-time TradingView technical "
                "analysis (RSI, MACD, Stochastics, ADX, Hull MA), institutional Floor Pivots (Pivot, S1-S3, R1-R3), "
                "Bollinger squeeze breakout screening across Binance, OANDA, and NASDAQ."
            ),
            "autonomous_lifecycle_memory": (
                "Continuously monitors open positions across closed candles until Take Profit (WIN), "
                "Stop Loss (LOSS), or Expiry. Resolved losses automatically trigger AI root-cause analysis "
                "to derive DO and DON'T rules stored persistently in SQLite to prevent repeating mistakes."
            ),
        },
        "supported_markets": {
            "crypto": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
            "forex": ["EURUSD", "GBPUSD", "USDJPY"],
            "commodities": ["XAUUSD (Gold)", "XAGUSD (Silver)", "WTICOUSD (Crude Oil)"],
        },
        "supported_time_limits": [
            "1 min (scalp)",
            "3 min",
            "5 min",
            "10 min",
            "15 min (standard default)",
            "20 min",
            "30 min",
            "1 hour (swing)",
        ],
        "primary_commands": [
            "`BTC 15m` or `signal ETH 1h` -- High-conviction signal with exact entry, stop loss, and take profit",
            "`tv btc` or `tradingview gold` -- Institutional TradingView technical indicators & pivot levels",
            "`tv breakouts` -- Screener for top volume breakout candidates on Binance",
            "`active trades` or `lifecycle` -- Real-time in-flight position tracker and automated settlements",
            "`what have you learned` -- Review lessons and DO/DON'T rules from the autonomous memory bank",
            "`status` -- Check connected market sources and AI brain health",
            "`time limits` -- Display supported holding periods and horizons",
        ],
    }


def get_live_platform_state() -> dict[str, Any]:
    """Inspect the live system state (active trades, memory bank, provider health)."""
    state: dict[str, Any] = {
        "active_trades_count": 0,
        "active_trades": [],
        "memory_total_trades": 0,
        "memory_win_rate": "N/A",
        "memory_top_rules": [],
        "provider_status": {},
    }

    # 1. Active signals from lifecycle
    try:
        from quantedge.services.lifecycle import get_active_signals_summary

        summary = get_active_signals_summary()
        state["active_trades_count"] = summary.get("total_active", 0)
        state["active_trades"] = summary.get("signals", [])[:5]
    except Exception as exc:
        log.debug("failed to fetch active signals for context", extra={"error": str(exc)})

    # 2. Memory bank rules
    try:
        from quantedge.services.memory import get_memory_bank_summary

        mem = get_memory_bank_summary()
        state["memory_total_trades"] = mem.get("total_memories", 0)
        state["memory_wins"] = mem.get("wins", 0)
        state["memory_losses"] = mem.get("losses", 0)
        if mem.get("total_memories", 0) > 0:
            rate = (mem.get("wins", 0) / mem.get("total_memories", 1)) * 100
            state["memory_win_rate"] = f"{rate:.1f}%"
        state["memory_top_rules"] = mem.get("recent_rules", [])[:4]
    except Exception as exc:
        log.debug("failed to fetch memory summary for context", extra={"error": str(exc)})

    # 3. Provider health
    try:
        from quantedge.providers.registry import get_registry

        registry = get_registry()
        health_list = registry.health_check_all()
        for h in health_list:
            state["provider_status"][h.provider] = h.status.value
    except Exception as exc:
        log.debug("failed to fetch provider health for context", extra={"error": str(exc)})

    return state


def build_platform_context() -> str:
    """Format full platform specifications and live data into a system prompt section."""
    know = get_platform_knowledge()
    live = get_live_platform_state()

    crypto_str = ", ".join(know["supported_markets"]["crypto"])
    forex_str = ", ".join(know["supported_markets"]["forex"])
    comm_str = ", ".join(know["supported_markets"]["commodities"])
    limits_str = ", ".join(know["supported_time_limits"])

    active_info = f"{live['active_trades_count']} trades currently in-flight."
    if live["active_trades"]:
        items = [
            f"{t.get('symbol')} ({t.get('direction')}) @ {t.get('reference_price')} (exp in {t.get('remaining_minutes')}m)"
            for t in live["active_trades"]
        ]
        active_info += f" Open setups: {', '.join(items)}."

    mem_info = f"Recorded trades in memory: {live['memory_total_trades']}."
    if live.get("memory_win_rate") != "N/A":
        mem_info += f" Win rate: {live['memory_win_rate']} ({live.get('memory_wins', 0)}W / {live.get('memory_losses', 0)}L)."
    if live["memory_top_rules"]:
        mem_info += f" Active DO/DON'T rules: {'; '.join(live['memory_top_rules'])}."

    providers_str = (
        ", ".join(f"{k.upper()}: {v}" for k, v in live["provider_status"].items())
        if live["provider_status"]
        else "Binance (Active), Twelve Data (Active), Alpha Vantage (Active)"
    )

    return (
        "### LIVE PLATFORM CONTEXT & GROUND TRUTH DATA\n"
        f"- Platform: {know['platform_name']} -- {know['tagline']}\n"
        f"- Mathematical Quant Engine: {know['architecture']['mathematical_quant_engine']}\n"
        f"- ZXL AI Brain: {know['architecture']['zxl_ai_brain']}\n"
        f"- TradingView Institutional Tools: {know['architecture']['tradingview_mcp']}\n"
        f"- Autonomous Lifecycle Engine: {know['architecture']['autonomous_lifecycle_memory']}\n"
        f"- Supported Cryptos: {crypto_str}\n"
        f"- Supported Forex: {forex_str}\n"
        f"- Supported Commodities: {comm_str}\n"
        f"- Supported Time Limits: {limits_str}\n"
        f"- Live Market Providers: {providers_str}\n"
        f"- Active In-Flight Positions: {active_info}\n"
        f"- Autonomous Memory Bank: {mem_info}\n"
        "- Available Quick Commands: " + " | ".join(know["primary_commands"]) + "\n"
    )
