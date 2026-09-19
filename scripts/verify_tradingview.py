"""Comprehensive verification suite for TradingView MCP Connector & Chat Intelligence.

Tests:
1. Target and venue resolution across Crypto, Commodities, Forex, and Equities.
2. Timeframe normalization.
3. Chat intent parsing for TradingView commands.
4. Live TradingView institutional technical analysis and pivot level extraction.
5. Multi-timeframe consensus.
6. Breakout and volume screener.
7. Chat layer dispatch and status reporting.
"""

from __future__ import annotations

import sys
from typing import Any

from quantedge.services import chat
from quantedge.services.tradingview import (
    format_tradingview_summary,
    get_tradingview_analysis,
    get_tradingview_multi_timeframe,
    normalize_timeframe,
    resolve_tradingview_target,
    scan_tradingview_gainers,
)


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{(': ' + detail) if detail else ''}")
    return ok


def main() -> int:
    results: list[bool] = []
    print("=== QUANTEDGE TRADINGVIEW MCP CONNECTOR VERIFICATION ===\n")

    # 1. Target and Venue Resolution
    print("--- 1. Target & Venue Resolution ---")
    res_btc = resolve_tradingview_target("BTC")
    results.append(check("resolve 'BTC'", res_btc == ("BTCUSDT", "BINANCE"), str(res_btc)))

    res_gold = resolve_tradingview_target("GOLD")
    results.append(check("resolve 'GOLD'", res_gold == ("XAUUSD", "OANDA"), str(res_gold)))

    res_eurusd = resolve_tradingview_target("EURUSD")
    results.append(check("resolve 'EURUSD'", res_eurusd == ("EURUSD", "FX_IDC"), str(res_eurusd)))

    res_nvda = resolve_tradingview_target("NVDA")
    results.append(check("resolve 'NVDA'", res_nvda == ("NVDA", "NASDAQ"), str(res_nvda)))

    # 2. Timeframe Normalization
    print("\n--- 2. Timeframe Normalization ---")
    results.append(check("normalize '15m'", normalize_timeframe("15m") == "15m"))
    results.append(check("normalize '1h'", normalize_timeframe("1h") == "1h"))
    results.append(check("normalize '4 hours'", normalize_timeframe("4 hours") == "4h"))
    results.append(check("normalize 'daily'", normalize_timeframe("daily") == "1D"))

    # 3. Chat Intent Parsing
    print("\n--- 3. Chat Intent Parsing ---")
    cases = [
        ("tv btc", "TRADINGVIEW", "BTCUSDT", None),
        ("tradingview gold 1h", "TRADINGVIEW", "XAUUSD", 60),
        ("tv breakouts", "TRADINGVIEW", None, None),
        ("tv mtf sol", "TRADINGVIEW", "SOLUSDT", None),
    ]
    for msg, expected_intent, expected_sym, expected_min in cases:
        parsed = chat.parse_intent(msg)
        ok = (
            parsed.intent.value == expected_intent
            and parsed.symbol == expected_sym
            and parsed.minutes == expected_min
        )
        results.append(
            check(f"parse {msg!r}", ok, f"intent={parsed.intent.value} sym={parsed.symbol} min={parsed.minutes}")
        )

    # 4. Live Technical Analysis: BTCUSDT (Crypto)
    print("\n--- 4. Live Technical Analysis (Crypto: BTCUSDT) ---")
    btc_ta = get_tradingview_analysis("BTCUSDT", "15m")
    btc_ok = btc_ta.get("status") == "ok" and btc_ta.get("price") is not None
    btc_trend = btc_ta.get("market_structure", {}).get("trend")
    btc_rsi = btc_ta.get("rsi", {}).get("value")
    results.append(
        check(
            "BTCUSDT 15m Analysis",
            btc_ok,
            f"Price={btc_ta.get('price')} Trend={btc_trend} RSI={btc_rsi}",
        )
    )
    if btc_ok:
        pivots = btc_ta.get("pivots", {})
        results.append(
            check(
                "BTCUSDT Pivots",
                pivots.get("pivot") is not None,
                f"Pivot={pivots.get('pivot')} S1={pivots.get('s1')} R1={pivots.get('r1')}",
            )
        )

    # 5. Live Technical Analysis: XAUUSD (Commodities / Gold)
    print("\n--- 5. Live Technical Analysis (Commodities: Gold / XAUUSD) ---")
    gold_ta = get_tradingview_analysis("XAUUSD", "1h")
    gold_ok = gold_ta.get("status") == "ok" and gold_ta.get("price") is not None
    results.append(
        check(
            "Gold 1h Analysis",
            gold_ok,
            f"Price={gold_ta.get('price')} S1={gold_ta.get('pivots', {}).get('s1')}",
        )
    )

    # 6. Multi-Timeframe Consensus
    print("\n--- 6. Multi-Timeframe Consensus ---")
    mtf = get_tradingview_multi_timeframe("BTCUSDT")
    mtf_ok = mtf.get("status") == "ok" and "alignment" in mtf
    results.append(
        check(
            "BTCUSDT MTF Consensus",
            mtf_ok,
            f"Alignment={mtf.get('alignment')} Rec={mtf.get('recommendation')}",
        )
    )

    # 7. Volume Breakout Scanner
    print("\n--- 7. Volume Breakout Scanner ---")
    gainers = scan_tradingview_gainers("BINANCE")
    results.append(
        check("Binance Breakout Scanner", isinstance(gainers, list), f"Found {len(gainers)} candidates")
    )

    # 8. End-to-End Chat Responses
    print("\n--- 8. End-to-End Chat Queries ---")
    chat_tv_btc = chat.handle_message("tv btc")
    tv_in_text = "TradingView Intelligence" in chat_tv_btc.text
    results.append(
        check(
            "chat 'tv btc'",
            chat_tv_btc.intent == chat.Intent.TRADINGVIEW and tv_in_text,
            chat_tv_btc.text.splitlines()[0] if chat_tv_btc.text else "empty",
        )
    )

    chat_status = chat.handle_message("status")
    results.append(
        check(
            "chat 'status' lists TradingView",
            "tradingview" in chat_status.text.lower(),
            "TradingView provider listed",
        )
    )

    print("\n--- Sample Output: chat.handle_message('tv btc') ---")
    print(chat_tv_btc.text)

    passed = sum(results)
    total = len(results)
    print("\n======================================")
    print(f"VERIFICATION SUMMARY: {passed}/{total} CHECKS PASSED")
    print("======================================")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
