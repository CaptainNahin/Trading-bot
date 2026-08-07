"""Live smoke test for the Twelve Data adapter.

Spends real quota against the free plan (8 req/min, 800/day), so the calls are
deliberately few and ordered cheapest-first. Nothing here is mocked: if the key
is missing or rejected, the script says so rather than pretending to pass.

Run:  ./.venv/Scripts/python.exe -u scripts/smoke_test_twelvedata.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge.contracts import Timeframe, utc_now
from quantedge.logging import configure_logging
from quantedge.providers.twelvedata.rest import TwelveDataProvider

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    marker = "[PASS]" if condition else "[FAIL]"
    print(f"  {marker} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


async def main() -> int:
    configure_logging()
    provider = TwelveDataProvider()

    print("=" * 70)
    print("TWELVE DATA ADAPTER -- LIVE SMOKE TEST")
    print("=" * 70)

    print("\n[1] health_check()")
    health = await provider.health_check()
    print(f"  status={health.status.value} latency={health.latency_ms}ms")
    print(f"  message={health.message}")
    for limitation in health.limitations:
        print(f"  limitation: {limitation}")
    check("credentials present", health.credentials_present)
    check("provider reachable", health.status.value == "ok", health.message or "")
    if health.status.value != "ok":
        print("\nAborting: without a working key the rest would prove nothing.")
        await provider.aclose()
        return 1

    print("\n[2] get_quote('EURUSD')")
    quote = await provider.get_quote("EURUSD")
    print(f"  last={quote.last} provider_time={quote.provider_time_utc.isoformat()}")
    print(f"  bid={quote.bid} ask={quote.ask} spread={quote.spread}")
    check("last price present", quote.last is not None)
    check("price is plausible for EURUSD", quote.last is not None and 0.5 < float(quote.last) < 2.0)
    check("symbol normalized to EURUSD", quote.symbol == "EURUSD", quote.symbol)
    check(
        "spread NOT fabricated (endpoint has no bid/ask)",
        quote.spread is None and quote.bid is None and quote.ask is None,
        f"spread={quote.spread}",
    )
    check("provider_time is tz-aware UTC", quote.provider_time_utc.tzinfo is not None)

    print("\n[3] get_candles('EURUSD', M5, count=10)  -- closed bars only")
    series = await provider.get_candles("EURUSD", Timeframe.M5, count=10)
    print(f"  bars={len(series)} source={series.source} forming={series.includes_forming_candle}")
    for candle in series.candles[-3:]:
        print(
            f"    {candle.open_time_utc.isoformat()} -> {candle.close_time_utc.isoformat()} "
            f"O={candle.open} H={candle.high} L={candle.low} C={candle.close} "
            f"closed={candle.is_closed} vol={candle.volume}"
        )
    check("bars returned", len(series) > 0, f"{len(series)} bars")
    check("respects count limit", len(series) <= 10, f"{len(series)} bars")
    check(
        "RULE 9: no forming candle leaked into history",
        all(c.is_closed for c in series.candles),
        f"{sum(1 for c in series.candles if not c.is_closed)} forming bars found",
    )
    check("includes_forming_candle flag is accurate", series.includes_forming_candle is False)
    times = [c.open_time_utc for c in series.candles]
    check("oldest-first ordering", times == sorted(times))
    check(
        "all timestamps tz-aware", all(c.open_time_utc.tzinfo is not None for c in series.candles)
    )

    if series.candles:
        newest = series.candles[-1]
        age_minutes = newest.age_seconds(now=utc_now()) / 60.0
        print(f"  newest closed bar age: {age_minutes:.1f} min")
        # A 5m bar that already closed is at most ~5 min old on a live market;
        # a large age means either a closed market (weekend) or a UTC mismatch.
        check(
            "newest closed bar is in the past (UTC alignment sane)",
            newest.close_time_utc <= utc_now(),
            newest.close_time_utc.isoformat(),
        )
        spacing = {
            int((b.open_time_utc - a.open_time_utc).total_seconds())
            for a, b in zip(series.candles, series.candles[1:], strict=False)
        }
        check(
            "bar spacing matches the 5m interval",
            spacing <= {300},
            f"spacing seconds seen: {sorted(spacing)}",
        )

    print("\n[4] get_candles('EURUSD', M5, include_forming=True)")
    live = await provider.get_candles("EURUSD", Timeframe.M5, count=5, include_forming=True)
    forming = live.forming
    print(f"  bars={len(live)} forming_present={forming is not None}")
    if forming is not None:
        print(f"    forming: {forming.open_time_utc.isoformat()} C={forming.close}")
        check("forming bar is last element", live.candles[-1] is forming)
        check("forming bar close_time is in the future", forming.close_time_utc > utc_now())
    else:
        print("    (no forming bar -- expected when the FX market is closed)")
    check(
        "includes_forming_candle flag matches reality",
        live.includes_forming_candle == (forming is not None),
    )

    print("\n[5] unsupported interval is refused, not silently substituted")
    try:
        await provider.get_candles("EURUSD", Timeframe.M3, count=5)
    except Exception as exc:
        check(
            "M3 raises UnsupportedTimeframeError",
            type(exc).__name__ == "UnsupportedTimeframeError",
            type(exc).__name__,
        )
    else:
        check("M3 raises UnsupportedTimeframeError", False, "no exception raised")

    print("\n[6] crypto is routed away from this provider")
    try:
        await provider.get_quote("BTCUSDT")
    except Exception as exc:
        check(
            "BTCUSDT raises UnsupportedSymbolError",
            type(exc).__name__ == "UnsupportedSymbolError",
            type(exc).__name__,
        )
    else:
        check("BTCUSDT raises UnsupportedSymbolError", False, "no exception raised")

    print("\n[7] rate limiter state")
    print(f"  {provider.rate_limit_snapshot()}")

    await provider.aclose()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: FAILED -- {len(FAILURES)} check(s) failed")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
