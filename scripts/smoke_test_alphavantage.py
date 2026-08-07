"""Live smoke test for the Alpha Vantage news adapter.

Costs real quota (free plan: 5/min, 25/day), so this makes at most three calls.

Run:  ./.venv/Scripts/python.exe -u scripts/smoke_test_alphavantage.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge.contracts import utc_now
from quantedge.logging import configure_logging
from quantedge.providers.alphavantage.news import AlphaVantageNewsProvider

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


async def main() -> int:
    configure_logging()
    provider = AlphaVantageNewsProvider()

    print("=" * 70)
    print("ALPHA VANTAGE NEWS ADAPTER -- LIVE SMOKE TEST")
    print("=" * 70)

    print("\n[1] health_check()")
    health = await provider.health_check()
    print(f"  status={health.status.value} latency={health.latency_ms}ms")
    print(f"  message={health.message}")
    for limitation in health.limitations:
        print(f"  limitation: {limitation}")
    check("credentials present", health.credentials_present)
    check("reachable", health.status.value == "ok", health.message or "")
    check(
        "declares NO economic calendar capability",
        provider.capabilities().get("economic_calendar") is False,
        str(provider.capabilities()),
    )
    if health.status.value != "ok":
        await provider.aclose()
        print("\nAborting: the feed is not reachable, so nothing below would prove anything.")
        return 1

    print("\n[2] get_market_news(limit=5)")
    items = await provider.get_market_news(limit=5)
    print(f"  items={len(items)}")
    for item in items[:3]:
        print(f"    {item.published_at_utc.isoformat()} [{item.source}] {item.headline[:70]}")
        print(f"      symbols={item.symbols[:6]}")
    check("news returned", len(items) > 0, f"{len(items)} items")
    check("respects limit", len(items) <= 5, f"{len(items)} items")
    check("all have headlines", all(i.headline.strip() for i in items))
    check("all timestamps tz-aware UTC", all(i.published_at_utc.tzinfo is not None for i in items))
    check(
        "newest first",
        [i.published_at_utc for i in items]
        == sorted((i.published_at_utc for i in items), reverse=True),
    )
    check(
        "no article dated in the future",
        all(i.published_at_utc <= utc_now() for i in items),
    )
    check("provider attributed", all(i.provider == "alphavantage" for i in items))

    print("\n[3] get_symbol_news('EURUSD', limit=3)  -- forex has no ticker filter")
    fx = await provider.get_symbol_news("EURUSD", limit=3)
    print(f"  items={len(fx)}")
    for item in fx[:2]:
        print(f"    {item.published_at_utc.isoformat()} {item.headline[:70]}")
    check("macro fallback returned news", len(fx) > 0, f"{len(fx)} items")
    check(
        "does NOT claim EURUSD-specific matching (symbols cleared)",
        all(item.symbols == [] for item in fx),
        f"symbols seen: {[i.symbols for i in fx if i.symbols]}",
    )

    print("\n[4] ticker mapping (no network)")
    check(
        "stock -> bare ticker",
        provider._to_ticker("AAPL") == "AAPL",
        str(provider._to_ticker("AAPL")),
    )
    check(
        "crypto -> CRYPTO:BTC",
        provider._to_ticker("BTCUSDT") == "CRYPTO:BTC",
        str(provider._to_ticker("BTCUSDT")),
    )
    check("forex -> None (unsupported)", provider._to_ticker("EURUSD") is None)

    print("\n[5] timestamp parser rejects the unparseable instead of stamping 'now'")
    check("garbage -> None", provider._parse_time("not-a-date") is None)
    check("empty -> None", provider._parse_time("") is None)
    check(
        "valid -> parsed UTC",
        (parsed := provider._parse_time("20260806T143000")) is not None
        and parsed.tzinfo is not None
        and parsed.hour == 14,
        str(provider._parse_time("20260806T143000")),
    )

    print("\n[6] rate limiter state")
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
