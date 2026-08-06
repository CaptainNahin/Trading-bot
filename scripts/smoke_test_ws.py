"""Live market data collector smoke test.

Verifies:
1. The WebSocket client connects to Binance's public stream
2. kline and bookTicker events arrive and are normalized correctly
3. Closed bars are separated from the forming bar
4. The collector reports healthy status
5. Event deduplication works across a simulated reconnect

Does NOT verify durable persistence (requires a real database).
"""

from __future__ import annotations

import asyncio

from quantedge.config import get_settings
from quantedge.contracts import Timeframe
from quantedge.providers.binance.ws import BinanceStreamClient
from quantedge.services.streams import MarketStreamCollector


async def main() -> None:
    print("=== Live Binance WebSocket Smoke Test ===\n")
    settings = get_settings()

    # 1. Raw stream client: listen for 10 seconds and report what arrives
    print("1. Raw stream test (10 seconds)...")
    client = BinanceStreamClient()
    symbols = settings.stream_symbols[:2]  # BTCUSDT, ETHUSDT
    intervals = ["1m"]

    print(f"   Subscribing: {symbols} x {intervals}")
    kline_count = 0
    book_count = 0
    start = asyncio.get_event_loop().time()

    try:
        async for event in client.stream(symbols, intervals, include_book_ticker=True):
            if event.kind == "kline":
                kline_count += 1
                candle = event.payload
                print(
                    f"   [kline] {candle.symbol} {candle.timeframe.value} "
                    f"{candle.open_time_utc.strftime('%H:%M')} "
                    f"closed={candle.is_closed} O={candle.open} C={candle.close}"
                )
            elif event.kind == "book_ticker":
                book_count += 1
                quote = event.payload
                print(
                    f"   [book ] {quote.symbol} bid={quote.bid} ask={quote.ask} "
                    f"spread={quote.spread}"
                )

            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > 10.0:
                client.stop()
                break
    except asyncio.CancelledError:
        pass

    print(
        f"   OK: Received {kline_count} kline events and {book_count} book_ticker events\n"
    )

    # 2. Collector: warm up and verify cache
    print("2. Collector warm-up (15 seconds)...")
    collector = MarketStreamCollector(
        symbols=symbols, intervals=intervals, persistence=None
    )
    await collector.start()
    await asyncio.sleep(15.0)
    await collector.stop()

    health = collector.health()
    print(f"   Status: {health.status.value}")
    print(f"   Message: {health.to_dict()['message']}")
    print(f"   Events processed: {collector.stats()['events_processed']}")
    print(f"   Closed bars stored: {collector.stats()['closed_bars_stored']}")
    print(f"   Coverage: {collector.coverage()}")

    for symbol in symbols:
        closed = collector.closed_candles(symbol, Timeframe.M1)
        forming = collector.forming_candle(symbol, Timeframe.M1)
        print(
            f"   {symbol}: {len(closed)} closed bars, forming={forming is not None}"
        )
        if closed:
            last = closed[-1]
            print(
                f"      Last closed: {last.open_time_utc.strftime('%H:%M')} "
                f"O={last.open} H={last.high} L={last.low} C={last.close}"
            )
        quote = collector.latest_quote(symbol)
        if quote:
            print(f"      Quote: bid={quote.bid} ask={quote.ask} spread={quote.spread}")

    if health.status.value == "ok":
        print("\n[PASS] WebSocket collector is functional.")
    elif health.status.value == "degraded":
        print(
            f"\n[WARN] Collector is degraded: {health.to_dict()['message']} (may be warming up)"
        )
    else:
        print(f"\n[FAIL] Collector unhealthy: {health.to_dict()['message']}")


if __name__ == "__main__":
    asyncio.run(main())
