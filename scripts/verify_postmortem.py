"""Verify the loss post-mortem distinguishes causes it can actually measure.

Each scenario below builds a candle sequence whose shape is known by
construction, then asserts the diagnosis names the cause that shape represents.
The point is that four different losses produce four different diagnoses -- the
templated version produced identical text for all of them.

ASCII markers only: the Windows console is cp1252 and a check mark raises.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quantedge.contracts import AssetClass, Candle, SignalDirection, Timeframe
from quantedge.services.postmortem import diagnose

START = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)


def bar(i: int, o: float, h: float, low: float, c: float) -> Candle:
    """One closed 5m candle with the provenance the contracts require."""
    return Candle(
        provider="test",
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M5,
        open_time_utc=START + timedelta(minutes=5 * i),
        close_time_utc=START + timedelta(minutes=5 * (i + 1)),
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(low)),
        close=Decimal(str(c)),
        volume=Decimal("100"),
        is_closed=True,
    )


def codes(mortem) -> list[str]:
    return [c.code for c in mortem.causes]


def check(name: str, got: list[str], expected: str) -> bool:
    ok = expected in got
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: expected {expected}, got {got or 'none'}")
    return ok


def main() -> int:
    results: list[bool] = []
    ref = Decimal("100.00")
    stop = Decimal("98.00")
    target = Decimal("104.00")

    # 1. Stop taken out at 97.5, price recovers to close at 101 -> STOP_TOO_TIGHT.
    wick = [
        bar(0, 100, 100.5, 99.0, 99.5),
        bar(1, 99.5, 99.8, 97.5, 98.2),  # dips through the stop
        bar(2, 98.2, 101.2, 98.0, 101.0),  # and comes back above entry
    ]
    m = diagnose(
        symbol="BTCUSDT",
        direction=SignalDirection.UP,
        reference_price=ref,
        holding_candles=wick,
        stop=stop,
        target=target,
    )
    results.append(check("stop run then recovery", codes(m), "STOP_TOO_TIGHT"))
    print(f"       adverse={m.max_adverse_excursion} favourable={m.max_favourable_excursion}")

    # 2. Straight down through the stop, no recovery -> STOP_HIT_AND_HELD.
    trend_down = [
        bar(0, 100, 100.1, 99.0, 99.2),
        bar(1, 99.2, 99.3, 97.8, 98.0),
        bar(2, 98.0, 98.1, 96.0, 96.3),
    ]
    m = diagnose(
        symbol="BTCUSDT",
        direction=SignalDirection.UP,
        reference_price=ref,
        holding_candles=trend_down,
        stop=stop,
        target=target,
    )
    results.append(check("stop hit and held", codes(m), "STOP_HIT_AND_HELD"))

    # 3. Runs to 103.5 (87.5% of a 4.00 target) then closes at 99.5 -> gave it back.
    reversal = [
        bar(0, 100, 101.5, 99.9, 101.4),
        bar(1, 101.4, 103.5, 101.0, 103.0),
        bar(2, 103.0, 103.1, 99.4, 99.5),
    ]
    m = diagnose(
        symbol="BTCUSDT",
        direction=SignalDirection.UP,
        reference_price=ref,
        holding_candles=reversal,
        stop=stop,
        target=target,
    )
    results.append(check("profit given back", codes(m), "GAVE_BACK_OPEN_PROFIT"))
    print(f"       favourable excursion={m.max_favourable_excursion} of 4.00 target")

    # 4. Drifts inside a 0.6 range and expires -> EXPIRED_BEFORE_RESOLUTION.
    chop = [
        bar(0, 100, 100.3, 99.7, 100.1),
        bar(1, 100.1, 100.4, 99.8, 99.9),
        bar(2, 99.9, 100.2, 99.6, 99.8),
    ]
    m = diagnose(
        symbol="BTCUSDT",
        direction=SignalDirection.UP,
        reference_price=ref,
        holding_candles=chop,
        stop=stop,
        target=target,
    )
    results.append(check("expired unresolved", codes(m), "EXPIRED_BEFORE_RESOLUTION"))

    # 5. No candles at all -> says so, does not guess.
    m = diagnose(
        symbol="BTCUSDT",
        direction=SignalDirection.UP,
        reference_price=ref,
        holding_candles=[],
        stop=stop,
        target=target,
    )
    results.append(check("no data", codes(m), "NO_SETTLEMENT_DATA"))

    # 6. A forming bar must not decide whether the stop was touched.
    forming = bar(3, 99.8, 99.9, 90.0, 95.0)
    forming = forming.model_copy(update={"is_closed": False})
    m = diagnose(
        symbol="BTCUSDT",
        direction=SignalDirection.UP,
        reference_price=ref,
        holding_candles=[*chop, forming],
        stop=stop,
        target=target,
    )
    excluded = m.bars_observed == 3 and m.stop_touched is False
    print(
        f"[{'PASS' if excluded else 'FAIL'}] forming bar excluded: "
        f"bars={m.bars_observed} (expected 3), stop_touched={m.stop_touched} (expected False)"
    )
    results.append(excluded)

    # Distinct root causes, not one template with the symbol swapped.
    print("\n--- sample root cause (scenario 1) ---")
    m1 = diagnose(
        symbol="BTCUSDT",
        direction=SignalDirection.UP,
        reference_price=ref,
        holding_candles=wick,
        stop=stop,
        target=target,
    )
    print(m1.root_cause())

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
