"""Structure verification -- hand-placed pivots and a no-lookahead proof.

The important section is [2]. Swing detection is the easiest place in a trading
system to leak future information, and the leak is invisible in normal output:
the swings look right, the backtest looks good, and the live system underperforms
because at bar ``i`` the pivot the backtest used had not happened yet.

So rather than eyeballing pivots, section 2 proves the property directly. For
every prefix length ``k``, the swings found in ``bars[:k]`` must equal exactly
the swings the full series reports whose confirmation index is below ``k``.
If any pivot were detected before its confirming bars existed -- or changed its
mind once later bars arrived -- some prefix would disagree.

Run:  ./.venv/Scripts/python.exe -u scripts/verify_structure.py
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge.contracts import AssetClass, Candle, CandleSeries, SignalDirection, Timeframe
from quantedge.errors import InsufficientDataError
from quantedge.services import structure as st

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


START = datetime(2026, 1, 1, tzinfo=UTC)


def bar(index: int, high: float, low: float, close: float, *, is_closed: bool = True) -> Candle:
    return Candle(
        provider="fixture",
        symbol="TESTUSDT",
        asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M5,
        open_time_utc=START + timedelta(minutes=5 * index),
        close_time_utc=START + timedelta(minutes=5 * (index + 1)),
        open=Decimal(str(round(close, 4))),
        high=Decimal(str(round(high, 4))),
        low=Decimal(str(round(low, 4))),
        close=Decimal(str(round(close, 4))),
        volume=Decimal("1000"),
        is_closed=is_closed,
    )


def bars_from_centers(centers: list[float], *, spread: float = 0.5) -> list[Candle]:
    """One bar per centre, with a fixed range around it.

    The ordering of centres is preserved in both highs and lows, so a centre
    that is a local maximum is a swing high and nothing else can be.
    """
    return [bar(i, c + spread, c - spread, c) for i, c in enumerate(centers)]


def zigzag(count: int, *, base: float, drift: float, amplitude: float, period: int) -> list[float]:
    """A triangle wave with linear drift -- deterministic, no randomness.

    Half-period ``period/2`` sets the pivot spacing; as long as that exceeds the
    swing lookback, every turn is a strict local extreme.
    """
    out: list[float] = []
    half = period / 2.0
    for i in range(count):
        phase = i % period
        tri = phase / half if phase < half else 2.0 - phase / half
        out.append(base + drift * i + amplitude * tri)
    return out


def tapering_zigzag(
    count: int, *, base: float, amplitude: float, period: int, taper: float
) -> list[float]:
    """Triangle wave whose amplitude scales by ``taper`` per bar.

    ``taper`` below 1 contracts (lower highs against higher lows -> RANGE);
    above 1 it expands (higher highs against lower lows -> genuinely UNCLEAR).

    The wave is centred on ``base`` and swings to +/- the current amplitude, so
    the taper moves highs and lows in opposite directions. Scaling ``tri - 1``
    instead would pin every peak to ``base`` and taper only the troughs, which
    is a flat-topped wedge, not a triangle -- the highs would be exactly equal
    and no high/low sequence would be monotonic at all.
    """
    out: list[float] = []
    half = period / 2.0
    for i in range(count):
        phase = i % period
        tri = phase / half if phase < half else 2.0 - phase / half
        out.append(base + amplitude * (taper**i) * (2.0 * tri - 1.0))
    return out


# --------------------------------------------------------------------------- #
# Sections                                                                     #
# --------------------------------------------------------------------------- #


def section_hand_placed() -> None:
    print("\n[1] Swings at hand-placed pivots")
    centers = [10, 11, 12, 11, 10, 9, 10, 11, 13, 12, 11, 10, 9, 10, 11]
    bars = bars_from_centers([float(c) for c in centers])
    highs, lows = st.find_swings(bars, lookback=2, confirmation_bars=2)

    high_idx = [s["index"] for s in highs]
    low_idx = [s["index"] for s in lows]
    check("swing highs at the two peaks", high_idx == [2, 8], f"{high_idx}")
    check("swing lows at the two troughs", low_idx == [5, 12], f"{low_idx}")
    check(
        "confirmation index is the pivot plus the confirmation window",
        all(s["confirmed_at_index"] == s["index"] + 2 for s in highs + lows),
    )
    check(
        "pivot price is the bar's high", highs[0]["price"] == str(bars[2].high), highs[0]["price"]
    )
    check("pivot price is the bar's low", lows[0]["price"] == str(bars[5].low), lows[0]["price"])

    # Ordering: oldest first, so a caller can read the sequence directly.
    check("returned oldest first", high_idx == sorted(high_idx))


def section_no_lookahead() -> None:
    print("\n[2] NO LOOKAHEAD: every prefix agrees with the full series")
    for label, lookback, confirmation in (
        ("lookback=5 confirmation=5", 5, 5),
        ("lookback=3 confirmation=7", 3, 7),
        ("lookback=7 confirmation=3", 7, 3),
    ):
        centers = zigzag(300, base=100.0, drift=0.05, amplitude=6.0, period=22)
        bars = bars_from_centers(centers)
        full_h, full_l = st.find_swings(bars, lookback=lookback, confirmation_bars=confirmation)
        horizon = max(lookback, confirmation)

        disagreements: list[str] = []
        for k in range(1, len(bars) + 1):
            got_h, got_l = st.find_swings(
                bars[:k], lookback=lookback, confirmation_bars=confirmation
            )
            want_h = [s for s in full_h if s["index"] + horizon < k]
            want_l = [s for s in full_l if s["index"] + horizon < k]
            if [s["index"] for s in got_h] != [s["index"] for s in want_h]:
                disagreements.append(f"k={k} highs")
            if [s["index"] for s in got_l] != [s["index"] for s in want_l]:
                disagreements.append(f"k={k} lows")

        check(
            f"{label}: all 300 prefixes agree",
            not disagreements,
            f"{len(disagreements)} disagreement(s): {disagreements[:3]}",
        )
        check(
            f"{label}: nothing confirmed in the trailing {horizon} bars",
            all(s["index"] <= len(bars) - 1 - horizon for s in full_h + full_l),
        )
        check(
            f"{label}: swings were actually found",
            len(full_h) >= 5 and len(full_l) >= 5,
            f"{len(full_h)} highs, {len(full_l)} lows",
        )


def section_ties() -> None:
    print("\n[3] Exact double top yields no pivot (strict comparison)")
    centers = [10.0, 11.0, 12.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0]
    bars = bars_from_centers(centers)
    highs, _lows = st.find_swings(bars, lookback=2, confirmation_bars=2)
    check(
        "neither equal high becomes the pivot",
        all(s["index"] not in (2, 4) for s in highs),
        f"{[s['index'] for s in highs]}",
    )

    # And the same series with the tie broken does produce one.
    centers[4] = 12.5
    highs2, _ = st.find_swings(bars_from_centers(centers), lookback=2, confirmation_bars=2)
    check(
        "breaking the tie produces the pivot",
        4 in [s["index"] for s in highs2],
        f"{[s['index'] for s in highs2]}",
    )


def section_classification() -> None:
    print("\n[4] Structure labels require agreement between highs and lows")

    up = st.analyze_structure(
        bars_from_centers(zigzag(200, base=100.0, drift=0.25, amplitude=5.0, period=24)), atr=1.0
    )
    check("rising zigzag is UPTREND", up.structure == "UPTREND", up.structure)
    check("  higher highs detected", up.has_higher_highs is True)
    check("  higher lows detected", up.has_higher_lows is True)

    down = st.analyze_structure(
        bars_from_centers(zigzag(200, base=200.0, drift=-0.25, amplitude=5.0, period=24)), atr=1.0
    )
    check("falling zigzag is DOWNTREND", down.structure == "DOWNTREND", down.structure)
    check("  lower highs detected", down.has_lower_highs is True)
    check("  lower lows detected", down.has_lower_lows is True)

    contracting = st.analyze_structure(
        bars_from_centers(tapering_zigzag(220, base=100.0, amplitude=12.0, period=24, taper=0.985)),
        atr=1.0,
    )
    check("contracting triangle is RANGE", contracting.structure == "RANGE", contracting.structure)
    check(
        "  lower highs against higher lows",
        contracting.has_lower_highs and contracting.has_higher_lows,
    )

    expanding = st.analyze_structure(
        bars_from_centers(tapering_zigzag(220, base=100.0, amplitude=2.0, period=24, taper=1.012)),
        atr=1.0,
    )
    check(
        "expanding range is UNCLEAR, not a trend",
        expanding.structure == "UNCLEAR",
        expanding.structure,
    )
    check(
        "  higher highs against lower lows", expanding.has_higher_highs and expanding.has_lower_lows
    )
    check(
        "  the ambiguity is explained in notes",
        any("expanding range" in n for n in expanding.notes),
        str(expanding.notes),
    )


def section_breakout_buffer() -> None:
    print("\n[5] A breakout must clear the level by the ATR buffer, not touch it")
    # Range between roughly 95 and 105, then a final bar placed by hand.
    centers = zigzag(140, base=100.0, drift=0.0, amplitude=5.0, period=20)
    base_bars = bars_from_centers(centers)

    report = st.analyze_structure(base_bars, atr=1.0)
    level = report.nearest_resistance
    check("a resistance level was identified", level is not None, str(level))
    assert level is not None
    # Buffer = 0.25 x ATR = 0.25 with atr=1.0.

    def with_final_close(price: float) -> list[Candle]:
        tail = bar(len(base_bars), price + 0.05, price - 0.05, price)
        return [*base_bars, tail]

    touching = st.analyze_structure(with_final_close(float(level) + 0.10), atr=1.0)
    check(
        "close 0.10 above the level is NOT a breakout (buffer is 0.25)",
        touching.breakout_candidate is False,
        f"close={float(level) + 0.10:.4f} level={level}",
    )
    clearing = st.analyze_structure(with_final_close(float(level) + 0.60), atr=1.0)
    check(
        "close 0.60 above the level IS a breakout",
        clearing.breakout_candidate is True,
        f"close={float(level) + 0.60:.4f} level={level}",
    )
    check(
        "breakout direction is UP",
        clearing.breakout_direction == SignalDirection.UP,
        str(clearing.breakout_direction),
    )

    # The buffer must scale with volatility: the same bars at 10x ATR are quiet.
    loud = st.analyze_structure(with_final_close(float(level) + 0.60), atr=10.0)
    check(
        "the same close is not a breakout when ATR is 10x larger",
        loud.breakout_candidate is False,
        "buffer becomes 2.5",
    )

    support = report.nearest_support
    assert support is not None
    downside = st.analyze_structure(with_final_close(float(support) - 0.60), atr=1.0)
    check("a downside break is detected too", downside.breakout_candidate is True)
    check(
        "  direction is DOWN",
        downside.breakout_direction == SignalDirection.DOWN,
        str(downside.breakout_direction),
    )

    print(f"    resistance={level} support={support}")


def section_failed_breakout() -> None:
    print("\n[6] Failed breakout is reported as its own outcome")
    centers = zigzag(140, base=100.0, drift=0.0, amplitude=5.0, period=20)
    base_bars = bars_from_centers(centers)
    report = st.analyze_structure(base_bars, atr=1.0)
    level = report.nearest_resistance
    assert level is not None
    lvl = float(level)

    # Two bars ago price closed well clear of the level; the last bar closed back
    # underneath it. That is the classic false break.
    poke = bar(len(base_bars), lvl + 1.2, lvl - 0.2, lvl + 1.0)
    back_inside = bar(len(base_bars) + 1, lvl + 0.3, lvl - 2.0, lvl - 1.5)
    failed = st.analyze_structure([*base_bars, poke, back_inside], atr=1.0)

    check("failed_breakout is True", failed.failed_breakout is True)
    check(
        "it is NOT reported as a live breakout",
        failed.breakout_candidate is False,
        str(failed.breakout_direction),
    )
    check(
        "the note names the level and how long ago",
        any("failed upside breakout" in n for n in failed.notes),
        str([n for n in failed.notes if "failed" in n]),
    )

    # A clean hold above the level is not a failure.
    held = st.analyze_structure(
        [*base_bars, poke, bar(len(base_bars) + 1, lvl + 1.5, lvl + 0.8, lvl + 1.3)], atr=1.0
    )
    check("holding above the level is not a failed breakout", held.failed_breakout is False)


def section_rule_9() -> None:
    print("\n[7] RULE 9: a forming bar cannot create a swing")
    centers = zigzag(160, base=100.0, drift=0.1, amplitude=5.0, period=22)
    bars = bars_from_centers(centers)
    forming = bars[-1].model_copy(update={"is_closed": False})
    with_forming = [*bars[:-1], forming]

    try:
        st.analyze_structure(with_forming, atr=1.0)
    except InsufficientDataError as exc:
        check("raises on a forming bar in a raw list", True, exc.code)
    else:
        check("raises on a forming bar in a raw list", False, "no exception raised")

    series = CandleSeries(
        provider="fixture",
        symbol="TESTUSDT",
        asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M5,
        candles=with_forming,
        includes_forming_candle=True,
    )
    from_series = st.analyze_structure(series, atr=1.0)
    from_closed = st.analyze_structure(bars[:-1], atr=1.0)
    check(
        "CandleSeries drops the forming bar and gives the identical report",
        from_series.model_dump() == from_closed.model_dump(),
    )

    # A forming bar spiking to a new extreme must not become a swing high: it is
    # inside the trailing blind spot, so no amount of price action can confirm it.
    spike = bars[-1].model_copy(
        update={"high": Decimal("99999"), "close": Decimal("99999"), "is_closed": False}
    )
    spiked = CandleSeries(
        provider="fixture",
        symbol="TESTUSDT",
        asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M5,
        candles=[*bars[:-1], spike],
        includes_forming_candle=True,
    )
    spiked_report = st.analyze_structure(spiked, atr=1.0)
    check(
        "a forming bar spiking to 99999 produces no swing high",
        all(Decimal(s["price"]) < Decimal("99999") for s in spiked_report.swing_highs),
    )
    check(
        "and does not move resistance",
        spiked_report.nearest_resistance == from_closed.nearest_resistance,
        f"{spiked_report.nearest_resistance}",
    )


def section_insufficient() -> None:
    print("\n[8] Too little data is declared, not guessed")
    empty = st.analyze_structure([])
    check("empty input is UNCLEAR", empty.structure == "UNCLEAR")
    check("  and says so", any("no closed candles" in n for n in empty.notes), str(empty.notes))

    short = st.analyze_structure(bars_from_centers([100.0 + i for i in range(8)]))
    check("8 bars cannot confirm a swing", short.structure == "UNCLEAR", short.structure)
    check(
        "  the note states the requirement",
        any("below the" in n for n in short.notes),
        str(short.notes),
    )
    check("  no swings are invented", not short.swing_highs and not short.swing_lows)
    check("  no breakout is claimed", short.breakout_candidate is False)

    # Enough bars to confirm swings, but too few swings to classify structure.
    quiet = st.analyze_structure(bars_from_centers([100.0] * 40 + [100.5] * 40), atr=1.0)
    check("a featureless series stays UNCLEAR", quiet.structure == "UNCLEAR", quiet.structure)


def section_no_volatility_reference() -> None:
    print("\n[9] Missing ATR degrades to a stated fallback, never to a zero buffer silently")
    centers = zigzag(140, base=100.0, drift=0.0, amplitude=5.0, period=20)
    bars = bars_from_centers(centers)
    report = st.analyze_structure(bars, atr=None)
    check(
        "the ATR substitution is disclosed in notes",
        any("ATR unavailable" in n for n in report.notes),
        str([n for n in report.notes if "ATR" in n]),
    )
    check(
        "structure is still classified",
        report.structure in {"UPTREND", "DOWNTREND", "RANGE", "UNCLEAR"},
    )

    # Zero-range bars: no volatility reference is derivable at all, and that is
    # said out loud rather than producing a zero-buffer breakout on any tick.
    flat = [bar(i, 100.0, 100.0, 100.0) for i in range(40)]
    flat_report = st.analyze_structure(flat, atr=None)
    check(
        "a truly flat series says no volatility reference exists",
        any("no volatility reference" in n for n in flat_report.notes)
        or any("below the" in n for n in flat_report.notes),
        str(flat_report.notes),
    )
    check("and claims no breakout", flat_report.breakout_candidate is False)


def section_determinism() -> None:
    print("\n[10] Same input, same output")
    centers = zigzag(200, base=100.0, drift=0.15, amplitude=5.0, period=22)
    bars = bars_from_centers(centers)
    first = st.analyze_structure(bars, atr=1.0).model_dump()
    second = st.analyze_structure(list(bars), atr=1.0).model_dump()
    check("two runs are byte-identical", first == second)
    check("swing lists are capped at 5 for the payload", len(first["swing_highs"]) <= 5)
    check(
        "no float NaN leaked into the report",
        not any(isinstance(v, float) and math.isnan(v) for v in first.values()),
    )


def main() -> int:
    print("=" * 70)
    print("STRUCTURE VERIFICATION -- hand-placed pivots + no-lookahead proof")
    print("=" * 70)
    section_hand_placed()
    section_no_lookahead()
    section_ties()
    section_classification()
    section_breakout_buffer()
    section_failed_breakout()
    section_rule_9()
    section_insufficient()
    section_no_volatility_reference()
    section_determinism()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
