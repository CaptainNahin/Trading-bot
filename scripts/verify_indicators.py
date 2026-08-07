"""Indicator verification against independently computed values.

Two kinds of check, deliberately:

1. **Published reference series.** RSI(14) is checked against Wilder's worked
   example as republished by StockCharts -- 33 closes with 19 expected RSI
   values. If our smoothing were an EMA instead of Wilder's, these would all be
   wrong by several points.
2. **Independent re-implementation.** SMA, EMA, true range, ATR and Bollinger
   are recomputed here in plain Python loops with no numpy, from the formula.
   Agreeing with a second implementation written from the definition is a real
   check; agreeing with itself is not.

Run:  ./.venv/Scripts/python.exe -u scripts/verify_indicators.py
"""

from __future__ import annotations

import math
import statistics
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge.contracts import AssetClass, Candle, Timeframe
from quantedge.errors import InsufficientDataError
from quantedge.services import indicators

FAILURES: list[str] = []
TOL = 1e-9


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


def close_enough(a: float | None, b: float | None, tol: float) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# Reference data                                                               #
# --------------------------------------------------------------------------- #

# Wilder's RSI example (StockCharts republication). 33 closes.
RSI_CLOSES = [
    44.3389,
    44.0902,
    44.1497,
    43.6124,
    44.3278,
    44.8264,
    45.0955,
    45.4245,
    45.8433,
    46.0826,
    45.8931,
    46.0328,
    45.6140,
    46.2820,
    46.2820,
    46.0028,
    46.0328,
    46.4116,
    46.2222,
    45.6439,
    46.2122,
    46.2521,
    45.7137,
    46.4515,
    45.7835,
    45.3548,
    44.0288,
    44.1783,
    44.2181,
    44.5672,
    43.4205,
    42.6628,
    43.1314,
]

# Expected RSI(14), aligned to RSI_CLOSES index 14 onward.
RSI_EXPECTED = [
    70.53,
    66.32,
    66.55,
    69.41,
    66.36,
    57.97,
    62.93,
    63.26,
    56.06,
    62.38,
    54.71,
    50.42,
    39.99,
    41.46,
    41.87,
    45.46,
    37.30,
    33.08,
    37.77,
]


def naive_sma(values: list[float], period: int, index: int) -> float:
    """SMA from the definition: the mean of the trailing window."""
    window = values[index - period + 1 : index + 1]
    return sum(window) / len(window)


def naive_ema(values: list[float], period: int) -> list[float | None]:
    """EMA from the definition, SMA-seeded, in a plain loop."""
    alpha = 2.0 / (period + 1.0)
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    prev = sum(values[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def naive_true_range(h: list[float], low: list[float], c: list[float]) -> list[float | None]:
    out: list[float | None] = [None]
    for i in range(1, len(h)):
        out.append(max(h[i] - low[i], abs(h[i] - c[i - 1]), abs(low[i] - c[i - 1])))
    return out


def naive_wilder(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    prev = sum(values[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = prev + (values[i] - prev) / period
        out[i] = prev
    return out


def synthetic_candles(count: int, *, seed_price: float = 100.0) -> list[Candle]:
    """Deterministic pseudo-random OHLCV bars.

    A fixed recurrence rather than ``random`` so every run of this script tests
    the identical series and a regression cannot hide behind a new sample.
    """
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars: list[Candle] = []
    price = seed_price
    for i in range(count):
        # Deterministic wobble: two incommensurate sinusoids plus a slow drift.
        drift = i * 0.02
        wobble = math.sin(i * 0.7) * 1.5 + math.cos(i * 0.31) * 0.8
        open_p = price
        close_p = seed_price + drift + wobble
        high_p = max(open_p, close_p) + abs(math.sin(i * 1.3)) * 0.6
        low_p = min(open_p, close_p) - abs(math.cos(i * 1.1)) * 0.6
        bars.append(
            Candle(
                provider="fixture",
                symbol="TESTUSDT",
                asset_class=AssetClass.CRYPTO,
                timeframe=Timeframe.M5,
                open_time_utc=start + timedelta(minutes=5 * i),
                close_time_utc=start + timedelta(minutes=5 * (i + 1)),
                open=Decimal(str(round(open_p, 4))),
                high=Decimal(str(round(high_p, 4))),
                low=Decimal(str(round(low_p, 4))),
                close=Decimal(str(round(close_p, 4))),
                volume=Decimal(str(round(1000 + math.sin(i * 0.5) * 300, 4))),
                is_closed=True,
            )
        )
        price = close_p
    return bars


# --------------------------------------------------------------------------- #
# Sections                                                                     #
# --------------------------------------------------------------------------- #


def section_sma() -> None:
    print("\n[1] SMA against the definition")
    values = [float(v) for v in range(1, 51)]
    got = indicators.sma(values, 10)
    check("undefined before warm-up", bool(math.isnan(got[8])), "index 8 is NaN")
    check("defined at exactly period-1", not math.isnan(got[9]), f"index 9 = {got[9]}")
    # mean(1..10) == 5.5
    check("first value correct", close_enough(float(got[9]), 5.5, TOL), f"{got[9]}")
    for idx in (9, 20, 33, 49):
        expected = naive_sma(values, 10, idx)
        check(f"matches naive SMA at {idx}", close_enough(float(got[idx]), expected, 1e-9))


def section_ema() -> None:
    print("\n[2] EMA against an independent loop implementation")
    bars = synthetic_candles(300)
    closes = [float(c.close) for c in bars]
    for period in (9, 20, 50, 200):
        got = indicators.ema(closes, period)
        expected = naive_ema(closes, period)
        mismatches = [
            i
            for i in range(len(closes))
            if (math.isnan(got[i]) != (expected[i] is None))
            or (expected[i] is not None and not close_enough(float(got[i]), expected[i], 1e-9))
        ]
        check(f"EMA-{period} matches at every index", not mismatches, f"{len(mismatches)} mismatch")
    got9 = indicators.ema(closes, 9)
    check("EMA seeded with SMA of first n", close_enough(float(got9[8]), sum(closes[:9]) / 9, 1e-9))


def section_rsi() -> None:
    print("\n[3] RSI(14) against Wilder's published example")
    got = indicators.rsi(RSI_CLOSES, 14)
    check("undefined at index 13", bool(math.isnan(got[13])))
    check("defined at index 14", not math.isnan(got[14]), f"{got[14]:.4f}")
    worst = 0.0
    for offset, expected in enumerate(RSI_EXPECTED):
        idx = 14 + offset
        worst = max(worst, abs(float(got[idx]) - expected))
    # Published values are rounded to 2dp, so 0.01 of rounding is unavoidable.
    check("all 19 values within 0.02 of published", worst <= 0.02, f"max deviation {worst:.4f}")
    check("stays within [0, 100]", all(0.0 <= v <= 100.0 for v in got[14:]))

    rising = float(indicators.rsi([float(x) for x in range(1, 40)], 14)[-1])
    check("pure uptrend gives RSI 100", close_enough(rising, 100.0, TOL), f"{rising}")
    falling = float(indicators.rsi([float(x) for x in range(40, 1, -1)], 14)[-1])
    check("pure downtrend gives RSI 0", close_enough(falling, 0.0, TOL), f"{falling}")


def section_atr() -> None:
    print("\n[4] True range and ATR against the definition")
    bars = synthetic_candles(120)
    h = [float(c.high) for c in bars]
    low = [float(c.low) for c in bars]
    c = [float(x.close) for x in bars]

    tr = indicators.true_range(h, low, c)
    naive_tr = naive_true_range(h, low, c)
    check("TR undefined at index 0 (no prior close)", bool(math.isnan(tr[0])))
    tr_mismatch = [i for i in range(1, len(h)) if not close_enough(float(tr[i]), naive_tr[i], 1e-9)]
    check("TR matches definition at every bar", not tr_mismatch, f"{len(tr_mismatch)} mismatch")

    got = indicators.atr(h, low, c, 14)
    expected = naive_wilder([v for v in naive_tr if v is not None], 14)
    atr_mismatch = [
        i
        for i in range(len(expected))
        if (math.isnan(got[i + 1]) != (expected[i] is None))
        or (expected[i] is not None and not close_enough(float(got[i + 1]), expected[i], 1e-9))
    ]
    check("ATR matches Wilder smoothing of TR", not atr_mismatch, f"{len(atr_mismatch)} mismatch")
    check("ATR is never negative", all(v >= 0 for v in got[15:]))

    # A gap must widen ATR: same bar ranges, different prior close.
    gapped = indicators.true_range([10.0, 12.0], [9.0, 11.0], [9.5, 11.5])
    check("TR includes the gap term", close_enough(float(gapped[1]), 2.5, TOL), f"{gapped[1]}")


def section_macd() -> None:
    print("\n[5] MACD = EMA12 - EMA26, signal = EMA9 of the MACD line")
    bars = synthetic_candles(300)
    closes = [float(x.close) for x in bars]
    line, signal, hist = indicators.macd(closes)

    ema12 = naive_ema(closes, 12)
    ema26 = naive_ema(closes, 26)
    line_mismatch = [
        i
        for i in range(len(closes))
        if ema12[i] is not None
        and ema26[i] is not None
        and not close_enough(float(line[i]), ema12[i] - ema26[i], 1e-9)
    ]
    check("MACD line matches EMA12-EMA26", not line_mismatch, f"{len(line_mismatch)} mismatch")
    check("line undefined before the slow EMA exists", bool(math.isnan(line[24])))

    # The signal EMA must be seeded on the MACD line's first defined value (25),
    # not on the leading NaNs. First signal value therefore lands at 25+8 = 33.
    check("signal undefined at index 32", bool(math.isnan(signal[32])))
    check("signal defined at index 33", not math.isnan(signal[33]), f"{signal[33]:.6f}")
    seed = sum(float(line[i]) for i in range(25, 34)) / 9.0
    check("signal seeded with SMA9 of the line", close_enough(float(signal[33]), seed, 1e-9))
    hist_ok = all(
        close_enough(float(hist[i]), float(line[i] - signal[i]), 1e-12) for i in range(33, 300)
    )
    check("histogram == line - signal", hist_ok)


def section_adx() -> None:
    print("\n[6] ADX / DI invariants")
    bars = synthetic_candles(300)
    h = [float(c.high) for c in bars]
    low = [float(c.low) for c in bars]
    c = [float(x.close) for x in bars]
    adx_v, plus_di, minus_di = indicators.adx(h, low, c, 14)

    defined = [i for i in range(len(h)) if not math.isnan(adx_v[i])]
    check("ADX becomes defined", bool(defined), f"first at index {defined[0] if defined else '-'}")
    check("ADX not defined before 2x warm-up", defined[0] >= 27, f"index {defined[0]}")
    check("ADX within [0, 100]", all(0.0 <= adx_v[i] <= 100.0 for i in defined))
    di_defined = [i for i in range(len(h)) if not math.isnan(plus_di[i])]
    check("+DI within [0, 100]", all(0.0 <= plus_di[i] <= 100.0 for i in di_defined))
    check("-DI within [0, 100]", all(0.0 <= minus_di[i] <= 100.0 for i in di_defined))

    # A monotonic ramp is a maximally directional market: +DI must dominate.
    ramp_h = [100.0 + i for i in range(120)]
    ramp_l = [99.0 + i for i in range(120)]
    ramp_c = [99.5 + i for i in range(120)]
    r_adx, r_plus, r_minus = indicators.adx(ramp_h, ramp_l, ramp_c, 14)
    check(
        "steady uptrend: +DI > -DI",
        float(r_plus[-1]) > float(r_minus[-1]),
        f"+DI={r_plus[-1]:.2f} -DI={r_minus[-1]:.2f}",
    )
    check("steady uptrend: ADX is high", float(r_adx[-1]) > 90.0, f"ADX={r_adx[-1]:.2f}")

    # A perfectly flat market is directionless, not unknown: DX is 0 by definition.
    flat = [100.0] * 120
    f_adx, f_plus, f_minus = indicators.adx(flat, flat, flat, 14)
    flat_adx = float(f_adx[-1])
    check("flat market: ADX is 0, not NaN", close_enough(flat_adx, 0.0, TOL), f"{flat_adx}")
    check(
        "flat market: both DIs are 0 (no directional movement occurred)",
        close_enough(float(f_plus[-1]), 0.0, TOL) and close_enough(float(f_minus[-1]), 0.0, TOL),
        f"+DI={f_plus[-1]} -DI={f_minus[-1]}",
    )


def section_bollinger() -> None:
    print("\n[7] Bollinger Bands with population stdev")
    bars = synthetic_candles(120)
    closes = [float(c.close) for c in bars]
    upper, middle, lower = indicators.bollinger(closes, 20, 2.0)

    idx = 100
    window = closes[idx - 19 : idx + 1]
    mean = sum(window) / 20
    pop_std = statistics.pstdev(window)
    check("middle band == SMA20", close_enough(float(middle[idx]), mean, 1e-9))
    check("upper == mean + 2*pop_stdev", close_enough(float(upper[idx]), mean + 2 * pop_std, 1e-9))
    check("lower == mean - 2*pop_stdev", close_enough(float(lower[idx]), mean - 2 * pop_std, 1e-9))
    sample_std = statistics.stdev(window)
    check(
        "NOT the sample stdev (ddof=1)",
        not close_enough(float(upper[idx]), mean + 2 * sample_std, 1e-6),
        f"pop={pop_std:.6f} sample={sample_std:.6f}",
    )
    check("undefined before 20 bars", bool(math.isnan(middle[18])))

    # Zero-variance input: bands collapse onto the mean rather than going NaN.
    flat = indicators.bollinger([50.0] * 40, 20, 2.0)
    check("flat series: bands collapse to the mean", close_enough(float(flat[0][-1]), 50.0, TOL))


def section_roc() -> None:
    print("\n[8] ROC")
    values = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 121.0]
    got = indicators.roc(values, 10)
    check("undefined before period", bool(math.isnan(got[9])))
    check("10% over 10 bars", close_enough(float(got[10]), 10.0, 1e-9), f"{got[10]}")
    expected_roc = (121.0 - 101.0) / 101.0 * 100.0
    check("+19.8% from 101 to 121", close_enough(float(got[11]), expected_roc, 1e-9))


def section_features_contract() -> None:
    print("\n[9] compute_features: warm-up gating and the missing_features list")
    short = synthetic_candles(30)
    snap = indicators.compute_features(short)
    check("bars_used reported honestly", snap.bars_used == 30, f"{snap.bars_used}")
    check("warmup_satisfied is False at 30 bars", snap.warmup_satisfied is False)
    check("ema_200 is None, not 0", snap.ema_200 is None)
    check("ema_200 named in missing_features", "ema_200" in snap.missing_features)
    check("sma_200 named in missing_features", "sma_200" in snap.missing_features)
    check("adx_14 present at 30 bars", snap.adx_14 is not None, f"{snap.adx_14}")
    check("rsi_14 present at 30 bars", snap.rsi_14 is not None, f"{snap.rsi_14:.2f}")
    check(
        "no fabricated zeros",
        all(getattr(snap, f) is None for f in snap.missing_features if hasattr(snap, f)),
    )

    full = synthetic_candles(260)
    snap2 = indicators.compute_features(full)
    check("warmup_satisfied is True at 260 bars", snap2.warmup_satisfied is True)
    check(
        "missing_features is empty when warmed",
        not snap2.missing_features,
        f"{snap2.missing_features}",
    )
    check("ema_200 computed", snap2.ema_200 is not None, f"{snap2.ema_200:.4f}")
    check("as_of is the last closed bar", snap2.as_of_candle_close_utc == full[-1].close_time_utc)
    check("close preserved as Decimal", isinstance(snap2.close, Decimal), f"{snap2.close}")
    check("close equals the last bar", snap2.close == full[-1].close)

    # Ratios describe the last bar's anatomy and must sum to 1.
    total = (snap2.body_ratio or 0) + (snap2.upper_wick_ratio or 0) + (snap2.lower_wick_ratio or 0)
    check("body + wicks == 1", close_enough(total, 1.0, 1e-9), f"{total}")


def section_rule_9() -> None:
    print("\n[10] RULE 9: a forming candle can never enter a calculation")
    bars = synthetic_candles(60)
    forming = bars[-1].model_copy(update={"is_closed": False})
    with_forming = [*bars[:-1], forming]

    try:
        indicators.compute_features(with_forming)
    except InsufficientDataError as exc:
        check("raises on a forming candle in a raw list", True, exc.code)
    else:
        check("raises on a forming candle in a raw list", False, "no exception raised")

    # Via CandleSeries the forming bar is dropped by `.closed`, and the result
    # must equal the same computation without it -- not merely "not crash".
    from quantedge.contracts import CandleSeries

    series = CandleSeries(
        provider="fixture",
        symbol="TESTUSDT",
        asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M5,
        candles=with_forming,
        includes_forming_candle=True,
    )
    from_series = indicators.compute_features(series)
    from_closed = indicators.compute_features(bars[:-1])
    check(
        "series result uses closed bars only",
        from_series.bars_used == 59,
        f"{from_series.bars_used}",
    )
    check(
        "identical to computing on closed bars alone",
        from_series.rsi_14 == from_closed.rsi_14 and from_series.ema_20 == from_closed.ema_20,
    )

    try:
        indicators.compute_features(bars[:1])
    except InsufficientDataError as exc:
        check("raises below 2 closed bars", True, exc.code)
    else:
        check("raises below 2 closed bars", False, "no exception raised")


def section_degenerate() -> None:
    print("\n[11] Degenerate inputs return None rather than inf/NaN")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    flat_bars = [
        Candle(
            provider="fixture",
            symbol="FLAT",
            asset_class=AssetClass.CRYPTO,
            timeframe=Timeframe.M5,
            open_time_utc=start + timedelta(minutes=5 * i),
            close_time_utc=start + timedelta(minutes=5 * (i + 1)),
            open=Decimal("50"),
            high=Decimal("50"),
            low=Decimal("50"),
            close=Decimal("50"),
            volume=Decimal("0"),
            is_closed=True,
        )
        for i in range(60)
    ]
    snap = indicators.compute_features(flat_bars)
    numeric = {name: value for name, value in snap.model_dump().items() if isinstance(value, float)}
    bad = {k: v for k, v in numeric.items() if math.isnan(v) or math.isinf(v)}
    check("no NaN or inf reaches the contract", not bad, f"{bad}")
    check("zero-range bar: body_ratio is None", snap.body_ratio is None)
    check("zero-volume: volume_change is None", snap.volume_change_percent is None)
    check("flat market: ATR is 0", close_enough(snap.atr_14, 0.0, TOL), f"{snap.atr_14}")
    check(
        "flat market: RSI is 50 (no movement either way)",
        close_enough(snap.rsi_14, 50.0, TOL),
        f"{snap.rsi_14}",
    )


def main() -> int:
    print("=" * 70)
    print("INDICATOR VERIFICATION -- independent recomputation + published series")
    print("=" * 70)
    section_sma()
    section_ema()
    section_rsi()
    section_atr()
    section_macd()
    section_adx()
    section_bollinger()
    section_roc()
    section_features_contract()
    section_rule_9()
    section_degenerate()

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
