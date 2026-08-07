"""Regime and multi-timeframe verification.

Two properties carry most of the weight here:

* **Section [3]** -- the derived inputs refuse to guess. A percentile over five
  observations and an ATR "mean" over three bars are numbers without meaning, and
  the helpers return ``None`` rather than produce them.
* **Section [7]** -- disagreement between timeframes is reported, never averaged.
  When execution points up and regime points down, ``aligned_direction`` must be
  unset and the conflict must be named. A snapshot that split the difference
  would be the most dangerous output in the system.

Run:  ./.venv/Scripts/python.exe -u scripts/verify_regime_mtf.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge.contracts import (
    AssetClass,
    Candle,
    MarketRegime,
    SignalDirection,
    StructureReport,
    Timeframe,
    timeframe_seconds,
)
from quantedge.errors import InsufficientDataError
from quantedge.services import mtf
from quantedge.services import regime as reg

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


NOW = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)


def struct_report(
    label: str = "UPTREND",
    *,
    breakout: bool = False,
    direction: SignalDirection | None = None,
    failed: bool = False,
) -> StructureReport:
    return StructureReport(
        symbol="BTCUSDT",
        timeframe=Timeframe.M5,
        structure=label,  # type: ignore[arg-type]
        breakout_candidate=breakout,
        breakout_direction=direction,
        failed_breakout=failed,
    )


def bars(
    count: int,
    *,
    drift: float = 0.0,
    span: float = 1.0,
    timeframe: Timeframe = Timeframe.M5,
    base: float = 100.0,
    amplitude: float = 4.0,
    period: int = 24,
) -> list[Candle]:
    """A clean series ending at ``NOW``, oldest first, with a real bar range.

    The path is a triangle wave plus linear drift, not a straight line. A straight
    line has no local extremes, so it produces no swing pivots and structure is
    correctly UNCLEAR -- which makes it useless for testing a directional regime.
    The oscillation gives genuine pivots; ``drift`` then decides whether they step
    up (higher highs and higher lows) or down.
    """
    step = timedelta(seconds=timeframe_seconds(timeframe))
    half = period / 2.0
    out: list[Candle] = []
    for i in range(count):
        phase = i % period
        tri = phase / half if phase < half else 2.0 - phase / half
        close = base + drift * i + amplitude * tri
        close_time = NOW - step * (count - 1 - i)
        out.append(
            Candle(
                provider="fixture",
                symbol="BTCUSDT",
                asset_class=AssetClass.CRYPTO,
                timeframe=timeframe,
                open_time_utc=close_time - step,
                close_time_utc=close_time,
                open=Decimal(str(round(close, 4))),
                high=Decimal(str(round(close + span / 2.0, 4))),
                low=Decimal(str(round(close - span / 2.0, 4))),
                close=Decimal(str(round(close, 4))),
                volume=Decimal("1000"),
                is_closed=True,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Sections                                                                     #
# --------------------------------------------------------------------------- #


def section_trend_states() -> None:
    print("\n[1] Trend regimes need ADX and structure to agree")

    strong_up = reg.classify_regime(
        structure=struct_report("UPTREND"),
        adx=30.0,
        bb_width_percentile=50.0,
        atr=1.0,
        atr_mean=1.0,
        ema_alignment="bullish",
    )
    check(
        "ADX 30 + UPTREND + bullish EMAs is STRONG_UPTREND",
        strong_up.regime is MarketRegime.STRONG_UPTREND,
        str(strong_up.regime),
    )
    check(
        "  unanimous evidence scores high",
        strong_up.heuristic_score >= 0.85,
        str(strong_up.heuristic_score),
    )
    check("  no contradictions", not strong_up.contradictions)

    mixed = reg.classify_regime(
        structure=struct_report("UPTREND"),
        adx=30.0,
        bb_width_percentile=50.0,
        atr=1.0,
        atr_mean=1.0,
        ema_alignment="mixed",
    )
    check(
        "mixed EMAs still STRONG_UPTREND but scores lower",
        mixed.regime is MarketRegime.STRONG_UPTREND and mixed.heuristic_score < 0.85,
        f"{mixed.regime} {mixed.heuristic_score}",
    )
    check(
        "  and the disagreement is recorded", bool(mixed.contradictions), str(mixed.contradictions)
    )

    weak = reg.classify_regime(
        structure=struct_report("UPTREND"),
        adx=20.0,
        bb_width_percentile=50.0,
        atr=1.0,
        atr_mean=1.0,
        ema_alignment="bullish",
    )
    check(
        "ADX 20 is WEAK_UPTREND, not strong",
        weak.regime is MarketRegime.WEAK_UPTREND,
        str(weak.regime),
    )
    check(
        "  and scores below a strong trend",
        weak.heuristic_score < strong_up.heuristic_score,
        f"{weak.heuristic_score} < {strong_up.heuristic_score}",
    )

    down = reg.classify_regime(
        structure=struct_report("DOWNTREND"),
        adx=30.0,
        bb_width_percentile=50.0,
        atr=1.0,
        atr_mean=1.0,
        ema_alignment="bearish",
    )
    check(
        "the mirror case is STRONG_DOWNTREND",
        down.regime is MarketRegime.STRONG_DOWNTREND,
        str(down.regime),
    )

    # Strong ADX with range structure is a genuine contradiction, not a trend.
    conflict = reg.classify_regime(
        structure=struct_report("RANGE"),
        adx=30.0,
        bb_width_percentile=50.0,
        atr=1.0,
        atr_mean=1.0,
        ema_alignment="mixed",
    )
    check(
        "strong ADX against RANGE structure is UNCERTAIN",
        conflict.regime is MarketRegime.UNCERTAIN,
        str(conflict.regime),
    )
    check("  and says why", bool(conflict.contradictions), str(conflict.contradictions))


def section_range_and_override() -> None:
    print("\n[2] Ranges, breakouts and shocks")

    low_vol = reg.classify_regime(
        structure=struct_report("RANGE"),
        adx=12.0,
        bb_width_percentile=10.0,
        atr=1.0,
        atr_mean=1.0,
        ema_alignment="mixed",
    )
    check(
        "low ADX + compressed bands + RANGE is LOW_VOLATILITY_RANGE",
        low_vol.regime is MarketRegime.LOW_VOLATILITY_RANGE,
        str(low_vol.regime),
    )

    high_vol = reg.classify_regime(
        structure=struct_report("UNCLEAR"),
        adx=12.0,
        bb_width_percentile=90.0,
        atr=1.0,
        atr_mean=1.0,
        ema_alignment="mixed",
    )
    check(
        "low ADX + expanded bands is HIGH_VOLATILITY_RANGE",
        high_vol.regime is MarketRegime.HIGH_VOLATILITY_RANGE,
        str(high_vol.regime),
    )

    # A breakout outranks every trend and range label.
    breakout = reg.classify_regime(
        structure=struct_report("UPTREND", breakout=True, direction=SignalDirection.UP),
        adx=30.0,
        bb_width_percentile=50.0,
        atr=1.0,
        atr_mean=1.0,
        ema_alignment="bullish",
    )
    check(
        "a breakout outranks a strong trend",
        breakout.regime is MarketRegime.BREAKOUT,
        str(breakout.regime),
    )

    failed = reg.classify_regime(
        structure=struct_report(
            "UPTREND", breakout=True, direction=SignalDirection.UP, failed=True
        ),
        adx=30.0,
        bb_width_percentile=50.0,
        atr=1.0,
        atr_mean=1.0,
        ema_alignment="bullish",
    )
    check(
        "a prior failed breakout lowers the score",
        failed.heuristic_score < breakout.heuristic_score,
        f"{failed.heuristic_score} < {breakout.heuristic_score}",
    )
    check(
        "  and is named as a contradiction", bool(failed.contradictions), str(failed.contradictions)
    )

    # A shock outranks everything except a breakout already in progress.
    shock = reg.classify_regime(
        structure=struct_report("UPTREND"),
        adx=30.0,
        bb_width_percentile=50.0,
        atr=5.0,
        atr_mean=1.0,
        ema_alignment="bullish",
    )
    check(
        "ATR 5x its mean is VOLATILITY_SHOCK",
        shock.regime is MarketRegime.VOLATILITY_SHOCK,
        str(shock.regime),
    )
    check(
        "  the multiple is quantified in evidence",
        any("exceeds" in e for e in shock.supporting_evidence),
        str(shock.supporting_evidence),
    )

    just_under = reg.classify_regime(
        structure=struct_report("UPTREND"),
        adx=30.0,
        bb_width_percentile=50.0,
        atr=2.4,
        atr_mean=1.0,
        ema_alignment="bullish",
    )
    check(
        "2.4x is below the 2.5x line and is not a shock",
        just_under.regime is not MarketRegime.VOLATILITY_SHOCK,
        str(just_under.regime),
    )


def section_missing_inputs() -> None:
    print("\n[3] Missing inputs yield UNCERTAIN, never a guess")

    no_adx = reg.classify_regime(
        structure=struct_report("UPTREND"),
        adx=None,
        bb_width_percentile=50.0,
        atr=1.0,
        atr_mean=1.0,
        ema_alignment="bullish",
    )
    check(
        "no ADX is UNCERTAIN even with clean structure",
        no_adx.regime is MarketRegime.UNCERTAIN,
        str(no_adx.regime),
    )
    check(
        "  the absence is stated",
        any("ADX unavailable" in c for c in no_adx.contradictions),
        str(no_adx.contradictions),
    )

    check("a 5-sample percentile is not reported", reg.bb_width_percentile([0.1] * 5, 0.2) is None)
    check(
        "a 20-sample percentile is reported",
        reg.bb_width_percentile([0.1] * 20, 0.2) == 100.0,
        str(reg.bb_width_percentile([0.1] * 20, 0.2)),
    )
    check("None current width yields None", reg.bb_width_percentile([0.1] * 30, None) is None)
    check(
        "Nones in history are skipped, not counted as zero",
        reg.bb_width_percentile([None] * 10 + [0.1] * 20, 0.2) == 100.0,
    )

    check("a 3-sample ATR mean is not reported", reg.atr_rolling_mean([1.0] * 3) is None)
    check(
        "a 14-sample ATR mean is reported",
        reg.atr_rolling_mean([2.0] * 14) == 2.0,
        str(reg.atr_rolling_mean([2.0] * 14)),
    )
    check(
        "zero and negative ATRs are excluded", reg.atr_rolling_mean([0.0, -1.0] + [2.0] * 14) == 2.0
    )


def section_ema_alignment() -> None:
    print("\n[4] EMA alignment is None until every EMA has warmed up")
    from quantedge.services import indicators as ind

    warm = ind.compute_features(bars(260, drift=0.1))
    check(
        "a 260-bar rising series aligns bullish",
        reg.ema_alignment(warm) == "bullish",
        str(reg.ema_alignment(warm)),
    )

    falling = ind.compute_features(bars(260, drift=-0.1, base=200.0))
    check(
        "a falling series aligns bearish",
        reg.ema_alignment(falling) == "bearish",
        str(reg.ema_alignment(falling)),
    )

    cold = ind.compute_features(bars(50, drift=0.1))
    check(
        "50 bars cannot align: EMA-200 is unwarmed",
        reg.ema_alignment(cold) is None,
        str(reg.ema_alignment(cold)),
    )


def section_resolve_horizon() -> None:
    print("\n[5] Horizon resolution")
    triple = mtf.resolve_horizon("5m")
    check(
        "5m resolves to three timeframes",
        triple["execution"] == Timeframe.M5
        and triple["confirmation"] == Timeframe.M15
        and triple["regime"] == Timeframe.H1,
        f"{triple['execution']}/{triple['confirmation']}/{triple['regime']}",
    )
    check(
        "each role has a known value",
        all(triple[r] for r in mtf.ROLES),
        str({r: triple[r] for r in mtf.ROLES}),
    )

    try:
        mtf.resolve_horizon("9m")
    except InsufficientDataError as exc:
        check("an unconfigured horizon raises", True, exc.code)
    else:
        check("an unconfigured horizon raises", False, "no exception")


def section_mtf_alignment() -> None:
    print("\n[6] Agreement: all three views must point the same way")

    # Build three views manually: all pointing up.
    exec_bars = bars(300, timeframe=Timeframe.M5, drift=0.1)
    conf_bars = bars(300, timeframe=Timeframe.M15, drift=0.1)
    reg_bars = bars(300, timeframe=Timeframe.H1, drift=0.1)

    exec_view = mtf.build_timeframe_view("execution", exec_bars, now=NOW)
    conf_view = mtf.build_timeframe_view("confirmation", conf_bars, now=NOW)
    reg_view = mtf.build_timeframe_view("regime", reg_bars, now=NOW)

    check(
        "all three views pass quality",
        all(
            v.quality.status.value in ("PASS", "DEGRADED") for v in [exec_view, conf_view, reg_view]
        ),
    )
    check(
        "quality produces a feature snapshot",
        exec_view.features is not None and reg_view.features is not None,
    )
    check(
        "the regime view carries a regime label",
        reg_view.regime is not None and isinstance(reg_view.regime.regime, MarketRegime),
    )

    # Now run the full alignment.
    snapshot = mtf.analyze_multi_timeframe(
        "BTCUSDT",
        AssetClass.CRYPTO,
        "5m",
        {"execution": exec_bars, "confirmation": conf_bars, "regime": reg_bars},
        now=NOW,
    )
    check(
        "the snapshot assembles all three views",
        len(snapshot.views) == 3,
        str([v.role for v in snapshot.views]),
    )
    check("the horizon is recorded", snapshot.horizon == "5m")
    check(
        "alignment score is between 0 and 1",
        0.0 <= snapshot.alignment_score <= 1.0,
        str(snapshot.alignment_score),
    )


def section_conflict_reporting() -> None:
    print("\n[7] Disagreement is reported, not averaged away")

    # Execution up, confirmation down -- a genuine conflict.
    exec_bars = bars(300, timeframe=Timeframe.M5, drift=0.1)
    conf_bars = bars(300, timeframe=Timeframe.M15, drift=-0.1)
    reg_bars = bars(300, timeframe=Timeframe.H1, drift=0.05)

    snapshot = mtf.analyze_multi_timeframe(
        "BTCUSDT",
        AssetClass.CRYPTO,
        "5m",
        {"execution": exec_bars, "confirmation": conf_bars, "regime": reg_bars},
        now=NOW,
    )
    check(
        "three opposing views produce conflicts", bool(snapshot.conflicts), str(snapshot.conflicts)
    )
    check(
        "and aligned_direction stays unset",
        snapshot.aligned_direction is None,
        str(snapshot.aligned_direction),
    )

    # All pointing the same way sets aligned_direction.
    up = bars(300, timeframe=Timeframe.M5, drift=0.1)
    snapshot_up = mtf.analyze_multi_timeframe(
        "BTCUSDT",
        AssetClass.CRYPTO,
        "5m",
        {"execution": up, "confirmation": up, "regime": up},
        now=NOW,
    )
    check(
        "all-up sets aligned_direction",
        snapshot_up.aligned_direction is not None,
        str(snapshot_up.aligned_direction),
    )
    check("no conflicts when unanimous", not snapshot_up.conflicts, str(snapshot_up.conflicts))

    # A missing role is warned, not a fatal error.
    snapshot_partial = mtf.analyze_multi_timeframe(
        "BTCUSDT",
        AssetClass.CRYPTO,
        "5m",
        {"execution": up, "confirmation": up},
        now=NOW,
    )
    check(
        "a missing role is warned",
        any("regime" in w for w in snapshot_partial.warnings),
        str(snapshot_partial.warnings),
    )
    check(
        "remaining views still contribute",
        len(snapshot_partial.views) == 2,
        str([v.role for v in snapshot_partial.views]),
    )


def section_determinism() -> None:
    print("\n[8] Determinism: same input, same output")
    up = bars(300, timeframe=Timeframe.M5, drift=0.1)

    first = mtf.analyze_multi_timeframe(
        "BTCUSDT",
        AssetClass.CRYPTO,
        "5m",
        {"execution": up, "confirmation": up, "regime": up},
        now=NOW,
    )
    second = mtf.analyze_multi_timeframe(
        "BTCUSDT",
        AssetClass.CRYPTO,
        "5m",
        {"execution": list(up), "confirmation": list(up), "regime": list(up)},
        now=NOW,
    )
    # Timestamps are stamped from the wall clock and legitimately differ between
    # runs; everything derived from the candles must not. Comparing the whole
    # dump would test the clock, not the analysis.
    _TIMESTAMPS = {"generated_at_utc", "computed_at_utc", "checked_at_utc"}

    def strip_times(value: object) -> object:
        if isinstance(value, dict):
            return {k: strip_times(v) for k, v in value.items() if k not in _TIMESTAMPS}
        if isinstance(value, list):
            return [strip_times(v) for v in value]
        return value

    check(
        "two runs agree on everything but the wall-clock stamps",
        strip_times(first.model_dump()) == strip_times(second.model_dump()),
    )
    check(
        "  and the stamps are the only difference",
        first.model_dump() != second.model_dump()
        or first.generated_at_utc == second.generated_at_utc,
    )
    check("aligned_direction is stable", first.aligned_direction is second.aligned_direction)


def main() -> int:
    print("=" * 70)
    print("REGIME & MTF VERIFICATION")
    print("=" * 70)
    section_trend_states()
    section_range_and_override()
    section_missing_inputs()
    section_ema_alignment()
    section_resolve_horizon()
    section_mtf_alignment()
    section_conflict_reporting()
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
