"""Quality-engine verification: every check fired deliberately.

For each of the fifteen checks there are two fixtures -- one that trips it and
one that does not -- because a check that never fires and a check that always
fires are equally useless, and only testing the happy path cannot tell them
apart.

Two properties get their own sections because they encode judgement rather than
mechanics:

* **Section [16]** -- unusual markets must not fail. A violent spike, a flat
  session and a volumeless forex feed are all real data. If the engine failed
  them it would blind the system exactly when conditions are most informative.
* **Section [17]** -- ``FAIL`` is a hard gate. ``is_blocking`` must be true
  whenever any blocking reason exists, and the reasons must be readable enough
  to act on.

Run:  ./.venv/Scripts/python.exe -u scripts/verify_quality.py
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
    CandleSeries,
    QualityStatus,
    Quote,
    Timeframe,
)
from quantedge.services.quality import CHECK_NAMES, assess_quality

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


# A fixed "now" so staleness is a property of the fixture, not of when the
# script happens to run.
NOW = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
BAR = timedelta(minutes=5)


def candle(
    index: int,
    *,
    close: float = 100.0,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: float | None = 1000.0,
    is_closed: bool = True,
    provider: str = "fixture",
    symbol: str = "BTCUSDT",
    timeframe: Timeframe = Timeframe.M5,
    asset_class: AssetClass = AssetClass.CRYPTO,
    end: datetime = NOW,
) -> Candle:
    """One bar, indexed backwards from ``end`` so index 0 is the newest."""
    close_time = end - BAR * index
    o = close if open_ is None else open_
    return Candle(
        provider=provider,
        symbol=symbol,
        asset_class=asset_class,
        timeframe=timeframe,
        open_time_utc=close_time - BAR,
        close_time_utc=close_time,
        open=Decimal(str(o)),
        high=Decimal(str(max(o, close) if high is None else high)),
        low=Decimal(str(min(o, close) if low is None else low)),
        close=Decimal(str(close)),
        volume=None if volume is None else Decimal(str(volume)),
        is_closed=is_closed,
    )


def healthy_bars(count: int = 60, **kwargs: object) -> list[Candle]:
    """A clean ascending series ending exactly at ``NOW``, oldest first."""
    return [
        candle(count - 1 - i, close=100.0 + i * 0.1, **kwargs)  # type: ignore[arg-type]
        for i in range(count)
    ]


def ranged_bars(count: int = 60, *, span: float = 1.0) -> list[Candle]:
    """Bars with a real high-low range, so the median range is meaningful.

    ``healthy_bars`` produces high == low == close, which is fine for the checks
    that only read closes but useless for the spike check: with every range at
    zero, the median of the nonzero ranges is the outlier itself, and nothing can
    exceed 12x itself.
    """
    out: list[Candle] = []
    for i in range(count):
        close = 100.0 + i * 0.1
        out.append(
            candle(
                count - 1 - i,
                close=close,
                high=close + span / 2.0,
                low=close - span / 2.0,
                open_=close,
            )
        )
    return out


def good_quote(**kwargs: object) -> Quote:
    defaults: dict[str, object] = {
        "provider": "fixture",
        "symbol": "BTCUSDT",
        "asset_class": AssetClass.CRYPTO,
        "bid": Decimal("100.00"),
        "ask": Decimal("100.02"),
        "provider_time_utc": NOW,
    }
    defaults.update(kwargs)
    return Quote(**defaults)  # type: ignore[arg-type]


def status_of(bars: list[Candle], **kwargs: object) -> tuple[QualityStatus, list[str], list[str]]:
    report = assess_quality(bars, now=NOW, **kwargs)  # type: ignore[arg-type]
    return report.status, report.blocking_reasons, report.warnings


def mentions(messages: list[str], fragment: str) -> bool:
    return any(fragment in m for m in messages)


# --------------------------------------------------------------------------- #
# Sections                                                                     #
# --------------------------------------------------------------------------- #


def section_baseline() -> None:
    print("\n[1] A clean series passes with no warnings")
    report = assess_quality(healthy_bars(), quote=good_quote(), now=NOW)
    check("status is PASS", report.status is QualityStatus.PASS, report.status.value)
    check("no blocking reasons", not report.blocking_reasons, str(report.blocking_reasons))
    check("no warnings", not report.warnings, str(report.warnings))
    check("score is 1.0", report.quality_score == 1.0, str(report.quality_score))
    check("all 15 checks ran", len(report.checks_run) == 15, f"{len(report.checks_run)}")
    check("checks_run matches CHECK_NAMES", set(report.checks_run) == set(CHECK_NAMES))
    check("provider recorded", report.provider == "fixture", report.provider)
    check("candle count recorded", report.candles_checked == 60, str(report.candles_checked))
    check(
        "freshness is zero on a just-closed bar", report.freshness_ms == 0, str(report.freshness_ms)
    )
    print(f"    score={report.quality_score} freshness_ms={report.freshness_ms}")


def section_empty_and_count() -> None:
    print("\n[2] series_present / bar_count_sufficient")
    report = assess_quality([], now=NOW)
    check("empty series FAILs", report.status is QualityStatus.FAIL, report.status.value)
    check(
        "  reason names the absence",
        mentions(report.blocking_reasons, "no candles"),
        str(report.blocking_reasons[:1]),
    )
    check(
        "  every check is still reported", len(report.checks_run) == 15, f"{len(report.checks_run)}"
    )
    check(
        "  unevaluable checks are FAIL, never a silent pass",
        mentions(report.blocking_reasons, "not evaluated"),
    )
    # The score is not forced to zero: it is a continuous measure of the data,
    # and the FAIL status is the gate. 0.15 is the two quote checks and the
    # disabled bar-count check passing out of 30 total weight -- well under the
    # 0.60 gate, and honest about which checks had a subject to examine.
    check(
        "  score reflects only the checks that had a subject",
        report.quality_score == 0.15,
        str(report.quality_score),
    )
    check("  and is far below the 0.60 gate", report.quality_score < 0.60)

    status, blocking, _ = status_of(healthy_bars(10), min_bars=100)
    check("10 of 100 required bars FAILs", status is QualityStatus.FAIL, status.value)
    check("  reason states both numbers", mentions(blocking, "10 bars available, 100 required"))

    status, _, warnings = status_of(healthy_bars(70), min_bars=100)
    check("70 of 100 degrades rather than fails", status is QualityStatus.DEGRADED, status.value)
    check("  warning explains the consequence", mentions(warnings, "long-warm-up"))

    status, _, _ = status_of(healthy_bars(60), min_bars=0)
    check("min_bars=0 disables the check", status is QualityStatus.PASS, status.value)


def section_provenance() -> None:
    print("\n[3] single_provider / symbol_timeframe_consistent")
    mixed = healthy_bars(30)
    mixed[10] = mixed[10].model_copy(update={"provider": "other_vendor"})
    status, blocking, _ = status_of(mixed)
    check("two providers in one series FAILs", status is QualityStatus.FAIL, status.value)
    check("  both providers named", mentions(blocking, "other_vendor"), str(blocking[:1]))

    mixed_symbol = healthy_bars(30)
    mixed_symbol[5] = mixed_symbol[5].model_copy(update={"symbol": "ETHUSDT"})
    status, blocking, _ = status_of(mixed_symbol)
    check("two symbols in one series FAILs", status is QualityStatus.FAIL, status.value)
    check("  both symbols named", mentions(blocking, "ETHUSDT"))

    mixed_tf = healthy_bars(30)
    mixed_tf[7] = mixed_tf[7].model_copy(update={"timeframe": Timeframe.M15})
    status, blocking, _ = status_of(mixed_tf)
    check("two timeframes in one series FAILs", status is QualityStatus.FAIL, status.value)
    check("  both timeframes named", mentions(blocking, "15m"))


def section_ordering() -> None:
    print("\n[4] chronological_order")
    bars = healthy_bars(30)
    swapped = [*bars[:10], bars[11], bars[10], *bars[12:]]
    status, blocking, _ = status_of(swapped)
    check("out-of-order bars FAIL", status is QualityStatus.FAIL, status.value)
    check("  the fault is located", mentions(blocking, "follows"), str(blocking[:1]))

    duplicated = [*bars[:10], bars[10], *bars[10:]]
    status, blocking, _ = status_of(duplicated)
    check("a duplicated open time FAILs", status is QualityStatus.FAIL, status.value)
    check("  reported as a duplicate", mentions(blocking, "duplicate open time"))

    status, _, _ = status_of(bars)
    check("correctly ordered bars pass", status is QualityStatus.PASS, status.value)


def section_ohlc() -> None:
    print("\n[5] ohlc_internally_consistent / positive_prices")
    # The Candle contract rejects high < low outright, so the fault the engine
    # must catch is the subtler one: a high below the close.
    bars = healthy_bars(30)
    bad = bars[15].model_copy(update={"high": bars[15].close - Decimal("5")})
    status, blocking, _ = status_of([*bars[:15], bad, *bars[16:]])
    check("high below the close FAILs", status is QualityStatus.FAIL, status.value)
    check("  described as an impossible bar", mentions(blocking, "cannot describe a real bar"))

    low_bad = bars[20].model_copy(update={"low": bars[20].close + Decimal("5")})
    status, _, _ = status_of([*bars[:20], low_bad, *bars[21:]])
    check("low above the close FAILs", status is QualityStatus.FAIL, status.value)

    zero = bars[8].model_copy(
        update={
            "open": Decimal("0"),
            "high": Decimal("0"),
            "low": Decimal("0"),
            "close": Decimal("0"),
        }
    )
    status, blocking, _ = status_of([*bars[:8], zero, *bars[9:]])
    check("a zero price FAILs", status is QualityStatus.FAIL, status.value)
    check("  reported as non-positive", mentions(blocking, "non-positive price"))


def section_rule_9() -> None:
    print("\n[6] RULE 9: closed_candles_only")
    bars = healthy_bars(30)
    forming = bars[-1].model_copy(update={"is_closed": False})
    status, blocking, _ = status_of([*bars[:-1], forming])
    check("a forming bar in the scored set FAILs", status is QualityStatus.FAIL, status.value)
    check("  the reason states the rule", mentions(blocking, "forming bars must be excluded"))

    # Through a CandleSeries the forming bar is excluded before scoring, so the
    # same data passes -- and closed_candles reflects what was actually scored.
    series = CandleSeries(
        provider="fixture",
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M5,
        candles=[*bars[:-1], forming],
        includes_forming_candle=True,
    )
    report = assess_quality(series, now=NOW)
    check(
        "CandleSeries scores closed bars only",
        report.status is QualityStatus.PASS,
        report.status.value,
    )
    check(
        "  the forming bar is not counted",
        report.candles_checked == 29,
        str(report.candles_checked),
    )
    check(
        "  closed_candles equals candles_checked",
        report.closed_candles == 29,
        str(report.closed_candles),
    )


def section_gaps() -> None:
    print("\n[7] no_missing_bars, and the weekend exemption")
    bars = healthy_bars(60)
    with_gap = [*bars[:30], *bars[33:]]  # three bars removed mid-series
    status, _, warnings = status_of(with_gap)
    check("a small gap degrades", status is QualityStatus.DEGRADED, status.value)
    check(
        "  the gap is counted and located",
        mentions(warnings, "3 bar(s) missing") or mentions(warnings, "3 bars missing"),
        str(warnings[:1]),
    )

    sparse = bars[::4]  # 75% of bars absent
    status, blocking, _ = status_of(sparse)
    check("a large gap FAILs", status is QualityStatus.FAIL, status.value)
    check("  the proportion is reported", mentions(blocking, "% of the window"), str(blocking[:1]))

    # Forex over a weekend: Friday 21:00 UTC to Sunday 22:00 UTC is a closure,
    # not missing data.
    friday_close = datetime(2026, 1, 2, 21, 0, tzinfo=UTC)  # 2026-01-02 is a Friday
    before = [
        candle(
            i,
            close=1.1 + i * 0.0001,
            symbol="EURUSD",
            asset_class=AssetClass.FOREX,
            volume=None,
            end=friday_close,
        )
        for i in range(20, 0, -1)
    ]
    sunday_open = datetime(2026, 1, 4, 22, 5, tzinfo=UTC)  # Sunday evening
    after = [
        candle(
            i,
            close=1.1 + i * 0.0001,
            symbol="EURUSD",
            asset_class=AssetClass.FOREX,
            volume=None,
            end=sunday_open,
        )
        for i in range(10, 0, -1)
    ]
    weekend_now = sunday_open
    report = assess_quality(before + after, now=weekend_now)
    check(
        "a forex weekend is not counted as missing bars",
        not mentions(report.warnings + report.blocking_reasons, "missing"),
        str([m for m in report.warnings if "missing" in m][:1]),
    )

    # The same shaped gap in crypto IS missing data: that market never closed.
    crypto_before = [candle(i, close=100.0, end=friday_close) for i in range(20, 0, -1)]
    crypto_after = [candle(i, close=100.0, end=sunday_open) for i in range(10, 0, -1)]
    crypto = assess_quality(crypto_before + crypto_after, now=weekend_now)
    check(
        "the same gap in crypto IS missing data",
        mentions(crypto.warnings + crypto.blocking_reasons, "missing"),
        crypto.status.value,
    )


def section_freshness() -> None:
    print("\n[8] freshness is measured in bar durations, not seconds")
    bars = healthy_bars(30)

    report = assess_quality(bars, now=NOW + timedelta(minutes=4))
    check(
        "4 minutes into a 5m bar is fresh", report.status is QualityStatus.PASS, report.status.value
    )
    check("  freshness_ms is reported", report.freshness_ms == 240_000, str(report.freshness_ms))

    report = assess_quality(bars, now=NOW + timedelta(minutes=10))
    check(
        "2 bar durations late degrades",
        report.status is QualityStatus.DEGRADED,
        report.status.value,
    )

    report = assess_quality(bars, now=NOW + timedelta(minutes=30))
    check("6 bar durations late FAILs", report.status is QualityStatus.FAIL, report.status.value)
    check(
        "  the reason says the feed is behind", mentions(report.blocking_reasons, "feed is behind")
    )

    # The identical wall-clock age on a 15m series is still fresh.
    m15 = [candle(29 - i, close=100.0 + i, timeframe=Timeframe.M15, end=NOW) for i in range(30)]
    report = assess_quality(m15, now=NOW + timedelta(minutes=10))
    check(
        "the same 10-minute age passes on a 15m series",
        report.status is QualityStatus.PASS,
        report.status.value,
    )

    # A bar closing in the future is a clock fault.
    report = assess_quality(bars, now=NOW - timedelta(minutes=30))
    check(
        "a bar closing in the future FAILs",
        report.status is QualityStatus.FAIL,
        report.status.value,
    )
    check("  named as a future close", mentions(report.blocking_reasons, "in the future"))
    check("  freshness_ms is never negative", report.freshness_ms >= 0, str(report.freshness_ms))


def section_quote_checks() -> None:
    print("\n[9] quote_freshness / quote_spread_sane")
    bars = healthy_bars(30)

    report = assess_quality(bars, quote=good_quote(), now=NOW)
    check("a fresh, tight quote passes", report.status is QualityStatus.PASS, report.status.value)

    stale = good_quote(provider_time_utc=NOW - timedelta(seconds=30))
    report = assess_quality(bars, quote=stale, now=NOW)
    check("a 30s-old quote degrades", report.status is QualityStatus.DEGRADED, report.status.value)

    ancient = good_quote(provider_time_utc=NOW - timedelta(minutes=5))
    report = assess_quality(bars, quote=ancient, now=NOW)
    check("a 5-minute-old quote FAILs", report.status is QualityStatus.FAIL, report.status.value)

    # A crossed book is rejected by the Quote contract at construction, so to
    # test the engine's defence-in-depth we bypass validation.
    crossed = Quote.model_construct(
        provider="fixture",
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        bid=Decimal("100.10"),
        ask=Decimal("100.00"),
        provider_time_utc=NOW,
    )
    report = assess_quality(bars, quote=crossed, now=NOW)
    check("a crossed book FAILs", report.status is QualityStatus.FAIL, report.status.value)
    check("  named as crossed", mentions(report.blocking_reasons, "crossed book"))

    # And the contract itself rejects a crossed book at construction.
    try:
        Quote(
            provider="fixture",
            symbol="BTCUSDT",
            asset_class=AssetClass.CRYPTO,
            bid=Decimal("100.10"),
            ask=Decimal("100.00"),
            provider_time_utc=NOW,
        )
    except ValueError as exc:
        check("the Quote contract rejects a crossed book", "crossed market" in str(exc), str(exc))
    else:
        check("the Quote contract rejects a crossed book", False, "no exception raised")

    wide = good_quote(bid=Decimal("100.00"), ask=Decimal("110.00"), mid=Decimal("105.00"))
    report = assess_quality(bars, quote=wide, now=NOW)
    check("a 9.5% spread degrades", report.status is QualityStatus.DEGRADED, report.status.value)
    check("  the consequence is stated", mentions(report.warnings, "execution assumptions"))

    report = assess_quality(bars, quote=None, now=NOW)
    check(
        "no quote is not a data fault",
        report.status is QualityStatus.PASS,
        report.status.value,
    )


def section_unusual_but_real() -> None:
    print("\n[10] Unusual markets DEGRADE; they never FAIL")
    bars = ranged_bars(60, span=1.0)

    # Range 50 against a median range of 1.0 -- comfortably past the 12x line.
    spiked = bars[30].model_copy(
        update={"high": bars[30].close + Decimal("25"), "low": bars[30].close - Decimal("25")}
    )
    report = assess_quality([*bars[:30], spiked, *bars[31:]], now=NOW)
    check(
        "a 50x-range bar does not FAIL",
        report.status is not QualityStatus.FAIL,
        report.status.value,
    )
    check("  it is flagged", mentions(report.warnings, "median range"), str(report.warnings[:1]))
    check(
        "  and the ambiguity is admitted",
        mentions(report.warnings, "real volatility or a bad tick"),
    )

    flat = [candle(29 - i, close=100.0, high=100.0, low=100.0, open_=100.0) for i in range(30)]
    report = assess_quality(flat, now=NOW)
    check(
        "a flatlined feed does not FAIL",
        report.status is not QualityStatus.FAIL,
        report.status.value,
    )
    check("  it is flagged as possibly stalled", mentions(report.warnings, "stalled"))

    novol = healthy_bars(30, volume=None)
    report = assess_quality(novol, now=NOW)
    check(
        "absent volume does not FAIL", report.status is not QualityStatus.FAIL, report.status.value
    )
    check("  it is noted", mentions(report.warnings, "no volume reported"))

    zerovol = healthy_bars(30, volume=0.0)
    report = assess_quality(zerovol, now=NOW)
    check("zero volume does not FAIL", report.status is not QualityStatus.FAIL, report.status.value)

    # Negative volume, by contrast, is impossible and must fail.
    negative = bars[5].model_copy(update={"volume": Decimal("-10")})
    report = assess_quality([*bars[:5], negative, *bars[6:]], now=NOW)
    check("negative volume FAILs", report.status is QualityStatus.FAIL, report.status.value)


def section_hard_gate() -> None:
    print("\n[11] FAIL is a hard gate and says why")
    bars = healthy_bars(30)
    broken = bars[10].model_copy(update={"high": bars[10].close - Decimal("5")})
    report = assess_quality([*bars[:10], broken, *bars[11:]], now=NOW)

    check("status is FAIL", report.status is QualityStatus.FAIL, report.status.value)
    check("is_blocking is True", report.is_blocking is True)
    check("blocking_reasons is non-empty", bool(report.blocking_reasons))
    # The score stays high -- one bad bar in thirty really is 90% good data --
    # and that is exactly why the score must not be the gate. A caller reading
    # quality_score alone would see 0.9 and proceed on a series containing an
    # impossible bar. is_blocking is the gate; the score is a description.
    check(
        "the score alone would not have caught this",
        report.quality_score > 0.60,
        str(report.quality_score),
    )
    check("but the report still blocks", report.is_blocking is True)

    clean = assess_quality(bars, quote=good_quote(), now=NOW)
    check("a PASS report is not blocking", clean.is_blocking is False)

    degraded = assess_quality(healthy_bars(70), min_bars=100, now=NOW)
    check("a DEGRADED report is not blocking", degraded.is_blocking is False, degraded.status.value)
    check(
        "DEGRADED still carries its warnings",
        bool(degraded.warnings),
        str(degraded.warnings[:1]),
    )

    # Several independent faults at once: all must be reported, not just the first.
    multi = healthy_bars(40)
    multi[5] = multi[5].model_copy(update={"provider": "elsewhere"})
    multi[9] = multi[9].model_copy(update={"high": multi[9].close - Decimal("3")})
    multi[12] = multi[12].model_copy(update={"is_closed": False})
    report = assess_quality(multi, now=NOW)
    check(
        "three distinct faults yield three reasons",
        len(report.blocking_reasons) >= 3,
        f"{len(report.blocking_reasons)} reasons",
    )
    print(f"    reasons: {report.blocking_reasons}")


def section_score_and_determinism() -> None:
    print("\n[12] Score behaviour and determinism")
    clean = assess_quality(healthy_bars(60), quote=good_quote(), now=NOW)
    degraded = assess_quality(healthy_bars(70), min_bars=100, now=NOW)
    bars = healthy_bars(30)
    failed = assess_quality(
        [
            *bars[:10],
            bars[10].model_copy(update={"high": bars[10].close - Decimal("5")}),
            *bars[11:],
        ],
        now=NOW,
    )
    check(
        "clean > degraded > failed",
        clean.quality_score > degraded.quality_score > failed.quality_score,
        f"{clean.quality_score} / {degraded.quality_score} / {failed.quality_score}",
    )
    check(
        "score stays within [0, 1]",
        all(0.0 <= r.quality_score <= 1.0 for r in (clean, degraded, failed)),
    )

    first = assess_quality(healthy_bars(60), quote=good_quote(), now=NOW)
    second = assess_quality(healthy_bars(60), quote=good_quote(), now=NOW)
    check("two runs are identical", first.model_dump() == second.model_dump())

    check(
        "checked_at_utc is the supplied clock, not wall time",
        first.checked_at_utc == NOW,
        first.checked_at_utc.isoformat(),
    )


def main() -> int:
    print("=" * 70)
    print("QUALITY ENGINE VERIFICATION -- each check fired on purpose")
    print("=" * 70)
    section_baseline()
    section_empty_and_count()
    section_provenance()
    section_ordering()
    section_ohlc()
    section_rule_9()
    section_gaps()
    section_freshness()
    section_quote_checks()
    section_unusual_but_real()
    section_hard_gate()
    section_score_and_determinism()

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
