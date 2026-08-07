"""The data-quality engine: fifteen deterministic checks over market data.

A ``FAIL`` here is a hard gate. No candidate, no regime call and no LLM analysis
may be released on top of failing data -- callers check
:attr:`DataQualityReport.is_blocking` and stop.

The distinction that runs through every check is between *bad data* and
*unusual markets*. A crossed book, a negative price, a bar out of order or a
series stitched from two providers are faults: the data is wrong, and anything
computed from it is wrong. A violent price spike, a flat hour or a session with
no volume are not faults -- they are what the market did, and a system that
refuses to look at them will be blind exactly when it matters. The first group
fails; the second degrades and is written into ``warnings`` so the reader knows
the conditions.

Two consequences worth stating outright:

* **Zero volume never fails.** Most spot-forex feeds report no volume at all,
  and several report a tick count instead. Failing on it would disable forex
  entirely on a cosmetic difference.
* **Weekend gaps in forex are not gaps.** The market was shut. Counting those
  bars as missing would mark every Monday morning as degraded.

The score is a weighted mean of the individual check scores, not a tally of
passes. It is called ``quality_score`` and it measures the data -- it is not a
confidence in any trade and must never be presented as one.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from itertools import pairwise
from typing import TYPE_CHECKING

from quantedge.contracts import (
    AssetClass,
    CandleSeries,
    DataQualityReport,
    QualityStatus,
    timeframe_seconds,
    utc_now,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from quantedge.contracts import Candle, Quote, Timeframe

__all__ = [
    "CHECK_NAMES",
    "QUALITY_VERSION",
    "assess_quality",
]

QUALITY_VERSION = "quality-1.0.0"

#: Every check, in the order it runs. Reported in ``checks_run`` so a caller can
#: tell "this check passed" from "this check never ran".
CHECK_NAMES = (
    "series_present",
    "bar_count_sufficient",
    "single_provider",
    "symbol_timeframe_consistent",
    "chronological_order",
    "ohlc_internally_consistent",
    "positive_prices",
    "closed_candles_only",
    "no_missing_bars",
    "freshness",
    "no_price_spikes",
    "no_flatline",
    "volume_sanity",
    "quote_freshness",
    "quote_spread_sane",
)

# Weights for the score. Faults that invalidate downstream maths carry more than
# advisory observations; the weights are normalized, so only ratios matter.
_WEIGHTS: dict[str, float] = {
    "series_present": 3.0,
    "bar_count_sufficient": 2.0,
    "single_provider": 2.0,
    "symbol_timeframe_consistent": 2.0,
    "chronological_order": 3.0,
    "ohlc_internally_consistent": 3.0,
    "positive_prices": 3.0,
    "closed_candles_only": 3.0,
    "no_missing_bars": 2.0,
    "freshness": 2.0,
    "no_price_spikes": 1.0,
    "no_flatline": 1.0,
    "volume_sanity": 0.5,
    "quote_freshness": 1.0,
    "quote_spread_sane": 1.5,
}

# Tolerances. Deliberately conservative: a check that fires constantly gets
# ignored, and an ignored quality gate is worse than none.
_MAX_MISSING_RATIO_DEGRADED = 0.02  # up to 2% of bars absent -> warn
_MAX_MISSING_RATIO_FAIL = 0.10  # beyond 10% the series is not the series asked for
_SPIKE_MEDIAN_MULTIPLE = 12.0  # bar range vs the median bar range
_FLATLINE_BARS = 10  # identical closes in a row
_STALENESS_DEGRADED_MULTIPLE = 1.5  # x one bar duration past the expected close
_STALENESS_FAIL_MULTIPLE = 2.5  # matches gates.max_candle_staleness_multiplier
_QUOTE_STALE_DEGRADED_MS = 15_000  # matches gates.max_quote_staleness_ms
_QUOTE_STALE_FAIL_MS = 120_000
_MAX_SPREAD_RATIO = 0.05  # 5% of mid is not a tradeable quote


@dataclass(slots=True)
class _Outcome:
    """One check's verdict.

    ``score`` is separate from ``status`` because a degraded check is not
    uniformly half-bad: three missing bars in a thousand should cost less than
    thirty.
    """

    name: str
    status: QualityStatus
    score: float
    message: str | None = None


def _ok(name: str) -> _Outcome:
    return _Outcome(name, QualityStatus.PASS, 1.0)


def _warn(name: str, message: str, score: float = 0.5) -> _Outcome:
    return _Outcome(name, QualityStatus.DEGRADED, score, message)


def _fail(name: str, message: str) -> _Outcome:
    return _Outcome(name, QualityStatus.FAIL, 0.0, message)


# --------------------------------------------------------------------------- #
# Individual checks                                                            #
# --------------------------------------------------------------------------- #


def _check_series_present(bars: Sequence[Candle]) -> _Outcome:
    if not bars:
        return _fail("series_present", "no candles were returned")
    return _ok("series_present")


def _check_bar_count(bars: Sequence[Candle], min_bars: int) -> _Outcome:
    name = "bar_count_sufficient"
    if min_bars <= 0:
        return _ok(name)
    if len(bars) >= min_bars:
        return _ok(name)
    # Below half the requirement nothing meaningful can be computed; between
    # half and full, short-window indicators still work and long ones will
    # report themselves missing.
    if len(bars) < min_bars // 2:
        return _fail(name, f"{len(bars)} bars available, {min_bars} required")
    return _warn(
        name,
        f"{len(bars)} bars available, {min_bars} requested; "
        "long-warm-up indicators will be unavailable",
        score=len(bars) / min_bars,
    )


def _check_single_provider(bars: Sequence[Candle], series_provider: str | None) -> _Outcome:
    name = "single_provider"
    providers = {c.provider for c in bars}
    if len(providers) > 1:
        # Two vendors' bars in one series do not share alignment or a close
        # convention; an indicator over the mixture describes neither feed.
        return _fail(name, f"series mixes providers: {sorted(providers)}")
    if series_provider and providers and series_provider not in providers:
        return _fail(
            name,
            f"series claims provider '{series_provider}' but bars are from {sorted(providers)}",
        )
    return _ok(name)


def _check_symbol_timeframe(bars: Sequence[Candle]) -> _Outcome:
    name = "symbol_timeframe_consistent"
    symbols = {c.symbol for c in bars}
    timeframes = {c.timeframe for c in bars}
    if len(symbols) > 1:
        return _fail(name, f"series contains multiple symbols: {sorted(symbols)}")
    if len(timeframes) > 1:
        return _fail(
            name, f"series contains multiple timeframes: {sorted(t.value for t in timeframes)}"
        )
    return _ok(name)


def _check_chronological(bars: Sequence[Candle]) -> _Outcome:
    name = "chronological_order"
    problems: list[str] = []
    for previous, current in pairwise(bars):
        if current.open_time_utc < previous.open_time_utc:
            problems.append(
                f"{current.open_time_utc.isoformat()} follows {previous.open_time_utc.isoformat()}"
            )
        elif current.open_time_utc == previous.open_time_utc:
            problems.append(f"duplicate open time {current.open_time_utc.isoformat()}")
    if problems:
        return _fail(name, f"{len(problems)} ordering fault(s): {problems[0]}")
    return _ok(name)


def _check_ohlc(bars: Sequence[Candle]) -> _Outcome:
    name = "ohlc_internally_consistent"
    bad: list[str] = []
    for candle in bars:
        top = max(candle.open, candle.close)
        bottom = min(candle.open, candle.close)
        if candle.high < top or candle.low > bottom or candle.high < candle.low:
            bad.append(candle.open_time_utc.isoformat())
    if bad:
        return _fail(
            name,
            f"{len(bad)} bar(s) where OHLC cannot describe a real bar, first at {bad[0]}",
        )
    return _ok(name)


def _check_positive_prices(bars: Sequence[Candle]) -> _Outcome:
    name = "positive_prices"
    zero = [c.open_time_utc.isoformat() for c in bars if min(c.open, c.high, c.low, c.close) <= 0]
    if zero:
        return _fail(name, f"{len(zero)} bar(s) with a non-positive price, first at {zero[0]}")
    return _ok(name)


def _check_closed_only(bars: Sequence[Candle]) -> _Outcome:
    name = "closed_candles_only"
    forming = [c for c in bars if not c.is_closed]
    if forming:
        # Rule 9. The caller was handed a set to score; a forming bar in it means
        # the forming bar is being treated as history somewhere upstream.
        return _fail(
            name,
            f"{len(forming)} forming candle(s) in the scored set; "
            "forming bars must be excluded before analysis",
        )
    return _ok(name)


def _is_weekend_gap(previous: Candle, current: Candle, asset_class: AssetClass) -> bool:
    """Does the gap between two bars span a weekend market closure?

    Crypto is exempt from the exemption: it trades continuously, so a Sunday gap
    there is a real gap.
    """
    if asset_class is AssetClass.CRYPTO:
        return False
    # Walk the covered days; any Saturday or Sunday explains the absence.
    start = previous.close_time_utc
    end = current.open_time_utc
    if end <= start:
        return False
    probe = start
    step = min(end - start, timedelta(hours=6))
    while probe < end:
        if probe.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return True
        probe += step
    return end.weekday() >= 5 or start.weekday() >= 5


def _check_missing_bars(bars: Sequence[Candle], asset_class: AssetClass) -> _Outcome:
    name = "no_missing_bars"
    if len(bars) < 2:
        return _ok(name)
    expected = timeframe_seconds(bars[0].timeframe)
    missing = 0
    first_gap: str | None = None
    for previous, current in pairwise(bars):
        delta = (current.open_time_utc - previous.open_time_utc).total_seconds()
        if delta <= expected:
            continue
        absent = round(delta / expected) - 1
        if absent <= 0 or _is_weekend_gap(previous, current, asset_class):
            continue
        missing += absent
        if first_gap is None:
            first_gap = f"{absent} bar(s) missing after {previous.open_time_utc.isoformat()}"
    if missing == 0:
        return _ok(name)
    ratio = missing / (len(bars) + missing)
    if ratio > _MAX_MISSING_RATIO_FAIL:
        return _fail(name, f"{missing} bars missing ({ratio:.1%} of the window); {first_gap}")
    if ratio > _MAX_MISSING_RATIO_DEGRADED:
        return _warn(name, f"{missing} bars missing ({ratio:.1%}); {first_gap}", score=1.0 - ratio)
    return _warn(name, f"{missing} bar(s) missing ({ratio:.2%}); {first_gap}", score=1.0 - ratio)


def _check_freshness(bars: Sequence[Candle], now: datetime) -> tuple[_Outcome, int]:
    """Age of the newest bar against how old it is entitled to be.

    A just-closed 15m bar is up to 15 minutes old and perfectly fresh; the same
    age on a 1m series means fourteen bars never arrived. So staleness is
    measured in bar durations, not wall-clock seconds.
    """
    name = "freshness"
    if not bars:
        return _fail(name, "no bars to age"), 0
    last = bars[-1]
    age_seconds = (now - last.close_time_utc).total_seconds()
    freshness_ms = max(0, int(age_seconds * 1000))
    bar_seconds = timeframe_seconds(last.timeframe)
    multiples = age_seconds / bar_seconds if bar_seconds else 0.0

    if age_seconds < 0:
        # A close time in the future is a clock or timezone fault, not fresh data.
        return (
            _fail(name, f"newest bar closes {abs(age_seconds):.0f}s in the future"),
            0,
        )
    if multiples > _STALENESS_FAIL_MULTIPLE:
        return (
            _fail(
                name,
                f"newest bar is {age_seconds:.0f}s old ({multiples:.1f} bar durations); "
                f"the feed is behind",
            ),
            freshness_ms,
        )
    if multiples > _STALENESS_DEGRADED_MULTIPLE:
        return (
            _warn(
                name,
                f"newest bar is {age_seconds:.0f}s old ({multiples:.1f} bar durations)",
                score=0.6,
            ),
            freshness_ms,
        )
    return _ok(name), freshness_ms


def _check_spikes(bars: Sequence[Candle]) -> _Outcome:
    """Outlier bar ranges.

    Degraded, never failed. A 12x-median bar is usually a news release or a
    genuine liquidation cascade -- real data, and the most informative bar on
    the chart. Occasionally it is a bad tick. The engine cannot tell them apart,
    so it flags rather than discards.
    """
    name = "no_price_spikes"
    if len(bars) < 20:
        return _ok(name)
    ranges = [float(c.high - c.low) for c in bars]
    median = statistics.median(r for r in ranges if r > 0) if any(r > 0 for r in ranges) else 0.0
    if median <= 0:
        return _ok(name)
    spikes = [
        bars[i].open_time_utc.isoformat()
        for i, r in enumerate(ranges)
        if r > median * _SPIKE_MEDIAN_MULTIPLE
    ]
    if spikes:
        return _warn(
            name,
            f"{len(spikes)} bar(s) exceed {_SPIKE_MEDIAN_MULTIPLE:.0f}x the median range "
            f"(first at {spikes[0]}); may be real volatility or a bad tick",
            score=0.7,
        )
    return _ok(name)


def _check_flatline(bars: Sequence[Candle]) -> _Outcome:
    """A run of identical closes: often a stalled feed repeating its last value."""
    name = "no_flatline"
    if len(bars) < _FLATLINE_BARS:
        return _ok(name)
    longest = 1
    run = 1
    for previous, current in pairwise(bars):
        if current.close == previous.close and current.high == current.low:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    if longest >= _FLATLINE_BARS:
        return _warn(
            name,
            f"{longest} consecutive bars with an identical close and no range; "
            "the feed may be stalled or the market shut",
            score=0.4,
        )
    return _ok(name)


def _check_volume(bars: Sequence[Candle]) -> _Outcome:
    """Negative volume is impossible; absent volume is normal on spot forex."""
    name = "volume_sanity"
    volumes = [c.volume for c in bars if c.volume is not None]
    if not volumes:
        return _warn(name, "no volume reported by this provider", score=0.8)
    negative = [v for v in volumes if v < 0]
    if negative:
        return _fail(name, f"{len(negative)} bar(s) report negative volume")
    if all(v == 0 for v in volumes):
        return _warn(name, "every bar reports zero volume", score=0.8)
    return _ok(name)


def _check_quote_freshness(quote: Quote | None, now: datetime) -> _Outcome:
    name = "quote_freshness"
    if quote is None:
        return _ok(name)
    age_ms = (now - quote.provider_time_utc).total_seconds() * 1000
    if age_ms < -1000:
        return _fail(name, f"quote timestamp is {abs(age_ms) / 1000:.1f}s in the future")
    if age_ms > _QUOTE_STALE_FAIL_MS:
        return _fail(name, f"quote is {age_ms / 1000:.0f}s old")
    if age_ms > _QUOTE_STALE_DEGRADED_MS:
        return _warn(name, f"quote is {age_ms / 1000:.1f}s old", score=0.5)
    return _ok(name)


def _check_spread(quote: Quote | None) -> _Outcome:
    name = "quote_spread_sane"
    if quote is None or quote.bid is None or quote.ask is None:
        return _ok(name)
    if quote.bid > quote.ask:
        # A crossed book is arbitrage or corruption, and it is always corruption.
        return _fail(name, f"crossed book: bid {quote.bid} above ask {quote.ask}")
    mid = quote.mid if quote.mid is not None else (quote.bid + quote.ask) / Decimal(2)
    if mid <= 0:
        return _fail(name, "quote mid price is not positive")
    ratio = float((quote.ask - quote.bid) / mid)
    if ratio > _MAX_SPREAD_RATIO:
        return _warn(
            name,
            f"spread is {ratio:.2%} of mid; execution assumptions will not hold",
            score=0.3,
        )
    return _ok(name)


# --------------------------------------------------------------------------- #
# Engine                                                                       #
# --------------------------------------------------------------------------- #


def assess_quality(
    series: CandleSeries | Sequence[Candle],
    *,
    quote: Quote | None = None,
    min_bars: int = 0,
    now: datetime | None = None,
) -> DataQualityReport:
    """Run all fifteen checks and return a verdict.

    Parameters
    ----------
    series:
        The bars to score. A :class:`~quantedge.contracts.CandleSeries` is
        scored on its **closed** bars only; a raw sequence is scored as given,
        and a forming bar in it fails ``closed_candles_only`` rather than being
        quietly removed.
    quote:
        Optional live quote. Its two checks pass trivially when absent -- the
        engine reports on what it was given, and a missing quote is the caller's
        decision, not a data fault.
    min_bars:
        Bars the caller needs for its analysis. ``0`` disables the check.

    Notes
    -----
    Every check always runs; there is no early return on the first failure. A
    report listing all four things wrong with a feed is worth far more when
    diagnosing than one that stops at the first.
    """
    now = now or utc_now()

    # isinstance rather than hasattr(series, "closed"): it narrows the union for
    # the type checker, and it does not mistake an unrelated sequence that
    # happens to expose a `closed` attribute for a CandleSeries.
    if isinstance(series, CandleSeries):
        bars = list(series.closed)
        series_provider: str | None = series.provider
        asset_class = series.asset_class
        symbol: str | None = series.symbol
        timeframe: Timeframe | None = series.timeframe
    else:
        bars = list(series)
        series_provider = bars[0].provider if bars else None
        asset_class = bars[0].asset_class if bars else AssetClass.CRYPTO
        symbol = bars[0].symbol if bars else None
        timeframe = bars[0].timeframe if bars else None

    freshness_ms = 0
    outcomes: list[_Outcome] = [
        _check_series_present(bars),
        _check_bar_count(bars, min_bars),
    ]

    if bars:
        outcomes.append(_check_single_provider(bars, series_provider))
        outcomes.append(_check_symbol_timeframe(bars))
        outcomes.append(_check_chronological(bars))
        outcomes.append(_check_ohlc(bars))
        outcomes.append(_check_positive_prices(bars))
        outcomes.append(_check_closed_only(bars))
        outcomes.append(_check_missing_bars(bars, asset_class))
        freshness_outcome, freshness_ms = _check_freshness(bars, now)
        outcomes.append(freshness_outcome)
        outcomes.append(_check_spikes(bars))
        outcomes.append(_check_flatline(bars))
        outcomes.append(_check_volume(bars))
    else:
        # With no bars the candle checks have no subject. They are recorded as
        # failed rather than skipped: reporting "passed" for a check that never
        # examined anything is the kind of false assurance this engine exists to
        # prevent.
        for check_name in CHECK_NAMES[2:13]:
            outcomes.append(_fail(check_name, "not evaluated: no candles"))

    outcomes.append(_check_quote_freshness(quote, now))
    outcomes.append(_check_spread(quote))

    warnings = [o.message for o in outcomes if o.status is QualityStatus.DEGRADED and o.message]
    blocking = [o.message for o in outcomes if o.status is QualityStatus.FAIL and o.message]

    if blocking:
        status = QualityStatus.FAIL
    elif warnings:
        status = QualityStatus.DEGRADED
    else:
        status = QualityStatus.PASS

    total_weight = sum(_WEIGHTS[o.name] for o in outcomes)
    weighted = sum(_WEIGHTS[o.name] * max(0.0, min(1.0, o.score)) for o in outcomes)
    score = weighted / total_weight if total_weight else 0.0

    return DataQualityReport(
        status=status,
        quality_score=round(score, 4),
        freshness_ms=freshness_ms,
        provider=series_provider or "unknown",
        symbol=symbol,
        timeframe=timeframe,
        candles_checked=len(bars),
        closed_candles=sum(1 for c in bars if c.is_closed),
        warnings=warnings,
        blocking_reasons=blocking,
        checks_run=[o.name for o in outcomes],
        checked_at_utc=now,
    )
