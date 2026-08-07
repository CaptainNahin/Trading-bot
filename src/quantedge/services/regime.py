"""Market regime classification: 9-state deterministic classifier.

The classifier runs after indicators and structure, combining both to label the
current regime. A regime is a broad characterization of market behaviour -- is
this a trending environment or a ranging one, is volatility high or low, is a
breakout in progress -- and it filters which strategies deserve consideration.

The nine regimes
----------------
**STRONG_UPTREND** / **STRONG_DOWNTREND**: ADX above ``adx_strong_trend`` (25)
and structure confirming the direction. A directional bias is appropriate here.

**WEAK_UPTREND** / **WEAK_DOWNTREND**: ADX between ``adx_weak_trend`` (18) and
``adx_strong_trend``, structure still directional. The trend exists but lacks
conviction; continuation is less certain than in a strong trend.

**LOW_VOLATILITY_RANGE**: ADX below ``adx_range_max`` (18), Bollinger Band
width below its 25th percentile, and structure classified as RANGE. Price is
compressed; a breakout may be building, but range strategies dominate until it
actually happens.

**HIGH_VOLATILITY_RANGE**: ADX below ``adx_range_max``, BB width above its 75th
percentile. Price is swinging without direction -- profitable for mean-reversion
when you catch the extremes, costly when a swing becomes a trend.

**BREAKOUT**: structure reports ``breakout_candidate`` as true and the breakout
cleared the nearest level by the ATR-scaled buffer. This is the moment: price
has left the prior range and not yet established a new trend. Fading it early
is how failed breakouts hurt; waiting too long misses the move.

**VOLATILITY_SHOCK**: current ATR exceeds ``atr_shock_multiplier`` (2.5) times
its own rolling mean. A news spike, a liquidation cascade, or a gap-open after
hours. Every strategy's assumptions are suspect here; the safest move is usually
to wait.

**UNCERTAIN**: None of the above conditions held clearly, or two contradictory
signals arrived at once. This is an honest "I don't know," and it is written
into ``contradictions`` so the caller understands why.

Heuristic score
---------------
A rule-based confidence in the label, 0 to 1. It is explicitly **not** a
calibrated probability -- it was not trained on data, and it has not been tested
on unseen outcomes to verify that "0.80" truly means "correct 80% of the time."
Renaming it to ``regime_probability`` is prohibited. It measures how cleanly the
regime's conditions were met: a strong trend with unanimous EMA alignment and no
structural contradictions scores near 1.0; a weak trend with mixed EMAs and
conflicting swings scores near 0.5.

Evidence and contradictions
----------------------------
Every factor that contributed to the label goes into ``supporting_evidence``,
and every factor that argued against it goes into ``contradictions``. When the
label is UNCERTAIN, contradictions is where the explanation lives. A report
listing three pieces of conflicting evidence is worth far more when diagnosing
than one that simply returns UNCERTAIN.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantedge.config import get_scanner_config
from quantedge.contracts import MarketRegime, RegimeReport

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantedge.contracts import FeatureSnapshot, StructureReport

# Below these sample counts the derived inputs are not reported at all: a
# percentile over 5 observations and a "mean" ATR over 3 bars are numbers
# without meaning, and the classifier is better told nothing than told those.
_MIN_PERCENTILE_SAMPLES = 20
_MIN_ATR_SAMPLES = 14

__all__ = [
    "REGIME_VERSION",
    "atr_rolling_mean",
    "bb_width_percentile",
    "classify_regime",
    "classify_regime_from_features",
    "ema_alignment",
]

REGIME_VERSION = "regime-1.0.0"


def ema_alignment(features: FeatureSnapshot) -> str | None:
    """Summarize EMA ordering as ``"bullish"``, ``"bearish"`` or ``"mixed"``.

    Returns ``None`` when any EMA is still inside its warm-up: an alignment
    computed from a partially-seeded EMA-200 would read as fact while resting on
    a value the series cannot yet support.
    """
    levels = (features.ema_9, features.ema_20, features.ema_50, features.ema_200)
    if any(v is None for v in levels):
        return None
    e9, e20, e50, e200 = levels
    assert e9 is not None and e20 is not None and e50 is not None and e200 is not None
    if e9 > e20 > e50 > e200:
        return "bullish"
    if e9 < e20 < e50 < e200:
        return "bearish"
    return "mixed"


def bb_width_percentile(history: Sequence[float | None], current: float | None) -> float | None:
    """Percentile rank of ``current`` within ``history``, 0 to 100.

    A raw Bollinger width is not comparable across instruments -- 0.004 is wide
    for EURUSD and flat for a small-cap. The percentile makes the comparison
    self-referential: wide *for this symbol, lately*.

    ``None`` when there is too little history to rank against; the caller then
    has no volatility percentile rather than a fabricated one.
    """
    if current is None:
        return None
    usable = [v for v in history if v is not None]
    if len(usable) < _MIN_PERCENTILE_SAMPLES:
        return None
    below = sum(1 for v in usable if v < current)
    return round(100.0 * below / len(usable), 2)


def atr_rolling_mean(history: Sequence[float | None]) -> float | None:
    """Mean of the available ATR history, or ``None`` if too little exists.

    The shock test compares current ATR against this, so a mean taken over three
    bars would call almost anything a shock.
    """
    usable = [v for v in history if v is not None and v > 0]
    if len(usable) < _MIN_ATR_SAMPLES:
        return None
    return sum(usable) / len(usable)


def _regime_config() -> dict[str, Any]:
    return get_scanner_config().get("regime", {})


def classify_regime(
    *,
    structure: StructureReport,
    adx: float | None,
    bb_width_percentile: float | None,
    atr: float | None,
    atr_mean: float | None,
    ema_alignment: str | None,
) -> RegimeReport:
    """Classify the current market regime from indicators and structure.

    Parameters
    ----------
    structure:
        The structure report from the same timeframe.
    adx:
        Current ADX(14). Measures trend strength, not direction.
    bb_width_percentile:
        Where the current Bollinger Band width sits in its recent distribution,
        0 to 100. Below 25 is compressed; above 75 is expanded.
    atr:
        Current ATR(14), absolute volatility.
    atr_mean:
        Rolling mean of ATR over the recent window, used to detect shocks.
    ema_alignment:
        A string summary of EMA ordering: "bullish" when 9>20>50>200, "bearish"
        when inverted, "mixed" otherwise. Produced by the indicator service.

    Returns
    -------
    A :class:`~quantedge.contracts.RegimeReport` with the classified regime,
    heuristic score, and the evidence that led to it.
    """
    cfg = _regime_config()
    evidence: list[str] = []
    contradictions: list[str] = []

    # Prioritize breakout and shock — they override everything else.
    if structure.breakout_candidate:
        direction = structure.breakout_direction
        evidence.append(f"breakout candidate: cleared the level in direction {direction}")
        score = 0.80
        if structure.failed_breakout:
            contradictions.append("a prior breakout already failed in this window")
            score = 0.55
        return RegimeReport(
            regime=MarketRegime.BREAKOUT,
            heuristic_score=score,
            supporting_evidence=evidence,
            contradictions=contradictions,
            version=REGIME_VERSION,
            symbol=structure.symbol,
            timeframe=structure.timeframe,
        )

    if atr is not None and atr_mean is not None and atr_mean > 0:
        shock_multiple = cfg.get("atr_shock_multiplier", 2.5)
        if atr > atr_mean * shock_multiple:
            evidence.append(f"ATR {atr:.4f} exceeds {shock_multiple:.1f}x its mean {atr_mean:.4f}")
            return RegimeReport(
                regime=MarketRegime.VOLATILITY_SHOCK,
                heuristic_score=0.85,
                supporting_evidence=evidence,
                contradictions=contradictions,
                version=REGIME_VERSION,
                symbol=structure.symbol,
                timeframe=structure.timeframe,
            )

    # Trend classification requires both ADX and structure agreement.
    strong_adx = cfg.get("adx_strong_trend", 25.0)
    weak_adx = cfg.get("adx_weak_trend", 18.0)
    range_adx = cfg.get("adx_range_max", 18.0)

    regime: MarketRegime = MarketRegime.UNCERTAIN
    score = 0.0

    if adx is not None and adx >= strong_adx:
        evidence.append(f"ADX {adx:.1f} above {strong_adx:.1f} (strong trend)")
        if structure.structure == "UPTREND":
            regime = MarketRegime.STRONG_UPTREND
            score = 0.75
            if ema_alignment == "bullish":
                evidence.append("EMA alignment is bullish")
                score = 0.90
            elif ema_alignment == "mixed":
                contradictions.append("EMAs are mixed despite uptrend structure")
                score = 0.65
        elif structure.structure == "DOWNTREND":
            regime = MarketRegime.STRONG_DOWNTREND
            score = 0.75
            if ema_alignment == "bearish":
                evidence.append("EMA alignment is bearish")
                score = 0.90
            elif ema_alignment == "mixed":
                contradictions.append("EMAs are mixed despite downtrend structure")
                score = 0.65
        else:
            contradictions.append(f"ADX shows strong trend but structure is {structure.structure}")
            regime = MarketRegime.UNCERTAIN
            score = 0.30

    elif adx is not None and weak_adx <= adx < strong_adx:
        evidence.append(f"ADX {adx:.1f} in weak-trend range [{weak_adx}, {strong_adx})")
        if structure.structure == "UPTREND":
            regime = MarketRegime.WEAK_UPTREND
            score = 0.60
            if ema_alignment == "bullish":
                evidence.append("EMA alignment supports the trend")
                score = 0.70
        elif structure.structure == "DOWNTREND":
            regime = MarketRegime.WEAK_DOWNTREND
            score = 0.60
            if ema_alignment == "bearish":
                evidence.append("EMA alignment supports the trend")
                score = 0.70
        else:
            contradictions.append(f"ADX suggests weak trend but structure is {structure.structure}")
            regime = MarketRegime.UNCERTAIN
            score = 0.35

    elif adx is not None and adx < range_adx:
        evidence.append(f"ADX {adx:.1f} below {range_adx:.1f} (ranging)")
        low_vol_pct = cfg.get("bb_width_low_vol_percentile", 25.0)
        high_vol_pct = cfg.get("bb_width_high_vol_percentile", 75.0)

        if bb_width_percentile is not None and bb_width_percentile < low_vol_pct:
            evidence.append(
                f"Bollinger Band width at {bb_width_percentile:.0f}th percentile (compressed)"
            )
            if structure.structure == "RANGE":
                regime = MarketRegime.LOW_VOLATILITY_RANGE
                score = 0.75
                evidence.append("structure confirms range")
            else:
                regime = MarketRegime.LOW_VOLATILITY_RANGE
                score = 0.55
                contradictions.append(
                    f"volatility is compressed but structure is {structure.structure}"
                )

        elif bb_width_percentile is not None and bb_width_percentile > high_vol_pct:
            evidence.append(
                f"Bollinger Band width at {bb_width_percentile:.0f}th percentile (expanded)"
            )
            regime = MarketRegime.HIGH_VOLATILITY_RANGE
            score = 0.70
            if structure.structure in ("UPTREND", "DOWNTREND"):
                contradictions.append(
                    f"high volatility swings but structure shows {structure.structure}"
                )
                score = 0.50

        else:
            # ADX says range, but BB width is middling and structure might disagree.
            regime = MarketRegime.UNCERTAIN
            score = 0.40
            contradictions.append(
                f"ADX is low but BB width is middling ({bb_width_percentile:.0f}th percentile) "
                f"and structure is {structure.structure}"
            )

    else:
        # ADX is None — cannot classify without a trend-strength measure.
        contradictions.append("ADX unavailable; trend strength cannot be assessed")
        regime = MarketRegime.UNCERTAIN
        score = 0.25

    # Final cross-check: if the regime is still UNCERTAIN and we have no evidence,
    # that is itself a finding worth stating.
    if regime == MarketRegime.UNCERTAIN and not evidence:
        contradictions.append("no single regime condition was clearly met")

    return RegimeReport(
        regime=regime,
        heuristic_score=score,
        supporting_evidence=evidence,
        contradictions=contradictions,
        version=REGIME_VERSION,
        symbol=structure.symbol,
        timeframe=structure.timeframe,
    )


def classify_regime_from_features(
    *,
    structure: StructureReport,
    features: FeatureSnapshot,
    bb_width_history: Sequence[float | None] = (),
    atr_history: Sequence[float | None] = (),
) -> RegimeReport:
    """Classify using a feature snapshot, deriving the aggregate inputs.

    The plain :func:`classify_regime` takes the four numbers it reasons about
    directly, which keeps it testable against hand-chosen values. This wrapper is
    what the scanner calls: it derives EMA alignment, the Bollinger width
    percentile and the ATR mean from a snapshot plus history.

    Passing no history is legitimate -- it yields ``None`` for the percentile and
    the ATR mean, and the classifier degrades to what it can still establish
    rather than inventing a distribution from one observation.
    """
    return classify_regime(
        structure=structure,
        adx=features.adx_14,
        bb_width_percentile=bb_width_percentile(bb_width_history, features.bb_width),
        atr=features.atr_14,
        atr_mean=atr_rolling_mean(atr_history),
        ema_alignment=ema_alignment(features),
    )
