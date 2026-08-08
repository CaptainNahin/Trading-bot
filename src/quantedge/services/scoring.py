"""Deterministic component scores for the scanner.

Every score here is a pure function of closed-candle features and structure.
Same bars in, same number out, on any machine -- which is what makes a signal
reproducible after the fact.

What these numbers are and are not
----------------------------------
A score is a **weighted sum of rule agreement**, normalised to 0..1. It is not
a probability, a win rate, or a confidence level, and nothing here has been
calibrated against unseen outcomes. ``config/scanner.yaml`` says the same thing
next to the weights: renaming these to accuracy/probability/win-rate is
prohibited. The honest reading of 0.72 is "most of the configured rules point
the same way", nothing more.

Absent inputs stay absent
-------------------------
When a feature is ``None`` -- warmup not satisfied, or the indicator undefined
on this many bars -- its rule is dropped from the weighted average and the
remaining weights are renormalised. Substituting a neutral 0.5 would silently
manufacture agreement out of missing data, and the caller could not tell the
difference. Each component reports which rules it could actually evaluate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from quantedge.contracts import SignalDirection

if TYPE_CHECKING:
    from quantedge.contracts import FeatureSnapshot, StructureReport

__all__ = [
    "ComponentScore",
    "composite_score",
    "momentum_score",
    "trend_score",
    "volatility_score",
]


class ComponentScore(NamedTuple):
    """A component score plus the rules behind it.

    ``rules_evaluated`` and ``rules_skipped`` are carried so a score can be
    explained rather than asserted: 0.5 from four agreeing rules and 0.5 from
    one lone rule are different claims about the market.
    """

    score: float
    rules_evaluated: list[str]
    rules_skipped: list[str]
    detail: dict[str, float]


def _blend(parts: dict[str, tuple[float | None, float]]) -> ComponentScore:
    """Weighted mean over rules that produced a value.

    ``parts`` maps rule name to ``(score_or_None, weight)``. Renormalises over
    present rules; returns 0.0 with everything listed as skipped when none are.
    """
    evaluated: list[str] = []
    skipped: list[str] = []
    detail: dict[str, float] = {}
    weighted = 0.0
    total_weight = 0.0

    for name, (value, weight) in parts.items():
        if value is None:
            skipped.append(name)
            continue
        clamped = max(0.0, min(1.0, float(value)))
        evaluated.append(name)
        detail[name] = round(clamped, 4)
        weighted += clamped * weight
        total_weight += weight

    if total_weight <= 0.0:
        return ComponentScore(0.0, [], skipped, {})
    return ComponentScore(round(weighted / total_weight, 4), evaluated, skipped, detail)


def _directional(value: float | None, direction: SignalDirection, scale: float) -> float | None:
    """Map a signed quantity to 0..1 by how well it backs ``direction``.

    ``scale`` is the magnitude treated as full agreement; beyond it the score
    saturates at 1.0 rather than growing without bound, so one extreme reading
    cannot dominate the blend.
    """
    if value is None:
        return None
    signed = value if direction == SignalDirection.UP else -value
    return max(0.0, min(1.0, 0.5 + 0.5 * (signed / scale))) if scale > 0 else None


def trend_score(
    features: FeatureSnapshot,
    structure: StructureReport,
    direction: SignalDirection,
    weights: dict[str, float],
) -> ComponentScore:
    """How strongly trend evidence supports ``direction``."""
    close = features.close
    up = direction == SignalDirection.UP

    ema_alignment: float | None = None
    emas = [features.ema_9, features.ema_20, features.ema_50, features.ema_200]
    present = [e for e in emas if e is not None]
    if close is not None and len(present) >= 2:
        # Fraction of available EMAs price sits on the correct side of.
        agree = sum(1 for e in present if (close > e) == up)
        ema_alignment = agree / len(present)

    slope = features.ema_20_slope if features.ema_20_slope is not None else features.ema_50_slope
    ema_slope = _directional(slope, direction, scale=0.004)

    adx_strength: float | None = None
    if features.adx_14 is not None:
        # ADX is directionless: 25+ is a conventional "trending" reading. Pair
        # it with DI dominance so a strong trend against us does not score high.
        strength = max(0.0, min(1.0, (features.adx_14 - 10.0) / 30.0))
        plus_di, minus_di = features.plus_di_14, features.minus_di_14
        if plus_di is not None and minus_di is not None:
            favoured = plus_di > minus_di if up else minus_di > plus_di
            adx_strength = strength if favoured else 1.0 - strength
        else:
            adx_strength = strength

    structure_hh_hl: float | None = None
    wanted = "UPTREND" if up else "DOWNTREND"
    opposed = "DOWNTREND" if up else "UPTREND"
    if structure.structure == wanted:
        # structure_confidence, when the engine supplies it, is a better signal
        # than the categorical label alone.
        structure_hh_hl = (
            structure.structure_confidence if structure.structure_confidence is not None else 0.85
        )
    elif structure.structure == opposed:
        structure_hh_hl = 0.0
    else:
        structure_hh_hl = 0.5

    return _blend(
        {
            "ema_alignment": (ema_alignment, float(weights.get("ema_alignment", 0.30))),
            "ema_slope": (ema_slope, float(weights.get("ema_slope", 0.20))),
            "adx_strength": (adx_strength, float(weights.get("adx_strength", 0.25))),
            "structure_hh_hl": (structure_hh_hl, float(weights.get("structure_hh_hl", 0.25))),
        }
    )


def momentum_score(
    features: FeatureSnapshot,
    direction: SignalDirection,
    weights: dict[str, float],
) -> ComponentScore:
    """How strongly momentum evidence supports ``direction``."""
    up = direction == SignalDirection.UP

    rsi_position: float | None = None
    if features.rsi_14 is not None:
        # Distance from 50 in the favoured direction, saturating at the
        # conventional 70/30 bands rather than at 100/0.
        offset = (features.rsi_14 - 50.0) / 20.0
        rsi_position = max(0.0, min(1.0, 0.5 + 0.5 * (offset if up else -offset)))

    macd_histogram: float | None = None
    if features.macd_histogram is not None and features.close:
        # Normalised by price so the scale is comparable across instruments.
        macd_histogram = _directional(
            float(features.macd_histogram) / float(features.close), direction, scale=0.002
        )

    roc = _directional(features.roc_10, direction, scale=2.0)

    body_ratio: float | None = None
    if features.body_ratio is not None:
        # Conviction of the last closed bar; direction comes from the wicks.
        wick_up, wick_dn = features.upper_wick_ratio, features.lower_wick_ratio
        if wick_up is not None and wick_dn is not None:
            favoured = wick_dn >= wick_up if up else wick_up >= wick_dn
            body_ratio = features.body_ratio if favoured else 1.0 - features.body_ratio
        else:
            body_ratio = features.body_ratio

    return _blend(
        {
            "rsi_position": (rsi_position, float(weights.get("rsi_position", 0.30))),
            "macd_histogram": (macd_histogram, float(weights.get("macd_histogram", 0.30))),
            "roc": (roc, float(weights.get("roc", 0.20))),
            "body_ratio": (body_ratio, float(weights.get("body_ratio", 0.20))),
        }
    )


def volatility_score(features: FeatureSnapshot, weights: dict[str, float]) -> ComponentScore:
    """Whether volatility is in a tradeable band.

    Directionless on purpose. Dead-flat tape gives a stop no room and offers no
    move to capture; a volatility shock invalidates levels faster than they can
    be acted on. Both ends score low, the middle scores high.
    """

    def band(value: float | None, low: float, high: float) -> float | None:
        if value is None:
            return None
        if value <= 0.0:
            return 0.0
        if value < low:
            return max(0.0, value / low)
        if value > high:
            return max(0.0, min(1.0, high / value))
        return 1.0

    atr_percentile = band(features.atr_percent, 0.10, 1.50)
    bollinger_width = band(features.bb_width, 0.30, 4.00)
    realized_vol = band(features.realized_volatility_20, 0.10, 3.00)

    return _blend(
        {
            "atr_percentile": (atr_percentile, float(weights.get("atr_percentile", 0.40))),
            "bollinger_width": (bollinger_width, float(weights.get("bollinger_width", 0.30))),
            "realized_vol": (realized_vol, float(weights.get("realized_vol", 0.30))),
        }
    )


def composite_score(
    *,
    trend: float,
    momentum: float,
    volatility: float,
    data_quality: float,
    evidence_agreement: float,
    weights: dict[str, float],
) -> float:
    """Blend components using ``weights.composite`` from scanner.yaml.

    Weights come from config rather than literals here so the scoring policy
    lives in one auditable place; a scan can be reproduced from the config that
    produced it.
    """
    parts: dict[str, tuple[float | None, float]] = {
        "trend_score": (trend, float(weights.get("trend_score", 0.35))),
        "momentum_score": (momentum, float(weights.get("momentum_score", 0.30))),
        "volatility_score": (volatility, float(weights.get("volatility_score", 0.15))),
        "data_quality_score": (data_quality, float(weights.get("data_quality_score", 0.10))),
        "evidence_agreement_score": (
            evidence_agreement,
            float(weights.get("evidence_agreement_score", 0.10)),
        ),
    }
    return _blend(parts).score
