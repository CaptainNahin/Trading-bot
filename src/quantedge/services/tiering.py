"""Deterministic conviction tiering.

The scanner used to make a single binary decision at the agreement gate: either
the weighted multi-timeframe agreement cleared 0.50 (signal) or it did not
(NO_TRADE). That threw away every real-but-partial read -- most often a lone
execution-timeframe vote worth 0.25 of the stack -- and returned nothing.

This module keeps the *numbers* exactly as the engine measured them and instead
maps them onto an actionability tier. A 0.25-agreement setup is not promoted to
look like a 0.75 one; it is labelled a low-conviction B scalp, sized to a third,
and told plainly what would raise it. STAND_ASIDE is still a first-class outcome
for genuine no-edge, timeframe conflict, or missing data -- the tier ladder
lowers the *conviction* bar, never the honesty bar.

The classifier is pure: same inputs, same tier, no I/O, no clock, no provider.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantedge.contracts import ConvictionTier
from quantedge.services.risk import position_size_fraction

__all__ = ["TierResult", "TierThresholds", "classify_tier"]


@dataclass(frozen=True, slots=True)
class TierThresholds:
    """Ladder cut-offs. Defaults match ``config/scanner.yaml``'s ``tiers`` block.

    A_PLUS and A require *both* broad agreement and a strong composite score. B
    only requires a real directional vote (agreement + participation past their
    floors) and a composite score above pure noise -- it is the scalp band.
    """

    a_plus_min_agreement: float = 0.75
    a_plus_min_score: float = 0.70
    a_min_agreement: float = 0.50
    a_min_score: float = 0.60
    b_min_agreement: float = 0.25
    b_min_participation: float = 0.25
    b_min_score: float = 0.35

    @classmethod
    def from_config(cls, cfg: dict | None) -> TierThresholds:
        """Build from the parsed ``tiers`` mapping; missing keys keep defaults."""
        if not cfg:
            return cls()
        a_plus = cfg.get("a_plus") or {}
        a = cfg.get("a") or {}
        b = cfg.get("b") or {}
        return cls(
            a_plus_min_agreement=float(a_plus.get("min_agreement", cls.a_plus_min_agreement)),
            a_plus_min_score=float(a_plus.get("min_score", cls.a_plus_min_score)),
            a_min_agreement=float(a.get("min_agreement", cls.a_min_agreement)),
            a_min_score=float(a.get("min_score", cls.a_min_score)),
            b_min_agreement=float(b.get("min_agreement", cls.b_min_agreement)),
            b_min_participation=float(b.get("min_participation", cls.b_min_participation)),
            b_min_score=float(b.get("min_score", cls.b_min_score)),
        )


@dataclass(frozen=True, slots=True)
class TierResult:
    """The tier a set of measurements earns, plus how to act on it.

    ``reason_code`` / ``reason`` are populated only for ``STAND_ASIDE`` and use
    the scanner's existing rejection vocabulary (``WEAK_EVIDENCE_AGREEMENT``,
    ``INSUFFICIENT_TIMEFRAME_PARTICIPATION``, ``LOW_HEURISTIC_SCORE``) so the
    taxonomy and the INSUFFICIENT_DATA-vs-NO_TRADE mapping do not drift.
    """

    tier: ConvictionTier
    size_fraction: float
    rationale: str
    upgrade_condition: str
    reason_code: str | None = None
    reason: str | None = None

    @property
    def actionable(self) -> bool:
        """True when this tier represents an open, tradeable position."""
        return self.tier in (ConvictionTier.A_PLUS, ConvictionTier.A, ConvictionTier.B)


def classify_tier(
    *,
    alignment_score: float,
    participation: float,
    heuristic_score: float,
    has_conflict: bool,
    direction: str | None,
    thresholds: TierThresholds,
) -> TierResult:
    """Map measured agreement/participation/score onto a conviction tier.

    Inputs are the engine's own numbers (``mtf.alignment_score``,
    ``mtf.participation``, ``scoring.composite_score``); none is rescaled. When
    no tier is earned the result is ``STAND_ASIDE`` carrying the scanner reason
    code for the *most fundamental* unmet floor.
    """
    t = thresholds
    dir_txt = direction or "directional"

    # A genuine timeframe conflict is never a scalp: the views disagree, so
    # there is no shared direction to size down into. Honest NO_TRADE.
    if has_conflict:
        return TierResult(
            tier=ConvictionTier.STAND_ASIDE,
            size_fraction=0.0,
            rationale="Timeframes disagree on direction; no shared edge to trade.",
            upgrade_condition=(
                "needs the conflicting timeframes to resolve onto one side before any tier arms"
            ),
            reason_code="WEAK_EVIDENCE_AGREEMENT",
            reason="timeframes carry opposing directions; abstaining rather than picking a side",
        )

    if alignment_score >= t.a_plus_min_agreement and heuristic_score >= t.a_plus_min_score:
        return TierResult(
            tier=ConvictionTier.A_PLUS,
            size_fraction=position_size_fraction(ConvictionTier.A_PLUS),
            rationale=(
                f"Prime setup: {alignment_score:.0%} timeframe agreement with "
                f"{heuristic_score:.2f} composite evidence — full size."
            ),
            upgrade_condition="",
        )

    if alignment_score >= t.a_min_agreement and heuristic_score >= t.a_min_score:
        return TierResult(
            tier=ConvictionTier.A,
            size_fraction=position_size_fraction(ConvictionTier.A),
            rationale=(
                f"Strong setup: {alignment_score:.0%} agreement, {heuristic_score:.2f} "
                f"composite — reduced size."
            ),
            upgrade_condition=(
                f"needs agreement ≥{t.a_plus_min_agreement:.0%} and score "
                f"≥{t.a_plus_min_score:.2f} to reach A+ (full size)"
            ),
        )

    if (
        alignment_score >= t.b_min_agreement
        and participation >= t.b_min_participation
        and heuristic_score >= t.b_min_score
    ):
        return TierResult(
            tier=ConvictionTier.B,
            size_fraction=position_size_fraction(ConvictionTier.B),
            rationale=(
                f"Low-conviction {dir_txt} scalp: only {alignment_score:.0%} of the "
                f"timeframe weight carries the vote (score {heuristic_score:.2f}). "
                f"Small size — trade it as a scalp, not a swing."
            ),
            upgrade_condition=(
                f"needs a second timeframe to confirm (agreement ≥{t.a_min_agreement:.0%}) "
                f"and score ≥{t.a_min_score:.2f} to become an A setup"
            ),
        )

    # Nothing earned even a scalp: name the most fundamental floor that was
    # missed, in the same order the scanner gates once ran.
    if alignment_score < t.b_min_agreement:
        code, why = (
            "WEAK_EVIDENCE_AGREEMENT",
            f"agreement {alignment_score:.2f} is below the {t.b_min_agreement:.2f} scalp floor",
        )
    elif participation < t.b_min_participation:
        code, why = (
            "INSUFFICIENT_TIMEFRAME_PARTICIPATION",
            f"participation {participation:.2f} is below the {t.b_min_participation:.2f} floor",
        )
    else:
        code, why = (
            "LOW_HEURISTIC_SCORE",
            f"composite score {heuristic_score:.2f} is below the {t.b_min_score:.2f} scalp floor",
        )
    return TierResult(
        tier=ConvictionTier.STAND_ASIDE,
        size_fraction=0.0,
        rationale=f"No tradeable edge: {why}.",
        upgrade_condition=(
            f"needs agreement ≥{t.b_min_agreement:.0%}, participation "
            f"≥{t.b_min_participation:.0%} and score ≥{t.b_min_score:.2f} to arm a scalp"
        ),
        reason_code=code,
        reason=why,
    )
