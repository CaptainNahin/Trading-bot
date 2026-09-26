"""Regime-routed trade playbooks.

For most of its life the scanner had a single playbook: follow an established
multi-timeframe trend. When no trend was aligned it emitted
``NO_CLEAR_DIRECTION`` and rejected the symbol -- correct for a trend-follower,
wrong for a product that must stay actionable across regimes, since most
short-horizon time is *not* trending.

This module supplies the missing playbooks and a deterministic router that
picks one from the regime the classifier has *already* established. Nothing here
invents a direction: ``range_fade`` fires only at a confirmed range edge,
``breakout_arm`` is an ARMED plan (size 0) until price actually breaks, and
``directional_lean`` is the honest last resort for dead chop -- it states the
marginal tilt, labels it as *no edge*, and is size 0.

Stops, targets and reward:risk are NOT decided here; they stay with
``signal.derive_risk_levels`` off confirmed pivots, so the honesty guarantees
(real levels only, RR floor, decline when structure is missing) are untouched.
This module only chooses the *direction*, the *tier* and the *labels*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quantedge.contracts import ConvictionTier, MarketRegime, SignalDirection
from quantedge.services.risk import SIZE_FRACTION_BY_TIER

_RANGE_REGIMES = {MarketRegime.LOW_VOLATILITY_RANGE, MarketRegime.HIGH_VOLATILITY_RANGE}

# Fractions of range height that count as "at an edge". Between them the middle
# is no-man's-land and no fade is fabricated.
_UPPER_EDGE = 0.66
_LOWER_EDGE = 0.34

@dataclass(frozen=True)
class PlaybookSetup:
    """A non-trend directional read chosen by :func:`select_playbook`.

    ``quality_score`` is a playbook-specific heuristic in [0, 1] -- how clean the
    setup is for *its own* strategy, never a win probability. ``armed`` marks a
    conditional plan holding no live position yet (breakout / lean).
    """

    playbook: str
    direction: SignalDirection
    tier: ConvictionTier
    size_fraction: float
    quality_score: float
    rationale: str
    risk_note: str
    upgrade_condition: str = ""
    armed: bool = False
    supporting: list[str] = field(default_factory=list)
    contradicting: list[str] = field(default_factory=list)


def _range_position(structure, features) -> float | None:
    """Where price sits in the range: 0.0 = at support, 1.0 = at resistance.

    Prefers confirmed structural range position, falls back to Bollinger %b, and
    returns ``None`` when neither exists (then no fade is offered).
    """
    pos = getattr(structure, "range_position", None)
    if pos is not None:
        return max(0.0, min(1.0, float(pos)))
    pct_b = getattr(features, "bb_percent_b", None)
    if pct_b is not None:
        return max(0.0, min(1.0, float(pct_b)))
    return None

def _range_fade(structure, features, regime) -> PlaybookSetup | None:
    if structure.nearest_support is None or structure.nearest_resistance is None:
        return None
    pos = _range_position(structure, features)
    if pos is None:
        return None

    vol = "high-volatility" if regime == MarketRegime.HIGH_VOLATILITY_RANGE else "low-volatility"
    adx = features.adx_14
    # A weaker trend (lower ADX) makes a range fade more reliable; being right at
    # the edge makes the entry better. Both are honest, bounded heuristics.
    adx_factor = 1.0 if adx is None else max(0.0, min(1.0, (25.0 - adx) / 25.0))

    if pos >= _UPPER_EDGE:
        direction, edge, opp = SignalDirection.DOWN, structure.nearest_resistance, structure.nearest_support
        where, break_word = "top", "above"
        edge_factor = (pos - _UPPER_EDGE) / (1.0 - _UPPER_EDGE)
    elif pos <= _LOWER_EDGE:
        direction, edge, opp = SignalDirection.UP, structure.nearest_support, structure.nearest_resistance
        where, break_word = "bottom", "below"
        edge_factor = (_LOWER_EDGE - pos) / _LOWER_EDGE
    else:
        return None  # mid-range: no fade, let the router fall through

    quality = round(0.40 + 0.15 * adx_factor + 0.05 * max(0.0, min(1.0, edge_factor)), 4)
    tier = ConvictionTier.B
    return PlaybookSetup(
        playbook="range_fade",
        direction=direction,
        tier=tier,
        size_fraction=SIZE_FRACTION_BY_TIER[tier],
        quality_score=quality,
        rationale=(
            f"Range-fade: price is at the {where} of a {vol} range "
            f"({opp} - {edge}); fading back toward {opp}."
        ),
        risk_note=(
            "This is a mean-reversion fade inside a range, not a trend trade. If "
            f"price instead breaks {break_word} {edge}, the fade is wrong and the "
            "stop takes you out -- that is the risk. Sized to a third."
        ),
        upgrade_condition=(
            f"A confirmed close beyond {edge} invalidates the fade and flips the read to a breakout."
        ),
        supporting=[f"{vol} range: support {structure.nearest_support}, resistance {structure.nearest_resistance}"],
    )

def _breakout_arm(structure, features, regime) -> PlaybookSetup | None:
    direction = structure.breakout_direction
    if direction is None:
        if regime != MarketRegime.BREAKOUT and not structure.breakout_candidate:
            return None
        ema = features.ema_9 or features.ema_20
        if ema is None:
            return None
        direction = SignalDirection.UP if float(features.close) >= ema else SignalDirection.DOWN
    level = structure.nearest_resistance if direction == SignalDirection.UP else structure.nearest_support
    break_word = "above" if direction == SignalDirection.UP else "below"
    lvl_txt = str(level) if level is not None else f"the {break_word} edge"
    tier = ConvictionTier.ARMED
    return PlaybookSetup(
        playbook="breakout_arm",
        direction=direction,
        tier=tier,
        size_fraction=SIZE_FRACTION_BY_TIER[tier],
        quality_score=round(0.45 if structure.breakout_candidate else 0.35, 4),
        rationale=(
            f"Coiled / squeeze: a breakout is primed to the {direction.value.lower()} side. "
            "This is an ARMED plan, not a live position."
        ),
        risk_note=(
            "No position yet, so nothing is at risk right now. Place a stop-entry order "
            f"just {break_word} {lvl_txt}; it becomes a live trade only on a confirmed "
            "break. If price never breaks, you hold nothing."
        ),
        upgrade_condition=f"Fills to a live setup on a confirmed close {break_word} {lvl_txt}.",
        armed=True,
        supporting=["Bollinger squeeze / breakout candidate on the execution timeframe"],
    )

def _directional_lean(features) -> PlaybookSetup:
    """Honest last resort for dead chop: state the tilt, label it as no edge."""
    votes = 0
    signals: list[str] = []
    if features.rsi_14 is not None:
        votes += 1 if features.rsi_14 >= 50 else -1
        signals.append(f"RSI {round(features.rsi_14, 1)}")
    ema = features.ema_9 or features.ema_20
    if ema is not None:
        votes += 1 if float(features.close) >= ema else -1
        signals.append("price vs EMA")
    if features.macd_histogram is not None:
        votes += 1 if features.macd_histogram >= 0 else -1
        signals.append("MACD histogram")
    direction = SignalDirection.UP if votes >= 0 else SignalDirection.DOWN
    tier = ConvictionTier.ARMED
    return PlaybookSetup(
        playbook="directional_lean",
        direction=direction,
        tier=tier,
        size_fraction=SIZE_FRACTION_BY_TIER[tier],
        quality_score=0.2,
        rationale=(
            "No range edge, coil or trend to trade -- this is a directional LEAN only "
            f"({', '.join(signals) or 'marginal tilt'} leaning {direction.value.lower()}). "
            "There is no statistical edge here."
        ),
        risk_note=(
            "LOW / NO EDGE: this is close to a coin flip. Do not size a normal position "
            "on it. Wait for a level to break or for the tape to choose a regime."
        ),
        upgrade_condition=(
            "Becomes a real setup if a range forms (fade the edge) or a level breaks (armed breakout)."
        ),
        armed=True,
    )

def select_playbook(*, regime, structure, features) -> PlaybookSetup:
    """Choose a non-trend playbook from the established regime.

    Called only when no multi-timeframe trend direction is aligned (the trend
    path stays in the scanner). Always returns a :class:`PlaybookSetup`: a fade
    or armed breakout when structure supports one, otherwise an honest,
    explicitly-labelled directional lean. It never returns ``None`` -- with valid
    warmed-up features the right answer to "give me a read" is a labelled lean,
    not a refusal. Genuine data faults are caught upstream as INSUFFICIENT_DATA.
    """
    if regime in _RANGE_REGIMES or structure.structure == "RANGE":
        fade = _range_fade(structure, features, regime)
        if fade is not None:
            return fade
    if structure.breakout_candidate or regime == MarketRegime.BREAKOUT:
        arm = _breakout_arm(structure, features, regime)
        if arm is not None:
            return arm
    return _directional_lean(features)
