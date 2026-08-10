"""Deterministic risk levels: stop, target and the ratio between them.

Levels are *derived*, never chosen. The previous implementation multiplied the
reference price by fixed constants (0.985 / 1.030) and then asserted
``risk_reward_ratio=2.0`` regardless of where those levels actually sat -- a
number that was true by declaration rather than by arithmetic.

Here the stop comes from volatility (ATR) widened to clear the nearest
structural level, the target comes from the opposing structural level when one
exists, and the ratio is computed from the two distances. If a level cannot be
derived from real data the caller is told so; nothing is substituted.

Why ATR and structure rather than a percentage
----------------------------------------------
A fixed 1.5% stop is 5x the average bar range on a quiet 15m chart and a third
of it during a volatility shock. The same constant therefore means two entirely
different things, and neither is the trader's actual risk. ATR expresses the
stop in the units the market is currently moving in.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from quantedge.contracts import SignalDirection

if TYPE_CHECKING:
    from quantedge.contracts import FeatureSnapshot, StructureReport

__all__ = ["MIN_ACCEPTABLE_RR", "RiskLevels", "derive_risk_levels"]

# Stop distance as a multiple of ATR. 1.5 keeps the stop outside ordinary bar
# noise without sitting so far away that the target becomes unreachable.
_ATR_STOP_MULTIPLE = Decimal("1.5")

# Hard ceiling on stop distance, also in ATR. Beyond this a structural level is
# not protecting the entry, it is describing a trade with a different premise.
_MAX_STOP_ATR_MULTIPLE = Decimal("2.5")

# Minimum reward:risk we are willing to present. Below this the setup is not
# worth taking even when direction is right, so the caller is told rather than
# handed a target that has been quietly stretched to hit a nicer ratio.
MIN_ACCEPTABLE_RR = Decimal("1.2")

# Fallback target distance when no opposing structural level exists, expressed
# as a multiple of the stop distance -- not of price.
_DEFAULT_TARGET_R = Decimal("2.0")


class RiskLevels:
    """Stop, target and the arithmetic that connects them."""

    __slots__ = (
        "basis",
        "notes",
        "reward_distance",
        "risk_distance",
        "rr",
        "stop",
        "target",
        "target_from_structure",
    )

    def __init__(
        self,
        stop: Decimal,
        target: Decimal,
        rr: float,
        risk_distance: Decimal,
        reward_distance: Decimal,
        basis: str,
        notes: list[str],
        target_from_structure: bool = True,
    ) -> None:
        self.stop = stop
        self.target = target
        self.rr = rr
        self.risk_distance = risk_distance
        self.reward_distance = reward_distance
        self.basis = basis
        self.notes = notes
        self.target_from_structure = target_from_structure

    @property
    def acceptable(self) -> bool:
        """Whether the derived ratio clears the minimum worth presenting.

        A ratio computed against an R-multiple target is ``_DEFAULT_TARGET_R`` by
        construction and would clear any threshold below it, so it is not
        evidence and does not count as clearing this gate. Only a target taken
        from a real structural level can.
        """
        if not self.target_from_structure:
            return False
        return Decimal(str(self.rr)) >= MIN_ACCEPTABLE_RR


def derive_risk_levels(
    *,
    reference_price: Decimal,
    direction: SignalDirection,
    features: FeatureSnapshot,
    structure: StructureReport | None,
) -> RiskLevels | None:
    """Derive stop and target from ATR and structure.

    Returns ``None`` when ATR is unavailable: without a volatility measure there
    is no honest way to place a stop, and inventing one would reintroduce the
    fabrication this module exists to remove.
    """
    atr = features.atr_14
    if atr is None or atr <= 0:
        return None

    atr_d = Decimal(str(atr))
    notes: list[str] = []
    quantum = _quantum_for(reference_price)

    stop_distance = atr_d * _ATR_STOP_MULTIPLE
    basis = f"{_ATR_STOP_MULTIPLE}x ATR({atr_d.quantize(quantum)})"

    # Widen the stop past the protective structural level when one is close, so
    # it is not sitting exactly where price is most likely to wick. "Close" is
    # the operative word and it used to be unbounded: a support 4x ATR below the
    # entry dragged the stop out to 4.5x ATR, which is no longer a volatility
    # stop but a different trade, and it destroyed the ratio arithmetically
    # before geometry was even considered. Past the cap the level is too far to
    # be protecting this entry, so the ATR stop stands and the distance is
    # reported instead of silently absorbed.
    protective = _protective_level(structure, direction)
    if protective is not None:
        needed = abs(reference_price - protective) + (atr_d * Decimal("0.25"))
        if needed > atr_d * _MAX_STOP_ATR_MULTIPLE:
            notes.append(
                f"nearest protective level is {(needed / atr_d).quantize(Decimal('0.1'))}x ATR "
                f"away; too distant to anchor the stop, so {_ATR_STOP_MULTIPLE}x ATR stands"
            )
        elif needed > stop_distance:
            stop_distance = needed
            level_name = "support" if direction is SignalDirection.UP else "resistance"
            basis = f"beyond {level_name} {protective.quantize(quantum)} + 0.25x ATR"
            notes.append("stop widened to clear the nearest protective level")

    objective = _objective_level(structure, direction)
    min_reward = stop_distance * MIN_ACCEPTABLE_RR
    if objective is not None and abs(objective - reference_price) >= min_reward:
        reward_distance = abs(objective - reference_price)
        target_basis = "opposing structural level"
        target_from_structure = True
    else:
        # No structural level far enough away to aim at. The target is then a
        # multiple of the stop, which means the resulting ratio is arithmetic on
        # our own constant rather than a measurement of the setup -- it comes out
        # at exactly _DEFAULT_TARGET_R every time. That is fine as a level to
        # trade toward and useless as evidence the trade is worth taking, so it
        # is flagged and excluded from the acceptability gate below. Presenting
        # it as "reward:risk 2.00" alongside genuine ratios was the defect: it
        # made every such setup look like it had cleared a bar it never faced.
        reward_distance = stop_distance * _DEFAULT_TARGET_R
        target_basis = f"{_DEFAULT_TARGET_R}R from the stop; no structural target in range"
        target_from_structure = False
        notes.append(
            "no opposing structural level within reach: the target is derived from "
            "the stop, so the ratio below is definitional, not measured"
        )

    if direction is SignalDirection.UP:
        stop = reference_price - stop_distance
        target = reference_price + reward_distance
    else:
        stop = reference_price + stop_distance
        target = reference_price - reward_distance

    if stop <= 0 or target <= 0:
        return None

    rr = float((reward_distance / stop_distance).quantize(Decimal("0.01")))
    notes.append(f"target from {target_basis}")

    return RiskLevels(
        stop=stop.quantize(quantum),
        target=target.quantize(quantum),
        rr=rr,
        risk_distance=stop_distance.quantize(quantum),
        reward_distance=reward_distance.quantize(quantum),
        basis=basis,
        notes=notes,
        target_from_structure=target_from_structure,
    )


def _protective_level(
    structure: StructureReport | None, direction: SignalDirection
) -> Decimal | None:
    """The level the stop must sit beyond: support for longs, resistance for shorts."""
    if structure is None:
        return None
    level = (
        structure.nearest_support
        if direction is SignalDirection.UP
        else structure.nearest_resistance
    )
    return Decimal(str(level)) if level is not None else None


def _objective_level(
    structure: StructureReport | None, direction: SignalDirection
) -> Decimal | None:
    """The level the target aims at: resistance for longs, support for shorts."""
    if structure is None:
        return None
    level = (
        structure.nearest_resistance
        if direction is SignalDirection.UP
        else structure.nearest_support
    )
    return Decimal(str(level)) if level is not None else None


def _quantum_for(price: Decimal) -> Decimal:
    """Rounding precision appropriate to the price magnitude.

    A single fixed ``0.0001`` quantum is wrong at both ends: it is meaningless
    precision on a 65,000 index and it truncates a 0.00001234 altcoin to zero.
    """
    if price >= 1000:
        return Decimal("0.01")
    if price >= 1:
        return Decimal("0.0001")
    return Decimal("0.00000001")
