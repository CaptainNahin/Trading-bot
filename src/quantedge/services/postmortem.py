"""Why a trade lost: measured from the candles that formed after entry.

The previous implementation picked a root cause from an ``if outcome == LOSS``
branch and filled in a sentence -- "counter-trend rejection or volatility shock"
-- without looking at a single bar. Two different losses produced identical
text, so the memory bank accumulated rows that could not distinguish a stop run
from a regime flip, and the DO/DONT rules built from them were assertions about
markets rather than observations of one.

This module answers the question with arithmetic. It fetches the closed candles
covering the holding period, measures what price actually did, recomputes
structure and features at expiry, and compares them against what was true at
entry. Every cause it reports carries the numbers it was derived from.

What it will not do
-------------------
It does not rank causes by a learned weight, and it does not express confidence
as a percentage: no calibration set exists, so a "78% likely stop hunt" would be
invented. Causes are reported as *observations that hold*, most specific first,
and a loss that fits none of them is recorded as unexplained rather than
assigned the nearest template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from quantedge.contracts import SignalDirection
from quantedge.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime

    from quantedge.contracts import Candle

__all__ = ["LossCause", "PostMortem", "diagnose"]

log = get_logger(__name__)

# An adverse excursion smaller than this fraction of the intended risk means
# price never really threatened the position -- the loss came from time running
# out, not from being wrong about direction.
_SHALLOW_ADVERSE = Decimal("0.5")

# Favourable excursion beyond this fraction of the intended reward means the
# trade was winning before it turned. That distinguishes "the read was wrong"
# from "the read was right and the exit was late".
_WAS_IN_PROFIT = Decimal("0.5")


@dataclass(frozen=True)
class LossCause:
    """One observation that held, with the measurement behind it."""

    code: str
    explanation: str
    evidence: dict[str, str] = field(default_factory=dict)


@dataclass
class PostMortem:
    """Measured account of a holding period."""

    symbol: str
    direction: SignalDirection
    reference_price: Decimal
    exit_price: Decimal
    bars_observed: int
    max_adverse_excursion: Decimal
    max_favourable_excursion: Decimal
    stop_touched: bool | None
    target_touched: bool | None
    causes: list[LossCause] = field(default_factory=list)

    @property
    def price_change(self) -> Decimal:
        return self.exit_price - self.reference_price

    def root_cause(self) -> str:
        """One sentence naming the causes that held, or stating that none did."""
        moved = "against" if self._moved_against() else "in favour of"
        header = (
            f"{self.symbol} {self.direction.value}: price closed at {self.exit_price} "
            f"from {self.reference_price} ({moved} the position) over "
            f"{self.bars_observed} closed bars; worst adverse excursion "
            f"{self.max_adverse_excursion}, best favourable excursion "
            f"{self.max_favourable_excursion}."
        )
        if not self.causes:
            return (
                f"{header} No single measurable cause held: the move was within "
                "the range the setup allowed for and no structural, regime or "
                "quality condition changed. Recorded as unexplained."
            )
        return header + " " + " ".join(c.explanation for c in self.causes)

    def _moved_against(self) -> bool:
        if self.direction is SignalDirection.UP:
            return self.exit_price < self.reference_price
        return self.exit_price > self.reference_price

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "reference_price": str(self.reference_price),
            "exit_price": str(self.exit_price),
            "bars_observed": self.bars_observed,
            "max_adverse_excursion": str(self.max_adverse_excursion),
            "max_favourable_excursion": str(self.max_favourable_excursion),
            "stop_touched": self.stop_touched,
            "target_touched": self.target_touched,
            "causes": [
                {"code": c.code, "explanation": c.explanation, "evidence": c.evidence}
                for c in self.causes
            ],
        }


def diagnose(
    *,
    symbol: str,
    direction: SignalDirection,
    reference_price: Decimal,
    holding_candles: list[Candle],
    entry_time: datetime | None = None,
    stop: Decimal | None = None,
    target: Decimal | None = None,
    entry_structure: object | None = None,
    exit_structure: object | None = None,
    entry_features: object | None = None,
    exit_features: object | None = None,
) -> PostMortem:
    """Measure a holding period and name the causes that hold.

    ``holding_candles`` must be closed bars from entry to expiry. Bars still
    forming are excluded by the caller; including one would let a partial bar's
    high decide whether a stop was touched, and that reading changes as the bar
    develops -- the repainting Rule 9 exists to prevent.

    Structure and feature snapshots from entry and expiry are optional. When both
    are supplied the diagnosis can distinguish a regime that changed underneath
    the trade from one that held while price simply went the other way; without
    them the excursion analysis still stands on its own.
    """
    closed = [c for c in holding_candles if c.is_closed]
    if entry_time is not None:
        closed = [c for c in closed if c.close_time_utc > entry_time]

    if not closed:
        return PostMortem(
            symbol=symbol,
            direction=direction,
            reference_price=reference_price,
            exit_price=reference_price,
            bars_observed=0,
            max_adverse_excursion=Decimal("0"),
            max_favourable_excursion=Decimal("0"),
            stop_touched=None,
            target_touched=None,
            causes=[
                LossCause(
                    "NO_SETTLEMENT_DATA",
                    "No closed candles cover the holding period, so nothing about "
                    "this trade can be measured. The outcome is recorded, the "
                    "cause is not inferred.",
                )
            ],
        )

    exit_price = closed[-1].close
    highest = max(c.high for c in closed)
    lowest = min(c.low for c in closed)

    if direction is SignalDirection.UP:
        adverse = reference_price - lowest
        favourable = highest - reference_price
        stop_touched = lowest <= stop if stop is not None else None
        target_touched = highest >= target if target is not None else None
    else:
        adverse = highest - reference_price
        favourable = reference_price - lowest
        stop_touched = highest >= stop if stop is not None else None
        target_touched = lowest <= target if target is not None else None

    mortem = PostMortem(
        symbol=symbol,
        direction=direction,
        reference_price=reference_price,
        exit_price=exit_price,
        bars_observed=len(closed),
        max_adverse_excursion=max(adverse, Decimal("0")),
        max_favourable_excursion=max(favourable, Decimal("0")),
        stop_touched=stop_touched,
        target_touched=target_touched,
    )
    mortem.causes = _causes_for(
        mortem,
        stop=stop,
        target=target,
        entry_structure=entry_structure,
        exit_structure=exit_structure,
        entry_features=entry_features,
        exit_features=exit_features,
    )
    log.info(
        "post-mortem completed",
        extra={
            "symbol": symbol,
            "bars": len(closed),
            "causes": [c.code for c in mortem.causes],
        },
    )
    return mortem


def _causes_for(
    mortem: PostMortem,
    *,
    stop: Decimal | None,
    target: Decimal | None,
    entry_structure: object | None,
    exit_structure: object | None,
    entry_features: object | None,
    exit_features: object | None,
) -> list[LossCause]:
    """Every observation that holds for this holding period.

    Ordered most specific first, and deliberately not reduced to one: a trade
    can be both stopped out and have had its structure break, and recording only
    the first would lose the fact that the two happened together.
    """
    causes: list[LossCause] = []
    risk = abs(mortem.reference_price - stop) if stop is not None else None
    reward = abs(target - mortem.reference_price) if target is not None else None

    # 1. Stopped out, then price recovered past entry: the level was too tight
    # for the volatility, which is a sizing observation rather than a directional
    # one and calls for a different fix.
    if mortem.stop_touched and _recovered(mortem):
        causes.append(
            LossCause(
                "STOP_TOO_TIGHT",
                f"The stop at {stop} was touched and price then returned through "
                f"entry to {mortem.exit_price}, so direction was not the error -- "
                "the stop sat inside the range the market was moving in.",
                {
                    "stop": str(stop),
                    "max_adverse_excursion": str(mortem.max_adverse_excursion),
                    "exit_price": str(mortem.exit_price),
                },
            )
        )
    elif mortem.stop_touched:
        causes.append(
            LossCause(
                "STOP_HIT_AND_HELD",
                f"The stop at {stop} was touched and price did not recover: "
                f"adverse excursion reached {mortem.max_adverse_excursion} and the "
                f"period closed at {mortem.exit_price}. The directional read was wrong.",
                {"stop": str(stop), "exit_price": str(mortem.exit_price)},
            )
        )

    # 2. Was in profit and gave it back -- an exit-timing observation. Only
    # meaningful when a target existed to measure against.
    if (
        reward is not None
        and reward > 0
        and not mortem.target_touched
        and mortem.max_favourable_excursion >= reward * _WAS_IN_PROFIT
    ):
        causes.append(
            LossCause(
                "GAVE_BACK_OPEN_PROFIT",
                f"Price ran {mortem.max_favourable_excursion} in favour "
                f"({_pct(mortem.max_favourable_excursion, reward)} of the "
                f"{reward} target) before reversing. The entry was timed well and "
                "the exit was not.",
                {
                    "max_favourable_excursion": str(mortem.max_favourable_excursion),
                    "target_distance": str(reward),
                },
            )
        )

    # 3. Neither level reached, the move was shallow both ways: the trade expired
    # rather than failed. Requires the favourable excursion to be small too --
    # a trade that ran most of the way to target and came back did resolve, it
    # just resolved badly, and that is the cause above rather than this one.
    ran_meaningfully = (
        reward is not None
        and reward > 0
        and mortem.max_favourable_excursion >= reward * _WAS_IN_PROFIT
    )
    if (
        risk is not None
        and risk > 0
        and not mortem.stop_touched
        and not mortem.target_touched
        and not ran_meaningfully
        and mortem.max_adverse_excursion < risk * _SHALLOW_ADVERSE
    ):
        causes.append(
            LossCause(
                "EXPIRED_BEFORE_RESOLUTION",
                f"Neither stop nor target was reached in {mortem.bars_observed} "
                f"bars and the worst adverse excursion was only "
                f"{mortem.max_adverse_excursion} against {risk} of intended risk. "
                "The time limit expired before the setup resolved.",
                {
                    "bars_observed": str(mortem.bars_observed),
                    "intended_risk": str(risk),
                    "max_adverse_excursion": str(mortem.max_adverse_excursion),
                },
            )
        )

    causes.extend(_structure_causes(mortem, entry_structure, exit_structure))
    causes.extend(_feature_causes(entry_features, exit_features))
    return causes


def _recovered(mortem: PostMortem) -> bool:
    """Whether the period closed on the favourable side of entry."""
    if mortem.direction is SignalDirection.UP:
        return mortem.exit_price > mortem.reference_price
    return mortem.exit_price < mortem.reference_price


def _pct(part: Decimal, whole: Decimal) -> str:
    if whole <= 0:
        return "n/a"
    return f"{(part / whole * 100).quantize(Decimal('0.1'))}%"


def _structure_causes(
    mortem: PostMortem,
    entry: object | None,
    exit_: object | None,
) -> list[LossCause]:
    """Structural changes between entry and expiry, if both were captured."""
    causes: list[LossCause] = []
    if exit_ is None:
        return causes

    wanted = "UPTREND" if mortem.direction is SignalDirection.UP else "DOWNTREND"
    opposed = "DOWNTREND" if mortem.direction is SignalDirection.UP else "UPTREND"

    entry_state = getattr(entry, "structure", None) if entry is not None else None
    exit_state = getattr(exit_, "structure", None)

    if exit_state == opposed and entry_state == wanted:
        causes.append(
            LossCause(
                "STRUCTURE_FLIPPED",
                f"Market structure was {entry_state} at entry and {exit_state} at "
                "expiry: the trend the setup was built on reversed while the "
                "position was open.",
                {"entry_structure": str(entry_state), "exit_structure": str(exit_state)},
            )
        )
    elif exit_state == opposed:
        causes.append(
            LossCause(
                "STRUCTURE_OPPOSED",
                f"Structure at expiry was {exit_state}, against the position.",
                {"exit_structure": str(exit_state)},
            )
        )

    if getattr(exit_, "failed_breakout", False):
        causes.append(
            LossCause(
                "FAILED_BREAKOUT",
                "The structure engine recorded a failed breakout during the "
                "holding period: price cleared the level and closed back inside it.",
                {"failed_breakout": "true"},
            )
        )

    choch = getattr(exit_, "last_choch", None)
    if choch is not None and choch != getattr(entry, "last_choch", None):
        # Summarised to the two numbers that matter -- which way it broke and the
        # level it broke -- rather than interpolating the event object. Its repr
        # carries ``datetime.datetime(...)`` and ``<SignalDirection.UP: 'UP'>``,
        # and this string is stored in the memory bank and read back to the
        # trader, so a repr here becomes permanent user-facing noise.
        where = getattr(choch, "level", None)
        which = getattr(getattr(choch, "direction", None), "value", None)
        detail = " ".join(
            part
            for part in (
                f"{which} through" if which else None,
                f"{where}" if where is not None else None,
            )
            if part
        )
        causes.append(
            LossCause(
                "CHANGE_OF_CHARACTER",
                "A change of character formed during the trade"
                + (f" ({detail})" if detail else "")
                + ", which is the structural signal that the prior trend is no "
                "longer in control.",
                {"last_choch": detail or "recorded"},
            )
        )
    return causes


def _feature_causes(entry: object | None, exit_: object | None) -> list[LossCause]:
    """Volatility and trend-strength changes, if both snapshots were captured."""
    causes: list[LossCause] = []
    if entry is None or exit_ is None:
        return causes

    entry_atr = getattr(entry, "atr_14", None)
    exit_atr = getattr(exit_, "atr_14", None)
    if entry_atr and exit_atr and entry_atr > 0:
        ratio = exit_atr / entry_atr
        if ratio >= 1.75:
            causes.append(
                LossCause(
                    "VOLATILITY_EXPANSION",
                    f"ATR rose from {round(entry_atr, 6)} to {round(exit_atr, 6)} "
                    f"({ratio:.2f}x) during the trade. Levels derived from the "
                    "entry-time ATR were too narrow for the volatility that arrived.",
                    {"entry_atr": str(round(entry_atr, 6)), "exit_atr": str(round(exit_atr, 6))},
                )
            )

    entry_adx = getattr(entry, "adx_14", None)
    exit_adx = getattr(exit_, "adx_14", None)
    if entry_adx is not None and exit_adx is not None and entry_adx >= 20 and exit_adx < 15:
        causes.append(
            LossCause(
                "TREND_DECAYED",
                f"ADX fell from {round(entry_adx, 2)} to {round(exit_adx, 2)}: the "
                "trend that justified a directional trade stopped trending.",
                {"entry_adx": str(round(entry_adx, 2)), "exit_adx": str(round(exit_adx, 2))},
            )
        )
    return causes
