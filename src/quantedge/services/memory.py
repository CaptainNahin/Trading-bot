"""The trade memory bank: what happened, and for a loss, why.

Two paths, deliberately asymmetric
----------------------------------
A win is recorded and moved on from. A loss is diagnosed first: the holding
period is fetched, measured in :mod:`quantedge.services.postmortem`, and the
memory is written with the causes that actually held and the numbers behind them.
That asymmetry is the requested behaviour -- "if successful it will just add it
to its memory... if it goes to loss it will first make sure what's the problem
and why the trade came in loss then add the loss to its memory with the reason".

Rules are derived, not authored
-------------------------------
The DO/DONT rules attached to a memory come from the diagnosed cause codes. The
previous version wrote rules like "wait for candle close confirmation" onto every
loss regardless of what happened, so the memory bank filled with advice that no
observation supported. Here a rule exists only because a specific cause was
measured, and it names that cause.
"""

from __future__ import annotations

import uuid
from collections import Counter
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from quantedge.contracts import (
    AssetClass,
    MarketRegime,
    SettlementOutcome,
    SignalDirection,
    TradeMemory,
    utc_now,
)
from quantedge.logging import get_logger
from quantedge.repositories import get_repository
from quantedge.services.postmortem import PostMortem, diagnose

if TYPE_CHECKING:
    from datetime import datetime

    from quantedge.contracts import Candle

__all__ = [
    "get_memory_bank_summary",
    "get_relevant_memories",
    "record_trade_outcome_and_analyze",
    "recurring_loss_rules",
]

log = get_logger(__name__)

# Lessons and rules keyed by the cause the post-mortem measured. Each is a
# statement about this trade's mechanics, not general trading advice.
_CAUSE_RULES: dict[str, tuple[str, str | None, str | None]] = {
    # code: (lesson, do_rule, dont_rule)
    "STOP_TOO_TIGHT": (
        "The stop was inside normal bar range: price took it out and came back.",
        "Size the stop from the ATR that applies at entry, and widen it past the "
        "nearest protective level before taking the trade.",
        None,
    ),
    "STOP_HIT_AND_HELD": (
        "Price moved against the position and stayed there; the directional read "
        "was wrong rather than early.",
        None,
        "Do not re-enter the same direction until structure confirms it again.",
    ),
    "GAVE_BACK_OPEN_PROFIT": (
        "The trade was well into profit before reversing, so entry timing was "
        "sound and the exit was late.",
        "On a fixed time limit, treat a target reached early as the exit rather "
        "than holding for expiry.",
        None,
    ),
    "EXPIRED_BEFORE_RESOLUTION": (
        "Neither level was reached: the time limit was shorter than the setup needed to resolve.",
        "Match the time limit to the horizon the structure implies, or pick a "
        "longer limit for this pattern.",
        None,
    ),
    "STRUCTURE_FLIPPED": (
        "Structure reversed while the position was open.",
        None,
        "Do not hold through a structure flip on the execution timeframe.",
    ),
    "STRUCTURE_OPPOSED": (
        "Structure at expiry was against the position.",
        "Require the execution timeframe's structure to agree with the direction at entry.",
        None,
    ),
    "FAILED_BREAKOUT": (
        "The breakout this entry relied on failed: price cleared the level and "
        "closed back inside it.",
        "Wait for a closed bar beyond the level before treating a breakout as real.",
        None,
    ),
    "CHANGE_OF_CHARACTER": (
        "A change of character formed during the trade, ending the prior trend.",
        None,
        "Do not add to or hold a position after a CHoCH against it.",
    ),
    "VOLATILITY_EXPANSION": (
        "Volatility expanded well beyond the level it was measured at.",
        "Re-derive levels when ATR changes materially during the holding period.",
        None,
    ),
    "TREND_DECAYED": (
        "ADX collapsed during the trade: the trend stopped trending.",
        None,
        "Do not hold a trend-following entry once ADX falls back below the "
        "threshold that qualified it.",
    ),
    "NO_SETTLEMENT_DATA": (
        "No closed candles covered the holding period, so this loss could not be diagnosed.",
        "Record the settlement provider and expiry so the holding period can be "
        "reconstructed next time.",
        None,
    ),
}


def record_trade_outcome_and_analyze(
    signal_id: str,
    outcome: SettlementOutcome | str,
    *,
    symbol: str = "BTCUSDT",
    asset_class: AssetClass | str = AssetClass.CRYPTO,
    horizon: str = "15m",
    regime: MarketRegime | str = MarketRegime.UNCERTAIN,
    pattern: str = "general",
    direction: SignalDirection | str | None = None,
    reference_price: Decimal | None = None,
    exit_price: Decimal | None = None,
    stop: Decimal | None = None,
    target: Decimal | None = None,
    holding_candles: list[Candle] | None = None,
    entry_time: datetime | None = None,
    entry_structure: object | None = None,
    exit_structure: object | None = None,
    entry_features: object | None = None,
    exit_features: object | None = None,
    user_notes: str | None = None,
) -> TradeMemory:
    """Record a settled trade, diagnosing the cause first when it lost.

    ``holding_candles`` are the closed bars from entry to expiry. Supplying them
    is what makes a loss diagnosable; without them the memory records the outcome
    and says plainly that the cause could not be measured, rather than filling in
    a plausible one.
    """
    repo = get_repository()

    out_enum = (
        outcome
        if isinstance(outcome, SettlementOutcome)
        else SettlementOutcome(str(outcome).upper())
    )
    reg_enum = regime if isinstance(regime, MarketRegime) else MarketRegime(str(regime).upper())
    ast_enum = (
        asset_class if isinstance(asset_class, AssetClass) else AssetClass(str(asset_class).lower())
    )
    dir_enum: SignalDirection | None
    if direction is None or isinstance(direction, SignalDirection):
        dir_enum = direction
    else:
        dir_enum = SignalDirection(str(direction).upper())

    mem_id = f"mem-{uuid.uuid4().hex[:12]}"
    mortem: PostMortem | None = None

    if out_enum is SettlementOutcome.WIN:
        root_cause, lessons, do_rules, dont_rules = _win_record(
            symbol=symbol, horizon=horizon, regime=reg_enum, pattern=pattern
        )
    elif out_enum is SettlementOutcome.LOSS:
        root_cause, lessons, do_rules, dont_rules, mortem = _loss_record(
            symbol=symbol,
            direction=dir_enum,
            reference_price=reference_price,
            holding_candles=holding_candles,
            entry_time=entry_time,
            stop=stop,
            target=target,
            entry_structure=entry_structure,
            exit_structure=exit_structure,
            entry_features=entry_features,
            exit_features=exit_features,
        )
    else:
        root_cause = (
            f"{symbol} settled {out_enum.value} on the {horizon} horizon: the "
            "position closed at the reference price, so neither direction was "
            "resolved. No cause is assigned to a non-directional outcome."
        )
        lessons = [f"{out_enum.value} outcomes carry no directional information."]
        do_rules = []
        dont_rules = []

    if user_notes:
        root_cause += f" Trader's note: {user_notes}"

    memory = TradeMemory(
        memory_id=mem_id,
        signal_id=signal_id,
        symbol=symbol,
        asset_class=ast_enum,
        horizon=horizon,
        regime=reg_enum,
        pattern=pattern,
        outcome=out_enum,
        reference_price=reference_price,
        exit_price=exit_price if exit_price is not None else _exit_of(mortem),
        root_cause=root_cause,
        key_lessons=lessons,
        do_rules=do_rules,
        dont_rules=dont_rules,
        user_notes=user_notes,
        created_at_utc=utc_now(),
    )

    repo.save_trade_memory(memory)
    log.info(
        "trade memory recorded",
        extra={
            "memory_id": mem_id,
            "symbol": symbol,
            "outcome": out_enum.value,
            "causes": [c.code for c in mortem.causes] if mortem else [],
        },
    )
    return memory


def _exit_of(mortem: PostMortem | None) -> Decimal | None:
    return mortem.exit_price if mortem is not None else None


def _win_record(
    *, symbol: str, horizon: str, regime: MarketRegime, pattern: str
) -> tuple[str, list[str], list[str], list[str]]:
    """A win, recorded plainly.

    No post-mortem is run and no rule is derived. A win says the setup worked
    *this time*; turning that into "prioritise setups like this" would be
    inferring an edge from one favourable sample, which is the reasoning error
    that makes a memory bank actively harmful. The record exists so the win
    counts in the summary and the conditions are searchable later.
    """
    root_cause = (
        f"{symbol} {horizon} settled WIN in a {regime.value} regime on the "
        f"{pattern} pattern. Recorded without a post-mortem: the outcome matched "
        "the setup, and one favourable result is not evidence of an edge."
    )
    lessons = [
        f"{pattern} in {regime.value} on the {horizon} horizon produced a win here.",
    ]
    return root_cause, lessons, [], []


def _loss_record(
    *,
    symbol: str,
    direction: SignalDirection | None,
    reference_price: Decimal | None,
    holding_candles: list[Candle] | None,
    entry_time: datetime | None,
    stop: Decimal | None,
    target: Decimal | None,
    entry_structure: object | None,
    exit_structure: object | None,
    entry_features: object | None,
    exit_features: object | None,
) -> tuple[str, list[str], list[str], list[str], PostMortem | None]:
    """A loss, diagnosed before it is written.

    When direction, entry price or the holding period is missing, nothing can be
    measured -- and the memory says exactly that instead of substituting a
    template. A memory bank whose rows cannot be told apart is worse than an
    empty one, because it is consulted as though it held evidence.
    """
    if direction is None or reference_price is None or not holding_candles:
        missing = [
            name
            for name, present in (
                ("direction", direction is not None),
                ("reference price", reference_price is not None),
                ("holding-period candles", bool(holding_candles)),
            )
            if not present
        ]
        root_cause = (
            f"{symbol} settled LOSS. The cause could not be determined: "
            f"{', '.join(missing)} unavailable, so no measurement of the holding "
            "period was possible. Recorded as undiagnosed rather than assigned a "
            "presumed cause."
        )
        return root_cause, [], [], [], None

    mortem = diagnose(
        symbol=symbol,
        direction=direction,
        reference_price=reference_price,
        holding_candles=holding_candles,
        entry_time=entry_time,
        stop=stop,
        target=target,
        entry_structure=entry_structure,
        exit_structure=exit_structure,
        entry_features=entry_features,
        exit_features=exit_features,
    )

    lessons: list[str] = []
    do_rules: list[str] = []
    dont_rules: list[str] = []
    for cause in mortem.causes:
        lesson, do_rule, dont_rule = _CAUSE_RULES.get(cause.code, (None, None, None))
        if lesson:
            lessons.append(lesson)
        if do_rule:
            do_rules.append(do_rule)
        if dont_rule:
            dont_rules.append(dont_rule)

    return (
        mortem.root_cause(),
        list(dict.fromkeys(lessons)),
        list(dict.fromkeys(do_rules)),
        list(dict.fromkeys(dont_rules)),
        mortem,
    )


def get_relevant_memories(
    symbol: str | None = None,
    regime: str | None = None,
    limit: int = 50,
) -> list[TradeMemory]:
    """Stored post-mortems, most recent first."""
    repo = get_repository()
    return repo.list_trade_memories(symbol=symbol, regime=regime, limit=limit)


def recurring_loss_rules(
    symbol: str,
    *,
    horizon: str | None = None,
    min_occurrences: int = 2,
    limit: int = 5,
) -> list[str]:
    """DON'T rules whose loss cause recurred on ``symbol``, most frequent first.

    A rule earns a place here by having been derived from at least
    ``min_occurrences`` separate losing trades. One loss is an event; the same
    measured cause twice is the weakest thing that can honestly be called a
    pattern, and surfacing a single occurrence as a rule would dress one bad
    trade up as a tendency.

    The returned strings carry their own occurrence count, so a consumer cannot
    present "seen twice" and "seen nine times" as the same claim. Nothing here
    adjusts a score: these are records of trades that already closed, and
    letting them move a number would manufacture a calibration this system has
    never measured. They qualify a setup; they do not rate it.
    """
    memories = [
        m
        for m in get_relevant_memories(symbol=symbol, limit=200)
        if m.outcome == SettlementOutcome.LOSS
        and (horizon is None or m.horizon == horizon)
        and m.dont_rules
    ]

    counts: Counter[str] = Counter()
    for memory in memories:
        # ``.keys()`` matters: Counter.update given a mapping adds its values, and
        # dict.fromkeys supplies None. Passing the keys counts occurrences, while
        # the dedup keeps one trade from voting twice for its own rule.
        counts.update(dict.fromkeys(memory.dont_rules).keys())

    return [
        f"{rule} (recorded on {count} past losing trade(s) for {symbol})"
        for rule, count in counts.most_common(limit)
        if count >= min_occurrences
    ]


def get_memory_bank_summary() -> dict[str, Any]:
    """Counts, the observed win rate, and the rules that were derived.

    ``observed_win_rate`` is the realised frequency in this record set and
    nothing more. ``sample_too_small`` is carried alongside it because a 3-trade
    sample reading 0.67 is noise, and a bare 0.67 next to it reads as skill.
    """
    repo = get_repository()
    memories = repo.list_trade_memories(limit=500)

    wins = sum(1 for m in memories if m.outcome == SettlementOutcome.WIN)
    losses = sum(1 for m in memories if m.outcome == SettlementOutcome.LOSS)
    decided = wins + losses

    do_rules: list[str] = []
    dont_rules: list[str] = []
    for m in memories:
        do_rules.extend(m.do_rules)
        dont_rules.extend(m.dont_rules)

    return {
        "total_memories": len(memories),
        "wins": wins,
        "losses": losses,
        "observed_win_rate": round(wins / decided, 4) if decided else None,
        "sample_too_small": decided < 30,
        "top_do_rules": list(dict.fromkeys(do_rules))[:10],
        "top_dont_rules": list(dict.fromkeys(dont_rules))[:10],
        "last_updated_utc": utc_now().isoformat(),
    }
