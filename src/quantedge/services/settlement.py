"""Deterministic Signal Settlement & Performance Tracking Engine.

Evaluates settled signal outcomes strictly from closed candle price history.
All settled records are immutable and appended via the repository layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from quantedge.contracts import (
    AIDecision,
    Candle,
    PerformanceSummary,
    SettledSignal,
    SettlementOutcome,
    SignalDirection,
    SignalStatus,
    utc_now,
)
from quantedge.repositories import get_repository


def settle_decision(
    decision: AIDecision,
    closed_candles: Sequence[Candle],
    *,
    settlement_provider: str = "binance",
    outcome: SettlementOutcome | None = None,
    settlement_price: Decimal | None = None,
    notes: list[str] | None = None,
) -> SettledSignal | None:
    """Evaluate settlement for an AIDecision using closed candles.

    Returns
    -------
    SettledSignal | None
        The immutable settlement record, or None if the signal is not eligible
        (NO_TRADE, no direction, no reference price, no bars, or no expiry).

    Raises
    ------
    PersistenceError
        If the record could not be written -- including a second settlement of
        the same signal, which the immutable table refuses. This propagates
        rather than being swallowed: returning the record after a failed write
        told the caller a trade had been scored and stored when nothing had
        been stored, and a worker counting those returns reported settlements
        that were not in the database.
    """
    if (
        decision is None
        or decision.status != SignalStatus.SIGNAL
        or not decision.direction
        or not decision.reference_price
    ):
        return None

    if not closed_candles:
        return None

    # No expiry, no settlement. The window a trade is scored over is part of the
    # signal; substituting "now" would score it over whatever interval the
    # settlement job happened to run in, which is not a property of the trade.
    if decision.expiry_utc is None:
        return None

    final_price = settlement_price if settlement_price is not None else closed_candles[-1].close
    ref_price = decision.reference_price

    if outcome is None:
        if decision.direction == SignalDirection.UP:
            if final_price > ref_price:
                derived_outcome = SettlementOutcome.WIN
            elif final_price < ref_price:
                derived_outcome = SettlementOutcome.LOSS
            else:
                derived_outcome = SettlementOutcome.FLAT
        elif decision.direction == SignalDirection.DOWN:
            if final_price < ref_price:
                derived_outcome = SettlementOutcome.WIN
            elif final_price > ref_price:
                derived_outcome = SettlementOutcome.LOSS
            else:
                derived_outcome = SettlementOutcome.FLAT
        else:
            derived_outcome = SettlementOutcome.FLAT
    else:
        derived_outcome = outcome

    settled_notes = notes if notes is not None else [f"Settled against closing price {final_price}"]

    settled = SettledSignal(
        signal_id=decision.decision_id or "unknown",
        symbol=decision.symbol,
        horizon=decision.horizon,
        direction=decision.direction,
        reference_price=ref_price,
        settlement_price=final_price,
        outcome=derived_outcome,
        expiry_utc=decision.expiry_utc,
        settled_at_utc=utc_now(),
        settlement_provider=settlement_provider,
        notes=settled_notes,
    )

    get_repository().settle_signal(settled)
    return settled


def get_performance_summary(
    symbol: str | None = None,
    horizon: str | None = None,
) -> PerformanceSummary:
    """Calculate realized historical performance metrics across settled signals."""
    repo = get_repository()
    settled_list = repo.settled_signals(symbol=symbol, horizon=horizon)

    total = len(settled_list)
    wins = sum(1 for s in settled_list if s.outcome == SettlementOutcome.WIN)
    losses = sum(1 for s in settled_list if s.outcome == SettlementOutcome.LOSS)
    flat = sum(1 for s in settled_list if s.outcome == SettlementOutcome.FLAT)
    void = sum(1 for s in settled_list if s.outcome == SettlementOutcome.VOID)

    win_rate = float(wins / (wins + losses)) if (wins + losses) > 0 else None
    sample_small = (wins + losses) < 30

    return PerformanceSummary(
        symbol=symbol,
        horizon=horizon,
        total_signals=total,
        settled_signals=total,
        pending_signals=0,
        wins=wins,
        losses=losses,
        flat=flat,
        void=void,
        observed_win_rate=win_rate,
        sample_too_small=sample_small,
    )
