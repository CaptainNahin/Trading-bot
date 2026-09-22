"""Autonomous Signal Lifecycle Monitoring & Closed-Loop Memory Engine.

Monitors active in-flight trading signals and manages their full lifecycle:
1. Detects Take Profit (TP) hits during the holding window (resolving as early WIN).
2. Detects Stop Loss (SL) breaches during the holding window (resolving as early LOSS).
3. Detects expiry timestamp arrival (resolving at market close price).
4. Scores and settles the signal immutably into ``settled_signals``.
5. Feeds the holding period into the Memory Bank (``quantedge.services.memory``):
   - For a WIN: records positive setup metrics.
   - For a LOSS: runs algorithmic post-mortem diagnostics AND the ZXL AI Brain
     (ZLM 5.3 via Seek AI) deep contextual post-mortem to extract actionable
     DO and DON'T rules.
   - The derived rules immediately guard future scans on this symbol/horizon.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence

from quantedge.contracts import (
    AIDecision,
    Candle,
    SettledSignal,
    SettlementOutcome,
    SignalDirection,
    SignalStatus,
    Timeframe,
    utc_now,
)
from quantedge.errors import PersistenceError, QuantEdgeError
from quantedge.logging import get_logger
from quantedge.providers.registry import get_registry
from quantedge.repositories import get_repository
from quantedge.services import memory, settlement
from quantedge.services.signal import _risk_levels_for

__all__ = [
    "get_active_signals_summary",
    "monitor_and_settle_active_signals",
]

log = get_logger(__name__)

_DEFAULT_TIMEFRAME = Timeframe.M1
_MAX_BARS = 500


def monitor_and_settle_active_signals(
    *,
    limit: int = 50,
    candle_fetcher: Any = None,
    trigger_ai_postmortem: bool = False,
) -> dict[str, Any]:
    """Inspect all open signals, check TP/SL hits or expiry, settle and update memory.

    Returns
    -------
    dict[str, Any]
        Summary containing:
        - ``checked``: Total open signals inspected
        - ``active_in_flight``: Count of signals still open and running
        - ``settled_now``: Count of signals resolved in this pass
        - ``settled_details``: List of signals settled in this pass
        - ``active_details``: List of ongoing signals with live status
    """
    repo = get_repository()
    registry = get_registry()

    try:
        candidates = repo.unsettled_signals(include_unexpired=True, limit=limit)
    except QuantEdgeError as exc:
        log.error("could not retrieve open signals", extra={"code": exc.code})
        return {
            "checked": 0,
            "active_in_flight": 0,
            "settled_now": 0,
            "settled_details": [],
            "active_details": [],
            "error": exc.message,
        }

    now = utc_now()
    settled_records: list[dict[str, Any]] = []
    active_records: list[dict[str, Any]] = []

    for decision in candidates:
        if (
            decision.status != SignalStatus.SIGNAL
            or not decision.direction
            or not decision.reference_price
            or not decision.expiry_utc
        ):
            continue

        symbol = decision.symbol
        horizon = decision.horizon
        ref_price = decision.reference_price
        direction = decision.direction
        signal_id = decision.decision_id or "unknown"
        entry_time = decision.created_at_utc

        # For non-crypto assets (forex, commodities, metals), use TradingView directly
        is_crypto = symbol.endswith("USDT") or symbol.endswith("BUSD") or symbol in (
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"
        )
        if not is_crypto:
            try:
                from quantedge.services.tradingview import get_tradingview_analysis

                tv = get_tradingview_analysis(symbol, timeframe="1m")
                cur_price = tv.get("price")
                if cur_price:
                    p_dec = Decimal(str(cur_price))
                    now_ts = utc_now()

                    class _LiveBar:
                        def __init__(self, p: Decimal, ts: Any):
                            self.open = p
                            self.high = p
                            self.low = p
                            self.close = p
                            self.is_closed = True
                            self.close_time_utc = ts

                    series = type("Series", (), {"candles": [_LiveBar(p_dec, now_ts)]})()
                else:
                    continue
            except Exception as tv_err:
                log.warning(
                    "no candles available to monitor signal",
                    extra={"symbol": symbol, "signal_id": signal_id, "error": str(tv_err)},
                )
                continue
        else:
            try:
                if candle_fetcher is not None:
                    series = candle_fetcher(symbol, _DEFAULT_TIMEFRAME)
                else:
                    series = registry.get_candles(symbol, _DEFAULT_TIMEFRAME, limit=_MAX_BARS)
            except Exception as exc:
                continue

        # Closed candles at or after entry time
        window = [
            c
            for c in series.candles
            if c.is_closed and (entry_time is None or c.close_time_utc >= entry_time)
        ]
        if not window:
            # If no bars strictly >= entry_time, take the latest closed bars
            window = [c for c in series.candles if c.is_closed]

        if not window:
            continue

        # 2. Derive or estimate risk levels (Stop Loss and Take Profit)
        stop: Decimal | None = None
        target: Decimal | None = None
        if is_crypto:
            try:
                levels = _risk_levels_for(symbol, horizon, direction, ref_price, candle_fetcher)
                if levels is not None:
                    stop = levels.stop
                    target = levels.target
            except Exception:
                pass

        # Fallback levels if ATR calculation failed
        if stop is None or target is None:
            pct_stop = Decimal("0.015")
            pct_target = Decimal("0.030")
            if direction == SignalDirection.UP:
                stop = ref_price * (Decimal("1.0") - pct_stop)
                target = ref_price * (Decimal("1.0") + pct_target)
            else:
                stop = ref_price * (Decimal("1.0") + pct_stop)
                target = ref_price * (Decimal("1.0") - pct_target)

        # 3. Scan bars chronologically for TP / SL hit
        resolved = False
        outcome: SettlementOutcome = SettlementOutcome.FLAT
        exit_price = window[-1].close
        reason = ""

        for bar in window:
            if direction == SignalDirection.UP:
                if bar.high >= target:
                    resolved = True
                    outcome = SettlementOutcome.WIN
                    exit_price = target
                    reason = f"Take Profit reached at {target} (high {bar.high})"
                    break
                if bar.low <= stop:
                    resolved = True
                    outcome = SettlementOutcome.LOSS
                    exit_price = stop
                    reason = f"Stop Loss breached at {stop} (low {bar.low})"
                    break
            elif direction == SignalDirection.DOWN:
                if bar.low <= target:
                    resolved = True
                    outcome = SettlementOutcome.WIN
                    exit_price = target
                    reason = f"Take Profit reached at {target} (low {bar.low})"
                    break
                if bar.high >= stop:
                    resolved = True
                    outcome = SettlementOutcome.LOSS
                    exit_price = stop
                    reason = f"Stop Loss breached at {stop} (high {bar.high})"
                    break

        # 4. If neither TP nor SL was hit, check for expiry
        if not resolved:
            if now >= decision.expiry_utc:
                resolved = True
                exit_price = window[-1].close
                if direction == SignalDirection.UP:
                    if exit_price > ref_price:
                        outcome = SettlementOutcome.WIN
                    elif exit_price < ref_price:
                        outcome = SettlementOutcome.LOSS
                    else:
                        outcome = SettlementOutcome.FLAT
                elif direction == SignalDirection.DOWN:
                    if exit_price < ref_price:
                        outcome = SettlementOutcome.WIN
                    elif exit_price > ref_price:
                        outcome = SettlementOutcome.LOSS
                    else:
                        outcome = SettlementOutcome.FLAT
                reason = f"Expiry reached at {decision.expiry_utc}; settled at close {exit_price}"
            else:
                # Still running!
                curr_price = window[-1].close
                if direction == SignalDirection.UP:
                    pnl_pct = float((curr_price - ref_price) / ref_price * 100)
                else:
                    pnl_pct = float((ref_price - curr_price) / ref_price * 100)

                remaining_sec = max(0, int((decision.expiry_utc - now).total_seconds()))
                active_records.append(
                    {
                        "signal_id": signal_id,
                        "symbol": symbol,
                        "horizon": horizon,
                        "direction": direction.value,
                        "reference_price": float(ref_price),
                        "current_price": float(curr_price),
                        "stop_loss": float(stop),
                        "take_profit": float(target),
                        "unrealized_pnl_pct": round(pnl_pct, 2),
                        "remaining_minutes": int(remaining_sec // 60),
                        "status": "RUNNING",
                    }
                )

        # 5. Settle and record into Memory Bank if resolved
        if resolved:
            try:
                settled = settlement.settle_decision(
                    decision,
                    window,
                    settlement_provider=getattr(series, "provider", "binance"),
                    outcome=outcome,
                    settlement_price=exit_price,
                    notes=[reason],
                )
            except PersistenceError as exc:
                log.warning(
                    "lifecycle settlement refused",
                    extra={"signal_id": signal_id, "error": exc.message},
                )
                continue

            if settled is not None:
                # Record in Memory Bank (triggers AI Brain post-mortem on LOSS)
                try:
                    memory.record_trade_outcome_and_analyze(
                        signal_id=signal_id,
                        outcome=outcome,
                        symbol=symbol,
                        horizon=horizon,
                        direction=direction,
                        reference_price=ref_price,
                        exit_price=exit_price,
                        stop=stop,
                        target=target,
                        holding_candles=window,
                        entry_time=entry_time,
                        trigger_ai_postmortem=trigger_ai_postmortem,
                    )
                except Exception as mem_exc:
                    log.warning(
                        "trade memory recording failed during lifecycle settlement",
                        extra={"signal_id": signal_id, "error": str(mem_exc)},
                    )

                settled_records.append(
                    {
                        "signal_id": signal_id,
                        "symbol": symbol,
                        "direction": direction.value,
                        "outcome": outcome.value,
                        "reference_price": float(ref_price),
                        "exit_price": float(exit_price),
                        "reason": reason,
                    }
                )
                log.info(
                    "autonomous lifecycle: trade resolved and settled",
                    extra={
                        "signal_id": signal_id,
                        "symbol": symbol,
                        "outcome": outcome.value,
                        "exit_price": float(exit_price),
                        "reason": reason,
                    },
                )

    return {
        "checked": len(candidates),
        "active_in_flight": len(active_records),
        "settled_now": len(settled_records),
        "settled_details": settled_records,
        "active_details": active_records,
    }


def get_active_signals_summary() -> dict[str, Any]:
    """Quick read-only query of all currently open active signals."""
    repo = get_repository()
    candidates = repo.unsettled_signals(include_unexpired=True, limit=50)
    now = utc_now()
    active: list[dict[str, Any]] = []

    for d in candidates:
        if d.expiry_utc and d.expiry_utc > now and d.reference_price:
            remaining_min = int((d.expiry_utc - now).total_seconds() // 60)
            active.append(
                {
                    "signal_id": d.decision_id,
                    "symbol": d.symbol,
                    "direction": d.direction.value if d.direction else None,
                    "reference_price": float(d.reference_price),
                    "expiry_utc": d.expiry_utc.isoformat(),
                    "remaining_minutes": remaining_min,
                }
            )

    return {
        "total_active": len(active),
        "signals": active,
    }
