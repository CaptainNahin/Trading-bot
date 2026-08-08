"""Background signal settlement worker.

Scores signals whose expiry has passed against the closed bars that followed
them, and writes each result to the immutable ``settled_signals`` table.

An earlier revision only read the performance summary and logged it, so nothing
was ever settled: the summary it printed was of rows some other code path had
written, and a deployment running this worker alone accumulated expired signals
that were never scored.

Two properties make it safe to run on a timer:

* **Idempotent.** Candidates come from an anti-join against ``settled_signals``,
  and that table refuses a second write for the same signal. A restart cannot
  re-score a closed trade.
* **No look-ahead.** Only bars that closed at or before the signal's expiry are
  passed to the scorer, so a signal is judged on the window it specified rather
  than on everything available by the time the worker happened to run.
"""

from __future__ import annotations

import time

from quantedge.contracts import Timeframe
from quantedge.errors import PersistenceError, QuantEdgeError
from quantedge.logging import get_logger
from quantedge.providers.registry import get_registry
from quantedge.repositories import get_repository
from quantedge.services import settlement

logger = get_logger(__name__)

# Bars fine enough to place the expiry accurately without asking for a year of
# 1m history on a long horizon.
_SETTLEMENT_TIMEFRAME = Timeframe.M1
_MAX_BARS = 1000


def settle_due_signals(*, limit: int = 100) -> int:
    """Settle every expired signal that has no settlement row. Returns the count."""
    repo = get_repository()
    registry = get_registry()

    try:
        due = repo.unsettled_expired_signals(limit=limit)
    except QuantEdgeError as exc:
        logger.error("could not read due signals", extra={"code": exc.code})
        return 0

    settled_count = 0
    for decision in due:
        if decision.expiry_utc is None:
            continue
        try:
            series = registry.get_candles(decision.symbol, _SETTLEMENT_TIMEFRAME, limit=_MAX_BARS)
        except QuantEdgeError as exc:
            logger.warning(
                "no bars to settle against",
                extra={"symbol": decision.symbol, "code": exc.code},
            )
            continue

        # Closed bars up to the expiry only. A forming bar has no final close,
        # and a bar that closed after expiry is information the trade never had.
        window = [
            c for c in series.candles if c.is_closed and c.close_time_utc <= decision.expiry_utc
        ]
        if not window:
            logger.warning(
                "expired signal has no closed bars in its window; leaving unsettled",
                extra={"signal_id": decision.decision_id, "symbol": decision.symbol},
            )
            continue

        try:
            result = settlement.settle_decision(decision, window)
        except PersistenceError as exc:
            logger.warning(
                "settlement refused",
                extra={"signal_id": decision.decision_id, "error": exc.message},
            )
            continue

        if result is not None:
            settled_count += 1
            logger.info(
                "signal settled",
                extra={
                    "signal_id": result.signal_id,
                    "symbol": result.symbol,
                    "outcome": result.outcome.value,
                    "bars": len(window),
                },
            )

    return settled_count


def run_settlement_worker(interval_seconds: int = 300) -> None:
    """Periodically settle expired signals."""
    logger.info("starting settlement worker", extra={"interval_seconds": interval_seconds})

    while True:
        try:
            count = settle_due_signals()
            summary = settlement.get_performance_summary()
            logger.info(
                "settlement pass complete",
                extra={
                    "settled_now": count,
                    "total": summary.total_signals,
                    "wins": summary.wins,
                    "losses": summary.losses,
                },
            )
        except Exception as exc:  # noqa: BLE001 - the loop must survive one bad pass
            logger.error("settlement worker error", extra={"error": type(exc).__name__})
        time.sleep(interval_seconds)


if __name__ == "__main__":
    run_settlement_worker()
