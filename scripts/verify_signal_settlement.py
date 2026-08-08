"""Signal and Settlement verification script."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge.contracts import (
    AIDecision,
    AssetClass,
    Candle,
    SettlementOutcome,
    SignalDirection,
    SignalStatus,
    Timeframe,
)
from quantedge.errors import PersistenceError
from quantedge.repositories import get_repository
from quantedge.services.settlement import get_performance_summary, settle_decision

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


START = datetime(2026, 1, 1, tzinfo=UTC)


def main() -> int:
    print("=" * 70)
    print("SIGNAL & SETTLEMENT VERIFICATION -- outcome calculation checks")
    print("=" * 70)

    # An in-memory bank, so the run is repeatable and does not write test rows
    # into the real settled table. The settled table is append-only, so writing
    # there made this script pass exactly once and fail on every rerun.
    get_repository(force_memory=True)

    decision = AIDecision(
        decision_id="test-dec-123",
        symbol="BTCUSDT",
        horizon="swing",
        status=SignalStatus.SIGNAL,
        direction=SignalDirection.UP,
        reference_price=Decimal("50000"),
        # The expiry bounds the window the trade is scored over, so it is part of
        # the signal rather than a detail of when settlement ran. Without one,
        # settlement declines instead of scoring against an arbitrary "now".
        expiry_utc=START + timedelta(minutes=30),
    )

    closed_winning = [
        Candle(
            provider="mock",
            symbol="BTCUSDT",
            asset_class=AssetClass.CRYPTO,
            timeframe=Timeframe.M5,
            open_time_utc=START,
            close_time_utc=START + timedelta(minutes=5),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49900"),
            close=Decimal("50500"),
            volume=Decimal("100"),
            is_closed=True,
        )
    ]

    settled = settle_decision(decision, closed_winning)
    check("Settlement object created", settled is not None)
    if settled:
        check("UP direction with higher close is WIN", settled.outcome == SettlementOutcome.WIN)
        check(
            "Settlement price matches last candle close",
            settled.settlement_price == Decimal("50500"),
        )

    summary = get_performance_summary(symbol="BTCUSDT")
    check("Performance summary retrieves total settled", summary.settled_signals >= 1)

    # A signal with no expiry has no window to be scored over. Settling it
    # against the moment the job happened to run would make the outcome a
    # property of the scheduler rather than of the trade.
    no_expiry = decision.model_copy(
        update={"decision_id": "test-dec-no-expiry", "expiry_utc": None}
    )
    check(
        "A signal without an expiry is not settled",
        settle_decision(no_expiry, closed_winning) is None,
    )

    # The settled table is immutable. A repeat settlement must surface as a
    # refused write, not return a record that was never stored.
    try:
        settle_decision(decision, closed_winning)
    except PersistenceError as exc:
        check("Re-settling the same signal is refused", True, exc.message[:60])
    else:
        check("Re-settling the same signal is refused", False, "the duplicate write was accepted")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
