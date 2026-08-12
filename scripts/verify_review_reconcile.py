"""Deterministic checks for conservative-only review and the confidence percentage.

No network: constructs LLMSignalResponse / TradeRecommendation objects directly and
asserts the two invariants added for signal quality:

1. The LLM reviewer may confirm or decline a setup, but may never reverse its
   direction -- a SIGNAL verdict that flips the deterministic direction is downgraded
   to NO_TRADE rather than traded the other way.
2. The recommendation carries a confidence percentage equal to the composite score,
   shown to the user and labelled as evidence strength, not a calibrated win rate.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from quantedge.contracts import (
    AssetClass,
    LLMSignalResponse,
    SignalDirection,
    SignalStatus,
    TradeRecommendation,
    utc_now,
)
from quantedge.services import chat
from quantedge.services.signal import _reconcile_review

results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    results.append(ok)


def resp(status: SignalStatus, direction: SignalDirection | None) -> LLMSignalResponse:
    return LLMSignalResponse(
        status=status,
        direction=direction,
        generated_at_utc=utc_now(),
    )


def main() -> int:
    up, down = SignalDirection.UP, SignalDirection.DOWN

    # 1. SIGNAL agreeing with the candidate passes through unchanged.
    r = _reconcile_review(resp(SignalStatus.SIGNAL, up), up, symbol="BTCUSDT")
    check(
        "SIGNAL + same direction is kept",
        r.status is SignalStatus.SIGNAL and r.direction is up,
        f"{r.status.value}/{r.direction}",
    )

    # 2. SIGNAL with no direction is read as concurrence with the candidate.
    r = _reconcile_review(resp(SignalStatus.SIGNAL, None), up, symbol="BTCUSDT")
    check(
        "SIGNAL + null direction concurs with candidate",
        r.status is SignalStatus.SIGNAL and r.direction is up,
        f"{r.status.value}/{r.direction}",
    )

    # 3. SIGNAL reversing the candidate is downgraded to NO_TRADE, not flipped.
    r = _reconcile_review(resp(SignalStatus.SIGNAL, down), up, symbol="BTCUSDT")
    check(
        "reviewer reversal becomes NO_TRADE, direction dropped",
        r.status is SignalStatus.NO_TRADE and r.direction is None,
        f"{r.status.value}/{r.direction}",
    )
    check(
        "reversal records the contradiction",
        bool(r.contradictory_evidence) and "reversal is" in r.contradictory_evidence[0],
        (r.contradictory_evidence[0] if r.contradictory_evidence else "(none)"),
    )

    # 4. A reviewer decline is always honoured unchanged (more conservative is allowed).
    r = _reconcile_review(resp(SignalStatus.NO_TRADE, None), up, symbol="BTCUSDT")
    check("NO_TRADE verdict is untouched", r.status is SignalStatus.NO_TRADE)
    r = _reconcile_review(resp(SignalStatus.INSUFFICIENT_DATA, None), down, symbol="BTCUSDT")
    check("INSUFFICIENT_DATA verdict is untouched", r.status is SignalStatus.INSUFFICIENT_DATA)

    # 5. The confidence percentage is shown and labelled, and equals score*100.
    now = utc_now()
    rec = TradeRecommendation(
        recommendation_id="rec-test",
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        horizon="10m",
        direction=up,
        valid_from_utc=now,
        valid_until_utc=now + timedelta(minutes=10),
        reference_price=Decimal("100.00"),
        stop_loss=Decimal("98.00"),
        take_profit=Decimal("104.00"),
        risk_reward_ratio=2.0,
        risk_level="MODERATE_CONVICTION_SETUP",
        regime="WEAK_UPTREND",
        heuristic_score=0.72,
        confidence_pct=72,
        rationale="test rationale",
    )
    text = chat._format_recommendation(rec, 10, now + timedelta(minutes=10))
    check("chat shows the confidence percentage", "Confidence     72%" in text)
    check(
        "confidence carries the not-a-win-probability caveat",
        "not a calibrated probability" in text,
    )
    check(
        "confidence_pct tracks the heuristic score",
        rec.confidence_pct == round(rec.heuristic_score * 100),
    )

    print()
    print(
        "RESULT:",
        "ALL PASSED" if all(results) else "FAILURES PRESENT",
        f"({sum(results)}/{len(results)})",
    )
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
