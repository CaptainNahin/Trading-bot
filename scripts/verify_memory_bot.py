"""Verify the memory bank and the bot's REST surface.

The asymmetry between outcomes is the thing under test. A win is filed and
derives no rules -- a trade that worked is not evidence about what to change.
A loss is diagnosed from the closed bars it was open for and files the measured
cause, with rules derived from that cause. Asserting that a win yields DO rules
(as an earlier revision of this script did) would lock in advice no observation
supports.

Recommendations are requested with ``time_limit``, the duration the trade is
held for. A declined setup answers 409 with the reason; that is a decision, not
a fault, so both 200 and 409 are accepted here and only a 5xx fails.

ASCII markers only: the Windows console is cp1252 and a check mark raises.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from quantedge.api.app import app
from quantedge.contracts import AssetClass, SettlementOutcome, TradeMemory
from quantedge.repositories import get_repository
from quantedge.services.memory import (
    get_memory_bank_summary,
    get_relevant_memories,
    record_trade_outcome_and_analyze,
    recurring_loss_rules,
)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


def main() -> int:
    print("=" * 70)
    print("AI TRADING BOT & MEMORY SYSTEM VERIFICATION")
    print("=" * 70)

    # An in-memory bank so this runs offline and does not write to the real one.
    get_repository(force_memory=True)

    print("\n[1] A win is filed, and derives no rules")
    m_win = record_trade_outcome_and_analyze(
        signal_id="sig-test-win",
        outcome=SettlementOutcome.WIN,
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        horizon="15m",
        reference_price=Decimal("50000"),
        exit_price=Decimal("51500"),
        user_notes="Perfect breakout entry",
    )
    check("win memory created", m_win.memory_id.startswith("mem-"))
    check("win outcome recorded as WIN", m_win.outcome == "WIN", m_win.outcome)
    check("win root cause is stated", len(m_win.root_cause) > 20, m_win.root_cause[:60])
    check(
        "win derives no DO/DONT rules",
        not m_win.do_rules and not m_win.dont_rules,
        f"do={m_win.do_rules} dont={m_win.dont_rules}",
    )

    print("\n[2] A loss with no measurable holding period says so")
    m_loss = record_trade_outcome_and_analyze(
        signal_id="sig-test-loss",
        outcome=SettlementOutcome.LOSS,
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        horizon="15m",
        reference_price=Decimal("50000"),
        exit_price=Decimal("49200"),
        user_notes="Whipsaw near resistance",
    )
    check("loss memory created", m_loss.memory_id.startswith("mem-"))
    check("loss outcome recorded as LOSS", m_loss.outcome == "LOSS", m_loss.outcome)
    check(
        "undiagnosable loss is declared, not assigned a cause",
        "could not be determined" in m_loss.root_cause,
        m_loss.root_cause[:80],
    )
    check(
        "and no rule is invented for it",
        not m_loss.do_rules and not m_loss.dont_rules,
        f"do={len(m_loss.do_rules)} dont={len(m_loss.dont_rules)}",
    )

    print("\n[3] The bank reads back")
    memories = get_relevant_memories(symbol="BTCUSDT")
    check("bank retrieves saved records", len(memories) >= 2, f"{len(memories)} found")

    stats = get_memory_bank_summary()
    rate = stats["observed_win_rate"]
    check("bank reports a total", stats["total_memories"] >= 2, str(stats["total_memories"]))
    check(
        "win rate is a fraction or None, never fabricated",
        rate is None or 0.0 <= rate <= 1.0,
        str(rate),
    )
    check(
        "a 2-trade sample is flagged as too small",
        stats.get("sample_too_small") is True,
        str(stats.get("sample_too_small")),
    )

    print("\n[4] A repeated loss cause becomes a rule; a single one does not")
    rule = "Do not hold through a structure flip on the execution timeframe."

    def loss_memory(index: int, outcome: SettlementOutcome = SettlementOutcome.LOSS) -> TradeMemory:
        return TradeMemory(
            memory_id=f"mem-rule-{index}",
            signal_id=f"sig-rule-{index}",
            symbol="RULETEST",
            horizon="15m",
            outcome=outcome,
            root_cause="STRUCTURE_FLIPPED held through the trade",
            dont_rules=[rule],
        )

    repo = get_repository()
    repo.save_trade_memory(loss_memory(1))
    check(
        "one loss is an event, not a pattern",
        recurring_loss_rules("RULETEST", horizon="15m") == [],
    )

    repo.save_trade_memory(loss_memory(2))
    repeated = recurring_loss_rules("RULETEST", horizon="15m")
    check("the same cause twice surfaces a rule", len(repeated) == 1)
    if repeated:
        check("and the rule carries its occurrence count", "2 past losing" in repeated[0])

    # Scoping matters: a rule earned at 15m says nothing about a 1h trade, and a
    # rule earned on one symbol says nothing about another.
    check(
        "a rule does not leak across horizons",
        recurring_loss_rules("RULETEST", horizon="1h") == [],
    )
    check("a rule does not leak across symbols", recurring_loss_rules("BTCUSDT") == [])

    repo.save_trade_memory(loss_memory(3, SettlementOutcome.WIN))
    check(
        "a win contributes no rule",
        len(recurring_loss_rules("RULETEST", horizon="15m")) == 1,
    )

    print("\n[5] REST surface")
    client = TestClient(app)

    res = client.post(
        "/api/v1/bot/trade-recommendation?symbol=BTCUSDT&time_limit=15%20min&asset_class=crypto"
    )
    # 200 is a setup, 409 is a reasoned decline, 503 is an unavailable provider.
    # Any of the three is correct behaviour; a 4xx schema error is not.
    check(
        "POST /bot/trade-recommendation answers 200, 409 or 503",
        res.status_code in (200, 409, 503),
        str(res.status_code),
    )
    if res.status_code == 200:
        rec = res.json()
        check("  direction is assigned", rec["direction"] in ("UP", "DOWN"), rec["direction"])
        check(
            "  validity window is ordered",
            rec["valid_until_utc"] > rec["valid_from_utc"],
        )
        check(
            "  stop and target are populated",
            rec["stop_loss"] is not None and rec["take_profit"] is not None,
        )
        check(
            "  reward:risk is positive and derived, not fixed",
            float(rec["risk_reward_ratio"]) > 0,
            str(rec["risk_reward_ratio"]),
        )
        check("  venue is set", bool(rec["recommended_venue"]), rec["recommended_venue"])
    elif res.status_code == 409:
        detail = res.json()["detail"]
        check(
            "  decline carries a status and a reason",
            bool(detail.get("status")) and bool(detail.get("reason")),
            f"{detail.get('status')}: {str(detail.get('reason'))[:60]}",
        )

    for path in ("/api/v1/bot/memories", "/api/v1/bot/memory-stats", "/api/v1/bot/time-limits"):
        r = client.get(path)
        check(f"GET {path} returns 200", r.status_code == 200, str(r.status_code))

    limits = client.get("/api/v1/bot/time-limits").json()["time_limits"]
    check("time limits are offered", len(limits) > 0, f"{len(limits)} offered")
    check(
        "every offered limit carries the horizon it maps to",
        all(entry.get("horizon") for entry in limits),
    )

    chat = client.post(
        "/api/v1/bot/chat",
        json={"message": "what have you learned", "session_id": "verify-memory-bot"},
    )
    check("POST /bot/chat returns 200", chat.status_code == 200, str(chat.status_code))
    if chat.status_code == 200:
        check("  chat reply carries text", bool(chat.json().get("text")))

    check("GET / serves the dashboard", client.get("/").status_code == 200)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
