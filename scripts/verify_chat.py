"""Verify the chat layer: intent parsing, and the win/loss memory asymmetry.

Two things are proved here.

1. Intent classification is deterministic and cannot be talked out of itself.
   "what's your win rate" must not be read as a report that a trade won -- that
   would write a fabricated WIN into the memory bank.

2. The outcome path does what was asked for: a win is filed with no rules
   derived, and a loss is *diagnosed first* from the closed bars between entry
   and expiry, then filed with the measured cause.

Candles are synthetic and generated here, so this runs with no network and no
provider credential. They are shaped by construction, which is what lets the
expected diagnosis be asserted rather than eyeballed.

ASCII markers only: the Windows console is cp1252 and a check mark raises.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quantedge.contracts import AssetClass, Candle, CandleSeries, Timeframe
from quantedge.repositories import get_repository
from quantedge.services import chat

ENTRY = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
BARS_BEFORE = 60


def bar(i: int, o: float, h: float, low: float, c: float) -> Candle:
    """One closed 5m candle, indexed from BARS_BEFORE bars before entry."""
    open_time = ENTRY + timedelta(minutes=5 * (i - BARS_BEFORE))
    return Candle(
        provider="synthetic",
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M5,
        open_time_utc=open_time,
        close_time_utc=open_time + timedelta(minutes=5),
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(low)),
        close=Decimal(str(c)),
        volume=Decimal("100"),
        is_closed=True,
    )


def approach() -> list[Candle]:
    """Sixty bars drifting up into entry, so an ATR and a structure exist."""
    out = []
    for i in range(BARS_BEFORE):
        base = 97.0 + i * 0.05
        out.append(bar(i, base, base + 0.35, base - 0.30, base + 0.05))
    return out


def series_for(holding: list[Candle]) -> CandleSeries:
    return CandleSeries(
        provider="synthetic",
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M5,
        candles=[*approach(), *holding],
    )


class StubRegistry:
    """Stands in for the provider registry, returning the shaped series."""

    def __init__(self, series: CandleSeries) -> None:
        self._series = series

    def get_candles(self, symbol: str, timeframe: object, limit: int = 100) -> CandleSeries:
        return self._series


def session_for(direction: str = "UP") -> dict[str, object]:
    """The session payload _handle_signal stores, built by hand."""
    return {
        "last_recommendation": {
            "recommendation_id": "rec-verifychat01",
            "symbol": "BTCUSDT",
            "asset_class": "crypto",
            "horizon": "5m",
            "direction": direction,
            "reference_price": "100.00",
            "stop_loss": "98.00",
            "take_profit": "104.00",
            "regime": "WEAK_UPTREND",
            "risk_level": "MODERATE_CONVICTION_SETUP",
            "valid_from_utc": ENTRY.isoformat(),
            "valid_until_utc": (ENTRY + timedelta(minutes=15)).isoformat(),
            "expiry_utc": (ENTRY + timedelta(minutes=15)).isoformat(),
            "time_limit_minutes": 15,
        }
    }


def install_stub(holding: list[Candle]) -> None:
    """Point the holding-period fetch at synthetic bars, not a live provider."""
    from quantedge.providers import registry as reg

    stub = StubRegistry(series_for(holding))
    reg.get_registry = lambda: stub  # type: ignore[assignment]


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    return ok


def main() -> int:
    results: list[bool] = []

    # ---------------- intent parsing ----------------
    cases = [
        ("give me a signal on BTC 10 min", "SIGNAL", "BTCUSDT", 10),
        ("BTC 20 minutes", "SIGNAL", "BTCUSDT", 20),
        ("analyse gold for 1 hour", "SIGNAL", "XAUUSD", 60),
        ("my BTC 10 min trade lost", "REPORT_OUTCOME", "BTCUSDT", 10),
        ("that one won", "REPORT_OUTCOME", None, None),
        ("stopped out", "REPORT_OUTCOME", None, None),
        ("what's your win rate", "PERFORMANCE", None, None),
        ("what have you learned", "MEMORY", None, None),
        ("what time limits do you have", "TIME_LIMITS", None, None),
        ("status", "STATUS", None, None),
        ("help", "HELP", None, None),
        ("blah blah nothing", "UNKNOWN", None, None),
    ]
    for message, intent, symbol, minutes in cases:
        p = chat.parse_intent(message)
        ok = p.intent.value == intent and p.symbol == symbol and p.minutes == minutes
        results.append(
            check(
                f"parse {message!r}",
                ok,
                f"{p.intent.value} sym={p.symbol} min={p.minutes}",
            )
        )

    # A time limit must not be read out of a bare price in the sentence.
    p = chat.parse_intent("BTC is at 65000 right now")
    results.append(check("bare price is not a duration", p.minutes is None, f"min={p.minutes}"))

    # ---------------- outcome: no trade to attach to ----------------
    reply = chat.handle_message("that one lost", session={})
    results.append(
        check(
            "loss with no open trade is refused, not invented",
            reply.data.get("recorded") is False,
            reply.data.get("outcome", "?"),
        )
    )

    # ---------------- WIN: filed, no rules derived ----------------
    get_repository(force_memory=True)
    install_stub([])
    state = session_for()
    reply = chat.handle_message("that one won", session=state)
    memory = reply.data.get("memory", {})
    results.append(
        check(
            "win recorded",
            reply.data.get("recorded") is True and memory.get("outcome") == "WIN",
            memory.get("memory_id", "none"),
        )
    )
    results.append(
        check(
            "win derives no DO/DONT rules",
            not memory.get("do_rules") and not memory.get("dont_rules"),
            f"do={memory.get('do_rules')} dont={memory.get('dont_rules')}",
        )
    )
    results.append(
        check(
            "settled trade is cleared from the session",
            "last_recommendation" not in state,
        )
    )

    # ---------------- LOSS: stop run, then recovery ----------------
    stop_run = [
        bar(BARS_BEFORE, 100.0, 100.5, 99.0, 99.5),
        bar(BARS_BEFORE + 1, 99.5, 99.8, 97.5, 98.2),
        bar(BARS_BEFORE + 2, 98.2, 101.2, 98.0, 101.0),
    ]
    install_stub(stop_run)
    state = session_for()
    reply = chat.handle_message("that one lost", session=state)
    memory = reply.data.get("memory", {})
    root = memory.get("root_cause", "")
    results.append(
        check(
            "loss diagnosed from the holding period",
            reply.data.get("bars_measured") == 3,
            f"bars={reply.data.get('bars_measured')}",
        )
    )
    results.append(
        check(
            "stop-run loss names the tight stop, with numbers",
            "the stop sat inside the range" in root and "2.50" in root,
        )
    )
    results.append(
        check(
            "loss derives a rule from the measured cause",
            bool(memory.get("do_rules") or memory.get("dont_rules")),
            f"{len(memory.get('do_rules', []))} do / {len(memory.get('dont_rules', []))} dont",
        )
    )

    # ---------------- LOSS: expired unresolved, a different diagnosis ----------------
    chop = [
        bar(BARS_BEFORE, 100.0, 100.3, 99.7, 100.1),
        bar(BARS_BEFORE + 1, 100.1, 100.4, 99.8, 99.9),
        bar(BARS_BEFORE + 2, 99.9, 100.2, 99.6, 99.8),
    ]
    install_stub(chop)
    reply2 = chat.handle_message("it lost", session=session_for())
    root2 = reply2.data.get("memory", {}).get("root_cause", "")
    results.append(
        check(
            "a differently-shaped loss gets a different diagnosis",
            root2 != root and "time limit expired" in root2,
        )
    )

    # ---------------- forming bar must not decide the stop ----------------
    forming = bar(BARS_BEFORE + 3, 99.8, 99.9, 90.0, 95.0).model_copy(update={"is_closed": False})
    install_stub([*chop, forming])
    reply3 = chat.handle_message("that lost", session=session_for())
    results.append(
        check(
            "forming bar excluded from the diagnosis",
            reply3.data.get("bars_measured") == 3,
            f"bars={reply3.data.get('bars_measured')}",
        )
    )

    # ---------------- bars after expiry must not be diagnosed ----------------
    after_expiry = [
        *chop,
        bar(BARS_BEFORE + 3, 99.8, 99.9, 90.0, 91.0),  # closes 20 min after entry
    ]
    install_stub(after_expiry)
    reply4 = chat.handle_message("that lost", session=session_for())
    results.append(
        check(
            "bars past expiry excluded from the diagnosis",
            reply4.data.get("bars_measured") == 3,
            f"bars={reply4.data.get('bars_measured')}",
        )
    )

    # ---------------- memory and performance read back ----------------
    reply5 = chat.handle_message("what have you learned")
    results.append(
        check(
            "memory review lists the recorded trades",
            "recorded trade" in reply5.text,
        )
    )
    reply6 = chat.handle_message("how are you doing")
    results.append(
        check(
            "performance states the sample is too small",
            "not computable" in reply6.text or "sample" in reply6.text,
        )
    )

    print("\n--- sample loss diagnosis (stop run then recovery) ---")
    install_stub(stop_run)
    print(chat.handle_message("that one lost", session=session_for()).text)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
