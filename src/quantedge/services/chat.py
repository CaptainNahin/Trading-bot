"""The conversational layer: intent in, grounded answer out.

How a message becomes an answer
-------------------------------
1. :func:`parse_intent` classifies the message and pulls out symbol, time limit
   and outcome **with regular expressions, in code**. No model decides what the
   user asked for, so "BTC 10 min" cannot be misread as a request for ETH.
2. The matching deterministic service runs -- scanner, risk, memory, settlement.
   Every number in the answer originates there.
3. The reply is assembled from those numbers by :mod:`quantedge.services.chat`
   itself. If a reviewer model is configured it *annotates* the answer; it never
   supplies a level, a direction or a price.

Step 3 is why the model being unreachable degrades the wording and nothing else.
A chat that fabricates a price when its LLM is down is worse than one that says
it cannot reach the model, and this ordering makes the latter the only option.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

from quantedge.contracts import (
    MarketRegime,
    SettlementOutcome,
    SignalDirection,
    SignalStatus,
    utc_now,
)
from quantedge.errors import QuantEdgeError, ValidationError
from quantedge.logging import get_logger
from quantedge.services.horizons import (
    available_time_limits,
    expiry_for,
    horizon_for_minutes,
)
from quantedge.symbols import is_supported

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "ChatIntent",
    "ChatReply",
    "Intent",
    "handle_message",
    "holding_period_for",
    "parse_intent",
]

log = get_logger(__name__)

_MAX_MESSAGE_CHARS = 2000

# Hold assumed when a symbol is named without a duration. 15m is the shortest
# horizon whose confirmation and regime timeframes (1h/4h) are slow enough to be
# meaningful, so it is the least presumptuous default rather than the fastest.
_DEFAULT_HOLD_MINUTES = 15


class Intent(str, Enum):
    """What the user is asking for."""

    SIGNAL = "SIGNAL"
    REPORT_OUTCOME = "REPORT_OUTCOME"
    MEMORY = "MEMORY"
    PERFORMANCE = "PERFORMANCE"
    TIME_LIMITS = "TIME_LIMITS"
    STATUS = "STATUS"
    HELP = "HELP"
    UNKNOWN = "UNKNOWN"


@dataclass
class ChatIntent:
    """A parsed message: what was asked, and what was named in it."""

    intent: Intent
    symbol: str | None = None
    minutes: int | None = None
    outcome: SettlementOutcome | None = None
    signal_id: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "symbol": self.symbol,
            "minutes": self.minutes,
            "outcome": self.outcome.value if self.outcome else None,
            "signal_id": self.signal_id,
        }


@dataclass
class ChatReply:
    """An answer, plus the structured payload it was built from.

    ``data`` carries the deterministic result so the UI can render levels and an
    expiry clock without re-parsing prose, and so a reader can check the sentence
    against the numbers it came from.
    """

    text: str
    intent: Intent
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    generated_at_utc: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "intent": self.intent.value,
            "data": self.data,
            "warnings": self.warnings,
            "generated_at_utc": self.generated_at_utc.isoformat(),
        }


# Words that mean "give me a trade". Matched as whole words so "signalling" in a
# sentence about something else does not trigger a scan.
_SIGNAL_WORDS = re.compile(
    r"\b(signal|setup|trade|entry|call|analy[sz]e|scan|what.?s\s+next|up\s+or\s+down)\b",
    re.IGNORECASE,
)
# "win" is excluded before "rate"/"ratio": "what's your win rate" is a question
# about the record, not a report that a trade won, and reading it as the latter
# would write a fabricated WIN into the memory bank.
_WIN_WORDS = re.compile(
    r"\b(win(?!\s*(rate|ratio))|won|winner|winning|profit|profitable|hit\s+target"
    r"|it\s+worked|successful|success)\b",
    re.IGNORECASE,
)
_LOSS_WORDS = re.compile(
    r"\b(loss|lost|lose|losing|loser|stopped\s+out|stop\s+out|failed|didn.?t\s+work|went\s+against)\b",
    re.IGNORECASE,
)
_MEMORY_WORDS = re.compile(
    r"\b(memor(y|ies)|remember|learn(ed|ing)?|lesson|past\s+trades?|history|rules?)\b",
    re.IGNORECASE,
)
_PERF_WORDS = re.compile(
    r"\b(performance|win\s*rate|track\s+record|how\s+(are|am)\s+(you|i)\s+doing|stats|statistics)\b",
    re.IGNORECASE,
)
_LIMIT_WORDS = re.compile(
    r"\b(time\s*limits?|expir(y|ies|ations?)|durations?|how\s+long|available\s+times?)\b",
    re.IGNORECASE,
)
_STATUS_WORDS = re.compile(
    r"\b(status|health|providers?|are\s+you\s+(ok|online|working)|connected)\b",
    re.IGNORECASE,
)
_HELP_WORDS = re.compile(r"\b(help|what\s+can\s+you\s+do|commands?|how\s+do\s+i)\b", re.IGNORECASE)

# "10 min", "10m", "20 minutes", "1 hour", "1h". The unit is required so a bare
# price in the sentence is not read as a duration.
_DURATION = re.compile(
    r"\b(\d{1,3})\s*(m|min|mins|minute|minutes|h|hr|hour|hours)\b", re.IGNORECASE
)

# Common shorthands the user is likely to type for a configured symbol.
_SYMBOL_ALIASES: dict[str, str] = {
    "BTC": "BTCUSDT",
    "BITCOIN": "BTCUSDT",
    "ETH": "ETHUSDT",
    "ETHEREUM": "ETHUSDT",
    "BNB": "BNBUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "DOGE": "DOGEUSDT",
    "ADA": "ADAUSDT",
    "AVAX": "AVAXUSDT",
    "GOLD": "XAUUSD",
    "SILVER": "XAGUSD",
    "OIL": "WTICOUSD",
    "EUR": "EURUSD",
    "GBP": "GBPUSD",
    "YEN": "USDJPY",
}

_SIGNAL_ID = re.compile(r"\b((?:rec|mem|sig)-[0-9a-f]{6,})\b", re.IGNORECASE)


def parse_intent(message: str) -> ChatIntent:
    """Classify a message and extract its parameters, in code.

    Deliberately not delegated to the model. A misclassified intent is not a
    wording problem: reading "my BTC trade lost" as a request for a new signal
    would skip the post-mortem the user asked for and open a position instead.
    """
    text = message.strip()[:_MAX_MESSAGE_CHARS]
    if not text:
        return ChatIntent(Intent.UNKNOWN)

    symbol = _extract_symbol(text)
    minutes = _extract_minutes(text)
    signal_id = _extract_signal_id(text)

    # Order matters: an outcome report often also names a symbol and a duration,
    # so it is tested before the signal words that such a sentence also contains
    # ("my BTC 10 min trade lost").
    if _LOSS_WORDS.search(text):
        return ChatIntent(
            Intent.REPORT_OUTCOME,
            symbol=symbol,
            minutes=minutes,
            outcome=SettlementOutcome.LOSS,
            signal_id=signal_id,
            notes=text,
        )
    if _WIN_WORDS.search(text):
        return ChatIntent(
            Intent.REPORT_OUTCOME,
            symbol=symbol,
            minutes=minutes,
            outcome=SettlementOutcome.WIN,
            signal_id=signal_id,
            notes=text,
        )
    if _PERF_WORDS.search(text):
        return ChatIntent(Intent.PERFORMANCE, symbol=symbol)
    if _MEMORY_WORDS.search(text):
        return ChatIntent(Intent.MEMORY, symbol=symbol)
    if _LIMIT_WORDS.search(text):
        return ChatIntent(Intent.TIME_LIMITS)
    if _STATUS_WORDS.search(text):
        return ChatIntent(Intent.STATUS)
    if _SIGNAL_WORDS.search(text) or (symbol is not None and minutes is not None):
        return ChatIntent(Intent.SIGNAL, symbol=symbol, minutes=minutes)
    if _HELP_WORDS.search(text):
        return ChatIntent(Intent.HELP)
    # A bare symbol is a request about that symbol; anything else is unknown.
    if symbol is not None:
        return ChatIntent(Intent.SIGNAL, symbol=symbol, minutes=minutes)
    return ChatIntent(Intent.UNKNOWN)


def _extract_symbol(text: str) -> str | None:
    """The first supported symbol named in the message.

    Checked against the symbol registry rather than pattern-matched, so an
    unsupported ticker is reported as unsupported instead of being sent to a
    provider that will reject it.
    """
    tokens = re.findall(r"[A-Za-z]{2,12}(?:USDT|USD)?", text.upper())
    for token in tokens:
        if is_supported(token):
            return token
        alias = _SYMBOL_ALIASES.get(token)
        if alias is not None and is_supported(alias):
            return alias
    return None


def _extract_minutes(text: str) -> int | None:
    """The duration named in the message, in minutes."""
    match = _DURATION.search(text)
    if match is None:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    minutes = value * 60 if unit.startswith("h") else value
    return minutes if 0 < minutes <= 1440 else None


def _extract_signal_id(text: str) -> str | None:
    match = _SIGNAL_ID.search(text)
    return match.group(1) if match else None


# ---------------------------------------------------------------------- #
# dispatch                                                              #
# ---------------------------------------------------------------------- #


def handle_message(
    message: str,
    *,
    default_symbol: str = "BTCUSDT",
    default_minutes: int | None = None,
    session: dict[str, Any] | None = None,
) -> ChatReply:
    """Answer one message.

    ``session`` carries the last recommendation so a follow-up ("that one lost")
    can be attached to the trade it refers to. It is a plain dict owned by the
    caller -- the API stores it per browser session -- because a conversation's
    working memory is not the same thing as the trade memory bank, and mixing
    them would let an unsent message pollute the record.
    """
    parsed = parse_intent(message)
    state = session if session is not None else {}

    if parsed.intent is Intent.SIGNAL:
        return _handle_signal(parsed, default_symbol, default_minutes, state)
    if parsed.intent is Intent.REPORT_OUTCOME:
        return _handle_outcome(parsed, state)
    if parsed.intent is Intent.MEMORY:
        return _handle_memory(parsed)
    if parsed.intent is Intent.PERFORMANCE:
        return _handle_performance(parsed)
    if parsed.intent is Intent.TIME_LIMITS:
        return _handle_time_limits()
    if parsed.intent is Intent.STATUS:
        return _handle_status()
    return _handle_help(parsed.intent)


def _handle_signal(
    parsed: ChatIntent,
    default_symbol: str,
    default_minutes: int | None,
    state: dict[str, Any],
) -> ChatReply:
    """Run the deterministic pipeline and report exactly what it returned."""
    from quantedge.services.horizons import horizon_minutes
    from quantedge.services.signal import (
        NoTradeReason,
        generate_best_trade_recommendation,
        generate_trade_recommendation,
    )

    assumption_note = ""
    alternatives: list[dict[str, Any]] = []

    if parsed.symbol is None:
        try:
            rec = generate_best_trade_recommendation(
                time_limit_minutes=parsed.minutes,
                alternatives_out=alternatives,
            )
        except NoTradeReason as exc:
            return _no_trade_reply("any symbol", parsed.minutes or 0, exc)
        except ValidationError as exc:
            return ChatReply(
                text=f"I can't analyse that: {exc.message}",
                intent=Intent.SIGNAL,
                data={"symbol": "ANY"},
                warnings=[exc.message],
            )
        except QuantEdgeError as exc:
            log.warning("global signal request failed", extra={"code": exc.code})
            return ChatReply(
                text=(
                    f"I couldn't complete the global analysis: {exc.message}. "
                    "No signal is being issued, because I'd be guessing."
                ),
                intent=Intent.SIGNAL,
                data={"symbol": "ANY", "error_code": exc.code},
                warnings=[exc.message],
            )

        minutes_used = parsed.minutes or horizon_minutes(rec.horizon)
        assumption_note = _alternatives_note(alternatives)
    else:
        symbol = parsed.symbol
        # Asking for a symbol is a request for that symbol. Blocking on a missing
        # duration turned "signal for BTC" into a question, so the one thing the
        # user actually named went unanswered. A duration is needed to pick the
        # timeframes, so the shortest configured horizon that is not a scalp is
        # assumed, stated in the reply, and overridden the moment one is given.
        requested = parsed.minutes or default_minutes or _DEFAULT_HOLD_MINUTES
        if parsed.minutes is None and default_minutes is None:
            assumption_note = (
                f"\n\nI assumed a {_DEFAULT_HOLD_MINUTES}-minute hold since you didn't say. "
                f"Other options: {', '.join(t.label for t in available_time_limits())}."
            )

        minutes_used = requested
        try:
            horizon = horizon_for_minutes(minutes_used)
            rec = generate_trade_recommendation(symbol, time_limit=horizon)
        except NoTradeReason as exc:
            return _no_trade_reply(symbol, minutes_used, exc, note=assumption_note)
        except ValidationError as exc:
            return ChatReply(
                text=f"I can't analyse that: {exc.message}",
                intent=Intent.SIGNAL,
                data={"symbol": symbol},
                warnings=[exc.message],
            )
        except QuantEdgeError as exc:
            log.warning("signal request failed", extra={"symbol": symbol, "code": exc.code})
            return ChatReply(
                text=(
                    f"I couldn't complete the analysis for {symbol}: {exc.message}. "
                    "No signal is being issued, because I'd be guessing."
                ),
                intent=Intent.SIGNAL,
                data={"symbol": symbol, "error_code": exc.code},
                warnings=[exc.message],
            )

    # The expiry the user is told is the duration they chose, not the horizon's.
    expiry = expiry_for(minutes_used, rec.generated_at_utc)
    payload = rec.model_dump(mode="json")
    payload["expiry_utc"] = expiry.isoformat()
    payload["time_limit_minutes"] = minutes_used
    state["last_recommendation"] = payload

    return ChatReply(
        text=_format_recommendation(rec, minutes_used, expiry) + assumption_note,
        intent=Intent.SIGNAL,
        data=payload,
        warnings=list(rec.warnings),
    )


def _alternatives_note(alternatives: list[dict[str, Any]]) -> str:
    """The rest of the board, so one top scorer does not look like the whole market.

    The sweep ranks every candidate and returns one. Showing only that one made a
    6-UP/3-DOWN board read as "the bot only ever says UP", which was a reporting
    artefact rather than a directional bias. These are scanner candidates that have
    not been through the risk gates, so they are labelled as such and no entry,
    stop or target is quoted for them.
    """
    if not alternatives:
        return ""

    # Top two of each direction rather than the top four overall. Ranking by score
    # alone listed four UPs under an "11 UP / 6 DOWN" header, which still read as a
    # one-way board even though the DOWN setups were right there. Order within each
    # direction stays score-descending, so this is a different slice of the same
    # deterministic ranking, not a reordering of it.
    ups = [a for a in alternatives if a["direction"] == "UP"]
    downs = [a for a in alternatives if a["direction"] != "UP"]
    top = ups[:2] + downs[:2]
    listed = ", ".join(
        f"{a['symbol']} {a['direction']} ({a['horizon']}, {a['heuristic_score']:.2f})"
        for a in top
    )
    return (
        f"\n\nAlso on the board ({len(ups)} UP / {len(downs)} DOWN, not risk-checked): "
        f"{listed}. Name one and I'll run the full analysis on it."
    )


def _no_trade_reply(symbol: str, minutes: int, exc: Any, note: str = "") -> ChatReply:
    """Declining is an answer. Say why, and do not offer a direction anyway."""

    time_text = "any time limit" if minutes == 0 else f"a {minutes}-minute hold"

    headline = (
        f"No trade on {symbol} for {time_text}."
        if exc.status is SignalStatus.NO_TRADE
        else f"I don't have usable data for {symbol} right now."
    )
    body = f" {exc.reason}." if exc.reason else ""
    detail = f" Contributing factors: {exc.detail}." if exc.detail else ""
    return ChatReply(
        text=(
            f"{headline}{body}{detail} I'd rather tell you there's nothing here "
            "than hand you a direction the data doesn't support." + note
        ),
        intent=Intent.SIGNAL,
        data={
            "symbol": symbol,
            "status": exc.status.value,
            "reason": exc.reason,
            "detail": exc.detail,
            "time_limit_minutes": minutes,
        },
    )


def _format_recommendation(rec: Any, minutes: int, expiry: datetime) -> str:
    """The answer the user asked for: direction, time, and the variables."""
    arrow = "UP" if rec.direction.value == "UP" else "DOWN"
    lines = [
        f"{rec.symbol} -- {arrow} for the next {minutes} minutes.",
        "",
        f"  Enter around   {rec.reference_price}",
        f"  Expires        {expiry.strftime('%H:%M:%S')} UTC ({minutes} min from now)",
        f"  Stop           {rec.stop_loss}",
        f"  Target         {rec.take_profit}",
        f"  Reward:risk    {rec.risk_reward_ratio:.2f}",
        f"  Regime         {rec.regime or 'unclassified'}",
        f"  Setup quality  {rec.risk_level.replace('_', ' ').lower()}"
        f" (heuristic {rec.heuristic_score:.2f})",
        f"  Venue          {rec.recommended_venue}",
    ]
    if rec.memory_consulted_count:
        lines.append(f"  Memory         {rec.memory_consulted_count} past outcome(s) consulted")
    if rec.key_lessons_applied:
        lines.append("")
        lines.append("From past trades on this symbol:")
        lines.extend(f"  - {lesson}" for lesson in rec.key_lessons_applied)
    # Shown under their own heading rather than mixed in with the lessons above:
    # these are failure modes this symbol has repeated at this horizon, which is
    # a stronger statement than a one-off observation and reads as one.
    if rec.memory_rules_applied:
        lines.append("")
        lines.append("This setup has failed this way before:")
        lines.extend(f"  - {rule}" for rule in rec.memory_rules_applied)
    lines.extend(
        [
            "",
            rec.rationale,
            "",
            "Tell me how it went when it closes and I'll record it -- if it loses "
            "I'll work out why first.",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# outcome reporting: the win/loss asymmetry the user asked for           #
# ---------------------------------------------------------------------- #


def _handle_outcome(parsed: ChatIntent, state: dict[str, Any]) -> ChatReply:
    """Record a reported outcome; for a loss, diagnose it first.

    The trade being reported is the one this conversation last issued. There is
    no lookup by symbol alone: two BTCUSDT trades an hour apart have different
    entries, stops and holding periods, and attaching a loss to the wrong one
    would file a real post-mortem against a trade that never happened.
    """
    from quantedge.services.memory import record_trade_outcome_and_analyze

    outcome = parsed.outcome or SettlementOutcome.FLAT
    last = state.get("last_recommendation")

    if last is None:
        return ChatReply(
            text=(
                f"I can record that as a {outcome.value}, but I don't have the trade "
                "it refers to in this conversation -- I need the entry, stop, target "
                "and the time it was opened to work out what happened. Ask me for a "
                "signal first and report back on that one, and the diagnosis will be "
                "measured rather than guessed."
            ),
            intent=Intent.REPORT_OUTCOME,
            data={"outcome": outcome.value, "recorded": False},
            warnings=["no recommendation in this session to attach the outcome to"],
        )

    if parsed.symbol is not None and parsed.symbol != last.get("symbol"):
        return ChatReply(
            text=(
                f"You mentioned {parsed.symbol}, but the last trade I issued here was "
                f"{last.get('symbol')}. I haven't recorded anything. Tell me which one "
                "you mean and I'll file it against the right entry."
            ),
            intent=Intent.REPORT_OUTCOME,
            data={
                "outcome": outcome.value,
                "recorded": False,
                "reported_symbol": parsed.symbol,
                "last_symbol": last.get("symbol"),
            },
            warnings=["reported symbol does not match the last recommendation"],
        )

    symbol = str(last["symbol"])
    horizon = str(last["horizon"])
    direction = SignalDirection(str(last["direction"]))
    reference_price = Decimal(str(last["reference_price"]))
    stop = Decimal(str(last["stop_loss"]))
    target = Decimal(str(last["take_profit"]))
    entry_time = _parse_utc(last.get("valid_from_utc"))
    expiry = _parse_utc(last.get("expiry_utc")) or _parse_utc(last.get("valid_until_utc"))

    period = _holding_period(symbol, horizon, entry_time, expiry)

    memory = record_trade_outcome_and_analyze(
        str(last.get("recommendation_id") or "unknown"),
        outcome,
        symbol=symbol,
        asset_class=str(last.get("asset_class") or "crypto"),
        horizon=horizon,
        regime=_regime_of(last.get("regime")),
        pattern=str(last.get("risk_level") or "general"),
        direction=direction,
        reference_price=reference_price,
        stop=stop,
        target=target,
        holding_candles=period.candles,
        entry_time=entry_time,
        entry_structure=period.entry_structure,
        exit_structure=period.exit_structure,
        entry_features=period.entry_features,
        exit_features=period.exit_features,
        user_notes=parsed.notes,
    )

    # The trade is settled; a later "it lost" must not be filed against it again.
    state.pop("last_recommendation", None)
    state.setdefault("recorded_memories", []).append(memory.memory_id)

    text = (
        _win_text(memory, symbol)
        if outcome is SettlementOutcome.WIN
        else _loss_text(memory, symbol, period)
    )
    return ChatReply(
        text=text,
        intent=Intent.REPORT_OUTCOME,
        data={
            "outcome": outcome.value,
            "recorded": True,
            "memory": memory.model_dump(mode="json"),
            "bars_measured": len(period.candles),
        },
        warnings=period.warnings,
    )


def _win_text(memory: Any, symbol: str) -> str:
    """A win is filed and nothing is concluded from it."""
    summary = _safe_memory_summary()
    tally = ""
    if summary is not None:
        tally = (
            f"\n\nThe bank now holds {summary['total_memories']} trade(s): "
            f"{summary['wins']} win, {summary['losses']} loss."
            if summary["wins"] == 1
            else (
                f"\n\nThe bank now holds {summary['total_memories']} trade(s): "
                f"{summary['wins']} wins, {summary['losses']} losses."
            )
        )
    return (
        f"Recorded: {symbol} WIN ({memory.memory_id}).\n\n"
        "No post-mortem was run and no rule was derived from it. One favourable "
        "result tells me the setup worked this time, not that it has an edge -- "
        "I'd be teaching myself a rule from a single sample."
        f"{tally}"
    )


def _loss_text(memory: Any, symbol: str, period: _HoldingPeriod) -> str:
    """A loss is diagnosed first, and the diagnosis is what gets filed."""
    lines = [f"Recorded: {symbol} LOSS ({memory.memory_id}). Here's what actually happened.", ""]
    lines.append(memory.root_cause)

    if period.candles:
        lines.append("")
        lines.append(
            f"Measured over {len(period.candles)} closed "
            f"{period.timeframe or 'execution'} bar(s) between entry and expiry."
        )
    if memory.key_lessons:
        lines.extend(["", "What that means:"])
        lines.extend(f"  - {lesson}" for lesson in memory.key_lessons)
    if memory.do_rules:
        lines.extend(["", "Added to my DO rules:"])
        lines.extend(f"  - {rule}" for rule in memory.do_rules)
    if memory.dont_rules:
        lines.extend(["", "Added to my DON'T rules:"])
        lines.extend(f"  - {rule}" for rule in memory.dont_rules)
    if not memory.key_lessons and not memory.do_rules and not memory.dont_rules:
        lines.extend(
            [
                "",
                "No rule was derived: nothing measurable distinguished this loss, and "
                "inventing a rule for it would put advice in the bank that no "
                "observation supports.",
            ]
        )
    return "\n".join(lines)


@dataclass
class _HoldingPeriod:
    """The closed bars a trade was open for, plus snapshots at each end."""

    candles: list[Any] = field(default_factory=list)
    entry_features: Any = None
    entry_structure: Any = None
    exit_features: Any = None
    exit_structure: Any = None
    timeframe: str | None = None
    warnings: list[str] = field(default_factory=list)


def holding_period_for(
    *,
    symbol: str,
    horizon: str,
    entry_time: datetime | None,
    expiry: datetime | None,
) -> _HoldingPeriod:
    """Public entry point for callers outside chat -- the HTTP feedback route.

    Exposed so the REST path diagnoses a loss the same way the conversation
    does. Two implementations of "which bars was this trade open for" would
    eventually disagree, and the memory bank would hold rows measured two
    different ways with nothing recording which was which.
    """
    return _holding_period(symbol, horizon, entry_time, expiry)


def _holding_period(
    symbol: str,
    horizon: str,
    entry_time: datetime | None,
    expiry: datetime | None,
) -> _HoldingPeriod:
    """Fetch and split the execution-timeframe bars covering a trade.

    Bars are cut at ``expiry`` as well as at entry. Letting the window run past
    expiry would diagnose price action the trade was never exposed to, and a
    reversal that happened ten minutes after the position closed would be
    written into the record as the reason it lost.

    Only closed bars are used, so a forming bar's high cannot decide whether the
    stop was touched -- that reading changes as the bar develops.
    """
    from quantedge.contracts import Timeframe
    from quantedge.providers.registry import get_registry
    from quantedge.services import indicators as ind
    from quantedge.services import structure as st
    from quantedge.services.horizons import horizon_timeframes

    period = _HoldingPeriod()
    if entry_time is None:
        period.warnings.append("entry time unknown; the holding period cannot be bounded")
        return period

    try:
        tf = Timeframe(horizon_timeframes(horizon)["execution"])
        period.timeframe = tf.value
        series = get_registry().get_candles(symbol, tf, limit=500)
    except QuantEdgeError as exc:
        period.warnings.append(f"settlement candles unavailable: {exc.message}")
        return period

    closed = [c for c in series.candles if c.is_closed]
    cutoff = min(expiry, utc_now()) if expiry is not None else utc_now()

    before_entry = [c for c in closed if c.close_time_utc <= entry_time]
    period.candles = [c for c in closed if entry_time < c.close_time_utc <= cutoff]
    up_to_exit = [c for c in closed if c.close_time_utc <= cutoff]

    if not period.candles:
        period.warnings.append(
            "no closed bars cover the holding period yet -- the outcome is recorded, "
            "the cause is not inferred"
        )

    period.entry_features, period.entry_structure = _snapshot(
        before_entry, series.provider, ind, st
    )
    period.exit_features, period.exit_structure = _snapshot(up_to_exit, series.provider, ind, st)
    return period


def _snapshot(candles: list[Any], provider: str, ind: Any, st: Any) -> tuple[Any, Any]:
    """Features and structure as of the last bar in ``candles``.

    Returns ``(None, None)`` below the warm-up length rather than computing an
    indicator from too few bars -- a 14-period ATR over 9 bars is a number, but
    it is not an ATR, and the post-mortem would compare it against a real one.
    """
    if len(candles) < 30:
        return None, None
    try:
        features = ind.compute_features(candles, provider=provider)
        report = st.analyze_structure(candles, atr=features.atr_14)
    except QuantEdgeError:
        return None, None
    return features, report


def _parse_utc(value: Any) -> datetime | None:
    """Parse an ISO timestamp from the session payload, or ``None``."""
    from datetime import datetime as _dt

    if value is None:
        return None
    if isinstance(value, _dt):
        return value
    try:
        return _dt.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _regime_of(value: Any) -> MarketRegime:
    """The stored regime, or ``UNCERTAIN`` when it is absent or unrecognised."""
    if isinstance(value, MarketRegime):
        return value
    try:
        return MarketRegime(str(value).upper())
    except ValueError:
        return MarketRegime.UNCERTAIN


def _safe_memory_summary() -> dict[str, Any] | None:
    from quantedge.services.memory import get_memory_bank_summary

    try:
        return get_memory_bank_summary()
    except QuantEdgeError as exc:
        log.warning("memory bank unavailable", extra={"code": exc.code})
        return None


# ---------------------------------------------------------------------- #
# memory, performance, and the informational intents                     #
# ---------------------------------------------------------------------- #


def _handle_memory(parsed: ChatIntent) -> ChatReply:
    """What the bot has learned, and from which trades."""
    from quantedge.services.memory import get_relevant_memories

    try:
        memories = get_relevant_memories(symbol=parsed.symbol, limit=20)
        summary = get_memory_bank_summary_safe()
    except QuantEdgeError as exc:
        return ChatReply(
            text=f"I can't read the memory bank right now: {exc.message}",
            intent=Intent.MEMORY,
            warnings=[exc.message],
        )

    scope = f" on {parsed.symbol}" if parsed.symbol else ""
    if not memories:
        return ChatReply(
            text=(
                f"I have no recorded trades{scope} yet. Memory is built from outcomes "
                "you report back to me -- take a signal, tell me how it closed, and "
                "the record starts there."
            ),
            intent=Intent.MEMORY,
            data={"memories": [], "summary": summary},
        )

    lines = [f"{len(memories)} recorded trade(s){scope}, most recent first:", ""]
    for m in memories[:5]:
        lines.append(
            f"  {m.created_at_utc:%Y-%m-%d %H:%M} UTC  {m.symbol} {m.horizon}  {m.outcome.value}"
        )
        lines.append(f"    {m.root_cause}")
        lines.append("")

    if summary:
        do_rules = summary.get("top_do_rules") or []
        dont_rules = summary.get("top_dont_rules") or []
        if do_rules:
            lines.append("DO rules derived from losses:")
            lines.extend(f"  - {r}" for r in do_rules[:5])
            lines.append("")
        if dont_rules:
            lines.append("DON'T rules derived from losses:")
            lines.extend(f"  - {r}" for r in dont_rules[:5])
            lines.append("")
        if not do_rules and not dont_rules:
            lines.append(
                "No rules have been derived yet: rules come from diagnosed losses, "
                "and wins deliberately produce none."
            )

    return ChatReply(
        text="\n".join(lines).rstrip(),
        intent=Intent.MEMORY,
        data={
            "memories": [m.model_dump(mode="json") for m in memories],
            "summary": summary,
        },
    )


def get_memory_bank_summary_safe() -> dict[str, Any] | None:
    """The bank summary, or ``None`` when persistence is unavailable."""
    return _safe_memory_summary()


def _handle_performance(parsed: ChatIntent) -> ChatReply:
    """Realised results only, with the sample size stated next to them."""
    from quantedge.services.settlement import get_performance_summary

    try:
        perf = get_performance_summary(symbol=parsed.symbol)
        summary = _safe_memory_summary()
    except QuantEdgeError as exc:
        return ChatReply(
            text=f"I can't read the performance record right now: {exc.message}",
            intent=Intent.PERFORMANCE,
            warnings=[exc.message],
        )

    scope = parsed.symbol or "all symbols"
    decided = perf.wins + perf.losses
    lines = [f"Settled record for {scope}:", ""]
    lines.append(f"  Settled        {perf.settled_signals}")
    lines.append(f"  Wins           {perf.wins}")
    lines.append(f"  Losses         {perf.losses}")
    lines.append(f"  Flat / void    {perf.flat} / {perf.void}")

    if perf.observed_win_rate is None:
        lines.append("  Win rate       not computable -- nothing has settled either way yet")
    else:
        lines.append(
            f"  Win rate       {perf.observed_win_rate:.1%} observed over {decided} trade(s)"
        )

    if perf.sample_too_small:
        lines.extend(
            [
                "",
                f"That rate is descriptive, not predictive. With {decided} settled "
                "trade(s) the sample is far below the ~30 needed to distinguish a "
                "result from noise, and I will not present it as a probability of "
                "the next trade winning.",
            ]
        )

    if summary and summary.get("total_memories"):
        lines.extend(
            [
                "",
                f"Memory bank: {summary['total_memories']} recorded trade(s), "
                f"{len(summary.get('top_do_rules') or [])} DO rule(s) and "
                f"{len(summary.get('top_dont_rules') or [])} DON'T rule(s) derived "
                "from diagnosed losses.",
            ]
        )

    return ChatReply(
        text="\n".join(lines),
        intent=Intent.PERFORMANCE,
        data={"performance": perf.model_dump(mode="json"), "memory_summary": summary},
    )


def _handle_time_limits() -> ChatReply:
    """The durations the scanner is actually configured for."""
    limits = available_time_limits()
    lines = ["Time limits I can analyse (each maps to a configured horizon):", ""]
    lines.extend(f"  {limit.label:<8} analysed on the {limit.horizon} horizon" for limit in limits)
    lines.extend(
        [
            "",
            'Ask for a signal with the limit you want -- "BTC 10 min", "gold 1 hour". '
            "The expiry clock I give you is the limit you picked, not the horizon's.",
        ]
    )
    return ChatReply(
        text="\n".join(lines),
        intent=Intent.TIME_LIMITS,
        data={"time_limits": [limit.to_dict() for limit in limits]},
    )


def _handle_status() -> ChatReply:
    """Which providers answered, and whether a reviewer model is reachable."""
    from quantedge.providers.llm import default_llm_provider
    from quantedge.providers.registry import get_registry

    rows: list[dict[str, Any]] = []
    try:
        for health in get_registry().health_check_all():
            rows.append(
                {
                    "provider": health.provider,
                    "kind": health.kind,
                    "status": health.status.value,
                    "message": health.message,
                }
            )
    except QuantEdgeError as exc:
        rows.append({"provider": "registry", "status": "error", "message": exc.message})

    llm_row: dict[str, Any]
    provider = default_llm_provider()
    if provider is None:
        llm_row = {
            "provider": "llm",
            "status": "disabled",
            "message": "no reviewer configured or credential missing",
        }
    else:
        try:
            health = provider.health()
            llm_row = {
                "provider": provider.provider_name,
                "model": provider.model_name,
                "status": health.status.value,
                "message": health.message,
            }
        except QuantEdgeError as exc:
            llm_row = {
                "provider": provider.provider_name,
                "status": "error",
                "message": exc.message,
            }

    lines = ["Market data providers:", ""]
    for row in rows:
        note = f" -- {row['message']}" if row.get("message") else ""
        lines.append(f"  {row['provider']:<14} {row['status']}{note}")

    lines.extend(["", "Reviewer model:", ""])
    note = f" -- {llm_row['message']}" if llm_row.get("message") else ""
    lines.append(f"  {llm_row['provider']:<14} {llm_row['status']}{note}")
    lines.extend(
        [
            "",
            "Signals are produced by the deterministic scanner. The reviewer model "
            "annotates them; when it is unreachable the analysis still runs and the "
            "answer says so rather than substituting anything for it.",
        ]
    )
    return ChatReply(
        text="\n".join(lines),
        intent=Intent.STATUS,
        data={"providers": rows, "llm": llm_row},
    )


def _handle_help(intent: Intent) -> ChatReply:
    """What the bot does, phrased as the sentences that actually work."""
    limits = ", ".join(limit.label for limit in available_time_limits())
    text = (
        "Here's what I can do.\n"
        "\n"
        '  Get a signal        "BTC 10 min", "give me a signal on gold, 30 minutes"\n'
        "                      I answer with UP or DOWN, the entry, an expiry clock,\n"
        "                      a stop, a target and the reward:risk behind them.\n"
        "\n"
        '  Report an outcome   "that one won", "the BTC trade lost"\n'
        "                      A win I just file. A loss I diagnose first: I pull the\n"
        "                      closed bars from entry to expiry, measure what price\n"
        "                      did, and record the cause with the numbers behind it.\n"
        "\n"
        '  Review memory       "what have you learned", "show me your rules"\n'
        '  Check results       "how are you doing", "performance"\n'
        '  See time limits     "what time limits do you have"\n'
        '  Check connections   "status"\n'
        "\n"
        f"Time limits available: {limits}.\n"
        "\n"
        "Two things I won't do: give you a direction when the data doesn't support "
        "one -- you'll get NO_TRADE and the reason -- and quote you a win "
        "probability, because nothing here has been calibrated to produce one."
    )
    if intent is Intent.UNKNOWN:
        text = "I'm not sure what you're asking for.\n\n" + text
    return ChatReply(text=text, intent=intent, data={"help": True})
