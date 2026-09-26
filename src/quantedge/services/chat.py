"""The conversational layer: intent in, grounded answer out.

How a message becomes an answer
-------------------------------
1. :func:`parse_intent` classifies the message and pulls out symbol, time limit
   and outcome **with regular expressions, in code**. No model decides what the
   user asked for, so "BTC 10 min" cannot be misread as a request for ETH.
2. The matching deterministic service runs -- scanner, risk, memory, settlement.
   Every number in the answer originates there.
3. The reply is assembled from those numbers by :mod:`quantedge.services.chat`
   itself. In the default ``llm_first`` mode the GLM 5.3 Flash brain is the
   decision authority -- it chooses the direction and whether to trade at all --
   but it never supplies a price or a level; entry, stop and target are always
   grounded from the deterministic engine's real ATR and pivots.

Step 3 is why the model being unreachable degrades the decision path and nothing
else: when the brain cannot be reached the deterministic gate decides, so an
unreachable model falls back to the tested deterministic answer rather than a
fabricated one. A chat that invents a price when its LLM is down is worse than one
that decides deterministically and says so, and this ordering makes the latter the
only option.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
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
    TRADINGVIEW = "TRADINGVIEW"
    LIFECYCLE = "LIFECYCLE"
    HELP = "HELP"
    CONVERSATION = "CONVERSATION"
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
_TRADINGVIEW_WORDS = re.compile(
    r"\b(tradingview|tv\b|trading\s*view)\b",
    re.IGNORECASE,
)
_LIFECYCLE_WORDS = re.compile(
    r"\b(lifecycle|monitor|in\s*flight|open\s+trades?|open\s+signals?|active\s+trades?|active\s+signals?|settle\s+due|settle)\b",
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
    "SOLANA": "SOLUSDT",
    "XRP": "XRPUSDT",
    "RIPPLE": "XRPUSDT",
    "DOGE": "DOGEUSDT",
    "DOGECOIN": "DOGEUSDT",
    "ADA": "ADAUSDT",
    "CARDANO": "ADAUSDT",
    "AVAX": "AVAXUSDT",
    "AVALANCHE": "AVAXUSDT",
    "DOT": "DOTUSDT",
    "POLKADOT": "DOTUSDT",
    "LINK": "LINKUSDT",
    "CHAINLINK": "LINKUSDT",
    "LTC": "LTCUSDT",
    "LITECOIN": "LTCUSDT",
    "POL": "POLUSDT",
    "MATIC": "POLUSDT",
    "GOLD": "XAUUSD",
    "SILVER": "XAGUSD",
    "OIL": "WTICOUSD",
    "CRUDE": "WTICOUSD",
    "EUR": "EURUSD",
    "EURO": "EURUSD",
    "GBP": "GBPUSD",
    "POUND": "GBPUSD",
    "CABLE": "GBPUSD",
    "YEN": "USDJPY",
    "JPY": "USDJPY",
    "CAD": "USDCAD",
    "CHF": "USDCHF",
    "AUD": "AUDUSD",
    "NZD": "NZDUSD",
}

_SIGNAL_ID = re.compile(r"\b((?:rec|mem|sig)-[0-9a-f]{6,})\b", re.IGNORECASE)


_ASSET_NAME_MAPPINGS: dict[str, str] = {
    "ARGENTINE PESO": "USDARS",
    "ARGENTINE": "USDARS",
    "ARGENTINA": "USDARS",
    "PESO": "USDARS",
    "TURKISH LIRA": "USDTRY",
    "LIRA": "USDTRY",
    "BRAZILIAN REAL": "USDBRL",
    "REAL": "USDBRL",
    "MEXICAN PESO": "USDMXN",
    "INDIAN RUPEE": "USDINR",
    "RUPEE": "USDINR",
    "SOUTH AFRICAN RAND": "USDZAR",
    "RAND": "USDZAR",
    "JAPANESE YEN": "USDJPY",
    "YEN": "USDJPY",
    "EURO": "EURUSD",
    "BRITISH POUND": "GBPUSD",
    "POUND": "GBPUSD",
    "STERLING": "GBPUSD",
    "SWISS FRANC": "USDCHF",
    "FRANC": "USDCHF",
    "CANADIAN DOLLAR": "USDCAD",
    "LOONIE": "USDCAD",
    "AUSTRALIAN DOLLAR": "AUDUSD",
    "AUSSIE": "AUDUSD",
    "NEW ZEALAND DOLLAR": "NZDUSD",
    "KIWI": "NZDUSD",
    "GOLD": "XAUUSD",
    "SILVER": "XAGUSD",
    "CRUDE OIL": "USOIL",
    "CRUDE": "USOIL",
    "OIL": "USOIL",
    "BRENT": "UKOIL",
    "NATURAL GAS": "NATGAS",
    "COPPER": "COPPER",
    "BITCOIN": "BTCUSDT",
    "ETHEREUM": "ETHUSDT",
    "SOLANA": "SOLUSDT",
    "RIPPLE": "XRPUSDT",
    "DOGECOIN": "DOGEUSDT",
    "DOGE": "DOGEUSDT",
    "CARDANO": "ADAUSDT",
    "BINANCE COIN": "BNBUSDT",
    "APPLE": "AAPL",
    "NVIDIA": "NVDA",
    "TESLA": "TSLA",
    "MICROSOFT": "MSFT",
    "AMAZON": "AMZN",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "META": "META",
    "FACEBOOK": "META",
    "S&P 500": "SPX",
    "S&P": "SPX",
    "SP500": "SPX",
    "NASDAQ": "NDX",
    "DOW JONES": "DJI",
    "DOW": "DJI",
}


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
    if _TRADINGVIEW_WORDS.search(text):
        return ChatIntent(Intent.TRADINGVIEW, symbol=symbol, minutes=minutes, notes=text)
    if _LIFECYCLE_WORDS.search(text):
        return ChatIntent(Intent.LIFECYCLE, symbol=symbol)

    # Exclude feedback/questions about problems from being mistaken for a signal request
    is_meta_question = bool(re.search(r"\b(can'?t|cannot|why|how\s+come|problem|issue|bug|not\s+working|failed\s+to|what\s+markets?|explain|difference)\b", text, re.I))
    is_explicit_signal_command = bool(re.search(r"\b(give\s+me|send\s+me|generate|scan\s+for|issue|recommend|signal|trade\s+setup)\b", text, re.I))

    if is_meta_question and not is_explicit_signal_command:
        return ChatIntent(Intent.CONVERSATION, symbol=symbol, minutes=minutes, notes=text)

    if _SIGNAL_WORDS.search(text) or (symbol is not None and minutes is not None):
        return ChatIntent(Intent.SIGNAL, symbol=symbol, minutes=minutes)
    if _HELP_WORDS.search(text):
        return ChatIntent(Intent.HELP)
    # A bare symbol is a request about that symbol; anything else is unknown.
    if symbol is not None and not is_meta_question:
        return ChatIntent(Intent.SIGNAL, symbol=symbol, minutes=minutes)
    return ChatIntent(Intent.UNKNOWN)


def _extract_symbol(text: str, default: str | None = None, default_symbol: str | None = None) -> str | None:
    """The first supported symbol or asset named in the message.

    Supports canonical tickers (USDJPY, BTCUSDT, USDARS), natural language names
    ("Argentine Peso", "Gold", "Tesla"), separator-delimited pairs (USD/JPY, USD/ARS),
    space-separated pairs (USD ARS), and common asset aliases.
    """
    from quantedge.symbols import normalize_symbol

    upper_text = text.upper()

    # 1. Natural language asset names (e.g. "Argentine Peso" -> "USDARS", "Gold" -> "XAUUSD")
    for name, sym in sorted(_ASSET_NAME_MAPPINGS.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", upper_text):
            return sym

    # 2. Separator pairs: e.g. "USD/JPY", "EUR/USD", "USD/ARS", "BTC/USDT", "BTC-USDT"
    sep_matches = re.findall(r"\b[A-Za-z0-9]{2,6}\s*[/_\-:]\s*[A-Za-z0-9]{2,6}\b", text)
    for m in sep_matches:
        try:
            cand = normalize_symbol(m)
            if is_supported(cand):
                return cand
        except Exception:
            pass

    _STOP_WORDS = {
        "GIVE", "SHOW", "SEND", "MAKE", "NEED", "WANT", "HAVE", "TAKE", "TELL",
        "HOLD", "TIME", "CALL", "OPEN", "FAST", "SLOW", "VERY", "GOOD", "HIGH",
        "FROM", "WHAT", "WHEN", "WITH", "THIS", "THAT", "THEM", "THEY", "YOUR",
        "SOME", "MORE", "LESS", "NEXT", "LAST", "HERE", "TRUE", "REAL", "FREE",
        "BEST", "LOOK", "SEEK", "FIND", "RATE", "STOP", "LOSS", "GAIN", "SIGN",
        "AUTO", "PLAN", "WARN", "LONG", "SHORT", "RULE", "TEST", "TRADE", "SETUP",
        "SIGNAL", "SIGNALS", "MARKET", "MARKETS", "PLEASE", "WHICH", "ABOUT",
        "UNKNOWN", "ACTIVE", "CURRENT", "PLATFORM", "BRAIN", "SYSTEM", "ENGINE",
        "COIN", "COINS", "PAIR", "PAIRS", "CURRENCY", "CURRENCIES", "STOCK", "STOCKS",
        "MIN", "MINS", "MINUTE", "MINUTES", "HOUR", "HOURS", "NOW", "TODAY", "OUT",
        "ANALYZE", "ANALYSIS", "ENTRY", "SCAN", "DOWN",
    }

    from quantedge.symbols import _ISO_CURRENCIES

    _PREPOSITIONS = {"FOR", "IN", "ON", "THE", "AND", "OF", "TO", "AT", "BY", "WITH", "FROM"}
    _EFFECTIVE_STOP = _STOP_WORDS | _PREPOSITIONS

    # 3. Space-separated currency/asset pairs: e.g. "USD ARS", "USD JPY", "EUR USD", "BTC USDT"
    tokens = re.findall(r"[A-Za-z0-9]{2,12}", upper_text)
    for i in range(len(tokens) - 1):
        t1, t2 = tokens[i], tokens[i + 1]
        if t1 in _EFFECTIVE_STOP or t2 in _EFFECTIVE_STOP:
            continue
        # Forex: both halves must be valid ISO currency codes (e.g. USD JPY, EUR USD, USD ARS)
        if t1 in _ISO_CURRENCIES and t2 in _ISO_CURRENCIES:
            return normalize_symbol(t1 + t2)
        # Crypto: e.g. BTC USDT, ETH USDT, SOL USDT
        if t2 in ("USDT", "BUSD", "USDC") and len(t1) >= 2:
            return normalize_symbol(t1 + t2)

    # 4. Direct supported tokens and aliases
    from quantedge.services.tradingview import _EXCHANGE_MAP

    for token in tokens:
        if token in _EFFECTIVE_STOP or len(token) < 2:
            continue
        if token in _SYMBOL_ALIASES:
            return _SYMBOL_ALIASES[token]
        if token in _EXCHANGE_MAP:
            return token
        if token.endswith("USDT") or token.endswith("BUSD") or token.endswith("USDC"):
            return normalize_symbol(token)
        if len(token) == 6 and (token[:3] in _ISO_CURRENCIES and token[3:] in _ISO_CURRENCIES):
            return normalize_symbol(token)

    return default or default_symbol



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

    # Run background check to automatically settle finished signals (TP/SL/expiry)
    try:
        from quantedge.services.lifecycle import monitor_and_settle_active_signals

        monitor_and_settle_active_signals()
    except Exception:
        pass

    if parsed.intent is Intent.SIGNAL:
        return _handle_signal(parsed, default_symbol, default_minutes, state)
    if parsed.intent is Intent.REPORT_OUTCOME:
        return _handle_outcome(parsed, state)
    if parsed.intent is Intent.TRADINGVIEW:
        return _handle_tradingview(parsed, default_symbol)
    if parsed.intent is Intent.LIFECYCLE:
        return _handle_lifecycle()
    if parsed.intent is Intent.MEMORY:
        return _handle_memory(parsed)
    if parsed.intent is Intent.PERFORMANCE:
        return _handle_performance(parsed)
    if parsed.intent is Intent.TIME_LIMITS:
        return _handle_time_limits()
    if parsed.intent is Intent.STATUS:
        return _handle_status()
    if parsed.intent is Intent.HELP:
        return _handle_help(parsed.intent)
    return _handle_conversation(message, state)


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

    # 1. Resolve target symbol: user explicitly typed one, OR passed one in default_symbol
    target_symbol = parsed.symbol
    if not target_symbol and default_symbol and default_symbol.upper() not in ("ANY", "", "NONE", "SELECT"):
        target_symbol = default_symbol.strip().upper()

    # 2. Resolve hold duration (e.g. 1m, 5m, 10m, 15m, 60m)
    requested = parsed.minutes or default_minutes or _DEFAULT_HOLD_MINUTES
    if parsed.minutes is None and default_minutes is None and target_symbol:
        assumption_note = (
            f"\n\nI assumed a {_DEFAULT_HOLD_MINUTES}-minute hold since you didn't say. "
            f"Other options: {', '.join(t.label for t in available_time_limits())}."
        )
    minutes_used = requested

    # 3. Universal market routing: non-crypto, exotic, forex, commodities, and equities route to TradingView MCP
    from quantedge.services.tradingview import generate_tradingview_recommendation

    def _is_tradingview_direct(sym: str) -> bool:
        s = sym.upper()
        is_binance_crypto = (
            s.endswith("USDT")
            and any(s.startswith(c) for c in ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "MATIC", "POL", "LTC", "NEAR", "SUI", "PEPE", "SHIB", "TRX"))
        )
        return not is_binance_crypto

    if target_symbol is None:
        try:
            rec = generate_best_trade_recommendation(
                time_limit_minutes=minutes_used,
                alternatives_out=alternatives,
            )
            assumption_note = _alternatives_note(alternatives)
        except NoTradeReason:
            # Fall back to TradingView for top institutional assets
            log.info("scanner found no trade; falling back to TradingView institutional screener")
            tv_cands = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSD", "USDJPY"]
            rec = None
            for cand in tv_cands:
                try:
                    rec = generate_tradingview_recommendation(cand, minutes=minutes_used)
                    break
                except Exception:
                    continue
            if rec is None:
                return _no_trade_reply("any symbol", minutes_used, NoTradeReason(SignalStatus.NO_TRADE, "all markets in consolidation"))
        except Exception as exc:
            log.warning("global scan failed, falling back to TradingView: %s", exc)
            try:
                rec = generate_tradingview_recommendation("BTCUSDT", minutes=minutes_used)
            except Exception:
                return ChatReply(
                    text=f"Market analysis is temporarily unavailable ({type(exc).__name__}). Please retry with `BTC 15m`.",
                    intent=Intent.SIGNAL,
                    data={"symbol": "ANY", "error": str(exc)},
                )
    else:
        symbol = target_symbol
        # All non-crypto and exotic markets (like USDARS, EURTRY, XAUUSD, NVDA) route directly to TradingView MCP
        if _is_tradingview_direct(symbol):
            try:
                rec = generate_tradingview_recommendation(symbol, minutes=minutes_used)
            except NoTradeReason as exc:
                # The TradingView engine ran and REACHED a decision -- most often
                # GLM's own NO_TRADE, or a genuine INSUFFICIENT_DATA it raised. Pass
                # it through with the status it set. Re-wrapping every case as
                # INSUFFICIENT_DATA produced the contradiction the user saw on
                # CHFJPY ("I don't have usable data ... the AI brain decided not to
                # trade") -- a real decision mislabelled as a data gap. The two are
                # different answers and the honesty seam depends on not blurring them.
                return _no_trade_reply(symbol, minutes_used, exc, note=assumption_note)
            except (QuantEdgeError, ValidationError) as tv_exc:
                log.warning("tradingview recommendation failed for %s: %s", symbol, tv_exc)
                return _no_trade_reply(
                    symbol, minutes_used, NoTradeReason(SignalStatus.INSUFFICIENT_DATA, str(tv_exc))
                )
            except Exception as tv_exc:
                log.exception("tradingview recommendation crashed for %s", symbol)
                return _no_trade_reply(
                    symbol, minutes_used, NoTradeReason(SignalStatus.INSUFFICIENT_DATA, str(tv_exc))
                )
        else:
            try:
                horizon = horizon_for_minutes(minutes_used)
                rec = generate_trade_recommendation(symbol, time_limit=horizon, hold_minutes=minutes_used)
            except (NoTradeReason, QuantEdgeError, ValidationError) as exc:
                log.info("deterministic engine declined %s (%s); trying TradingView institutional analysis", symbol, exc)
                try:
                    rec = generate_tradingview_recommendation(symbol, minutes=minutes_used)
                except Exception:
                    return _no_trade_reply(symbol, minutes_used, exc, note=assumption_note)
            except Exception as exc:
                log.exception("signal recommendation failed for %s", symbol)
                try:
                    rec = generate_tradingview_recommendation(symbol, minutes=minutes_used)
                except Exception:
                    return ChatReply(
                        text=f"Market data or analysis for {symbol} is temporarily unavailable ({type(exc).__name__}).",
                        intent=Intent.SIGNAL,
                        data={"symbol": symbol, "error": str(exc)},
                    )

    # The expiry the user is told is the duration they chose, not the horizon's.
    expiry = expiry_for(minutes_used, rec.generated_at_utc)
    payload = rec.model_dump(mode="json")
    payload["expiry_utc"] = expiry.isoformat()
    payload["time_limit_minutes"] = minutes_used
    state["last_recommendation"] = payload

    tv_note = ""
    try:
        from quantedge.services.tradingview import get_tradingview_analysis

        tv_data = get_tradingview_analysis(rec.symbol, timeframe="15m")
        if tv_data.get("status") == "ok":
            pivots = tv_data.get("pivots", {})
            bb = tv_data.get("bollinger_bands", {})
            tv_lines = ["", "TradingView Institutional Context:"]
            if pivots.get("pivot"):
                tv_lines.append(f"  Pivot: {pivots.get('pivot')} | S1: {pivots.get('s1')} | R1: {pivots.get('r1')}")
            if bb.get("squeeze"):
                tv_lines.append("  Bollinger Squeeze: ACTIVE (breakout pending)")
            if tv_data.get("sentiment", {}).get("signal"):
                tv_lines.append(
                    f"  TV Consensus: {tv_data['sentiment']['signal']} (Rating: {tv_data['sentiment'].get('rating', '')})"
                )
            tv_note = "\n" + "\n".join(tv_lines)
            payload["tradingview"] = tv_data
    except Exception:
        pass

    return ChatReply(
        text=_format_recommendation(rec, minutes_used, expiry) + tv_note + assumption_note,
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
    _board_tier = {"A_PLUS": "A+", "A": "A", "B": "B", "ARMED": "armed", "STAND_ASIDE": "-"}
    listed = ", ".join(
        f"{a['symbol']} {a['direction']} "
        f"({a['horizon']}, {_board_tier.get(a.get('conviction_tier', ''), '')}"
        f"{'/' if a.get('conviction_tier') else ''}{a['heuristic_score']:.2f})"
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
    # A conditional watch plan rides along only on a genuine directional decline
    # (NO_TRADE), built from real structural levels. It is explicitly not a
    # position -- it names what would have to happen first -- so it turns "nothing
    # here" into something actionable without inventing a trade.
    watch = getattr(exc, "watch_plan", "") or ""
    watch_block = f"\n\nWhat would change this: {watch}" if watch else ""
    return ChatReply(
        text=(
            f"{headline}{body}{detail} I'd rather tell you there's nothing here "
            "than hand you a direction the data doesn't support." + watch_block + note
        ),
        intent=Intent.SIGNAL,
        data={
            "symbol": symbol,
            "status": exc.status.value,
            "reason": exc.reason,
            "detail": exc.detail,
            "watch_plan": watch,
            "time_limit_minutes": minutes,
        },
    )


def _tier_label(tier: Any) -> str:
    """Human label for a conviction tier. Falls back to the raw value."""
    mapping = {
        "A_PLUS": "A+ Prime (full size)",
        "A": "A Strong (reduced size)",
        "B": "B Scalp (LOW conviction, small size)",
        "ARMED": "Armed (conditional -- not yet triggered)",
        "STAND_ASIDE": "Stand aside (no edge)",
    }
    val = getattr(tier, "value", tier)
    return mapping.get(str(val), str(val))


def _format_recommendation(rec: Any, minutes: int, expiry: datetime) -> str:
    """The answer the user asked for: direction, time, and the variables."""
    arrow = "UP" if rec.direction.value == "UP" else "DOWN"
    tier = getattr(rec, "conviction_tier", None)
    size_frac = getattr(rec, "position_size_fraction", 0.0) or 0.0
    lines = [
        f"{rec.symbol} -- {arrow} for the next {minutes} minutes.",
    ]
    if tier is not None:
        lines.append(f"  Conviction     {_tier_label(tier)}")
        # Position size is a RELATIVE multiplier on the trader's own per-trade
        # risk budget, never a dollar figure and never a win probability. A B
        # scalp risks a third of what an A+ prime does because the directional
        # evidence is a third as broad, not because we can quote its odds.
        lines.append(
            f"  Position size  x{size_frac:.2f} of your normal per-trade risk"
        )
    lines.extend(
        [
            f"  Confidence     {rec.confidence_pct}% ({rec.risk_level.replace('_', ' ').lower()})",
            "",
            f"  Enter around   {rec.reference_price}",
            f"  Expires        {expiry.strftime('%H:%M:%S')} UTC ({minutes} min from now)",
        ]
    )
    # An ARMED read holds no live position, so it carries no stop, target or
    # reward:risk -- naming any would fabricate levels on a trade that does not
    # exist. Show the honest "no live levels yet" line and lean on the trigger /
    # upgrade condition below. Every real A+/A/B setup still prints its RR-gated
    # stop and target here.
    if rec.stop_loss is None or rec.take_profit is None:
        lines.append(
            "  Stop / Target  none yet -- ARMED lean, size 0 (no live position to protect)"
        )
    else:
        lines.extend(
            [
                f"  Stop           {rec.stop_loss}",
                f"  Target         {rec.take_profit}",
                f"  Reward:risk    {rec.risk_reward_ratio:.2f}",
            ]
        )
    lines.extend(
        [
            f"  Regime         {rec.regime or 'unclassified'}",
            f"  Venue          {rec.recommended_venue}",
        ]
    )
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
    # What would raise the conviction tier, stated plainly. Only shown when the
    # setup is below full size -- an A+ prime has nothing to upgrade to.
    upgrade = getattr(rec, "upgrade_condition", "") or ""
    if upgrade:
        lines.append("")
        lines.append(f"To upgrade this setup: it {upgrade}.")
    # Caveats that qualify the setup without withdrawing it -- a DEGRADED feed or
    # the short-horizon freshness disclosure. Memory rules already appear under
    # their own heading above, so they are filtered out here rather than repeated.
    _shown = set(getattr(rec, "memory_rules_applied", []))
    caveats = [w for w in getattr(rec, "warnings", []) if w not in _shown]
    if caveats:
        lines.append("")
        lines.append("Before you take it:")
        lines.extend(f"  - {c}" for c in caveats)
    # Who actually made this call. In the default llm_first mode GLM 5.3 Flash is
    # the decision authority; the deterministic engine grounds the levels and is
    # the fallback when the brain is unreachable. Stated plainly because the user
    # asked to be told the brain is the one deciding -- and kept truthful by keying
    # off the live decision mode rather than asserting it unconditionally.
    from quantedge.config import decision_mode

    if decision_mode() == "llm_first":
        brain_line = (
            "GLM-5.3-Flash made this call -- the direction and whether to trade at "
            "all -- from the real evidence and levels the deterministic engine "
            "computed. (If the brain can't be reached, that deterministic gate "
            "decides instead and the answer says so.)"
        )
    else:
        brain_line = (
            "The deterministic engine chose this direction; GLM-5.3-Flash reviewed it."
        )
    lines.extend(
        [
            "",
            rec.rationale,
            "",
            brain_line,
            f"Confidence ({rec.confidence_pct}%) is how strongly the evidence backs "
            "this setup -- not a calibrated probability that it wins.",
            "",
            "Tell me how it went when it closes and I'll record it -- if it loses "
            "I'll work out why first.",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# outcome reporting: the win/loss asymmetry the user asked for           #
# ---------------------------------------------------------------------- #


def _maybe_decimal(value: Any) -> Decimal | None:
    """Parse a stored price level into a Decimal, tolerating a missing/None level.

    An ARMED lean carries ``stop_loss``/``take_profit`` of ``None`` -- there is no
    live position, so quoting a level would fabricate one. ``Decimal(str(None))``
    raises ``InvalidOperation``, and surfaced through the chat route that became the
    "invalid operation" message the user saw when reporting a loss on a size-0 lean
    -- the report was rejected instead of learned from. The whole point of recording
    a loss is to learn from it, so a missing level must flow through as ``None``; the
    memory and post-mortem layers already treat ``None`` levels as "not measurable"
    (they diagnose what they can and skip stop/target-dependent causes) rather than
    crashing.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


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
    # None-safe: an ARMED lean has no live stop/target, and may not have grounded a
    # reference price. Parse each defensively so reporting a loss on a size-0 lean
    # records and diagnoses (as far as the data allows) instead of raising.
    reference_price = _maybe_decimal(last.get("reference_price"))
    stop = _maybe_decimal(last.get("stop_loss"))
    target = _maybe_decimal(last.get("take_profit"))
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
    except Exception as exc:  # noqa: BLE001 -- diagnosis is enrichment, never fatal
        # An unknown horizon, a bad symbol map or a registry fault must degrade to
        # an undiagnosed record, not stop the loss from being saved. The caller
        # (the /bot/feedback route) reads only the warnings and empty fields.
        period.warnings.append(f"settlement candles unavailable: {exc}")
        return period

    try:
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
        period.exit_features, period.exit_structure = _snapshot(
            up_to_exit, series.provider, ind, st
        )
    except Exception as exc:  # noqa: BLE001 -- diagnosis is enrichment, never fatal
        period.warnings.append(f"holding-period diagnosis incomplete: {exc}")
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

    # TradingView MCP status
    try:
        from quantedge.services.tradingview import resolve_tradingview_target

        rows.append(
            {
                "provider": "tradingview",
                "kind": "mcp_intelligence",
                "status": "online",
                "message": "public technical analysis, pivots, and breakout scanner active",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "provider": "tradingview",
                "kind": "mcp_intelligence",
                "status": "degraded",
                "message": str(exc),
            }
        )

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

    lines.extend(["", "Decision brain:", ""])
    note = f" -- {llm_row['message']}" if llm_row.get("message") else ""
    lines.append(f"  {llm_row['provider']:<14} {llm_row['status']}{note}")
    from quantedge.config import decision_mode

    if decision_mode() == "llm_first":
        lines.extend(
            [
                "",
                "GLM-5.3-Flash (via Seek AI) is the decision authority: it decides the "
                "direction and whether to trade at all. The deterministic scanner "
                "gathers the real, verified evidence and grounds every price level; "
                "when the brain is unreachable that deterministic gate decides instead, "
                "and the answer says so rather than substituting anything for it.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Signals are produced by the deterministic scanner; GLM-5.3-Flash "
                "reviews them. When it is unreachable the analysis still runs and the "
                "answer says so rather than substituting anything for it.",
            ]
        )
    return ChatReply(
        text="\n".join(lines),
        intent=Intent.STATUS,
        data={"providers": rows, "llm": llm_row},
    )


def _handle_tradingview(
    parsed: ChatIntent,
    default_symbol: str,
) -> ChatReply:
    """Query TradingView MCP connector for institutional analysis, pivots, and screeners."""
    from quantedge.services.tradingview import (
        format_tradingview_summary,
        get_tradingview_analysis,
        get_tradingview_multi_timeframe,
        scan_tradingview_gainers,
    )

    text_lower = (parsed.notes or "").lower()
    symbol = parsed.symbol

    # 1. Breakout / Gainers screener request
    if any(word in text_lower for word in ("scan", "breakout", "screener", "gainer", "gainers")):
        exchange = "BINANCE"
        if "oanda" in text_lower or "forex" in text_lower or "fx" in text_lower:
            exchange = "OANDA"
        elif "nasdaq" in text_lower or "stock" in text_lower:
            exchange = "NASDAQ"

        gainers = scan_tradingview_gainers(exchange)
        if not gainers:
            return ChatReply(
                text=(
                    f"TradingView Breakout Scanner ({exchange}): "
                    "No current breakout candidates triggered threshold."
                ),
                intent=Intent.TRADINGVIEW,
                data={"scanner": exchange, "results": []},
            )

        lines = [f"**TradingView Breakout & Volume Scanner ({exchange}) Top Results:**"]
        for idx, item in enumerate(gainers[:8], 1):
            s = item.get("symbol", "N/A")
            price = item.get("price") or item.get("close", "N/A")
            chg = item.get("change") or item.get("change_pct", "N/A")
            vol = item.get("volume") or item.get("relative_volume", "N/A")
            lines.append(f"{idx}. **{s}** -- Price: {price} | Change: {chg}% | Vol: {vol}")
        lines.append("\nQuery `tv <symbol>` to run full technical analysis on any coin.")
        return ChatReply(
            text="\n".join(lines),
            intent=Intent.TRADINGVIEW,
            data={"scanner": exchange, "results": gainers},
        )

    # 2. Multi-Timeframe consensus request
    if symbol and any(word in text_lower for word in ("mtf", "multi", "consensus")):
        mtf_data = get_tradingview_multi_timeframe(symbol)
        if mtf_data.get("status") != "ok":
            err_msg = mtf_data.get("error", "unknown")
            return ChatReply(
                text=f"TradingView MTF analysis for {symbol} failed: {err_msg}",
                intent=Intent.TRADINGVIEW,
                data=mtf_data,
                warnings=[str(err_msg)],
            )

        sym = mtf_data.get("symbol")
        venue = mtf_data.get("exchange")
        align = mtf_data.get("alignment")
        rec = mtf_data.get("recommendation")
        tfs = mtf_data.get("timeframes", {})

        lines = [
            f"**TradingView Multi-Timeframe Consensus [{sym} on {venue}]:**",
            f"- **Alignment:** {align} | **Overall Recommendation:** {rec}",
            "- **Timeframe Breakdown:**",
        ]
        for tf_name, tf_val in tfs.items():
            if isinstance(tf_val, dict):
                rating = tf_val.get("rating", "N/A")
                score = tf_val.get("score", "N/A")
                lines.append(f"  - **{tf_name}:** Rating {rating} (Score {score})")
            else:
                lines.append(f"  - **{tf_name}:** {tf_val}")

        return ChatReply(
            text="\n".join(lines),
            intent=Intent.TRADINGVIEW,
            data=mtf_data,
        )

    # 3. Standard TA and Pivot Point lookup for symbol
    target_sym = symbol or default_symbol or "BTCUSDT"
    timeframe = "15m"
    if parsed.minutes:
        if parsed.minutes <= 1:
            timeframe = "1m"
        elif parsed.minutes <= 5:
            timeframe = "5m"
        elif parsed.minutes <= 15:
            timeframe = "15m"
        elif parsed.minutes <= 30:
            timeframe = "30m"
        elif parsed.minutes <= 60:
            timeframe = "1h"
        elif parsed.minutes <= 120:
            timeframe = "2h"
        elif parsed.minutes <= 240:
            timeframe = "4h"
        else:
            timeframe = "1D"
    elif "1h" in text_lower or "60m" in text_lower:
        timeframe = "1h"
    elif "4h" in text_lower:
        timeframe = "4h"
    elif "1d" in text_lower or "daily" in text_lower:
        timeframe = "1D"
    elif "5m" in text_lower:
        timeframe = "5m"
    elif "1w" in text_lower or "weekly" in text_lower:
        timeframe = "1W"

    analysis = get_tradingview_analysis(target_sym, timeframe=timeframe)
    if analysis.get("status") != "ok":
        err_msg = analysis.get("error", "unknown error")
        return ChatReply(
            text=f"TradingView analysis for {target_sym} is currently unavailable: {err_msg}",
            intent=Intent.TRADINGVIEW,
            data=analysis,
            warnings=[str(err_msg)],
        )

    summary_text = format_tradingview_summary(analysis)
    return ChatReply(
        text=summary_text,
        intent=Intent.TRADINGVIEW,
        data=analysis,
    )


def _handle_lifecycle() -> ChatReply:
    """Monitor in-flight trades, check for TP/SL hits or expiries, and report status."""
    from quantedge.services.lifecycle import monitor_and_settle_active_signals

    report = monitor_and_settle_active_signals()
    lines = ["**Autonomous Signal Lifecycle & Trade Monitor:**", ""]

    settled = report.get("settled_details") or []
    active = report.get("active_details") or []

    if settled:
        lines.append(f"**Newly Resolved & Settled Trades ({len(settled)}):**")
        for s in settled:
            outcome_label = (
                "✅ WIN"
                if s["outcome"] == "WIN"
                else ("❌ LOSS" if s["outcome"] == "LOSS" else "⚖️ FLAT")
            )
            lines.append(f"- **{s['symbol']}** ({s['direction']}): {outcome_label}")
            lines.append(f"  Entry: {s['reference_price']} | Exit: {s['exit_price']}")
            lines.append(f"  Outcome Reason: {s['reason']}")
        lines.append("")

    if active:
        lines.append(f"**Active In-Flight Positions ({len(active)}):**")
        for a in active:
            pnl = a["unrealized_pnl_pct"]
            pnl_str = f"+{pnl:.2f}%" if pnl >= 0 else f"{pnl:.2f}%"
            lines.append(
                f"- **{a['symbol']}** ({a['direction']} {a['horizon']}) | Unrealized PnL: {pnl_str} | Remaining: ~{a['remaining_minutes']}m"
            )
            lines.append(
                f"  Entry: {a['reference_price']} | Current: {a['current_price']} | SL: {a['stop_loss']} | TP: {a['take_profit']}"
            )
        lines.append("")

    if not settled and not active:
        lines.append(
            "No signals currently in flight. Ask for a trade (e.g. `BTC 15m`) to start tracking."
        )

    lines.append("")
    lines.append(
        "Signals are automatically tracked until resolution (TP hit, SL hit, or expiry), "
        "and the diagnosed cause of each loss is stored in the Memory Bank so the same "
        "mistake is flagged the next time the setup recurs."
    )

    return ChatReply(
        text="\n".join(lines).strip(),
        intent=Intent.LIFECYCLE,
        data=report,
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
        '  TradingView TA      "tv btc", "tradingview gold 1h", "tv pivots sol"\n'
        '  Breakout screener   "tv breakouts", "tv scan"\n'
        '  MTF consensus       "tv mtf btc"\n'
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


_CONVERSATION_SYSTEM_PROMPT = """You are QuantEdge AI, a world-class institutional quantitative trading analyst and a sharp, fully capable general assistant.

RESPONSE DISCIPLINE (read first, this governs every reply):
- Reason INTERNALLY as briefly as possible, then answer directly. Do NOT deliberate at length, do NOT restate the question, do NOT think out loud across many steps. A good answer lands in 2-6 tight sentences (or a short list). Speed and precision are both required -- a slow answer is a failed answer.
- Lead with the substance the user asked for. Add a trading action or command only if it genuinely helps.

Your dual-brain architecture:
1. Deterministic Mathematical Quant Engine: market structure (CHoCH, BOS, swing highs/lows), 200 EMA trend, ATR volatility bands, multi-timeframe consensus (1W/1D/4H/1H/15m), order-book depth, and regime classification. Every price level (entry, stop, target) is grounded in real ATR and pivots.
2. AI Brain (a reasoning LLM served via Seek AI): the DECISION AUTHORITY, not a reviewer -- it decides direction and whether to trade at all, reasoning over the deterministic engine's verified evidence. It never invents a number; every price comes from the math engine. If asked what model you are, say you run on a reasoning LLM via Seek AI and that the served model can vary -- never claim a specific model identity you cannot verify. When the brain is unreachable, the deterministic gate decides and the answer says so.

You also integrate: TradingView institutional intelligence (live TA, pivots Pivot/S1-S3/R1-R3, Bollinger-squeeze, exchange-wide volume breakouts across Crypto, Forex/Commodities, Equities) and an autonomous closed-loop memory engine that tracks live trades to TP/SL and learns rules from losses.

Conversation principles:
- Answer ANY question -- markets, general knowledge, casual chat, coding, explanations. You are never only a trading bot and never only a chatbot. Never deflect an off-topic question; answer what was asked, then offer a relevant action if one fits.
- Engage directly with the user's actual point. If they push back ("you only ever say DOWN"), address THAT specific claim with reasoning -- agree, disagree, or nuance it. Never respond to a real question with a menu of commands.
- Give a directional read on any liquid market named (crypto, forex, metals, indices, equities). When the edge is thin, say so plainly and call it a low-conviction lean rather than refusing; decline only when there is genuinely no basis (no data, hard timeframe conflict, or extreme event risk).
- NEVER quote, promise, or imply a win rate or "guaranteed"/"perfect" accuracy. If conviction is low, say conviction is low. Honest abstention (NO_TRADE) is a feature, not a failure.
- Pure greetings ("hi", "hello"): warm one-liner introducing yourself and inviting an asset to analyze. A greeting that PREFIXES a real question is a real question -- answer it.
- Handy commands, mention only when relevant: `BTC 15m` (signal), `tv btc` (TradingView TA & pivots), `tv breakouts` (volume gainers), `active trades`, `status`.
- Markdown, concise, no disclaimer padding. Remind users when relevant that QuantEdge is an analysis gateway, not an execution broker.
"""


def _generate_contextual_fallback(user_text: str, platform_ctx: str) -> str:
    """Generate an intelligent, question-specific response when upstream LLM is experiencing transient delays."""
    lower = user_text.lower().strip()

    # 1. Friendly greetings -- but ONLY when the message is essentially JUST a
    # greeting. A greeting-PREFIXED real question ("hey, why are you only giving
    # DOWN signals?") must fall through to the substantive branches below: a bare
    # "hey" matching here and returning the canned hello, swallowing the actual
    # question, was exactly the reported "bot ignores what I asked" bug.
    greeting_re = r"\b(hi|hello|hey|good\s+morning|good\s+evening|good\s+day|sup|yo|howdy|greetings)\b"
    if re.search(greeting_re, lower):
        residue = re.sub(greeting_re, " ", lower)
        residue = re.sub(r"[^a-z0-9]+", " ", residue).strip()
        # A "?" is a hard tell the user asked something; so is any real content
        # left once the greeting words are removed. Keep the pure-greeting reply
        # only for "hi" / "hello there" / "good morning" and the like.
        if "?" not in user_text and len(residue.split()) <= 2:
            return (
                "Hello! I'm **QuantEdge AI**, your dual-brain quantitative trading intelligence assistant.\n\n"
                "I'm ready to analyze markets, scan technical indicators, or discuss trading strategies. "
                "How can I help your market analysis today? You can command me with `BTC 15m`, `tv btc`, "
                "`gold 10m`, or ask me any questions about trading indicators and platform strategies!"
            )

    # 2. Specific Technical Indicator Inquiries
    if "rsi" in lower:
        return (
            "**Relative Strength Index (RSI)** in QuantEdge AI:\n\n"
            "- **Institutional Calibration**: We use a standard 14-period RSI combined with TradingView multi-oscillator consensus.\n"
            "- **Key Thresholds**: Above 70 denotes overbought conditions (potential mean-reversion short), while below 30 denotes oversold conditions.\n"
            "- **Signal Integration**: We never trade RSI in isolation. RSI momentum must align with our 200 EMA trend regime to validate a directional entry.\n\n"
            "Try typing `tv btc` to view live RSI and momentum readings on Bitcoin right now!"
        )

    if "macd" in lower:
        return (
            "**Moving Average Convergence Divergence (MACD)**:\n\n"
            "- **Formula**: Evaluates the relationship between the 12-period fast EMA and 26-period slow EMA, smoothed by a 9-period signal line.\n"
            "- **Histogram**: QuantEdge measures histogram expansion to gauge accelerating institutional momentum before confirming breakouts.\n\n"
            "Type `tv eth` to inspect live MACD momentum on Ethereum!"
        )

    if any(k in lower for k in ("ema", "exponential moving average", "moving average")):
        return (
            "**200 EMA Trend Regime Filter**:\n\n"
            "- **Trend Baseline**: The 200 Exponential Moving Average is our foundational gate across all timeframes.\n"
            "- **Directional Gate**: Long setups are strictly restricted to prices holding above the 200 EMA, while Short setups require price below the EMA.\n"
            "- **Multi-Timeframe Alignment**: We enforce trend agreement across execution (e.g. 15m), confirmation (1H), and regime (4H) timeframes.\n\n"
            "Type `BTC 15m` to see how the 200 EMA filters our active market setups!"
        )

    if "atr" in lower:
        return (
            "**Average True Range (ATR) & Dynamic Risk**:\n\n"
            "- **Mathematical Levels**: Instead of arbitrary fixed pips or percentages, QuantEdge places the stop at 1.5x ATR (widened to clear the nearest structural level, capped at 2.5x ATR).\n"
            "- **Volatility Normalization**: This ensures stops are positioned outside normal market noise instead of at a fixed percentage that means different things in quiet and volatile tape.\n\n"
            "Type `gold 10m` to see ATR-calculated stop and target levels in action!"
        )

    if any(k in lower for k in ("bollinger", "squeeze", "bands")):
        return (
            "**Bollinger Bands & Institutional Squeeze**:\n\n"
            "- **Band Squeeze**: When Bollinger Bands contract inside Keltner Channels, it signals volatility compression before an explosive breakout.\n"
            "- **TradingView Integration**: Our MCP server actively scans for active squeeze patterns across crypto and forex markets.\n\n"
            "Type `tv breakouts` to discover markets currently breaking out of volatility squeezes!"
        )

    if any(k in lower for k in ("pivot", "pivots", "s1", "r1")):
        return (
            "**Institutional Floor Pivots**:\n\n"
            "- **Calculations**: Daily pivot $(H+L+C)/3$ with support ($S_1, S_2, S_3$) and resistance ($R_1, R_2, R_3$) levels.\n"
            "- **TradingView Feed**: We pull exact institutional pivot levels directly through our TradingView MCP integration to use as commonly-defended price targets.\n\n"
            "Type `tv pivots btc` to view institutional pivot levels for Bitcoin!"
        )

    # 3. Strategy and Risk Management
    if any(k in lower for k in ("risk", "reward", "r:r", "ratio", "stop loss", "take profit", "money management")):
        return (
            "**Institutional Risk Architecture**:\n\n"
            "- **Minimum $R:R \\ge 1.2$**: Every trade recommendation must clear a 1.2:1 "
            "reward-to-risk floor. A setup below it is declined (NO_TRADE), never stretched to "
            "hit a nicer ratio -- the target is the opposing structural level, or where none is "
            "in reach it is flagged as derived from the stop rather than measured.\n"
            "- **Hard Invalidation Gates**: Candidates that fail event-risk checks, data quality thresholds, or multi-timeframe alignment are immediately vetoed (NO_TRADE).\n"
            "- **Conviction tiers**: Setups that pass the gates are sized by evidence breadth -- A+ Prime (full), A Strong (reduced), B Scalp (small, low-conviction) -- never by a quoted win probability, which we do not claim.\n\n"
            "Try generating a trade setup with `ETH 15m` or `USDJPY 5m`!"
        )

    # 4. Questions about active trades / positions
    if any(phrase in lower for phrase in ("active trades", "open trades", "in flight", "current trades", "open positions")):
        try:
            from quantedge.services.lifecycle import get_active_signals_summary
            summary = get_active_signals_summary()
            count = summary.get("total_active", 0)
            if count == 0:
                return "There are currently no active in-flight trades open. You can generate a new high-conviction setup by typing e.g. `BTC 15m`!"
            signals = summary.get("signals", [])
            lines = [f"- **{s['symbol']}** ({s.get('direction', 'TRADE')}) at {s['reference_price']} (expires in {s['remaining_minutes']}m)" for s in signals[:5]]
            return f"Currently monitoring **{count} active in-flight trade(s)**:\n" + "\n".join(lines)
        except Exception:
            return "You can check all in-flight positions and automated settlements anytime by typing `active trades`."

    # 5. Questions about performance, win rate, or memory
    if any(phrase in lower for phrase in ("win rate", "performance", "memory bank", "what have you learned", "learned rules", "track record", "past trades")):
        try:
            from quantedge.services.memory import get_memory_bank_summary
            mem = get_memory_bank_summary()
            total = mem.get("total_memories", 0)
            wins = mem.get("wins", 0)
            losses = mem.get("losses", 0)
            rules = mem.get("recent_rules", [])
            rules_str = "\n".join(f"- {r}" for r in rules[:3]) if rules else "No recurring failure rules yet."
            return (
                f"**Autonomous Memory Bank Status**:\n"
                f"- Total Settled Trades: {total}\n"
                f"- Wins: {wins} | Losses: {losses}\n\n"
                f"**Active Learned Rules**:\n{rules_str}\n\n"
                f"Type `what have you learned` to inspect the full trade journal."
            )
        except Exception:
            return "Type `what have you learned` to inspect the complete memory journal and learned rules."

    # 6. Questions about markets / supported assets
    if any(phrase in lower for phrase in ("markets", "assets", "symbols", "what coins", "what crypto", "supported pairs")):
        return (
            "I support three institutional asset classes across global exchanges:\n\n"
            "- **Crypto**: Bitcoin (`BTCUSDT`), Ethereum (`ETHUSDT`), Solana (`SOLUSDT`), Binance Coin (`BNBUSDT`), Ripple (`XRPUSDT`).\n"
            "- **Forex**: Euro (`EURUSD`), British Pound (`GBPUSD`), Japanese Yen (`USDJPY`), Swiss Franc (`USDCHF`), Canadian Dollar (`USDCAD`).\n"
            "- **Commodities**: Gold (`XAUUSD`), Silver (`XAGUSD`), Crude Oil (`WTICOUSD`).\n\n"
            "You can request trades on horizons from 1m up to 1h (e.g. `gold 10m` or `USDJPY 5m`)."
        )

    # 7. Explicit questions about the bot identity / architecture
    if any(phrase in lower for phrase in (
        "what is quantedge", "who are you", "what is this bot", "how does the bot work",
        "how does this bot work", "tell me about this bot", "what can you do", "bot features",
        "how does the platform work", "dual brain", "architecture"
    )):
        return (
            "I am **QuantEdge AI**, an institutional-grade quantitative trading platform featuring a **Dual-Brain Architecture**:\n\n"
            "1. **Mathematical Quant Engine**: Analyzes 200 EMA trend alignment, ATR volatility bands, multi-timeframe consensus (15m, 1H, 4H, 1D), and order book volume delta. It grounds every price level from real ATR and pivots.\n"
            "2. **AI Brain (a reasoning LLM via Seek AI)**: The decision authority -- it decides the direction and whether to trade at all from the engine's real evidence, and conducts post-mortem diagnostics on resolved trades. It never invents a price; the levels come from the math engine, and if the brain is unreachable the deterministic gate decides.\n"
            "3. **TradingView FastMCP Tools**: Live technical summaries, floor pivots (S1-S3, R1-R3), and volume breakout screeners.\n"
            "4. **Autonomous Memory Engine**: Automatically settles active trades and learns DO/DON'T rules from losses to improve over time.\n\n"
            "Try commanding me with `BTC 15m`, `tv btc`, `gold 10m`, or `active trades`!"
        )

    # 8. General fallback -- reached only when the AI brain timed out or errored.
    # It must NOT dump a command menu as if ignoring the question (that was the
    # reported bug): acknowledge the actual question, be honest that the reasoned
    # take is momentarily unavailable, and offer a retry plus compact options.
    return (
        f"On your question — *\"{user_text}\"* — my AI brain took longer than the response "
        "window allowed just now, so I couldn't finish the full reasoned answer this time. "
        "This is a transient latency spike on the reasoning model, not a refusal — **ask me "
        "again in a few seconds** and it usually comes straight back.\n\n"
        "In the meantime I can run something deterministic and instant for you:\n"
        "- A grounded signal — e.g. `BTC 15m`, `gold 10m`, `EURGBP 5m`\n"
        "- Live TradingView TA & pivots — `tv btc`\n"
        "- Volume breakouts — `tv breakouts`  ·  in-flight trades — `active trades`"
    )


def _handle_conversation(message: str, state: dict[str, Any]) -> ChatReply:
    """Handle freeform conversational messages and user questions using the AI Brain."""
    text = message.strip()
    history = state.get("conversation_history", [])

    # Fetch live ground-truth platform context to ground the AI's answers
    try:
        from quantedge.services.platform_context import build_platform_context
        platform_context = build_platform_context()
    except Exception as ctx_err:
        log.debug("could not build dynamic platform context", extra={"error": str(ctx_err)})
        platform_context = ""

    # Conditionally include platform context on platform-specific questions to keep general chats fast
    is_platform_query = any(w in text.lower() for w in ("platform", "bot", "active", "position", "rule", "learned", "status", "performance", "system", "engine", "architecture"))
    if is_platform_query and platform_context:
        full_system_prompt = f"{_CONVERSATION_SYSTEM_PROMPT}\n\n{platform_context}"
    else:
        full_system_prompt = _CONVERSATION_SYSTEM_PROMPT

    reason_for_fallback = "AI brain was not consulted"
    try:
        from quantedge.providers.llm import default_llm_provider

        provider = default_llm_provider()
        if provider is not None and hasattr(provider, "generate_chat_reply"):
            reply_text = provider.generate_chat_reply(
                message=text,
                conversation_history=history,
                system_prompt=full_system_prompt,
            )
            if reply_text and reply_text.strip():
                # Update conversational history (keep last 8 turns)
                history.append({"role": "user", "content": text})
                history.append({"role": "assistant", "content": reply_text})
                state["conversation_history"] = history[-8:]

                return ChatReply(
                    text=reply_text,
                    intent=Intent.CONVERSATION,
                    data={"conversation": True, "provider": getattr(provider, "provider_name", "ai")},
                )
            # The brain answered with nothing. This path used to fall through in
            # silence, so a blank reply was indistinguishable from "no brain
            # configured" and the canned fallback looked like the bot ignoring the
            # question. Record why so the fallback is never an unexplained mystery.
            reason_for_fallback = "AI brain returned an empty reply"
            log.warning("conversational AI returned an empty reply; using grounded fallback")
        else:
            reason_for_fallback = "no AI brain provider is configured"
            log.warning("no conversational AI provider available; using grounded fallback")
    except Exception as exc:
        reason_for_fallback = f"AI brain unavailable ({type(exc).__name__})"
        log.warning(
            "conversational AI reply unavailable; falling back to grounded response",
            extra={"error": str(exc)},
        )

    # Grounded question-specific fallback if AI endpoint is experiencing transient network/quota delay
    fallback_text = _generate_contextual_fallback(text, platform_context)
    return ChatReply(
        text=fallback_text,
        intent=Intent.CONVERSATION,
        data={"conversation": True, "fallback": True, "fallback_reason": reason_for_fallback},
    )

