"""FastAPI routes for market data, indicators, scanning, and signals."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from quantedge.contracts import (
    AIDecision,
    DataQualityReport,
    FeatureSnapshot,
    PerformanceSummary,
    ProviderHealth,
    RegimeReport,
    ScanResult,
    StructureReport,
    Timeframe,
    TradeMemory,
    TradeRecommendation,
)
from quantedge.errors import QuantEdgeError, ValidationError
from quantedge.providers.registry import get_registry
from quantedge.services import (
    chat as bot_chat,
)
from quantedge.services import (
    indicators as ind,
)
from quantedge.services import (
    quality as qual,
)
from quantedge.services import (
    regime as reg,
)
from quantedge.services import (
    scanner as scan,
)
from quantedge.services import (
    settlement as setl,
)
from quantedge.services import (
    signal as sig,
)
from quantedge.services import (
    structure as st,
)
from quantedge.services.horizons import available_time_limits

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=list[ProviderHealth])
def health_check() -> list[ProviderHealth]:
    """Check health across all configured providers."""
    registry = get_registry()
    return registry.health_check_all()


@router.get("/quote")
def get_quote(symbol: str, provider: str = "binance") -> Any:
    """Fetch live quote for a symbol."""
    registry = get_registry()
    quote = registry.get_quote(symbol, provider_name=provider)
    if not quote:
        raise HTTPException(status_code=404, detail=f"Quote unavailable for symbol {symbol}")
    return quote


@router.get("/candles")
def get_candles(
    symbol: str,
    timeframe: str = "1m",
    limit: int = Query(default=100, ge=1, le=1000),
    provider: str = "binance",
) -> Any:
    """Fetch closed candle history."""
    registry = get_registry()
    tf = Timeframe(timeframe)
    candles = registry.get_candles(symbol, tf, limit=limit, provider_name=provider)
    return candles


@router.get("/indicators", response_model=FeatureSnapshot)
def get_indicators(symbol: str, timeframe: str = "1m", limit: int = 100) -> FeatureSnapshot:
    """Compute feature snapshot from closed candles."""
    registry = get_registry()
    tf = Timeframe(timeframe)
    candles = registry.get_candles(symbol, tf, limit=limit)
    if not candles:
        raise HTTPException(status_code=404, detail="No closed candles found")
    return ind.compute_features(candles)


@router.get("/structure", response_model=StructureReport)
def get_structure(symbol: str, timeframe: str = "1m", limit: int = 100) -> StructureReport:
    """Analyze price structure and confirmed swings."""
    registry = get_registry()
    tf = Timeframe(timeframe)
    candles = registry.get_candles(symbol, tf, limit=limit)
    if not candles:
        raise HTTPException(status_code=404, detail="No closed candles found")
    return st.analyze_structure(candles)


@router.get("/regime", response_model=RegimeReport)
def get_regime(symbol: str, timeframe: str = "1m", limit: int = 100) -> RegimeReport:
    """Classify 9-state market regime."""
    registry = get_registry()
    tf = Timeframe(timeframe)
    candles = registry.get_candles(symbol, tf, limit=limit)
    if not candles:
        raise HTTPException(status_code=404, detail="No closed candles found")
    feats = ind.compute_features(candles)
    struct = st.analyze_structure(candles, atr=feats.atr_14)
    # ``classify_regime_from_features`` is the wrapper the scanner uses. The
    # earlier call here passed the candle series positionally into the ``structure``
    # slot of the low-level function, so this route raised TypeError on every
    # request. Going through one entry point keeps the route's answer identical
    # to the scanner's for the same bars.
    return reg.classify_regime_from_features(structure=struct, features=feats)


@router.get("/quality", response_model=DataQualityReport)
def get_quality(symbol: str, timeframe: str = "1m", limit: int = 100) -> DataQualityReport:
    """Evaluate 15 data quality checks."""
    registry = get_registry()
    tf = Timeframe(timeframe)
    candles = registry.get_candles(symbol, tf, limit=limit)
    return qual.evaluate_quality(candles, expected_timeframe=tf)


@router.post("/scan", response_model=ScanResult)
def run_scanner(symbols: list[str], horizon: str = "swing") -> ScanResult:
    """Execute candidate scan across requested symbols."""
    registry = get_registry()
    return scan.run_scan(symbols, horizon=horizon, registry=registry)


@router.post("/signal/evaluate", response_model=AIDecision)
def evaluate_signal(symbol: str, horizon: str = "swing") -> AIDecision:
    """Generate AI signal decision for a symbol."""
    return sig.generate_signal_decision(symbol, horizon=horizon)


@router.get("/performance", response_model=PerformanceSummary)
def get_performance(symbol: str | None = None, horizon: str | None = None) -> PerformanceSummary:
    """Fetch realized historical performance summary."""
    return setl.get_performance_summary(symbol=symbol, horizon=horizon)


# ------------------------------------------------------------------ #
# AI Trading Bot & Memory System Endpoints                            #
# ------------------------------------------------------------------ #


@router.post("/bot/trade-recommendation", response_model=TradeRecommendation)
def get_bot_trade_recommendation(
    symbol: str = Query(default="BTCUSDT"),
    time_limit: str = Query(default="15m", description='e.g. "10 min", "20", "1h"'),
    asset_class: str = Query(default="crypto"),
) -> TradeRecommendation:
    """Generate a memory-augmented recommendation for a chosen time limit.

    ``time_limit`` is the duration the trade will be held for; it resolves to a
    configured horizon so the timeframes analysed and the expiry quoted agree.
    A declined setup is a 409 carrying the reason, not an empty 200 -- an empty
    success reads as a system fault rather than a decision not to trade.
    """
    try:
        return sig.generate_trade_recommendation(
            symbol, time_limit=time_limit, asset_class=asset_class
        )
    except sig.NoTradeReason as exc:
        raise HTTPException(
            status_code=409,
            detail={"status": exc.status.value, "reason": exc.reason, "detail": exc.detail},
        ) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    except QuantEdgeError as exc:
        raise HTTPException(status_code=503, detail=f"{exc.code}: {exc.message}") from exc


class TradeFeedback(BaseModel):
    """A settled trade, reported by the trader.

    ``direction``, ``reference_price``, ``stop``, ``target`` and ``entry_time``
    are what make a loss diagnosable: they bound the holding period and give the
    post-mortem the levels to measure against. Supply them and the recorded
    cause is measured; omit them and the memory says plainly that the cause
    could not be determined, which is the honest alternative to a template.
    """

    model_config = ConfigDict(extra="forbid")

    signal_id: str = Field(min_length=1, max_length=64)
    outcome: str = Field(pattern="^(?i)(WIN|LOSS|FLAT|VOID|PENDING)$")
    symbol: str = Field(default="BTCUSDT", max_length=20)
    asset_class: str = "crypto"
    horizon: str = "15m"
    regime: str = "UNCERTAIN"
    pattern: str = "general"
    direction: str | None = Field(default=None, pattern="^(?i)(UP|DOWN)$")
    reference_price: Decimal | None = None
    exit_price: Decimal | None = None
    stop: Decimal | None = None
    target: Decimal | None = None
    entry_time_utc: datetime | None = None
    expiry_utc: datetime | None = None
    user_notes: str | None = Field(default=None, max_length=2000)


@router.post("/bot/feedback", response_model=TradeMemory)
def post_trade_feedback(feedback: TradeFeedback) -> TradeMemory:
    """Record a settled trade, diagnosing the holding period when it lost.

    The closed bars between entry and expiry are fetched here and handed to the
    post-mortem, so the stored cause is a measurement of this trade rather than
    a sentence chosen by its outcome.
    """
    from quantedge.contracts import SettlementOutcome
    from quantedge.services import memory as mem

    period = bot_chat.holding_period_for(
        symbol=feedback.symbol,
        horizon=feedback.horizon,
        entry_time=feedback.entry_time_utc,
        expiry=feedback.expiry_utc,
    )

    return mem.record_trade_outcome_and_analyze(
        feedback.signal_id,
        SettlementOutcome(feedback.outcome.upper()),
        symbol=feedback.symbol,
        asset_class=feedback.asset_class,
        horizon=feedback.horizon,
        regime=feedback.regime,
        pattern=feedback.pattern,
        direction=feedback.direction.upper() if feedback.direction else None,
        reference_price=feedback.reference_price,
        exit_price=feedback.exit_price,
        stop=feedback.stop,
        target=feedback.target,
        holding_candles=period.candles,
        entry_time=feedback.entry_time_utc,
        entry_structure=period.entry_structure,
        exit_structure=period.exit_structure,
        entry_features=period.entry_features,
        exit_features=period.exit_features,
        user_notes=feedback.user_notes,
    )


@router.get("/bot/memories", response_model=list[TradeMemory])
def get_bot_memories(
    symbol: str | None = None,
    regime: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[TradeMemory]:
    """Retrieve stored post-mortem memories from the bot's memory bank."""
    from quantedge.services import memory as mem

    return mem.get_relevant_memories(symbol=symbol, regime=regime, limit=limit)


@router.get("/bot/memory-stats")
def get_bot_memory_stats() -> dict[str, Any]:
    """Fetch memory bank performance statistics and learned DO/DONT rules."""
    from quantedge.services import memory as mem

    return mem.get_memory_bank_summary()


# ------------------------------------------------------------------ #
# Chat                                                                #
# ------------------------------------------------------------------ #

# Conversation working memory, keyed by session id. This holds the trade a
# follow-up like "that one lost" refers to -- entry, stop, target and the time
# it opened -- which is what makes the loss diagnosable rather than guessed.
#
# Deliberately separate from the trade memory bank: an in-flight conversation is
# not evidence, and letting it write into the bank would file trades that were
# never settled. It is also deliberately in-process and bounded: it is scratch
# state, and losing it on restart costs a user one re-request.
_SESSIONS: dict[str, dict[str, Any]] = {}
_MAX_SESSIONS = 500


class ChatRequest(BaseModel):
    """One chat turn."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=64)
    symbol: str = Field(default="BTCUSDT", max_length=20)
    minutes: int | None = Field(default=None, ge=1, le=1440)


def _session(session_id: str) -> dict[str, Any]:
    """Working memory for one conversation, created on first use."""
    state = _SESSIONS.get(session_id)
    if state is None:
        if len(_SESSIONS) >= _MAX_SESSIONS:
            # Drop the oldest rather than growing without bound. Dicts preserve
            # insertion order, so this evicts the least recently created.
            _SESSIONS.pop(next(iter(_SESSIONS)))
        state = _SESSIONS[session_id] = {}
    return state


@router.post("/bot/chat")
def post_bot_chat(request: ChatRequest) -> dict[str, Any]:
    """Answer one chat message.

    Intent is classified in code and every number in the reply comes from the
    deterministic pipeline. A failure here returns the failure -- there is no
    path that substitutes a plausible answer for an unavailable one.
    """
    try:
        reply = bot_chat.handle_message(
            request.message,
            default_symbol=request.symbol,
            default_minutes=request.minutes,
            session=_session(request.session_id),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    except QuantEdgeError as exc:
        raise HTTPException(status_code=503, detail=f"{exc.code}: {exc.message}") from exc
    return reply.to_dict()


@router.get("/bot/time-limits")
def get_time_limits() -> list[dict[str, str]]:
    """Get offered validity windows and their corresponding horizon names."""
    return [
        {"limit": str(limit), "horizon": horizon.name}
        for limit, horizon in config.HORIZON_MAP.items()
    ]


@router.get("/debug-db")
def debug_db():
    try:
        from quantedge.repositories.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            res = conn.execute(text("SELECT 1")).scalar()
            return {"status": "ok", "result": res}
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
@router.get("/debug-db")
def debug_db():
    try:
        from quantedge.repositories.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            res = conn.execute(text("SELECT 1")).scalar()
            return {"status": "ok", "result": res}
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
