"""End-to-end signal engine.

Coordinates scanning, context assembly, LLM review and persistence.

The contract this module holds
------------------------------
A recommendation is only ever returned when the deterministic scanner produced a
candidate *and* real risk levels could be derived from real volatility. When the
scan finds nothing the caller receives ``NO_TRADE`` / ``INSUFFICIENT_DATA`` and
no levels -- not a setup assembled from fallback constants.

This is worth stating because the previous implementation did the opposite: it
substituted ``SignalDirection.UP`` for a missing direction, ``Decimal("50000")``
for a missing price, multiplied those by fixed percentages to get a stop and
target, and asserted ``risk_reward_ratio=2.0`` over the result. A user reading
that output could not tell a genuine setup from the scanner declining to find
one, which is the single most consequential thing they need to know.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from quantedge.contracts import (
    AIDecision,
    AssetClass,
    ConvictionTier,
    DataQualityReport,
    LLMSignalResponse,
    QualityStatus,
    SignalDirection,
    SignalStatus,
    TradeRecommendation,
    utc_now,
)
from quantedge.config import decision_mode
from quantedge.errors import QuantEdgeError
from quantedge.logging import get_logger
from quantedge.providers.llm.base import BaseLLMProvider
from quantedge.repositories import get_repository
from quantedge.services.horizons import (
    expiry_for,
    horizon_minutes,
    resolve_time_limit,
)
from quantedge.services.llm_review import validate_llm_response
from quantedge.services.risk import (
    MIN_ACCEPTABLE_RR,
    SIZE_FRACTION_BY_TIER,
    derive_risk_levels,
)
from quantedge.services.scanner import run_scan
from quantedge.services.signal_context import build_signal_context
from quantedge.symbols import asset_class_for

if TYPE_CHECKING:
    from quantedge.services.scanner import ScanResult

__all__ = [
    "NoTradeReason",
    "generate_best_trade_recommendation",
    "generate_signal_decision",
    "generate_trade_recommendation",
]

log = get_logger(__name__)


class NoTradeReason(Exception):
    """Raised when no honest recommendation can be produced.

    Carries the machine-readable reason so callers -- the chat service in
    particular -- can explain *why* nothing was returned instead of showing an
    empty result, which reads as a system failure rather than a decision.
    """

    def __init__(
        self,
        status: SignalStatus,
        reason: str,
        *,
        detail: str = "",
        watch_plan: str = "",
    ) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.detail = detail
        # An optional, purely conditional watch plan built from REAL confirmed
        # structural levels. It never asserts a direction the evidence does not
        # support and never quotes odds: it names the price that, if reached,
        # would give the setup something to work with. Empty when no confirmed
        # structure exists (never fabricated to fill the space).
        self.watch_plan = watch_plan


# Symbols with a native Binance USDT candle feed. Everything else -- forex, metals,
# indices, equities, and exotic crypto without a Binance pair -- has no candle feed
# here and must be decided from live TradingView evidence, not the candle scan.
_BINANCE_NATIVE_PREFIXES = (
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX",
    "DOT", "LINK", "MATIC", "POL", "LTC", "NEAR", "SUI", "PEPE", "SHIB", "TRX",
)


def _is_binance_native(symbol: str) -> bool:
    clean = symbol.upper().replace("/", "").replace("-", "")
    return clean.endswith("USDT") and any(clean.startswith(c) for c in _BINANCE_NATIVE_PREFIXES)


def generate_signal_decision(
    symbol: str,
    *,
    horizon: str = "15m",
    provider_name: str | None = None,
    llm_provider: BaseLLMProvider | None = None,
    candle_fetcher: Any = None,
    allow_brain: bool = True,
) -> AIDecision:
    """Scan, review and persist a decision for one symbol.

    The returned :class:`AIDecision` reports the real outcome, including
    ``NO_TRADE`` with the scanner's own rejection reason. The data-quality status
    on the record is the status the quality engine actually returned.
    """
    # Non-crypto symbols (forex, metals, indices, equities, exotic crypto without a
    # Binance pair) have no candle feed for the deterministic scan below, so
    # run_scan cannot serve them -- the failure that made evaluate_signal return a
    # data-fetch error for e.g. XAUUSD. Route them through the same universal
    # TradingView path the recommendation entry uses, with GLM as the brain. An
    # honest abstention there surfaces as a real NO_TRADE / INSUFFICIENT_DATA
    # record, never a manufactured setup.
    if not _is_binance_native(symbol):
        from quantedge.services.tradingview import generate_tradingview_decision

        try:
            return generate_tradingview_decision(symbol, minutes=horizon_minutes(horizon))
        except NoTradeReason as ntr:
            is_no_trade = ntr.status is SignalStatus.NO_TRADE
            decision = AIDecision(
                decision_id=str(uuid.uuid4()),
                symbol=symbol,
                horizon=horizon,
                status=ntr.status,
                conviction_tier=ConvictionTier.STAND_ASIDE,
                contradictory_evidence=[ntr.reason] if is_no_trade else [],
                missing_information=[] if is_no_trade else [ntr.reason],
                created_at_utc=utc_now(),
            )
            _persist(get_repository(), decision)
            return decision

    repo = get_repository()
    scan = run_scan(
        [symbol],
        horizon=horizon,
        # Passing provider_name (or None) allows the provider registry to route
        # automatically based on the asset class of the symbol (crypto -> binance,
        # forex/commodity -> twelvedata).
        provider=provider_name,
        candle_fetcher=candle_fetcher,
    )
    candidate = scan.candidates[0] if scan.candidates else None
    quality = _quality_report(scan, symbol)

    if candidate is None or quality is None:
        decision = _no_trade_decision(symbol, horizon, scan)
        _persist(repo, decision)
        return decision

    # Hand over the analysis the scan already did. Calling this with only the
    # candidate and the quality report meant the reviewer was told, on every
    # request, that the multi-timeframe snapshot and the regime report were "not
    # available" -- while the scan had just computed both to decide the direction
    # it was being asked to review. The reviewer then correctly reported a thin
    # evidence base and declined, so the review looked like caution when it was
    # really a plumbing gap.
    context = build_signal_context(
        candidate,
        quality,
        multi_timeframe=scan.mtf_snapshots.get(symbol),
        regime=scan.regime_reports.get(symbol),
    )

    provider = llm_provider if llm_provider is not None else _default_llm_provider()

    # llm_first (the default): the trained brain (GLM 5.3 Flash via Seek AI) is the
    # decision authority. Hand it the full verified evidence the scan just built --
    # every real indicator, the multi-timeframe views by role, the deterministic
    # lean and score -- and let IT decide the direction and whether to trade at
    # all, including flipping the deterministic lean or standing aside. The
    # deterministic candidate is not discarded: it still grounds every price level
    # (in generate_trade_recommendation, which grounds the returned direction) and
    # it is the tested fallback. When the brain is unreachable within budget, or
    # allow_brain is off (a rate-limit-bounded market sweep asks the brain about the
    # single best candidate only), we fall through to the conservative reviewer path
    # below unchanged. deterministic_first skips this branch entirely and keeps the
    # reviewer as a veto-only guard -- the pre-inversion behaviour, one env flip away.
    # Authority of the decision built on the fallthrough path below. It is
    # DETERMINISTIC by design (deterministic_first mode, or a rate-limited sweep
    # that deliberately did not consult the brain for this symbol). It becomes
    # DETERMINISTIC_FALLBACK only when the brain WAS the intended authority for this
    # symbol but returned no verdict within budget -- spec item 7: the deterministic
    # result must then be labelled a fallback, never read as the brain's own call.
    det_authority = "DETERMINISTIC"
    det_mode = decision_mode()
    det_fallback_reason: str | None = None
    if (
        allow_brain
        and det_mode == "llm_first"
        and provider is not None
        and callable(getattr(provider, "decide_trade", None))
    ):
        _brain_diag: dict[str, Any] = {}
        brain_decision = _glm_decide_crypto(
            provider,
            candidate,
            context,
            symbol=symbol,
            horizon=horizon,
            scan=scan,
            minutes=horizon_minutes(horizon),
            diag=_brain_diag,
        )
        if brain_decision is not None:
            _persist(repo, brain_decision, context=context)
            return brain_decision
        # The brain was the intended decision authority for this symbol but did not
        # return a usable verdict within the request budget (timeout, rate limit,
        # missing key, or malformed output). Mark the deterministic decision below
        # as a fallback so nothing downstream attributes it to the AI brain, and
        # carry the honest reason so an infra fault is visible, not hidden.
        det_authority = "DETERMINISTIC_FALLBACK"
        det_fallback_reason = _brain_diag.get("fallback_reason")

    validated = None
    # An ARMED read (a dead-chop directional lean or a coiled breakout plan) holds
    # NO live position: size is 0 and its labels already say "no edge / not yet
    # triggered". The conservative reviewer exists to veto *live* setups it
    # distrusts -- there is nothing to veto here, and letting it collapse the
    # labelled read into STAND_ASIDE would recreate the exact false abstention the
    # directive removes. So skip the review round-trip for ARMED and return the
    # honest, explicitly-low-conviction read as the scanner built it. B/A/A+ sized
    # setups keep the full reviewer veto.
    skip_review = candidate.conviction_tier is ConvictionTier.ARMED
    if provider is not None and not skip_review:
        try:
            validated = validate_llm_response(provider.evaluate_signal_context(context), context)
        except (QuantEdgeError, Exception) as exc:
            # A failed review must not upgrade into a signal or block execution.
            # The deterministic candidate stands on its own and is reported as such.
            log.warning(
                "llm review unavailable; returning the deterministic candidate",
                extra={"symbol": symbol, "error": str(exc)},
            )

    # Conservative-only review. The scanner owns the direction; the reviewer owns
    # the veto. A SIGNAL verdict that reverses the deterministic direction is
    # treated as a decline, never as a flipped trade -- otherwise the reviewer
    # could turn a correct deterministic setup into its opposite, which is the one
    # thing the documented contract says it must not do ("the LLM can only ever
    # make the system more conservative"). Enforced in code rather than trusted to
    # the prompt, because a directional flip is too costly to leave to model
    # compliance.
    if validated is not None:
        validated = _reconcile_review(validated, candidate.direction, symbol=symbol)

    # Carry the conviction tier the scanner assigned onto the decision. When the
    # reviewer vetoes (final status is not SIGNAL) there is no open position to
    # tier or size, so it collapses to STAND_ASIDE at zero size -- the tier field
    # must never advertise conviction for a trade the system is declining.
    final_status = validated.status if validated is not None else SignalStatus.SIGNAL
    if final_status is SignalStatus.SIGNAL:
        conviction_tier = candidate.conviction_tier
        size_fraction = candidate.position_size_fraction
        tier_rationale = candidate.tier_rationale
        upgrade_condition = candidate.upgrade_condition
    else:
        conviction_tier = ConvictionTier.STAND_ASIDE
        size_fraction = 0.0
        tier_rationale = "Reviewer vetoed the deterministic setup; standing aside."
        upgrade_condition = ""

    decision = AIDecision(
        decision_id=str(uuid.uuid4()),
        symbol=symbol,
        horizon=horizon,
        status=validated.status if validated is not None else SignalStatus.SIGNAL,
        direction=validated.direction if validated is not None else candidate.direction,
        reference_price=candidate.reference_price,
        expiry_utc=expiry_for(horizon_minutes(horizon), candidate.generated_at_utc),
        regime=candidate.regime.value,
        heuristic_score=candidate.heuristic_score,
        calibrated_probability=None,
        conviction_tier=conviction_tier,
        position_size_fraction=size_fraction,
        tier_rationale=tier_rationale,
        upgrade_condition=upgrade_condition,
        supporting_evidence=(
            validated.supporting_evidence
            if validated is not None
            else candidate.supporting_evidence
        ),
        contradictory_evidence=(
            validated.contradictory_evidence
            if validated is not None
            else candidate.contradictory_evidence
        ),
        invalidation_conditions=validated.invalidation_conditions if validated is not None else [],
        missing_information=validated.missing_information if validated is not None else [],
        llm_provider=provider.provider_name if provider is not None and validated else None,
        llm_model=provider.model_name if provider is not None and validated else None,
        decision_authority=det_authority,
        decision_mode=det_mode,
        llm_fallback_reason=det_fallback_reason,
        scanner_version=candidate.scanner_version,
        data_quality_status=_quality_status(scan, symbol),
        created_at_utc=utc_now(),
    )
    _persist(repo, decision, context=context, llm_response=validated)
    return decision


def generate_trade_recommendation(
    symbol: str,
    *,
    time_limit: str = "15m",
    asset_class: AssetClass | str | None = None,
    provider_name: str | None = None,
    candle_fetcher: Any = None,
    hold_minutes: int | None = None,
    include_watch_plan: bool = True,
    allow_brain: bool = True,
) -> TradeRecommendation:
    """Produce a memory-augmented recommendation, or raise :class:`NoTradeReason`.

    ``time_limit`` is the expiry the user picked ("10 min", "20m", "1h"); it is
    resolved to a configured horizon so the analysis timeframes and the stated
    expiry agree with each other.
    """
    from quantedge.services.horizons import horizon_minutes
    from quantedge.services.memory import get_relevant_memories, recurring_loss_rules
    from quantedge.services.tradingview import generate_tradingview_recommendation

    horizon = resolve_time_limit(time_limit)
    dur = hold_minutes if hold_minutes is not None else horizon_minutes(horizon)

    # Universal market routing: non-crypto or non-Binance pairs execute directly via TradingView MCP
    if not _is_binance_native(symbol):
        # Non-crypto (forex, metals, indices, equities) has no Binance candle feed;
        # TradingView is the only valid path. An honest abstention there IS the
        # answer -- propagate it rather than falling through to the crypto engine,
        # which would fetch the wrong series and manufacture a setup.
        try:
            return generate_tradingview_recommendation(symbol, minutes=dur)
        except NoTradeReason:
            raise
        except Exception as tv_exc:
            log.warning("TradingView direct recommendation failed for %s: %s", symbol, tv_exc)
            raise NoTradeReason(
                SignalStatus.INSUFFICIENT_DATA,
                f"no live market data reachable for {symbol}",
            ) from tv_exc

    try:
        decision = generate_signal_decision(
            symbol,
            horizon=horizon,
            provider_name=provider_name,
            candle_fetcher=candle_fetcher,
            allow_brain=allow_brain,
        )
    except Exception as exc:
        log.info("deterministic engine declined %s (%s); trying TradingView institutional analysis", symbol, exc)
        try:
            return generate_tradingview_recommendation(symbol, minutes=dur)
        except Exception:
            raise

    if decision.status is not SignalStatus.SIGNAL or decision.direction is None:
        # A deterministic NO_TRADE is a considered decision -- honour it. Only reach
        # for TradingView when the engine genuinely lacked data (INSUFFICIENT_DATA),
        # so the fallback adds evidence instead of overriding an abstention.
        if decision.status is SignalStatus.INSUFFICIENT_DATA:
            try:
                return generate_tradingview_recommendation(symbol, minutes=dur)
            except NoTradeReason:
                raise
            except Exception:
                pass

        # Which list holds the reason depends on why the decision came back.
        if decision.status is SignalStatus.INSUFFICIENT_DATA:
            reasons = list(decision.missing_information)
        else:
            reasons = list(decision.contradictory_evidence) or list(
                decision.missing_information
            )
        # A NO_TRADE is a directional decline, not a data fault, so it earns a
        # conditional watch plan built from real structure. INSUFFICIENT_DATA does
        # not: naming levels off a feed we could not read would be the exact
        # "infrastructure failure dressed as a decision" the honesty rules forbid.
        watch = ""
        if include_watch_plan and decision.status is SignalStatus.NO_TRADE:
            watch = _watch_plan(symbol, horizon, candle_fetcher)
        raise NoTradeReason(
            decision.status,
            (reasons or ["no setup met the configured criteria"])[0],
            detail="; ".join(reasons[1:4]),
            watch_plan=watch,
        )
    if decision.reference_price is None:
        try:
            return generate_tradingview_recommendation(symbol, minutes=dur)
        except Exception:
            raise NoTradeReason(
                SignalStatus.INSUFFICIENT_DATA,
                "no reference price was available from any provider",
            )

    # ARMED read: a size-0 conditional plan -- a dead-chop directional lean or a
    # primed-but-unbroken breakout. No live position exists, so the RR/target
    # gate below does not apply; forcing an ARMED read through it raises NO_TRADE
    # and silences exactly the "give me the lean, just say it's risky" answer the
    # product must give on every request. Return the direction honestly labelled
    # as low/no edge with its trigger, and NEVER a fabricated live stop or target
    # (honesty rule: no invented levels on a trade that does not exist).
    if decision.conviction_tier is ConvictionTier.ARMED:
        return _armed_recommendation(
            symbol,
            decision,
            horizon=horizon,
            dur=dur,
            asset_class=asset_class,
            candle_fetcher=candle_fetcher,
            include_watch_plan=include_watch_plan,
        )

    levels = _risk_levels_for(
        symbol, horizon, decision.direction, decision.reference_price, candle_fetcher
    )
    if levels is None:
        # No ATR (or too few closed bars) to place an honest stop. In llm_first the
        # brain has already named a real direction; degrade to a size-0 ARMED lean
        # that states plainly WHY no stop is shown, rather than refusing outright --
        # the warning names the missing input, so nothing is disguised. In
        # deterministic_first, keep the strict INSUFFICIENT_DATA decline.
        if decision_mode() == "llm_first":
            return _armed_recommendation(
                symbol,
                decision,
                horizon=horizon,
                dur=dur,
                asset_class=asset_class,
                candle_fetcher=candle_fetcher,
                include_watch_plan=include_watch_plan,
                tier_rationale=(
                    "ARMED / NO EDGE: the AI brain chose a direction, but ATR is "
                    "unavailable for this series, so no honest stop can be placed. "
                    "Directional lean only -- size 0."
                ),
                upgrade_condition="a clean volatility (ATR) read so a real stop and target can be derived",
                extra_warning=(
                    "No stop or target is shown because volatility (ATR) could not be "
                    "computed for this series -- none is invented for a trade that cannot yet be sized."
                ),
            )
        raise NoTradeReason(
            SignalStatus.INSUFFICIENT_DATA,
            "stop and target could not be derived: ATR is unavailable for this series",
        )

    # Unfavourable geometry is a decline, not a caveat. A setup whose target is
    # nearer than its stop needs to be right more often than not just to break
    # even, so emitting it with a warning attached invited exactly the trade the
    # ratio says to skip. The levels are real -- ATR-derived and structural --
    # which is why the fix is to refuse the trade rather than move the target.
    # In llm_first, the brain has still named a real direction: degrade to a
    # size-0 ARMED lean (no invented levels) so the read stays actionable and
    # honest, exactly as the TradingView engine does at its RR gate. In
    # deterministic_first, keep the strict NO_TRADE decline.
    if not levels.acceptable:
        if decision_mode() == "llm_first":
            return _armed_recommendation(
                symbol,
                decision,
                horizon=horizon,
                dur=dur,
                asset_class=asset_class,
                candle_fetcher=candle_fetcher,
                include_watch_plan=include_watch_plan,
                tier_rationale=(
                    f"ARMED / NO EDGE: the AI brain chose {decision.direction.value.lower()}, "
                    f"but the reward:risk on real levels ({levels.rr:.2f}) is below the "
                    f"{MIN_ACCEPTABLE_RR} minimum. Directional lean only -- size 0."
                ),
                upgrade_condition=(
                    f"the geometry improving so a reachable structural target clears "
                    f"reward:risk >= {MIN_ACCEPTABLE_RR}"
                ),
                extra_warning=(
                    f"Levels are real ({levels.basis}) but do not clear the reward:risk "
                    "minimum, so no sized setup is offered -- neither stop nor target is "
                    "moved to manufacture a better ratio."
                ),
            )
        if not levels.target_from_structure:
            raise NoTradeReason(
                SignalStatus.NO_TRADE,
                "no price level within reach to target",
                detail=(
                    f"the stop sits at {levels.stop} on volatility, but there is no "
                    "structural level ahead to aim at, so any target would be a "
                    "multiple of the stop rather than a place the market has defended"
                ),
            )
        raise NoTradeReason(
            SignalStatus.NO_TRADE,
            f"reward:risk {levels.rr:.2f} is below the {MIN_ACCEPTABLE_RR} minimum",
            detail=(
                f"stop {levels.stop} and target {levels.target} derived from {levels.basis}; "
                "moving either to improve the ratio would misreport the levels"
            ),
        )

    memories = get_relevant_memories(symbol=symbol, limit=20)
    lessons: list[str] = []
    for m in memories:
        lessons.extend(m.key_lessons[:2])
    lessons = list(dict.fromkeys(lessons))[:5]

    # Losses whose diagnosed cause has recurred at this horizon. These reach the
    # recommendation as caveats rather than as a score adjustment: the bank has
    # never been calibrated against unseen outcomes, so moving the number would
    # invent a precision that was never measured. Read instead as "this setup has
    # failed this way before on this symbol", which is exactly what was recorded.
    memory_rules = recurring_loss_rules(symbol, horizon=horizon)

    ast = _resolve_asset_class(symbol, asset_class)
    now = utc_now()
    dur = hold_minutes if hold_minutes is not None else horizon_minutes(horizon)
    expiry = expiry_for(dur, now)

    # The composite score as a percentage. Derived, not measured: it is exactly
    # decision.heuristic_score * 100, the same evidence-agreement score already
    # shown, surfaced as the "how sure" figure the user asked for. It is not a
    # calibrated win probability and is never presented as one (Rule 3).
    confidence_pct = round((decision.heuristic_score or 0.0) * 100)

    rationale = (
        f"{decision.regime} on the {horizon} horizon, heuristic score "
        f"{decision.heuristic_score:.2f}. Stop {levels.basis}; "
        f"reward:risk {levels.rr:.2f}."
    )
    if lessons:
        rationale += f" {len(memories)} past outcome(s) on {symbol} consulted."
    if memory_rules:
        rationale += f" {len(memory_rules)} recurring loss pattern(s) flagged from memory."

    # Caveats that qualify the setup without withdrawing it. A DEGRADED quality
    # report did not block the signal, but the trader is entitled to know the
    # feed was imperfect before committing to the trade.
    warnings: list[str] = []
    if decision.data_quality_status is QualityStatus.DEGRADED:
        warnings.append(
            "Data quality is DEGRADED for this series; the setup stands but the "
            "inputs are not clean."
        )
    # Short-horizon freshness honesty. The serverless deployment holds no
    # persistent market-data socket, so a 1m/3m setup is computed on the last
    # *closed* bar polled over REST -- not a live tick. That bar can be nearly a
    # full interval old at the moment the setup is read, which on a scalp is a
    # large fraction of the whole intended move. Disclosed rather than assumed,
    # and never described as live (Rule: never label delayed data as live).
    if horizon in {"1m", "3m"}:
        warnings.append(
            f"{horizon} scalp: computed from the last CLOSED {horizon} bar polled over "
            "REST, not a live tick stream. Treat the entry as that bar's close -- the "
            "bar can be up to one interval old, and slippage on a scalp this short can "
            "rival the edge itself."
        )
    warnings.extend(memory_rules)

    # Conviction tier from the deterministic scan, carried onto the trade the
    # user acts on. Coerced against None so a decision produced without a tier
    # (e.g. a future non-scan path) still lands on a defined, small-size B rather
    # than crashing the size-bounded contract field.
    conviction_tier = decision.conviction_tier or ConvictionTier.B
    size_fraction = (
        decision.position_size_fraction
        if decision.position_size_fraction is not None
        else 0.0
    )

    return TradeRecommendation(
        recommendation_id=f"rec-{uuid.uuid4().hex[:12]}",
        symbol=symbol,
        asset_class=ast,
        horizon=horizon,
        direction=decision.direction,
        valid_from_utc=now,
        valid_until_utc=expiry,
        reference_price=decision.reference_price,
        stop_loss=levels.stop,
        take_profit=levels.target,
        risk_reward_ratio=levels.rr,
        risk_level=_risk_level(decision.heuristic_score, levels.acceptable),
        recommended_venue=_venue_for(ast),
        regime=decision.regime,
        memory_consulted_count=len(memories),
        key_lessons_applied=lessons,
        memory_rules_applied=memory_rules,
        heuristic_score=decision.heuristic_score or 0.0,
        confidence_pct=confidence_pct,
        conviction_tier=conviction_tier,
        position_size_fraction=size_fraction,
        tier_rationale=decision.tier_rationale or "",
        upgrade_condition=decision.upgrade_condition or "",
        rationale=rationale,
        warnings=warnings,
        decision_authority=decision.decision_authority,
        decision_mode=decision.decision_mode,
        llm_provider=decision.llm_provider,
        llm_requested_model=decision.llm_requested_model,
        llm_response_model=decision.llm_response_model,
        model_verified=decision.model_verified,
        llm_latency_ms=decision.llm_latency_ms,
        llm_fallback_reason=decision.llm_fallback_reason,
        generated_at_utc=now,
    )


def generate_best_trade_recommendation(
    time_limit_minutes: int | None = None,
    provider_name: str | None = None,
    candle_fetcher: Any = None,
    alternatives_out: list[dict[str, Any]] | None = None,
) -> TradeRecommendation:
    """Scan major symbols to find the best setup, avoiding API rate limits.

    ``alternatives_out``, when given, is filled with the other candidates the
    sweep scored. Returning only the single top scorer made a balanced board look
    one-directional: the DOWN setups were found, ranked and then discarded before
    anyone saw them. The caller can now show what else was on the board without
    paying for a second scan.
    """
    from quantedge.services.scanner import run_scan
    from quantedge.symbols import supported_symbols

    # The crypto allowlist, which is what this deployment actually trades. An
    # earlier five-name list excluded the symbols that were signalling and also
    # named forex/metal pairs no healthy provider serves here, so the sweep was
    # searching a set that could not answer.
    supported = set(supported_symbols())
    # Prioritise the most liquid primary symbols to bound latency within serverless timeouts
    primary_crypto = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    all_syms = [s for s in primary_crypto if s in supported]
    if not all_syms:
        all_syms = [s for s in supported_symbols("crypto") if s in supported][:4]
    if not all_syms:
        all_syms = supported_symbols()[:4]

    if time_limit_minutes is not None:
        target_hz = resolve_time_limit(f"{time_limit_minutes}m")
        # Tailored search horizons for the user's desired trade duration (1m, 5m, 10m, 15m, etc.)
        if time_limit_minutes <= 3:
            search_horizons = ["1m", "3m", "5m"]
        elif time_limit_minutes <= 7:
            search_horizons = ["5m", "3m", "10m", "1m"]
        elif time_limit_minutes <= 12:
            search_horizons = ["10m", "5m", "15m"]
        elif time_limit_minutes <= 25:
            search_horizons = ["15m", "10m", "5m", "30m"]
        elif time_limit_minutes <= 45:
            search_horizons = ["30m", "15m", "1h"]
        else:
            search_horizons = ["1h", "30m", "15m"]
        horizons = [target_hz] + [h for h in search_horizons if h != target_hz]
    else:
        # Default representative horizons: prioritize fast, active intraday timeframes over 1h
        horizons = ["5m", "15m", "10m", "1h"]

    scored: list[tuple[float, str, str, str, str]] = []
    for hz in horizons:
        try:
            scan_res = run_scan(
                all_syms,
                horizon=hz,
                provider=provider_name,
                candle_fetcher=candle_fetcher,
            )
            scored.extend(
                (c.heuristic_score, c.symbol, hz, c.direction.value, c.conviction_tier.value)
                for c in scan_res.candidates
            )
        except Exception as exc:
            log.warning("Scan failed for horizon %s", hz, exc_info=exc)

    if not scored:
        raise NoTradeReason(
            SignalStatus.NO_TRADE,
            "No trade setups found across any symbol or timeframe right now.",
        )

    # Best first, then walk down. Conviction tier leads the ranking so an A+ prime
    # always outranks a B scalp even when their raw scores sit close: "the best
    # trade" should mean the most-supported one, not merely the highest number.
    # Score breaks ties within a tier. When a specific time limit is requested,
    # an exact horizon match takes precedence over everything else.
    tier_rank = {"A_PLUS": 3, "A": 2, "B": 1, "ARMED": 0, "STAND_ASIDE": 0}
    first_decline: NoTradeReason | None = None
    if time_limit_minutes is not None:
        target_hz = resolve_time_limit(f"{time_limit_minutes}m")
        ranked = sorted(
            scored,
            key=lambda row: (row[2] == target_hz, tier_rank.get(row[4], 0), row[0]),
            reverse=True,
        )
    else:
        ranked = sorted(
            scored,
            key=lambda row: (tier_rank.get(row[4], 0), row[0]),
            reverse=True,
        )

    for _rank_idx, (score, symbol, hz, _direction, _tier) in enumerate(ranked):
        try:
            rec = generate_trade_recommendation(
                symbol=symbol,
                time_limit=hz,
                provider_name=provider_name,
                candle_fetcher=candle_fetcher,
                hold_minutes=time_limit_minutes,
                include_watch_plan=False,
                # Rate-limit + serverless-timeout bound: consult the brain (one
                # decide_trade round-trip) on the single best-ranked candidate
                # only; the rest use the deterministic path. Seek AI throttles the
                # whole account to ~5 req/min, and the Vercel wall clock is 60s, so
                # one brain call per sweep is the honest ceiling.
                allow_brain=(_rank_idx == 0),
            )
        except NoTradeReason as exc:
            if first_decline is None:
                first_decline = exc
            continue

        if alternatives_out is not None:
            alternatives_out.extend(
                {
                    "symbol": s,
                    "horizon": h,
                    "direction": d,
                    "heuristic_score": sc,
                    "conviction_tier": tv,
                }
                for sc, s, h, d, tv in ranked
                if not (s == symbol and h == hz and sc == score)
            )
        return rec

    raise first_decline if first_decline is not None else NoTradeReason(
        SignalStatus.NO_TRADE,
        "No trade setups found across any symbol or timeframe right now.",
    )


# ---------------------------------------------------------------------- #
# helpers                                                               #
# ---------------------------------------------------------------------- #


def _tier_from_conviction(conviction: Any) -> tuple[ConvictionTier, float]:
    """Map the brain's conviction (0..1) to a conviction tier and relative size.

    Used only when the brain overrode the deterministic lean, or agreed with a
    candidate that was itself ARMED -- i.e. when there is no richer structure-aware
    tier from the scanner to keep. The size is the RELATIVE risk multiplier from
    ``SIZE_FRACTION_BY_TIER``, never a dollar amount or a win probability.
    """
    c = float(conviction) if isinstance(conviction, (int, float)) else 0.0
    if c >= 0.7:
        tier = ConvictionTier.A
    elif c >= 0.45:
        tier = ConvictionTier.B
    else:
        tier = ConvictionTier.ARMED
    return tier, SIZE_FRACTION_BY_TIER[tier]


def _tf_slice(view: Any) -> dict[str, Any]:
    """One timeframe's real indicator + structure read, for the brain's evidence.

    Every value is what the scan actually computed; ``None`` means the feature had
    insufficient warmup, which is stated as-is rather than filled with a default.
    """
    f = view.features
    s = view.structure
    return {
        "role": view.role,
        "timeframe": getattr(view.timeframe, "value", str(view.timeframe)),
        "bars_available": view.bars_available,
        "rsi_14": f.rsi_14 if f else None,
        "macd_histogram": f.macd_histogram if f else None,
        "adx_14": f.adx_14 if f else None,
        "ema_20_slope": f.ema_20_slope if f else None,
        "atr_percent": f.atr_percent if f else None,
        "bb_percent_b": f.bb_percent_b if f else None,
        "structure": s.structure if s else None,
        "nearest_support": str(s.nearest_support) if s and s.nearest_support is not None else None,
        "nearest_resistance": (
            str(s.nearest_resistance) if s and s.nearest_resistance is not None else None
        ),
    }


def _build_crypto_evidence(candidate: Any, context: Any, minutes: int) -> dict[str, Any]:
    """Assemble the full real evidence packet the brain decides on.

    Everything here is verified data the deterministic scan already produced: the
    per-role multi-timeframe views (execution / confirmation / regime) with their
    real indicators and structure, the deterministic lean and its component scores,
    the data-quality status and the event-risk classification. The brain is told
    what the maths concluded and why -- and is free to disagree. No price level is
    included for the brain to anchor on; levels are grounded downstream from ATR
    and real structure, never from anything the brain says.
    """
    mtf = context.multi_timeframe
    views = [_tf_slice(v) for v in mtf.views] if mtf is not None else []
    return {
        "symbol": candidate.symbol,
        "asset_class": getattr(candidate.asset_class, "value", str(candidate.asset_class)),
        "hold_minutes": minutes,
        "horizon": candidate.horizon,
        "price": str(candidate.reference_price),
        "regime": getattr(candidate.regime, "value", str(candidate.regime)),
        "deterministic_candidate": candidate.direction.value,
        "deterministic_playbook": candidate.playbook,
        "deterministic_score": round(candidate.heuristic_score, 3),
        "trend_score": round(candidate.trend_score, 3),
        "momentum_score": round(candidate.momentum_score, 3),
        "volatility_score": round(candidate.volatility_score, 3),
        "evidence_agreement": round(candidate.evidence_agreement_score, 3),
        "multi_timeframe": {
            "aligned_direction": (
                mtf.aligned_direction.value if mtf and mtf.aligned_direction else None
            ),
            "alignment_score": round(mtf.alignment_score, 3) if mtf else None,
            "participation": round(mtf.participation, 3) if mtf else None,
            "abstaining_roles": list(mtf.abstaining_roles) if mtf else [],
            "conflicts": list(mtf.conflicts) if mtf else [],
            "views": views,
        },
        "supporting_evidence": list(candidate.supporting_evidence[:6]),
        "contradictory_evidence": list(candidate.contradictory_evidence[:6]),
        "data_quality": getattr(context.quality.status, "value", None) if context.quality else None,
        "event_risk": getattr(candidate.event_risk, "value", None) if candidate.event_risk else None,
    }


def _glm_decide_crypto(
    provider: Any,
    candidate: Any,
    context: Any,
    *,
    symbol: str,
    horizon: str,
    scan: Any,
    minutes: int,
    diag: dict[str, Any] | None = None,
) -> AIDecision | None:
    """Let the brain decide this crypto symbol; return the AIDecision, or None.

    ``None`` means the brain was unreachable within budget (no key, timeout, rate
    limit, or a malformed response) -- the caller then falls through to the tested
    deterministic reviewer path. A returned decision is the brain's own verdict:
    an honest STAND_ASIDE ``NO_TRADE`` when it declines, or a ``SIGNAL`` carrying
    the direction it chose (which may flip the deterministic lean). The brain never
    names a price; the direction it returns is grounded into real levels downstream.

    When ``diag`` is provided and the brain does not return a usable verdict, the
    reason (exception type + message, or why it was skipped) is written to
    ``diag["fallback_reason"]`` so the caller can surface it -- an infra fault must
    never masquerade as a clean deterministic no-edge call.
    """
    decide = getattr(provider, "decide_trade", None)
    if not callable(decide):
        if diag is not None:
            diag["fallback_reason"] = "provider exposes no decide_trade()"
        return None
    evidence = _build_crypto_evidence(candidate, context, minutes)
    try:
        verdict = decide(evidence)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        log.warning(
            "AI brain decide_trade unavailable for %s (%s); deterministic reviewer decides",
            symbol,
            reason,
        )
        if diag is not None:
            diag["fallback_reason"] = reason
        return None

    gd = str(verdict.get("decision", "")).upper()
    brain = str(verdict.get("brain") or getattr(provider, "model_name", "glm"))
    reason = str(verdict.get("reason") or "")
    invalidation = str(verdict.get("invalidation") or "")
    conviction = verdict.get("conviction")
    quality_status = _quality_status(scan, symbol)
    provider_name = getattr(provider, "provider_name", None)

    # Model-identity honesty (spec items 4-7): the brain returns the model id the
    # server itself declared, and whether it belongs to the family we requested.
    # A False here means the endpoint substituted a different model (e.g. a
    # glm-5.3-flash request answered by MiniMax); the decision is still the brain's
    # to make, but it must NOT be attributed to the requested model. We relabel the
    # brain to the model that actually answered and mark the authority UNVERIFIED,
    # so no rationale string or persisted row claims GLM decided when it did not.
    requested_model = verdict.get("requested_model")
    response_model = verdict.get("response_model")
    model_verified = verdict.get("model_verified")
    latency_ms = verdict.get("latency_ms")
    mode = decision_mode()
    if model_verified is False:
        authority = "LLM_UNVERIFIED"
        brain = (response_model or f"{brain} (UNVERIFIED substitute)")
    else:
        authority = "LLM"

    if gd == "NO_TRADE":
        return AIDecision(
            decision_id=str(uuid.uuid4()),
            symbol=symbol,
            horizon=horizon,
            status=SignalStatus.NO_TRADE,
            reference_price=candidate.reference_price,
            regime=candidate.regime.value,
            heuristic_score=candidate.heuristic_score,
            conviction_tier=ConvictionTier.STAND_ASIDE,
            position_size_fraction=0.0,
            tier_rationale=f"AI brain ({brain}) stood aside." + (f" {reason}" if reason else ""),
            contradictory_evidence=[reason] if reason else ["AI brain declined this setup."],
            invalidation_conditions=[invalidation] if invalidation else [],
            llm_provider=provider_name,
            llm_model=brain,
            decision_authority=authority,
            decision_mode=mode,
            llm_requested_model=requested_model,
            llm_response_model=response_model,
            model_verified=model_verified,
            llm_latency_ms=latency_ms,
            scanner_version=candidate.scanner_version,
            data_quality_status=quality_status,
            created_at_utc=utc_now(),
        )

    # __GLM_SIGNAL_PATH__
    glm_dir = SignalDirection.UP if gd == "UP" else SignalDirection.DOWN

    # When the brain confirms a sized deterministic candidate, keep the scanner's
    # richer, structure-aware tier and size -- it was computed from the full
    # candidate, not just a scalar conviction. When the brain overrode the lean, or
    # the candidate was itself ARMED (no sized tier to keep), derive the tier from
    # the brain's conviction. Either way the tier is a relative multiplier, never a
    # win probability.
    if glm_dir is candidate.direction and candidate.conviction_tier is not ConvictionTier.ARMED:
        tier = candidate.conviction_tier
        size = candidate.position_size_fraction
        tier_rationale = candidate.tier_rationale or f"AI brain ({brain}) confirmed the {glm_dir.value} setup."
        upgrade_condition = candidate.upgrade_condition or ""
    else:
        tier, size = _tier_from_conviction(conviction)
        verb = "confirmed" if glm_dir is candidate.direction else f"overrode the deterministic {candidate.direction.value} lean and chose"
        conv_txt = f" (conviction {float(conviction):.2f})" if isinstance(conviction, (int, float)) else ""
        tier_rationale = f"AI brain ({brain}) {verb} {glm_dir.value}{conv_txt}." + (f" {reason}" if reason else "")
        upgrade_condition = ""

    supporting = [f"AI brain ({brain}) decided {glm_dir.value}" + (f": {reason}" if reason else "")]
    supporting.extend(list(candidate.supporting_evidence[:4]))

    return AIDecision(
        decision_id=str(uuid.uuid4()),
        symbol=symbol,
        horizon=horizon,
        status=SignalStatus.SIGNAL,
        direction=glm_dir,
        reference_price=candidate.reference_price,
        expiry_utc=expiry_for(horizon_minutes(horizon), candidate.generated_at_utc),
        regime=candidate.regime.value,
        heuristic_score=candidate.heuristic_score,
        calibrated_probability=None,
        conviction_tier=tier,
        position_size_fraction=size,
        tier_rationale=tier_rationale,
        upgrade_condition=upgrade_condition,
        supporting_evidence=supporting,
        contradictory_evidence=list(candidate.contradictory_evidence),
        invalidation_conditions=[invalidation] if invalidation else [],
        missing_information=[],
        llm_provider=provider_name,
        llm_model=brain,
        decision_authority=authority,
        decision_mode=mode,
        llm_requested_model=requested_model,
        llm_response_model=response_model,
        model_verified=model_verified,
        llm_latency_ms=latency_ms,
        scanner_version=candidate.scanner_version,
        data_quality_status=quality_status,
        created_at_utc=utc_now(),
    )


def _armed_recommendation(
    symbol: str,
    decision: AIDecision,
    *,
    horizon: str,
    dur: int,
    asset_class: AssetClass | str | None,
    candle_fetcher: Any,
    include_watch_plan: bool,
    tier_rationale: str | None = None,
    upgrade_condition: str | None = None,
    extra_warning: str = "",
) -> TradeRecommendation:
    """A size-0 ARMED directional lean: a real direction, no groundable sized setup.

    The honest actionable read when a direction exists (the brain's, or the
    scanner's) but there is no sized setup to place: dead chop, a primed-but-unbroken
    breakout, or -- in llm_first -- the brain chose a side that real levels cannot
    yet ground into a stop and a reachable target. Size is 0, stop and target are
    ``None`` (no level is invented for a trade that does not exist), and the trigger
    to watch lives in ``upgrade_condition`` and the warnings. Shared by the explicit
    ARMED branch and both llm_first degrade paths so all three read identically.
    """
    now = utc_now()
    ast = _resolve_asset_class(symbol, asset_class)
    expiry = expiry_for(dur, now)
    watch = _watch_plan(symbol, horizon, candle_fetcher) if include_watch_plan else ""
    warnings: list[str] = [
        "ARMED / NO EDGE: this is a directional lean, not a live setup. Position "
        "size is 0 -- do not stake a normal position on it. Wait for the trigger "
        "below (a level breaking, or price reaching a range edge) before treating "
        "it as a trade.",
    ]
    if extra_warning:
        warnings.append(extra_warning)
    if horizon in {"1m", "3m"}:
        warnings.append(
            f"{horizon} scalp: computed from the last CLOSED {horizon} bar polled "
            "over REST, not a live tick stream; the bar can be up to one interval old."
        )
    if watch:
        warnings.append(watch)
    # __ARMED_REC_RETURN__
    return TradeRecommendation(
        recommendation_id=f"rec-{uuid.uuid4().hex[:12]}",
        symbol=symbol,
        asset_class=ast,
        horizon=horizon,
        direction=decision.direction,
        valid_from_utc=now,
        valid_until_utc=expiry,
        reference_price=decision.reference_price,
        stop_loss=None,
        take_profit=None,
        risk_reward_ratio=0.0,
        risk_level="NO_EDGE",
        recommended_venue=_venue_for(ast),
        regime=decision.regime,
        heuristic_score=decision.heuristic_score or 0.0,
        confidence_pct=round((decision.heuristic_score or 0.0) * 100),
        conviction_tier=ConvictionTier.ARMED,
        position_size_fraction=0.0,
        tier_rationale=(tier_rationale if tier_rationale is not None else (decision.tier_rationale or "")),
        upgrade_condition=(
            upgrade_condition if upgrade_condition is not None else (decision.upgrade_condition or "")
        ),
        rationale=(
            f"{decision.regime} on the {horizon} horizon, heuristic score "
            f"{(decision.heuristic_score or 0.0):.2f}. ARMED directional lean "
            f"({decision.direction.value.lower()}) -- no live position, size 0."
        ),
        warnings=warnings,
        decision_authority=decision.decision_authority,
        decision_mode=decision.decision_mode,
        llm_provider=decision.llm_provider,
        llm_requested_model=decision.llm_requested_model,
        llm_response_model=decision.llm_response_model,
        model_verified=decision.model_verified,
        llm_latency_ms=decision.llm_latency_ms,
        llm_fallback_reason=decision.llm_fallback_reason,
        generated_at_utc=now,
    )


def _reconcile_review(
    validated: LLMSignalResponse,
    candidate_direction: SignalDirection,
    *,
    symbol: str,
) -> LLMSignalResponse:
    """Enforce the reviewer as conservative-only: it may confirm or decline, never flip.

    - A ``SIGNAL`` verdict that keeps the candidate direction passes through.
    - A ``SIGNAL`` verdict that omits a direction is read as concurrence with the
      candidate, so an unstated direction is not mistaken for a decline.
    - A ``SIGNAL`` verdict whose direction *reverses* the candidate is downgraded
      to ``NO_TRADE``. The scanner sets direction deterministically; the reviewer's
      role is a veto, not a reversal. Trading the opposite of a correct setup on a
      model's say-so is exactly the failure the documented contract rules out.
    - Any non-``SIGNAL`` verdict (``NO_TRADE`` / ``INSUFFICIENT_DATA``) is returned
      unchanged: the reviewer is always free to be *more* conservative.
    """
    if validated.status is not SignalStatus.SIGNAL:
        return validated
    if validated.direction is None:
        return validated.model_copy(update={"direction": candidate_direction})
    if validated.direction is candidate_direction:
        return validated
    log.info(
        "reviewer reversed the candidate direction; treating as a decline",
        extra={
            "symbol": symbol,
            "candidate": candidate_direction.value,
            "reviewer": validated.direction.value,
        },
    )
    return validated.model_copy(
        update={
            "status": SignalStatus.NO_TRADE,
            "direction": None,
            "contradictory_evidence": [
                f"Reviewer read the evidence as {validated.direction.value} while the "
                f"deterministic scan set up {candidate_direction.value}; a reversal is "
                "declined, not traded in the opposite direction.",
                *validated.contradictory_evidence,
            ],
        }
    )


def _no_trade_decision(symbol: str, horizon: str, scan: ScanResult) -> AIDecision:
    rejection = scan.rejections[0] if scan.rejections else None
    reason = rejection.reason if rejection else "no setup met the configured criteria"
    status = (
        SignalStatus.INSUFFICIENT_DATA
        if rejection is not None and _is_data_problem(rejection.reason_code)
        else SignalStatus.NO_TRADE
    )
    return AIDecision(
        decision_id=str(uuid.uuid4()),
        symbol=symbol,
        horizon=horizon,
        status=status,
        conviction_tier=ConvictionTier.STAND_ASIDE,
        missing_information=[reason],
        data_quality_status=_quality_status(scan, symbol),
        created_at_utc=utc_now(),
    )


def _is_data_problem(reason_code: str) -> bool:
    """Distinguish "no setup here" from "we could not see the market"."""
    return reason_code in {
        "FETCH_ERROR",
        "INSUFFICIENT_BARS",
        "STALE_QUOTE",
        "QUALITY_BLOCKED",
        "NO_QUOTE",
        "WARMUP_INCOMPLETE",
    }


def _quality_report(scan: ScanResult, symbol: str) -> DataQualityReport | None:
    """The execution-timeframe quality report the scanner actually produced.

    ``None`` when the scan never got as far as evaluating quality -- a provider
    failure, typically. That is a different statement from any of PASS /
    DEGRADED / FAIL, and the field is optional precisely so it can be made.
    """
    return scan.quality_reports.get(symbol)


def _quality_status(scan: ScanResult, symbol: str) -> QualityStatus | None:
    """The status the quality engine returned, not one inferred from a score."""
    report = _quality_report(scan, symbol)
    return report.status if report is not None else None


def _risk_levels_for(
    symbol: str,
    horizon: str,
    direction: SignalDirection,
    reference_price: Decimal,
    candle_fetcher: Any,
) -> Any:
    """Recompute execution-timeframe features to place the stop and target."""
    from quantedge.contracts import Timeframe
    from quantedge.providers.registry import get_registry
    from quantedge.services import indicators as ind
    from quantedge.services import structure as st
    from quantedge.services.horizons import horizon_timeframes

    tf = Timeframe(horizon_timeframes(horizon)["execution"])
    if candle_fetcher is not None:
        series = candle_fetcher(symbol, tf)
    else:
        series = get_registry().get_candles(symbol, tf, limit=300)

    closed = [c for c in series.candles if c.is_closed]
    if len(closed) < 30:
        return None
    features = ind.compute_features(closed, provider=series.provider)
    report = st.analyze_structure(closed, atr=features.atr_14)
    return derive_risk_levels(
        reference_price=reference_price,
        direction=direction,
        features=features,
        structure=report,
    )


def _watch_plan(symbol: str, horizon: str, candle_fetcher: Any = None) -> str:
    """An honest, conditional watch line built from confirmed structure only.

    Called when the engine declines a *directional* setup (not a data fault): the
    decline is real, but the trader still deserves to know what would change it.
    Uses only :func:`structure.analyze_structure` output -- confirmed swing levels
    and the structure engine's own deterministic breakout read -- so nothing here
    asserts a direction the evidence does not carry or quotes a probability.
    Returns ``""`` when no confirmed structure exists, so the space is never
    filled with a fabricated level.
    """
    from quantedge.contracts import Timeframe
    from quantedge.providers.registry import get_registry
    from quantedge.services import indicators as ind
    from quantedge.services import structure as st
    from quantedge.services.horizons import horizon_timeframes

    try:
        tf = Timeframe(horizon_timeframes(horizon)["execution"])
        if candle_fetcher is not None:
            series = candle_fetcher(symbol, tf)
        else:
            series = get_registry().get_candles(symbol, tf, limit=300)
        closed = [c for c in series.candles if c.is_closed]
        if len(closed) < 30:
            return ""
        features = ind.compute_features(closed, provider=series.provider)
        report = st.analyze_structure(closed, atr=features.atr_14)
    except Exception as exc:  # pragma: no cover - watch plan is best-effort
        log.info("watch plan unavailable for %s %s: %s", symbol, horizon, exc)
        return ""

    res = report.nearest_resistance
    sup = report.nearest_support
    if res is None and sup is None:
        return ""

    # The structure engine's own breakout read, when it has one, lets us name the
    # single level that matters rather than both. It is a deterministic structural
    # classification, not a forecast, so it may be stated as the trigger.
    if report.breakout_candidate and report.breakout_direction is not None:
        up = report.breakout_direction.value == "UP"
        level = res if up else sup
        if level is not None:
            side = "above" if up else "below"
            return (
                f"Not armed yet. Structure is coiled {report.breakout_direction.value}: a "
                f"confirmed {horizon} close {side} {level} would be the trigger that gives a "
                f"{report.breakout_direction.value} setup something to work with -- it is not "
                "a position now, so ask again if that level closes."
            )

    parts: list[str] = []
    if res is not None:
        parts.append(f"a confirmed close above {res} opens the upside")
    if sup is not None:
        parts.append(f"a confirmed close below {sup} opens the downside")
    return (
        "Not armed yet. Watch the edges: " + "; ".join(parts) + ". Neither has "
        "broken, so there is no position here -- ask again if one closes."
    )


def _resolve_asset_class(symbol: str, given: AssetClass | str | None) -> AssetClass:
    if isinstance(given, AssetClass):
        return given
    if isinstance(given, str) and given:
        return AssetClass(given.lower())
    return asset_class_for(symbol)


def _risk_level(score: float | None, rr_acceptable: bool) -> str:
    """A qualitative band, deliberately not a probability.

    Naming this a win rate or confidence percentage would imply calibration that
    has not been performed, which Rule 3 prohibits.
    """
    if not rr_acceptable:
        return "UNFAVOURABLE_GEOMETRY"
    if score is None:
        return "UNRATED"
    if score >= 0.80:
        return "HIGH_CONVICTION_SETUP"
    if score >= 0.65:
        return "MODERATE_CONVICTION_SETUP"
    return "LOW_CONVICTION_SETUP"


def _venue_for(asset_class: AssetClass) -> str:
    match asset_class:
        case AssetClass.FOREX:
            return "OANDA"
        case AssetClass.STOCK:
            return "Interactive Brokers"
        case AssetClass.COMMODITY:
            return "OANDA (CFD)"
        case _:
            return "Binance (spot)"


def _default_llm_provider() -> BaseLLMProvider | None:
    """The configured reviewer, or ``None`` when no credential is present.

    Returning ``None`` keeps the deterministic path working without a model:
    the scanner's candidate is reported on its own rather than being blocked.
    """
    from quantedge.providers.llm import default_llm_provider

    return default_llm_provider()


def _as_llm_response(decision: AIDecision) -> LLMSignalResponse:
    """Shape a decision into the response contract the signals table stores.

    Used when the deterministic candidate stands without a model review, so the
    persisted row is the same shape either way. ``llm_provider`` on the decision
    stays ``None`` in that case, which is how a reader tells the two apart.
    """
    return LLMSignalResponse(
        status=decision.status,
        asset=decision.symbol,
        direction=decision.direction,
        generated_at_utc=decision.created_at_utc,
        expiry_utc=decision.expiry_utc,
        horizon=decision.horizon,
        reference_price=decision.reference_price,
        regime=decision.regime,
        calibrated_probability=decision.calibrated_probability,
        heuristic_score=decision.heuristic_score,
        supporting_evidence=list(decision.supporting_evidence),
        contradictory_evidence=list(decision.contradictory_evidence),
        invalidation_conditions=list(decision.invalidation_conditions),
        missing_information=list(decision.missing_information),
    )


def _persist(
    repo: Any,
    decision: AIDecision,
    *,
    context: Any = None,
    llm_response: LLMSignalResponse | None = None,
) -> None:
    """Record the decision, logging rather than swallowing a failure.

    A directional signal goes to the ``signals`` table. A NO_TRADE or
    INSUFFICIENT_DATA outcome goes to the audit log instead: the signals schema
    requires a direction, and inventing one to satisfy a column would put a
    trade in the history that was never issued.

    The original code used a bare ``except Exception: pass`` here, so a broken
    database looked identical to a working one from the caller's side.
    """
    decision_id = decision.decision_id or str(uuid.uuid4())
    try:
        if decision.status is SignalStatus.SIGNAL and decision.direction is not None:
            repo.save_signal(
                decision_id,
                f"scan-{decision_id[:12]}",
                llm_response if llm_response is not None else _as_llm_response(decision),
                context,
            )
        else:
            repo.log_event(
                "signal_declined",
                symbol=decision.symbol,
                message=decision.status.value,
                details={
                    "decision_id": decision_id,
                    "horizon": decision.horizon,
                    "reasons": list(decision.missing_information),
                    "data_quality_status": (
                        decision.data_quality_status.value
                        if decision.data_quality_status is not None
                        else None
                    ),
                },
            )
    except QuantEdgeError as exc:
        log.warning("decision not persisted", extra={"symbol": decision.symbol, "code": exc.code})
    except Exception as exc:  # noqa: BLE001 - persistence must not break the answer
        log.warning(
            "decision not persisted",
            extra={"symbol": decision.symbol, "error": type(exc).__name__},
        )
