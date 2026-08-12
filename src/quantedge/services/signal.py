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
    DataQualityReport,
    LLMSignalResponse,
    QualityStatus,
    SignalDirection,
    SignalStatus,
    TradeRecommendation,
    utc_now,
)
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
from quantedge.services.risk import MIN_ACCEPTABLE_RR, derive_risk_levels
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

    def __init__(self, status: SignalStatus, reason: str, *, detail: str = "") -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.detail = detail


def generate_signal_decision(
    symbol: str,
    *,
    horizon: str = "15m",
    provider_name: str | None = None,
    llm_provider: BaseLLMProvider | None = None,
    candle_fetcher: Any = None,
) -> AIDecision:
    """Scan, review and persist a decision for one symbol.

    The returned :class:`AIDecision` reports the real outcome, including
    ``NO_TRADE`` with the scanner's own rejection reason. The data-quality status
    on the record is the status the quality engine actually returned.
    """
    repo = get_repository()
    scan = run_scan(
        [symbol],
        horizon=horizon,
        # ``run_scan`` pins a single vendor; None here means "no preference", so
        # it falls through to the routed default rather than being passed on as
        # a provider named None.
        provider=provider_name or "binance",
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
    validated = None
    if provider is not None:
        try:
            validated = validate_llm_response(provider.evaluate_signal_context(context), context)
        except QuantEdgeError as exc:
            # A failed review must not upgrade into a signal. The deterministic
            # candidate stands on its own and is reported as such.
            log.warning(
                "llm review unavailable; returning the deterministic candidate",
                extra={"symbol": symbol, "code": exc.code},
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
) -> TradeRecommendation:
    """Produce a memory-augmented recommendation, or raise :class:`NoTradeReason`.

    ``time_limit`` is the expiry the user picked ("10 min", "20m", "1h"); it is
    resolved to a configured horizon so the analysis timeframes and the stated
    expiry agree with each other.
    """
    from quantedge.services.memory import get_relevant_memories, recurring_loss_rules

    horizon = resolve_time_limit(time_limit)
    decision = generate_signal_decision(
        symbol,
        horizon=horizon,
        provider_name=provider_name,
        candle_fetcher=candle_fetcher,
    )

    if decision.status is not SignalStatus.SIGNAL or decision.direction is None:
        # Which list holds the reason depends on why the decision came back.
        # INSUFFICIENT_DATA means something was missing, so missing_information
        # names the cause. NO_TRADE means the evidence was read and found
        # wanting, and the cause is the contradiction that decided it -- taking
        # missing_information[0] in that case led with an unrelated data gap
        # ("Liquidity session state not available") while the reasons that
        # actually declined the trade were pushed into the detail line.
        if decision.status is SignalStatus.INSUFFICIENT_DATA:
            reasons = list(decision.missing_information)
        else:
            reasons = list(decision.contradictory_evidence) or list(
                decision.missing_information
            )
        raise NoTradeReason(
            decision.status,
            (reasons or ["no setup met the configured criteria"])[0],
            detail="; ".join(reasons[1:4]),
        )
    if decision.reference_price is None:
        raise NoTradeReason(
            SignalStatus.INSUFFICIENT_DATA,
            "no reference price was available from any provider",
        )

    levels = _risk_levels_for(
        symbol, horizon, decision.direction, decision.reference_price, candle_fetcher
    )
    if levels is None:
        raise NoTradeReason(
            SignalStatus.INSUFFICIENT_DATA,
            "stop and target could not be derived: ATR is unavailable for this series",
        )

    # Unfavourable geometry is a decline, not a caveat. A setup whose target is
    # nearer than its stop needs to be right more often than not just to break
    # even, so emitting it with a warning attached invited exactly the trade the
    # ratio says to skip. The levels are real -- ATR-derived and structural --
    # which is why the fix is to refuse the trade rather than move the target.
    if not levels.acceptable:
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
    expiry = expiry_for(horizon_minutes(horizon), now)

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
    warnings.extend(memory_rules)

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
        rationale=rationale,
        warnings=warnings,
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
    all_syms = [s for s in supported_symbols("crypto") if s in supported]
    if not all_syms:
        all_syms = supported_symbols()[:3]

    if time_limit_minutes is not None:
        horizons = [resolve_time_limit(f"{time_limit_minutes}m")]
    else:
        # Default representative horizons to find the best trade without hitting rate limits
        horizons = ["15m", "1h"]

    # Every candidate, not just the strongest. The scanner scores a setup; the
    # recommendation stage then applies gates the scanner never saw -- reward:risk
    # above all. Keeping only the top scorer meant one candidate failing on
    # geometry was reported as "no setup on any symbol", which is a different and
    # much worse claim than the truth, and it hid setups that were ready to go.
    scored: list[tuple[float, str, str, str]] = []
    for hz in horizons:
        try:
            scan_res = run_scan(
                all_syms,
                horizon=hz,
                provider=provider_name or "binance",
                candle_fetcher=candle_fetcher,
            )
            scored.extend(
                (c.heuristic_score, c.symbol, hz, c.direction.value)
                for c in scan_res.candidates
            )
        except Exception as exc:
            log.warning("Scan failed for horizon %s", hz, exc_info=exc)

    if not scored:
        raise NoTradeReason(
            SignalStatus.NO_TRADE,
            "No trade setups found across any symbol or timeframe right now.",
        )

    # Best first, then walk down. The first decline is kept so that if every
    # candidate is refused the caller learns why the strongest one was refused,
    # rather than a generic "nothing found".
    first_decline: NoTradeReason | None = None
    ranked = sorted(scored, key=lambda row: row[0], reverse=True)
    for score, symbol, hz, _direction in ranked:
        try:
            rec = generate_trade_recommendation(
                symbol=symbol,
                time_limit=hz,
                provider_name=provider_name,
                candle_fetcher=candle_fetcher,
            )
        except NoTradeReason as exc:
            if first_decline is None:
                first_decline = exc
            continue

        if alternatives_out is not None:
            alternatives_out.extend(
                {"symbol": s, "horizon": h, "direction": d, "heuristic_score": sc}
                for sc, s, h, d in ranked
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
