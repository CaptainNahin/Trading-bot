"""SignalContext assembly service.

Prepares verified market data and deterministic analysis into a strict
``SignalContext`` contract before handing off to LLM review.
"""

from __future__ import annotations

from quantedge.contracts import (
    DataQualityReport,
    EventRiskReport,
    MultiTimeframeSnapshot,
    RegimeReport,
    ScanCandidate,
    SessionState,
    SignalContext,
    utc_now,
)


def build_signal_context(
    candidate: ScanCandidate,
    quality: DataQualityReport,
    *,
    multi_timeframe: MultiTimeframeSnapshot | None = None,
    regime: RegimeReport | None = None,
    session: SessionState | None = None,
    event_risk: EventRiskReport | None = None,
    extra_missing_info: list[str] | None = None,
) -> SignalContext:
    """Build a SignalContext object from verified components.

    Explicitly enumerates missing information to prevent LLM hallucination.
    """
    missing_info: list[str] = list(extra_missing_info or [])

    if multi_timeframe is None:
        missing_info.append("Multi-timeframe snapshot not available")
    if regime is None:
        missing_info.append("Regime report not available")
    if session is None:
        missing_info.append("Liquidity session state not available")
    if event_risk is None:
        missing_info.append("Economic event risk report not available")

    # Pull memory bank rules derived from past losses on this asset & horizon
    learned_rules: list[str] = []
    try:
        from quantedge.services.memory import recurring_loss_rules

        learned_rules = recurring_loss_rules(candidate.symbol, horizon=candidate.horizon)
    except Exception:
        pass

    # Force calibration model flag to False per system rules
    return SignalContext(
        generated_at_utc=utc_now(),
        symbol=candidate.symbol,
        asset_class=candidate.asset_class,
        horizon=candidate.horizon,
        data_sources={"provider": candidate.provider},
        quality=quality,
        multi_timeframe=multi_timeframe,
        regime=regime,
        session=session,
        event_risk=event_risk,
        candidate_direction=candidate.direction,
        heuristic_score=candidate.heuristic_score,
        supporting_evidence=candidate.supporting_evidence,
        contradictory_evidence=candidate.contradictory_evidence,
        learned_rules=learned_rules,
        missing_information=missing_info,
        calibration_model_available=False,
    )
