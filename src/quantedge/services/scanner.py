"""12-step deterministic scanning pipeline.

The scanner evaluates market candidates using pure, reproducible formulas.
A symbol with a failing data quality report is immediately rejected without
proceeding to indicator calculation or regime classification.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from quantedge.config import get_scanner_config
from quantedge.contracts import (
    AssetClass,
    Candle,
    DataQualityReport,
    EventRiskStatus,
    ScanCandidate,
    ScanRejection,
    ScanResult,
    SignalDirection,
)
from quantedge.errors import QuantEdgeError
from quantedge.services import (
    indicators as ind,
)
from quantedge.services import (
    mtf,
    scoring,
)
from quantedge.services import (
    quality as qual,
)
from quantedge.services import (
    regime as reg,
)
from quantedge.services import (
    structure as st,
)
from quantedge.services.horizons import normalize_horizon
from quantedge.symbols import asset_class_for

if TYPE_CHECKING:
    from quantedge.providers.registry import ProviderRegistry

SCANNER_VERSION = "scanner-1.0.0"


def _indicator_history(
    closed_candles: Sequence[Candle],
    *,
    provider: str | None = None,
    symbol: str | None = None,
) -> tuple[list[float | None], list[float | None]]:
    """Per-bar ``bb_width`` and ``atr_14`` over the closed series.

    Computed by recomputing features over expanding windows, because a
    percentile or rolling mean is only meaningful against a distribution.
    O(n^2) in the bar count, but the scan runs a handful of symbols at most
    and each window is a pure recomputation -- deterministic, no look-ahead.
    """
    bb: list[float | None] = []
    atr: list[float | None] = []
    for i in range(1, len(closed_candles) + 1):
        window = closed_candles[:i]
        if len(window) < 15:
            bb.append(None)
            atr.append(None)
            continue
        feat = ind.compute_features(window, provider=provider, symbol=symbol)
        bb.append(feat.bb_width)
        atr.append(feat.atr_14)
    return bb, atr


def _event_risk_for(symbol: str) -> EventRiskStatus:
    """Event risk for a symbol, or ``UNKNOWN`` when no calendar is available.

    Rule 5 in reverse: absence of data is reported as absence, never as
    "benign". ``UNKNOWN`` is deliberately not downgraded to ``LOW`` -- the
    contract for :class:`ScanCandidate` says so explicitly, because a trader
    reading LOW would believe a calendar had been checked and had come back
    clear.
    """
    from quantedge.providers.registry import get_registry

    try:
        registry = get_registry()
    except QuantEdgeError:
        return EventRiskStatus.UNKNOWN

    if not registry.calendar_chain():
        return EventRiskStatus.UNKNOWN

    # Every configured calendar provider is credential-gated; when none can be
    # tried the answer is unknown rather than clear.
    for name in registry.calendar_chain():
        try:
            provider = registry.get(name)
        except QuantEdgeError:
            continue
        if provider.enabled and provider.circuit_state != "open":
            # A provider is reachable, but wiring the live lookup into the
            # synchronous scan path is deferred; claiming LOW here would be
            # the same fabrication with extra steps.
            return EventRiskStatus.UNKNOWN
    return EventRiskStatus.UNKNOWN


def _session_liquidity(asset_class: AssetClass) -> str:
    """Coarse session liquidity by asset class and UTC hour.

    Crypto trades continuously, so its liquidity is reported ``continuous``
    rather than ``high`` -- "high" would imply a measurement of depth that
    nothing here performs. For session-bound markets the value describes which
    session is open, which is a fact about the clock, not about the book.
    """
    if asset_class == AssetClass.CRYPTO:
        return "continuous"

    hour = datetime.now(UTC).hour
    weekday = datetime.now(UTC).weekday()
    if weekday >= 5:
        return "closed_weekend"
    if 12 <= hour < 17:  # London/New York overlap
        return "london_newyork_overlap"
    if 7 <= hour < 16:
        return "london"
    if 13 <= hour < 22:
        return "newyork"
    if hour >= 23 or hour < 9:
        return "asia"
    return "off_session"


def run_scan(
    symbols: Sequence[str],
    *,
    horizon: str = "swing",
    registry: ProviderRegistry | None = None,
    provider: str = "binance",
    candle_fetcher: Any = None,
) -> ScanResult:
    """Execute the 12-step scan pipeline across ``symbols``.

    Parameters
    ----------
    symbols:
        List of symbol strings (e.g. ``["BTCUSDT", "ETHUSDT"]``).
    horizon:
        Trading horizon (``scalp``, ``intraday``, ``swing``, ``position``).
    registry:
        Optional ProviderRegistry instance.
    provider:
        Default provider to query.
    candle_fetcher:
        Optional callable ``(symbol, timeframe, limit) -> list[Candle]`` for
        dependency injection or offline testing.
    """
    candidates: list[ScanCandidate] = []
    rejections: list[ScanRejection] = []
    warnings: list[str] = []
    # Keyed by symbol, kept whatever the outcome: a caller needs to distinguish
    # "the data was unusable" from "the setup was not there".
    quality_reports: dict[str, DataQualityReport] = {}

    horizon = normalize_horizon(horizon)
    timeframe_map = mtf.get_horizon_timeframes(horizon)
    exec_tf = timeframe_map["execution"]
    conf_tf = timeframe_map["confirmation"]
    reg_tf = timeframe_map["regime"]
    lookback = int(timeframe_map.get("lookback", 300))

    cfg = get_scanner_config()
    # Nested under output:, not top-level. Reading the wrong key silently fell
    # back to a default that no config file could override.
    output_cfg = cfg.get("output", {})
    min_heuristic_score = float(output_cfg.get("min_heuristic_score_to_return", 0.55))
    gates = cfg.get("gates", {})
    weights = cfg.get("weights", {})
    composite_weights = weights.get("composite", {})
    warmup_cfg = cfg.get("warmup", {})
    min_bars = int(warmup_cfg.get("minimum_bars", 210))
    min_quality = float(gates.get("min_data_quality_score", 0.60))
    min_agreement = float(gates.get("min_evidence_agreement_score", 0.50))
    # Agreement is renormalised over the views that carry a direction, so it
    # cannot on its own distinguish three unanimous timeframes from one lone
    # execution view. The quorum is the other half of that gate: it requires a
    # minimum share of the timeframe stack to have actually voted before the
    # unanimity means anything.
    min_participation = float(gates.get("min_timeframe_participation", 0.50))
    allowed_quality = {
        str(s).upper() for s in gates.get("require_quality_status", ["PASS", "DEGRADED"])
    }
    blocking_event_risk = {str(s).upper() for s in gates.get("block_on_event_risk", ["HIGH"])}
    block_unknown_event_risk = bool(gates.get("block_on_unknown_event_risk", False))

    # Fall back to the shared registry when no data source was injected. Without
    # this every caller has to remember to pass one, and the one that forgot got
    # a NO_DATA_SOURCE rejection that reads exactly like "no setup here" -- a
    # missing wire reported as a trading decision. Tests still inject their own.
    if candle_fetcher is None and registry is None:
        from quantedge.providers.registry import get_registry

        registry = get_registry()

    for symbol in symbols:
        # Step 1 & 2: Fetch candles & run Quality check
        try:
            if candle_fetcher is not None:
                exec_series = candle_fetcher(symbol, exec_tf, lookback)
                conf_series = candle_fetcher(symbol, conf_tf, lookback)
                reg_series = candle_fetcher(symbol, reg_tf, lookback)
            elif registry is not None:
                exec_series = registry.get_candles(
                    symbol, exec_tf, limit=lookback, provider_name=provider
                )
                conf_series = registry.get_candles(
                    symbol, conf_tf, limit=lookback, provider_name=provider
                )
                reg_series = registry.get_candles(
                    symbol, reg_tf, limit=lookback, provider_name=provider
                )
            else:
                rejections.append(
                    ScanRejection(
                        symbol=symbol,
                        reason_code="NO_DATA_SOURCE",
                        reason="Neither registry nor candle_fetcher was provided to scanner",
                        stage="DATA_FETCH",
                    )
                )
                continue
        except QuantEdgeError as exc:
            rejections.append(
                ScanRejection(
                    symbol=symbol,
                    reason_code=exc.code or "FETCH_ERROR",
                    reason=f"Failed to fetch market data: {exc.message}",
                    stage="DATA_FETCH",
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not end the scan
            rejections.append(
                ScanRejection(
                    symbol=symbol,
                    reason_code="FETCH_ERROR",
                    reason=f"Failed to fetch market data: {type(exc).__name__}: {exc}",
                    stage="DATA_FETCH",
                )
            )
            continue

        # The series carries provider, timeframe and forming-bar metadata that
        # the quality engine checks; unwrapping to a bare list here would throw
        # away exactly what Rule 9 and the staleness checks are built on.
        exec_candles = list(getattr(exec_series, "candles", exec_series))
        conf_candles = list(getattr(conf_series, "candles", conf_series))
        reg_candles = list(getattr(reg_series, "candles", reg_series))

        # Data Quality Gate
        exec_quality = qual.evaluate_quality(
            exec_candles, expected_timeframe=exec_tf, provider=provider
        )
        quality_reports[symbol] = exec_quality
        if exec_quality.is_blocking:
            rejections.append(
                ScanRejection(
                    symbol=symbol,
                    reason_code="DATA_QUALITY_FAIL",
                    reason=(
                        "Execution timeframe quality check failed: "
                        f"{exec_quality.blocking_reasons}"
                    ),
                    stage="QUALITY_GATE",
                )
            )
            continue
        # ``require_quality_status`` is a documented gate in scanner.yaml, and a
        # second profile in that file sets it to [PASS] alone. FAIL is already
        # refused above; enforcing the list here is what makes "DEGRADED never
        # escalates" mean something rather than being an inert setting.
        if exec_quality.status.value.upper() not in allowed_quality:
            rejections.append(
                ScanRejection(
                    symbol=symbol,
                    reason_code="QUALITY_STATUS_NOT_ALLOWED",
                    reason=(
                        f"Execution timeframe quality status {exec_quality.status.value} "
                        f"is not in the configured allow-list {sorted(allowed_quality)}"
                    ),
                    stage="QUALITY_GATE",
                )
            )
            continue
        # The configured gate on quality is a score floor (0.60 by default),
        # applied independently of the blocking reasons the engine raised.
        if exec_quality.quality_score < min_quality:
            rejections.append(
                ScanRejection(
                    symbol=symbol,
                    reason_code="LOW_DATA_QUALITY",
                    reason=(
                        f"Execution timeframe quality score {exec_quality.quality_score} "
                        f"is below minimum {min_quality}"
                    ),
                    stage="QUALITY_GATE",
                )
            )
            continue

        # Step 3: Rule 9 closed candles filter
        closed_exec = [c for c in exec_candles if c.is_closed]
        closed_conf = [c for c in conf_candles if c.is_closed]
        closed_reg = [c for c in reg_candles if c.is_closed]

        # config/scanner.yaml sets warmup.minimum_bars 210 because EMA-200
        # needs 200 closed bars before it means anything. The old literal 50
        # let a scan proceed with EMA-200 undefined and score it anyway.
        if len(closed_exec) < min_bars:
            rejections.append(
                ScanRejection(
                    symbol=symbol,
                    reason_code="INSUFFICIENT_CLOSED_CANDLES",
                    reason=(
                        f"Only {len(closed_exec)} closed candles on execution timeframe; "
                        f"warmup requires {min_bars}"
                    ),
                    stage="WARMUP",
                )
            )
            continue

        # Step 4: Indicators & Feature Snapshot
        feat_exec = ind.compute_features(closed_exec, provider=provider)

        # Step 5: Structure Report
        atr_val = feat_exec.atr_14
        struct_exec = st.analyze_structure(closed_exec, atr=atr_val)

        # Step 6: Regime Report
        # The classifier needs a distribution, not a single reading: "is
        # Bollinger width unusually narrow" and "is ATR above its own mean" are
        # only answerable against history. Recomputing features over expanding
        # windows of the same closed bars keeps it deterministic and free of
        # look-ahead -- window i ends at bar i, so no future bar informs it.
        bb_history, atr_history = _indicator_history(closed_exec, provider=provider, symbol=symbol)
        regime_report = reg.classify_regime_from_features(
            structure=struct_exec,
            features=feat_exec,
            bb_width_history=bb_history,
            atr_history=atr_history,
        )

        # Step 7: Multi-timeframe views & Alignment
        conf_quality = qual.evaluate_quality(
            closed_conf, expected_timeframe=conf_tf, provider=provider
        )
        reg_quality = qual.evaluate_quality(
            closed_reg, expected_timeframe=reg_tf, provider=provider
        )

        conf_struct = (
            st.analyze_structure(
                closed_conf, atr=ind.compute_features(closed_conf, provider=provider).atr_14
            )
            if len(closed_conf) >= 30
            else None
        )
        reg_struct = (
            st.analyze_structure(
                closed_reg, atr=ind.compute_features(closed_reg, provider=provider).atr_14
            )
            if len(closed_reg) >= 30
            else None
        )

        mtf_snapshot = mtf.build_mtf_snapshot(
            symbol=symbol,
            horizon=horizon,
            exec_candles=closed_exec,
            conf_candles=closed_conf,
            reg_candles=closed_reg,
            exec_quality=exec_quality,
            conf_quality=conf_quality,
            reg_quality=reg_quality,
            exec_struct=struct_exec,
            conf_struct=conf_struct,
            reg_struct=reg_struct,
        )

        # Step 8: Directional Bias Determination
        direction = mtf_snapshot.aligned_direction
        if direction is None:
            if struct_exec.structure == "UPTREND":
                direction = SignalDirection.UP
            elif struct_exec.structure == "DOWNTREND":
                direction = SignalDirection.DOWN

        if direction is None:
            rejections.append(
                ScanRejection(
                    symbol=symbol,
                    reason_code="NO_CLEAR_DIRECTION",
                    reason="No clear directional bias established across MTF and structure",
                    stage="DIRECTION_FILTER",
                )
            )
            continue

        # Step 9: Scoring Components
        # Weighted rule agreement from config/scanner.yaml, not step functions.
        # The previous binary scores ("0.8 if trending else 0.4") could not
        # distinguish a marginal trend from an overwhelming one, so a coin-flip
        # setup and a textbook one landed on the same number.
        trend = scoring.trend_score(feat_exec, struct_exec, direction, weights.get("trend", {}))
        momentum = scoring.momentum_score(feat_exec, direction, weights.get("momentum", {}))
        volatility = scoring.volatility_score(feat_exec, weights.get("volatility", {}))

        trend_score = trend.score
        momentum_score = momentum.score
        volatility_score = volatility.score
        quality_score = exec_quality.quality_score
        agreement_score = mtf_snapshot.alignment_score

        if agreement_score < min_agreement:
            # Two different failures land here and the message has to say which.
            # A conflict means the views pointed opposite ways; no conflict means
            # they simply did not all speak, and reporting that as "timeframes
            # disagree" describes a fight that never happened -- the reader then
            # looks for an opposing trend that is not there.
            if mtf_snapshot.conflicts:
                cause = "; ".join(mtf_snapshot.conflicts)
            elif mtf_snapshot.abstaining_roles:
                cause = (
                    f"no conflict, but only part of the stack carries a direction; "
                    f"abstaining: {', '.join(mtf_snapshot.abstaining_roles)}"
                )
            else:
                cause = "no timeframe carries a direction"
            rejections.append(
                ScanRejection(
                    symbol=symbol,
                    reason_code="WEAK_EVIDENCE_AGREEMENT",
                    reason=(
                        f"Multi-timeframe agreement {round(agreement_score, 4)} is below "
                        f"minimum {min_agreement}: {cause}"
                    ),
                    stage="AGREEMENT_GATE",
                )
            )
            continue

        if mtf_snapshot.participation < min_participation:
            rejections.append(
                ScanRejection(
                    symbol=symbol,
                    reason_code="INSUFFICIENT_TIMEFRAME_PARTICIPATION",
                    reason=(
                        f"Only {round(mtf_snapshot.participation, 4)} of the timeframe "
                        f"stack carries a direction (minimum {min_participation}); "
                        f"abstaining: {', '.join(mtf_snapshot.abstaining_roles) or 'none'}"
                    ),
                    stage="AGREEMENT_GATE",
                )
            )
            continue

        heuristic_score = scoring.composite_score(
            trend=trend_score,
            momentum=momentum_score,
            volatility=volatility_score,
            data_quality=quality_score,
            evidence_agreement=agreement_score,
            weights=composite_weights,
        )

        skipped_rules = trend.rules_skipped + momentum.rules_skipped + volatility.rules_skipped

        # Step 10: Score Threshold Check
        if heuristic_score < min_heuristic_score:
            rejections.append(
                ScanRejection(
                    symbol=symbol,
                    reason_code="LOW_HEURISTIC_SCORE",
                    reason=(
                        f"Heuristic score {heuristic_score} is below minimum "
                        f"threshold {min_heuristic_score}"
                    ),
                    stage="SCORE_FILTER",
                )
            )
            continue

        # Step 11: Event risk.
        # No calendar credential is configured in this deployment, so the
        # honest value is UNKNOWN. The previous hardcoded LOW asserted "no
        # economic event is near" without consulting anything -- the exact
        # claim a trader would rely on before sizing a position.
        event_risk = _event_risk_for(symbol)
        if event_risk.value.upper() in blocking_event_risk or (
            block_unknown_event_risk and event_risk == EventRiskStatus.UNKNOWN
        ):
            rejections.append(
                ScanRejection(
                    symbol=symbol,
                    reason_code="EVENT_RISK_BLOCK",
                    reason=f"Event risk is {event_risk.value} and configuration blocks it",
                    stage="EVENT_RISK_GATE",
                )
            )
            continue

        # Step 12: Construct Candidate
        contradictions = list(regime_report.contradictions) + list(mtf_snapshot.conflicts)
        if skipped_rules:
            contradictions.append(
                "scoring rules not evaluated (inputs unavailable): "
                f"{', '.join(sorted(set(skipped_rules)))}"
            )

        candidate = ScanCandidate(
            symbol=symbol,
            asset_class=asset_class_for(symbol),
            horizon=horizon,
            direction=direction,
            provider=exec_series.provider or provider,
            heuristic_score=heuristic_score,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            data_quality_score=quality_score,
            evidence_agreement_score=agreement_score,
            regime=regime_report.regime,
            reference_price=closed_exec[-1].close,
            quote_freshness_ms=exec_quality.freshness_ms,
            supporting_evidence=regime_report.supporting_evidence + struct_exec.notes,
            contradictory_evidence=contradictions,
            event_risk=event_risk,
            session_liquidity=_session_liquidity(asset_class_for(symbol)),
            scanner_version=SCANNER_VERSION,
        )
        candidates.append(candidate)

    return ScanResult(
        horizon=horizon,
        requested_symbols=list(symbols),
        scanned=len(symbols),
        candidates=candidates,
        rejections=rejections,
        quality_reports=quality_reports,
        scanner_version=SCANNER_VERSION,
        warnings=warnings,
    )
