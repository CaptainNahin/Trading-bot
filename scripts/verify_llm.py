"""LLM Provider and Response Validation verification script."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge.contracts import (
    AssetClass,
    DataQualityReport,
    HealthStatus,
    LLMSignalResponse,
    MarketRegime,
    QualityStatus,
    ScanCandidate,
    SignalDirection,
    SignalStatus,
    utc_now,
)
from quantedge.providers.llm.agentrouter import AgentRouterLLMProvider
from quantedge.providers.llm.anthropic import AnthropicLLMProvider
from quantedge.services.llm_review import validate_llm_response
from quantedge.services.signal_context import build_signal_context

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'[PASS]' if condition else '[FAIL]'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


def main() -> int:
    print("=" * 70)
    print("LLM VERIFICATION -- provider abstraction & output validation checks")
    print("=" * 70)

    # 1. Health checks. Three outcomes are legitimate and which one applies
    #    depends on the environment, not on the code: DISABLED when no key is
    #    configured, OK when one works, ERROR when one is present but the host
    #    rejects it. What must hold in every case is that the status is not
    #    overstated and that a failure names its cause -- a provider that
    #    reported OK on a 401 would let the pipeline believe review ran.
    allowed = (HealthStatus.OK, HealthStatus.DISABLED, HealthStatus.ERROR)
    for name, provider in (
        ("AgentRouter", AgentRouterLLMProvider()),
        ("Anthropic", AnthropicLLMProvider()),
    ):
        h = provider.health()
        check(f"{name} health is OK, DISABLED or ERROR", h.status in allowed, h.status.value)
        if h.status == HealthStatus.ERROR:
            check(f"  {name} error states why", bool(h.message), h.message or "(no message)")
        if h.status == HealthStatus.DISABLED:
            check(
                f"  {name} disabled means no credential, and says which",
                not h.credentials_present,
                f"missing_env={h.missing_env}",
            )

    # 2. Context assembly
    cand = ScanCandidate(
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        horizon="swing",
        direction=SignalDirection.UP,
        provider="binance",
        heuristic_score=0.75,
        trend_score=0.8,
        momentum_score=0.7,
        volatility_score=0.7,
        data_quality_score=1.0,
        evidence_agreement_score=0.8,
        regime=MarketRegime.STRONG_UPTREND,
        reference_price=Decimal("50000"),
        quote_freshness_ms=100,
        event_risk="LOW",
        session_liquidity="high",
        scanner_version="scanner-1.0.0",
    )
    qual = DataQualityReport(
        status=QualityStatus.PASS,
        quality_score=1.0,
        freshness_ms=100,
        provider="binance",
        symbol="BTCUSDT",
    )
    ctx = build_signal_context(cand, qual)
    check(
        "SignalContext forces calibration_model_available=False",
        ctx.calibration_model_available is False,
    )

    # 3. Validation forcing calibrated_probability to None
    raw_llm = LLMSignalResponse(
        status=SignalStatus.SIGNAL,
        asset="BTCUSDT",
        direction=SignalDirection.UP,
        generated_at_utc=utc_now(),
        calibrated_probability=0.85,
    )
    val = validate_llm_response(raw_llm, ctx)
    check("Validation forces calibrated_probability to None", val.calibrated_probability is None)

    # 4. Refuse upgrading QualityStatus FAIL
    fail_qual = DataQualityReport(
        status=QualityStatus.FAIL,
        quality_score=0.0,
        freshness_ms=99999,
        provider="binance",
        symbol="BTCUSDT",
        blocking_reasons=["stale data"],
    )
    fail_ctx = build_signal_context(cand, fail_qual)
    val_fail = validate_llm_response(raw_llm, fail_ctx)
    check(
        "Validation forces NO_TRADE on failing quality data",
        val_fail.status == SignalStatus.NO_TRADE,
    )

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
