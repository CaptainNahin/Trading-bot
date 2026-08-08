"""Analysis contracts: quality, features, regime, events, scanning, signals."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantedge.contracts.enums import (
    AssetClass,
    EventImpact,
    EventRiskStatus,
    HealthStatus,
    MarketRegime,
    QualityStatus,
    SettlementOutcome,
    SignalDirection,
    SignalStatus,
    Timeframe,
)
from quantedge.contracts.market import utc_now

__all__ = [
    "AIDecision",
    "DataQualityReport",
    "EconomicEvent",
    "EventRiskReport",
    "FeatureSnapshot",
    "LLMSignalResponse",
    "MultiTimeframeSnapshot",
    "NewsItem",
    "PerformanceSummary",
    "ProviderHealth",
    "RegimeReport",
    "ScanCandidate",
    "ScanRejection",
    "ScanResult",
    "SessionState",
    "SignalContext",
    "StructureReport",
    "TimeframeView",
]

# The disclaimer that must accompany every signal-bearing response.
STANDARD_WARNING = "Probabilistic research assessment, not a guaranteed outcome."


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# Health                                                                       #
# --------------------------------------------------------------------------- #


class ProviderHealth(_Model):
    """Health of a single provider. Contains no credential material."""

    provider: str
    kind: Literal["market_data", "economic_calendar", "news", "llm", "persistence"]
    status: HealthStatus
    enabled: bool
    credentials_present: bool
    latency_ms: float | None = None
    checked_at_utc: datetime = Field(default_factory=utc_now)
    asset_classes: list[AssetClass] = Field(default_factory=list)
    capabilities: dict[str, bool] = Field(default_factory=dict)
    # Credential *names* only -- values never appear.
    missing_env: list[str] = Field(default_factory=list)
    message: str | None = None
    limitations: list[str] = Field(default_factory=list)
    circuit_state: Literal["closed", "open", "half_open"] = "closed"


# --------------------------------------------------------------------------- #
# Data quality                                                                 #
# --------------------------------------------------------------------------- #


class DataQualityReport(_Model):
    """Result of the deterministic data-quality engine.

    ``status == FAIL`` is a hard gate: no candidate, no regime call and no LLM
    analysis may be released on top of failing data.
    """

    status: QualityStatus
    quality_score: float = Field(ge=0.0, le=1.0)
    freshness_ms: int = Field(ge=0)
    provider: str
    symbol: str | None = None
    timeframe: Timeframe | None = None
    candles_checked: int = 0
    closed_candles: int = 0
    warnings: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    checks_run: list[str] = Field(default_factory=list)
    checked_at_utc: datetime = Field(default_factory=utc_now)

    @property
    def is_blocking(self) -> bool:
        return self.status is QualityStatus.FAIL


# --------------------------------------------------------------------------- #
# Features and structure                                                       #
# --------------------------------------------------------------------------- #


class FeatureSnapshot(_Model):
    """Deterministic indicator values at the last **closed** bar.

    Every field is computed in :mod:`quantedge.services.indicators` from closed
    candles only. ``None`` means insufficient warm-up, never zero.
    """

    provider: str
    symbol: str
    timeframe: Timeframe
    computed_at_utc: datetime = Field(default_factory=utc_now)
    as_of_candle_close_utc: datetime
    bars_used: int
    warmup_satisfied: bool

    close: Decimal

    simple_return: float | None = None
    log_return: float | None = None

    ema_9: float | None = None
    ema_20: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None

    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    atr_14: float | None = None
    atr_percent: float | None = None
    adx_14: float | None = None
    plus_di_14: float | None = None
    minus_di_14: float | None = None

    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_width: float | None = None
    bb_percent_b: float | None = None

    roc_10: float | None = None
    realized_volatility_20: float | None = None
    volume_change_percent: float | None = None

    body_ratio: float | None = None
    upper_wick_ratio: float | None = None
    lower_wick_ratio: float | None = None

    ema_20_slope: float | None = None
    ema_50_slope: float | None = None
    sma_50_slope: float | None = None

    distance_from_high_20: float | None = None
    distance_from_low_20: float | None = None
    distance_from_high_50: float | None = None
    distance_from_low_50: float | None = None

    missing_features: list[str] = Field(default_factory=list)


class StructureEvent(_Model):
    """A single structural event: a break, a character change or a sweep.

    ``direction`` is always the *implied directional bias* the event carries,
    never merely which side of the chart it happened on. A sweep of highs takes
    liquidity above the market and is then rejected, so its bias is ``DOWN``.
    Encoding it the other way round would make an event's direction mean two
    different things depending on ``event_type``.

    ``level_confirmed_at_index`` exists to make the no-lookahead property
    auditable from the output alone: it must always be strictly less than
    ``occurred_at_index``, i.e. the level was knowable before the bar that acted
    on it. A consumer can assert this without re-reading the candles.

    ``confidence`` is a transparent rule-based weight in ``[0, 1]`` derived from
    displacement, level maturity and trend agreement. It is **not** a calibrated
    probability, not an accuracy figure and not a win rate.
    """

    event_type: Literal["BOS", "CHOCH", "LIQUIDITY_SWEEP"]
    direction: SignalDirection
    level: Decimal = Field(description="The swing price that was broken or swept")
    price: Decimal = Field(description="Close that broke it, or extreme that swept it")
    occurred_at_index: int = Field(ge=0)
    occurred_at_utc: datetime
    level_index: int = Field(ge=0, description="Bar index of the swing forming the level")
    level_confirmed_at_index: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    invalidation: str


class LiquidityPool(_Model):
    """Two or more confirmed swings resting at effectively the same price.

    Equal highs and equal lows are where stop orders accumulate. The cluster is
    reported with the tolerance that produced it, because "equal" is a
    volatility-relative judgement -- two highs 3 ticks apart are equal on an
    index future and are not on a low-priced altcoin.
    """

    kind: Literal["EQUAL_HIGHS", "EQUAL_LOWS"]
    price: Decimal = Field(description="Mean price of the cluster members")
    touches: int = Field(ge=2)
    indices: list[int] = Field(default_factory=list)
    confirmed_at_index: int = Field(ge=0)
    tolerance_used: Decimal
    swept: bool = Field(
        default=False, description="A later bar's extreme cleared the pool by the tolerance"
    )


class StructureReport(_Model):
    """Price-structure analysis from confirmed swings only.

    A swing is confirmed only after ``swing_confirmation_bars`` later bars
    exist. No forward-looking pivots are ever produced.
    """

    symbol: str
    timeframe: Timeframe
    swing_highs: list[dict[str, Any]] = Field(default_factory=list)
    swing_lows: list[dict[str, Any]] = Field(default_factory=list)
    has_higher_highs: bool = False
    has_higher_lows: bool = False
    has_lower_highs: bool = False
    has_lower_lows: bool = False
    structure: Literal["UPTREND", "DOWNTREND", "RANGE", "UNCLEAR"] = "UNCLEAR"
    breakout_candidate: bool = False
    breakout_direction: SignalDirection | None = None
    failed_breakout: bool = False
    nearest_resistance: Decimal | None = None
    nearest_support: Decimal | None = None
    notes: list[str] = Field(default_factory=list)

    # -- structural events and liquidity ------------------------------------ #
    events: list[StructureEvent] = Field(
        default_factory=list, description="BOS / CHOCH / sweep events, oldest first"
    )
    last_bos: StructureEvent | None = None
    last_choch: StructureEvent | None = None
    equal_highs: list[LiquidityPool] = Field(default_factory=list)
    equal_lows: list[LiquidityPool] = Field(default_factory=list)

    # -- internal vs external structure ------------------------------------- #
    # Two swing scales. External is the structure a higher timeframe would see;
    # internal is the detail inside the current external leg. They disagree
    # constantly, and that disagreement is information (a pullback in a trend),
    # not noise to be averaged away -- so both are reported.
    internal_structure: Literal["UPTREND", "DOWNTREND", "RANGE", "UNCLEAR"] = "UNCLEAR"
    external_structure: Literal["UPTREND", "DOWNTREND", "RANGE", "UNCLEAR"] = "UNCLEAR"

    # -- position within the last external range ---------------------------- #
    premium_discount: Literal["PREMIUM", "DISCOUNT", "EQUILIBRIUM"] | None = None
    range_position: float | None = Field(
        default=None,
        description="Close as a fraction of the last external swing range. "
        "Deliberately unbounded: <0 or >1 means price has left that range, "
        "which is a real and important state that clamping would erase.",
    )

    structure_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Rule-based agreement score across swing count, internal/external "
        "consistency and recent events. Not a probability.",
    )


class TimeframeView(_Model):
    """One timeframe's contribution to a multi-timeframe snapshot."""

    role: Literal["execution", "confirmation", "regime"]
    timeframe: Timeframe
    provider: str
    bars_available: int
    last_closed_candle_utc: datetime | None = None
    forming_candle_present: bool = False
    forming_candle_excluded_from_analysis: bool = True
    quality: DataQualityReport
    features: FeatureSnapshot | None = None
    structure: StructureReport | None = None
    regime: RegimeReport | None = None


class MultiTimeframeSnapshot(_Model):
    """Execution / confirmation / regime alignment for one symbol."""

    symbol: str
    asset_class: AssetClass
    horizon: str
    generated_at_utc: datetime = Field(default_factory=utc_now)
    views: list[TimeframeView]
    aligned_direction: SignalDirection | None = None
    alignment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Agreement among the views that actually carry a direction, and how much of
    # the timeframe stack that was. They answer different questions and a caller
    # needs both: one directional view out of three renormalises to a perfect
    # 1.0 agreement, which is true and also nearly worthless on its own.
    # ``participation`` is the fraction of non-failed weight that voted, so a
    # gate can require both unanimity and a quorum.
    participation: float = Field(default=0.0, ge=0.0, le=1.0)
    abstaining_roles: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RegimeReport(_Model):
    """Deterministic market-regime classification.

    ``heuristic_score`` is a transparent rule-based confidence in the label. It
    is explicitly **not** a calibrated probability and must never be presented
    as one.
    """

    regime: MarketRegime
    heuristic_score: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    version: str
    symbol: str | None = None
    timeframe: Timeframe | None = None
    computed_at_utc: datetime = Field(default_factory=utc_now)


# --------------------------------------------------------------------------- #
# Sessions and events                                                          #
# --------------------------------------------------------------------------- #


class SessionState(_Model):
    """Which liquidity sessions are open, in UTC."""

    timestamp_utc: datetime = Field(default_factory=utc_now)
    open_sessions: list[str] = Field(default_factory=list)
    active_overlaps: list[str] = Field(default_factory=list)
    liquidity: Literal["low", "medium", "high", "highest", "closed"] = "medium"
    is_weekend_forex_closed: bool = False
    in_thin_liquidity_window: bool = False
    notes: list[str] = Field(default_factory=list)


class EconomicEvent(_Model):
    """A scheduled economic release, normalized across providers."""

    provider: str
    event_id: str | None = None
    title: str
    country: str | None = None
    currency: str | None = None
    impact: EventImpact = EventImpact.UNKNOWN
    scheduled_utc: datetime
    actual: str | None = None
    forecast: str | None = None
    previous: str | None = None
    retrieved_at_utc: datetime = Field(default_factory=utc_now)


class EventRiskReport(_Model):
    """Unified event-risk assessment.

    ``status`` is ``UNKNOWN`` whenever no calendar provider could answer. It is
    never assumed to be ``LOW``.
    """

    status: EventRiskStatus
    events: list[EconomicEvent] = Field(default_factory=list)
    blocked_until_utc: datetime | None = None
    provider: str | None = None
    data_freshness: str | None = None
    warnings: list[str] = Field(default_factory=list)
    currencies_checked: list[str] = Field(default_factory=list)
    window_start_utc: datetime | None = None
    window_end_utc: datetime | None = None


class NewsItem(_Model):
    provider: str
    headline: str
    summary: str | None = None
    url: str | None = None
    source: str | None = None
    symbols: list[str] = Field(default_factory=list)
    published_at_utc: datetime
    retrieved_at_utc: datetime = Field(default_factory=utc_now)


# --------------------------------------------------------------------------- #
# Scanner                                                                      #
# --------------------------------------------------------------------------- #


class ScanCandidate(_Model):
    """A deterministic pre-scan candidate.

    Naming discipline: every score below is a transparent heuristic derived
    from published formulas. None of them is an accuracy figure, a win rate, or
    a probability. ``calibrated_probability`` deliberately does not exist on
    this model -- it appears only on :class:`LLMSignalResponse`, and only once a
    calibration model fitted on unseen data is registered.
    """

    symbol: str
    asset_class: AssetClass
    horizon: str
    direction: SignalDirection
    provider: str

    heuristic_score: float = Field(ge=0.0, le=1.0)
    trend_score: float = Field(ge=0.0, le=1.0)
    momentum_score: float = Field(ge=0.0, le=1.0)
    volatility_score: float = Field(ge=0.0, le=1.0)
    data_quality_score: float = Field(ge=0.0, le=1.0)
    evidence_agreement_score: float = Field(ge=0.0, le=1.0)

    regime: MarketRegime
    reference_price: Decimal
    quote_freshness_ms: int
    supporting_evidence: list[str] = Field(default_factory=list)
    contradictory_evidence: list[str] = Field(default_factory=list)
    event_risk: EventRiskStatus
    session_liquidity: str
    generated_at_utc: datetime = Field(default_factory=utc_now)
    scanner_version: str
    warning: str = STANDARD_WARNING


class ScanRejection(_Model):
    """Why a symbol did not become a candidate. Transparency by default."""

    symbol: str
    reason_code: str
    reason: str
    stage: str


class ScanResult(_Model):
    """Outcome of one scan: what qualified, what did not, and on what data.

    ``quality_reports`` carries the execution-timeframe
    :class:`DataQualityReport` per symbol -- for rejected symbols too, since
    "the data was too poor to judge" and "the setup was not there" are different
    answers. Downstream callers report that verdict instead of inferring one
    from the candidate's score.
    """

    horizon: str
    requested_symbols: list[str] = Field(default_factory=list)
    scanned: int = 0
    candidates: list[ScanCandidate] = Field(default_factory=list)
    rejections: list[ScanRejection] = Field(default_factory=list)
    quality_reports: dict[str, DataQualityReport] = Field(default_factory=dict)
    generated_at_utc: datetime = Field(default_factory=utc_now)
    scanner_version: str
    warnings: list[str] = Field(default_factory=list)
    warning: str = STANDARD_WARNING


# --------------------------------------------------------------------------- #
# Signal context and LLM contracts                                             #
# --------------------------------------------------------------------------- #


class SignalContext(_Model):
    """The complete, verified payload handed to the LLM.

    Contains only data that was actually retrieved and validated.
    ``missing_information`` is mandatory and explicit: the model must be told
    what is *not* known so it cannot quietly assume it.
    """

    generated_at_utc: datetime = Field(default_factory=utc_now)
    symbol: str
    asset_class: AssetClass
    horizon: str

    data_sources: dict[str, str] = Field(default_factory=dict)
    quality: DataQualityReport
    quote: dict[str, Any] | None = None
    multi_timeframe: MultiTimeframeSnapshot | None = None
    regime: RegimeReport | None = None
    session: SessionState | None = None
    event_risk: EventRiskReport | None = None

    candidate_direction: SignalDirection | None = None
    heuristic_score: float | None = None
    supporting_evidence: list[str] = Field(default_factory=list)
    contradictory_evidence: list[str] = Field(default_factory=list)
    historical_statistics: dict[str, Any] | None = None
    missing_information: list[str] = Field(default_factory=list)

    calibration_model_available: bool = False
    warning: str = STANDARD_WARNING


class LLMSignalResponse(_Model):
    """Strict LLM output contract.

    ``extra="forbid"`` is the anti-fabrication mechanism: a model that invents
    an extra field (``"win_rate": 0.87``) fails validation instead of having
    the value silently accepted.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: SignalStatus
    asset: str | None = None
    direction: SignalDirection | None = None
    generated_at_utc: datetime
    entry_window_start_utc: datetime | None = None
    entry_window_end_utc: datetime | None = None
    expiry_utc: datetime | None = None
    horizon: str | None = None
    reference_price: Decimal | None = None
    regime: str | None = None
    calibrated_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    heuristic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradictory_evidence: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    warning: str = STANDARD_WARNING

    @field_validator("warning")
    @classmethod
    def _force_standard_warning(cls, v: str) -> str:
        """The disclaimer is not negotiable by the model."""
        return STANDARD_WARNING


class AIDecision(_Model):
    """A persisted LLM decision, joined to the deterministic context."""

    decision_id: str | None = None
    symbol: str
    horizon: str
    status: SignalStatus
    direction: SignalDirection | None = None
    reference_price: Decimal | None = None
    expiry_utc: datetime | None = None
    regime: str | None = None
    heuristic_score: float | None = None
    calibrated_probability: float | None = None
    supporting_evidence: list[str] = Field(default_factory=list)
    contradictory_evidence: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    llm_provider: str | None = None
    llm_model: str | None = None
    strategy_version: str | None = None
    scanner_version: str | None = None
    data_quality_status: QualityStatus | None = None
    created_at_utc: datetime = Field(default_factory=utc_now)
    warning: str = STANDARD_WARNING


class SettledSignal(_Model):
    """Immutable settlement record for a previously issued decision."""

    signal_id: str
    symbol: str
    horizon: str
    direction: SignalDirection | None = None
    reference_price: Decimal | None = None
    settlement_price: Decimal | None = None
    outcome: SettlementOutcome
    expiry_utc: datetime | None = None
    settled_at_utc: datetime = Field(default_factory=utc_now)
    settlement_provider: str | None = None
    notes: list[str] = Field(default_factory=list)


class PerformanceSummary(_Model):
    """Realized outcome statistics.

    These are **observed historical frequencies over a finite sample**, not
    predictions and not a calibrated probability of future outcomes. The
    ``disclaimer`` field travels with the data so the caveat cannot be dropped.
    """

    symbol: str | None = None
    horizon: str | None = None
    total_signals: int = 0
    settled_signals: int = 0
    pending_signals: int = 0
    wins: int = 0
    losses: int = 0
    flat: int = 0
    void: int = 0
    observed_win_rate: float | None = None
    sample_too_small: bool = True
    generated_at_utc: datetime = Field(default_factory=utc_now)
    disclaimer: str = (
        "Observed historical frequency on a finite sample. Not a calibrated "
        "probability and not predictive of future outcomes."
    )


class TradeMemory(_Model):
    """Post-mortem root-cause analysis record saved to memory."""

    memory_id: str
    signal_id: str
    symbol: str
    asset_class: AssetClass = AssetClass.CRYPTO
    horizon: str = "swing"
    regime: MarketRegime = MarketRegime.UNCERTAIN
    pattern: str = "general"
    outcome: SettlementOutcome
    reference_price: Decimal | None = None
    exit_price: Decimal | None = None
    root_cause: str
    key_lessons: list[str] = Field(default_factory=list)
    do_rules: list[str] = Field(default_factory=list)
    dont_rules: list[str] = Field(default_factory=list)
    user_notes: str | None = None
    created_at_utc: datetime = Field(default_factory=utc_now)


class TradeRecommendation(_Model):
    """Memory-augmented trade recommendation payload."""

    recommendation_id: str
    symbol: str
    asset_class: AssetClass
    horizon: str
    direction: SignalDirection
    valid_from_utc: datetime
    valid_until_utc: datetime
    reference_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    risk_reward_ratio: float
    risk_level: str = "MODERATE_RISK"
    recommended_venue: str = "Binance"
    # The regime the classifier returned, carried as its own field so consumers
    # do not have to parse it back out of ``rationale``. A settled trade is
    # filed in memory under this value, and a wrong parse would file it under a
    # regime it never traded in.
    regime: str | None = None
    memory_consulted_count: int = 0
    key_lessons_applied: list[str] = Field(default_factory=list)
    # DON'T rules carried forward from losses whose measured cause recurred on
    # this symbol and horizon. Each is a statement about trades that already
    # happened, so it qualifies the setup without claiming anything about how
    # this one resolves; the count of prior occurrences travels with the text so
    # a rule seen once is not presented as a pattern.
    memory_rules_applied: list[str] = Field(default_factory=list)
    heuristic_score: float = 0.75
    rationale: str = ""
    # Caveats that qualify the recommendation without withdrawing it: degraded
    # data quality, unfavourable reward:risk. A setup can clear the bar to be
    # emitted and still carry something the trader should know, and burying that
    # in ``rationale`` prose leaves consumers parsing sentences to find it.
    warnings: list[str] = Field(default_factory=list)
    generated_at_utc: datetime = Field(default_factory=utc_now)


# Resolve the forward reference used by TimeframeView.regime.
TimeframeView.model_rebuild()
