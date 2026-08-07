"""SQLAlchemy table definitions for the 15 persisted tables.

Three properties are structural rather than conventional, and each exists to
prevent a specific way stored market data goes wrong.

**Candles are unique on ``(provider, symbol, timeframe, open_time_utc)``.**
Two providers legitimately disagree about the same minute -- different feeds,
different aggregation, different last-trade. Keying on the provider keeps both
without either overwriting the other, and makes "which feed said this" answerable
after the fact. Dropping ``provider`` from the key would silently merge two
different measurements of the same instant.

**``audit_logs`` is append-only and ``settled_signals`` is immutable.** Both are
enforced at the repository layer rather than trusted to callers. A settled signal
that can be edited is a performance record that can be edited, and a performance
record that can be edited is not evidence of anything.

**Every timestamp column is timezone-aware UTC.** SQLite does not enforce this,
so :class:`UtcDateTime` normalizes on the way in and re-attaches UTC on the way
out. A naive datetime read back from storage is the kind of bug that shifts a
candle by the host's offset and is invisible until the host moves.

Only closed candles are stored. The forming candle lives in memory and never
reaches this layer -- Rule 9 applies to persistence as much as to calculation,
because a stored forming bar becomes indistinguishable from history the moment
the process restarts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

__all__ = [
    "TABLE_NAMES",
    "AuditLog",
    "Base",
    "Candle",
    "DataQualityReportRow",
    "EconomicEventRow",
    "FeatureSnapshotRow",
    "NewsItemRow",
    "ProviderHealthRow",
    "ProviderQuota",
    "Quote",
    "RegimeReportRow",
    "ScanResultRow",
    "SettledSignalRow",
    "SignalRow",
    "StructureReportRow",
    "Symbol",
]

# Prices need exact decimal arithmetic; 24 digits with 12 after the point covers
# both a $100k index and a satoshi-denominated altcoin without rounding either.
_PRICE = Numeric(24, 12)
_QTY = Numeric(32, 12)


class UtcDateTime(TypeDecorator[datetime]):
    """A ``DateTime`` that refuses to store or return a naive value.

    SQLite has no native timezone handling and Postgres will happily accept a
    naive datetime into a ``timestamptz`` column by assuming the server zone.
    Both paths produce timestamps that are wrong by an offset nobody recorded, so
    conversion happens here where it cannot be forgotten.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"naive datetime rejected: {value!r} has no timezone")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    """Declarative base. ``JSON`` maps to ``JSONB`` on Postgres automatically."""

    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON, Decimal: _PRICE}


class _Row(Base):
    """Surrogate key plus the row's own creation time.

    ``created_at_utc`` records when *this system* learned the fact, which is a
    different question from when the fact was true. Both are kept on every table
    that has an event time, because a candle written four hours late is a
    different situation from one written on time and the row should say so.
    """

    __abstract__ = True

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)


# --------------------------------------------------------------------------- #
# 1. symbols -- the allowlist, not merely a cache                              #
# --------------------------------------------------------------------------- #


class Symbol(_Row):
    """Tradeable instruments, one row per provider's view of a symbol.

    This table doubles as the symbol allowlist the request validators check
    against. A symbol absent here is rejected rather than forwarded to a
    provider, which keeps an arbitrary user string out of an outbound URL.
    """

    __tablename__ = "symbols"
    __table_args__ = (
        UniqueConstraint("provider", "symbol", name="uq_symbols_provider_symbol"),
        Index("ix_symbols_asset_class", "asset_class"),
    )

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(32), nullable=False)
    base_asset: Mapped[str | None] = mapped_column(String(32))
    quote_asset: Mapped[str | None] = mapped_column(String(32))
    price_precision: Mapped[int | None] = mapped_column(Integer)
    quantity_precision: Mapped[int | None] = mapped_column(Integer)
    min_quantity: Mapped[Decimal | None] = mapped_column(_QTY)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    enabled_for_scanning: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON)


# --------------------------------------------------------------------------- #
# 2. candles -- closed bars only                                               #
# --------------------------------------------------------------------------- #


class Candle(_Row):
    """One closed candle from one provider.

    ``is_closed`` is stored and constrained to ``True`` by the repository rather
    than being omitted. Keeping the column makes the guarantee legible in the
    data itself: a reader querying this table can confirm the property instead of
    having to trust that the writer honoured it.
    """

    __tablename__ = "candles"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "symbol",
            "timeframe",
            "open_time_utc",
            name="uq_candles_provider_symbol_timeframe_open",
        ),
        # The scanner's hot path: newest N bars for one symbol and timeframe.
        Index("ix_candles_lookup", "symbol", "timeframe", "open_time_utc"),
    )

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    open_time_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    close_time_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    open: Mapped[Decimal] = mapped_column(_PRICE, nullable=False)
    high: Mapped[Decimal] = mapped_column(_PRICE, nullable=False)
    low: Mapped[Decimal] = mapped_column(_PRICE, nullable=False)
    close: Mapped[Decimal] = mapped_column(_PRICE, nullable=False)
    volume: Mapped[Decimal] = mapped_column(_QTY, nullable=False)
    quote_volume: Mapped[Decimal | None] = mapped_column(_QTY)
    trade_count: Mapped[int | None] = mapped_column(Integer)
    is_closed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


# --------------------------------------------------------------------------- #
# 3. quotes                                                                    #
# --------------------------------------------------------------------------- #


class Quote(_Row):
    """A point-in-time bid/ask snapshot.

    Quotes are sampled, not exhaustive: this table is a record of what was
    observed when something asked, not a tick archive. ``as_of_utc`` is the
    provider's timestamp where one was given and never a local substitute, so a
    stale feed reads as stale rather than as fresh.
    """

    __tablename__ = "quotes"
    __table_args__ = (Index("ix_quotes_symbol_time", "symbol", "as_of_utc"),)

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(32), nullable=False)
    as_of_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    bid: Mapped[Decimal | None] = mapped_column(_PRICE)
    ask: Mapped[Decimal | None] = mapped_column(_PRICE)
    mid: Mapped[Decimal | None] = mapped_column(_PRICE)
    last: Mapped[Decimal | None] = mapped_column(_PRICE)
    spread_absolute: Mapped[Decimal | None] = mapped_column(_PRICE)
    spread_basis_points: Mapped[float | None] = mapped_column(Numeric(12, 4))


# --------------------------------------------------------------------------- #
# 4-5. provider health and quota                                               #
# --------------------------------------------------------------------------- #


class ProviderHealthRow(_Row):
    """Result of one provider health probe.

    Kept as history rather than a single current-state row: "Alpha Vantage is
    down" and "Alpha Vantage has been down for six hours" call for different
    responses, and only the second is answerable from a log.
    """

    __tablename__ = "provider_health"
    __table_args__ = (Index("ix_provider_health_provider_time", "provider", "checked_at_utc"),)

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    checked_at_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Numeric(12, 3))
    credential_present: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    endpoint_reachable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Redacted before it arrives here. The repository does not re-check, so the
    # redaction must already have happened at the error boundary.
    message: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class ProviderQuota(_Row):
    """Request counters that must survive a process restart.

    This table exists because an in-process rate limiter cannot enforce a daily
    cap. Alpha Vantage allows 25 requests a day; a limiter holding that count in
    memory resets to zero every time the CLI is invoked, so twenty invocations
    spend the day's budget while each one believes it has spent nothing.

    The window is keyed by ``(provider, window_kind, window_start_utc)`` so the
    daily and per-minute budgets are separate rows rather than one counter trying
    to mean both.
    """

    __tablename__ = "provider_quota"
    __table_args__ = (
        UniqueConstraint(
            "provider", "window_kind", "window_start_utc", name="uq_provider_quota_window"
        ),
    )

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    window_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    window_start_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    requests_made: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    requests_allowed: Mapped[int | None] = mapped_column(Integer)
    last_request_utc: Mapped[datetime | None] = mapped_column(UtcDateTime)


# --------------------------------------------------------------------------- #
# 6-9. analysis outputs                                                        #
# --------------------------------------------------------------------------- #


class DataQualityReportRow(_Row):
    """A stored quality assessment.

    ``quality_score`` is persisted alongside ``status`` because they answer
    different questions and neither substitutes for the other: the score is how
    good the data was, the status is whether it was usable at all. A row can
    legitimately carry a middling score and a hard FAIL.
    """

    __tablename__ = "data_quality_reports"
    __table_args__ = (Index("ix_quality_symbol_time", "symbol", "assessed_at_utc"),)

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    # Nullable, both of them: a report can assess a provider's feed as a whole
    # rather than one series, and a freshness failure is frequently exactly that.
    # Substituting a placeholder symbol would file a feed-wide failure against an
    # instrument that was never checked.
    symbol: Mapped[str | None] = mapped_column(String(64))
    timeframe: Mapped[str | None] = mapped_column(String(16))
    assessed_at_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    quality_score: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    is_blocking: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    bars_checked: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    checks: Mapped[list[Any] | None] = mapped_column(JSON)
    failures: Mapped[list[Any] | None] = mapped_column(JSON)
    warnings: Mapped[list[Any] | None] = mapped_column(JSON)


class FeatureSnapshotRow(_Row):
    """Deterministically computed indicators for one bar.

    Stored as JSON rather than one column per indicator. The set of indicators
    changes as the system grows, and a schema migration per indicator would make
    adding one a database event. ``indicator_version`` records which code
    produced the values, so a later change to a formula does not silently
    reinterpret old rows.
    """

    __tablename__ = "feature_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "symbol", "timeframe", "as_of_utc", name="uq_features_symbol_timeframe_time"
        ),
    )

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    as_of_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    bars_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    indicator_version: Mapped[str] = mapped_column(String(32), nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    unavailable: Mapped[list[Any] | None] = mapped_column(JSON)


class StructureReportRow(_Row):
    """Stored swing/structure analysis."""

    __tablename__ = "structure_reports"
    __table_args__ = (
        UniqueConstraint(
            "symbol", "timeframe", "as_of_utc", name="uq_structure_symbol_timeframe_time"
        ),
    )

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    as_of_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    bars_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    structure_version: Mapped[str] = mapped_column(String(32), nullable=False)
    structure_label: Mapped[str] = mapped_column(String(32), nullable=False)
    breakout_candidate: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    breakout_direction: Mapped[str | None] = mapped_column(String(16))
    failed_breakout: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    nearest_support: Mapped[Decimal | None] = mapped_column(_PRICE)
    nearest_resistance: Mapped[Decimal | None] = mapped_column(_PRICE)
    swing_highs: Mapped[list[Any] | None] = mapped_column(JSON)
    swing_lows: Mapped[list[Any] | None] = mapped_column(JSON)
    notes: Mapped[list[Any] | None] = mapped_column(JSON)


class RegimeReportRow(_Row):
    """Stored regime classification."""

    __tablename__ = "regime_reports"
    __table_args__ = (
        UniqueConstraint(
            "symbol", "timeframe", "as_of_utc", name="uq_regime_symbol_timeframe_time"
        ),
    )

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    as_of_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    regime_version: Mapped[str] = mapped_column(String(32), nullable=False)
    regime: Mapped[str] = mapped_column(String(32), nullable=False)
    heuristic_score: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    supporting_evidence: Mapped[list[Any] | None] = mapped_column(JSON)
    contradictions: Mapped[list[Any] | None] = mapped_column(JSON)


# --------------------------------------------------------------------------- #
# 10-11. external events                                                      #
# --------------------------------------------------------------------------- #


class EconomicEventRow(_Row):
    """A scheduled economic release, normalized across providers."""

    __tablename__ = "economic_events"
    __table_args__ = (Index("ix_events_time_currency", "scheduled_utc", "currency"),)

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    event_id_from_provider: Mapped[str | None] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    country: Mapped[str | None] = mapped_column(String(8))
    currency: Mapped[str | None] = mapped_column(String(8))
    impact: Mapped[str] = mapped_column(String(16), nullable=False, default="UNKNOWN")
    scheduled_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    actual: Mapped[str | None] = mapped_column(String(64))
    forecast: Mapped[str | None] = mapped_column(String(64))
    previous: Mapped[str | None] = mapped_column(String(64))
    retrieved_at_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)


class NewsItemRow(_Row):
    """A news headline, normalized across providers."""

    __tablename__ = "news_items"
    __table_args__ = (Index("ix_news_published", "published_at_utc"),)

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(512))
    source: Mapped[str | None] = mapped_column(String(128))
    symbols: Mapped[list[Any] | None] = mapped_column(JSON)
    published_at_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    retrieved_at_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)


# --------------------------------------------------------------------------- #
# 12-15. scanner outputs and settled signals                                   #
# --------------------------------------------------------------------------- #


class ScanResultRow(_Row):
    """A scan cycle's output: candidates, rejections, and summary stats."""

    __tablename__ = "scan_results"
    __table_args__ = (Index("ix_scan_results_scanned_at", "scanned_at_utc"),)

    scan_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    scanned_at_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    horizon: Mapped[str] = mapped_column(String(16), nullable=False)
    symbols_scanned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    candidates_found: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejections_logged: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    runtime_ms: Mapped[float | None] = mapped_column(Numeric(12, 3))
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class SignalRow(_Row):
    """A candidate that passed gates and reached the LLM.

    The LLM's output is stored inline as JSON, not split across columns. The
    signal's attributes (direction, status, confidence) come from parsing that
    JSON rather than being duplicated into typed columns.
    """

    __tablename__ = "signals"
    __table_args__ = (Index("ix_signals_symbol_time", "symbol", "signal_time_utc"),)

    scan_id: Mapped[str] = mapped_column(String(64), nullable=False)
    signal_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    signal_time_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon: Mapped[str] = mapped_column(String(16), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="CANDIDATE")
    llm_response: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    context_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    version: Mapped[str] = mapped_column(String(32), nullable=False)


class SettledSignalRow(_Row):
    """A signal whose outcome is known and recorded.

    This table is **immutable**: the repository forbids updates. A settled signal
    is a closed record, and editing one is editing the performance history. To
    record a change in view about a signal after it has settled, write a new row
    with a note pointing to the original.
    """

    __tablename__ = "settled_signals"
    __table_args__ = (Index("ix_settled_signals_settled_at", "settled_at_utc"),)

    signal_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon: Mapped[str] = mapped_column(String(16), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    reference_price: Mapped[Decimal] = mapped_column(_PRICE, nullable=False)
    settlement_price: Mapped[Decimal] = mapped_column(_PRICE, nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    expiry_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    settled_at_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    # Which feed priced the settlement. Two providers disagreeing by a tick
    # decides a win from a loss on a short horizon, so the source is recorded
    # rather than assumed to be whichever provider is primary today.
    settlement_provider: Mapped[str | None] = mapped_column(String(64))
    notes: Mapped[str | None] = mapped_column(Text)


class AuditLog(_Row):
    """An append-only log of system actions and state changes.

    This table exists to answer "what happened" after an unexpected state. It is
    **append-only**: the repository forbids updates and deletes. Dropping a row
    that records an error would turn a readable event stream into one where events
    vanish, which is a more severe failure than the original error.

    ``user_id`` is a string rather than a foreign key because the audit log must
    survive even when the user table is unavailable. Same for ``symbol``.
    """

    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_logs_event_time", "event_time_utc"),)

    event_time_utc: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[str | None] = mapped_column(String(128))
    symbol: Mapped[str | None] = mapped_column(String(64))
    provider: Mapped[str | None] = mapped_column(String(64))
    # Redacted before it arrives here. Repository does not sanitize; the caller must.
    message: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON)


TABLE_NAMES = [
    "symbols",
    "candles",
    "quotes",
    "provider_health",
    "provider_quota",
    "data_quality_reports",
    "feature_snapshots",
    "structure_reports",
    "regime_reports",
    "economic_events",
    "news_items",
    "scan_results",
    "signals",
    "settled_signals",
    "audit_logs",
]
