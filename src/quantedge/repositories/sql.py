"""SQL repository: the only sanctioned path from contracts into storage.

Every method takes or returns a Pydantic contract, never an ORM row. Rows are an
implementation detail of this module, and letting one escape would put a live
database session behind an object the caller thinks is plain data -- which is how
a lazy-load ends up firing inside a request that already closed its session.

Two guarantees are enforced here, not documented and hoped for
---------------------------------------------------------------
``settled_signals`` is immutable and ``audit_logs`` is append-only. The
enforcement is a pair of ORM event listeners rather than the absence of an update
method, because the absence of a method only stops the caller who reads the API.
A listener stops any code path in the process, including a future one written by
someone who never read this docstring.

The cost is honest: SQL issued outside the ORM (``engine.execute("UPDATE ...")``)
bypasses the listener, and so does a database administrator. This is a guard
against accident and drift, not against a determined actor with a psql prompt.
Postgres deployments should add a trigger for that; the listener is the portable
half that also covers SQLite.

Idempotence
-----------
Writes that can plausibly repeat -- the same candle arriving from a backfill and
a stream, a scan re-running after a crash -- are idempotent on their natural key.
The candle path uses ``INSERT ... ON CONFLICT DO NOTHING``, so re-ingesting an
overlapping window is a no-op rather than a duplicate-key error. The first
observation wins deliberately: a provider that restates a closed bar is
reporting a correction, and a silent overwrite would erase the fact that the
value changed.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, event, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError

from quantedge.contracts import (
    AIDecision,
    AssetClass,
    Candle,
    DataQualityReport,
    EconomicEvent,
    FeatureSnapshot,
    MarketRegime,
    NewsItem,
    ProviderHealth,
    Quote,
    RegimeReport,
    SettledSignal,
    SettlementOutcome,
    SignalDirection,
    SignalStatus,
    StructureReport,
    SymbolInfo,
    Timeframe,
    TradeMemory,
    utc_now,
)
from quantedge.errors import PersistenceError
from quantedge.logging import get_logger
from quantedge.repositories import models as m
from quantedge.repositories.database import get_session_factory, session_scope

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.orm import Session, sessionmaker

__all__ = ["SqlRepository", "install_immutability_guards"]

log = get_logger(__name__)

_IMMUTABLE = (m.SettledSignalRow, m.AuditLog)


def _parse_utc(raw: str) -> datetime | None:
    """Parse a stored ISO timestamp, or ``None`` when it cannot be read.

    JSON columns round-trip datetimes as strings, and SQLite gives them back
    without the tzinfo Postgres preserves. A naive result is assumed UTC because
    every write in this schema goes through :func:`utc_now`; anything
    unparseable returns ``None`` so the caller skips the row rather than
    settling against a timestamp it had to guess.
    """
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _window_start(stamp: datetime, kind: str) -> datetime:
    """Truncate ``stamp`` to the start of the window.

    A day window starts at midnight UTC; a minute window starts at the top of
    the minute. The window is keyed by its start so rows naturally partition by
    calendar unit instead of sliding with the query time.
    """
    if kind == "day":
        return stamp.replace(hour=0, minute=0, second=0, microsecond=0)
    if kind == "minute":
        return stamp.replace(second=0, microsecond=0)
    return stamp


def _spread_bps(quote: Quote) -> float | None:
    """Spread in basis points, or None when the spread or mid is unavailable."""
    if quote.spread is None or quote.mid is None or quote.mid == 0:
        return None
    return float(quote.spread / quote.mid * 10_000)


def _candle_from_row(row: m.Candle) -> Candle:
    return Candle(
        provider=row.provider,
        symbol=row.symbol,
        asset_class=AssetClass(row.asset_class),
        timeframe=Timeframe(row.timeframe),
        open_time_utc=row.open_time_utc,
        close_time_utc=row.close_time_utc,
        open=row.open,
        high=row.high,
        low=row.low,
        close=row.close,
        volume=row.volume,
        quote_volume=row.quote_volume,
        trade_count=row.trade_count,
        is_closed=row.is_closed,
        # The row's insert time, not a fresh stamp. Letting the field default
        # would make two reads of the same bar unequal, which quietly breaks any
        # caller that dedups or caches candles by value. What is stored is when
        # this system took delivery of the bar; nothing here invents a time.
        received_at_utc=row.created_at_utc,
    )


def _reject_mutation(_mapper: Any, _connection: Any, target: Any) -> None:
    raise PersistenceError(
        f"{type(target).__tablename__} is append-only; "
        f"UPDATE and DELETE are refused. Write a new row instead."
    )


def install_immutability_guards() -> None:
    """Attach the append-only listeners. Idempotent.

    Called at import time so the guard is in place before any session exists.
    A guard installed by the first repository construction would leave a window
    in which a direct ORM user could mutate the tables freely.
    """
    for cls in _IMMUTABLE:
        for verb in ("before_update", "before_delete"):
            if not event.contains(cls, verb, _reject_mutation):
                event.listen(cls, verb, _reject_mutation)


install_immutability_guards()


def _insert_ignore(
    session: Session, table: Any, rows: list[dict[str, Any]], *, index: list[str]
) -> Any:
    """Dialect-appropriate ``INSERT ... ON CONFLICT DO NOTHING``.

    SQLAlchemy has no portable spelling for this, and the alternative -- SELECT
    then INSERT -- is a race: two workers checking the same missing candle both
    see it absent and both insert.
    """
    backend = session.get_bind().dialect.name
    if backend == "postgresql":
        # The two dialects' Insert types are unrelated, so the variable is Any
        # rather than either one of them.
        stmt: Any = pg_insert(table).values(rows).on_conflict_do_nothing(index_elements=index)
    elif backend == "sqlite":
        stmt = sqlite_insert(table).values(rows).on_conflict_do_nothing(index_elements=index)
    else:  # pragma: no cover - only two backends are supported
        raise PersistenceError(f"unsupported backend for upsert: {backend}")
    return session.execute(stmt)


class SqlRepository:
    """Contract-in, contract-out persistence over SQLAlchemy.

    Each method opens and closes its own transaction unless a session is passed,
    so a caller doing one write does not have to think about transactions and a
    caller doing several can still make them atomic.
    """

    def __init__(self, factory: sessionmaker[Session] | None = None) -> None:
        self._factory = factory or get_session_factory()

    # ------------------------------------------------------------------ #
    # candles                                                            #
    # ------------------------------------------------------------------ #

    def save_candles(self, candles: Iterable[Candle]) -> int:
        """Persist closed candles. Returns the number of rows actually inserted.

        Unclosed candles are refused rather than filtered. Silently dropping them
        would let a caller believe a 200-bar write succeeded when 199 landed, and
        the missing bar is always the most recent one -- the bar the next
        calculation depends on.
        """
        batch = list(candles)
        if not batch:
            return 0
        forming = [c for c in batch if not c.is_closed]
        if forming:
            raise PersistenceError(
                f"refusing to store {len(forming)} forming candle(s): "
                f"only closed bars may be persisted (Rule 9)",
                details={"symbols": sorted({c.symbol for c in forming})},
            )

        now = utc_now()
        rows = [
            {
                "created_at_utc": now,
                "provider": c.provider,
                "symbol": c.symbol,
                "asset_class": c.asset_class.value,
                "timeframe": c.timeframe.value,
                "open_time_utc": c.open_time_utc,
                "close_time_utc": c.close_time_utc,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "quote_volume": c.quote_volume,
                "trade_count": c.trade_count,
                "is_closed": True,
            }
            for c in batch
        ]
        with session_scope(self._factory) as session:
            result = _insert_ignore(
                session,
                m.Candle,
                rows,
                index=["provider", "symbol", "timeframe", "open_time_utc"],
            )
            inserted = int(result.rowcount or 0)
        log.debug(
            "candles stored",
            extra={"offered": len(rows), "inserted": inserted},
        )
        return inserted

    def load_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        limit: int = 300,
        provider: str | None = None,
        since: datetime | None = None,
    ) -> list[Candle]:
        """The newest ``limit`` closed candles, returned oldest-first.

        The query orders newest-first to use the index and let ``LIMIT`` stop
        early, then reverses in Python. Every indicator in this system consumes
        oldest-first, so returning newest-first here would push a reversal into
        every call site and guarantee that one of them forgets.
        """
        tf = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe)
        stmt = (
            select(m.Candle)
            .where(
                m.Candle.symbol == symbol,
                m.Candle.timeframe == tf,
                m.Candle.is_closed.is_(True),
            )
            .order_by(m.Candle.open_time_utc.desc())
            .limit(max(1, limit))
        )
        if provider:
            stmt = stmt.where(m.Candle.provider == provider)
        if since is not None:
            stmt = stmt.where(m.Candle.open_time_utc >= since)

        with session_scope(self._factory) as session:
            rows = list(session.scalars(stmt))
        return [_candle_from_row(r) for r in reversed(rows)]

    def latest_candle(
        self, symbol: str, timeframe: Timeframe | str, *, provider: str | None = None
    ) -> Candle | None:
        found = self.load_candles(symbol, timeframe, limit=1, provider=provider)
        return found[-1] if found else None

    def candle_count(self, symbol: str, timeframe: Timeframe | str) -> int:
        tf = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe)
        stmt = (
            select(func.count())
            .select_from(m.Candle)
            .where(m.Candle.symbol == symbol, m.Candle.timeframe == tf)
        )
        with session_scope(self._factory) as session:
            return int(session.scalar(stmt) or 0)

    def prune_candles_before(self, cutoff_utc: datetime) -> int:
        """Delete candles older than ``cutoff_utc``. Returns rows removed.

        Retention is a deliberate, explicitly-invoked operation with a caller and
        a cutoff, not a side effect of writing. Nothing here prunes on insert.
        """
        with session_scope(self._factory) as session:
            # A DELETE yields a CursorResult, which carries rowcount; the
            # declared return type of Session.execute does not.
            result: Any = session.execute(
                delete(m.Candle).where(m.Candle.open_time_utc < cutoff_utc)
            )
            removed = int(result.rowcount or 0)
        log.info("candles pruned", extra={"removed": removed, "cutoff": cutoff_utc.isoformat()})
        return removed

    # ------------------------------------------------------------------ #
    # quotes                                                             #
    # ------------------------------------------------------------------ #

    def save_quote(self, quote: Quote) -> None:
        """Append one observed quote.

        ``as_of_utc`` takes the provider's timestamp. Substituting local time
        would make every stored quote look fresh, which defeats the staleness
        check that reads this column.
        """
        with session_scope(self._factory) as session:
            session.add(
                m.Quote(
                    created_at_utc=utc_now(),
                    provider=quote.provider,
                    symbol=quote.symbol,
                    asset_class=quote.asset_class.value,
                    as_of_utc=quote.provider_time_utc or quote.received_at_utc,
                    bid=quote.bid,
                    ask=quote.ask,
                    mid=quote.mid,
                    last=quote.last,
                    spread_absolute=quote.spread,
                    spread_basis_points=_spread_bps(quote),
                )
            )

    def latest_quote(self, symbol: str) -> Quote | None:
        stmt = (
            select(m.Quote)
            .where(m.Quote.symbol == symbol)
            .order_by(m.Quote.as_of_utc.desc())
            .limit(1)
        )
        with session_scope(self._factory) as session:
            row = session.scalars(stmt).first()
            if row is None:
                return None
            return Quote(
                provider=row.provider,
                symbol=row.symbol,
                asset_class=AssetClass(row.asset_class),
                bid=row.bid,
                ask=row.ask,
                mid=row.mid,
                last=row.last,
                spread=row.spread_absolute,
                provider_time_utc=row.as_of_utc,
            )

    # ------------------------------------------------------------------ #
    # symbols -- the allowlist                                           #
    # ------------------------------------------------------------------ #

    def upsert_symbols(self, symbols: Iterable[SymbolInfo]) -> int:
        """Refresh the symbol catalogue. Returns rows written.

        Symbols do change -- precision is revised, an instrument is delisted --
        so unlike candles this path updates on conflict. ``enabled_for_scanning``
        is deliberately absent from the update set: it is an operator decision,
        and a provider refresh must not re-enable something a human switched off.
        """
        batch = list(symbols)
        if not batch:
            return 0
        now = utc_now()
        written = 0
        with session_scope(self._factory) as session:
            for info in batch:
                existing = session.scalars(
                    select(m.Symbol).where(
                        m.Symbol.provider == info.provider, m.Symbol.symbol == info.symbol
                    )
                ).first()
                values = {
                    "asset_class": info.asset_class.value,
                    "base_asset": info.base_asset,
                    "quote_asset": info.quote_asset,
                    "price_precision": info.price_precision,
                    "min_quantity": info.min_quantity,
                    "is_active": bool(info.is_tradable),
                }
                if existing is None:
                    session.add(
                        m.Symbol(
                            created_at_utc=now,
                            provider=info.provider,
                            symbol=info.symbol,
                            enabled_for_scanning=False,
                            **values,
                        )
                    )
                else:
                    for key, value in values.items():
                        setattr(existing, key, value)
                written += 1
        log.info("symbols upserted", extra={"count": written})
        return written

    def is_symbol_allowed(self, symbol: str, *, provider: str | None = None) -> bool:
        """Whether ``symbol`` is a known, active instrument.

        The allowlist check the request validators call. An unknown symbol is
        rejected here rather than forwarded, which keeps an arbitrary caller
        string out of an outbound provider URL.
        """
        stmt = (
            select(func.count())
            .select_from(m.Symbol)
            .where(m.Symbol.symbol == symbol, m.Symbol.is_active.is_(True))
        )
        if provider:
            stmt = stmt.where(m.Symbol.provider == provider)
        with session_scope(self._factory) as session:
            return int(session.scalar(stmt) or 0) > 0

    def scannable_symbols(self) -> list[str]:
        stmt = (
            select(m.Symbol.symbol)
            .where(m.Symbol.is_active.is_(True), m.Symbol.enabled_for_scanning.is_(True))
            .distinct()
            .order_by(m.Symbol.symbol)
        )
        with session_scope(self._factory) as session:
            return [str(s) for s in session.scalars(stmt)]

    def set_symbol_scanning(
        self, symbol: str, *, enabled: bool, provider: str | None = None
    ) -> int:
        """Turn scanning on or off for ``symbol``. Returns rows affected.

        The only way ``enabled_for_scanning`` changes. :meth:`upsert_symbols`
        refuses to touch it precisely so that this decision has one owner and one
        entry point -- an operator, through here -- rather than being a side
        effect of whichever provider refreshed its catalogue most recently.

        Enabling an untradable symbol is permitted and does nothing on its own:
        :meth:`scannable_symbols` requires both flags, so a delisted instrument
        stays out of the scan regardless of what is set here.
        """
        stmt = select(m.Symbol).where(m.Symbol.symbol == symbol)
        if provider:
            stmt = stmt.where(m.Symbol.provider == provider)
        with session_scope(self._factory) as session:
            rows = list(session.scalars(stmt))
            for row in rows:
                row.enabled_for_scanning = enabled
            affected = len(rows)
        log.info(
            "symbol scanning changed",
            extra={"symbol": symbol, "enabled": enabled, "rows": affected},
        )
        return affected

    # ------------------------------------------------------------------ #
    # provider health                                                    #
    # ------------------------------------------------------------------ #

    def record_health(self, health: ProviderHealth) -> None:
        """Append one health probe result.

        ``message`` is stored as given. The redactor runs at the error boundary,
        so a secret that reaches this point was never redacted, and re-checking
        here would hide that the boundary is broken.
        """
        with session_scope(self._factory) as session:
            session.add(
                m.ProviderHealthRow(
                    created_at_utc=utc_now(),
                    provider=health.provider,
                    status=health.status.value,
                    checked_at_utc=health.checked_at_utc,
                    latency_ms=health.latency_ms,
                    credential_present=bool(health.credentials_present),
                    endpoint_reachable=health.status.value != "DOWN",
                    message=health.message,
                    details={
                        "kind": health.kind,
                        "enabled": health.enabled,
                        "capabilities": list(health.capabilities),
                        "missing_env": list(health.missing_env),
                        "limitations": list(health.limitations),
                        "circuit_state": health.circuit_state,
                    },
                )
            )

    def recent_health(self, provider: str, *, limit: int = 20) -> list[dict[str, Any]]:
        stmt = (
            select(m.ProviderHealthRow)
            .where(m.ProviderHealthRow.provider == provider)
            .order_by(m.ProviderHealthRow.checked_at_utc.desc())
            .limit(max(1, limit))
        )
        with session_scope(self._factory) as session:
            return [
                {
                    "provider": r.provider,
                    "status": r.status,
                    "checked_at_utc": r.checked_at_utc,
                    "latency_ms": float(r.latency_ms) if r.latency_ms is not None else None,
                    "credential_present": r.credential_present,
                    "message": r.message,
                }
                for r in session.scalars(stmt)
            ]

    # ------------------------------------------------------------------ #
    # provider quota -- the counter that must survive a restart          #
    # ------------------------------------------------------------------ #

    def consume_quota(
        self,
        provider: str,
        *,
        window_kind: str = "day",
        limit: int | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Claim one request against a persisted budget.

        Returns ``{"allowed", "requests_made", "requests_allowed", "window_start_utc"}``.
        ``allowed`` is false when the budget is spent; the caller degrades rather
        than being handed an exception it has to translate back into a decision.

        The counter is incremented *before* the request is made, not after. A
        request that fails still consumed the provider's quota, so counting only
        successes lets a run of failures overspend the budget invisibly. The
        pessimistic direction is the safe one: at worst the last request of the
        day goes unused.
        """
        stamp = now or utc_now()
        start = _window_start(stamp, window_kind)
        with session_scope(self._factory) as session:
            stmt = select(m.ProviderQuota).where(
                m.ProviderQuota.provider == provider,
                m.ProviderQuota.window_kind == window_kind,
                m.ProviderQuota.window_start_utc == start,
            )
            if session.get_bind().dialect.name == "postgresql":
                # Two workers claiming the last request of the day must not both
                # win. SQLite serialises writers already, so the lock is only
                # needed -- and only available -- on Postgres.
                stmt = stmt.with_for_update()
            row = session.scalars(stmt).first()

            if row is None:
                row = m.ProviderQuota(
                    created_at_utc=stamp,
                    provider=provider,
                    window_kind=window_kind,
                    window_start_utc=start,
                    requests_made=0,
                    requests_allowed=limit,
                )
                session.add(row)
                session.flush()

            if limit is not None and row.requests_allowed != limit:
                # The configured cap changed. Record the new one rather than
                # enforcing a stale number nobody can find in the config.
                row.requests_allowed = limit

            cap = row.requests_allowed
            if cap is not None and row.requests_made >= cap:
                return {
                    "allowed": False,
                    "requests_made": row.requests_made,
                    "requests_allowed": cap,
                    "window_start_utc": start,
                }

            row.requests_made += 1
            row.last_request_utc = stamp
            return {
                "allowed": True,
                "requests_made": row.requests_made,
                "requests_allowed": cap,
                "window_start_utc": start,
            }

    def quota_state(
        self, provider: str, *, window_kind: str = "day", now: datetime | None = None
    ) -> dict[str, Any] | None:
        """Read the current window without consuming from it."""
        start = _window_start(now or utc_now(), window_kind)
        with session_scope(self._factory) as session:
            row = session.scalars(
                select(m.ProviderQuota).where(
                    m.ProviderQuota.provider == provider,
                    m.ProviderQuota.window_kind == window_kind,
                    m.ProviderQuota.window_start_utc == start,
                )
            ).first()
            if row is None:
                return None
            cap = row.requests_allowed
            remaining = None if cap is None else max(0, cap - row.requests_made)
            return {
                "provider": row.provider,
                "window_kind": row.window_kind,
                "window_start_utc": row.window_start_utc,
                "requests_made": row.requests_made,
                "requests_allowed": row.requests_allowed,
                "remaining": remaining,
                "last_request_utc": row.last_request_utc,
            }

    # ------------------------------------------------------------------ #
    # analysis outputs                                                   #
    # ------------------------------------------------------------------ #

    def save_quality_report(self, report: DataQualityReport) -> None:
        """Persist one quality assessment."""
        with session_scope(self._factory) as session:
            session.add(
                m.DataQualityReportRow(
                    created_at_utc=utc_now(),
                    provider=report.provider,
                    symbol=report.symbol,
                    # Stored as NULL, not as the string "unknown": a feed-wide
                    # report has no timeframe, and a sentinel would be
                    # indistinguishable from a real one in later queries.
                    timeframe=report.timeframe.value if report.timeframe else None,
                    assessed_at_utc=report.checked_at_utc,
                    status=report.status.value,
                    quality_score=float(report.quality_score),
                    is_blocking=report.is_blocking,
                    bars_checked=report.candles_checked,
                    checks=list(report.checks_run),
                    failures=list(report.blocking_reasons),
                    warnings=list(report.warnings),
                )
            )

    def save_feature_snapshot(self, snapshot: FeatureSnapshot) -> None:
        with session_scope(self._factory) as session:
            _insert_ignore(
                session,
                m.FeatureSnapshotRow,
                [
                    {
                        "created_at_utc": utc_now(),
                        "provider": snapshot.provider,
                        "symbol": snapshot.symbol,
                        "timeframe": snapshot.timeframe.value,
                        "as_of_utc": snapshot.as_of_candle_close_utc,
                        "bars_used": snapshot.bars_used,
                        "indicator_version": "indicators-1.0.0",
                        "features": snapshot.model_dump(exclude_none=True),
                        "unavailable": list(snapshot.missing_features),
                    }
                ],
                index=["symbol", "timeframe", "as_of_utc"],
            )

    def save_structure_report(self, report: StructureReport) -> None:
        with session_scope(self._factory) as session:
            _insert_ignore(
                session,
                m.StructureReportRow,
                [
                    {
                        "created_at_utc": utc_now(),
                        "provider": "unknown",
                        "symbol": report.symbol,
                        "timeframe": report.timeframe.value,
                        "as_of_utc": utc_now(),
                        "bars_used": 0,
                        "structure_version": "structure-1.0.0",
                        "structure_label": report.structure,
                        "breakout_candidate": report.breakout_candidate,
                        "breakout_direction": (
                            report.breakout_direction.value if report.breakout_direction else None
                        ),
                        "failed_breakout": report.failed_breakout,
                        "nearest_support": report.nearest_support,
                        "nearest_resistance": report.nearest_resistance,
                        "swing_highs": report.swing_highs,
                        "swing_lows": report.swing_lows,
                        "notes": report.notes,
                    }
                ],
                index=["symbol", "timeframe", "as_of_utc"],
            )

    def save_regime_report(self, report: RegimeReport) -> None:
        with session_scope(self._factory) as session:
            _insert_ignore(
                session,
                m.RegimeReportRow,
                [
                    {
                        "created_at_utc": utc_now(),
                        "provider": "unknown",
                        "symbol": report.symbol,
                        "timeframe": report.timeframe.value if report.timeframe else "unknown",
                        "as_of_utc": report.computed_at_utc,
                        "regime_version": report.version,
                        "regime": report.regime.value,
                        "heuristic_score": float(report.heuristic_score),
                        "supporting_evidence": list(report.supporting_evidence),
                        "contradictions": list(report.contradictions),
                    }
                ],
                index=["symbol", "timeframe", "as_of_utc"],
            )

    # ------------------------------------------------------------------ #
    # external events                                                    #
    # ------------------------------------------------------------------ #

    def save_economic_events(self, events: Iterable[EconomicEvent]) -> int:
        batch = list(events)
        if not batch:
            return 0
        rows = [
            {
                "created_at_utc": utc_now(),
                "provider": e.provider,
                "event_id_from_provider": e.event_id,
                "title": e.title,
                "country": e.country,
                "currency": e.currency,
                "impact": e.impact.value,
                "scheduled_utc": e.scheduled_utc,
                "actual": e.actual,
                "forecast": e.forecast,
                "previous": e.previous,
                "retrieved_at_utc": e.retrieved_at_utc,
            }
            for e in batch
        ]
        with session_scope(self._factory) as session:
            result = _insert_ignore(
                session,
                m.EconomicEventRow,
                rows,
                index=["provider", "event_id_from_provider", "scheduled_utc"],
            )
            return int(result.rowcount or 0)

    def upcoming_events(
        self, *, currency: str | None = None, from_utc: datetime | None = None, limit: int = 50
    ) -> list[EconomicEvent]:
        stmt = (
            select(m.EconomicEventRow)
            .where(m.EconomicEventRow.scheduled_utc >= (from_utc or utc_now()))
            .order_by(m.EconomicEventRow.scheduled_utc)
            .limit(max(1, limit))
        )
        if currency:
            stmt = stmt.where(m.EconomicEventRow.currency == currency)
        with session_scope(self._factory) as session:
            return [_event_from_row(r) for r in session.scalars(stmt)]

    def save_news(self, items: Iterable[NewsItem]) -> int:
        batch = list(items)
        if not batch:
            return 0
        rows = [
            {
                "created_at_utc": utc_now(),
                "provider": n.provider,
                "headline": n.headline,
                "summary": n.summary,
                "url": n.url,
                "source": n.source,
                "symbols": list(n.symbols),
                "published_at_utc": n.published_at_utc,
                "retrieved_at_utc": n.retrieved_at_utc,
            }
            for n in batch
        ]
        with session_scope(self._factory) as session:
            result = _insert_ignore(
                session,
                m.NewsItemRow,
                rows,
                index=["provider", "headline", "published_at_utc"],
            )
            return int(result.rowcount or 0)

    def recent_news(self, *, symbol: str | None = None, limit: int = 50) -> list[NewsItem]:
        stmt = (
            select(m.NewsItemRow)
            .order_by(m.NewsItemRow.published_at_utc.desc())
            .limit(max(1, limit))
        )
        if symbol:
            stmt = stmt.where(m.NewsItemRow.symbols.contains([symbol]))
        with session_scope(self._factory) as session:
            rows = list(session.scalars(stmt))
        return [_news_from_row(r) for r in rows]

    # ------------------------------------------------------------------ #
    # scanner outputs and settled signals                                #
    # ------------------------------------------------------------------ #

    def save_scan_result(self, scan_id: str, result: Any, runtime_ms: float | None = None) -> None:
        """Persist one scan cycle's summary."""
        with session_scope(self._factory) as session:
            session.add(
                m.ScanResultRow(
                    created_at_utc=utc_now(),
                    scan_id=scan_id,
                    scanned_at_utc=result.generated_at_utc,
                    horizon=result.horizon,
                    symbols_scanned=len(result.scanned),
                    candidates_found=len(result.candidates),
                    rejections_logged=len(result.rejections),
                    runtime_ms=runtime_ms,
                    extra={
                        "warnings": result.warnings,
                        "scanner_version": result.scanner_version,
                    },
                )
            )

    def save_signal(self, signal_id: str, scan_id: str, llm_response: Any, context: Any) -> None:
        """Persist one LLM-approved signal."""
        with session_scope(self._factory) as session:
            session.add(
                m.SignalRow(
                    created_at_utc=utc_now(),
                    scan_id=scan_id,
                    signal_id=signal_id,
                    signal_time_utc=llm_response.generated_at_utc,
                    symbol=llm_response.asset,
                    horizon=llm_response.horizon,
                    direction=llm_response.direction.value,
                    status="CANDIDATE",
                    llm_response=llm_response.model_dump(),
                    context_snapshot=context.model_dump() if context else None,
                    version="signal-1.0.0",
                )
            )

    def settle_signal(self, settled: SettledSignal) -> None:
        """Write one closed signal. Immutable once stored.

        The four scoring fields are optional on the contract -- an unsettled
        signal legitimately has none of them -- and required here. A settlement
        row missing its direction or either price cannot be scored, and filling
        the gap with a placeholder would put a price in the performance history
        that no provider ever quoted (Rule 2). So the write is refused instead,
        naming the fields, and the caller learns it settled nothing.
        """
        missing = [
            name
            for name, value in (
                ("direction", settled.direction),
                ("reference_price", settled.reference_price),
                ("settlement_price", settled.settlement_price),
                ("expiry_utc", settled.expiry_utc),
            )
            if value is None
        ]
        if missing:
            raise PersistenceError(
                f"cannot settle a signal without its scoring fields; missing: {', '.join(missing)}",
                details={"signal_id": settled.signal_id, "missing": missing},
            )
        assert settled.direction is not None  # narrowed by the check above

        try:
            with session_scope(self._factory) as session:
                session.add(
                    m.SettledSignalRow(
                        created_at_utc=utc_now(),
                        signal_id=settled.signal_id,
                        symbol=settled.symbol,
                        horizon=settled.horizon,
                        direction=settled.direction.value,
                        reference_price=settled.reference_price,
                        settlement_price=settled.settlement_price,
                        outcome=settled.outcome.value,
                        expiry_utc=settled.expiry_utc,
                        settled_at_utc=settled.settled_at_utc,
                        settlement_provider=settled.settlement_provider,
                        notes="\n".join(settled.notes) if settled.notes else None,
                    )
                )
        except IntegrityError as exc:
            # The unique constraint on signal_id is the immutability guarantee
            # doing its job -- a second settlement of the same signal would be a
            # re-scoring of a closed trade. Translated rather than propagated so
            # callers handle one persistence error type, not SQLAlchemy's.
            raise PersistenceError(
                "settled_signals is immutable; a signal with this ID already exists",
                details={"signal_id": settled.signal_id},
            ) from exc

    def unsettled_expired_signals(self, *, limit: int = 100) -> list[AIDecision]:
        """Signals whose expiry has passed and which have no settlement row yet.

        The anti-join against ``settled_signals`` is what keeps the settlement
        worker idempotent: a signal already scored is not a candidate, so a
        restarted worker cannot re-score a closed trade.

        A row whose stored JSON has no ``expiry_utc`` is skipped rather than
        assigned one. Settling against an invented expiry would score the trade
        over a window the signal never specified.
        """
        settled = select(m.SettledSignalRow.signal_id)
        stmt = (
            select(m.SignalRow)
            .where(m.SignalRow.signal_id.not_in(settled))
            .order_by(m.SignalRow.signal_time_utc.asc())
            .limit(max(1, limit))
        )
        with session_scope(self._factory) as session:
            rows = list(session.scalars(stmt))

        now = utc_now()
        due: list[AIDecision] = []
        for row in rows:
            payload = row.llm_response or {}
            expiry = payload.get("expiry_utc")
            if not expiry:
                continue
            parsed = expiry if isinstance(expiry, datetime) else _parse_utc(str(expiry))
            if parsed is None or parsed > now:
                continue
            raw_price = payload.get("reference_price")
            price: Decimal | None = None
            if raw_price is not None:
                with contextlib.suppress(InvalidOperation, ValueError):
                    price = Decimal(str(raw_price))
            due.append(
                AIDecision(
                    decision_id=row.signal_id,
                    symbol=row.symbol,
                    horizon=row.horizon,
                    status=SignalStatus.SIGNAL,
                    direction=SignalDirection(row.direction),
                    reference_price=price,
                    expiry_utc=parsed,
                )
            )
        return due

    def settled_signals(
        self, *, symbol: str | None = None, horizon: str | None = None, limit: int = 100
    ) -> list[SettledSignal]:
        stmt = (
            select(m.SettledSignalRow)
            .order_by(m.SettledSignalRow.settled_at_utc.desc())
            .limit(max(1, limit))
        )
        if symbol:
            stmt = stmt.where(m.SettledSignalRow.symbol == symbol)
        if horizon:
            stmt = stmt.where(m.SettledSignalRow.horizon == horizon)
        with session_scope(self._factory) as session:
            rows = list(session.scalars(stmt))
        return [_settled_from_row(r) for r in rows]

    # ------------------------------------------------------------------ #
    # audit log -- append-only                                           #
    # ------------------------------------------------------------------ #

    def log_event(
        self,
        event_type: str,
        *,
        actor: str | None = None,
        symbol: str | None = None,
        provider: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append one event. Message is stored as given; redaction happens upstream."""
        with session_scope(self._factory) as session:
            session.add(
                m.AuditLog(
                    created_at_utc=utc_now(),
                    event_time_utc=utc_now(),
                    event_type=event_type,
                    actor=actor,
                    symbol=symbol,
                    provider=provider,
                    message=message,
                    details=details,
                )
            )

    def recent_events(
        self, *, event_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        stmt = select(m.AuditLog).order_by(m.AuditLog.event_time_utc.desc()).limit(max(1, limit))
        if event_type:
            stmt = stmt.where(m.AuditLog.event_type == event_type)
        with session_scope(self._factory) as session:
            return [
                {
                    "event_time_utc": r.event_time_utc,
                    "event_type": r.event_type,
                    "actor": r.actor,
                    "symbol": r.symbol,
                    "provider": r.provider,
                    "message": r.message,
                    "details": r.details,
                }
                for r in session.scalars(stmt)
            ]

    def save_trade_memory(self, memory: TradeMemory) -> None:
        """Persist one post-mortem trade memory."""
        with session_scope(self._factory) as session:
            session.add(
                m.TradeMemoryRow(
                    created_at_utc=utc_now(),
                    memory_id=memory.memory_id,
                    signal_id=memory.signal_id,
                    symbol=memory.symbol,
                    asset_class=memory.asset_class.value
                    if hasattr(memory.asset_class, "value")
                    else str(memory.asset_class),
                    horizon=memory.horizon,
                    regime=memory.regime.value
                    if hasattr(memory.regime, "value")
                    else str(memory.regime),
                    pattern=memory.pattern,
                    outcome=memory.outcome.value
                    if hasattr(memory.outcome, "value")
                    else str(memory.outcome),
                    reference_price=memory.reference_price,
                    exit_price=memory.exit_price,
                    root_cause=memory.root_cause,
                    key_lessons=list(memory.key_lessons),
                    do_rules=list(memory.do_rules),
                    dont_rules=list(memory.dont_rules),
                    user_notes=memory.user_notes,
                )
            )

    def list_trade_memories(
        self,
        *,
        symbol: str | None = None,
        regime: str | None = None,
        outcome: str | None = None,
        limit: int = 100,
    ) -> list[TradeMemory]:
        """Query trade memories by symbol, regime, or outcome."""
        stmt = (
            select(m.TradeMemoryRow)
            .order_by(m.TradeMemoryRow.created_at_utc.desc())
            .limit(max(1, limit))
        )
        if symbol:
            stmt = stmt.where(m.TradeMemoryRow.symbol == symbol)
        if regime:
            stmt = stmt.where(m.TradeMemoryRow.regime == regime)
        if outcome:
            stmt = stmt.where(m.TradeMemoryRow.outcome == outcome)

        with session_scope(self._factory) as session:
            rows = list(session.scalars(stmt))
        return [_memory_from_row(r) for r in rows]


def _event_from_row(row: m.EconomicEventRow) -> EconomicEvent:
    return EconomicEvent(
        provider=row.provider,
        event_id=row.event_id_from_provider,
        title=row.title,
        country=row.country,
        currency=row.currency,
        impact=row.impact,
        scheduled_utc=row.scheduled_utc,
        actual=row.actual,
        forecast=row.forecast,
        previous=row.previous,
        retrieved_at_utc=row.retrieved_at_utc,
    )


def _news_from_row(row: m.NewsItemRow) -> NewsItem:
    return NewsItem(
        provider=row.provider,
        headline=row.headline,
        summary=row.summary,
        url=row.url,
        source=row.source,
        symbols=row.symbols or [],
        published_at_utc=row.published_at_utc,
        retrieved_at_utc=row.retrieved_at_utc,
    )


def _settled_from_row(row: m.SettledSignalRow) -> SettledSignal:
    return SettledSignal(
        signal_id=row.signal_id,
        symbol=row.symbol,
        horizon=row.horizon,
        direction=SignalDirection(row.direction),
        reference_price=row.reference_price,
        settlement_price=row.settlement_price,
        outcome=SettlementOutcome(row.outcome),
        expiry_utc=row.expiry_utc,
        settled_at_utc=row.settled_at_utc,
        settlement_provider=row.settlement_provider,
        notes=row.notes.split("\n") if row.notes else [],
    )


def _memory_from_row(row: m.TradeMemoryRow) -> TradeMemory:
    return TradeMemory(
        memory_id=row.memory_id,
        signal_id=row.signal_id,
        symbol=row.symbol,
        asset_class=AssetClass(row.asset_class) if row.asset_class else AssetClass.CRYPTO,
        horizon=row.horizon,
        regime=MarketRegime(row.regime) if row.regime else MarketRegime.UNCERTAIN,
        pattern=row.pattern,
        outcome=SettlementOutcome(row.outcome),
        reference_price=row.reference_price,
        exit_price=row.exit_price,
        root_cause=row.root_cause,
        key_lessons=row.key_lessons or [],
        do_rules=row.do_rules or [],
        dont_rules=row.dont_rules or [],
        user_notes=row.user_notes,
        created_at_utc=row.created_at_utc,
    )
