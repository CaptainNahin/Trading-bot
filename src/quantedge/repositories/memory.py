"""In-memory fallback repository when persistence is unavailable.

Every method signature matches :class:`SqlRepository` so switching between them
is a configuration change rather than a code change. The in-memory version is a
deliberate fallback, not a toy: it lets the scanner run and the MCP server
answer queries even when no database is configured.

The cost is honest: nothing survives a restart, and the memory footprint is
unbounded. A caller that persists must check ``persistence_available`` before
claiming durability, and a long-running process that uses this should shed old
data periodically or accept that it will eventually exhaust memory.

Immutability guards
-------------------
``audit_logs`` is append-only and ``settled_signals`` is immutable here too.
The enforcement is a lock rather than an event listener -- there is no ORM --
but the guarantee is the same as the SQL path, and callers that read
"append-only" in the models docstring should find it append-only everywhere.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from quantedge.contracts import (
    Candle,
    DataQualityReport,
    EconomicEvent,
    FeatureSnapshot,
    NewsItem,
    ProviderHealth,
    Quote,
    RegimeReport,
    SettledSignal,
    StructureReport,
    SymbolInfo,
    Timeframe,
    utc_now,
)
from quantedge.errors import PersistenceError
from quantedge.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["MemoryRepository"]

log = get_logger(__name__)


class MemoryRepository:
    """In-memory contract store with no durability guarantee.

    All data lives in dicts and lists inside this object. When the process
    exits, everything is lost. Methods match :class:`SqlRepository` so callers
    can switch by configuration.
    """

    def __init__(self) -> None:
        # keyed (provider, symbol, timeframe); open_time uniqueness is checked
        # within the bucket rather than being part of the key, because
        # load_candles has to scan a whole series at once.
        self._candles: dict[tuple[str, str, str], list[Candle]] = defaultdict(list)
        self._quotes: dict[str, list[Quote]] = defaultdict(list)
        self._symbols: dict[tuple[str, str], SymbolInfo] = {}
        # Scanning is an operator flag held apart from the catalogue, so that
        # re-upserting a symbol cannot silently change it.
        self._scanning: set[str] = set()
        self._health: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._quota: dict[tuple[str, str, datetime], dict[str, Any]] = {}
        self._quality_reports: list[DataQualityReport] = []
        self._feature_snapshots: list[FeatureSnapshot] = []
        self._structure_reports: list[StructureReport] = []
        self._regime_reports: list[RegimeReport] = []
        self._economic_events: list[EconomicEvent] = []
        self._news: list[NewsItem] = []
        self._scan_results: list[dict[str, Any]] = []
        self._signals: list[dict[str, Any]] = []
        self._settled: list[SettledSignal] = []
        self._audit_log: list[dict[str, Any]] = []
        self._lock = threading.RLock()

        log.warning(
            "in-memory repository active: no data survives a restart",
            extra={"persistence_available": False},
        )

    @property
    def persistence_available(self) -> bool:
        return False

    # ------------------------------------------------------------------ #
    # candles                                                            #
    # ------------------------------------------------------------------ #

    def save_candles(self, candles: Iterable[Candle]) -> int:
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

        with self._lock:
            inserted = 0
            touched: set[tuple[str, str, str]] = set()
            for c in batch:
                bucket = (c.provider, c.symbol, c.timeframe.value)
                store = self._candles[bucket]
                # First observation wins, matching the SQL path's ON CONFLICT DO
                # NOTHING: a provider restating a closed bar is reporting a
                # correction, and overwriting would erase the fact that it changed.
                if not any(existing.open_time_utc == c.open_time_utc for existing in store):
                    store.append(c)
                    inserted += 1
                    touched.add(bucket)
            # Only the buckets that grew need reordering; load_candles relies on
            # each series being oldest-first.
            for bucket in touched:
                self._candles[bucket].sort(key=lambda c: c.open_time_utc)
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
        tf = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe)
        with self._lock:
            candidates = []
            for (p, s, t), bars in self._candles.items():
                if s != symbol or t != tf:
                    continue
                if provider and p != provider:
                    continue
                candidates.extend(bars)
            if since:
                candidates = [c for c in candidates if c.open_time_utc >= since]
            candidates.sort(key=lambda c: c.open_time_utc)
            return candidates[-limit:]

    def latest_candle(
        self, symbol: str, timeframe: Timeframe | str, *, provider: str | None = None
    ) -> Candle | None:
        found = self.load_candles(symbol, timeframe, limit=1, provider=provider)
        return found[-1] if found else None

    def candle_count(self, symbol: str, timeframe: Timeframe | str) -> int:
        tf = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe)
        with self._lock:
            total = 0
            for (_, s, t), bars in self._candles.items():
                if s == symbol and t == tf:
                    total += len(bars)
            return total

    def prune_candles_before(self, cutoff_utc: datetime) -> int:
        with self._lock:
            removed = 0
            for key in list(self._candles.keys()):
                before_count = len(self._candles[key])
                kept = [c for c in self._candles[key] if c.open_time_utc >= cutoff_utc]
                self._candles[key] = kept
                removed += before_count - len(kept)
                if not self._candles[key]:
                    del self._candles[key]
        return removed

    # ------------------------------------------------------------------ #
    # quotes                                                             #
    # ------------------------------------------------------------------ #

    def save_quote(self, quote: Quote) -> None:
        with self._lock:
            self._quotes[quote.symbol].append(quote)

    def latest_quote(self, symbol: str) -> Quote | None:
        with self._lock:
            quotes = self._quotes.get(symbol, [])
            if not quotes:
                return None
            return max(quotes, key=lambda q: q.provider_time_utc or q.received_at_utc)

    # ------------------------------------------------------------------ #
    # symbols                                                            #
    # ------------------------------------------------------------------ #

    def upsert_symbols(self, symbols: Iterable[SymbolInfo]) -> int:
        # Materialised before the loop, not counted after it: a generator is
        # exhausted by the loop, so counting afterwards reports zero written
        # while the rows are sitting in the dict.
        batch = list(symbols)
        with self._lock:
            for info in batch:
                self._symbols[(info.provider, info.symbol)] = info
        return len(batch)

    def is_symbol_allowed(self, symbol: str, *, provider: str | None = None) -> bool:
        with self._lock:
            for (p, s), info in self._symbols.items():
                if s == symbol and (provider is None or p == provider) and info.is_tradable:
                    return True
            return False

    def scannable_symbols(self) -> list[str]:
        """Symbols that are both tradable and switched on for scanning.

        Both flags, matching the SQL path. Returning every tradable symbol here
        would make the memory backend scan instruments the operator had disabled,
        which is exactly the divergence the parity section exists to catch.
        """
        with self._lock:
            unique = {
                s
                for (_, s), info in self._symbols.items()
                if info.is_tradable and s in self._scanning
            }
            return sorted(unique)

    def set_symbol_scanning(
        self, symbol: str, *, enabled: bool, provider: str | None = None
    ) -> int:
        """Turn scanning on or off for ``symbol``. Returns symbols affected.

        Kept out of :meth:`upsert_symbols` for the same reason as the SQL path: a
        provider refresh must not re-enable something a human switched off.
        """
        with self._lock:
            matches = [
                s for (p, s) in self._symbols if s == symbol and (provider is None or p == provider)
            ]
            if not matches:
                return 0
            if enabled:
                self._scanning.add(symbol)
            else:
                self._scanning.discard(symbol)
            return len(matches)

    # ------------------------------------------------------------------ #
    # provider health                                                    #
    # ------------------------------------------------------------------ #

    def record_health(self, health: ProviderHealth) -> None:
        with self._lock:
            self._health[health.provider].append(
                {
                    "provider": health.provider,
                    "status": health.status.value,
                    "checked_at_utc": health.checked_at_utc,
                    "latency_ms": health.latency_ms,
                    "credential_present": bool(health.credentials_present),
                    "message": health.message,
                }
            )

    def recent_health(self, provider: str, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            history = self._health.get(provider, [])
            return sorted(history, key=lambda r: r["checked_at_utc"], reverse=True)[:limit]

    # ------------------------------------------------------------------ #
    # provider quota                                                     #
    # ------------------------------------------------------------------ #

    def consume_quota(
        self,
        provider: str,
        *,
        window_kind: str = "day",
        limit: int | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        from quantedge.repositories.sql import _window_start

        stamp = now or utc_now()
        start = _window_start(stamp, window_kind)
        key = (provider, window_kind, start)

        with self._lock:
            if key not in self._quota:
                self._quota[key] = {
                    "requests_made": 0,
                    "requests_allowed": limit,
                    "window_start_utc": start,
                }
            state = self._quota[key]
            if limit is not None:
                state["requests_allowed"] = limit

            cap = state["requests_allowed"]
            if cap is not None and state["requests_made"] >= cap:
                return {
                    "allowed": False,
                    "requests_made": state["requests_made"],
                    "requests_allowed": cap,
                    "window_start_utc": start,
                }

            state["requests_made"] += 1
            return {
                "allowed": True,
                "requests_made": state["requests_made"],
                "requests_allowed": cap,
                "window_start_utc": start,
            }

    def quota_state(
        self, provider: str, *, window_kind: str = "day", now: datetime | None = None
    ) -> dict[str, Any] | None:
        from quantedge.repositories.sql import _window_start

        start = _window_start(now or utc_now(), window_kind)
        key = (provider, window_kind, start)
        with self._lock:
            state = self._quota.get(key)
            if state is None:
                return None
            remaining = (
                None
                if state["requests_allowed"] is None
                else max(0, state["requests_allowed"] - state["requests_made"])
            )
            return {
                "provider": provider,
                "window_kind": window_kind,
                "window_start_utc": start,
                "requests_made": state["requests_made"],
                "requests_allowed": state["requests_allowed"],
                "remaining": remaining,
                "last_request_utc": None,
            }

    # ------------------------------------------------------------------ #
    # analysis outputs                                                   #
    # ------------------------------------------------------------------ #

    def save_quality_report(self, report: DataQualityReport) -> None:
        with self._lock:
            self._quality_reports.append(report)

    def save_feature_snapshot(self, snapshot: FeatureSnapshot) -> None:
        with self._lock:
            self._feature_snapshots.append(snapshot)

    def save_structure_report(self, report: StructureReport) -> None:
        with self._lock:
            self._structure_reports.append(report)

    def save_regime_report(self, report: RegimeReport) -> None:
        with self._lock:
            self._regime_reports.append(report)

    # ------------------------------------------------------------------ #
    # external events                                                    #
    # ------------------------------------------------------------------ #

    def save_economic_events(self, events: Iterable[EconomicEvent]) -> int:
        batch = list(events)
        with self._lock:
            self._economic_events.extend(batch)
        return len(batch)

    def upcoming_events(
        self, *, currency: str | None = None, from_utc: datetime | None = None, limit: int = 50
    ) -> list[EconomicEvent]:
        cutoff = from_utc or utc_now()
        with self._lock:
            candidates = [e for e in self._economic_events if e.scheduled_utc >= cutoff]
            if currency:
                candidates = [e for e in candidates if e.currency == currency]
            candidates.sort(key=lambda e: e.scheduled_utc)
            return candidates[:limit]

    def save_news(self, items: Iterable[NewsItem]) -> int:
        batch = list(items)
        with self._lock:
            self._news.extend(batch)
        return len(batch)

    def recent_news(self, *, symbol: str | None = None, limit: int = 50) -> list[NewsItem]:
        with self._lock:
            candidates = list(self._news)
            if symbol:
                candidates = [n for n in candidates if symbol in n.symbols]
            candidates.sort(key=lambda n: n.published_at_utc, reverse=True)
            return candidates[:limit]

    # ------------------------------------------------------------------ #
    # scanner outputs and settled signals                                #
    # ------------------------------------------------------------------ #

    def save_scan_result(self, scan_id: str, result: Any, runtime_ms: float | None = None) -> None:
        with self._lock:
            self._scan_results.append(
                {
                    "scan_id": scan_id,
                    "result": result,
                    "runtime_ms": runtime_ms,
                }
            )

    def save_signal(self, signal_id: str, scan_id: str, llm_response: Any, context: Any) -> None:
        with self._lock:
            self._signals.append(
                {
                    "signal_id": signal_id,
                    "scan_id": scan_id,
                    "llm_response": llm_response,
                    "context": context,
                }
            )

    def settle_signal(self, settled: SettledSignal) -> None:
        """Store one closed signal. Immutable once stored.

        The scoring fields are checked here even though this backend could hold
        a partial contract quite happily. If the two backends disagreed about
        what a valid settlement is, the same call would succeed on SQLite and
        fail on Postgres, and the difference would surface as a bug in whichever
        one was configured second.
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

        with self._lock:
            if any(s.signal_id == settled.signal_id for s in self._settled):
                raise PersistenceError(
                    "settled_signals is immutable; a signal with this ID already exists",
                    details={"signal_id": settled.signal_id},
                )
            self._settled.append(settled)

    def settled_signals(
        self, *, symbol: str | None = None, horizon: str | None = None, limit: int = 100
    ) -> list[SettledSignal]:
        with self._lock:
            candidates = list(self._settled)
            if symbol:
                candidates = [s for s in candidates if s.symbol == symbol]
            if horizon:
                candidates = [s for s in candidates if s.horizon == horizon]
            candidates.sort(key=lambda s: s.settled_at_utc, reverse=True)
            return candidates[:limit]

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
        with self._lock:
            self._audit_log.append(
                {
                    "event_time_utc": utc_now(),
                    "event_type": event_type,
                    "actor": actor,
                    "symbol": symbol,
                    "provider": provider,
                    "message": message,
                    "details": details,
                }
            )

    def recent_events(
        self, *, event_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._lock:
            candidates = list(self._audit_log)
            if event_type:
                candidates = [e for e in candidates if e["event_type"] == event_type]
            candidates.sort(key=lambda e: e["event_time_utc"], reverse=True)
            return candidates[:limit]
