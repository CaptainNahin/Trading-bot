"""Background live-market collector.

Consumes the Binance public WebSocket stream and maintains an in-process cache
of live market state, so that scanner and MCP requests can be answered without
a REST round trip per call.

The ten collector obligations from the specification, and where each is met:

============================================  ==========================================
Requirement                                   Implementation
============================================  ==========================================
1. reconnect automatically                    ``BinanceStreamClient._consume`` loop
2. exponential backoff                        ``BinanceStreamClient._backoff`` (jittered)
3. record last message time                   ``StreamStats.last_message_monotonic``
4. detect stale connections                   ``recv`` timeout -> forced reconnect;
                                              ``CollectorHealth.is_stale``
5. reject duplicate events                    ``_DedupeWindow`` keyed per bar/updateId
6. preserve provider event timestamps         ``mapping`` copies ``t``/``T``/``E`` as sent
7. normalize symbols                          ``normalize_symbol`` before subscribing and
                                              on every event
8. store completed candles                    ``_CandleBuffer.append_closed`` + optional
                                              durable ``PersistenceProvider``
9. store incomplete candles separately        ``_CandleBuffer.forming`` -- a distinct slot,
                                              never inside the closed deque
10. never treat incomplete as historical       ``closed_candles()`` reads only the deque;
                                              the forming bar is reachable only through
                                              ``forming_candle()``, which names itself
============================================  ==========================================

Rule 9 is the load-bearing one. A forming candle is a *partial* observation: its
high, low and close will all still change. Feeding it to an indicator produces a
number that silently rewrites itself on the next tick, which is how backtests
come to disagree with live trading. Here the forming bar is physically stored in
a different attribute from the history, so using it by accident requires calling
a method with "forming" in its name.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from quantedge.config import get_settings
from quantedge.contracts import (
    AssetClass,
    Candle,
    CandleSeries,
    HealthStatus,
    Quote,
    Timeframe,
    utc_now,
)
from quantedge.logging import get_logger
from quantedge.providers.binance.ws import BinanceStreamClient, StreamEvent
from quantedge.symbols import normalize_symbol

if TYPE_CHECKING:
    from quantedge.providers.base import PersistenceProvider

__all__ = ["CollectorHealth", "MarketStreamCollector", "get_collector"]

log = get_logger(__name__)

# Enough closed bars for the longest warm-up (200-period EMA + buffer) without
# unbounded memory: 600 bars x ~40 series is a few MB.
_MAX_BARS_PER_SERIES = 600
_PERSIST_BATCH = 50


class _CandleBuffer:
    """Closed history and the single forming bar, held apart on purpose."""

    __slots__ = ("_closed", "forming", "last_closed_at")

    def __init__(self, maxlen: int = _MAX_BARS_PER_SERIES) -> None:
        self._closed: deque[Candle] = deque(maxlen=maxlen)
        self.forming: Candle | None = None
        self.last_closed_at: float | None = None

    def append_closed(self, candle: Candle) -> bool:
        """Add a closed bar. Returns ``False`` if it was a duplicate or stale.

        Late-arriving bars are not merged into the middle of history; a series
        with a hole is detectable, whereas one silently back-filled out of order
        is not.
        """
        if not candle.is_closed:
            raise ValueError("append_closed refuses a forming candle")
        if self._closed:
            newest = self._closed[-1]
            if candle.open_time_utc <= newest.open_time_utc:
                return False
        self._closed.append(candle)
        self.last_closed_at = time.monotonic()
        # The bar that just closed is no longer "forming".
        if self.forming is not None and self.forming.open_time_utc <= candle.open_time_utc:
            self.forming = None
        return True

    def set_forming(self, candle: Candle) -> None:
        if candle.is_closed:
            raise ValueError("set_forming refuses a closed candle")
        self.forming = candle

    def closed(self, count: int | None = None) -> list[Candle]:
        """Closed bars only, oldest first. Never includes the forming bar."""
        bars = list(self._closed)
        return bars[-count:] if count is not None and count < len(bars) else bars

    def __len__(self) -> int:
        return len(self._closed)

    def __bool__(self) -> bool:
        """Always truthy.

        Without this, ``__len__`` would make a buffer that holds only a forming
        bar indistinguishable from a missing buffer under ``if buffer:``, which
        silently hid live forming candles until it was caught in testing.
        """
        return True


class CollectorHealth:
    """Point-in-time view of collector state, safe to serialize to a client."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)

    @property
    def status(self) -> HealthStatus:
        return HealthStatus(self._payload["status"])

    @property
    def is_stale(self) -> bool:
        return bool(self._payload["stream"]["is_stale"])


class MarketStreamCollector:
    """Owns the WebSocket lifecycle and the live cache built from it."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        intervals: list[str] | None = None,
        *,
        persistence: PersistenceProvider | None = None,
        client: BinanceStreamClient | None = None,
        include_book_ticker: bool = True,
    ) -> None:
        settings = get_settings()
        raw_symbols = symbols if symbols is not None else settings.stream_symbols
        raw_intervals = intervals if intervals is not None else settings.stream_intervals

        # Normalize once, up front: a bad symbol should fail at startup, not on
        # the first event three hours later.
        self.symbols = [normalize_symbol(s) for s in raw_symbols]
        self.intervals = [i.lower() for i in raw_intervals]
        self._include_book_ticker = include_book_ticker

        self._client = client or BinanceStreamClient()
        self._persistence = persistence
        self._buffers: dict[tuple[str, Timeframe], _CandleBuffer] = {}
        self._quotes: dict[str, Quote] = {}
        self._pending_persist: list[Candle] = []
        self._task: asyncio.Task[None] | None = None
        self._started_at: float | None = None
        self._events_processed = 0
        self._closed_bars_stored = 0
        self._persist_failures = 0
        self._last_persist_error: str | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # lifecycle                                                          #
    # ------------------------------------------------------------------ #

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start the consume loop in the background. Idempotent."""
        if self.running:
            return
        self._started_at = time.monotonic()
        self._task = asyncio.create_task(self._run(), name="quantedge-market-collector")
        log.info(
            "market collector started",
            extra={"symbols": len(self.symbols), "intervals": self.intervals},
        )

    async def stop(self) -> None:
        """Stop the loop and flush anything still queued for persistence."""
        self._client.stop()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        # Close the socket while the loop is still running; leaving it to GC
        # races interpreter shutdown.
        await self._client.aclose()
        await self._flush_persist(force=True)
        log.info("market collector stopped", extra=self.stats())

    async def run_forever(self) -> None:
        """Run in the foreground -- the entry point used by the worker process."""
        await self.start()
        assert self._task is not None
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def _run(self) -> None:
        """Consume events until cancelled. Never exits on a data error."""
        try:
            async for event in self._client.stream(
                self.symbols,
                self.intervals,
                include_book_ticker=self._include_book_ticker,
            ):
                try:
                    await self._handle(event)
                except Exception as exc:  # noqa: BLE001 - one bad event must not stop the feed
                    log.warning(
                        "collector failed to handle event",
                        extra={"error": type(exc).__name__, "kind": event.kind},
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("collector loop terminated", extra={"error": type(exc).__name__})
            raise

    # ------------------------------------------------------------------ #
    # event handling                                                     #
    # ------------------------------------------------------------------ #

    async def _handle(self, event: StreamEvent) -> None:
        self._events_processed += 1
        if event.kind == "kline":
            await self._handle_candle(event.payload)
        elif event.kind == "book_ticker":
            self._quotes[event.symbol] = event.payload

    async def _handle_candle(self, candle: Candle) -> None:
        key = (candle.symbol, candle.timeframe)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = self._buffers[key] = _CandleBuffer()

        if candle.is_closed:
            if buffer.append_closed(candle):
                self._closed_bars_stored += 1
                # Only closed bars are ever queued for durable storage.
                self._pending_persist.append(candle)
                await self._flush_persist()
        else:
            buffer.set_forming(candle)

    async def _flush_persist(self, *, force: bool = False) -> None:
        """Batch-write closed candles when persistence is available."""
        if self._persistence is None or not self._persistence.available:
            self._pending_persist.clear()  # nothing durable to write to
            return
        if not self._pending_persist:
            return
        if not force and len(self._pending_persist) < _PERSIST_BATCH:
            return

        async with self._lock:
            batch, self._pending_persist = self._pending_persist, []
        try:
            await self._persistence.save_candles(batch)
        except Exception as exc:  # noqa: BLE001 - storage must not kill the feed
            self._persist_failures += 1
            self._last_persist_error = type(exc).__name__
            log.error(
                "failed to persist candles",
                extra={"error": type(exc).__name__, "count": len(batch)},
            )

    # ------------------------------------------------------------------ #
    # read API -- what the services layer consumes                       #
    # ------------------------------------------------------------------ #

    def closed_candles(
        self, symbol: str, timeframe: Timeframe, count: int | None = None
    ) -> list[Candle]:
        """Closed bars from the live cache. Empty when not warmed up yet.

        The forming bar is *never* returned here. Callers wanting it must ask
        for it by name via :meth:`forming_candle`.
        """
        buffer = self._buffers.get((normalize_symbol(symbol), timeframe))
        # `is None`, not truthiness: _CandleBuffer defines __len__, so a buffer
        # holding only a forming bar is falsy despite existing.
        return buffer.closed(count) if buffer is not None else []

    def forming_candle(self, symbol: str, timeframe: Timeframe) -> Candle | None:
        """The current incomplete bar, if any.

        This is a partial observation whose high/low/close are still moving. It
        is valid for display and for spread/last-price purposes; it must not be
        fed to indicators or treated as historical evidence.
        """
        buffer = self._buffers.get((normalize_symbol(symbol), timeframe))
        return buffer.forming if buffer is not None else None

    def candle_series(
        self, symbol: str, timeframe: Timeframe, count: int | None = None
    ) -> CandleSeries | None:
        """Closed cache contents as a :class:`CandleSeries`, or ``None``."""
        canonical = normalize_symbol(symbol)
        bars = self.closed_candles(canonical, timeframe, count)
        if not bars:
            return None
        return CandleSeries(
            provider="binance",
            symbol=canonical,
            asset_class=AssetClass.CRYPTO,
            timeframe=timeframe,
            candles=bars,
            includes_forming_candle=False,
            source="websocket_cache",
        )

    def latest_quote(self, symbol: str) -> Quote | None:
        """Most recent best bid/ask seen on the stream."""
        return self._quotes.get(normalize_symbol(symbol))

    def coverage(self) -> dict[str, int]:
        """How many closed bars are cached per series -- the warm-up view."""
        return {
            f"{symbol}:{timeframe.value}": len(buffer)
            for (symbol, timeframe), buffer in sorted(
                self._buffers.items(), key=lambda kv: (kv[0][0], kv[0][1].value)
            )
        }

    # ------------------------------------------------------------------ #
    # health                                                             #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "uptime_seconds": (
                round(time.monotonic() - self._started_at, 1)
                if self._started_at is not None
                else None
            ),
            "events_processed": self._events_processed,
            "closed_bars_stored": self._closed_bars_stored,
            "series_cached": len(self._buffers),
            "quotes_cached": len(self._quotes),
            "pending_persist": len(self._pending_persist),
            "persist_failures": self._persist_failures,
            "last_persist_error": self._last_persist_error,
        }

    def health(self) -> CollectorHealth:
        """Collector status. Degraded and stale states are reported, not hidden."""
        stream = self._client.stats.snapshot()
        collector = self.stats()

        if not self.running:
            status = HealthStatus.DISABLED
            message = "collector is not running"
        elif not stream["connected"]:
            status = HealthStatus.ERROR
            message = stream.get("last_error") or "stream is disconnected"
        elif stream["is_stale"]:
            status = HealthStatus.DEGRADED
            message = "connected but no recent messages"
        elif collector["closed_bars_stored"] == 0:
            status = HealthStatus.DEGRADED
            message = "connected; waiting for the first bar to close"
        else:
            status = HealthStatus.OK
            message = "streaming"

        return CollectorHealth(
            {
                "status": status.value,
                "message": message,
                "checked_at_utc": utc_now().isoformat(),
                "persistence": (
                    "durable"
                    if self._persistence is not None and self._persistence.available
                    else "memory_only"
                ),
                "stream": stream,
                "collector": collector,
                "coverage": self.coverage(),
            }
        )


_collector: MarketStreamCollector | None = None


def get_collector(**kwargs: Any) -> MarketStreamCollector:
    """Process-wide collector singleton.

    One WebSocket connection is shared by the MCP server, the API and the
    workers within a process; opening one per caller would multiply connections
    against the venue for no benefit.
    """
    global _collector  # noqa: PLW0603 - deliberate process-level singleton
    if _collector is None:
        _collector = MarketStreamCollector(**kwargs)
    return _collector
