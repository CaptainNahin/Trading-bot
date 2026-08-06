"""Binance public WebSocket market streams.

Only public market streams are used: ``<symbol>@kline_<interval>`` and
``<symbol>@bookTicker``. There is no user-data stream, which would require a
listen key derived from an API secret.

Reliability concerns handled here
---------------------------------
* **Reconnect with exponential backoff and jitter** -- a dropped socket
  reconnects without stampeding the venue.
* **Heartbeat** -- ``websockets`` answers server pings automatically; we
  additionally track the last *application* message, because a socket that is
  open but silent is worse than one that is closed (it looks healthy while
  serving nothing).
* **Stale detection** -- if no message arrives within a timeout, the connection
  is treated as dead and recycled.
* **Duplicate rejection** -- Binance can redeliver an event across a reconnect.
  Events are keyed on ``(symbol, interval, open_time, is_closed)`` so a replayed
  bar is dropped rather than double-counted.
* **24-hour limit** -- Binance disconnects any stream after 24 hours; the
  reconnect loop treats that as routine, not as an error.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from quantedge.config import get_settings
from quantedge.contracts import utc_now
from quantedge.errors import ProviderUnavailableError
from quantedge.logging import get_logger
from quantedge.providers.binance import mapping
from quantedge.symbols import normalize_symbol

__all__ = ["BinanceStreamClient", "StreamEvent", "StreamStats"]

log = get_logger(__name__)

# Binance closes any stream connection after 24h; recycle before it does.
_MAX_CONNECTION_SECONDS = 23 * 3600
_DEFAULT_STALE_SECONDS = 90.0
_DEDUPE_CAPACITY = 4096

# Give up after this many consecutive connections that yielded no message at
# all. A reconnect loop is meant to ride out a network blip, not to retry a
# broken build forever.
_MAX_BARREN_ATTEMPTS = 5

# Errors that mean "this code is wrong", not "the network hiccuped". Retrying
# these forever burns CPU and hides the defect behind a plausible-looking
# reconnect log, which is exactly how a NameError once masqueraded as an outage
# here. They are re-raised immediately.
_PROGRAMMING_ERRORS = (
    NameError,
    AttributeError,
    TypeError,
    ImportError,
    IndentationError,
    UnboundLocalError,
)


@dataclass(slots=True)
class StreamEvent:
    """One normalized live event."""

    kind: str  # "kline" | "book_ticker"
    symbol: str
    payload: Any  # Candle for kline, Quote for book_ticker
    received_at_monotonic: float
    raw_stream: str | None = None


@dataclass
class StreamStats:
    """Observable health of a stream connection."""

    connected: bool = False
    connect_count: int = 0
    reconnect_count: int = 0
    messages_received: int = 0
    duplicates_rejected: int = 0
    malformed_rejected: int = 0
    closed_candles: int = 0
    last_message_monotonic: float | None = None
    last_message_utc: str | None = None
    connected_since_monotonic: float | None = None
    last_error: str | None = None
    subscribed_streams: list[str] = field(default_factory=list)

    @property
    def seconds_since_last_message(self) -> float | None:
        if self.last_message_monotonic is None:
            return None
        return time.monotonic() - self.last_message_monotonic

    def is_stale(self, threshold_seconds: float = _DEFAULT_STALE_SECONDS) -> bool:
        """A connection with no recent traffic is stale even if 'open'."""
        gap = self.seconds_since_last_message
        if gap is None:
            return self.connected  # connected but never received anything
        return gap > threshold_seconds

    def snapshot(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "connect_count": self.connect_count,
            "reconnect_count": self.reconnect_count,
            "messages_received": self.messages_received,
            "duplicates_rejected": self.duplicates_rejected,
            "malformed_rejected": self.malformed_rejected,
            "closed_candles": self.closed_candles,
            "seconds_since_last_message": (
                round(self.seconds_since_last_message, 2)
                if self.seconds_since_last_message is not None
                else None
            ),
            "last_message_utc": self.last_message_utc,
            "is_stale": self.is_stale(),
            "subscribed_streams": self.subscribed_streams,
            "last_error": self.last_error,
        }


class _DedupeWindow:
    """Bounded LRU of recently seen event keys."""

    def __init__(self, capacity: int = _DEDUPE_CAPACITY) -> None:
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._capacity = capacity

    def is_duplicate(self, key: str) -> bool:
        if key in self._seen:
            self._seen.move_to_end(key)
            return True
        self._seen[key] = None
        if len(self._seen) > self._capacity:
            self._seen.popitem(last=False)
        return False


class BinanceStreamClient:
    """Managed connection to Binance combined public market streams."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        stale_seconds: float = _DEFAULT_STALE_SECONDS,
        max_backoff_seconds: float = 60.0,
    ) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.binance_ws_base_url).rstrip("/")
        self._stale_seconds = stale_seconds
        self._max_backoff = max_backoff_seconds
        self.stats = StreamStats()
        self._dedupe = _DedupeWindow()
        self._stop = asyncio.Event()
        self._socket: Any = None

    # ------------------------------------------------------------------ #
    # lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> BinanceStreamClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Stop the loop and close the socket deterministically.

        A consumer that leaves ``async for`` via ``break`` suspends the
        generator at its ``yield`` rather than finishing it, so the generator's
        own ``finally`` never runs and the socket survives until garbage
        collection -- which, at interpreter shutdown, races the event loop and
        produces "aclose(): asynchronous generator is already running". Calling
        this (or using the client as an async context manager) closes the socket
        while the loop is still alive.
        """
        self._stop.set()
        socket, self._socket = self._socket, None
        if socket is not None:
            with contextlib.suppress(Exception):
                await socket.close()

    # ------------------------------------------------------------------ #
    # stream naming                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def kline_streams(symbols: list[str], intervals: list[str]) -> list[str]:
        """Build ``btcusdt@kline_1m``-style stream names."""
        out: list[str] = []
        for symbol in symbols:
            lower = normalize_symbol(symbol).lower()
            for interval in intervals:
                out.append(f"{lower}@kline_{interval.lower()}")
        return out

    @staticmethod
    def book_ticker_streams(symbols: list[str]) -> list[str]:
        return [f"{normalize_symbol(s).lower()}@bookTicker" for s in symbols]

    def _combined_url(self, streams: list[str]) -> str:
        return f"{self._base_url}/stream?streams={'/'.join(streams)}"

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #

    def stop(self) -> None:
        """Request a graceful shutdown of the consume loop."""
        self._stop.set()

    async def stream_klines(
        self, symbols: list[str], intervals: list[str]
    ) -> AsyncIterator[StreamEvent]:
        """Yield normalized kline events until stopped."""
        streams = self.kline_streams(symbols, intervals)
        async for event in self._consume(streams):
            yield event

    async def stream_book_ticker(self, symbols: list[str]) -> AsyncIterator[StreamEvent]:
        """Yield normalized best bid/ask events until stopped."""
        streams = self.book_ticker_streams(symbols)
        async for event in self._consume(streams):
            yield event

    async def stream(
        self, symbols: list[str], intervals: list[str], *, include_book_ticker: bool = True
    ) -> AsyncIterator[StreamEvent]:
        """Yield kline and (optionally) bookTicker events from one connection."""
        streams = self.kline_streams(symbols, intervals)
        if include_book_ticker:
            streams += self.book_ticker_streams(symbols)
        async for event in self._consume(streams):
            yield event

    # ------------------------------------------------------------------ #
    # connection loop                                                    #
    # ------------------------------------------------------------------ #

    async def _consume(self, streams: list[str]) -> AsyncIterator[StreamEvent]:
        """Connect, read, normalize and reconnect forever until stopped.

        The socket read loop is deliberately inlined rather than delegated to a
        second async generator. Nesting generators means that when a consumer
        breaks out of ``async for``, Python must close the outer generator while
        the inner one is suspended inside ``socket.recv()`` -- which raises
        "asynchronous generator is already running". One generator, one
        suspension point, no such race.
        """
        if not streams:
            raise ValueError("at least one stream must be requested")

        self._stop.clear()
        self.stats.subscribed_streams = streams
        url = self._combined_url(streams)
        attempt = 0
        barren_attempts = 0

        while not self._stop.is_set():
            socket: Any = None
            messages_at_connect = self.stats.messages_received
            try:
                socket = await websockets.connect(
                    url,
                    ping_interval=20,  # keepalive; the server also pings every 20s
                    ping_timeout=20,
                    close_timeout=5,
                    max_queue=1024,
                )
                self._socket = socket
                attempt = 0  # a successful connect resets backoff
                self.stats.connected = True
                self.stats.connect_count += 1
                self.stats.connected_since_monotonic = time.monotonic()
                self.stats.last_error = None
                log.info(
                    "binance stream connected",
                    extra={"streams": len(streams), "base_url": self._base_url},
                )

                connection_started = time.monotonic()
                while not self._stop.is_set():
                    if time.monotonic() - connection_started > _MAX_CONNECTION_SECONDS:
                        log.info("recycling connection before the 24h server limit")
                        break

                    try:
                        raw = await asyncio.wait_for(
                            socket.recv(), timeout=self._stale_seconds
                        )
                    except TimeoutError:
                        # Open but silent. A socket that looks healthy while
                        # serving nothing is worse than one that is closed.
                        log.warning(
                            "binance stream stale; forcing reconnect",
                            extra={"stale_seconds": self._stale_seconds},
                        )
                        self.stats.last_error = (
                            f"stale: no message for {self._stale_seconds}s"
                        )
                        break

                    self.stats.messages_received += 1
                    self.stats.last_message_monotonic = time.monotonic()
                    self.stats.last_message_utc = utc_now().isoformat()

                    event = self._normalize(raw)
                    if event is not None:
                        yield event

            except asyncio.CancelledError:
                self.stats.connected = False
                raise
            except _PROGRAMMING_ERRORS:
                # A defect in this module. Reconnecting cannot fix it, and
                # swallowing it would turn a one-line bug into a silent,
                # permanently-degraded feed. Fail loudly instead.
                self.stats.connected = False
                self.stats.last_error = "internal error (see traceback)"
                log.exception("stream loop hit a programming error; not retrying")
                raise
            except (ConnectionClosed, WebSocketException, OSError) as exc:
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "binance stream disconnected",
                    extra={"error": type(exc).__name__, "attempt": attempt + 1},
                )
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                log.error("unexpected stream error", extra={"error": type(exc).__name__})
            finally:
                self.stats.connected = False
                self._socket = None
                if socket is not None:
                    with contextlib.suppress(Exception):
                        await socket.close()

            if self._stop.is_set():
                break

            # A connection that delivered nothing did not really work, however
            # cleanly it opened. Track those separately from ordinary drops.
            if self.stats.messages_received == messages_at_connect:
                barren_attempts += 1
                if barren_attempts >= _MAX_BARREN_ATTEMPTS:
                    raise ProviderUnavailableError(
                        "binance",
                        f"{barren_attempts} consecutive connections delivered no messages; "
                        f"last error: {self.stats.last_error or 'none'}",
                    )
            else:
                barren_attempts = 0

            delay = self._backoff(attempt)
            attempt += 1
            self.stats.reconnect_count += 1
            log.info("reconnecting to binance stream", extra={"delay_seconds": round(delay, 2)})
            with contextlib.suppress(TimeoutError):
                # Sleep, but wake immediately if stop() is called.
                await asyncio.wait_for(self._stop.wait(), timeout=delay)

        self.stats.connected = False
        log.info("binance stream loop exited", extra=self.stats.snapshot())

    def _normalize(self, raw: str | bytes) -> StreamEvent | None:
        """Parse and normalize one frame. Returns ``None`` when it is dropped."""
        try:
            message = json.loads(raw)
        except (ValueError, TypeError):
            self.stats.malformed_rejected += 1
            log.warning("dropping non-JSON stream frame")
            return None

        # Combined streams wrap payloads as {"stream": ..., "data": {...}}.
        stream_name = message.get("stream") if isinstance(message, dict) else None
        data = message.get("data", message) if isinstance(message, dict) else None
        if not isinstance(data, dict):
            self.stats.malformed_rejected += 1
            return None

        event_type = data.get("e")

        try:
            if event_type == "kline":
                candle = mapping.normalize_kline_event(data)
                kline = data.get("k") or {}
                base = (
                    f"k:{candle.symbol}:{candle.timeframe.value}:"
                    f"{int(candle.open_time_utc.timestamp())}"
                )

                if candle.is_closed:
                    # A bar closes exactly once. Any repeat is a replay from a
                    # reconnect and must not be counted twice.
                    key = f"{base}:closed"
                else:
                    # A forming bar legitimately updates many times per minute,
                    # each with new OHLC. Keying only on open_time would collapse
                    # every update into one "duplicate" and freeze the live price
                    # at the first tick of the minute. Mixing in the event time
                    # and the bar's monotonically-growing volume/trade count keeps
                    # genuine updates distinct while still catching exact replays.
                    revision = data.get("E") or f"{kline.get('v')}:{kline.get('n')}"
                    key = f"{base}:forming:{revision}"

                if self._dedupe.is_duplicate(key):
                    self.stats.duplicates_rejected += 1
                    return None
                if candle.is_closed:
                    self.stats.closed_candles += 1
                return StreamEvent(
                    kind="kline",
                    symbol=candle.symbol,
                    payload=candle,
                    received_at_monotonic=time.monotonic(),
                    raw_stream=stream_name,
                )

            if "b" in data and "a" in data and "s" in data:
                quote = mapping.normalize_book_ticker_event(data)
                # bookTicker updates carry an updateId; use it to dedupe.
                update_id = data.get("u")
                if update_id is not None:
                    key = f"b:{quote.symbol}:{update_id}"
                    if self._dedupe.is_duplicate(key):
                        self.stats.duplicates_rejected += 1
                        return None
                return StreamEvent(
                    kind="book_ticker",
                    symbol=quote.symbol,
                    payload=quote,
                    received_at_monotonic=time.monotonic(),
                    raw_stream=stream_name,
                )
        except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the stream
            self.stats.malformed_rejected += 1
            log.warning(
                "dropping malformed stream event",
                extra={"error": type(exc).__name__, "stream": stream_name},
            )
            return None

        return None

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter, capped."""
        base = min(1.0 * (2**attempt), self._max_backoff)
        return random.uniform(0.5, max(0.5, base))  # noqa: S311 - jitter, not cryptography
