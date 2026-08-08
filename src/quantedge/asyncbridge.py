"""One long-lived event loop for calling async code from sync code.

Why a dedicated loop thread instead of :func:`asyncio.run`
---------------------------------------------------------
Providers cache an ``httpx.AsyncClient``, and an httpx client is bound to the
loop that created it -- its connection pool holds transports tied to that
loop's selector. ``asyncio.run`` creates a fresh loop and *closes it on exit*,
so the second sync call through a provider would reach for a pooled connection
whose loop is dead and fail with ``Event loop is closed``. That failure is
intermittent by nature: the first request of a process succeeds, so it survives
a smoke test and breaks under real use.

Running one loop on a daemon thread for the life of the process keeps every
cached client, connection pool, rate-limit window and circuit breaker on the
same loop, which is what those components already assume.

The alternative, ``nest_asyncio``, monkey-patches the running loop to re-enter
itself. That makes a blocking call inside an async handler *look* like it
works while the loop is actually being driven re-entrantly from within a task,
which reorders callbacks and can starve concurrent requests. It is also an
unpinned third-party dependency we do not otherwise need.

Callers
-------
FastAPI routes here are plain ``def``, so Starlette runs them in a worker
thread with no loop of its own; submitting to this loop is then a normal
cross-thread handoff. Async callers must ``await`` the coroutine directly and
must not come through here -- see the guard in :func:`run_sync`.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from typing import TYPE_CHECKING, Any, TypeVar

from quantedge.errors import QuantEdgeError
from quantedge.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Coroutine

__all__ = ["run_sync", "shutdown_loop"]

log = get_logger(__name__)

T = TypeVar("T")

# A sync call that never returns would hang a request thread forever. The
# per-request timeouts in ResilientHttpClient are much tighter than this; this
# is the backstop for a coroutine that never yields, not a request budget.
_DEFAULT_TIMEOUT_SECONDS = 180.0

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Return the shared loop, starting its thread on first use."""
    global _loop, _thread

    with _lock:
        if _loop is not None and not _loop.is_closed():
            return _loop

        ready = threading.Event()
        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(
            target=_run,
            name="quantedge-asyncbridge",
            daemon=True,
        )
        thread.start()
        ready.wait(timeout=10.0)

        _loop, _thread = loop, thread
        return loop


def run_sync(coro: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
    """Run ``coro`` on the shared loop and block until it finishes.

    Exceptions propagate unchanged, so a caller still sees the real
    ``ProviderError`` or ``ValidationError`` rather than a wrapper that hides
    which provider failed and why.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is not None and running is _loop:
        # Submitting here would block the only thread able to complete the
        # work. Fail loudly: a deadlocked request thread is far harder to
        # diagnose than an exception naming the mistake.
        coro.close()
        raise QuantEdgeError(
            "run_sync() called from inside the async bridge loop; "
            "await the coroutine directly instead",
            code="ASYNC_BRIDGE_REENTRY",
        )

    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout or _DEFAULT_TIMEOUT_SECONDS)
    except TimeoutError:
        future.cancel()
        raise


def shutdown_loop() -> None:
    """Stop the loop thread. Idempotent; safe when it never started."""
    global _loop, _thread  # noqa: PLW0603 - deliberate process-level singleton

    with _lock:
        loop, thread = _loop, _thread
        _loop = _thread = None

    if loop is None:
        return

    loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
        thread.join(timeout=5.0)
    if not loop.is_closed():
        loop.close()


atexit.register(shutdown_loop)
