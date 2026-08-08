"""Background market data streaming worker.

Runs the shared :class:`MarketStreamCollector`, which owns the WebSocket
lifecycle: connection, reconnection with backoff, dedupe, and the closed-bar
buffers everything downstream reads from.

This worker deliberately holds no client of its own. An earlier revision opened
a second WebSocket and fed messages to a private collector, which meant the API
and MCP server were reading a cache this worker was not filling -- two live
caches, silently disagreeing. ``get_collector`` returns the process-wide
singleton, so what this worker collects is what the rest of the process serves.

Symbols and intervals come from settings rather than being hardcoded here, so
the streamed set matches the configured one.
"""

from __future__ import annotations

import asyncio

from quantedge.logging import get_logger
from quantedge.services.streams import get_collector

logger = get_logger(__name__)


async def run_stream_worker() -> None:
    """Maintain WebSocket market streams until cancelled."""
    collector = get_collector()
    logger.info(
        "starting stream worker",
        extra={"symbols": len(collector.symbols), "intervals": collector.intervals},
    )
    try:
        await collector.run_forever()
    except asyncio.CancelledError:
        logger.info("stream worker cancelled; stopping collector")
        await collector.stop()
        raise


if __name__ == "__main__":
    asyncio.run(run_stream_worker())
