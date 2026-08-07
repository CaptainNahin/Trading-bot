"""QuantEdge Live Market Gateway.

A live market intelligence backend. Real provider data flows through
validation, deterministic analysis and a scanner into one service layer, which
is then exposed over two transports -- MCP for Claude Code and HTTP for a
future website. Both call the same services; neither owns business logic.

Non-negotiables, enforced in code rather than left to convention
----------------------------------------------------------------
* Nothing is fabricated. Prices, candles, indicators, news and calendar events
  come from a named provider or they are not returned at all.
* Missing or stale data yields ``INSUFFICIENT_DATA``; weak evidence yields
  ``NO_TRADE``. Neither is filled in with a plausible guess.
* Every numeric calculation is deterministic Python. The LLM interprets
  computed values; it never computes them.
* Forming candles are flagged and excluded from history.
* No broker execution, no Martingale or recovery staking, no login automation,
  no scraping, and no private exchange endpoints.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
