"""A single wall-clock deadline for one request, shared by every LLM call in it.

Why this exists (the "60-second / 180-second problem")
------------------------------------------------------
The bot runs on a serverless host (Vercel) that hard-kills any request at a fixed
wall clock -- 60 seconds (``maxDuration`` in ``vercel.json``). A trade DECISION on
the GLM 5.3 Flash reasoning model takes ~30-40s, and it runs *after* a scan that
has already spent time gathering live data. If the scan burns 30s and the decision
is then handed its full 40s budget, the request runs ~70s and the host kills it
with an opaque 504 -- so the GLM decision never returns and the deterministic
fallback *appears* to be what decided. The chat path had the mirror problem: a 12s
budget, too tight for the model to finish, so a canned fallback answered instead of
the brain. Meanwhile the config advertised an ``llm_timeout_seconds`` of 180 -- three
times the host's wall clock -- a value that could never be honoured.

The fix is one deadline per request. The HTTP layer records, at request start, the
instant by which everything must be done (``now + budget``). Every LLM call then
asks for the *smaller* of its own budget and the time actually left, and refuses to
start a call there is no longer time to finish -- failing fast and honestly into the
deterministic / grounded fallback instead of being killed mid-flight by the host.
Off a serverless host (CLI, tests) no deadline is set and calls use their configured
budgets unchanged.

This module is deliberately dependency-free (stdlib + logging only) so it can be
imported from both ``providers`` and ``services`` without an import cycle.
"""

from __future__ import annotations

import contextvars
import math
import os
import time

from quantedge.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "DEADLINE_MARGIN_SECONDS",
    "MIN_LLM_CALL_SECONDS",
    "SERVERLESS_BUDGET_SECONDS",
    "clamp_to_deadline",
    "clear_request_deadline",
    "remaining_seconds",
    "set_request_deadline",
]

# The serverless function's hard wall clock. Vercel kills the request at
# ``maxDuration`` (60s in vercel.json); override for another host via the env var.
SERVERLESS_BUDGET_SECONDS = float(os.getenv("QUANTEDGE_SERVERLESS_BUDGET_SECONDS", "60"))

# Seconds reserved at the tail for gathering the answer, serialising and returning
# the response before the host's guillotine falls. An LLM call is never allowed to
# eat into this, so ``remaining_seconds`` already subtracts it.
DEADLINE_MARGIN_SECONDS = float(os.getenv("QUANTEDGE_DEADLINE_MARGIN_SECONDS", "6"))

# Below this many seconds it is not worth starting an LLM call: it cannot finish in
# time, so we fail fast into the deterministic / grounded fallback rather than burn
# what budget remains on a call the host will kill mid-flight.
MIN_LLM_CALL_SECONDS = 5.0

_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "quantedge_request_deadline", default=None
)


def set_request_deadline(budget_seconds: float | None = None) -> float:
    """Anchor this request's deadline to *now*. Call once, from the HTTP layer.

    Returns the monotonic instant everything must finish by. Always resets, so a
    fresh HTTP request (or a new CLI invocation) starts with a full budget rather
    than inheriting an expired deadline from an earlier one. It is set by the route
    handler -- which runs in the same synchronous call stack as the LLM call it
    bounds -- so the value is guaranteed visible downstream without relying on any
    framework middleware/context-propagation behaviour.
    """
    budget = SERVERLESS_BUDGET_SECONDS if budget_seconds is None else float(budget_seconds)
    deadline = time.monotonic() + budget
    _deadline.set(deadline)
    return deadline


def clear_request_deadline() -> None:
    """Drop any deadline in this context; LLM calls revert to their own budgets."""
    _deadline.set(None)


def remaining_seconds() -> float:
    """Wall-clock seconds an LLM call may safely spend now, or +inf when unset.

    The response margin is already subtracted, so this is spendable time, not the
    raw time to the host kill. Off a serverless host (no deadline set) it is +inf,
    which makes :func:`clamp_to_deadline` a no-op.
    """
    deadline = _deadline.get()
    if deadline is None:
        return math.inf
    return (deadline - time.monotonic()) - DEADLINE_MARGIN_SECONDS


def clamp_to_deadline(desired_seconds: float) -> float:
    """The largest budget an LLM call may use now: ``min(its want, time left)``.

    With no deadline set (CLI, tests) the desired budget is returned unchanged. The
    result can be <= 0 or below :data:`MIN_LLM_CALL_SECONDS` when the request is
    nearly out of time; callers check for that and fail fast rather than dialling a
    doomed sub-second timeout.
    """
    left = remaining_seconds()
    if math.isinf(left):
        return desired_seconds
    return min(desired_seconds, left)
