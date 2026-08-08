"""Trade duration vocabulary: what the user picks, what the scanner runs.

The UI offers a *time limit* -- "10 minutes", "20 minutes" -- because that is
how a trade is actually placed. The scanner is configured by *horizon key* in
``config/scanner.yaml`` (``1m``, ``3m``, ``5m``, ``10m``, ``15m``, ``30m``,
``1h``), each mapping to an execution/confirmation/regime timeframe triple.

This module is the single translation between the two, and it resolves against
the config file rather than a hardcoded list, so adding a horizon to the YAML
makes it available to the UI without touching code.

Why words like "swing" are mapped, not rejected
-----------------------------------------------
Earlier UI code sent ``swing``/``scalp``/``intraday``/``position``, which are
not keys in the config; ``resolve_horizon`` raised and every scan failed. Those
words describe a duration, so they are mapped onto the nearest configured
horizon rather than silently defaulted -- and anything genuinely unrecognised
raises with the valid options listed, instead of quietly scanning a timeframe
the user did not ask for.

A 20-minute limit has no ``20m`` horizon; it maps to ``15m``. The expiry the
user is shown is always the duration they chose, never the horizon's -- see
:func:`expiry_for`.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, NamedTuple

from quantedge.config import get_scanner_config
from quantedge.errors import ValidationError

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "TimeLimit",
    "available_time_limits",
    "expiry_for",
    "horizon_for_minutes",
    "horizon_minutes",
    "horizon_timeframes",
    "normalize_horizon",
    "resolve_time_limit",
]

# Duration in minutes for each configured horizon key. Used to pick the closest
# horizon for a requested time limit.
_HORIZON_MINUTES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "1h": 60,
}

# Legacy/verbal names that reached the scanner from the UI. Mapped to the
# nearest configured horizon so an old client keeps working.
_ALIASES: dict[str, str] = {
    "scalp": "1m",
    "intraday": "5m",
    "swing": "15m",
    "position": "1h",
    "1min": "1m",
    "3min": "3m",
    "5min": "5m",
    "10min": "10m",
    "15min": "15m",
    "20min": "15m",
    "30min": "30m",
    "60m": "1h",
    "1hour": "1h",
    "hourly": "1h",
}


class TimeLimit(NamedTuple):
    """One selectable trade duration and the horizon that analyses it."""

    minutes: int
    label: str
    horizon: str

    def to_dict(self) -> dict[str, object]:
        """Serialize for the UI, naming the bars the setup is actually read on.

        ``execution_timeframe`` is included because a 20-minute limit resolves
        to the 15m horizon: showing only the label would leave the trader
        unable to tell which timeframe produced the levels.
        """
        return {
            "minutes": self.minutes,
            "label": self.label,
            "horizon": self.horizon,
            "execution_timeframe": horizon_timeframes(self.horizon)["execution"],
        }


def configured_horizons() -> list[str]:
    """Horizon keys actually present in ``config/scanner.yaml``."""
    horizons = get_scanner_config().get("horizons", {})
    return [str(k) for k in horizons]


def normalize_horizon(horizon: str) -> str:
    """Resolve any accepted spelling to a configured horizon key.

    Raises :class:`ValidationError` listing the valid keys when the input
    cannot be resolved -- a wrong horizon silently replaced by a default would
    analyse a different timeframe than the trade is held for.
    """
    raw = str(horizon).strip().lower()
    valid = configured_horizons()

    if raw in valid:
        return raw

    alias = _ALIASES.get(raw)
    if alias is not None and alias in valid:
        return alias

    # "10", "10 min", "10 minutes", "1 hour", "2h" -> nearest configured horizon.
    # Hours are handled because ``available_time_limits`` labels anything from
    # 60 minutes up as "N hour", and the UI sends that label straight back. A
    # parser that only understood minutes rejected the very strings this module
    # had just offered, so every hour-length limit failed to resolve.
    digits = "".join(ch for ch in raw if ch.isdigit())
    if digits:
        if "hour" in raw or "hr" in raw or raw.endswith("h"):
            return horizon_for_minutes(int(digits) * 60)
        if "min" in raw or raw.endswith("m") or raw == digits:
            return horizon_for_minutes(int(digits))

    raise ValidationError(
        f"unknown horizon '{horizon}'",
        details={"valid_horizons": ", ".join(valid)},
    )


def horizon_for_minutes(minutes: int) -> str:
    """Configured horizon whose duration is closest to ``minutes``.

    Ties resolve to the shorter horizon: analysing a faster timeframe than the
    trade duration is the conservative error, since it will not miss structure
    that forms inside the holding period.
    """
    if minutes <= 0:
        raise ValidationError(
            "time limit must be a positive number of minutes",
            details={"received": str(minutes)},
        )

    valid = configured_horizons()
    candidates = [(k, m) for k, m in _HORIZON_MINUTES.items() if k in valid]
    if not candidates:
        raise ValidationError(
            "config/scanner.yaml defines no usable horizons",
            details={"configured": ", ".join(valid) or "none"},
        )

    return min(candidates, key=lambda kv: (abs(kv[1] - minutes), kv[1]))[0]


def resolve_time_limit(time_limit: str | int) -> str:
    """Resolve a user-selected time limit to a configured horizon key.

    Accepts what a user or an old client actually sends: ``"10 min"``, ``"20"``,
    ``20``, ``"15m"``, ``"swing"``. Everything funnels through
    :func:`normalize_horizon` so there is one definition of a valid horizon.
    """
    if isinstance(time_limit, int):
        return horizon_for_minutes(time_limit)
    return normalize_horizon(time_limit)


def horizon_minutes(horizon: str) -> int:
    """Duration in minutes of a configured horizon key."""
    key = normalize_horizon(horizon)
    minutes = _HORIZON_MINUTES.get(key)
    if minutes is None:
        raise ValidationError(
            f"horizon '{key}' has no known duration",
            details={"known": ", ".join(_HORIZON_MINUTES)},
        )
    return minutes


def horizon_timeframes(horizon: str) -> dict[str, str]:
    """The execution/confirmation/regime timeframes configured for a horizon.

    Read from ``config/scanner.yaml`` rather than derived, so the timeframes the
    risk module recomputes on are the same ones the scanner scored.
    """
    key = normalize_horizon(horizon)
    spec = get_scanner_config().get("horizons", {}).get(key, {})
    if not spec:
        raise ValidationError(
            f"config/scanner.yaml has no timeframes for horizon '{key}'",
            details={"configured": ", ".join(configured_horizons())},
        )
    return {
        "execution": str(spec.get("execution")),
        "confirmation": str(spec.get("confirmation")),
        "regime": str(spec.get("regime")),
    }


def available_time_limits() -> list[TimeLimit]:
    """Selectable durations for the UI, derived from the config.

    Built from the configured horizons plus a 20-minute option, which traders
    expect and which maps to the 15m horizon. Each entry names the horizon it
    resolves to so the UI can show what will actually be analysed.
    """
    limits: list[TimeLimit] = []
    for key in configured_horizons():
        minutes = _HORIZON_MINUTES.get(key)
        if minutes is None:
            continue
        label = f"{minutes} min" if minutes < 60 else f"{minutes // 60} hour"
        limits.append(TimeLimit(minutes, label, key))

    if any(limit.minutes == 15 for limit in limits):
        limits.append(TimeLimit(20, "20 min", "15m"))

    return sorted(limits, key=lambda limit: limit.minutes)


def expiry_for(minutes: int, start: datetime) -> datetime:
    """Expiry timestamp for a trade of ``minutes`` starting at ``start``.

    Deliberately computed from the duration the user selected, not from the
    horizon it mapped to: a 20-minute trade expires in 20 minutes even though
    the 15m horizon analysed it.
    """
    if minutes <= 0:
        raise ValidationError(
            "time limit must be a positive number of minutes",
            details={"received": str(minutes)},
        )
    return start + timedelta(minutes=minutes)
