"""Multi-timeframe analysis: execution, confirmation and regime views.

A signal read from one timeframe is a signal read without context. The 5-minute
chart breaking upward means one thing inside a 1-hour uptrend and something quite
different inside a 1-hour downtrend, and the 5-minute chart cannot tell you which
it is in. So every analysis here carries three views, named by the job they do:

* **execution** -- the timeframe the trade is timed on. Entry, and the level that
  invalidates it, come from here.
* **confirmation** -- one step out. It answers whether the execution signal is
  moving with the immediate trend or against it.
* **regime** -- the broad context. It sets which strategies are even applicable,
  and it changes slowly enough that a single bar rarely revises it.

``config/scanner.yaml`` maps each requested horizon onto that triple, so a caller
asks for "5m" and receives 5m / 15m / 1h.

The forming-bar rule, restated because it bites hardest here
------------------------------------------------------------
Rule 9 says no calculation may use an incomplete candle, and multi-timeframe work
is where that rule is most tempting to break. At 10:05 the 1-hour bar is five
minutes old: it has an open, a high, a low and a current price, and it *looks*
like a candle. Treating it as one means the regime view is computed from a bar
that will not resemble itself in fifty-five minutes -- and the resulting regime
label will silently revise itself all hour.

So higher-timeframe forming bars are excluded from every calculation and reported
in :attr:`TimeframeView.forming_candle_present`. The consequence is stated rather
than hidden: on a 1-hour regime view the newest usable information can be up to
an hour old, and a caller that needs to know how stale its context is can read
``last_closed_candle_utc`` and find out. Blending the forming bar in would make
the number look fresher while making it mean less.

Alignment
---------
``aligned_direction`` is set only when the views do not contradict each other.
The alignment score is a weighted agreement measure, not a probability: it says
how much of the available evidence points one way, and it is weighted toward the
slower timeframes because they revise less often.

Disagreement is not averaged away. When the execution timeframe points up and the
regime timeframe points down, that conflict is the single most useful thing the
snapshot can report -- it is written into ``conflicts`` and the aligned direction
is left unset. A caller that wants to trade anyway may; it will do so knowing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from quantedge.config import get_scanner_config
from quantedge.contracts import (
    MultiTimeframeSnapshot,
    QualityStatus,
    SignalDirection,
    Timeframe,
    TimeframeView,
)
from quantedge.errors import InsufficientDataError
from quantedge.services import indicators as ind
from quantedge.services import quality as qual
from quantedge.services import regime as reg
from quantedge.services import structure as struct

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantedge.contracts import AssetClass, Candle, CandleSeries

__all__ = [
    "MTF_VERSION",
    "ROLES",
    "analyze_multi_timeframe",
    "build_mtf_snapshot",
    "build_timeframe_view",
    "get_horizon_timeframes",
    "resolve_horizon",
]

MTF_VERSION = "mtf-1.0.0"

ROLES = ("execution", "confirmation", "regime")

# The regime view moves slowest and revises least, so it carries the most weight
# in the agreement measure; execution moves fastest and is noisiest. These are
# judgement, not fitted values, and they are here rather than inline so the
# judgement is visible and changeable.
_ROLE_WEIGHTS = {"execution": 0.25, "confirmation": 0.35, "regime": 0.40}

# Regimes that carry a direction, and the direction they carry.
_DIRECTIONAL_REGIMES = {
    "STRONG_UPTREND": SignalDirection.UP,
    "WEAK_UPTREND": SignalDirection.UP,
    "STRONG_DOWNTREND": SignalDirection.DOWN,
    "WEAK_DOWNTREND": SignalDirection.DOWN,
}

_STRUCTURE_DIRECTIONS = {
    "UPTREND": SignalDirection.UP,
    "DOWNTREND": SignalDirection.DOWN,
}


def resolve_horizon(horizon: str) -> dict[str, Any]:
    """Map a requested horizon onto its execution/confirmation/regime triple.

    Raises
    ------
    InsufficientDataError
        When the horizon is not configured. The alternative -- guessing a
        plausible triple -- would produce an analysis the configuration never
        sanctioned, on timeframes the caller did not choose.
    """
    aliases = {
        "scalp": "1m",
        "intraday": "5m",
        "swing": "15m",
        "position": "1h",
    }
    resolved_name = aliases.get(horizon.lower(), horizon)

    horizons = get_scanner_config().get("horizons", {})
    entry = horizons.get(resolved_name)
    if not entry:
        available = ", ".join(sorted(horizons)) or "none configured"
        raise InsufficientDataError(
            f"horizon {horizon!r} is not configured; available: {available}"
        )
    return {
        "execution": Timeframe(entry["execution"]),
        "confirmation": Timeframe(entry["confirmation"]),
        "regime": Timeframe(entry["regime"]),
        "lookback": int(entry.get("lookback", 300)),
    }


def get_horizon_timeframes(horizon: str) -> dict[str, Any]:
    """Alias for resolve_horizon."""
    return resolve_horizon(horizon)


def build_mtf_snapshot(
    symbol: str,
    horizon: str,
    exec_candles: Sequence[Candle] | CandleSeries,
    conf_candles: Sequence[Candle] | CandleSeries,
    reg_candles: Sequence[Candle] | CandleSeries,
    exec_quality: Any = None,
    conf_quality: Any = None,
    reg_quality: Any = None,
    exec_struct: Any = None,
    conf_struct: Any = None,
    reg_struct: Any = None,
) -> MultiTimeframeSnapshot:
    """Construct a MultiTimeframeSnapshot from series by role."""
    series_by_role = {
        "execution": exec_candles,
        "confirmation": conf_candles,
        "regime": reg_candles,
    }
    from quantedge.contracts import AssetClass

    return analyze_multi_timeframe(
        symbol=symbol,
        asset_class=AssetClass.CRYPTO,
        horizon=horizon,
        series_by_role=series_by_role,
    )


def build_timeframe_view(
    role: str,
    candles: Sequence[Candle] | CandleSeries,
    *,
    min_bars: int = 0,
    now: Any = None,
) -> TimeframeView:
    """Assess one timeframe: quality, then features, structure and regime.

    Quality is the gate. When it fails, ``features``, ``structure`` and ``regime``
    are left unset rather than computed and marked suspect -- a number computed
    from data known to be broken is worse than an absent one, because absence
    cannot be misread as a measurement.
    """
    forming_present = bool(getattr(candles, "includes_forming_candle", False))
    all_bars = list(getattr(candles, "candles", candles))
    bars = [c for c in all_bars if c.is_closed]
    if not forming_present:
        forming_present = len(bars) != len(all_bars)

    provider = bars[0].provider if bars else "unknown"
    timeframe = bars[0].timeframe if bars else Timeframe.M1

    quality = qual.assess_quality(
        bars,
        min_bars=min_bars,
        now=now,
    )

    view = TimeframeView(
        role=role,  # type: ignore[arg-type]
        timeframe=timeframe,
        provider=provider,
        bars_available=len(bars),
        last_closed_candle_utc=bars[-1].close_time_utc if bars else None,
        forming_candle_present=forming_present,
        forming_candle_excluded_from_analysis=True,
        quality=quality,
    )

    if quality.status is QualityStatus.FAIL:
        return view

    features = ind.compute_features(bars)
    structure = struct.analyze_structure(bars, atr=features.atr_14)
    regime = reg.classify_regime_from_features(
        structure=structure,
        features=features,
        bb_width_history=_bb_width_history(bars),
        atr_history=_atr_history(bars),
    )
    return view.model_copy(update={"features": features, "structure": structure, "regime": regime})


def _bb_width_history(bars: Sequence[Candle], window: int = 60) -> list[float | None]:
    """Bollinger width over a trailing window, for percentile ranking.

    Recomputed per bar rather than cached: this is O(window x bb_period) on a few
    hundred bars, which is cheap enough that a cache would buy little and could
    go stale against the series it describes.
    """
    if len(bars) < 20:
        return []
    out: list[float | None] = []
    start = max(20, len(bars) - window)
    closes_all = [float(c.close) for c in bars]
    for end in range(start, len(bars) + 1):
        upper, middle, lower = ind.bollinger(closes_all[:end], 20, 2.0)
        up, mid, low = upper[-1], middle[-1], lower[-1]
        if np.isnan(up) or np.isnan(mid) or np.isnan(low) or mid == 0:
            out.append(None)
            continue
        # Same normalisation the indicator service uses, so the percentile is
        # ranking the current width against comparable historical values.
        out.append(float((up - low) / mid * 100.0))
    return out


def _atr_history(bars: Sequence[Candle], window: int = 60) -> list[float | None]:
    """ATR over a trailing window, for the shock comparison."""
    if len(bars) < 15:
        return []
    out: list[float | None] = []
    start = max(15, len(bars) - window)
    for end in range(start, len(bars) + 1):
        highs = [float(c.high) for c in bars[:end]]
        lows = [float(c.low) for c in bars[:end]]
        closes = [float(c.close) for c in bars[:end]]
        atr_arr = ind.atr(highs, lows, closes)
        out.append(float(atr_arr[-1]) if atr_arr.size and not np.isnan(atr_arr[-1]) else None)
    return out


def _view_direction(view: TimeframeView) -> SignalDirection | None:
    """The direction a single view implies, or ``None`` when it implies none.

    Regime is consulted before structure: it already incorporates structure plus
    trend strength, so a regime of ``HIGH_VOLATILITY_RANGE`` should not be
    overridden by the structure label it was computed from.
    """
    if view.regime is not None:
        direction = _DIRECTIONAL_REGIMES.get(str(view.regime.regime))
        if direction is not None:
            return direction
        # A breakout carries the direction of the break itself.
        if str(view.regime.regime) == "BREAKOUT" and view.structure is not None:
            return view.structure.breakout_direction
        # Every other regime -- ranges, shocks, uncertain -- is non-directional,
        # and reading a direction out of its structure would defeat the point.
        return None
    if view.structure is not None:
        return _STRUCTURE_DIRECTIONS.get(view.structure.structure)
    return None


def analyze_multi_timeframe(
    symbol: str,
    asset_class: AssetClass,
    horizon: str,
    series_by_role: dict[str, Sequence[Candle] | CandleSeries],
    *,
    min_bars: int = 0,
    now: Any = None,
) -> MultiTimeframeSnapshot:
    """Build the three views and measure their agreement.

    Parameters
    ----------
    series_by_role:
        Candles keyed by ``"execution"``, ``"confirmation"`` and ``"regime"``. A
        missing role is recorded as a warning and excluded from the agreement
        measure; it is not silently substituted from another timeframe.
    """
    views: list[TimeframeView] = []
    warnings: list[str] = []
    conflicts: list[str] = []

    expected = resolve_horizon(horizon)

    for role in ROLES:
        series = series_by_role.get(role)
        if series is None:
            warnings.append(f"no {role} series supplied; that view is absent from the analysis")
            continue
        view = build_timeframe_view(role, series, min_bars=min_bars, now=now)
        if view.timeframe != expected[role] and view.bars_available:
            # Not fatal, but the caller asked for a specific triple and did not
            # get it, so the analysis is not the one they think they requested.
            warnings.append(
                f"{role} view is {view.timeframe} but horizon {horizon!r} expects {expected[role]}"
            )
        if view.quality.status is QualityStatus.FAIL:
            warnings.append(
                f"{role} view ({view.timeframe}) failed data quality and contributes "
                f"no direction: {view.quality.blocking_reasons[:1]}"
            )
        elif view.quality.status is QualityStatus.DEGRADED:
            warnings.append(f"{role} view ({view.timeframe}) is DEGRADED but still counted")
        if view.forming_candle_present:
            warnings.append(
                f"{role} view ({view.timeframe}) has a forming candle; it is excluded, so "
                f"the newest closed information is from {view.last_closed_candle_utc}"
            )
        views.append(view)

    directions = {
        view.role: _view_direction(view)
        for view in views
        if view.quality.status is not QualityStatus.FAIL
    }
    voted = {role: d for role, d in directions.items() if d is not None}

    up_weight = sum(_ROLE_WEIGHTS[r] for r, d in voted.items() if d is SignalDirection.UP)
    down_weight = sum(_ROLE_WEIGHTS[r] for r, d in voted.items() if d is SignalDirection.DOWN)
    available_weight = sum(
        _ROLE_WEIGHTS[view.role] for view in views if view.quality.status is not QualityStatus.FAIL
    )

    for role, direction in voted.items():
        for other_role, other in voted.items():
            if role < other_role and direction is not other:
                conflicts.append(
                    f"{role} points {direction.value} while {other_role} points {other.value}"
                )

    # Abstention is not disagreement. A ranging or uncertain view carries no
    # directional information, so it is dropped from the average and the
    # remaining weights are renormalised -- the same rule ``scoring._blend``
    # applies to an indicator that is undefined on the bars available. Dividing
    # by the full stack instead scored an abstaining view exactly like an
    # opposing one, which put a hard ceiling on the achievable agreement: with
    # these weights a symbol whose confirmation and regime views were both
    # ranging could reach at most 0.25 and so could never clear a 0.5 gate, no
    # matter how cleanly its execution view was trending.
    voted_weight = up_weight + down_weight
    abstaining = [
        view.role
        for view in views
        if view.quality.status is not QualityStatus.FAIL and directions.get(view.role) is None
    ]

    aligned: SignalDirection | None = None
    score = 0.0
    if voted_weight > 0 and not conflicts:
        if up_weight > 0 and down_weight == 0:
            aligned = SignalDirection.UP
            score = up_weight / voted_weight
        elif down_weight > 0 and up_weight == 0:
            aligned = SignalDirection.DOWN
            score = down_weight / voted_weight

    # Renormalising makes the score say "the views that spoke were unanimous"
    # rather than "the whole stack agrees", so participation has to travel with
    # it or the two get confused. It is measured against the weight that was
    # usable, not the full stack, so a view lost to a data-quality failure does
    # not read as an abstention it never made.
    participation = voted_weight / available_weight if available_weight > 0 else 0.0

    if not voted and views:
        warnings.append(
            "no view carries a direction; every usable timeframe is ranging, shocked or uncertain"
        )
    elif abstaining:
        warnings.append(
            f"agreement is measured over {round(participation, 2)} of the usable timeframe "
            f"weight; these views abstain and are excluded: {', '.join(sorted(abstaining))}"
        )

    return MultiTimeframeSnapshot(
        symbol=symbol,
        asset_class=asset_class,
        horizon=horizon,
        views=views,
        aligned_direction=aligned,
        alignment_score=round(min(1.0, max(0.0, score)), 4),
        participation=round(min(1.0, max(0.0, participation)), 4),
        abstaining_roles=sorted(abstaining),
        conflicts=conflicts,
        warnings=warnings,
    )
