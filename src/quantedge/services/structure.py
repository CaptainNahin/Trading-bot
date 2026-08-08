"""Price structure: confirmed swings, trend structure, breakouts.

The single rule that shapes this whole module: **a pivot is only real once the
bars that confirm it exist.** A swing high at bar ``i`` is not knowable at bar
``i``; it becomes knowable after ``swing_confirmation_bars`` later bars have
closed without exceeding it. Detectors that scan a completed array and report
"the swing high was at bar i" are reading the future, and a strategy built on
them cannot be traded -- at bar ``i`` that pivot did not yet exist.

So every swing returned here carries a ``confirmed_at_index``, and the last
``lookback + confirmation`` bars of the series can never contain a confirmed
swing. That trailing blind spot is not a defect to be engineered away; it is the
honest cost of not looking ahead.

Structure classification
------------------------
``UPTREND`` requires *both* higher highs and higher lows -- price making higher
highs while also making lower lows is an expanding range, not a trend, and
labelling it ``UPTREND`` on the strength of the highs alone is how a broadening
top gets traded as continuation. When the two disagree, the label is ``UNCLEAR``
and the disagreement is written into ``notes``.

Breakouts
---------
A breakout requires the close to clear the level by a volatility-scaled buffer
(``breakout_atr_buffer`` x ATR), not merely to touch it. Without the buffer,
every level is "broken" several times an hour by noise. A *failed* breakout --
cleared the level, then closed back inside within ``failed_breakout_bars`` -- is
reported separately, because it is evidence against continuation, not a weaker
form of evidence for it.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from quantedge.config import get_scanner_config
from quantedge.contracts import LiquidityPool, SignalDirection, StructureEvent, StructureReport
from quantedge.errors import InsufficientDataError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantedge.contracts import Candle, CandleSeries

__all__ = [
    "STRUCTURE_VERSION",
    "analyze_structure",
    "find_swings",
]

STRUCTURE_VERSION = "structure-1.0.0"

StructureLabel = Literal["UPTREND", "DOWNTREND", "RANGE", "UNCLEAR"]


def _structure_config() -> dict[str, Any]:
    cfg = get_scanner_config().get("structure", {})
    return {
        "swing_lookback": int(cfg.get("swing_lookback", 5)),
        "swing_confirmation_bars": int(cfg.get("swing_confirmation_bars", 5)),
        "min_swings_for_structure": int(cfg.get("min_swings_for_structure", 4)),
        "breakout_confirmation_bars": int(cfg.get("breakout_confirmation_bars", 1)),
        "failed_breakout_bars": int(cfg.get("failed_breakout_bars", 3)),
    }


def find_swings(
    candles: Sequence[Candle],
    *,
    lookback: int = 5,
    confirmation_bars: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Locate confirmed swing highs and lows.

    A bar at index ``i`` is a swing high when its high is the strict maximum of
    the window ``[i - lookback, i + lookback]``, and it is *confirmed* only once
    index ``i + confirmation_bars`` exists in the series.

    Returns
    -------
    ``(swing_highs, swing_lows)``, each oldest first. Every entry records the
    index it occurred at and the index at which it became knowable, so a caller
    can verify no future information was used.

    Notes
    -----
    Strict comparison against neighbours means an exact double top produces no
    swing at either bar. That is deliberate: with two equal highs there is no
    single pivot, and picking one arbitrarily would make the output depend on
    scan direction.
    """
    highs: list[dict[str, Any]] = []
    lows: list[dict[str, Any]] = []
    n = len(candles)
    if n < lookback * 2 + 1:
        return highs, lows

    # A pivot at i needs `lookback` bars either side to be a local extreme, and
    # `confirmation_bars` bars after it to be confirmed. Both must exist.
    last_confirmable = n - 1 - max(lookback, confirmation_bars)

    for i in range(lookback, last_confirmable + 1):
        window = candles[i - lookback : i + lookback + 1]
        pivot = candles[i]
        others_high = [float(c.high) for j, c in enumerate(window) if j != lookback]
        others_low = [float(c.low) for j, c in enumerate(window) if j != lookback]

        if float(pivot.high) > max(others_high):
            highs.append(
                {
                    "index": i,
                    "confirmed_at_index": i + confirmation_bars,
                    "price": str(pivot.high),
                    "time_utc": pivot.close_time_utc.isoformat(),
                }
            )
        if float(pivot.low) < min(others_low):
            lows.append(
                {
                    "index": i,
                    "confirmed_at_index": i + confirmation_bars,
                    "price": str(pivot.low),
                    "time_utc": pivot.close_time_utc.isoformat(),
                }
            )

    return highs, lows


def _sequence_direction(swings: list[dict[str, Any]], *, want_rising: bool) -> bool:
    """Whether the two most recent confirmed swings step in one direction.

    "Higher high" is a relation between two swings, so two points is the
    definition, not a shortcut. Requiring three strictly-monotonic points here
    demanded two consecutive noise-free steps in *both* the highs and the lows
    at once -- four perfect steps -- which live intraday data almost never
    satisfies: every symbol and timeframe measured came back ``UNCLEAR``, so the
    trend branches of the regime classifier were unreachable and no signal could
    ever be emitted. That is a stricter rule than this module documents.

    Noise robustness comes from the label requiring highs *and* lows to agree
    (a single noisy pivot moves one series, not both), and trend *strength* is
    graded downstream by ADX and EMA alignment rather than by pivot counting.
    """
    if len(swings) < 2:
        return False
    prices = [Decimal(s["price"]) for s in swings[-2:]]
    if want_rising:
        return prices[0] < prices[1]
    return prices[0] > prices[1]


def analyze_structure(
    candles: Sequence[Candle] | CandleSeries,
    *,
    atr: float | None = None,
) -> StructureReport:
    """Classify price structure from confirmed swings only.

    Parameters
    ----------
    candles:
        Closed candles, oldest first. A :class:`~quantedge.contracts.CandleSeries`
        is accepted and its forming bar dropped.
    atr:
        Current ATR, used to size the breakout buffer. When ``None`` the buffer
        falls back to a fraction of the recent range; a breakout is never
        declared with no volatility reference at all.
    """
    bars = list(getattr(candles, "closed", candles))
    cfg = _structure_config()
    notes: list[str] = []

    forming = [c for c in bars if not c.is_closed]
    if forming:
        # Rule 9, and it matters more here than almost anywhere else: a forming
        # bar's high and low are still moving, so it can register as a swing
        # extreme that evaporates before the bar closes. Dropping it silently
        # would shift every index the caller reasons about, so this raises.
        raise InsufficientDataError(
            "analyze_structure requires closed candles only; "
            f"{len(forming)} forming candle(s) were supplied",
            symbol=bars[0].symbol if bars else None,
        )

    if not bars:
        return StructureReport(
            symbol="", timeframe="1m", structure="UNCLEAR", notes=["no closed candles supplied"]
        )

    symbol = bars[0].symbol
    timeframe = bars[0].timeframe
    lookback = cfg["swing_lookback"]
    confirmation = cfg["swing_confirmation_bars"]

    minimum_bars = lookback * 2 + confirmation + 1
    if len(bars) < minimum_bars:
        return StructureReport(
            symbol=symbol,
            timeframe=timeframe,
            structure="UNCLEAR",
            notes=[
                f"{len(bars)} bars is below the {minimum_bars} needed to confirm a swing "
                f"with lookback={lookback} and confirmation={confirmation}"
            ],
        )

    swing_highs, swing_lows = find_swings(bars, lookback=lookback, confirmation_bars=confirmation)

    total_swings = len(swing_highs) + len(swing_lows)
    if total_swings < cfg["min_swings_for_structure"]:
        notes.append(
            f"only {total_swings} confirmed swing(s); "
            f"{cfg['min_swings_for_structure']} required to classify structure"
        )

    has_hh = _sequence_direction(swing_highs, want_rising=True)
    has_hl = _sequence_direction(swing_lows, want_rising=True)
    has_lh = _sequence_direction(swing_highs, want_rising=False)
    has_ll = _sequence_direction(swing_lows, want_rising=False)

    structure: StructureLabel = "UNCLEAR"
    if total_swings >= cfg["min_swings_for_structure"]:
        if has_hh and has_hl:
            structure = "UPTREND"
        elif has_lh and has_ll:
            structure = "DOWNTREND"
        elif has_lh and has_hl:
            # Lower highs against higher lows: price is compressing.
            structure = "RANGE"
            notes.append("lower highs against higher lows -- contracting range")
        elif has_hh and has_ll:
            structure = "UNCLEAR"
            notes.append(
                "higher highs against lower lows -- expanding range, not a trend; "
                "direction is genuinely ambiguous here"
            )
        else:
            notes.append("swing sequence is not monotonic in either direction")

    # -- support and resistance from the most recent confirmed swings -------- #
    last_close = bars[-1].close
    resistance = _nearest_level(swing_highs, last_close, above=True)
    support = _nearest_level(swing_lows, last_close, above=False)

    # -- breakout assessment ------------------------------------------------- #
    buffer = _breakout_buffer(bars, atr, notes)
    breakout_candidate = False
    breakout_direction: SignalDirection | None = None
    failed_breakout = False

    if resistance is not None and last_close > resistance + buffer:
        breakout_candidate = True
        breakout_direction = SignalDirection.UP
    elif support is not None and last_close < support - buffer:
        breakout_candidate = True
        breakout_direction = SignalDirection.DOWN

    # A failed breakout is checked independently of the current one: price may
    # have cleared a level a few bars ago and already closed back inside.
    failed_breakout, failure_note = _detect_failed_breakout(
        bars, swing_highs, swing_lows, buffer, cfg["failed_breakout_bars"]
    )
    if failure_note:
        notes.append(failure_note)

    # -- structural events (BOS / CHOCH / SWEEP) ------------------------------- #
    events, last_bos, last_choch = _detect_structural_events(
        bars, swing_highs, swing_lows, structure, buffer
    )

    # -- liquidity pools (EQUAL HIGHS / EQUAL LOWS) ---------------------------- #
    equal_highs, equal_lows = _detect_liquidity_pools(bars, swing_highs, swing_lows, buffer)

    # -- internal vs external structure ---------------------------------------- #
    int_highs, int_lows = find_swings(
        bars, lookback=max(2, lookback // 2), confirmation_bars=max(2, confirmation // 2)
    )
    internal_structure = _classify_from_swings(int_highs, int_lows, min_swings=3)
    external_structure = structure

    # -- position within external range (PREMIUM / DISCOUNT / EQUILIBRIUM) ---- #
    prem_disc, range_pos = _calculate_premium_discount(last_close, swing_highs, swing_lows)

    # -- structure confidence --------------------------------------------------- #
    confidence = _calculate_structure_confidence(
        structure, internal_structure, total_swings, breakout_candidate, failed_breakout, events
    )

    return StructureReport(
        symbol=symbol,
        timeframe=timeframe,
        swing_highs=swing_highs[-5:],
        swing_lows=swing_lows[-5:],
        has_higher_highs=has_hh,
        has_higher_lows=has_hl,
        has_lower_highs=has_lh,
        has_lower_lows=has_ll,
        structure=structure,
        breakout_candidate=breakout_candidate,
        breakout_direction=breakout_direction,
        failed_breakout=failed_breakout,
        nearest_resistance=resistance,
        nearest_support=support,
        notes=notes,
        events=events,
        last_bos=last_bos,
        last_choch=last_choch,
        equal_highs=equal_highs,
        equal_lows=equal_lows,
        internal_structure=internal_structure,
        external_structure=external_structure,
        premium_discount=prem_disc,
        range_position=range_pos,
        structure_confidence=confidence,
    )


def _nearest_level(swings: list[dict[str, Any]], price: Decimal, *, above: bool) -> Decimal | None:
    """Closest confirmed swing level on one side of ``price``.

    Falls back to the most extreme recent swing when every level is on the wrong
    side -- if price has cleared all confirmed highs, the highest of them is
    still the level it broke, and reporting ``None`` would erase that.
    """
    if not swings:
        return None
    levels = [Decimal(s["price"]) for s in swings[-10:]]
    candidates = [lv for lv in levels if (lv > price if above else lv < price)]
    if candidates:
        return min(candidates) if above else max(candidates)
    return max(levels) if above else min(levels)


def _breakout_buffer(bars: list[Candle], atr: float | None, notes: list[str]) -> Decimal:
    """Minimum distance beyond a level before a break counts.

    Scaled by ATR so the same rule works on an instrument moving 0.05% a bar and
    one moving 3%. Without a usable ATR, a fraction of the recent average range
    substitutes -- still volatility-derived, and the substitution is noted.
    """
    cfg = get_scanner_config().get("regime", {})
    multiple = Decimal(str(cfg.get("breakout_atr_buffer", 0.25)))

    if atr is not None and atr > 0:
        return Decimal(str(atr)) * multiple

    recent = bars[-20:]
    ranges = [c.high - c.low for c in recent]
    if ranges:
        average = sum(ranges) / Decimal(len(ranges))
        if average > 0:
            notes.append("breakout buffer derived from mean bar range (ATR unavailable)")
            return average * multiple

    notes.append("no volatility reference available; breakout buffer is zero")
    return Decimal(0)


def _detect_failed_breakout(
    bars: list[Candle],
    swing_highs: list[dict[str, Any]],
    swing_lows: list[dict[str, Any]],
    buffer: Decimal,
    window: int,
) -> tuple[bool, str | None]:
    """Did price clear a level and then close back inside within ``window`` bars?

    Only levels confirmed *before* the breakout bar are considered, so a swing
    that became knowable after the break cannot be used to explain it.
    """
    if window <= 0 or len(bars) < window + 1:
        return False, None

    for offset in range(1, min(window, len(bars) - 1) + 1):
        idx = len(bars) - 1 - offset
        bar = bars[idx]

        highs_known = [s for s in swing_highs if s["confirmed_at_index"] <= idx]
        lows_known = [s for s in swing_lows if s["confirmed_at_index"] <= idx]

        for swing in highs_known[-5:]:
            level = Decimal(swing["price"])
            if bar.close > level + buffer and bars[-1].close < level:
                return True, (
                    f"failed upside breakout: cleared {level} {offset} bar(s) ago "
                    f"and closed back below it"
                )
        for swing in lows_known[-5:]:
            level = Decimal(swing["price"])
            if bar.close < level - buffer and bars[-1].close > level:
                return True, (
                    f"failed downside breakout: cleared {level} {offset} bar(s) ago "
                    f"and closed back above it"
                )
    return False, None


def _detect_structural_events(
    bars: list[Candle],
    swing_highs: list[dict[str, Any]],
    swing_lows: list[dict[str, Any]],
    trend: StructureLabel,
    buffer: Decimal,
) -> tuple[list[StructureEvent], StructureEvent | None, StructureEvent | None]:
    events: list[StructureEvent] = []
    last_bos: StructureEvent | None = None
    last_choch: StructureEvent | None = None

    if not bars or len(bars) < 5:
        return events, last_bos, last_choch

    for i in range(1, len(bars)):
        bar = bars[i]
        known_highs = [s for s in swing_highs if s["confirmed_at_index"] < i]
        known_lows = [s for s in swing_lows if s["confirmed_at_index"] < i]

        if not known_highs and not known_lows:
            continue

        if known_highs:
            latest_h = known_highs[-1]
            h_lvl = Decimal(latest_h["price"])
            if bar.close > h_lvl + buffer:
                is_bos = trend in ("UPTREND", "UNCLEAR", "RANGE")
                evt_type = "BOS" if is_bos else "CHOCH"
                direction = SignalDirection.UP
                evt = StructureEvent(
                    event_type=evt_type,
                    direction=direction,
                    level=h_lvl,
                    price=bar.close,
                    occurred_at_index=i,
                    occurred_at_utc=bar.close_time_utc,
                    level_index=latest_h["index"],
                    level_confirmed_at_index=latest_h["confirmed_at_index"],
                    confidence=0.8 if is_bos else 0.85,
                    evidence=[f"{evt_type} upside break of {h_lvl} at bar {i}"],
                    invalidation=f"Close below {h_lvl}",
                )
                events.append(evt)
                if evt_type == "BOS":
                    last_bos = evt
                else:
                    last_choch = evt

        if known_lows:
            latest_l = known_lows[-1]
            l_lvl = Decimal(latest_l["price"])
            if bar.close < l_lvl - buffer:
                is_bos = trend in ("DOWNTREND", "UNCLEAR", "RANGE")
                evt_type = "BOS" if is_bos else "CHOCH"
                direction = SignalDirection.DOWN
                evt = StructureEvent(
                    event_type=evt_type,
                    direction=direction,
                    level=l_lvl,
                    price=bar.close,
                    occurred_at_index=i,
                    occurred_at_utc=bar.close_time_utc,
                    level_index=latest_l["index"],
                    level_confirmed_at_index=latest_l["confirmed_at_index"],
                    confidence=0.8 if is_bos else 0.85,
                    evidence=[f"{evt_type} downside break of {l_lvl} at bar {i}"],
                    invalidation=f"Close above {l_lvl}",
                )
                events.append(evt)
                if evt_type == "BOS":
                    last_bos = evt
                else:
                    last_choch = evt

    return events, last_bos, last_choch


def _detect_liquidity_pools(
    bars: list[Candle],
    swing_highs: list[dict[str, Any]],
    swing_lows: list[dict[str, Any]],
    buffer: Decimal,
) -> tuple[list[LiquidityPool], list[LiquidityPool]]:
    eq_highs: list[LiquidityPool] = []
    eq_lows: list[LiquidityPool] = []

    if len(swing_highs) >= 2:
        for i in range(len(swing_highs) - 1):
            h1 = Decimal(swing_highs[i]["price"])
            h2 = Decimal(swing_highs[i + 1]["price"])
            tol = max(buffer, h1 * Decimal("0.001"))
            if abs(h1 - h2) <= tol:
                avg = (h1 + h2) / Decimal("2")
                swept = any(
                    b.high > avg + tol for b in bars[swing_highs[i + 1]["confirmed_at_index"] :]
                )
                eq_highs.append(
                    LiquidityPool(
                        kind="EQUAL_HIGHS",
                        price=avg,
                        touches=2,
                        indices=[swing_highs[i]["index"], swing_highs[i + 1]["index"]],
                        confirmed_at_index=max(
                            swing_highs[i]["confirmed_at_index"],
                            swing_highs[i + 1]["confirmed_at_index"],
                        ),
                        tolerance_used=tol,
                        swept=swept,
                    )
                )

    if len(swing_lows) >= 2:
        for i in range(len(swing_lows) - 1):
            l1 = Decimal(swing_lows[i]["price"])
            l2 = Decimal(swing_lows[i + 1]["price"])
            tol = max(buffer, l1 * Decimal("0.001"))
            if abs(l1 - l2) <= tol:
                avg = (l1 + l2) / Decimal("2")
                swept = any(
                    b.low < avg - tol for b in bars[swing_lows[i + 1]["confirmed_at_index"] :]
                )
                eq_lows.append(
                    LiquidityPool(
                        kind="EQUAL_LOWS",
                        price=avg,
                        touches=2,
                        indices=[swing_lows[i]["index"], swing_lows[i + 1]["index"]],
                        confirmed_at_index=max(
                            swing_lows[i]["confirmed_at_index"],
                            swing_lows[i + 1]["confirmed_at_index"],
                        ),
                        tolerance_used=tol,
                        swept=swept,
                    )
                )

    return eq_highs, eq_lows


def _classify_from_swings(
    highs: list[dict[str, Any]], lows: list[dict[str, Any]], min_swings: int = 3
) -> StructureLabel:
    total = len(highs) + len(lows)
    if total < min_swings:
        return "UNCLEAR"
    has_hh = _sequence_direction(highs, want_rising=True)
    has_hl = _sequence_direction(lows, want_rising=True)
    has_lh = _sequence_direction(highs, want_rising=False)
    has_ll = _sequence_direction(lows, want_rising=False)

    if has_hh and has_hl:
        return "UPTREND"
    if has_lh and has_ll:
        return "DOWNTREND"
    if has_lh and has_hl:
        return "RANGE"
    return "UNCLEAR"


def _calculate_premium_discount(
    last_close: Decimal, swing_highs: list[dict[str, Any]], swing_lows: list[dict[str, Any]]
) -> tuple[Literal["PREMIUM", "DISCOUNT", "EQUILIBRIUM"] | None, float | None]:
    if not swing_highs or not swing_lows:
        return None, None

    max_high = Decimal(max(s["price"] for s in swing_highs[-3:]))
    min_low = Decimal(min(s["price"] for s in swing_lows[-3:]))

    if max_high <= min_low:
        return None, None

    range_span = max_high - min_low
    pos = float((last_close - min_low) / range_span)

    if 0.45 <= pos <= 0.55:
        zone: Literal["PREMIUM", "DISCOUNT", "EQUILIBRIUM"] = "EQUILIBRIUM"
    elif pos > 0.55:
        zone = "PREMIUM"
    else:
        zone = "DISCOUNT"

    return zone, pos


def _calculate_structure_confidence(
    structure: StructureLabel,
    internal_structure: StructureLabel,
    total_swings: int,
    breakout_candidate: bool,
    failed_breakout: bool,
    events: list[StructureEvent],
) -> float:
    if structure == "UNCLEAR":
        return 0.2

    base = 0.5
    if structure in ("UPTREND", "DOWNTREND"):
        base += 0.2
    if structure == internal_structure:
        base += 0.15
    if total_swings >= 6:
        base += 0.05
    if breakout_candidate:
        base += 0.05
    if failed_breakout:
        base -= 0.25
    if events:
        base += 0.05

    return max(0.0, min(1.0, round(base, 2)))
