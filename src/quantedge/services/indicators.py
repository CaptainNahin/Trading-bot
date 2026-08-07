"""Deterministic technical indicators.

Every number the system reports about price is computed here, in Python, from
closed candles. **Rule 10 lives in this module**: the LLM never estimates an
RSI or eyeballs a trend -- it receives values produced by this code and
interprets them. If a value is not computable, this module returns ``None`` and
names the gap; it never approximates.

Design decisions that affect correctness
----------------------------------------
*Closed bars only.* :func:`compute_features` raises if handed a forming candle.
A forming bar's high, low and close all still move, so an indicator built on it
silently rewrites itself on the next tick -- which is exactly how a backtest
comes to disagree with live trading.

*Wilder smoothing where Wilder defined it.* RSI, ATR, ADX and the directional
indicators use Wilder's ``1/n`` recursive average, not an ``2/(n+1)`` EMA. The
two differ by roughly a factor of two in effective lookback, so mixing them
produces numbers that match no published chart and cannot be reconciled with a
broker's platform.

*SMA seeding for EMAs.* The first EMA value is the SMA of the first ``n``
closes, then the recursion runs. Seeding with ``close[0]`` instead is common in
quick implementations and leaves a visible transient for hundreds of bars on an
EMA-200.

*Population standard deviation for Bollinger Bands.* Bollinger specified
population (``ddof=0``); using the sample deviation widens every band slightly
and would not match any charting package.

*Float math, Decimal prices.* Prices arrive as :class:`~decimal.Decimal` and are
converted to float here. Indicator values are inherently approximate real
numbers -- an EMA of a price is not a price -- so float is the honest type. The
underlying prices themselves are never round-tripped through float.

Warm-up
-------
Each indicator needs a minimum history before its output means anything. The
requirements are declared in :data:`WARMUP_BARS` and enforced per indicator:
an under-warmed indicator is ``None`` and its name appears in
``FeatureSnapshot.missing_features``. A zero is never substituted -- zero is a
legitimate value for several of these, so it cannot double as "unknown".
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from quantedge.contracts import FeatureSnapshot, utc_now
from quantedge.errors import InsufficientDataError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantedge.contracts import Candle, CandleSeries

__all__ = [
    "WARMUP_BARS",
    "adx",
    "atr",
    "bollinger",
    "compute_features",
    "ema",
    "macd",
    "roc",
    "rsi",
    "sma",
    "true_range",
]

# Minimum closed bars before each indicator is trusted. These are the **first
# meaningful value*, not the point of full convergence: an EMA-200 seeded from
# an SMA-200 is defined at bar 200 but still carries seeding influence for a
# while. Callers wanting a converged value should ask for more history, which is
# why config/scanner.yaml sets minimum_bars to 210 and strict to 250.
WARMUP_BARS: dict[str, int] = {
    "simple_return": 2,
    "log_return": 2,
    "sma_20": 20,
    "sma_50": 50,
    "sma_200": 200,
    "ema_9": 9,
    "ema_20": 20,
    "ema_50": 50,
    "ema_200": 200,
    "rsi_14": 15,  # 14 changes require 15 closes
    "macd": 26,
    "macd_signal": 34,  # 26 slow EMA + 9-period signal seeding
    "atr_14": 15,  # 14 true ranges require 15 bars
    "adx_14": 28,  # 14 for DM/TR smoothing + 14 for DX smoothing
    "bollinger": 20,
    "roc_10": 11,
    "realized_volatility_20": 21,
    "volume_change_percent": 2,
    "ema_20_slope": 25,
    "ema_50_slope": 55,
    "sma_50_slope": 55,
    "distance_from_high_20": 20,
    "distance_from_low_20": 20,
    "distance_from_high_50": 50,
    "distance_from_low_50": 50,
}

# Bars used to measure a moving-average slope. Short enough to react, long
# enough that one noisy bar cannot flip the sign.
SLOPE_LOOKBACK = 5


# --------------------------------------------------------------------------- #
# Primitives                                                                   #
# --------------------------------------------------------------------------- #


def sma(values: Sequence[float], period: int) -> np.ndarray:
    """Simple moving average. Positions before warm-up are ``NaN``.

    ``NaN`` rather than a truncated array so every returned series stays index-
    aligned with the input candles; misaligned indicator arrays are a rich
    source of off-by-one errors that produce plausible-looking wrong numbers.
    """
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.size, np.nan)
    if arr.size < period or period <= 0:
        return out
    # Cumulative-sum windowing: exact for the magnitudes involved here and O(n).
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out[period - 1 :] = (cumsum[period:] - cumsum[:-period]) / float(period)
    return out


def ema(values: Sequence[float], period: int) -> np.ndarray:
    """Exponential moving average, seeded with the SMA of the first ``period``.

    Uses the standard ``alpha = 2 / (period + 1)``. For Wilder's smoothing --
    used by RSI, ATR and ADX -- see :func:`_wilder`.
    """
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.size, np.nan)
    if arr.size < period or period <= 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = float(arr[:period].mean())
    for i in range(period, arr.size):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _wilder(values: Sequence[float], period: int) -> np.ndarray:
    """Wilder's recursive smoothing: ``prev + (x - prev) / period``.

    Equivalent to an EMA with ``alpha = 1 / period``. Seeded with the simple
    mean of the first ``period`` values, exactly as in *New Concepts in
    Technical Trading Systems*.
    """
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.size, np.nan)
    if arr.size < period or period <= 0:
        return out
    out[period - 1] = float(arr[:period].mean())
    for i in range(period, arr.size):
        out[i] = out[i - 1] + (arr[i] - out[i - 1]) / float(period)
    return out


def true_range(high: Sequence[float], low: Sequence[float], close: Sequence[float]) -> np.ndarray:
    """Wilder's true range. The first element is ``NaN`` -- it has no prior close.

    ``max(h-l, |h-prev_close|, |l-prev_close|)``. The gap terms are what make
    this differ from a plain high-low range, and they are the reason ATR does
    not collapse to zero across an overnight gap.
    """
    h = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    out = np.full(h.size, np.nan)
    if h.size < 2:
        return out
    prev_close = c[:-1]
    out[1:] = np.maximum.reduce(
        [
            h[1:] - low_arr[1:],
            np.abs(h[1:] - prev_close),
            np.abs(low_arr[1:] - prev_close),
        ]
    )
    return out


def atr(
    high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14
) -> np.ndarray:
    """Average true range via Wilder smoothing."""
    tr = true_range(high, low, close)
    valid = tr[1:]  # drop the leading NaN
    smoothed = _wilder(valid, period)
    out = np.full(tr.size, np.nan)
    out[1:] = smoothed
    return out


def rsi(close: Sequence[float], period: int = 14) -> np.ndarray:
    """Relative strength index, Wilder's original formulation.

    A period of pure gains yields exactly 100. That is the defined value, not a
    saturation artefact, so it is returned as-is.
    """
    arr = np.asarray(close, dtype=float)
    out = np.full(arr.size, np.nan)
    if arr.size <= period:
        return out

    delta = np.diff(arr)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = _wilder(gains, period)
    avg_loss = _wilder(losses, period)

    # avg_* are indexed over deltas; delta[i] sits between close[i] and
    # close[i+1], so shift by one to realign with the close series.
    for i in range(period - 1, delta.size):
        loss = avg_loss[i]
        gain = avg_gain[i]
        if np.isnan(gain) or np.isnan(loss):
            continue
        if loss == 0.0:
            # No downward movement in the window: RS is infinite, RSI is 100.
            out[i + 1] = 100.0 if gain > 0.0 else 50.0
        else:
            rs = gain / loss
            out[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return out


def macd(
    close: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line and histogram.

    The signal line is an EMA of the MACD line computed only over the region
    where the MACD line exists; feeding the leading ``NaN``s into the EMA would
    poison every subsequent value.
    """
    arr = np.asarray(close, dtype=float)
    fast_ema = ema(arr, fast)
    slow_ema = ema(arr, slow)
    macd_line = fast_ema - slow_ema

    out_signal = np.full(arr.size, np.nan)
    defined = ~np.isnan(macd_line)
    if defined.any():
        start = int(np.argmax(defined))
        sig = ema(macd_line[start:], signal)
        out_signal[start:] = sig

    return macd_line, out_signal, macd_line - out_signal


def adx(
    high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average directional index with +DI and -DI, Wilder's method.

    Returns ``(adx, plus_di, minus_di)``. ADX measures trend *strength* and is
    directionless; the sign of the trend comes from which DI is on top. Reading
    a high ADX as bullish is a standard misuse, so the regime classifier
    consumes all three together.
    """
    h = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    size = h.size
    nan = np.full(size, np.nan)
    if size < period * 2:
        return nan, nan.copy(), nan.copy()

    up_move = h[1:] - h[:-1]
    down_move = low_arr[:-1] - low_arr[1:]
    # Only the larger of the two directional moves counts, and only if positive.
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(h, low_arr, c)[1:]

    smooth_tr = _wilder(tr, period)
    smooth_plus = _wilder(plus_dm, period)
    smooth_minus = _wilder(minus_dm, period)

    # Two distinct reasons a value can be unavailable here, and they must not be
    # conflated. Inside the warm-up window we genuinely do not know yet -> NaN,
    # which becomes ``None`` at the contract boundary. Once smoothing is defined
    # but true range is zero, the instrument did not move at all: both
    # directional movements are nil, so 0 is the measured value, not a guess.
    tr_defined = ~np.isnan(smooth_tr)
    tr_moving = tr_defined & (smooth_tr != 0.0)
    flat = tr_defined & ~tr_moving

    plus_di = np.full_like(smooth_tr, np.nan)
    minus_di = np.full_like(smooth_tr, np.nan)
    np.divide(100.0 * smooth_plus, smooth_tr, out=plus_di, where=tr_moving)
    np.divide(100.0 * smooth_minus, smooth_tr, out=minus_di, where=tr_moving)
    plus_di[flat] = 0.0
    minus_di[flat] = 0.0

    di_sum = plus_di + minus_di
    dx = np.full_like(di_sum, np.nan)
    np.divide(
        100.0 * np.abs(plus_di - minus_di),
        di_sum,
        out=dx,
        where=(di_sum != 0.0) & ~np.isnan(di_sum),
    )
    # Perfectly balanced directional movement -- including the nil-vs-nil case of
    # a flat market. DX is 0 by definition, so ADX reports "no trend" rather
    # than "unknown".
    dx[tr_defined & (di_sum == 0.0)] = 0.0

    adx_valid = _wilder_from_first_valid(dx, period)

    out_adx = np.full(size, np.nan)
    out_plus = np.full(size, np.nan)
    out_minus = np.full(size, np.nan)
    out_adx[1:] = adx_valid
    out_plus[1:] = plus_di
    out_minus[1:] = minus_di
    return out_adx, out_plus, out_minus


def _wilder_from_first_valid(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smooth a series that begins with ``NaN``s, preserving alignment."""
    out = np.full(values.size, np.nan)
    defined = ~np.isnan(values)
    if not defined.any():
        return out
    start = int(np.argmax(defined))
    tail = values[start:]
    if tail.size < period:
        return out
    out[start:] = _wilder(tail, period)
    return out


def bollinger(
    close: Sequence[float], period: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands: ``(upper, middle, lower)``.

    Population standard deviation (``ddof=0``), as originally specified.
    """
    arr = np.asarray(close, dtype=float)
    middle = sma(arr, period)
    std = np.full(arr.size, np.nan)
    if arr.size >= period > 0:
        windows = np.lib.stride_tricks.sliding_window_view(arr, period)
        std[period - 1 :] = windows.std(axis=1, ddof=0)
    return middle + num_std * std, middle, middle - num_std * std


def roc(close: Sequence[float], period: int = 10) -> np.ndarray:
    """Rate of change, in percent, over ``period`` bars."""
    arr = np.asarray(close, dtype=float)
    out = np.full(arr.size, np.nan)
    if arr.size <= period:
        return out
    prior = arr[:-period]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[period:] = np.where(prior != 0, (arr[period:] - prior) / prior * 100.0, np.nan)
    return out


# --------------------------------------------------------------------------- #
# Feature assembly                                                             #
# --------------------------------------------------------------------------- #


def _last(values: np.ndarray) -> float | None:
    """Final value of a series, or ``None`` when undefined.

    ``NaN`` and infinity both become ``None``. An infinite indicator value is
    the result of a degenerate input (a zero price, a zero-range window); it is
    not a number a caller can act on, and letting it through would break JSON
    serialization at the transport boundary anyway.
    """
    if values.size == 0:
        return None
    value = float(values[-1])
    return None if math.isnan(value) or math.isinf(value) else value


def _slope(values: np.ndarray, lookback: int = SLOPE_LOOKBACK) -> float | None:
    """Percent change of a moving average over ``lookback`` bars.

    Expressed as a percentage of the earlier value so slopes are comparable
    across instruments priced in wildly different units -- a 0.4 slope means
    something for both BTCUSDT and EURUSD, whereas a raw price delta does not.
    """
    if values.size <= lookback:
        return None
    current = float(values[-1])
    earlier = float(values[-1 - lookback])
    if math.isnan(current) or math.isnan(earlier) or earlier == 0.0:
        return None
    return (current - earlier) / abs(earlier) * 100.0


def _distance_percent(price: float, level: float | None) -> float | None:
    """Signed distance from ``price`` to ``level``, in percent of ``level``."""
    if level is None or math.isnan(level) or level == 0.0:
        return None
    return (price - level) / abs(level) * 100.0


def compute_features(
    candles: Sequence[Candle] | CandleSeries,
    *,
    provider: str | None = None,
    symbol: str | None = None,
) -> FeatureSnapshot:
    """Compute every indicator at the last closed bar.

    Parameters
    ----------
    candles:
        Closed candles, oldest first. A :class:`~quantedge.contracts.CandleSeries`
        is accepted directly and its forming bar is dropped.

    Raises
    ------
    InsufficientDataError
        If fewer than two closed candles are supplied, or if a forming candle
        reaches this function from a raw list.

    Notes
    -----
    Under-warmed indicators are ``None`` and named in ``missing_features``. That
    list is what lets a caller distinguish "this market has no momentum" from
    "we do not have enough bars to say", which are very different inputs to a
    decision.
    """
    bars = list(getattr(candles, "closed", candles))

    forming = [c for c in bars if not c.is_closed]
    if forming:
        # Rule 9. A caller that hands us a forming bar has a bug we must not
        # paper over by silently dropping it -- the resulting values would be
        # computed from a different bar count than the caller believes.
        raise InsufficientDataError(
            "compute_features requires closed candles only; "
            f"{len(forming)} forming candle(s) were supplied",
            symbol=symbol or (bars[0].symbol if bars else None),
        )

    if len(bars) < 2:
        raise InsufficientDataError(
            f"at least 2 closed candles are required to compute features; got {len(bars)}",
            symbol=symbol or (bars[0].symbol if bars else None),
            missing=["closed_candles"],
        )

    first = bars[0]
    resolved_provider = provider or first.provider
    resolved_symbol = symbol or first.symbol

    closes = np.array([float(c.close) for c in bars], dtype=float)
    highs = np.array([float(c.high) for c in bars], dtype=float)
    lows = np.array([float(c.low) for c in bars], dtype=float)
    volumes = np.array(
        [float(c.volume) if c.volume is not None else np.nan for c in bars], dtype=float
    )

    n = len(bars)
    last_close = float(closes[-1])
    missing: list[str] = []

    def gated(name: str, value: float | None) -> float | None:
        """Return ``value`` only if warm-up is satisfied; else record the gap."""
        required = WARMUP_BARS.get(name, 0)
        if n < required or value is None:
            missing.append(name)
            return None
        return value

    # -- returns ----------------------------------------------------------- #
    prev_close = float(closes[-2])
    simple_return = (last_close - prev_close) / prev_close * 100.0 if prev_close else None
    log_return = (
        math.log(last_close / prev_close) * 100.0 if prev_close > 0 and last_close > 0 else None
    )

    # -- trend ------------------------------------------------------------- #
    ema_9 = ema(closes, 9)
    ema_20 = ema(closes, 20)
    ema_50 = ema(closes, 50)
    ema_200 = ema(closes, 200)
    sma_20 = sma(closes, 20)
    sma_50 = sma(closes, 50)
    sma_200 = sma(closes, 200)

    # -- momentum ---------------------------------------------------------- #
    rsi_14 = rsi(closes, 14)
    macd_line, macd_signal_line, macd_hist = macd(closes)
    roc_10 = roc(closes, 10)

    # -- volatility -------------------------------------------------------- #
    atr_14 = atr(highs, lows, closes, 14)
    adx_14, plus_di, minus_di = adx(highs, lows, closes, 14)
    bb_upper, bb_middle, bb_lower = bollinger(closes, 20, 2.0)

    atr_value = _last(atr_14)
    atr_percent = (
        atr_value / last_close * 100.0 if atr_value is not None and last_close != 0 else None
    )

    bb_up, bb_mid, bb_low = _last(bb_upper), _last(bb_middle), _last(bb_lower)
    bb_width = (
        (bb_up - bb_low) / bb_mid * 100.0
        if None not in (bb_up, bb_mid, bb_low) and bb_mid
        else None
    )
    # %B places the close within the band: 0 at the lower band, 1 at the upper.
    # Outside the bands it legitimately exceeds [0, 1]; it is not clamped,
    # because a close beyond the band is exactly the signal it encodes.
    bb_percent_b = (
        (last_close - bb_low) / (bb_up - bb_low)
        if None not in (bb_up, bb_low) and bb_up != bb_low  # type: ignore[operator]
        else None
    )

    # Realized volatility: standard deviation of the last 20 log returns, in
    # percent per bar. Deliberately NOT annualized -- an annualization factor
    # implies a bar-count-per-year assumption that is wrong for forex weekends
    # and meaningless for crypto.
    realized_vol: float | None = None
    if n >= WARMUP_BARS["realized_volatility_20"] and (closes[-21:] > 0).all():
        log_returns = np.diff(np.log(closes[-21:]))
        realized_vol = float(np.std(log_returns, ddof=0) * 100.0)

    # -- volume ------------------------------------------------------------ #
    volume_change: float | None = None
    if n >= 2 and not math.isnan(volumes[-1]) and not math.isnan(volumes[-2]) and volumes[-2] != 0:
        volume_change = float((volumes[-1] - volumes[-2]) / volumes[-2] * 100.0)

    # -- candle anatomy ----------------------------------------------------- #
    last_bar = bars[-1]
    bar_range = float(last_bar.high) - float(last_bar.low)
    if bar_range > 0:
        body_ratio = abs(float(last_bar.close) - float(last_bar.open)) / bar_range
        upper_wick = (float(last_bar.high) - max(float(last_bar.open), last_close)) / bar_range
        lower_wick = (min(float(last_bar.open), last_close) - float(last_bar.low)) / bar_range
    else:
        # A zero-range bar (no ticks in the interval) is a real occurrence on
        # thin instruments. Ratios are undefined, not zero.
        body_ratio = upper_wick = lower_wick = None  # type: ignore[assignment]

    # -- rolling extremes --------------------------------------------------- #
    high_20 = float(highs[-20:].max()) if n >= 20 else None
    low_20 = float(lows[-20:].min()) if n >= 20 else None
    high_50 = float(highs[-50:].max()) if n >= 50 else None
    low_50 = float(lows[-50:].min()) if n >= 50 else None

    dist_high_20 = _distance_percent(last_close, high_20)
    dist_low_20 = _distance_percent(last_close, low_20)
    dist_high_50 = _distance_percent(last_close, high_50)
    dist_low_50 = _distance_percent(last_close, low_50)

    warmup_target = WARMUP_BARS["ema_200"]

    snapshot = FeatureSnapshot(
        provider=resolved_provider,
        symbol=resolved_symbol,
        timeframe=first.timeframe,
        computed_at_utc=utc_now(),
        as_of_candle_close_utc=last_bar.close_time_utc,
        bars_used=n,
        warmup_satisfied=n >= warmup_target,
        close=last_bar.close,
        simple_return=gated("simple_return", simple_return),
        log_return=gated("log_return", log_return),
        ema_9=gated("ema_9", _last(ema_9)),
        ema_20=gated("ema_20", _last(ema_20)),
        ema_50=gated("ema_50", _last(ema_50)),
        ema_200=gated("ema_200", _last(ema_200)),
        sma_20=gated("sma_20", _last(sma_20)),
        sma_50=gated("sma_50", _last(sma_50)),
        sma_200=gated("sma_200", _last(sma_200)),
        rsi_14=gated("rsi_14", _last(rsi_14)),
        macd=gated("macd", _last(macd_line)),
        macd_signal=gated("macd_signal", _last(macd_signal_line)),
        macd_histogram=gated("macd_signal", _last(macd_hist)),
        atr_14=gated("atr_14", atr_value),
        atr_percent=gated("atr_14", atr_percent),
        adx_14=gated("adx_14", _last(adx_14)),
        plus_di_14=gated("adx_14", _last(plus_di)),
        minus_di_14=gated("adx_14", _last(minus_di)),
        bb_upper=gated("bollinger", bb_up),
        bb_middle=gated("bollinger", bb_mid),
        bb_lower=gated("bollinger", bb_low),
        bb_width=gated("bollinger", bb_width),
        bb_percent_b=gated("bollinger", bb_percent_b),
        roc_10=gated("roc_10", _last(roc_10)),
        realized_volatility_20=gated("realized_volatility_20", realized_vol),
        volume_change_percent=gated("volume_change_percent", volume_change),
        body_ratio=body_ratio,
        upper_wick_ratio=upper_wick,
        lower_wick_ratio=lower_wick,
        ema_20_slope=gated("ema_20_slope", _slope(ema_20)),
        ema_50_slope=gated("ema_50_slope", _slope(ema_50)),
        sma_50_slope=gated("sma_50_slope", _slope(sma_50)),
        distance_from_high_20=gated("distance_from_high_20", dist_high_20),
        distance_from_low_20=gated("distance_from_low_20", dist_low_20),
        distance_from_high_50=gated("distance_from_high_50", dist_high_50),
        distance_from_low_50=gated("distance_from_low_50", dist_low_50),
        missing_features=sorted(set(missing)),
    )
    return snapshot
