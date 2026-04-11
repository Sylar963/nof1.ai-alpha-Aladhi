"""Technical indicators consumed by the options agent.

Per the user's spec, the options agent reads only **two** technical signals
on the underlying — opening range and Keltner channel. Everything else
(RSI, MACD, moving averages, etc.) lives on the perps side; the vol agent
reasons about IV regime and greeks, not chart patterns.

All functions in this module are pure: stateless, no IO, deterministic
inputs in / dicts out. They're called from
:func:`src.backend.options_intel.builder.build_options_context` to populate
the corresponding fields in the :class:`OptionsContext` digest.

Conventions
-----------
- ``compute_opening_range`` takes a list of ``(unix_timestamp, price)``
  tuples (Deribit's mark-price-history shape after timestamp parsing) and
  returns the high/low of any prices that fell within the first 60 minutes
  of the supplied ``today`` UTC date, plus a ``position`` label
  (``above`` / ``inside`` / ``below`` / ``unknown``) describing where the
  current spot sits relative to that range.

- ``compute_atr`` uses the close-to-close approximation
  (mean of ``|close[i] − close[i−1]|`` over the window). True ATR needs
  OHLC; for crypto's 24/7 tape the close-to-close proxy is close enough
  for the regime classifier and significantly simpler than pulling OHLC
  from a separate Deribit endpoint.

- ``compute_keltner_channel`` returns ``{ema20, upper, lower, position}``
  with the EMA centered band and ``±atr_multiplier × ATR`` half-widths.

All functions return ``None`` (or an "unknown" position) when the input
series is too short to compute the indicator — the snapshot serializer
handles ``None`` cleanly so the LLM gets a literal ``null`` rather than a
fake number.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Opening range
# ---------------------------------------------------------------------------


def compute_opening_range(
    intraday_prices: Iterable[tuple[int, float]],
    today: date,
    current_spot: Optional[float],
    minutes: int = 60,
) -> dict:
    """Return the high/low of the first ``minutes`` of the supplied UTC date.

    Args:
        intraday_prices: iterable of ``(unix_seconds, price)`` tuples. The
            order doesn't matter; the function filters by timestamp.
        today: UTC date whose opening range we want.
        current_spot: latest spot price; used only to compute the
            ``position`` label.
        minutes: window length, default 60.

    Returns:
        ``{"high": float|None, "low": float|None, "position": str}``.
        ``position`` is one of ``"above" | "inside" | "below" | "unknown"``.
    """
    start = int(datetime(today.year, today.month, today.day, tzinfo=timezone.utc).timestamp())
    end = start + minutes * 60

    in_window: list[float] = []
    for ts, price in intraday_prices:
        if ts is None or price is None:
            continue
        try:
            ts_int = int(ts)
            price_f = float(price)
        except (TypeError, ValueError):
            continue
        if start <= ts_int < end:
            in_window.append(price_f)

    if not in_window:
        return {"high": None, "low": None, "position": "unknown"}

    high = max(in_window)
    low = min(in_window)
    position = "unknown"
    if current_spot is not None:
        try:
            spot_f = float(current_spot)
            if spot_f > high:
                position = "above"
            elif spot_f < low:
                position = "below"
            else:
                position = "inside"
        except (TypeError, ValueError):
            position = "unknown"

    return {"high": high, "low": low, "position": position}


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


def compute_ema(closes: list[float], period: int) -> Optional[float]:
    """Standard EMA with alpha = 2 / (period + 1).

    Returns ``None`` when the series is shorter than ``period``. Seeds the
    EMA with the simple average of the first ``period`` closes (textbook
    initialization) and then folds the rest forward.
    """
    if not closes or len(closes) < period or period <= 0:
        return None
    seed = sum(closes[:period]) / period
    alpha = 2.0 / (period + 1)
    ema = seed
    for close in closes[period:]:
        ema = alpha * close + (1.0 - alpha) * ema
    return ema


# ---------------------------------------------------------------------------
# ATR (close-to-close approximation)
# ---------------------------------------------------------------------------


def compute_atr(closes: list[float], period: int = 14) -> Optional[float]:
    """Mean absolute close-to-close range over the last ``period`` bars.

    True ATR uses OHLC; this is a close-only approximation that works for
    crypto where there's no session gap. Returns None when the series is
    too short to fill the window.
    """
    if not closes or len(closes) < period + 1 or period <= 0:
        return None
    diffs = [abs(closes[i] - closes[i - 1]) for i in range(len(closes) - period, len(closes))]
    return sum(diffs) / period


# ---------------------------------------------------------------------------
# Keltner channel
# ---------------------------------------------------------------------------


def compute_keltner_channel(
    closes: list[float],
    period: int = 20,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
    current_spot: Optional[float] = None,
) -> dict:
    """Compute the Keltner channel and the position label vs current spot.

    Returns ``{ema20, upper, lower, position}``. When the input series is
    too short, all numeric fields are ``None`` and ``position == "unknown"``.
    """
    ema = compute_ema(closes, period=period)
    atr = compute_atr(closes, period=atr_period)
    if ema is None or atr is None:
        return {"ema20": None, "upper": None, "lower": None, "position": "unknown"}

    upper = ema + atr_multiplier * atr
    lower = ema - atr_multiplier * atr

    position = "unknown"
    if current_spot is not None:
        try:
            spot_f = float(current_spot)
            if spot_f > upper:
                position = "above"
            elif spot_f < lower:
                position = "below"
            else:
                position = "inside"
        except (TypeError, ValueError):
            position = "unknown"

    return {"ema20": ema, "upper": upper, "lower": lower, "position": position}
