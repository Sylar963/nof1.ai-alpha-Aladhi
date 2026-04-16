"""Local indicator computation from raw OHLCV candles.

Computes SMA, Keltner channels, anchored VWAP, and opening range entirely
from candle arrays — no external API needed.  Used as the primary indicator
source (from Hyperliquid candles) with TAAPI kept only as a fallback.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo


# Opening Range window: 20:00–20:15 America/Tijuana (DST-aware)
TIJUANA_TZ = ZoneInfo("America/Tijuana")
OR_START_HOUR = 20
OR_DURATION_MINUTES = 15


def _or_window_ms(now_utc: datetime) -> tuple[int, int]:
    """Return the (start_ms, end_ms) of the most recent 20:00–20:15 Tijuana window.

    If the current Tijuana time is before 20:00, returns yesterday's window.
    """
    now_tj = now_utc.astimezone(TIJUANA_TZ)
    start_tj = now_tj.replace(hour=OR_START_HOUR, minute=0, second=0, microsecond=0)
    if now_tj < start_tj:
        start_tj = start_tj - timedelta(days=1)
    end_tj = start_tj + timedelta(minutes=OR_DURATION_MINUTES)
    start_ms = int(start_tj.astimezone(timezone.utc).timestamp() * 1000)
    end_ms = int(end_tj.astimezone(timezone.utc).timestamp() * 1000)
    return start_ms, end_ms


def to_price_candles(candles: list[dict], max_bars: int = 200) -> dict:
    """Transform Hyperliquid ``{t,o,h,l,c,v}`` candle dicts into Plotly-ready arrays.

    Returns ``{}`` when ``candles`` is empty so downstream callers can treat
    "no OHLC" as falsy.
    """
    if not candles:
        return {}
    tail = candles[-max_bars:]
    return {
        "time": [int(c["t"]) for c in tail],
        "open": [float(c["o"]) for c in tail],
        "high": [float(c["h"]) for c in tail],
        "low": [float(c["l"]) for c in tail],
        "close": [float(c["c"]) for c in tail],
    }


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------

def compute_sma(closes: list[float], period: int) -> Optional[float]:
    """Simple moving average over the last ``period`` closes."""
    if not closes or len(closes) < period or period <= 0:
        return None
    return sum(closes[-period:]) / period


def compute_sma_series(closes: list[float], period: int, results: int = 5) -> list[float]:
    """Return the last ``results`` SMA values (oldest first)."""
    if not closes or len(closes) < period or period <= 0:
        return []
    series = []
    for i in range(max(period, len(closes) - results + 1), len(closes) + 1):
        window = closes[i - period:i]
        series.append(round(sum(window) / period, 4))
    return series[-results:]


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def compute_ema(closes: list[float], period: int) -> Optional[float]:
    """Standard EMA with alpha = 2 / (period + 1)."""
    if not closes or len(closes) < period or period <= 0:
        return None
    seed = sum(closes[:period]) / period
    alpha = 2.0 / (period + 1)
    ema = seed
    for close in closes[period:]:
        ema = alpha * close + (1.0 - alpha) * ema
    return ema


def compute_ema_series(closes: list[float], period: int, results: int = 5) -> list[float]:
    """Return the last ``results`` EMA values (oldest first)."""
    if not closes or len(closes) < period or period <= 0:
        return []
    alpha = 2.0 / (period + 1)
    seed = sum(closes[:period]) / period
    ema = seed
    all_ema = [ema]
    for close in closes[period:]:
        ema = alpha * close + (1.0 - alpha) * ema
        all_ema.append(ema)
    return [round(v, 4) for v in all_ema[-results:]]


# ---------------------------------------------------------------------------
# ATR (true range using OHLC)
# ---------------------------------------------------------------------------

def compute_true_range(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    """Compute true range series from OHLC data.

    TR = max(high - low, |high - prev_close|, |low - prev_close|)
    """
    trs = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        trs.append(max(hl, hc, lc))
    return trs


def compute_atr_ema(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> Optional[float]:
    """ATR using EMA smoothing of true range (matches TAAPI/TradingView)."""
    trs = compute_true_range(highs, lows, closes)
    if len(trs) < period:
        return None
    return compute_ema(trs, period)


# ---------------------------------------------------------------------------
# Keltner channels
# ---------------------------------------------------------------------------

def compute_keltner_series(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 130,
    atr_length: int = 130,
    multiplier: float = 4.0,
    results: int = 5,
) -> dict:
    """Compute the last ``results`` Keltner channel values.

    Returns ``{"lower": [...], "middle": [...], "upper": [...]}``.
    """
    n = len(closes)
    needed = max(period, atr_length + 1)
    if n < needed:
        logging.warning(
            "Keltner: only %d bars, need %d — returning empty series", n, needed,
        )
        return {"lower": [], "middle": [], "upper": []}

    # Compute full EMA series for the middle band
    alpha = 2.0 / (period + 1)
    seed = sum(closes[:period]) / period
    ema = seed
    ema_all = [None] * period
    ema_all[-1] = ema
    for i in range(period, n):
        ema = alpha * closes[i] + (1.0 - alpha) * ema
        ema_all.append(ema)

    # Compute full true-range series then EMA-smooth for ATR
    trs = [0.0]  # placeholder for index 0
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        trs.append(max(hl, hc, lc))

    atr_alpha = 2.0 / (atr_length + 1)
    atr_seed = sum(trs[1:atr_length + 1]) / atr_length
    atr = atr_seed
    atr_all = [None] * (atr_length + 1)
    atr_all[-1] = atr
    for i in range(atr_length + 1, n):
        atr = atr_alpha * trs[i] + (1.0 - atr_alpha) * atr
        atr_all.append(atr)

    # Extract last `results` where both EMA and ATR are available
    upper_series = []
    middle_series = []
    lower_series = []
    start = max(0, n - results)
    for i in range(start, n):
        e = ema_all[i] if i < len(ema_all) else None
        a = atr_all[i] if i < len(atr_all) else None
        if e is not None and a is not None:
            middle_series.append(round(e, 4))
            upper_series.append(round(e + multiplier * a, 4))
            lower_series.append(round(e - multiplier * a, 4))

    return {"lower": lower_series, "middle": middle_series, "upper": upper_series}


# ---------------------------------------------------------------------------
# Anchored VWAP
# ---------------------------------------------------------------------------

def compute_avwap(candles: list[dict]) -> Optional[float]:
    """Compute anchored VWAP from OHLCV candle dicts.

    Candle dicts must have ``h``, ``l``, ``c``, ``v`` keys (floats).
    """
    notional = 0.0
    volume_total = 0.0
    for c in candles:
        try:
            h = float(c["h"])
            l = float(c["l"])
            close = float(c["c"])
            vol = float(c["v"])
        except (KeyError, TypeError, ValueError):
            continue
        if vol <= 0:
            continue
        typical = (h + l + close) / 3.0
        notional += typical * vol
        volume_total += vol

    if volume_total <= 0:
        return None
    return round(notional / volume_total, 4)


# ---------------------------------------------------------------------------
# Opening range
# ---------------------------------------------------------------------------

def compute_opening_range(candles: list[dict], current_spot: Optional[float] = None) -> dict:
    """Compute the first-hour opening range from intraday candle highs/lows.

    Candle dicts must have ``h`` and ``l`` keys (floats).
    All candles passed in should already be filtered to the first hour.
    """
    highs = []
    lows = []
    for c in candles:
        try:
            highs.append(float(c["h"]))
            lows.append(float(c["l"]))
        except (KeyError, TypeError, ValueError):
            continue

    if not highs or not lows:
        return {"high": None, "low": None, "position": "unknown"}

    high = max(highs)
    low = min(lows)
    position = "unknown"
    if current_spot is not None:
        try:
            spot = float(current_spot)
            if spot > high:
                position = "above"
            elif spot < low:
                position = "below"
            else:
                position = "inside"
        except (TypeError, ValueError):
            pass
    return {"high": round(high, 4), "low": round(low, 4), "position": position}


# ---------------------------------------------------------------------------
# High-level: build full indicator bundle from candle arrays
# ---------------------------------------------------------------------------

def build_indicator_bundle(
    candles_5m: list[dict],
    candles_daily: list[dict],
    candles_long: list[dict],
    long_interval: str,
    current_spot: Optional[float] = None,
) -> dict:
    """Build the full indicator bundle matching TAAPI's ``fetch_asset_indicators`` output.

    Args:
        candles_5m: 5-minute OHLCV candles for today (or recent).
                    Must cover at least the first hour for opening range.
        candles_daily: Daily OHLCV candles from the AVWAP anchor date to now.
        candles_long: Higher-timeframe candles (e.g. 4h) — need >= 131 for
                      Keltner(130,130,4) + SMA99.
        long_interval: The interval label (e.g. "4h", "15m").
        current_spot: Latest spot price for position labels.

    Returns:
        Dict matching the ``fetch_asset_indicators`` shape:
        ``{"5m": {...}, "<long_interval>": {...}}``
    """
    result = {"5m": {}, long_interval: {}}

    # --- 5m indicators ---
    closes_5m = [c["c"] for c in candles_5m]
    highs_5m = [c["h"] for c in candles_5m]
    lows_5m = [c["l"] for c in candles_5m]

    result["5m"]["sma99"] = compute_sma_series(closes_5m, period=99, results=5)
    result["5m"]["keltner"] = compute_keltner_series(
        highs_5m, lows_5m, closes_5m,
        period=130, atr_length=130, multiplier=4.0, results=5,
    )

    # Opening range: 20:00–20:15 America/Tijuana (most recent completed window)
    or_start_ms, or_end_ms = _or_window_ms(datetime.now(timezone.utc))
    or_candles = [c for c in candles_5m if or_start_ms <= c["t"] < or_end_ms]
    result["5m"]["opening_range"] = compute_opening_range(or_candles, current_spot)

    # Raw OHLC for the chart (both intraday 5m and long-term frame)
    result["5m"]["price_candles"] = to_price_candles(candles_5m, max_bars=200)

    # Anchored VWAP from daily candles
    avwap = compute_avwap(candles_daily)
    result["5m"]["avwap"] = avwap

    # --- Long-term indicators ---
    closes_long = [c["c"] for c in candles_long]
    highs_long = [c["h"] for c in candles_long]
    lows_long = [c["l"] for c in candles_long]

    result[long_interval]["sma99"] = compute_sma_series(closes_long, period=99, results=5)
    result[long_interval]["keltner"] = compute_keltner_series(
        highs_long, lows_long, closes_long,
        period=130, atr_length=130, multiplier=4.0, results=5,
    )
    result[long_interval]["avwap"] = avwap
    result[long_interval]["price_candles"] = to_price_candles(candles_long, max_bars=200)

    logging.info(
        "Local indicators: 5m sma99=%d vals, keltner=%d vals, OR=%s | %s sma99=%d vals, keltner=%d vals",
        len(result["5m"]["sma99"]),
        len(result["5m"]["keltner"].get("middle", [])),
        result["5m"]["opening_range"]["position"],
        long_interval,
        len(result[long_interval]["sma99"]),
        len(result[long_interval]["keltner"].get("middle", [])),
    )

    return result
