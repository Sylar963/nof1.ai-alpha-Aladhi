"""Tests for the technicals helpers used by the options-intel snapshot.

These are the only technical signals the options agent reads (per the
user's spec — no RSI/MACD/etc). All functions are stateless and operate
on raw price arrays so they're trivially testable."""

from datetime import datetime, timezone

import pytest

from src.backend.options_intel.technicals import (
    compute_atr,
    compute_ema,
    compute_keltner_channel,
    compute_opening_range,
)


# ---------------------------------------------------------------------------
# compute_opening_range
# ---------------------------------------------------------------------------


def _ts(year, month, day, hour=0, minute=0):
    return int(datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp())


def test_opening_range_returns_min_max_of_first_hour():
    """Opening range = min/max of prices in the first 60 minutes of the UTC day."""
    today = datetime(2026, 4, 11, tzinfo=timezone.utc).date()
    intraday = [
        (_ts(2026, 4, 11, 0, 5), 60000),
        (_ts(2026, 4, 11, 0, 15), 60500),
        (_ts(2026, 4, 11, 0, 30), 59800),  # low
        (_ts(2026, 4, 11, 0, 45), 60800),  # high
        (_ts(2026, 4, 11, 0, 59), 60100),
        (_ts(2026, 4, 11, 1, 5), 59500),  # outside the first hour, must be excluded
        (_ts(2026, 4, 11, 2, 0), 65000),  # outside, must be excluded
    ]
    result = compute_opening_range(intraday, today=today, current_spot=60500)
    assert result["high"] == 60800
    assert result["low"] == 59800
    assert result["position"] == "inside"


def test_opening_range_position_above():
    today = datetime(2026, 4, 11, tzinfo=timezone.utc).date()
    intraday = [(_ts(2026, 4, 11, 0, 5), 60000), (_ts(2026, 4, 11, 0, 30), 60100)]
    result = compute_opening_range(intraday, today=today, current_spot=61000)
    assert result["position"] == "above"


def test_opening_range_position_below():
    today = datetime(2026, 4, 11, tzinfo=timezone.utc).date()
    intraday = [(_ts(2026, 4, 11, 0, 5), 60000), (_ts(2026, 4, 11, 0, 30), 60100)]
    result = compute_opening_range(intraday, today=today, current_spot=58000)
    assert result["position"] == "below"


def test_opening_range_returns_unknown_when_no_data():
    today = datetime(2026, 4, 11, tzinfo=timezone.utc).date()
    result = compute_opening_range([], today=today, current_spot=60000)
    assert result == {"high": None, "low": None, "position": "unknown"}


def test_opening_range_returns_unknown_when_no_first_hour_data():
    """Intraday data exists but none of it falls within today's first hour."""
    today = datetime(2026, 4, 11, tzinfo=timezone.utc).date()
    intraday = [
        (_ts(2026, 4, 10, 0, 30), 59000),  # yesterday
        (_ts(2026, 4, 11, 5, 0), 60000),    # today, but past first hour
    ]
    result = compute_opening_range(intraday, today=today, current_spot=60000)
    assert result["position"] == "unknown"


# ---------------------------------------------------------------------------
# compute_ema
# ---------------------------------------------------------------------------


def test_ema_constant_series_equals_constant():
    closes = [100.0] * 30
    assert compute_ema(closes, period=20) == pytest.approx(100.0)


def test_ema_returns_none_when_series_too_short():
    closes = [100.0, 101.0, 102.0]
    assert compute_ema(closes, period=20) is None


def test_ema_weights_recent_prices_more_heavily():
    """A linearly rising series ends with EMA closer to the recent values."""
    closes = list(range(1, 31))  # 1..30
    ema = compute_ema(closes, period=20)
    assert ema is not None
    # The simple average of 1..30 is 15.5; the EMA should be biased upward.
    assert ema > 15.5


# ---------------------------------------------------------------------------
# compute_atr
# ---------------------------------------------------------------------------


def test_atr_close_to_close_approximation():
    """Close-to-close ATR for synthetic series with constant 100-step jumps = 100."""
    closes = [60000, 60100, 60200, 60100, 60200, 60300, 60200, 60300, 60400, 60300,
              60400, 60500, 60400, 60500, 60600]
    atr = compute_atr(closes, period=14)
    assert atr == pytest.approx(100.0, rel=0.01)


def test_atr_returns_none_when_series_too_short():
    closes = [60000, 60100]
    assert compute_atr(closes, period=14) is None


def test_atr_handles_flat_series_with_zero_atr():
    closes = [60000.0] * 20
    assert compute_atr(closes, period=14) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_keltner_channel
# ---------------------------------------------------------------------------


def test_keltner_channel_bands_use_ema_plus_minus_atr():
    """Bands = EMA ± mult × ATR with the requested multiplier."""
    closes = [60000, 60100, 60050, 60150, 60100, 60200, 60150, 60250, 60200, 60300,
              60250, 60350, 60300, 60400, 60350, 60450, 60400, 60500, 60450, 60550,
              60500, 60600, 60550, 60650, 60600]
    result = compute_keltner_channel(closes, period=20, atr_period=14, atr_multiplier=2.0, current_spot=60600)
    assert result["ema20"] is not None
    assert result["upper"] is not None
    assert result["lower"] is not None
    # Upper - lower must equal 4*ATR (2 above, 2 below)
    spread = result["upper"] - result["lower"]
    atr = compute_atr(closes, period=14)
    assert spread == pytest.approx(4 * atr, abs=1e-6)


def test_keltner_position_above_when_spot_breaks_upper():
    closes = [60000.0] * 25  # constant → ATR=0, bands collapse to EMA
    result = compute_keltner_channel(closes, period=20, current_spot=60500)
    # Bands collapse to ema20=60000; spot=60500 > 60000 → above
    assert result["position"] == "above"


def test_keltner_position_below_when_spot_breaks_lower():
    closes = [60000.0] * 25
    result = compute_keltner_channel(closes, period=20, current_spot=59500)
    assert result["position"] == "below"


def test_keltner_position_inside_when_spot_is_inside_bands():
    closes = list(range(60000, 60030))
    result = compute_keltner_channel(closes, period=20, current_spot=60020)
    # Use whatever the bands compute to and just verify the position label is consistent.
    assert result["ema20"] is not None
    assert result["lower"] <= 60020 <= result["upper"]
    assert result["position"] == "inside"


def test_keltner_returns_none_bands_when_series_too_short():
    closes = [60000.0, 60100.0]
    result = compute_keltner_channel(closes, period=20, current_spot=60000)
    assert result["ema20"] is None
    assert result["upper"] is None
    assert result["lower"] is None
    assert result["position"] == "unknown"
