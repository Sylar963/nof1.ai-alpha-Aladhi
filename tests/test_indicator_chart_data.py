"""Contract tests for the data backing the Open Range & Keltner chart.

The indicator chart at ``src/gui/pages/market.py`` plots:
  * 15m candles (up to 2000 bars / ~21 days) from the ``chart_intraday``
    frame.
  * Keltner (130, 130, 4) bands aligned to those candles.
  * Open Range high/low horizontal lines.
  * Underlying close price.

These tests pin the backend contract that the chart consumes: a
``chart_intraday`` frame with up to 2000 bars of 15m price_candles,
Keltner sized to ``n - 130`` after warmup, and an ``opening_range`` that
always carries ``or_start_ms`` for the bot to use.
"""

from datetime import datetime, timezone

from src.backend.indicators.indicator_engine import (
    build_indicator_bundle,
    to_price_candles,
)

FIVE_MIN_MS = 5 * 60 * 1000
FIFTEEN_MIN_MS = 15 * 60 * 1000


def _make_candles(n: int, anchor_ms: int, step_ms: int,
                  base_price: float = 76000.0) -> list[dict]:
    candles = []
    for i in range(n):
        t = anchor_ms + i * step_ms
        price = base_price + i * 0.5
        candles.append({
            "t": t,
            "o": price,
            "h": price + 5.0,
            "l": price - 5.0,
            "c": price + 1.0,
            "v": 1.0,
        })
    return candles


def test_to_price_candles_caps_at_max_bars():
    """Slicing pins the most recent N bars regardless of input length."""
    anchor = int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp() * 1000)
    candles = _make_candles(2500, anchor, FIFTEEN_MIN_MS)

    out = to_price_candles(candles, max_bars=2000)

    assert len(out["time"]) == 2000
    assert len(out["close"]) == 2000
    # The slice keeps the most recent 2000 (drops the oldest 500).
    assert out["time"][0] == candles[500]["t"]
    assert out["time"][-1] == candles[-1]["t"]


def test_indicator_bundle_builds_chart_intraday_when_15m_supplied():
    anchor = int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp() * 1000)
    candles_5m = _make_candles(200, anchor, FIVE_MIN_MS)
    candles_15m = _make_candles(2130, anchor, FIFTEEN_MIN_MS)
    candles_long = _make_candles(200, anchor, FIFTEEN_MIN_MS)

    bundle = build_indicator_bundle(
        candles_5m=candles_5m,
        candles_daily=[],
        candles_long=candles_long,
        long_interval="4h",
        candles_15m=candles_15m,
        current_spot=candles_5m[-1]["c"],
    )

    assert "chart_intraday" in bundle
    frame = bundle["chart_intraday"]
    assert frame["interval"] == "15m"

    price = frame["price_candles"]
    assert len(price["close"]) == 2000
    assert len(price["time"]) == 2000

    # Keltner valid bars = n - period. With 2130 input candles, Keltner has
    # 2000 valid bars, lining up 1:1 with the 2000 displayed candles.
    keltner_middle = frame["keltner"]["middle"]
    assert len(keltner_middle) == 2000

    or_block = frame["opening_range"]
    assert or_block["or_start_ms"] is not None
    assert or_block["or_end_ms"] - or_block["or_start_ms"] == 15 * 60 * 1000


def test_indicator_bundle_skips_chart_frame_when_15m_omitted():
    """TAAPI fallback path doesn't supply 15m candles; build_indicator_bundle
    must not crash and must not emit a half-built chart_intraday frame."""
    anchor = int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp() * 1000)
    candles_5m = _make_candles(200, anchor, FIVE_MIN_MS)
    candles_long = _make_candles(200, anchor, FIFTEEN_MIN_MS)

    bundle = build_indicator_bundle(
        candles_5m=candles_5m,
        candles_daily=[],
        candles_long=candles_long,
        long_interval="4h",
        current_spot=candles_5m[-1]["c"],
    )

    assert "chart_intraday" not in bundle
    assert "5m" in bundle


def test_chart_frame_keltner_pads_to_match_displayed_candles():
    """If we fetch only 2000 15m candles (no warmup buffer), Keltner has
    n - 130 valid bars and the renderer pads the missing 130 with None.
    """
    anchor = int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp() * 1000)
    candles_15m = _make_candles(2000, anchor, FIFTEEN_MIN_MS)

    bundle = build_indicator_bundle(
        candles_5m=_make_candles(200, anchor, FIVE_MIN_MS),
        candles_daily=[],
        candles_long=_make_candles(200, anchor, FIFTEEN_MIN_MS),
        long_interval="4h",
        candles_15m=candles_15m,
        current_spot=candles_15m[-1]["c"],
    )

    frame = bundle["chart_intraday"]
    n_price = len(frame["price_candles"]["close"])
    n_keltner = len(frame["keltner"]["middle"])

    assert n_price == 2000
    assert n_keltner == n_price - 130
