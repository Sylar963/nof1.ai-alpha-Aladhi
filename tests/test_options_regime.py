"""Tests for the vol regime classifier.

The classifier combines two signals into a single label:

- Signal 1: ATM 15-day straddle expected-move test (your heuristic)
    - current_spot < lower_strike_15d_ago → 'cheap'
    - current_spot > upper_strike_15d_ago → 'rich'
    - inside the range                    → 'fair'

- Signal 2: realized vs implied vol ratio
    - RV/IV > 1.2 → 'cheap'   (real moves outran implied)
    - RV/IV < 0.8 → 'rich'    (implied was bigger than realized)
    - in between  → 'fair'

Combined:
    - both agree → use that label, confidence='high'
    - disagree   → use Signal 2 (more standard), confidence='low'
    - either input missing → 'unknown'
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.backend.options_intel.iv_history_store import IVHistoryRow, IVHistoryStore
from src.backend.options_intel.regime import (
    RegimeReading,
    classify_regime,
    realized_vol_close_to_close,
)


def _row(spot, lower, upper, age_days, atm_iv=0.65, em=0.08, tenor=15, now=None):
    now = now or datetime.now(timezone.utc)
    return IVHistoryRow(
        ts=now - timedelta(days=age_days),
        tenor_days=tenor,
        atm_iv=atm_iv,
        atm_straddle_em=em,
        spot_at_init=spot,
        lower_strike=lower,
        upper_strike=upper,
    )


# ---------------------------------------------------------------------------
# Signal 1 — straddle expected-move test
# ---------------------------------------------------------------------------


def test_classify_cheap_when_spot_below_historical_lower(tmp_path):
    """Spot dropped below the implied range from 15 days ago → 'cheap' (per user heuristic)."""
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))
    now = datetime.now(timezone.utc)
    store.write(_row(spot=60000, lower=58000, upper=62000, age_days=15, now=now))

    reading = classify_regime(
        store=store,
        current_spot=57000.0,
        current_atm_iv_15d=0.65,
        spot_history=[60000.0] * 16,  # flat → RV ≈ 0 → unknown for signal 2
        now=now,
    )
    assert reading.signal_1_label == "cheap"


def test_classify_rich_when_spot_above_historical_upper(tmp_path):
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))
    now = datetime.now(timezone.utc)
    store.write(_row(spot=60000, lower=58000, upper=62000, age_days=15, now=now))

    reading = classify_regime(
        store=store,
        current_spot=63000.0,
        current_atm_iv_15d=0.65,
        spot_history=[60000.0] * 16,
        now=now,
    )
    assert reading.signal_1_label == "rich"


def test_classify_fair_when_spot_inside_historical_range(tmp_path):
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))
    now = datetime.now(timezone.utc)
    store.write(_row(spot=60000, lower=58000, upper=62000, age_days=15, now=now))

    reading = classify_regime(
        store=store,
        current_spot=60500.0,
        current_atm_iv_15d=0.65,
        spot_history=[60000.0] * 16,
        now=now,
    )
    assert reading.signal_1_label == "fair"


# ---------------------------------------------------------------------------
# Signal 2 — RV/IV ratio
# ---------------------------------------------------------------------------


def test_classify_cheap_when_realized_vol_outran_implied(tmp_path):
    """RV/IV > 1.2 → vol was UNDERPRICED → 'cheap'."""
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))
    now = datetime.now(timezone.utc)
    # Construct a spot history with high realized vol (~10% daily moves)
    spot_history = [60000 * (1.1 if i % 2 == 0 else 0.9) for i in range(16)]

    reading = classify_regime(
        store=store,
        current_spot=60000.0,
        current_atm_iv_15d=0.30,  # implied is low
        spot_history=spot_history,
        now=now,
    )
    assert reading.signal_2_label == "cheap"


def test_classify_rich_when_realized_vol_below_implied(tmp_path):
    """RV/IV < 0.8 → market priced more move than happened → 'rich'."""
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))
    # Flat price history → RV ≈ 0 → ratio ≈ 0 → 'rich'
    spot_history = [60000.0] * 16

    reading = classify_regime(
        store=store,
        current_spot=60000.0,
        current_atm_iv_15d=1.50,  # absurdly high implied
        spot_history=spot_history,
    )
    assert reading.signal_2_label == "rich"


# ---------------------------------------------------------------------------
# Combined label
# ---------------------------------------------------------------------------


def test_combined_label_high_confidence_when_signals_agree(tmp_path):
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))
    now = datetime.now(timezone.utc)
    store.write(_row(spot=60000, lower=58000, upper=62000, age_days=15, now=now))

    # Spot below lower (signal 1 = cheap) AND high RV (signal 2 = cheap)
    spot_history = [60000 * (1.08 if i % 2 == 0 else 0.92) for i in range(16)]
    reading = classify_regime(
        store=store,
        current_spot=57000.0,
        current_atm_iv_15d=0.30,
        spot_history=spot_history,
        now=now,
    )
    assert reading.vol_regime == "cheap"
    assert reading.confidence == "high"


def test_combined_label_low_confidence_when_signals_disagree(tmp_path):
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))
    now = datetime.now(timezone.utc)
    store.write(_row(spot=60000, lower=58000, upper=62000, age_days=15, now=now))

    # Spot above upper (signal 1 = rich) but flat history (signal 2 = rich) → both rich
    # We need disagreement: signal 1 = rich, signal 2 = cheap
    spot_history = [60000 * (1.08 if i % 2 == 0 else 0.92) for i in range(16)]  # high RV
    reading = classify_regime(
        store=store,
        current_spot=63000.0,  # signal 1 = rich
        current_atm_iv_15d=0.30,  # signal 2 = cheap
        spot_history=spot_history,
        now=now,
    )
    assert reading.signal_1_label == "rich"
    assert reading.signal_2_label == "cheap"
    # Disagreement → use signal 2 (the more standard signal), low confidence
    assert reading.vol_regime == "cheap"
    assert reading.confidence == "low"


def test_combined_label_unknown_when_history_empty(tmp_path):
    """No persisted anchor → signal 1 unavailable → unknown overall."""
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))
    reading = classify_regime(
        store=store,
        current_spot=60000.0,
        current_atm_iv_15d=0.65,
        spot_history=[60000.0] * 16,
    )
    assert reading.vol_regime == "unknown"


# ---------------------------------------------------------------------------
# realized_vol_close_to_close helper
# ---------------------------------------------------------------------------


def test_realized_vol_zero_for_flat_series():
    rv = realized_vol_close_to_close([60000.0] * 20, periods_per_year=365)
    assert rv == pytest.approx(0.0, abs=1e-9)


def test_realized_vol_positive_for_volatile_series():
    series = [60000 * (1.05 if i % 2 == 0 else 0.95) for i in range(30)]
    rv = realized_vol_close_to_close(series, periods_per_year=365)
    assert rv > 0.5  # very high (synthetic ±5% daily)
