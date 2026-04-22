"""Nearest/interpolated ATM IV lookup for a target tenor.

BTC option expiries cluster on Fridays, so on most calendar days the
surface has no bucket exactly equal to the regime target. Without a
neighbor-aware lookup the regime classifier's Signal 2 input is always
0 and the LLM keeps seeing ``vol_regime=unknown`` forever.
"""

from __future__ import annotations

import pytest

from src.backend.options_intel.vol_surface import (
    atm_iv_for_target_tenor,
    _atm_straddle_anchor,
    SmileEntry,
)


def test_exact_tenor_hit_returns_value():
    assert atm_iv_for_target_tenor({30: 0.62}, 30) == pytest.approx(0.62)


def test_neighbors_on_both_sides_interpolate_in_total_variance():
    """28d at 0.60 and 35d at 0.70 → 30d interpolation uses total-variance linear."""
    out = atm_iv_for_target_tenor({28: 0.60, 35: 0.70}, 30)
    assert out is not None
    assert 0.55 < out < 0.70


def test_only_nearby_below_returns_that_iv():
    assert atm_iv_for_target_tenor({25: 0.55}, 30) == pytest.approx(0.55)


def test_only_nearby_above_returns_that_iv():
    assert atm_iv_for_target_tenor({35: 0.60}, 30) == pytest.approx(0.60)


def test_no_tenor_within_band_returns_none():
    assert atm_iv_for_target_tenor({2: 0.40, 90: 0.80}, 30) is None


def test_empty_dict_returns_none():
    assert atm_iv_for_target_tenor({}, 30) is None


def test_zero_ivs_are_skipped():
    assert atm_iv_for_target_tenor({30: 0.0, 31: 0.62}, 30) == pytest.approx(0.62)


def test_anchor_records_bucket_tenor_not_actual_tenor():
    """The persisted ``tenor_days`` must be the target bucket (30) so
    ``IVHistoryStore.lookback(tenor_days=30)`` finds it 30 days later."""
    smiles = {
        31: [
            SmileEntry(instrument_name="BTC-A-80000-C", strike=80000, kind="call", iv=0.60, price=1500),
            SmileEntry(instrument_name="BTC-A-80000-P", strike=80000, kind="put", iv=0.60, price=1400),
        ]
    }
    anchor = _atm_straddle_anchor(smiles, spot=80000.0, target_tenor=30)
    assert anchor is not None
    assert anchor["tenor_days"] == 30
    assert anchor["actual_tenor_days"] == 31


def test_anchor_is_none_when_no_tenor_close_to_target():
    smiles = {
        2: [
            SmileEntry(instrument_name="BTC-A-80000-C", strike=80000, kind="call", iv=0.60, price=1500),
            SmileEntry(instrument_name="BTC-A-80000-P", strike=80000, kind="put", iv=0.60, price=1400),
        ],
        90: [
            SmileEntry(instrument_name="BTC-B-80000-C", strike=80000, kind="call", iv=0.60, price=3500),
            SmileEntry(instrument_name="BTC-B-80000-P", strike=80000, kind="put", iv=0.60, price=3400),
        ],
    }
    assert _atm_straddle_anchor(smiles, spot=80000.0, target_tenor=30) is None


def test_anchor_honors_max_offset_kwarg():
    smiles = {
        36: [
            SmileEntry(instrument_name="BTC-A-80000-C", strike=80000, kind="call", iv=0.60, price=1500),
            SmileEntry(instrument_name="BTC-A-80000-P", strike=80000, kind="put", iv=0.60, price=1400),
        ]
    }
    assert _atm_straddle_anchor(smiles, spot=80000.0, target_tenor=30, max_offset_days=5) is None
    assert _atm_straddle_anchor(smiles, spot=80000.0, target_tenor=30, max_offset_days=10) is not None
