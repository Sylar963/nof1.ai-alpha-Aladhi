"""Tests for the vol surface builder.

Vol surface = a structured view of an option chain bucketed by tenor and
moneyness, with derived signals (ATM straddle expected move, 25Δ skew, term
structure slope). All tests use synthetic chains so we don't depend on a
live exchange feed."""

from datetime import date, timedelta

import pytest

from src.backend.options_intel.vol_surface import (
    VolSurface,
    build_vol_surface,
)


def _instr(name, expiry, strike, kind, mark_iv, mark_price=None, delta=None):
    """Build a synthetic Thalex-shape instrument record."""
    return {
        "instrument_name": name,
        "expiry_timestamp": int(expiry.timestamp()),
        "strike": strike,
        "option_type": kind,
        "mark_iv": mark_iv,
        "mark_price": mark_price or 0.0,
        "type": "option",
        "underlying": "BTCUSD",
        "delta": delta,
    }


@pytest.fixture
def synthetic_chain():
    """Two-tenor BTC option chain with a smile shape, ATM near 60000."""
    from datetime import datetime, timezone

    today = datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc)
    expiry_15d = datetime(2026, 4, 25, 8, 0, 0, tzinfo=timezone.utc)
    expiry_30d = datetime(2026, 5, 10, 8, 0, 0, tzinfo=timezone.utc)

    # 15-day expiry: smile bottom at ATM 60k, edges higher
    chain = [
        _instr("BTC-25APR26-50000-C", expiry_15d, 50000, "call", 0.85, 10500, 0.95),
        _instr("BTC-25APR26-55000-C", expiry_15d, 55000, "call", 0.72, 6000, 0.80),
        _instr("BTC-25APR26-60000-C", expiry_15d, 60000, "call", 0.65, 2500, 0.50),
        _instr("BTC-25APR26-65000-C", expiry_15d, 65000, "call", 0.70, 800, 0.25),
        _instr("BTC-25APR26-70000-C", expiry_15d, 70000, "call", 0.78, 200, 0.10),
        _instr("BTC-25APR26-50000-P", expiry_15d, 50000, "put", 0.88, 50, -0.05),
        _instr("BTC-25APR26-55000-P", expiry_15d, 55000, "put", 0.75, 250, -0.20),
        _instr("BTC-25APR26-60000-P", expiry_15d, 60000, "put", 0.66, 2400, -0.50),
        _instr("BTC-25APR26-65000-P", expiry_15d, 65000, "put", 0.72, 5800, -0.75),
        _instr("BTC-25APR26-70000-P", expiry_15d, 70000, "put", 0.80, 10100, -0.90),
        # 30-day expiry: similar shape, slightly higher IV (term structure contango)
        _instr("BTC-10MAY26-60000-C", expiry_30d, 60000, "call", 0.70, 3500, 0.52),
        _instr("BTC-10MAY26-60000-P", expiry_30d, 60000, "put", 0.71, 3400, -0.48),
        _instr("BTC-10MAY26-55000-P", expiry_30d, 55000, "put", 0.78, 700, -0.25),
        _instr("BTC-10MAY26-65000-C", expiry_30d, 65000, "call", 0.74, 1200, 0.30),
    ]
    return chain, today.date(), 60000.0


# ---------------------------------------------------------------------------
# build_vol_surface
# ---------------------------------------------------------------------------


def test_build_vol_surface_groups_by_tenor(synthetic_chain):
    chain, today, spot = synthetic_chain
    surface = build_vol_surface(chain, spot=spot, today=today)
    assert isinstance(surface, VolSurface)
    tenors = sorted(surface.smiles.keys())
    assert tenors == [15, 30]


def test_atm_iv_is_average_of_call_and_put_at_atm_strike(synthetic_chain):
    chain, today, spot = synthetic_chain
    surface = build_vol_surface(chain, spot=spot, today=today)
    # At spot=60000 the ATM strike is 60000.
    # 15d ATM: call IV 0.65, put IV 0.66 → mean 0.655
    assert surface.atm_iv_by_tenor[15] == pytest.approx(0.655, abs=1e-3)
    # 30d ATM: 0.70 / 0.71 → 0.705
    assert surface.atm_iv_by_tenor[30] == pytest.approx(0.705, abs=1e-3)


def test_atm_straddle_expected_move(synthetic_chain):
    chain, today, spot = synthetic_chain
    surface = build_vol_surface(chain, spot=spot, today=today)
    # 15d ATM straddle = 2500 (call) + 2400 (put) = 4900 BTC equiv → 4900/60000 ≈ 8.17%
    em_15d_pct = surface.expected_move_pct_by_tenor[15]
    assert em_15d_pct == pytest.approx(4900 / 60000, abs=1e-4)


def test_25_delta_skew_per_tenor(synthetic_chain):
    """25Δ skew = put 25Δ IV − call 25Δ IV. Positive = puts richer = bearish skew."""
    chain, today, spot = synthetic_chain
    surface = build_vol_surface(chain, spot=spot, today=today)
    # 15d: put with delta -0.20 (strike 55000) IV=0.75; call with delta 0.25 (strike 65000) IV=0.70
    skew_15d = surface.skew_25d_by_tenor[15]
    assert skew_15d == pytest.approx(0.05, abs=1e-3)


def test_term_structure_slope_positive_when_longer_tenor_has_higher_iv(synthetic_chain):
    """Slope is (long_tenor_iv − short_tenor_iv) / (long − short) days. Positive = contango."""
    chain, today, spot = synthetic_chain
    surface = build_vol_surface(chain, spot=spot, today=today)
    assert surface.term_structure_slope > 0


def test_build_vol_surface_skips_unparseable_instruments():
    """Garbage entries must be tolerated and excluded, not crash the build."""
    from datetime import datetime, timezone

    expiry = datetime(2026, 5, 10, 8, 0, 0, tzinfo=timezone.utc)
    today = date(2026, 4, 10)
    chain = [
        _instr("BTC-10MAY26-60000-C", expiry, 60000, "call", 0.65, 2500, 0.5),
        _instr("BTC-10MAY26-60000-P", expiry, 60000, "put", 0.66, 2450, -0.5),
        {"instrument_name": "GARBAGE", "type": "perpetual"},
        {"instrument_name": "BAD", "type": "option", "expiry_timestamp": "not-a-number"},
    ]
    surface = build_vol_surface(chain, spot=60000.0, today=today)
    assert 30 in surface.smiles
    assert len(surface.smiles[30]) == 2


def test_returns_empty_surface_for_empty_chain():
    surface = build_vol_surface([], spot=60000.0, today=date(2026, 4, 10))
    assert surface.smiles == {}
    assert surface.atm_iv_by_tenor == {}


def test_find_atm_15d_straddle_strike_returns_closest_to_spot(synthetic_chain):
    """Used by the regime classifier to anchor the straddle EM baseline."""
    chain, today, spot = synthetic_chain
    surface = build_vol_surface(chain, spot=spot, today=today)
    info = surface.atm_straddle_15d
    assert info is not None
    assert info["strike"] == 60000
    assert info["call_price"] == pytest.approx(2500)
    assert info["put_price"] == pytest.approx(2400)
    assert info["lower_strike"] == pytest.approx(60000 - 4900)
    assert info["upper_strike"] == pytest.approx(60000 + 4900)
