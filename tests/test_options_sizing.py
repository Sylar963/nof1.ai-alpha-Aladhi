"""Tests for the multi-tenor gamma distribution helper.

Given a target gamma exposure (in BTC) and a list of tenors, the helper
picks the ATM contract per tenor and computes how many contracts to buy
in each so that the equal-weighted total gamma matches the target. All
contract counts respect the min increment (0.001) and the per-trade cap
(0.1)."""

import pytest

from src.backend.options_intel.sizing import (
    LegSizing,
    distribute_target_gamma,
)
from src.backend.options_intel.vol_surface import SmileEntry, VolSurface


def _entry(strike, kind, iv, delta, price, name=None):
    return SmileEntry(
        instrument_name=name or f"BTC-X-{int(strike)}-{kind[0].upper()}",
        strike=strike,
        kind=kind,
        iv=iv,
        price=price,
        delta=delta,
    )


def _build_surface(spot=60000.0):
    """Synthetic surface with three tenors and a uniform 0.05 gamma at ATM."""
    smiles = {
        7: [
            _entry(60000, "call", 0.65, 0.50, 1500, name="BTC-7D-60000-C"),
            _entry(60000, "put", 0.66, -0.50, 1450, name="BTC-7D-60000-P"),
        ],
        14: [
            _entry(60000, "call", 0.66, 0.50, 2100, name="BTC-14D-60000-C"),
            _entry(60000, "put", 0.67, -0.50, 2050, name="BTC-14D-60000-P"),
        ],
        30: [
            _entry(60000, "call", 0.68, 0.50, 3000, name="BTC-30D-60000-C"),
            _entry(60000, "put", 0.69, -0.50, 2950, name="BTC-30D-60000-P"),
        ],
    }
    return VolSurface(spot=spot, today=None, smiles=smiles)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Equal-gamma distribution
# ---------------------------------------------------------------------------


def test_distribute_equal_split_three_tenors():
    """Target 0.0030 BTC of gamma split across 3 tenors → 0.001 per tenor."""
    surface = _build_surface()
    legs = distribute_target_gamma(
        target_btc=0.003,
        tenors=[7, 14, 30],
        surface=surface,
        atm_gamma_estimate=1.0,  # 1 BTC of gamma per contract → 0.001 contracts per tenor
        kind="call",
    )
    assert len(legs) == 3
    for leg in legs:
        assert leg.contracts == pytest.approx(0.001)
    assert sum(leg.contracts for leg in legs) == pytest.approx(0.003)


def test_distribute_clamps_to_min_contract():
    """Target so small that per-tenor share < 0.001 must round UP to 0.001."""
    surface = _build_surface()
    legs = distribute_target_gamma(
        target_btc=0.0001,  # tiny
        tenors=[7, 14, 30],
        surface=surface,
        atm_gamma_estimate=1.0,
        kind="call",
    )
    assert all(leg.contracts >= 0.001 for leg in legs)


def test_distribute_clamps_to_max_per_trade():
    """Target so large that per-tenor share > 0.1 must clamp to 0.1."""
    surface = _build_surface()
    legs = distribute_target_gamma(
        target_btc=10.0,  # huge
        tenors=[7, 14, 30],
        surface=surface,
        atm_gamma_estimate=1.0,
        kind="call",
        max_per_trade=0.1,
    )
    assert all(leg.contracts <= 0.1 for leg in legs)


def test_distribute_skips_tenor_with_no_atm_contract():
    """If a tenor has no matching ATM contract for the kind, it's dropped."""
    surface = _build_surface()
    surface.smiles[7] = [_entry(60000, "put", 0.66, -0.50, 1450, name="BTC-7D-60000-P")]  # no call!

    legs = distribute_target_gamma(
        target_btc=0.002,
        tenors=[7, 14, 30],
        surface=surface,
        atm_gamma_estimate=1.0,
        kind="call",
    )
    assert len(legs) == 2
    assert {leg.tenor_days for leg in legs} == {14, 30}


def test_distribute_picks_correct_kind():
    surface = _build_surface()
    legs = distribute_target_gamma(
        target_btc=0.002,
        tenors=[7, 14],
        surface=surface,
        atm_gamma_estimate=1.0,
        kind="put",
    )
    assert all(leg.kind == "put" for leg in legs)
    for leg in legs:
        assert leg.instrument_name.endswith("-P")


def test_distribute_returns_empty_when_surface_empty():
    surface = VolSurface(spot=60000.0, today=None, smiles={})  # type: ignore[arg-type]
    legs = distribute_target_gamma(
        target_btc=0.002,
        tenors=[7, 14],
        surface=surface,
        atm_gamma_estimate=1.0,
        kind="call",
    )
    assert legs == []


def test_leg_sizing_dataclass_fields():
    sizing = LegSizing(
        instrument_name="BTC-7D-60000-C",
        tenor_days=7,
        kind="call",
        contracts=0.001,
    )
    assert sizing.instrument_name == "BTC-7D-60000-C"
    assert sizing.contracts == 0.001
