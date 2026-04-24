"""Black-Scholes greeks smoke tests.

We're not trying to re-validate BS against a reference library — just lock
the sign + magnitude behavior the portfolio aggregator and the options
LLM rely on. Spot=80000, IV=60%, 30 DTE is a representative BTC option.
"""

from __future__ import annotations

import math

from src.backend.options_intel.greeks_bs import (
    black_scholes_greeks,
    infer_kind,
    years_between,
)


def test_atm_call_has_positive_delta_gamma_vega_and_negative_theta():
    g = black_scholes_greeks(
        spot=80000.0, strike=80000.0, iv=0.60, time_years=30 / 365.0, kind="call",
    )
    assert 0.4 < g["delta"] < 0.6, "ATM call delta should sit near 0.5"
    assert g["gamma"] > 0
    assert g["vega"] > 0
    assert g["theta"] < 0, "long option theta is negative (premium decays)"


def test_atm_put_delta_is_negative_and_near_minus_half():
    g = black_scholes_greeks(
        spot=80000.0, strike=80000.0, iv=0.60, time_years=30 / 365.0, kind="put",
    )
    assert -0.6 < g["delta"] < -0.4


def test_deep_itm_call_delta_approaches_one():
    g = black_scholes_greeks(
        spot=80000.0, strike=40000.0, iv=0.60, time_years=30 / 365.0, kind="call",
    )
    assert g["delta"] > 0.97


def test_deep_otm_call_has_tiny_gamma_and_small_vega():
    g = black_scholes_greeks(
        spot=80000.0, strike=160000.0, iv=0.60, time_years=30 / 365.0, kind="call",
    )
    assert g["delta"] < 0.05
    assert g["gamma"] > 0
    assert g["vega"] > 0


def test_degenerate_inputs_return_zeros():
    for args in (
        {"spot": 0, "strike": 80000, "iv": 0.6, "time_years": 0.1, "kind": "call"},
        {"spot": 80000, "strike": 0, "iv": 0.6, "time_years": 0.1, "kind": "call"},
        {"spot": 80000, "strike": 80000, "iv": 0, "time_years": 0.1, "kind": "call"},
        {"spot": 80000, "strike": 80000, "iv": 0.6, "time_years": 0, "kind": "call"},
    ):
        g = black_scholes_greeks(**args)
        assert g == {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}


def test_vega_is_per_one_percent_iv_move():
    """Vega convention: a 1-vol-point change (1% IV) ≈ mark price change by vega."""
    g = black_scholes_greeks(
        spot=80000.0, strike=80000.0, iv=0.60, time_years=30 / 365.0, kind="call",
    )
    # spot * pdf(0.5*sigma*sqrt(T)) * sqrt(T) / 100 — rough sanity band
    assert 80.0 < g["vega"] < 200.0, f"vega out of expected band: {g['vega']}"


def test_infer_kind_from_instrument_suffix():
    assert infer_kind("BTC-29MAY26-90000-C") == "call"
    assert infer_kind("BTC-29MAY26-90000-P") == "put"
    assert infer_kind(None) is None
    assert infer_kind("garbage") is None
    assert infer_kind("x", explicit_kind="put") == "put"


def test_years_between_boundary():
    assert years_between(1000, 1000) == 0.0
    assert math.isclose(
        years_between(1000 + 365 * 86400, 1000), 1.0, abs_tol=1e-9,
    )
