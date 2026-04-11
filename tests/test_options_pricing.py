"""Tests for the Black-Scholes-Merton pricing engine.

These tests lock down the math against well-known reference values from the
options-pricing literature, then exercise the greeks and the implied-vol
inversion. The pricing module has zero external dependencies — pure math —
so these tests are fast and deterministic."""

import math

import pytest

from src.backend.options_intel.pricing import (
    bsm_greeks,
    bsm_price,
    implied_vol,
)


# ---------------------------------------------------------------------------
# Reference price tests
# ---------------------------------------------------------------------------
#
# Reference values come from the standard Black-Scholes example used in most
# textbooks (e.g. Hull, Options Futures and Other Derivatives):
#
#   S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.20, q = 0.0
#   Call ≈ 10.4506
#   Put  ≈ 5.5735


def test_bsm_call_atm_one_year():
    price = bsm_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0, kind="call")
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_bsm_put_atm_one_year():
    price = bsm_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0, kind="put")
    assert price == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity_holds():
    """C - P = S e^{-qT} - K e^{-rT}  must hold for any inputs."""
    S, K, T, r, sigma, q = 50000.0, 60000.0, 0.25, 0.04, 0.65, 0.0
    call = bsm_price(S, K, T, r, sigma, q, "call")
    put = bsm_price(S, K, T, r, sigma, q, "put")
    parity = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert (call - put) == pytest.approx(parity, abs=1e-6)


def test_deep_itm_call_approaches_intrinsic_minus_pv_strike():
    """A deep-ITM call should price close to S − K·e^{-rT}."""
    S, K, T, r, sigma, q = 100.0, 50.0, 0.5, 0.05, 0.20, 0.0
    price = bsm_price(S, K, T, r, sigma, q, "call")
    intrinsic_pv = S - K * math.exp(-r * T)
    assert price >= intrinsic_pv  # always at least intrinsic
    assert price == pytest.approx(intrinsic_pv, rel=0.01)


def test_deep_otm_call_is_near_zero():
    S, K, T, r, sigma, q = 100.0, 200.0, 0.1, 0.05, 0.20, 0.0
    price = bsm_price(S, K, T, r, sigma, q, "call")
    assert 0.0 <= price < 0.05


def test_zero_time_to_expiry_returns_intrinsic():
    """At T=0 a call is worth max(S-K, 0) and a put is max(K-S, 0)."""
    assert bsm_price(110.0, 100.0, 0.0, 0.05, 0.30, 0.0, "call") == pytest.approx(10.0)
    assert bsm_price(90.0, 100.0, 0.0, 0.05, 0.30, 0.0, "call") == pytest.approx(0.0)
    assert bsm_price(90.0, 100.0, 0.0, 0.05, 0.30, 0.0, "put") == pytest.approx(10.0)
    assert bsm_price(110.0, 100.0, 0.0, 0.05, 0.30, 0.0, "put") == pytest.approx(0.0)


def test_zero_volatility_returns_intrinsic_pv():
    """sigma=0 collapses to a deterministic forward — call = max(S − K·e^{-rT}, 0)."""
    S, K, T, r = 100.0, 90.0, 1.0, 0.05
    expected_call = max(S - K * math.exp(-r * T), 0.0)
    expected_put = max(K * math.exp(-r * T) - S, 0.0)
    assert bsm_price(S, K, T, r, 0.0, 0.0, "call") == pytest.approx(expected_call, abs=1e-6)
    assert bsm_price(S, K, T, r, 0.0, 0.0, "put") == pytest.approx(expected_put, abs=1e-6)


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        bsm_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "straddle")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Greeks tests
# ---------------------------------------------------------------------------


def test_atm_call_delta_is_about_half():
    """For an ATM call with no dividends, delta ≈ N(d1) ≈ 0.5–0.6."""
    g = bsm_greeks(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0, kind="call")
    assert 0.5 < g["delta"] < 0.7


def test_atm_put_delta_is_negative_and_about_minus_half():
    g = bsm_greeks(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0, kind="put")
    assert -0.5 < g["delta"] < -0.3


def test_call_and_put_share_the_same_gamma_at_same_strike():
    call_g = bsm_greeks(100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "call")
    put_g = bsm_greeks(100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "put")
    assert call_g["gamma"] == pytest.approx(put_g["gamma"], abs=1e-9)


def test_call_and_put_share_the_same_vega_at_same_strike():
    call_g = bsm_greeks(100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "call")
    put_g = bsm_greeks(100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "put")
    assert call_g["vega"] == pytest.approx(put_g["vega"], abs=1e-9)


def test_call_theta_is_negative():
    """Long calls bleed time premium — theta is negative for the holder."""
    g = bsm_greeks(100.0, 100.0, 0.5, 0.05, 0.30, 0.0, "call")
    assert g["theta"] < 0


def test_greeks_dict_has_all_expected_keys():
    g = bsm_greeks(100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "call")
    assert set(g.keys()) == {"delta", "gamma", "vega", "theta", "rho"}


# ---------------------------------------------------------------------------
# Implied volatility inversion
# ---------------------------------------------------------------------------


def test_implied_vol_recovers_input_sigma_for_call():
    """Pricing forward then inverting must round-trip to the original sigma."""
    S, K, T, r, q, sigma_in = 100.0, 105.0, 0.5, 0.04, 0.0, 0.35
    price = bsm_price(S, K, T, r, sigma_in, q, "call")
    sigma_out = implied_vol(price=price, S=S, K=K, T=T, r=r, q=q, kind="call")
    assert sigma_out == pytest.approx(sigma_in, abs=1e-4)


def test_implied_vol_recovers_input_sigma_for_put():
    S, K, T, r, q, sigma_in = 100.0, 95.0, 0.25, 0.03, 0.0, 0.55
    price = bsm_price(S, K, T, r, sigma_in, q, "put")
    sigma_out = implied_vol(price=price, S=S, K=K, T=T, r=r, q=q, kind="put")
    assert sigma_out == pytest.approx(sigma_in, abs=1e-4)


def test_implied_vol_returns_nan_for_unachievable_price():
    """A price below intrinsic is unachievable — must surface a NaN, not a wrong number."""
    result = implied_vol(price=0.001, S=100.0, K=50.0, T=1.0, r=0.05, q=0.0, kind="call")
    assert math.isnan(result)
