"""Black-Scholes-Merton pricing engine for European options.

Pure functions, zero external dependencies — just the standard library. The
formulas follow the standard textbook presentation (Hull, Options Futures
and Other Derivatives) with continuous dividend yield ``q``:

    d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    Call = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
    Put  = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)

Greeks follow the same convention with ``vega`` quoted per 1.00 change in
sigma (multiply by 0.01 for "per vol point") and ``theta`` quoted per year
(divide by 365 for "per day").

Implied vol uses Newton-Raphson on the price equation with vega as the
derivative; falls back to bisection bounds if Newton wanders out of range.
"""

from __future__ import annotations

import math
from typing import Literal


Kind = Literal["call", "put"]


# ---------------------------------------------------------------------------
# Standard normal CDF / PDF
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def _validate_kind(kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _intrinsic(S: float, K: float, kind: Kind) -> float:
    if kind == "call":
        return max(S - K, 0.0)
    return max(K - S, 0.0)


def _intrinsic_pv(S: float, K: float, T: float, r: float, q: float, kind: Kind) -> float:
    """Present value of intrinsic at expiry given the deterministic forward."""
    forward_pv_diff = S * math.exp(-q * T) - K * math.exp(-r * T)
    if kind == "call":
        return max(forward_pv_diff, 0.0)
    return max(-forward_pv_diff, 0.0)


def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: Kind = "call",
) -> float:
    """Black-Scholes-Merton European option price.

    Args:
        S: spot price.
        K: strike.
        T: time to expiry in years (e.g. 30/365 for 30 days).
        r: continuously-compounded risk-free rate.
        sigma: implied volatility (annualized, decimal — 0.65 for 65%).
        q: continuous dividend / carry yield. Use 0.0 for crypto unless
            modelling funding-rate carry explicitly.
        kind: ``"call"`` or ``"put"``.

    Returns:
        Theoretical present-value price of the option.
    """
    _validate_kind(kind)
    if T <= 0.0:
        return _intrinsic(S, K, kind)
    if sigma <= 0.0:
        return _intrinsic_pv(S, K, T, r, q, kind)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount_q = math.exp(-q * T)
    discount_r = math.exp(-r * T)

    if kind == "call":
        return S * discount_q * _norm_cdf(d1) - K * discount_r * _norm_cdf(d2)
    return K * discount_r * _norm_cdf(-d2) - S * discount_q * _norm_cdf(-d1)


def bsm_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: Kind = "call",
) -> dict:
    """Return ``{delta, gamma, vega, theta, rho}`` for a European option.

    Conventions:
        - ``vega`` is per 1.00 change in sigma. Multiply by 0.01 for "per
          vol point" if you prefer.
        - ``theta`` is per year. Divide by 365 for "per day" if you prefer.
        - ``rho`` is per 1.00 change in r.

    At ``T=0`` or ``sigma=0`` returns degenerate but finite values: delta is
    a step at the strike, gamma/vega/theta are zero.
    """
    _validate_kind(kind)
    if T <= 0.0 or sigma <= 0.0:
        if kind == "call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount_q = math.exp(-q * T)
    discount_r = math.exp(-r * T)
    pdf_d1 = _norm_pdf(d1)

    gamma = discount_q * pdf_d1 / (S * sigma * sqrt_T)
    vega = S * discount_q * pdf_d1 * sqrt_T

    if kind == "call":
        delta = discount_q * _norm_cdf(d1)
        theta = (
            -(S * discount_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            - r * K * discount_r * _norm_cdf(d2)
            + q * S * discount_q * _norm_cdf(d1)
        )
        rho = K * T * discount_r * _norm_cdf(d2)
    else:
        delta = -discount_q * _norm_cdf(-d1)
        theta = (
            -(S * discount_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            + r * K * discount_r * _norm_cdf(-d2)
            - q * S * discount_q * _norm_cdf(-d1)
        )
        rho = -K * T * discount_r * _norm_cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ---------------------------------------------------------------------------
# Implied volatility inversion
# ---------------------------------------------------------------------------


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    kind: Kind = "call",
    *,
    initial_guess: float = 0.5,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> float:
    """Invert the BSM price equation to recover implied volatility.

    Newton-Raphson on the price residual using vega as the derivative. If
    Newton wanders out of (sigma_min, sigma_max) we fall back to bisection.

    Returns ``float('nan')`` when the requested price is unachievable
    (e.g. below intrinsic, above forward, or T<=0).
    """
    _validate_kind(kind)
    if T <= 0.0:
        return float("nan")

    intrinsic_pv = _intrinsic_pv(S, K, T, r, q, kind)
    if price < intrinsic_pv - 1e-9:
        return float("nan")

    sigma_min = 1e-6
    sigma_max = 5.0  # 500% IV upper bound — generous for crypto
    upper_price = bsm_price(S, K, T, r, sigma_max, q, kind)
    if price > upper_price + 1e-9:
        return float("nan")

    sigma = max(sigma_min, min(sigma_max, initial_guess))
    for _ in range(max_iter):
        residual = bsm_price(S, K, T, r, sigma, q, kind) - price
        if abs(residual) < tol:
            return sigma
        vega = bsm_greeks(S, K, T, r, sigma, q, kind)["vega"]
        if vega < 1e-12:
            break  # vega too small for Newton — fall through to bisection
        next_sigma = sigma - residual / vega
        if next_sigma < sigma_min or next_sigma > sigma_max or math.isnan(next_sigma):
            break
        sigma = next_sigma

    # Bisection fallback. We re-anchor on (sigma_min, sigma_max) and squeeze.
    lo, hi = sigma_min, sigma_max
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        price_mid = bsm_price(S, K, T, r, mid, q, kind)
        if abs(price_mid - price) < tol:
            return mid
        if price_mid < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
