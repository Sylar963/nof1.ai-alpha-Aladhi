"""Black-Scholes greeks for BTC options — local fallback.

Thalex ticker exposes only ``iv`` and ``delta`` at the top level. Gamma,
vega, and theta are not published, so we compute them locally from
spot + strike + iv + time. r=0 is a reasonable approximation for crypto
options (no carry, no dividend).

Conventions match Deribit so the LLM sees consistent numbers:
  - ``vega`` is per 1% IV move  (i.e. spot * pdf(d1) * sqrt(T) / 100)
  - ``theta`` is per calendar day (annual / 365)
  - ``delta`` is signed: call in [0, 1], put in [-1, 0]
"""

from __future__ import annotations

import math
from typing import Optional


_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def black_scholes_greeks(
    *,
    spot: float,
    strike: float,
    iv: float,
    time_years: float,
    kind: str,
    r: float = 0.0,
) -> dict:
    """Return ``{delta, gamma, vega, theta}`` for a European BTC option.

    All four keys are always present; degenerate inputs (non-positive spot/
    strike/iv/time) yield zeros so callers can merge unconditionally.
    """
    if spot <= 0 or strike <= 0 or iv <= 0 or time_years <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    vol_sqrt_t = iv * math.sqrt(time_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * time_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    pdf_d1 = _norm_pdf(d1)
    disc = math.exp(-r * time_years)

    gamma = pdf_d1 / (spot * vol_sqrt_t)
    vega = spot * pdf_d1 * math.sqrt(time_years) / 100.0

    if kind == "call":
        delta = _norm_cdf(d1)
        theta_year = -(spot * pdf_d1 * iv) / (2.0 * math.sqrt(time_years)) - r * strike * disc * _norm_cdf(d2)
    else:
        delta = _norm_cdf(d1) - 1.0
        theta_year = -(spot * pdf_d1 * iv) / (2.0 * math.sqrt(time_years)) + r * strike * disc * _norm_cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta_year / 365.0,
    }


def days_between(expiry_seconds: int, now_seconds: int) -> float:
    """Calendar days between two UTC timestamps (seconds)."""
    return max(0.0, (expiry_seconds - now_seconds) / 86400.0)


def years_between(expiry_seconds: int, now_seconds: int) -> float:
    """Calendar years (for the T factor in BS)."""
    return max(0.0, (expiry_seconds - now_seconds) / (365.0 * 86400.0))


def infer_kind(instrument_name: Optional[str], explicit_kind: Optional[str] = None) -> Optional[str]:
    """Best-effort kind resolution: explicit hint first, then instrument-name suffix."""
    if explicit_kind in ("call", "put"):
        return explicit_kind
    if not isinstance(instrument_name, str):
        return None
    suffix = instrument_name.rsplit("-", 1)[-1].upper()
    if suffix == "C":
        return "call"
    if suffix == "P":
        return "put"
    return None
