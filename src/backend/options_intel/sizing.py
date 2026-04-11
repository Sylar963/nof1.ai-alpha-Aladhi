"""Multi-tenor gamma distribution helper.

Given a target gamma exposure (in BTC) and a list of tenors, distribute that
exposure equally across the tenors at the ATM strike of each tenor. Each
per-tenor contract count is clamped to ``[min_contract, max_per_trade]`` and
rounded to the nearest min-contract increment.

Equal-weighted by design — easy to reason about and matches the user's
"buy across tenors to build gamma" intent. Future iterations can weight by
implied vol term structure or by liquidity once we have data on what the
real fills look like.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.backend.options_intel.vol_surface import SmileEntry, VolSurface


@dataclass
class LegSizing:
    """One leg of a multi-tenor sized order."""

    instrument_name: str
    tenor_days: int
    kind: str
    contracts: float
    strike: Optional[float] = None
    iv: Optional[float] = None


def _atm_contract(smile: list[SmileEntry], spot: float, kind: str) -> Optional[SmileEntry]:
    """Pick the contract whose strike is closest to spot for the requested kind."""
    matches = [e for e in smile if e.kind == kind]
    if not matches:
        return None
    return min(matches, key=lambda e: abs(e.strike - spot))


def _round_to_min(value: float, min_contract: float) -> float:
    """Round to the nearest multiple of ``min_contract`` (e.g. 0.001)."""
    if min_contract <= 0:
        return value
    return round(value / min_contract) * min_contract


def distribute_target_gamma(
    target_btc: float,
    tenors: list[int],
    surface: VolSurface,
    atm_gamma_estimate: float,
    kind: str = "call",
    min_contract: float = 0.001,
    max_per_trade: float = 0.1,
) -> list[LegSizing]:
    """Spread a target gamma exposure across multiple tenors.

    Args:
        target_btc: total gamma exposure in BTC equivalent (e.g. ``0.005``
            BTC of long gamma).
        tenors: list of tenor-day buckets to spread across (e.g. ``[7, 14, 30]``).
        surface: the current vol surface; we read the ATM contract per tenor.
        atm_gamma_estimate: BTC of gamma per ATM contract for the rough split.
            For BTC options at ATM this is typically a small number (~0.0001
            per contract on real chains); the caller can pass an estimate
            from the chain greeks. Tests pass 1.0 to make the math obvious.
        kind: ``"call"`` or ``"put"``.
        min_contract: smallest contract increment Thalex allows (default 0.001).
        max_per_trade: hard cap from the risk caps (default 0.1).

    Returns:
        List of :class:`LegSizing` — one per tenor that has a usable ATM
        contract. Tenors with no matching contract are silently dropped.
    """
    if not surface.smiles or not tenors:
        return []

    eligible_tenors: list[int] = []
    eligible_atm: dict[int, SmileEntry] = {}
    for tenor in tenors:
        smile = surface.smiles.get(tenor)
        if not smile:
            continue
        atm = _atm_contract(smile, surface.spot, kind)
        if atm is None:
            continue
        eligible_tenors.append(tenor)
        eligible_atm[tenor] = atm

    if not eligible_tenors:
        return []

    per_tenor_btc = target_btc / len(eligible_tenors)
    if atm_gamma_estimate <= 0:
        atm_gamma_estimate = 1.0  # avoid div-by-zero; degrade gracefully
    raw_contracts = per_tenor_btc / atm_gamma_estimate

    legs: list[LegSizing] = []
    for tenor in eligible_tenors:
        atm = eligible_atm[tenor]
        rounded = _round_to_min(raw_contracts, min_contract)
        clamped = max(min_contract, min(max_per_trade, rounded))
        legs.append(
            LegSizing(
                instrument_name=atm.instrument_name,
                tenor_days=tenor,
                kind=kind,
                contracts=clamped,
                strike=atm.strike,
                iv=atm.iv,
            )
        )
    return legs
