"""Vol surface builder.

Bucket an option chain by tenor (days to expiry) and moneyness (K/S),
compute the smile per tenor, and derive the signals the options agent and
the regime classifier need:

- ``atm_iv_by_tenor[days]``: average of ATM call IV and ATM put IV
- ``expected_move_pct_by_tenor[days]``: ATM straddle premium / spot
- ``skew_25d_by_tenor[days]``: put-25Δ IV minus call-25Δ IV
- ``term_structure_slope``: linear slope of ATM IV across tenors
- ``atm_straddle_15d``: explicit anchor for the regime classifier's
  15-day straddle expected-move test (the user's heuristic).

The builder is defensive against malformed instrument records — anything
that doesn't parse cleanly is silently dropped, never raised. The point is
to keep the surface usable even when the upstream chain has occasional
garbage entries (perpetuals mixed in, expired contracts, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional


@dataclass
class SmileEntry:
    """One option contract on the smile (one strike + kind)."""

    instrument_name: str
    strike: float
    kind: str  # "call" | "put"
    iv: float
    price: float
    delta: Optional[float] = None


@dataclass
class VolSurface:
    """Structured view of an option chain at a single point in time."""

    spot: float
    today: date
    smiles: dict[int, list[SmileEntry]] = field(default_factory=dict)
    atm_iv_by_tenor: dict[int, float] = field(default_factory=dict)
    expected_move_pct_by_tenor: dict[int, float] = field(default_factory=dict)
    skew_25d_by_tenor: dict[int, float] = field(default_factory=dict)
    term_structure_slope: float = 0.0
    atm_straddle_15d: Optional[dict] = None


# ---------------------------------------------------------------------------
# Build entrypoint
# ---------------------------------------------------------------------------


def build_vol_surface(
    chain: list[dict],
    spot: float,
    today: Optional[date] = None,
) -> VolSurface:
    """Convert a raw option chain into a structured VolSurface.

    Args:
        chain: list of instrument records (Thalex-shape: ``instrument_name``,
            ``expiry_timestamp``, ``strike``, ``option_type``, ``mark_iv``,
            ``mark_price``, optional ``delta``).
        spot: current underlying spot price.
        today: reference date for tenor computation. Defaults to today (UTC).
    """
    today = today or datetime.now(timezone.utc).date()

    smiles: dict[int, list[SmileEntry]] = {}
    for record in chain:
        entry = _parse_entry(record)
        if entry is None:
            continue
        days = _days_to_expiry(record, today)
        if days is None or days <= 0:
            continue
        smiles.setdefault(days, []).append(entry)

    surface = VolSurface(spot=spot, today=today, smiles=smiles)
    if not smiles:
        return surface

    for tenor in smiles:
        atm = _atm_iv(smiles[tenor], spot)
        if atm is not None:
            surface.atm_iv_by_tenor[tenor] = atm
        em = _atm_straddle_expected_move_pct(smiles[tenor], spot)
        if em is not None:
            surface.expected_move_pct_by_tenor[tenor] = em
        skew = _skew_25d(smiles[tenor])
        if skew is not None:
            surface.skew_25d_by_tenor[tenor] = skew

    surface.term_structure_slope = _term_structure_slope(surface.atm_iv_by_tenor)
    surface.atm_straddle_15d = _atm_straddle_anchor(smiles, spot, target_tenor=15)

    return surface


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_entry(record: dict) -> Optional[SmileEntry]:
    if not isinstance(record, dict):
        return None
    if record.get("type") != "option":
        return None
    name = record.get("instrument_name")
    if not isinstance(name, str) or not name:
        return None
    try:
        strike = float(record.get("strike", 0))
        iv = float(record.get("mark_iv", 0))
    except (TypeError, ValueError):
        return None
    if strike <= 0 or iv <= 0:
        return None
    kind = record.get("option_type") or record.get("kind") or ""
    if kind not in ("call", "put"):
        return None
    try:
        price = float(record.get("mark_price", 0) or 0)
    except (TypeError, ValueError):
        price = 0.0
    delta_raw = record.get("delta")
    try:
        delta = float(delta_raw) if delta_raw is not None else None
    except (TypeError, ValueError):
        delta = None
    return SmileEntry(
        instrument_name=name,
        strike=strike,
        kind=kind,
        iv=iv,
        price=price,
        delta=delta,
    )


def _days_to_expiry(record: dict, today: date) -> Optional[int]:
    ts = record.get("expiry_timestamp")
    if not isinstance(ts, (int, float)):
        return None
    try:
        expiry = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
    except (OverflowError, OSError, ValueError):
        return None
    return (expiry - today).days


# ---------------------------------------------------------------------------
# Smile-derived metrics
# ---------------------------------------------------------------------------


def _atm_strike(smile: list[SmileEntry], spot: float) -> Optional[float]:
    if not smile:
        return None
    return min((e.strike for e in smile), key=lambda k: abs(k - spot))


def _atm_iv(smile: list[SmileEntry], spot: float) -> Optional[float]:
    """Mean of the call IV and put IV at the strike closest to spot."""
    strike = _atm_strike(smile, spot)
    if strike is None:
        return None
    call = next((e for e in smile if e.strike == strike and e.kind == "call"), None)
    put = next((e for e in smile if e.strike == strike and e.kind == "put"), None)
    if call is None and put is None:
        return None
    if call is None:
        return put.iv  # type: ignore[union-attr]
    if put is None:
        return call.iv
    return (call.iv + put.iv) / 2.0


def _atm_straddle_expected_move_pct(smile: list[SmileEntry], spot: float) -> Optional[float]:
    """ATM straddle (call_price + put_price) / spot — implied move over the tenor."""
    strike = _atm_strike(smile, spot)
    if strike is None or spot <= 0:
        return None
    call = next((e for e in smile if e.strike == strike and e.kind == "call"), None)
    put = next((e for e in smile if e.strike == strike and e.kind == "put"), None)
    if call is None or put is None:
        return None
    straddle = call.price + put.price
    if straddle <= 0:
        return None
    return straddle / spot


def _skew_25d(smile: list[SmileEntry]) -> Optional[float]:
    """25-delta skew = (25Δ put IV) − (25Δ call IV).

    Picks the put closest to delta=-0.25 and the call closest to delta=+0.25.
    Requires deltas to be present in the chain; returns None if not.
    """
    puts = [e for e in smile if e.kind == "put" and e.delta is not None]
    calls = [e for e in smile if e.kind == "call" and e.delta is not None]
    if not puts or not calls:
        return None
    target_put = min(puts, key=lambda e: abs(e.delta + 0.25))  # type: ignore[operator]
    target_call = min(calls, key=lambda e: abs(e.delta - 0.25))  # type: ignore[operator]
    return target_put.iv - target_call.iv


def _term_structure_slope(atm_iv_by_tenor: dict[int, float]) -> float:
    """Linear slope of ATM IV across tenors. Positive = contango, negative = backwardation."""
    if len(atm_iv_by_tenor) < 2:
        return 0.0
    items = sorted(atm_iv_by_tenor.items())
    short_t, short_iv = items[0]
    long_t, long_iv = items[-1]
    if long_t == short_t:
        return 0.0
    return (long_iv - short_iv) / (long_t - short_t)


def _atm_straddle_anchor(
    smiles: dict[int, list[SmileEntry]],
    spot: float,
    target_tenor: int = 15,
) -> Optional[dict]:
    """Find the smile closest to ``target_tenor`` and return its ATM straddle anchor.

    Used by the regime classifier as the baseline for the user's
    "is price outside the implied range" heuristic.
    """
    if not smiles:
        return None
    closest_tenor = min(smiles.keys(), key=lambda t: abs(t - target_tenor))
    smile = smiles[closest_tenor]
    strike = _atm_strike(smile, spot)
    if strike is None:
        return None
    call = next((e for e in smile if e.strike == strike and e.kind == "call"), None)
    put = next((e for e in smile if e.strike == strike and e.kind == "put"), None)
    if call is None or put is None:
        return None
    straddle = call.price + put.price
    return {
        "tenor_days": closest_tenor,
        "strike": strike,
        "spot_at_init": spot,
        "call_price": call.price,
        "put_price": put.price,
        "straddle_premium": straddle,
        "lower_strike": strike - straddle,
        "upper_strike": strike + straddle,
        "atm_iv": (call.iv + put.iv) / 2.0,
    }
