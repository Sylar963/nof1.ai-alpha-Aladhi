"""Vol surface builder.

Bucket an option chain by tenor (days to expiry) and moneyness (K/S),
compute the smile per tenor, and derive the signals the options agent and
the regime classifier need:

- ``atm_iv_by_tenor[days]``: average of ATM call IV and ATM put IV
- ``expected_move_pct_by_tenor[days]``: ATM straddle premium / spot
- ``skew_25d_by_tenor[days]``: put-25Δ IV minus call-25Δ IV
- ``term_structure_slope``: linear slope of ATM IV across tenors
- ``atm_straddle_30d``: explicit anchor for the regime classifier's
  30-day straddle expected-move test (the user's heuristic).

The builder is defensive against malformed instrument records — anything
that doesn't parse cleanly is silently dropped, never raised. The point is
to keep the surface usable even when the upstream chain has occasional
garbage entries (perpetuals mixed in, expired contracts, etc.).
"""

from __future__ import annotations

import math
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
    atm_straddle_30d: Optional[dict] = None


# ---------------------------------------------------------------------------
# Surface merging
# ---------------------------------------------------------------------------


def merge_surfaces(primary: "VolSurface", fallback: "VolSurface") -> "VolSurface":
    """Merge two venue surfaces, preferring ``primary`` per tenor.

    Used to aggregate Thalex (primary) and Deribit (fallback). For every
    tenor present in either surface, we emit a single merged entry:
      - atm_iv, 25d skew, and expected-move come from primary when that
        tenor is populated there, else from fallback
      - the smile list is the union (primary first) so ATM-strike search
        and mispricing scans have the richest data to work with
    Term-structure slope and 30d anchor are recomputed from the merged
    atm_iv_by_tenor so the derived metrics stay self-consistent.
    """
    merged = VolSurface(
        spot=primary.spot or fallback.spot,
        today=primary.today or fallback.today,
    )

    for tenor in sorted(set(primary.smiles) | set(fallback.smiles)):
        primary_smile = primary.smiles.get(tenor, [])
        fallback_smile = fallback.smiles.get(tenor, [])
        merged.smiles[tenor] = list(primary_smile) + list(fallback_smile)

    for tenor in sorted(set(primary.atm_iv_by_tenor) | set(fallback.atm_iv_by_tenor)):
        merged.atm_iv_by_tenor[tenor] = (
            primary.atm_iv_by_tenor.get(tenor)
            if tenor in primary.atm_iv_by_tenor
            else fallback.atm_iv_by_tenor.get(tenor)
        )

    for tenor in sorted(set(primary.skew_25d_by_tenor) | set(fallback.skew_25d_by_tenor)):
        merged.skew_25d_by_tenor[tenor] = (
            primary.skew_25d_by_tenor.get(tenor)
            if tenor in primary.skew_25d_by_tenor
            else fallback.skew_25d_by_tenor.get(tenor)
        )

    for tenor in sorted(set(primary.expected_move_pct_by_tenor) | set(fallback.expected_move_pct_by_tenor)):
        merged.expected_move_pct_by_tenor[tenor] = (
            primary.expected_move_pct_by_tenor.get(tenor)
            if tenor in primary.expected_move_pct_by_tenor
            else fallback.expected_move_pct_by_tenor.get(tenor)
        )

    merged.term_structure_slope = _term_structure_slope(merged.atm_iv_by_tenor)
    merged.atm_straddle_30d = primary.atm_straddle_30d or fallback.atm_straddle_30d
    return merged


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
    surface.atm_straddle_30d = _atm_straddle_anchor(smiles, spot, target_tenor=30)

    return surface


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _first_present(record: dict, *keys: str):
    """Return the first key's value that is not None. Lets us accept both
    Thalex (``strike_price``, ``iv``, ``expiration_timestamp``) and the
    Deribit/test-fixture (``strike``, ``mark_iv``, ``expiry_timestamp``)
    naming without branching at every call site."""
    for key in keys:
        raw = record.get(key)
        if raw is not None:
            return raw
    return None


def _parse_entry(record: dict) -> Optional[SmileEntry]:
    if not isinstance(record, dict):
        return None
    rec_type = record.get("type")
    if rec_type is not None and rec_type != "option":
        return None
    name = record.get("instrument_name")
    if not isinstance(name, str) or not name:
        return None
    try:
        strike = float(_first_present(record, "strike", "strike_price") or 0)
        iv = float(_first_present(record, "iv", "mark_iv") or 0)
    except (TypeError, ValueError):
        return None
    if strike <= 0 or iv <= 0:
        return None
    if iv > 2.0:
        iv = iv / 100.0
    kind = record.get("option_type") or record.get("kind") or ""
    if kind not in ("call", "put"):
        return None
    try:
        price = float(_first_present(record, "mark_price") or 0)
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
    ts = _first_present(record, "expiry_timestamp", "expiration_timestamp")
    if not isinstance(ts, (int, float)):
        return None
    ts_float = float(ts)
    if ts_float > 1e11:
        ts_float = ts_float / 1000.0
    try:
        expiry = datetime.fromtimestamp(int(ts_float), tz=timezone.utc).date()
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
    target_tenor: int = 30,
    max_offset_days: int = 7,
) -> Optional[dict]:
    """Find the smile closest to ``target_tenor`` and return its ATM straddle anchor.

    Used by the regime classifier as the baseline for the user's
    "is price outside the implied range" heuristic. The returned dict's
    ``tenor_days`` carries the **bucket** id (``target_tenor``) so the
    persistence layer can ``lookback(tenor_days=target_tenor)`` on day+30
    without caring that the actual closest expiry shifts with the weekly
    Friday calendar. ``actual_tenor_days`` keeps the real expiry for
    inspection. Returns ``None`` when no smile is within
    ``max_offset_days`` of the target.
    """
    if not smiles:
        return None
    closest_tenor = min(smiles.keys(), key=lambda t: abs(t - target_tenor))
    if abs(closest_tenor - target_tenor) > max_offset_days:
        return None
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
        "tenor_days": target_tenor,
        "actual_tenor_days": closest_tenor,
        "strike": strike,
        "spot_at_init": spot,
        "call_price": call.price,
        "put_price": put.price,
        "straddle_premium": straddle,
        "lower_strike": strike - straddle,
        "upper_strike": strike + straddle,
        "atm_iv": (call.iv + put.iv) / 2.0,
    }


def atm_iv_for_target_tenor(
    atm_iv_by_tenor: dict[int, float],
    target_tenor: int,
    max_offset_days: int = 7,
) -> Optional[float]:
    """Return the ATM IV at ``target_tenor``, interpolating across neighbors.

    BTC option expiries cluster on Fridays, so on most calendar days there
    is no tenor that hashes to exactly ``target_tenor``. This resolves the
    target by:
      1. exact hit → return it
      2. neighbors on both sides within ``max_offset_days`` → linear
         interpolation in total variance (``iv**2 * t``)
      3. only one side within band → that side's IV
      4. nothing within band → ``None``

    Regime Signal 2 needs a 30-day ATM IV anchor; without this the
    classifier reads 0 and falls back to ``unknown`` on every cycle.
    """
    if not atm_iv_by_tenor:
        return None
    exact = atm_iv_by_tenor.get(target_tenor)
    if exact and exact > 0:
        return float(exact)

    tenors = sorted(t for t, iv in atm_iv_by_tenor.items() if iv and iv > 0)
    if not tenors:
        return None

    below = [t for t in tenors if t < target_tenor]
    above = [t for t in tenors if t > target_tenor]
    lower = below[-1] if below else None
    upper = above[0] if above else None

    if lower is not None and upper is not None:
        if (target_tenor - lower) > max_offset_days and (upper - target_tenor) > max_offset_days:
            return None
        iv_lo = float(atm_iv_by_tenor[lower])
        iv_hi = float(atm_iv_by_tenor[upper])
        tv_lo = (iv_lo * iv_lo) * lower
        tv_hi = (iv_hi * iv_hi) * upper
        frac = (target_tenor - lower) / (upper - lower)
        tv = tv_lo + frac * (tv_hi - tv_lo)
        interpolated_var = tv / max(target_tenor, 1)
        if interpolated_var <= 0:
            return None
        return math.sqrt(interpolated_var)

    nearest = lower if lower is not None else upper
    if nearest is None or abs(nearest - target_tenor) > max_offset_days:
        return None
    iv = atm_iv_by_tenor[nearest]
    return float(iv) if iv and iv > 0 else None
