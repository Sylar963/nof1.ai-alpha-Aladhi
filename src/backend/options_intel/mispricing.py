"""Cross-venue IV mispricing scanner: Thalex vs Deribit.

Aligns Thalex option instruments with their exact Deribit counterparts by
``(expiry, strike, kind)`` and computes the IV gap in basis points. The top-N
largest gaps (by absolute value) become arbitrage candidates the LLM can act
on via the ``vol_arb`` strategy.

PR A behavior is **strict exact-match alignment** — any Thalex instrument
without a Deribit counterpart at the same expiry-strike-kind is skipped, with
the count exposed in the report so we can see how much coverage we're losing.
PR C will add a surface interpolation path so unmatched tenors get a synthetic
Deribit IV from the closest two expiries; the entry point lives here as the
``interpolate_deribit_surface`` no-op stub.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass
class MispricingScanReport:
    """Outcome of one Thalex × Deribit mispricing scan."""

    top: list[dict] = field(default_factory=list)
    matched_count: int = 0
    skipped_count: int = 0


def _normalize_expiry_seconds(record: dict) -> Optional[int]:
    """Extract expiry as a UTC timestamp in *seconds*.

    Thalex publishes ``expiry_timestamp`` in seconds; Deribit publishes
    ``expiration_timestamp`` in milliseconds. We normalize both to seconds
    and round to the nearest day so off-by-an-hour intra-day differences
    don't break exact-match alignment.
    """
    raw = record.get("expiry_timestamp")
    if raw is None:
        raw = record.get("expiration_timestamp")
        if isinstance(raw, (int, float)):
            raw = int(raw / 1000)
    if not isinstance(raw, (int, float)):
        return None
    try:
        as_date = datetime.fromtimestamp(int(raw), tz=timezone.utc).date()
        return int(datetime(as_date.year, as_date.month, as_date.day, tzinfo=timezone.utc).timestamp())
    except (OverflowError, OSError, ValueError):
        return None


def _kind_of(record: dict) -> Optional[str]:
    kind = record.get("option_type") or record.get("kind") or ""
    if kind in ("call", "put"):
        return kind
    return None


def _strike_of(record: dict) -> Optional[float]:
    raw = record.get("strike")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _iv_of(record: dict) -> Optional[float]:
    raw = record.get("mark_iv")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _index_deribit(deribit_chain: list[dict]) -> dict[tuple, dict]:
    """Build a lookup keyed by ``(expiry_day_seconds, strike, kind)``."""
    index: dict[tuple, dict] = {}
    for record in deribit_chain:
        if not isinstance(record, dict):
            continue
        expiry = _normalize_expiry_seconds(record)
        kind = _kind_of(record)
        strike = _strike_of(record)
        if expiry is None or kind is None or strike is None:
            continue
        index[(expiry, strike, kind)] = record
    return index


def scan_mispricings(
    thalex_chain: list[dict],
    deribit_chain: list[dict],
    top_n: int = 5,
    min_edge_bps: float = 0.0,
    use_interpolation: bool = False,
) -> MispricingScanReport:
    """Compute the largest Thalex - Deribit IV gaps after alignment.

    Args:
        thalex_chain: Thalex option instruments with ``mark_iv`` populated.
        deribit_chain: Deribit option instruments (book summary shape works).
        top_n: how many of the largest |IV diff| entries to return.
        min_edge_bps: drop entries whose absolute edge is below this threshold.
        use_interpolation: when True, fill in synthetic Deribit IVs for any
            Thalex tenor Deribit doesn't list directly, by linear-in-variance
            interpolation between the two bracketing Deribit expiries. PR C
            default is False — match exactly to keep the surface conservative.

    Returns:
        :class:`MispricingScanReport` with the ranked top list, plus
        match/skip counters for visibility.
    """
    deribit_index = _index_deribit(deribit_chain)

    if use_interpolation:
        thalex_target_expiries = _collect_thalex_expiries(thalex_chain)
        for target_expiry in thalex_target_expiries:
            if any(key[0] == target_expiry for key in deribit_index):
                continue  # already covered exactly
            synthetic = interpolate_deribit_surface(deribit_chain, target_expiry=target_expiry)
            for record in synthetic:
                # Re-index with the (target_expiry, strike, kind) key.
                deribit_index[(target_expiry, record["strike"], record["option_type"])] = record

    matched = 0
    skipped = 0
    candidates: list[dict] = []

    for record in thalex_chain:
        if not isinstance(record, dict):
            continue
        expiry = _normalize_expiry_seconds(record)
        kind = _kind_of(record)
        strike = _strike_of(record)
        thalex_iv = _iv_of(record)
        if expiry is None or kind is None or strike is None or thalex_iv is None:
            continue

        deribit_record = deribit_index.get((expiry, strike, kind))
        if deribit_record is None:
            skipped += 1
            continue

        deribit_iv = _iv_of(deribit_record)
        if deribit_iv is None:
            skipped += 1
            continue

        edge_bps = (thalex_iv - deribit_iv) * 10_000
        if abs(edge_bps) < min_edge_bps:
            matched += 1
            continue

        candidates.append({
            "instrument_name": record.get("instrument_name", ""),
            "expiry_seconds": expiry,
            "strike": strike,
            "kind": kind,
            "iv_thalex": thalex_iv,
            "iv_deribit": deribit_iv,
            "edge_bps": edge_bps,
        })
        matched += 1

    candidates.sort(key=lambda c: abs(c["edge_bps"]), reverse=True)
    if skipped:
        logger.info("mispricing scan skipped %d Thalex instruments without Deribit match", skipped)
    return MispricingScanReport(
        top=candidates[:top_n],
        matched_count=matched,
        skipped_count=skipped,
    )


def interpolate_deribit_surface(
    deribit_chain: list[dict],
    target_expiry: int,
) -> list[dict]:
    """Linear-in-variance interpolation between two bracketing Deribit expiries.

    For each strike that exists in BOTH the nearest expiry below and the
    nearest expiry above ``target_expiry``, compute a synthetic IV at the
    target tenor using the standard practitioner formula:

        var(T) = (1 - w) * var(T1) + w * var(T2)
        iv(T)  = sqrt(var(T))
        where w = (T - T1) / (T2 - T1)

    Returns a list of synthetic instrument records (one per strike per
    kind) shaped like Deribit book summary entries so the mispricing
    scanner can consume them via the same alignment path.

    Returns an empty list when:
    - There are not at least one expiry below AND one expiry above the target.
    - No strikes are present in both bracketing expiries.
    - Any required IV is missing or non-positive.
    """
    # Normalize the target the same way we normalize chain expiries (midnight
    # UTC of the date) so weight math is consistent regardless of whether the
    # caller passed an intra-day timestamp.
    normalized_target = _normalize_expiry_seconds({"expiry_timestamp": int(target_expiry)})
    if normalized_target is None:
        return []

    by_expiry: dict[int, list[dict]] = {}
    for record in deribit_chain:
        if not isinstance(record, dict):
            continue
        expiry = _normalize_expiry_seconds(record)
        if expiry is None:
            continue
        by_expiry.setdefault(expiry, []).append(record)

    expiries_below = sorted([e for e in by_expiry if e < normalized_target])
    expiries_above = sorted([e for e in by_expiry if e > normalized_target])
    if not expiries_below or not expiries_above:
        return []

    t1 = expiries_below[-1]  # closest below
    t2 = expiries_above[0]   # closest above
    if t2 == t1:
        return []
    weight = (normalized_target - t1) / (t2 - t1)

    near_index: dict[tuple, dict] = {}
    far_index: dict[tuple, dict] = {}
    for record in by_expiry[t1]:
        kind = _kind_of(record)
        strike = _strike_of(record)
        if kind is None or strike is None:
            continue
        near_index[(strike, kind)] = record
    for record in by_expiry[t2]:
        kind = _kind_of(record)
        strike = _strike_of(record)
        if kind is None or strike is None:
            continue
        far_index[(strike, kind)] = record

    interpolated: list[dict] = []
    for key in near_index.keys() & far_index.keys():
        near_iv = _iv_of(near_index[key])
        far_iv = _iv_of(far_index[key])
        if near_iv is None or far_iv is None or near_iv <= 0 or far_iv <= 0:
            continue
        var_t1 = near_iv * near_iv
        var_t2 = far_iv * far_iv
        var_target = (1.0 - weight) * var_t1 + weight * var_t2
        if var_target <= 0:
            continue
        iv_target = math.sqrt(var_target)
        strike, kind = key
        interpolated.append({
            "instrument_name": f"INTERPOLATED-{int(normalized_target)}-{int(strike)}-{kind[0].upper()}",
            "kind": "option",
            "option_type": kind,
            "strike": strike,
            "mark_iv": iv_target,
            "expiration_timestamp": int(normalized_target) * 1000,
            "interpolated_from": [t1, t2],
        })
    return interpolated


def _collect_thalex_expiries(thalex_chain: list[dict]) -> set[int]:
    """Set of unique normalized expiries (in seconds) found in a Thalex chain."""
    expiries: set[int] = set()
    for record in thalex_chain:
        if not isinstance(record, dict):
            continue
        expiry = _normalize_expiry_seconds(record)
        if expiry is not None:
            expiries.add(expiry)
    return expiries
