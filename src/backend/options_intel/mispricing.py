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
) -> MispricingScanReport:
    """Compute the largest Thalex - Deribit IV gaps after exact-match alignment.

    Args:
        thalex_chain: Thalex option instruments with ``mark_iv`` populated.
        deribit_chain: Deribit option instruments (book summary shape works).
        top_n: how many of the largest |IV diff| entries to return.
        min_edge_bps: drop entries whose absolute edge is below this threshold.

    Returns:
        :class:`MispricingScanReport` with the ranked top list, plus
        match/skip counters for visibility.
    """
    deribit_index = _index_deribit(deribit_chain)
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
    """Future-coverage stub: interpolate Deribit IVs to a Thalex tenor that
    Deribit doesn't list directly.

    PR A returns an empty list. PR C will implement linear-in-variance
    interpolation between the two closest Deribit expiries so unmatched Thalex
    instruments still get a synthetic Deribit IV for mispricing scoring. The
    function signature is fixed now so the rest of the pipeline can call it
    without changes when the implementation lands.
    """
    return []
