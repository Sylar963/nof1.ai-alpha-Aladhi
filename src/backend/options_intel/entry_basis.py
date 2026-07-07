"""Fill-based entry price basis for open option legs.

Walks the venue's trade history newest-to-oldest and takes the most recent
same-direction fills that add up to each leg's current size (LIFO basis).
The resulting per-contract VWAP is the price actually paid/received at
entry, replacing the first-observed-mark proxy used before. Legs whose
fills can't cover the position size (history horizon exceeded, transferred
positions) are omitted so callers fall back to the persisted mark-based
entry premium.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def _fill_field(fill: dict, *names, default=None):
    for name in names:
        value = fill.get(name)
        if value is not None:
            return value
    return default


def compute_fill_entry_prices(
    fills: list,
    legs: Dict[str, Tuple[str, float]],
) -> Dict[str, Decimal]:
    """Per-contract entry VWAP for each leg covered by fill history.

    Args:
        fills: venue trade-history dicts (any recency order; sorted here).
        legs: instrument_name -> (side "long"|"short", current abs size).

    Returns:
        instrument_name -> Decimal VWAP, only for legs whose same-direction
        fills cover >= 99% of the current size.
    """
    remaining: Dict[str, float] = {}
    acc_cost: Dict[str, float] = {}
    acc_qty: Dict[str, float] = {}
    for name, (_side, size) in legs.items():
        try:
            size_f = float(size)
        except (TypeError, ValueError):
            continue
        if size_f > 0:
            remaining[name] = size_f
            acc_cost[name] = 0.0
            acc_qty[name] = 0.0
    if not remaining:
        return {}

    def _ts(fill: dict) -> float:
        try:
            return float(_fill_field(fill, "time", "timestamp", default=0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    ordered = sorted(
        (f for f in fills or [] if isinstance(f, dict)),
        key=_ts,
        reverse=True,
    )
    for fill in ordered:
        name = str(_fill_field(fill, "instrument_name", "asset") or "")
        if name not in remaining or remaining[name] <= 0:
            continue
        direction = str(_fill_field(fill, "direction", "side", default="") or "").lower()
        wanted = "buy" if legs[name][0] == "long" else "sell"
        if not direction.startswith(wanted):
            continue
        try:
            amount = float(_fill_field(fill, "amount", "quantity", "size", default=0.0) or 0.0)
            price = float(_fill_field(fill, "price", "px", default=0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if amount <= 0 or price <= 0:
            continue
        take = min(amount, remaining[name])
        acc_cost[name] += take * price
        acc_qty[name] += take
        remaining[name] -= take

    out: Dict[str, Decimal] = {}
    for name in acc_qty:
        size_f = float(legs[name][1])
        if acc_qty[name] > 0 and remaining[name] <= size_f * 0.01:
            out[name] = Decimal(str(acc_cost[name] / acc_qty[name]))
    return out


async def fill_entry_price_map(
    adapter: Any,
    legs: Dict[str, Tuple[str, float]],
    limit: int = 200,
) -> Dict[str, Decimal]:
    """Fetch venue fills and compute the entry basis; fail-soft to {}."""
    get_fills = getattr(adapter, "get_recent_fills", None)
    if not callable(get_fills) or not legs:
        return {}
    try:
        fills = await get_fills(limit=limit)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("entry basis: fills fetch failed: %s", exc)
        return {}
    return compute_fill_entry_prices(fills or [], legs)
