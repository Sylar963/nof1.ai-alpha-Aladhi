"""Enrich a Thalex option chain with ticker data.

``public/instruments`` on Thalex returns contract metadata only — no IV,
no mark price. To build a vol surface we need ``iv``, ``mark_price``, and
``delta``, which live on ``public/ticker`` (per-instrument). Fetching
every ticker would be 200+ RPCs, so we filter to the relevant subset
first:

  - DTE ≤ ``max_dte_days`` (strategies above this horizon aren't used)
  - strike within ±``strike_band_pct`` of spot (far-OTM tails don't
    contribute to the ATM IV / 25-delta skew we need)

Records outside the filter are passed through untouched. The enrichment
is additive: we merge ticker fields into a copy of the instrument
record, never mutating the caller's list.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Iterable, Optional


logger = logging.getLogger(__name__)


_TICKER_FIELDS_TO_MERGE: tuple[str, ...] = (
    "iv",
    "mark_price",
    "delta",
    "forward",
    "index",
    "open_interest",
    "volume_24h",
)


def _record_expiry_seconds(record: dict) -> Optional[int]:
    raw = record.get("expiry_timestamp")
    if raw is None:
        raw = record.get("expiration_timestamp")
    if not isinstance(raw, (int, float)):
        return None
    ts = float(raw)
    if ts > 1e11:
        ts = ts / 1000.0
    return int(ts)


def _record_strike(record: dict) -> Optional[float]:
    for key in ("strike", "strike_price"):
        raw = record.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _is_target(record: dict, *, spot: float, max_dte_days: int, strike_band_pct: float, now_s: int) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("type") not in (None, "option"):
        return False
    if (record.get("option_type") or record.get("kind")) not in ("call", "put"):
        return False
    expiry = _record_expiry_seconds(record)
    if expiry is None or expiry <= now_s:
        return False
    dte = (expiry - now_s) / 86400.0
    if dte > max_dte_days:
        return False
    strike = _record_strike(record)
    if strike is None or strike <= 0:
        return False
    lo = spot * (1.0 - strike_band_pct)
    hi = spot * (1.0 + strike_band_pct)
    return lo <= strike <= hi


async def enrich_chain_with_tickers(
    chain: list[dict],
    *,
    thalex: Any,
    spot: float,
    max_dte_days: int = 90,
    strike_band_pct: float = 0.35,
    concurrency: int = 12,
    per_request_timeout_s: float = 4.0,
) -> tuple[list[dict], dict]:
    """Return ``(enriched_chain, stats)``.

    ``enriched_chain`` is a new list with ticker fields merged into the
    filtered subset. ``stats`` reports how many records were targeted,
    how many enrichments succeeded, and how many failed so callers can
    surface degradation in the coverage map.
    """
    stats = {"targeted": 0, "enriched": 0, "failed": 0}
    if not chain or spot <= 0:
        return list(chain), stats

    now_s = int(time.time())
    copy = [dict(r) if isinstance(r, dict) else r for r in chain]

    target_indices = [
        i for i, r in enumerate(copy)
        if isinstance(r, dict)
        and _is_target(r, spot=spot, max_dte_days=max_dte_days, strike_band_pct=strike_band_pct, now_s=now_s)
    ]
    stats["targeted"] = len(target_indices)
    if not target_indices:
        return copy, stats

    sem = asyncio.Semaphore(concurrency)

    async def fetch_one(idx: int) -> bool:
        name = copy[idx].get("instrument_name")
        if not name:
            return False
        async with sem:
            try:
                ticker = await asyncio.wait_for(
                    thalex.get_ticker_snapshot(name),
                    timeout=per_request_timeout_s,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("ticker fetch failed for %s: %s", name, exc)
                return False
        if not isinstance(ticker, dict) or not ticker:
            return False
        for key in _TICKER_FIELDS_TO_MERGE:
            if key in ticker and ticker[key] is not None:
                copy[idx][key] = ticker[key]
        return True

    results = await asyncio.gather(*(fetch_one(i) for i in target_indices), return_exceptions=True)
    for ok in results:
        if isinstance(ok, Exception):
            stats["failed"] += 1
        elif ok:
            stats["enriched"] += 1
        else:
            stats["failed"] += 1
    return copy, stats
