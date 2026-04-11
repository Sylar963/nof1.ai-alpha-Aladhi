"""Portfolio aggregator: turn raw Thalex positions into the OptionsContext view.

Walks the open Thalex options positions, fetches per-instrument greeks via
the adapter, parses days-to-expiry from each instrument name, and returns
two compact structures the LLM-facing snapshot consumes:

- ``open_positions``: list of dicts (one per leg) carrying instrument name,
  size, side, per-contract greeks, and days-to-expiry. The decision agent
  reads this to know what's already on the books.

- ``portfolio_greeks``: net delta/gamma/vega/theta totals across all open
  positions, with sign flipped for short legs. The agent uses these to
  understand current risk exposure before sizing a new trade.

Failure modes are intentionally tolerant: if a single ``get_greeks`` call
fails or an instrument name doesn't parse as an option, the offending
position is silently dropped from both lists rather than blowing up the
whole snapshot. Per-position errors are logged so they're visible during
calibration but never crash the bot.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Iterable, Optional

from src.backend.trading.options import parse_instrument_name


logger = logging.getLogger(__name__)


_GREEKS_KEYS = ("delta", "gamma", "vega", "theta")


async def aggregate_portfolio_greeks(
    positions: Iterable[dict],
    greeks_source: Any,
    today: date,
) -> dict:
    """Build the open_positions list and net portfolio greeks dict.

    Args:
        positions: iterable of position dicts in the shape Thalex exposes
            (``instrument_name``, ``size``, ``side``).
        greeks_source: any object exposing
            ``async get_greeks(instrument_name) -> dict``.
        today: reference UTC date used to compute days-to-expiry per leg.

    Returns:
        ``{"open_positions": [...], "portfolio_greeks": {delta, gamma, vega, theta}}``
    """
    open_positions: list[dict] = []
    totals = {key: 0.0 for key in _GREEKS_KEYS}

    for raw in positions:
        if not isinstance(raw, dict):
            continue
        instrument_name = raw.get("instrument_name") or ""
        if not instrument_name:
            continue
        try:
            size = float(raw.get("size") or 0.0)
        except (TypeError, ValueError):
            continue
        if size <= 0:
            continue

        spec = parse_instrument_name(instrument_name)
        if spec is None:
            logger.debug("portfolio aggregator: skipping unparseable instrument %s", instrument_name)
            continue

        try:
            greeks = await greeks_source.get_greeks(instrument_name)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("portfolio aggregator: get_greeks failed for %s: %s", instrument_name, exc)
            continue
        if not isinstance(greeks, dict):
            continue

        side = (raw.get("side") or "long").lower()
        signed_size = size if side == "long" else -size
        days_to_expiry = (spec.expiry - today).days

        position_view = {
            "instrument_name": instrument_name,
            "size": size,
            "side": side,
            "days_to_expiry": days_to_expiry,
            "kind": spec.kind,
            "strike": spec.strike,
        }
        for key in _GREEKS_KEYS:
            value = greeks.get(key)
            if value is None:
                continue
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            position_view[key] = value_f
            totals[key] += signed_size * value_f

        open_positions.append(position_view)

    return {
        "open_positions": open_positions,
        "portfolio_greeks": totals,
    }
