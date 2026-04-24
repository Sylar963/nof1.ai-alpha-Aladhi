"""Portfolio aggregator: turn raw Thalex positions into the OptionsContext view.

Walks the open Thalex options positions, fetches per-instrument greeks via
the adapter, parses days-to-expiry from each instrument name, and returns
two compact structures the LLM-facing snapshot consumes:

- ``open_positions``: list of dicts (one per leg) carrying instrument name,
  size, side, days-to-expiry, and — when greeks were available — the
  per-contract ``delta``, ``gamma``, ``vega``, ``theta`` keys. The decision
  agent reads this to know what's already on the books.

- ``portfolio_greeks``: net delta/gamma/vega/theta totals across all open
  positions, with sign flipped for short legs. The agent uses these to
  understand current risk exposure before sizing a new trade.

Failure handling — **fail-open by design**:

- **``get_greeks`` failure**: the position is RETAINED in ``open_positions``
  but with NO per-contract greek fields, so the operator still sees what
  the bot is holding even when the ticker hiccups. The failing leg
  contributes zero to ``portfolio_greeks`` totals. See the
  ``greeks = {}`` fallback in :func:`aggregate_portfolio_greeks` around
  line 79 — that's the explicit reset point.

- **Unparseable instrument name** (perpetual, future, garbage): the
  position is dropped from both outputs because we can't compute
  days-to-expiry or bucket it sensibly with the rest of the options.

- Per-position errors are logged so they're visible during calibration
  but never crash the bot.

**Caller contract**: each entry in ``open_positions`` may or may not
contain the ``delta``/``gamma``/``vega``/``theta`` keys. Consumers must
not assume the greek fields are present — read them with ``.get(...)``
or check ``"delta" in entry`` first. The aggregate ``portfolio_greeks``
dict is always present and always has all four keys, but contributions
from greekless legs will simply be zero.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any, Iterable, Optional

from src.backend.options_intel.greeks_bs import black_scholes_greeks
from src.backend.trading.options import parse_instrument_name


logger = logging.getLogger(__name__)


_GREEKS_KEYS = ("delta", "gamma", "vega", "theta")


async def aggregate_portfolio_greeks(
    positions: Iterable[dict],
    greeks_source: Any,
    today: date,
    *,
    spot: Optional[float] = None,
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

    Greeks are fetched in parallel via ``asyncio.gather`` so the total wall
    time for N positions is roughly one RPC, not N sequential RPCs.
    """
    # First pass: materialize all parsable positions and record the spec so
    # we only fire one get_greeks call per valid instrument.
    parsed: list[tuple[dict, Any, float, str, float]] = []
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

        side = (raw.get("side") or "long").lower()
        signed_size = size if side == "long" else -size
        parsed.append((raw, spec, size, side, signed_size))

    # Second pass: fetch greeks for every leg concurrently. return_exceptions
    # so one flaky instrument doesn't sink the whole aggregation.
    greeks_results = await asyncio.gather(
        *(greeks_source.get_greeks(raw["instrument_name"]) for raw, *_ in parsed),
        return_exceptions=True,
    )

    open_positions: list[dict] = []
    totals = {key: 0.0 for key in _GREEKS_KEYS}
    for (raw, spec, size, side, signed_size), greeks in zip(parsed, greeks_results, strict=True):
        if isinstance(greeks, Exception):
            logger.warning(
                "portfolio aggregator: get_greeks failed for %s: %s",
                raw["instrument_name"], greeks,
            )
            greeks = {}
        if not isinstance(greeks, dict):
            greeks = {}

        days_to_expiry = (spec.expiry - today).days
        position_view = {
            "instrument_name": raw["instrument_name"],
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

        # Thalex ticker only publishes iv + delta; fill in gamma/vega/theta
        # locally via Black-Scholes so the LLM sees a complete greeks row
        # and the portfolio totals aren't silently under-reporting risk.
        iv_raw = greeks.get("mark_iv") if isinstance(greeks, dict) else None
        if iv_raw is None and isinstance(greeks, dict):
            iv_raw = greeks.get("iv")
        try:
            iv_for_bs = float(iv_raw) if iv_raw is not None else 0.0
        except (TypeError, ValueError):
            iv_for_bs = 0.0
        if iv_for_bs > 2.0:
            iv_for_bs = iv_for_bs / 100.0
        missing = [k for k in _GREEKS_KEYS if k not in position_view]
        if missing and spot and iv_for_bs > 0 and days_to_expiry > 0:
            local = black_scholes_greeks(
                spot=float(spot),
                strike=float(spec.strike),
                iv=iv_for_bs,
                time_years=days_to_expiry / 365.0,
                kind=spec.kind,
            )
            for key in missing:
                position_view[key] = local[key]
            position_view["greeks_source"] = "bs_local"

        for key in _GREEKS_KEYS:
            if key in position_view:
                totals[key] += signed_size * position_view[key]

        open_positions.append(position_view)

    return {
        "open_positions": open_positions,
        "portfolio_greeks": totals,
    }
