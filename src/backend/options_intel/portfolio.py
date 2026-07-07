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
from decimal import Decimal
from typing import Any, Iterable, Optional

from src.backend.options_intel.greeks_bs import black_scholes_greeks
from src.backend.options_intel.structure import OptionLeg, classify, classify_many
from src.backend.trading.options import parse_instrument_name


logger = logging.getLogger(__name__)


_GREEKS_KEYS = ("delta", "gamma", "vega", "theta")
_PRICE_KEYS = ("mark_price",)


def _bs_price(*, spot: float, strike: float, iv: float, time_years: float, kind: str) -> float:
    import math
    if spot <= 0 or strike <= 0 or iv <= 0 or time_years <= 0:
        return 0.0
    vol_sqrt_t = iv * math.sqrt(time_years)
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * time_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    def _ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    if kind == "call":
        return spot * _ncdf(d1) - strike * _ncdf(d2)
    return strike * _ncdf(-d2) - spot * _ncdf(-d1)


async def aggregate_portfolio_greeks(
    positions: Iterable[dict],
    greeks_source: Any,
    today: date,
    *,
    spot: Optional[float] = None,
    entry_premium_by_structure_id: Optional[dict] = None,
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
        for key in _GREEKS_KEYS + _PRICE_KEYS:
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
        if "mark_price" not in position_view and spot and iv_for_bs > 0 and days_to_expiry > 0:
            position_view["mark_price"] = _bs_price(
                spot=float(spot),
                strike=float(spec.strike),
                iv=iv_for_bs,
                time_years=days_to_expiry / 365.0,
                kind=spec.kind,
            )

        for key in _GREEKS_KEYS:
            if key in position_view:
                totals[key] += signed_size * position_view[key]

        open_positions.append(position_view)

    structure_legs: list[OptionLeg] = []
    for position_view in open_positions:
        try:
            structure_legs.append(
                OptionLeg(
                    instrument_name=position_view["instrument_name"],
                    kind=position_view["kind"],
                    strike=Decimal(str(position_view["strike"])),
                    side=position_view["side"],
                    contracts=Decimal(str(position_view["size"])),
                    days_to_expiry=int(position_view["days_to_expiry"]),
                    mark_price=Decimal(str(position_view["mark_price"])) if "mark_price" in position_view else Decimal("0"),
                    delta=Decimal(str(position_view["delta"])) if "delta" in position_view else None,
                    gamma=Decimal(str(position_view["gamma"])) if "gamma" in position_view else None,
                    vega=Decimal(str(position_view["vega"])) if "vega" in position_view else None,
                    theta=Decimal(str(position_view["theta"])) if "theta" in position_view else None,
                )
            )
        except (KeyError, ValueError) as exc:
            logger.debug("portfolio aggregator: skipping leg for classifier: %s", exc)
            continue

    structures: list[dict] = []
    if structure_legs:
        from src.backend.options_intel.entry_basis import fill_entry_price_map

        entry_prices = await fill_entry_price_map(
            greeks_source,
            {
                leg.instrument_name: (leg.side, float(leg.contracts))
                for leg in structure_legs
            },
        )
        results = classify_many(
            structure_legs,
            entry_net_premium_by_id=entry_premium_by_structure_id,
            entry_price_by_instrument=entry_prices,
        )
        for result in results:
            structures.append({
                "structure_id": result.structure_id,
                "kind": result.kind.value,
                "underlying": result.underlying,
                "tenor_days_min": result.tenor_days_min,
                "tenor_days_max": result.tenor_days_max,
                "net_premium": float(result.net_premium),
                "entry_net_premium": (
                    float(result.entry_net_premium)
                    if result.entry_net_premium is not None else None
                ),
                "is_credit": result.is_credit,
                "max_loss": float(result.max_loss) if result.max_loss is not None else None,
                "max_profit": float(result.max_profit) if result.max_profit is not None else None,
                "breakevens": [float(b) for b in result.breakevens],
                "short_leg_delta": float(result.short_leg_delta) if result.short_leg_delta is not None else None,
                "breach_state": result.breach_state.value,
                "pnl_abs": float(result.pnl_abs),
                "pnl_pct": float(result.pnl_pct),
                "aggregate_greeks": {k: float(v) for k, v in result.aggregate_greeks.items()},
                "confidence": result.confidence,
                "legs": [leg.instrument_name for leg in result.legs],
            })

    return {
        "open_positions": open_positions,
        "portfolio_greeks": totals,
        "structures": structures,
    }
