"""Event-driven delta-hedge manager for Thalex options positions.

This is the runtime glue that turns the existing pure-logic
:class:`DeltaHedger` into a continuously running hedge controller. It owns:

- a registry of currently hedged Thalex positions (instrument → contracts/kind),
- ticker subscriptions on the Thalex adapter (one per open instrument),
- a callback that fires on every ticker push, computes drift between the
  target perp delta and the current Hyperliquid perp position, and submits a
  rebalance trade only when the drift breaches the configured threshold.

Hedging is **threshold-driven, not time-driven**. Below the threshold no
trade is sent — the underlying's normal noise doesn't churn perp fees. Above
the threshold the rebalance brings the perp leg back to target at the new
(worse) underlying price; subsequent rebalance-to-target trades automatically
buy low and sell high as drift mean-reverts, capturing the pullback as
realized P&L. That's gamma scalping in disguise.

The manager itself is small and stateless beyond the position registry; the
:class:`DeltaHedger` instance handles all the math.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.backend.trading.exchange_adapter import ExchangeAdapter
from src.backend.trading.options_strategies import DeltaHedger


logger = logging.getLogger(__name__)


@dataclass
class _HedgedPosition:
    """In-memory record of a single hedged option position."""

    instrument_name: str
    contracts: float
    kind: str  # "call" | "put"
    underlying: str  # "BTC", etc.


class DeltaHedgeManager:
    """Subscribes to ticker pushes and rebalances the perp leg on threshold breach."""

    def __init__(
        self,
        thalex: ExchangeAdapter,
        hyperliquid: ExchangeAdapter,
        hedger: Optional[DeltaHedger] = None,
    ) -> None:
        self.thalex = thalex
        self.hyperliquid = hyperliquid
        self.hedger = hedger or DeltaHedger()
        self._positions: dict[str, _HedgedPosition] = {}

    async def add_position(
        self,
        instrument_name: str,
        contracts: float,
        kind: str,
        underlying: str,
    ) -> None:
        """Register a position and subscribe to its ticker channel.

        Called from the bot engine immediately after a ``long_call_delta_hedged``
        or ``long_put_delta_hedged`` decision opens a Thalex order.
        """
        self._positions[instrument_name] = _HedgedPosition(
            instrument_name=instrument_name,
            contracts=contracts,
            kind=kind,
            underlying=underlying,
        )

        async def _callback(payload):
            await self._on_ticker(instrument_name, payload)

        if hasattr(self.thalex, "subscribe_ticker"):
            await self.thalex.subscribe_ticker(instrument_name, _callback)
        logger.info(
            "DeltaHedgeManager added position %s (%s %.4f contracts)",
            instrument_name, kind, contracts,
        )

    async def remove_position(self, instrument_name: str) -> None:
        """Unsubscribe and forget the position. Safe to call on a missing key."""
        self._positions.pop(instrument_name, None)
        if hasattr(self.thalex, "unsubscribe_ticker"):
            await self.thalex.unsubscribe_ticker(instrument_name)
        logger.info("DeltaHedgeManager removed position %s", instrument_name)

    async def _on_ticker(self, instrument_name: str, payload) -> None:
        """Single ticker push handler. Computes drift and rebalances if breached.

        ``payload`` is whatever the Thalex subscription channel pushed for
        this instrument. We use the same defensive parser as ``get_greeks``
        so the manager works regardless of the actual notification layout.
        """
        position = self._positions.get(instrument_name)
        if position is None:
            return

        delta_per_contract = _extract_delta(payload)
        if delta_per_contract is None:
            logger.debug("ticker push for %s lacked delta — skipping", instrument_name)
            return

        # For a long call/put, the perp leg should hold MINUS the option's
        # signed delta exposure. Long call (positive delta) → short perp.
        # Long put (negative delta) → long perp.
        target_perp_delta = -(position.contracts * delta_per_contract)
        current_perp_delta = await self._current_perp_delta(position.underlying)

        action = self.hedger.compute_rebalance(
            target_delta=target_perp_delta,
            current_perp_delta=current_perp_delta,
        )
        if action.side == "noop" or action.contracts_to_trade <= 0:
            logger.debug(
                "delta drift below threshold for %s (target=%.4f current=%.4f)",
                instrument_name, target_perp_delta, current_perp_delta,
            )
            return

        try:
            if action.side == "buy":
                await self.hyperliquid.place_buy_order(position.underlying, action.contracts_to_trade)
            else:
                await self.hyperliquid.place_sell_order(position.underlying, action.contracts_to_trade)
            logger.info(
                "delta rebalance %s %.4f %s on Hyperliquid (target=%.4f current=%.4f)",
                action.side, action.contracts_to_trade, position.underlying,
                target_perp_delta, current_perp_delta,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("delta rebalance failed for %s: %s", instrument_name, exc)

    async def _current_perp_delta(self, underlying: str) -> float:
        """Return the signed BTC perp delta currently held on Hyperliquid.

        Positive = long, negative = short. Reads the live Hyperliquid account
        state and finds the entry for the requested underlying. Returns 0.0
        if no perp position exists.
        """
        try:
            state = await self.hyperliquid.get_user_state()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("get_user_state failed: %s", exc)
            return 0.0

        positions = []
        if isinstance(state, dict):
            positions = state.get("positions") or []
        else:
            positions = getattr(state, "positions", []) or []

        for pos in positions:
            asset = pos.get("asset") if isinstance(pos, dict) else getattr(pos, "asset", None)
            if asset != underlying:
                continue
            size = pos.get("size") if isinstance(pos, dict) else getattr(pos, "size", 0.0)
            side = pos.get("side") if isinstance(pos, dict) else getattr(pos, "side", "long")
            try:
                magnitude = abs(float(size or 0.0))
            except (TypeError, ValueError):
                magnitude = 0.0
            return magnitude if side == "long" else -magnitude
        return 0.0


def _extract_delta(payload) -> Optional[float]:
    """Pull a `delta` field out of an arbitrary ticker notification payload.

    Mirrors the field paths in :func:`thalex_api._extract_greeks` so the
    manager keeps working if Thalex pushes greeks under any of the plausible
    layouts.
    """
    if not isinstance(payload, dict):
        return None
    for path in ((), ("greeks",), ("greek",), ("data",)):
        node = payload
        for segment in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(segment)
        if isinstance(node, dict) and "delta" in node:
            try:
                return float(node["delta"])
            except (TypeError, ValueError):
                continue
    return None
