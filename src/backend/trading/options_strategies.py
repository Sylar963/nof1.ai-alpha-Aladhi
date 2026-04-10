"""Strategy executor + delta-hedge rebalancer for Thalex options.

The :class:`OptionsExecutor` is the bridge between a parsed
:class:`TradeDecision` and the underlying ExchangeAdapters. For each strategy
it:

1. Runs the Thalex risk-cap preflight on every leg.
2. Resolves the option intent to a real instrument (via the Thalex adapter).
3. Submits the leg orders to Thalex.
4. For delta-hedged longs, opens an offsetting Hyperliquid perp position sized
   by the option's delta × contracts.

The :class:`DeltaHedger` is a pure-logic helper that the bot engine calls on a
schedule to rebalance the perp leg whenever the position's delta drifts beyond
a configurable threshold (default 0.05 BTC equivalent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.backend.agent.decision_schema import TradeDecision
from src.backend.trading.exchange_adapter import ExchangeAdapter
from src.backend.trading.options import OptionIntent


@dataclass
class ExecutionResult:
    """Outcome of executing one TradeDecision through the OptionsExecutor."""

    ok: bool
    reason: str = ""
    thalex_orders: list = None  # type: ignore[assignment]
    hyperliquid_orders: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.thalex_orders is None:
            self.thalex_orders = []
        if self.hyperliquid_orders is None:
            self.hyperliquid_orders = []


@dataclass
class HedgeAction:
    """A single perp-leg trade the delta hedger wants to submit."""

    side: str  # "buy" | "sell" | "noop"
    contracts_to_trade: float


class OptionsExecutor:
    """Executes options decisions across Thalex (and Hyperliquid for hedging)."""

    def __init__(self, thalex: ExchangeAdapter, hyperliquid: ExchangeAdapter) -> None:
        self.thalex = thalex
        self.hyperliquid = hyperliquid

    async def execute(self, decision: TradeDecision, open_positions_count: int) -> ExecutionResult:
        if decision.venue != "thalex":
            return ExecutionResult(ok=False, reason=f"OptionsExecutor only handles thalex, got {decision.venue}")
        if not decision.strategy:
            return ExecutionResult(ok=False, reason="missing strategy")

        if decision.strategy == "credit_put":
            return await self._execute_single_leg(decision, open_positions_count, side_override="sell")
        if decision.strategy in {"long_call_delta_hedged", "long_put_delta_hedged"}:
            return await self._execute_delta_hedged(decision, open_positions_count)
        if decision.strategy == "credit_spread":
            return await self._execute_spread(decision, open_positions_count)
        return ExecutionResult(ok=False, reason=f"unknown strategy {decision.strategy}")

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    async def _execute_single_leg(
        self,
        decision: TradeDecision,
        open_positions_count: int,
        side_override: Optional[str] = None,
    ) -> ExecutionResult:
        contracts = decision.contracts or 0.0
        ok, reason = self.thalex.preflight(  # type: ignore[attr-defined]
            underlying=decision.underlying or decision.asset,
            contracts=contracts,
            open_positions_count=open_positions_count,
        )
        if not ok:
            return ExecutionResult(ok=False, reason=reason)

        intent = decision.to_option_intent()
        if intent is None:
            return ExecutionResult(ok=False, reason="cannot derive OptionIntent from decision")
        instrument_name = await self.thalex.resolve_intent(intent)  # type: ignore[attr-defined]
        if not instrument_name:
            return ExecutionResult(ok=False, reason="no matching instrument on Thalex")

        side = side_override or decision.action
        if side == "sell":
            order = await self.thalex.place_sell_order(instrument_name, contracts)
        else:
            order = await self.thalex.place_buy_order(instrument_name, contracts)
        return ExecutionResult(ok=True, thalex_orders=[order])

    async def _execute_delta_hedged(
        self,
        decision: TradeDecision,
        open_positions_count: int,
    ) -> ExecutionResult:
        contracts = decision.contracts or 0.0
        ok, reason = self.thalex.preflight(  # type: ignore[attr-defined]
            underlying=decision.underlying or decision.asset,
            contracts=contracts,
            open_positions_count=open_positions_count,
        )
        if not ok:
            return ExecutionResult(ok=False, reason=reason)

        intent = decision.to_option_intent()
        if intent is None:
            return ExecutionResult(ok=False, reason="cannot derive OptionIntent from decision")
        instrument_name = await self.thalex.resolve_intent(intent)  # type: ignore[attr-defined]
        if not instrument_name:
            return ExecutionResult(ok=False, reason="no matching instrument on Thalex")

        thalex_order = await self.thalex.place_buy_order(instrument_name, contracts)

        # Determine hedge direction + size from the (possibly cached) delta.
        delta_per_contract = self._lookup_delta(instrument_name, decision.kind or "call")
        hedge_size = abs(contracts * delta_per_contract)
        hl_orders = []
        if hedge_size > 0:
            if (decision.kind or "call") == "call":
                # long call ⇒ positive delta ⇒ hedge by SHORTING perp
                hl_orders.append(await self.hyperliquid.place_sell_order(decision.underlying or "BTC", hedge_size))
            else:
                # long put ⇒ negative delta ⇒ hedge by going LONG perp
                hl_orders.append(await self.hyperliquid.place_buy_order(decision.underlying or "BTC", hedge_size))

        return ExecutionResult(ok=True, thalex_orders=[thalex_order], hyperliquid_orders=hl_orders)

    async def _execute_spread(
        self,
        decision: TradeDecision,
        open_positions_count: int,
    ) -> ExecutionResult:
        if not decision.legs:
            return ExecutionResult(ok=False, reason="credit_spread requires legs[]")

        # Risk caps apply per-leg
        for leg in decision.legs:
            ok, reason = self.thalex.preflight(  # type: ignore[attr-defined]
                underlying=decision.underlying or decision.asset,
                contracts=leg.contracts,
                open_positions_count=open_positions_count,
            )
            if not ok:
                return ExecutionResult(ok=False, reason=f"leg rejected: {reason}")

        thalex_orders = []
        for leg in decision.legs:
            intent = OptionIntent(
                underlying=decision.underlying or decision.asset,
                kind=leg.kind,
                tenor_days=decision.tenor_days or 14,
                target_strike=leg.target_strike,
                target_delta=leg.target_delta,
            )
            instrument_name = await self.thalex.resolve_intent(intent)  # type: ignore[attr-defined]
            if not instrument_name:
                return ExecutionResult(ok=False, reason=f"no instrument for leg {leg}")
            if leg.side == "sell":
                order = await self.thalex.place_sell_order(instrument_name, leg.contracts)
            else:
                order = await self.thalex.place_buy_order(instrument_name, leg.contracts)
            thalex_orders.append(order)

        return ExecutionResult(ok=True, thalex_orders=thalex_orders)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lookup_delta(self, instrument_name: str, kind: str) -> float:
        """Best-effort delta lookup. Falls back to a 0.5 / -0.5 default for ATM options.

        Real implementations enrich the Thalex adapter with a ticker call that
        returns greeks; the test fakes seed `delta_per_position` directly.
        """
        delta_map = getattr(self.thalex, "delta_per_position", {}) or {}
        if instrument_name in delta_map:
            return float(delta_map[instrument_name])
        return 0.5 if kind == "call" else -0.5


class DeltaHedger:
    """Computes when (and how much) to rebalance the perp delta hedge."""

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def compute_rebalance(self, target_delta: float, current_perp_delta: float) -> HedgeAction:
        """Return the trade needed to bring perp delta back to target.

        ``target_delta`` is the delta of the perp leg we WANT to hold (signed).
        ``current_perp_delta`` is the signed delta currently held on Hyperliquid.
        Negative = short, positive = long.
        """
        drift = target_delta - current_perp_delta
        if abs(drift) < self.threshold:
            return HedgeAction(side="noop", contracts_to_trade=0.0)
        if drift > 0:
            return HedgeAction(side="buy", contracts_to_trade=drift)
        return HedgeAction(side="sell", contracts_to_trade=abs(drift))
