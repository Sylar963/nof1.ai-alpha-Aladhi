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

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, TypeVar

from src.backend.agent.decision_schema import TradeDecision
from src.backend.config_loader import CONFIG
from src.backend.trading.exchange_adapter import ExchangeAdapter, OrderResult
from src.backend.trading.hyperliquid_api import HyperliquidAPI
from src.backend.trading.options import OptionIntent


# Accepted canonical statuses from ExchangeAdapter.OrderResult — matches the
# contract declared on :class:`OrderResult` ("ok | filled | resting | rejected
# | error"). "resting" means the order is live on the book but not yet filled,
# which is a legitimate success for credit/iron-condor flows (we want the GTC
# limit to rest and collect the credit). Anything outside this set is treated
# as a rejection so the bot never records a phantom fill.
_ACCEPTED_ORDER_STATUSES = {"ok", "filled", "resting", "partially_filled"}

# Strictly-filled statuses. Delta-hedged flows require the option leg to be
# filled (not merely resting) before opening a perp hedge, because hedging a
# pending order would leave the book in an inconsistent state if the option
# never fills.
_FILLED_ORDER_STATUSES = {"filled"}


def _order_ok(order) -> tuple[bool, bool, str]:
    """Return ``(accepted, filled, reason)`` for a submitted venue order.

    - ``accepted``: the venue received and acknowledged the order. A
      credit/iron-condor flow can proceed on this.
    - ``filled``: the order actually crossed and took a position. Delta-
      hedged flows MUST gate the perp hedge on ``filled==True``; opening a
      hedge against a resting-only option leaves the book with a naked
      perp if the option never fills.
    - ``reason``: human-readable rejection detail when ``accepted`` is
      False; empty string otherwise.

    Works for both normalized :class:`OrderResult` (Thalex) and the raw
    Hyperliquid SDK dict — Hyperliquid rejections surface as a per-status
    ``{"error": "..."}`` entry inside ``response.data.statuses`` with the
    top-level status still ``"ok"``, which is exactly the shape
    :meth:`HyperliquidAPI.parse_order_response` knows how to decode.
    """
    if order is None:
        return False, False, "no order result returned"
    if isinstance(order, OrderResult):
        status = str(order.status or "").lower()
        if status in _FILLED_ORDER_STATUSES:
            return True, True, ""
        if status in _ACCEPTED_ORDER_STATUSES:
            return True, False, ""
        err = order.error or status or "unknown"
        return False, False, f"status={status!r} error={err}"
    if isinstance(order, dict):
        accepted, reason = HyperliquidAPI.parse_order_response(order)
        if not accepted:
            return False, False, reason
        # ``parse_order_response`` treats both ``filled`` and ``resting`` as
        # success; introspect the statuses to tell them apart for the
        # delta-hedge gate.
        filled = False
        try:
            statuses = order["response"]["data"]["statuses"]
            filled = any(
                isinstance(st, dict) and "filled" in st for st in statuses
            )
        except (KeyError, TypeError):
            filled = False
        return True, filled, ""
    return False, False, f"unknown order type: {type(order).__name__}"


def _filled_amount(order: Any, requested: float) -> float:
    """Best-effort filled quantity for a submitted Thalex leg.

    Prefers the explicit ``filled_amount`` attribute ThalexAPI attaches to its
    OrderResults; falls back to the raw payload, then to the status
    (``filled`` ⇒ fully filled, anything else ⇒ nothing filled).
    """
    fa = getattr(order, "filled_amount", None)
    if fa is not None:
        try:
            return max(min(float(fa), float(requested)), 0.0)
        except (TypeError, ValueError):
            pass
    if isinstance(order, OrderResult):
        raw = order.raw if isinstance(order.raw, dict) else {}
        for key in ("filled_amount", "filled", "fill_amount", "cumulative_filled", "total_filled"):
            value = raw.get(key)
            if value is None:
                continue
            try:
                return max(min(float(value), float(requested)), 0.0)
            except (TypeError, ValueError):
                continue
        if str(order.status or "").lower() == "filled":
            return float(requested)
        return 0.0
    return float(requested)


# Default tenor ladder for the multi-tenor target_gamma_btc expansion path.
# Spreads gamma across short, medium, and longer-dated expiries so a single
# decision builds curve exposure rather than concentrating in one tenor.
_DEFAULT_GAMMA_TENORS: tuple[int, ...] = (7, 14, 30, 60)

# Smallest contract increment Thalex accepts for BTC options.
_THALEX_MIN_CONTRACT: float = 0.001

# Sanity ceiling for per-contract gamma. ATM BTC option gamma is ~1e-4–1e-5
# per contract; anything above this is a corrupt or mis-scaled greek and the
# leg must be refused rather than sized off nonsense.
_MAX_SANE_GAMMA_PER_CONTRACT: float = 0.01

# Exponential-backoff retry config for adapter calls in the multi-tenor
# submit path. Aligned with HyperliquidAPI._retry's defaults so the whole
# project uses one set of numbers for transient failure handling.
_RETRY_MAX_ATTEMPTS: int = 3
_RETRY_BACKOFF_BASE_SECONDS: float = 0.5


logger = logging.getLogger(__name__)


_T = TypeVar("_T")


async def _retry_async(
    coro_factory: Callable[[], Awaitable[_T]],
    *,
    max_attempts: int = _RETRY_MAX_ATTEMPTS,
    backoff_base: float = _RETRY_BACKOFF_BASE_SECONDS,
    description: str = "adapter call",
) -> _T:
    """Run an async coroutine factory with exponential-backoff retry.

    Used by the multi-tenor submit path so a single transient adapter
    hiccup doesn't unwind a whole curve build. The factory is invoked
    fresh on every attempt (so any internal state from a failed attempt
    is discarded).

    Sleep schedule: ``backoff_base × 2**attempt`` between attempts. With
    the defaults (``base=0.5s``, ``max_attempts=3``) that's 0.5s, 1.0s
    between attempts — bounded so a hung loop can't deadlock the surface
    refresh task.

    Raises whatever the underlying coroutine raised on the final attempt.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            if attempt < max_attempts - 1:
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying",
                    description, attempt + 1, max_attempts, exc,
                )
                await asyncio.sleep(backoff_base * (2 ** attempt))
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    description, max_attempts, exc,
                )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{description} retry exited without result or exception")


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

    def __init__(
        self,
        thalex: ExchangeAdapter,
        hyperliquid: ExchangeAdapter,
        delta_hedge_enabled: bool = True,
    ) -> None:
        self.thalex = thalex
        self.hyperliquid = hyperliquid
        self._delta_hedge_enabled = bool(delta_hedge_enabled)

    def set_delta_hedge_enabled(self, enabled: bool) -> None:
        self._delta_hedge_enabled = bool(enabled)

    def is_delta_hedge_enabled(self) -> bool:
        return self._delta_hedge_enabled

    async def _submit_multi_leg_entry_order(
        self,
        instrument_name: str,
        side: str,
        contracts: float,
    ):
        """Submit a spread/condor leg in a way that can rest on the book.

        Defined-risk entries often need a working limit rather than IOC. When
        the Thalex adapter exposes its native limit-order helper, use GTC so a
        spread leg can rest instead of coming back as an immediate IOC
        cancellation. Test doubles and non-Thalex adapters fall back to the
        generic adapter methods.
        """
        submit_limit = getattr(self.thalex, "_submit_limit_order", None)
        entry_limit_price = getattr(self.thalex, "_entry_limit_price", None)
        if callable(submit_limit) and callable(entry_limit_price):
            from thalex import TimeInForce

            limit_price = await entry_limit_price(instrument_name, side, 0.01)
            return await submit_limit(
                instrument_name=instrument_name,
                amount=contracts,
                side=side,
                limit_price=limit_price,
                time_in_force=TimeInForce.GTC,
                description=f"Thalex {side} {instrument_name} (multi-leg)",
            )
        if side == "sell":
            return await self.thalex.place_sell_order(instrument_name, contracts)
        return await self.thalex.place_buy_order(instrument_name, contracts)

    async def execute(self, decision: TradeDecision, open_positions_count: int) -> ExecutionResult:
        if decision.venue != "thalex":
            return ExecutionResult(ok=False, reason=f"OptionsExecutor only handles thalex, got {decision.venue}")
        if not decision.strategy:
            return ExecutionResult(ok=False, reason="missing strategy")

        if decision.strategy in {"long_call_delta_hedged", "long_put_delta_hedged"}:
            return await self._execute_delta_hedged(decision, open_positions_count)

        # All multi-leg defined-risk plays go through the same path: each leg
        # is resolved to its own instrument and submitted in order. Iron condor
        # is just a 4-leg credit spread (two verticals, one each side).
        if decision.strategy in {
            "credit_put_spread",
            "credit_call_spread",
            "iron_condor",
            "vol_arb",
        }:
            return await self._execute_multi_leg(decision, open_positions_count)

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
        accepted, _filled, reason = _order_ok(order)
        if not accepted:
            return ExecutionResult(ok=False, reason=f"thalex {side} {instrument_name} rejected: {reason}")
        # Credit spread / iron condor flows are OK with a resting order —
        # the limit is supposed to wait on the book to capture the credit.
        return ExecutionResult(ok=True, thalex_orders=[order])

    async def _execute_delta_hedged(
        self,
        decision: TradeDecision,
        open_positions_count: int,
    ) -> ExecutionResult:
        # Multi-tenor curve build via target_gamma_btc takes precedence over
        # the single-leg path: when the LLM declares a gamma target, expand
        # into N legs across the default tenor ladder before hitting the
        # single-tenor execution path.
        if decision.target_gamma_btc and decision.target_gamma_btc > 0:
            return await self._execute_delta_hedged_multi_tenor(decision, open_positions_count)

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

        delta_per_contract = await self._lookup_delta(instrument_name)
        if delta_per_contract is None:
            return ExecutionResult(
                ok=False,
                reason=f"missing live delta for {instrument_name}",
            )

        thalex_order = await self.thalex.place_buy_order(instrument_name, contracts)
        accepted, _fully_filled, reason = _order_ok(thalex_order)
        if not accepted:
            return ExecutionResult(
                ok=False,
                reason=f"thalex buy {instrument_name} rejected: {reason}",
            )
        filled_contracts = _filled_amount(thalex_order, contracts)
        if filled_contracts <= 0:
            # Hedging a resting option leaves the book naked on perp if the
            # option never fills. Cancel the pending leg so the operator can
            # re-enter when liquidity supports a fill.
            await self._cancel_and_reverse_leg(thalex_order, instrument_name, "buy", contracts)
            return ExecutionResult(
                ok=False,
                reason=(
                    f"thalex buy {instrument_name} accepted but not filled; "
                    "refusing to hedge a pending option"
                ),
            )
        if filled_contracts < contracts:
            # Partial fill: cancel the resting remainder so the hedge matches
            # the position actually held.
            await self._cancel_resting_remainder(
                thalex_order, instrument_name, filled_contracts, contracts
            )

        if not self._delta_hedge_enabled:
            return ExecutionResult(ok=True, thalex_orders=[thalex_order], hyperliquid_orders=[])

        # Determine hedge direction + size from the live delta and the
        # quantity that actually filled (never the requested contracts).
        hedge_size = abs(filled_contracts * delta_per_contract)
        hl_orders = []
        if hedge_size > 0:
            hedge_asset = decision.underlying or "BTC"
            try:
                if (decision.kind or "call") == "call":
                    # long call ⇒ positive delta ⇒ hedge by SHORTING perp
                    hl_orders.append(
                        await _retry_async(
                            lambda: self.hyperliquid.place_sell_order(hedge_asset, hedge_size),
                            description=f"hyperliquid sell {hedge_asset} hedge",
                        )
                    )
                else:
                    # long put ⇒ negative delta ⇒ hedge by going LONG perp
                    hl_orders.append(
                        await _retry_async(
                            lambda: self.hyperliquid.place_buy_order(hedge_asset, hedge_size),
                            description=f"hyperliquid buy {hedge_asset} hedge",
                        )
                    )
            except Exception as exc:  # pylint: disable=broad-except
                await self._cancel_and_reverse_leg(thalex_order, instrument_name, "buy", contracts)
                return ExecutionResult(
                    ok=False,
                    reason=f"hyperliquid hedge failed for {hedge_asset}: {exc}",
                )
            # Exchange could have accepted the HTTP call but rejected the order
            # (e.g. insufficient margin on perps). Validate + unwind the option
            # leg if we can't actually run hedged. Hedges are market orders —
            # they should be ``filled`` immediately; a merely-accepted
            # (resting) hedge means the book is effectively unhedged, so we
            # unwind rather than pretend the position is safe.
            if hl_orders:
                accepted, filled, reason = _order_ok(hl_orders[-1])
            else:
                accepted, filled, reason = False, False, "no hedge order"
            if not accepted or not filled:
                await self._cancel_and_reverse_leg(thalex_order, instrument_name, "buy", contracts)
                detail = reason if not accepted else "hedge accepted but not filled"
                return ExecutionResult(
                    ok=False,
                    reason=f"hyperliquid hedge rejected for {hedge_asset}: {detail}",
                )

        return ExecutionResult(ok=True, thalex_orders=[thalex_order], hyperliquid_orders=hl_orders)

    async def _cancel_resting_remainder(
        self,
        order: Any,
        instrument_name: str,
        filled: float,
        requested: float,
    ) -> None:
        """Cancel the unfilled remainder of an accepted order by id."""
        if filled >= requested:
            return
        order_id = str(getattr(order, "order_id", "") or "") if order is not None else ""
        if not order_id:
            logger.warning(
                "unwind: no order_id for %s — cannot cancel resting remainder (%.4f of %.4f unfilled)",
                instrument_name, requested - filled, requested,
            )
            return
        try:
            await self.thalex.cancel_order(instrument_name, order_id)
            logger.info(
                "unwind: cancelled resting %s order_id=%s (%.4f of %.4f unfilled)",
                instrument_name, order_id, requested - filled, requested,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "unwind: cancel %s order_id=%s failed: %s — manual intervention may be needed",
                instrument_name, order_id, exc,
            )

    async def _submit_reduce_only_close(self, instrument_name: str, side: str, contracts: float):
        """Close a filled leg with reduce-only semantics when the venue supports it."""
        submit_limit = getattr(self.thalex, "_submit_limit_order", None)
        entry_limit_price = getattr(self.thalex, "_entry_limit_price", None)
        if callable(submit_limit) and callable(entry_limit_price):
            limit_price = await entry_limit_price(instrument_name, side, 0.01)
            return await submit_limit(
                instrument_name=instrument_name,
                amount=contracts,
                side=side,
                limit_price=limit_price,
                reduce_only=True,
                description=f"Thalex {side} {instrument_name} (unwind)",
            )
        if side == "sell":
            return await self.thalex.place_sell_order(instrument_name, contracts)
        return await self.thalex.place_buy_order(instrument_name, contracts)

    async def _cancel_and_reverse_leg(
        self,
        order: Any,
        instrument_name: str,
        side: str,
        requested_contracts: float,
    ) -> None:
        """Best-effort rollback for one submitted leg.

        Cancels any resting (unfilled) remainder by order id so the GTC limit
        can't fill later, then reverses ONLY the quantity that actually
        filled. Never places an opposite-side order for contracts that were
        never held — that would self-match against the resting order or leave
        a naked short.
        """
        filled = _filled_amount(order, requested_contracts)
        await self._cancel_resting_remainder(order, instrument_name, filled, requested_contracts)
        if filled <= 0:
            return
        close_side = "buy" if side == "sell" else "sell"
        try:
            await self._submit_reduce_only_close(instrument_name, close_side, filled)
            logger.info(
                "unwind: reversed %s %s (%.4f filled contracts)",
                side, instrument_name, filled,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "unwind: reverse %s %s failed: %s — manual intervention may be needed",
                side, instrument_name, exc,
            )

    async def _unwind_multi_leg_orders(
        self,
        submitted_legs: list[tuple[str, str, float, Any]],
    ) -> None:
        """Best-effort rollback for partially-submitted multi-leg structures."""
        for instrument_name, side, contracts, order in reversed(submitted_legs):
            await self._cancel_and_reverse_leg(order, instrument_name, side, contracts)

    async def _execute_delta_hedged_multi_tenor(
        self,
        decision: TradeDecision,
        open_positions_count: int,
    ) -> ExecutionResult:
        """Distribute a target gamma exposure across the default tenor ladder.

        Two distinct phases for safety:

        **Stage phase** — resolve EVERY tenor's instrument via
        ``self.thalex.resolve_intent`` upfront, *before* submitting any
        orders. If any resolve returns ``None`` or raises, the whole
        expansion aborts cleanly with zero orders sent. This makes a
        half-built curve impossible.

        Sizing converts the BTC gamma target into contracts using each
        resolved instrument's live per-contract gamma:
        ``contracts_i = (target_gamma_btc / len(tenors)) / gamma_i``.
        Missing, non-positive, or absurd gamma (>
        ``_MAX_SANE_GAMMA_PER_CONTRACT``) fails the whole build closed —
        the gamma target is NEVER spent directly as a contract count.
        Per-leg contracts are rounded to the venue min increment and
        clamped to ``CONFIG["thalex_max_contracts_per_trade"]``.

        **Submit phase** — once all tenors are staged, place each Thalex
        buy + perp hedge with retry (:func:`_retry_async`, exponential
        backoff via ``_RETRY_MAX_ATTEMPTS`` and ``_RETRY_BACKOFF_BASE_SECONDS``).
        If a leg fails after retries, :meth:`_unwind_multi_tenor_legs`
        submits compensating orders for the legs that already landed
        (best-effort rollback to flat).

        The position-count cap is checked **once** against the existing
        open count before any leg lands — the gamma build is treated as
        a single logical position for cap purposes (not N positions).
        """
        underlying = decision.underlying or decision.asset
        kind = decision.kind or "call"
        tenors = _DEFAULT_GAMMA_TENORS
        per_leg_gamma = (decision.target_gamma_btc or 0.0) / len(tenors)
        max_contracts = float(
            CONFIG.get("thalex_max_contracts_per_trade", 0.1) or 0.1
        )

        ok, reason = self.thalex.preflight(  # type: ignore[attr-defined]
            underlying=underlying,
            contracts=_THALEX_MIN_CONTRACT,
            open_positions_count=open_positions_count,
        )
        if not ok:
            return ExecutionResult(ok=False, reason=reason)

        # ------------------------------------------------------------------
        # STAGE PHASE — resolve all tenors before any submit happens.
        # ------------------------------------------------------------------
        staged_legs: list[tuple[int, str, float, float]] = []
        for tenor in tenors:
            intent = OptionIntent(
                underlying=underlying,
                kind=kind,
                tenor_days=tenor,
                target_strike=decision.target_strike,
                target_delta=decision.target_delta,
            )
            try:
                instrument_name = await self.thalex.resolve_intent(intent)  # type: ignore[attr-defined]
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "multi-tenor resolve_intent raised for %s tenor=%d: %s",
                    kind, tenor, exc,
                )
                return ExecutionResult(
                    ok=False,
                    reason=f"resolve_intent failed for {kind} tenor={tenor}: {exc}",
                )
            if not instrument_name:
                logger.error(
                    "multi-tenor resolve_intent returned no instrument for %s tenor=%d",
                    kind, tenor,
                )
                return ExecutionResult(
                    ok=False,
                    reason=f"no instrument for {kind} tenor={tenor}",
                )

            delta_per_contract = await self._lookup_delta(instrument_name)
            if delta_per_contract is None:
                logger.error(
                    "multi-tenor missing live delta for %s tenor=%d",
                    instrument_name,
                    tenor,
                )
                return ExecutionResult(
                    ok=False,
                    reason=f"missing live delta for {instrument_name}",
                )

            gamma_per_contract = await self._lookup_gamma(instrument_name)
            if (
                gamma_per_contract is None
                or gamma_per_contract <= 0
                or gamma_per_contract > _MAX_SANE_GAMMA_PER_CONTRACT
            ):
                logger.error(
                    "multi-tenor unusable gamma %r for %s tenor=%d — refusing leg",
                    gamma_per_contract, instrument_name, tenor,
                )
                return ExecutionResult(
                    ok=False,
                    reason=(
                        f"unusable per-contract gamma {gamma_per_contract!r} for "
                        f"{instrument_name}; refusing to size leg from gamma target"
                    ),
                )

            raw_contracts = per_leg_gamma / gamma_per_contract
            rounded = round(raw_contracts / _THALEX_MIN_CONTRACT) * _THALEX_MIN_CONTRACT
            leg_contracts = max(_THALEX_MIN_CONTRACT, min(max_contracts, rounded))

            ok, reason = self.thalex.preflight(  # type: ignore[attr-defined]
                underlying=underlying,
                contracts=leg_contracts,
                open_positions_count=open_positions_count,
            )
            if not ok:
                return ExecutionResult(
                    ok=False,
                    reason=f"leg rejected for {instrument_name}: {reason}",
                )

            staged_legs.append((tenor, instrument_name, delta_per_contract, leg_contracts))

        # ------------------------------------------------------------------
        # SUBMIT PHASE — orders go out only now, with retry + unwind.
        # ------------------------------------------------------------------
        submitted_thalex: list[tuple[str, float, Any]] = []  # (instrument, contracts, order)
        submitted_hl: list[tuple[str, float]] = []            # (side, size)
        thalex_orders = []
        hl_orders = []

        for tenor, instrument_name, delta_per_contract, leg_contracts in staged_legs:
            # Submit Thalex leg with retry.
            try:
                thalex_order = await _retry_async(
                    lambda iname=instrument_name, c=leg_contracts: self.thalex.place_buy_order(
                        iname, c
                    ),
                    description=f"thalex buy {instrument_name}",
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "thalex submit failed for %s after retries: %s — unwinding",
                    instrument_name, exc,
                )
                await self._unwind_multi_tenor_legs(
                    submitted_thalex, submitted_hl, underlying, kind,
                )
                return ExecutionResult(
                    ok=False,
                    reason=f"thalex submit failed for {instrument_name}: {exc}",
                )
            accepted, filled, reason = _order_ok(thalex_order)
            if not accepted:
                logger.error(
                    "thalex order rejected for %s: %s — unwinding prior legs",
                    instrument_name, reason,
                )
                await self._unwind_multi_tenor_legs(
                    submitted_thalex, submitted_hl, underlying, kind,
                )
                return ExecutionResult(
                    ok=False,
                    reason=f"thalex order rejected for {instrument_name}: {reason}",
                )
            if self._delta_hedge_enabled and not filled:
                logger.error(
                    "thalex order %s accepted but not filled; refusing to hedge pending leg",
                    instrument_name,
                )
                await self._unwind_multi_tenor_legs(
                    submitted_thalex + [(instrument_name, leg_contracts, thalex_order)],
                    submitted_hl,
                    underlying,
                    kind,
                )
                return ExecutionResult(
                    ok=False,
                    reason=(
                        f"thalex order {instrument_name} accepted but not filled; "
                        "refusing to hedge a pending option"
                    ),
                )
            thalex_orders.append(thalex_order)
            submitted_thalex.append((instrument_name, leg_contracts, thalex_order))

            if not self._delta_hedge_enabled:
                continue

            hedge_size = abs(leg_contracts * delta_per_contract)
            if hedge_size <= 0:
                continue

            # Submit perp hedge with retry. Failures unwind both the
            # already-submitted hedges AND all already-submitted thalex legs.
            hedge_side = "sell" if kind == "call" else "buy"
            try:
                if hedge_side == "sell":
                    hl_order = await _retry_async(
                        lambda hs=hedge_size: self.hyperliquid.place_sell_order(
                            underlying, hs
                        ),
                        description=f"hyperliquid sell {underlying} hedge",
                    )
                else:
                    hl_order = await _retry_async(
                        lambda hs=hedge_size: self.hyperliquid.place_buy_order(
                            underlying, hs
                        ),
                        description=f"hyperliquid buy {underlying} hedge",
                    )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "hyperliquid hedge failed for %s after retries: %s — unwinding",
                    underlying, exc,
                )
                await self._unwind_multi_tenor_legs(
                    submitted_thalex, submitted_hl, underlying, kind,
                )
                return ExecutionResult(
                    ok=False,
                    reason=f"hyperliquid hedge failed for {underlying}: {exc}",
                )
            accepted, filled, reason = _order_ok(hl_order)
            if not accepted or not filled:
                detail = reason if not accepted else "hedge accepted but not filled"
                logger.error(
                    "hyperliquid hedge rejected for %s: %s — unwinding",
                    underlying, detail,
                )
                await self._unwind_multi_tenor_legs(
                    submitted_thalex, submitted_hl, underlying, kind,
                )
                return ExecutionResult(
                    ok=False,
                    reason=f"hyperliquid hedge rejected for {underlying}: {detail}",
                )
            hl_orders.append(hl_order)
            submitted_hl.append((hedge_side, hedge_size))

        return ExecutionResult(
            ok=True,
            thalex_orders=thalex_orders,
            hyperliquid_orders=hl_orders,
        )

    async def _unwind_multi_tenor_legs(
        self,
        submitted_thalex: list[tuple[str, float, Any]],
        submitted_hl: list[tuple[str, float]],
        underlying: str,
        kind: str,
    ) -> None:
        """Best-effort compensating rollback for partially-submitted multi-tenor legs.

        Called from :meth:`_execute_delta_hedged_multi_tenor` when a Thalex
        submit or perp hedge fails after retries. For each Thalex leg the
        resting remainder is cancelled by order id and only the filled
        quantity is reversed (see :meth:`_cancel_and_reverse_leg`). Perp
        hedges are flattened with the opposite-direction order.

        Each unwind is wrapped in its own try/except so a single failure
        in the rollback path doesn't abort the rest of the unwind. The
        unwind is best-effort by design: if it can't reach the venue, the
        operator gets a loud log line and the position stays half-open
        for manual intervention.
        """
        for instrument_name, contracts, order in submitted_thalex:
            await self._cancel_and_reverse_leg(order, instrument_name, "buy", contracts)

        for side, size in submitted_hl:
            try:
                if side == "sell":
                    await self.hyperliquid.place_buy_order(underlying, size)
                else:
                    await self.hyperliquid.place_sell_order(underlying, size)
                logger.info("unwind: reversed hyperliquid hedge (%.4f %s)", size, underlying)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "unwind: hyperliquid reverse %s failed: %s — manual intervention may be needed",
                    underlying, exc,
                )

    async def _execute_multi_leg(
        self,
        decision: TradeDecision,
        open_positions_count: int,
    ) -> ExecutionResult:
        """Execute any multi-leg defined-risk strategy (credit spreads, iron
        condors, vol arb, calendar spreads).

        Each leg is resolved independently — when ``leg.tenor_days`` is set it
        overrides the decision-level tenor (calendar/diagonal spreads). Risk
        caps are applied per-leg before any orders are submitted.
        """
        if not decision.legs:
            return ExecutionResult(
                ok=False,
                reason=f"{decision.strategy} requires legs[] to be non-empty",
            )

        # Risk caps apply per-leg
        for leg in decision.legs:
            ok, reason = self.thalex.preflight(  # type: ignore[attr-defined]
                underlying=decision.underlying or decision.asset,
                contracts=leg.contracts,
                open_positions_count=open_positions_count,
            )
            if not ok:
                return ExecutionResult(ok=False, reason=f"leg rejected: {reason}")

        staged_legs: list[tuple[str, str, float]] = []
        thalex_orders = []
        for leg in decision.legs:
            intent = OptionIntent(
                underlying=decision.underlying or decision.asset,
                kind=leg.kind,
                # Per-leg tenor wins; decision-level is the fallback
                tenor_days=leg.tenor_days or decision.tenor_days or 14,
                target_strike=leg.target_strike,
                target_delta=leg.target_delta,
            )
            try:
                instrument_name = await self.thalex.resolve_intent(intent)  # type: ignore[attr-defined]
            except Exception as exc:  # pylint: disable=broad-except
                return ExecutionResult(ok=False, reason=f"resolve_intent failed for leg {leg}: {exc}")
            if not instrument_name:
                return ExecutionResult(ok=False, reason=f"no instrument for leg {leg}")
            staged_legs.append((instrument_name, leg.side, leg.contracts))

        if decision.strategy in {"credit_put_spread", "credit_call_spread", "iron_condor"}:
            net_credit = await self._estimate_net_premium(staged_legs)
            if net_credit is not None:
                min_credit = float(CONFIG.get("options_min_net_credit_usd", 0.0))
                if net_credit < min_credit:
                    return ExecutionResult(
                        ok=False,
                        reason=(
                            f"{decision.strategy} expected net credit ${net_credit:.2f} "
                            f"below minimum ${min_credit:.2f} — structure would not "
                            "collect the intended credit"
                        ),
                    )

        # Protective (long) legs go in before short legs so the book is never
        # short an option without its defined-risk wing.
        ordered_legs = sorted(staged_legs, key=lambda leg: 0 if leg[1] == "buy" else 1)

        submitted_legs: list[tuple[str, str, float, Any]] = []
        for instrument_name, side, contracts in ordered_legs:
            try:
                order = await self._submit_multi_leg_entry_order(
                    instrument_name,
                    side,
                    contracts,
                )
            except Exception as exc:  # pylint: disable=broad-except
                await self._unwind_multi_leg_orders(submitted_legs)
                return ExecutionResult(
                    ok=False,
                    reason=f"thalex submit failed for {instrument_name}: {exc}",
                )
            accepted, _filled, reason = _order_ok(order)
            if not accepted:
                await self._unwind_multi_leg_orders(submitted_legs)
                return ExecutionResult(
                    ok=False,
                    reason=f"thalex leg rejected ({side} {instrument_name}): {reason}",
                )
            submitted_legs.append((instrument_name, side, contracts, order))
            thalex_orders.append(order)

        unfilled_legs = await self._await_multi_leg_fills(submitted_legs)
        if unfilled_legs:
            unfilled_set = {id(leg[3]) for leg in unfilled_legs}
            filled_siblings = [leg for leg in submitted_legs if id(leg[3]) not in unfilled_set]
            for instrument_name, side, contracts, order in unfilled_legs:
                await self._cancel_and_reverse_leg(order, instrument_name, side, contracts)
            await self._unwind_multi_leg_orders(filled_siblings)
            names = ", ".join(f"{side} {name}" for name, side, _c, _o in unfilled_legs)
            return ExecutionResult(
                ok=False,
                reason=f"multi-leg fill timeout: unfilled legs cancelled and siblings unwound ({names})",
                thalex_orders=thalex_orders,
            )

        return ExecutionResult(ok=True, thalex_orders=thalex_orders)

    async def _estimate_net_premium(
        self,
        staged_legs: list[tuple[str, str, float]],
    ) -> Optional[float]:
        """Expected net premium (USD, sells positive) from live leg quotes.

        Returns ``None`` when the adapter can't quote (test doubles, missing
        pricing helper) — callers then skip the credit gate rather than
        blocking on unknowable data.
        """
        entry_limit_price = getattr(self.thalex, "_entry_limit_price", None)
        if not callable(entry_limit_price):
            return None
        total = 0.0
        for instrument_name, side, contracts in staged_legs:
            try:
                price = await entry_limit_price(instrument_name, side, 0.0)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "net-premium estimate failed for %s: %s — skipping credit gate",
                    instrument_name, exc,
                )
                return None
            if side == "sell":
                total += float(price) * float(contracts)
            else:
                total -= float(price) * float(contracts)
        return total

    async def _await_multi_leg_fills(
        self,
        submitted_legs: list[tuple[str, str, float, Any]],
    ) -> list[tuple[str, str, float, Any]]:
        """Poll open orders until every leg fills or the timeout expires.

        A leg is considered done when it reported a full fill at submit time
        or when its order id no longer appears in the venue's open orders.
        Returns the legs still resting when the watchdog gives up.
        """
        pending: list[tuple[str, str, float, Any]] = []
        for leg in submitted_legs:
            instrument_name, side, contracts, order = leg
            if _filled_amount(order, contracts) >= contracts:
                continue
            if not str(getattr(order, "order_id", "") or ""):
                continue
            pending.append(leg)
        if not pending:
            return []

        get_open_orders = getattr(self.thalex, "get_open_orders", None)
        if not callable(get_open_orders):
            return []

        timeout = max(float(CONFIG.get("options_fill_timeout_seconds", 30)), 0.0)
        deadline = time.monotonic() + timeout
        while True:
            try:
                open_orders = await get_open_orders()
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("fill watchdog: get_open_orders failed: %s", exc)
                open_orders = None
            if isinstance(open_orders, list):
                open_ids = {
                    str(o.get("order_id") or o.get("id") or "")
                    for o in open_orders
                    if isinstance(o, dict)
                }
                pending = [
                    leg for leg in pending
                    if str(getattr(leg[3], "order_id", "") or "") in open_ids
                ]
            remaining = deadline - time.monotonic()
            if not pending or remaining <= 0:
                return pending
            await asyncio.sleep(min(1.0, remaining))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _lookup_delta(self, instrument_name: str) -> Optional[float]:
        """Return the live delta for an instrument, or ``None`` when unknown.

        The hedge engine must not trade on guessed ATM deltas. If no live greek
        is available, the caller should fail closed or degrade hedge management
        until the feed recovers.
        """
        delta_map = getattr(self.thalex, "delta_per_position", {}) or {}
        if instrument_name in delta_map:
            return float(delta_map[instrument_name])

        if hasattr(self.thalex, "get_greeks"):
            try:
                greeks = await self.thalex.get_greeks(instrument_name)
                if isinstance(greeks, dict) and "delta" in greeks:
                    return float(greeks["delta"])
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("get_greeks failed for %s: %s", instrument_name, exc)

        logger.error("Delta unknown for %s — refusing to hedge on guessed value", instrument_name)
        return None

    async def _lookup_gamma(self, instrument_name: str) -> Optional[float]:
        """Return the live per-contract gamma for an instrument, or ``None``.

        The multi-tenor sizer converts a BTC gamma target into contracts
        with this value. Never guess: without a live gamma the caller must
        refuse the leg rather than fall back to treating the gamma target
        as a contract count.
        """
        if hasattr(self.thalex, "get_greeks"):
            try:
                greeks = await self.thalex.get_greeks(instrument_name)
                if isinstance(greeks, dict) and greeks.get("gamma") is not None:
                    return float(greeks["gamma"])
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("get_greeks failed for %s: %s", instrument_name, exc)

        logger.error("Gamma unknown for %s — refusing to size on guessed value", instrument_name)
        return None


class DeltaHedger:
    """Computes when (and how much) to rebalance the perp delta hedge.

    Threshold-based gamma scalping: the rebalance trade only fires when the
    drift between target and current perp delta exceeds ``threshold`` BTC.
    Below the threshold the drift is left alone — when the underlying
    mean-reverts, the option's delta returns to where it started and no
    perp churn ever happened. Above the threshold, the rebalance trade
    re-hedges at the new (worse) level; subsequent rebalance-to-target
    trades automatically capture the pullback by buying low / selling high.

    Default threshold is 0.02 BTC: small enough to capture pullbacks during
    normal BTC moves, large enough that quiet markets don't churn perp fees.
    """

    def __init__(self, threshold: float = 0.02) -> None:
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
