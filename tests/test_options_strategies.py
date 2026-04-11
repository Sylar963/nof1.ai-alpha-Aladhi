"""Tests for options strategy execution and the delta-hedge auto-rebalancer.

These tests use in-memory fake adapters and a fake clock so the strategy logic
can be exercised end-to-end without touching the network. They cover:

- Credit put: validates the order is sized + routed correctly.
- Credit spread: produces two leg orders in the right direction.
- Long call delta-hedged: places a Thalex buy AND a Hyperliquid perp short.
- Long put delta-hedged: Thalex buy AND Hyperliquid perp long.
- Delta-hedge rebalance: only fires when |drift| > threshold.
- Risk caps: oversized contracts are rejected before any order goes out.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.backend.agent.decision_schema import parse_decision
from src.backend.trading.exchange_adapter import (
    AccountState,
    ExchangeAdapter,
    OrderResult,
    PositionSnapshot,
)
from src.backend.trading.options import RiskCaps
from src.backend.trading.options_strategies import (
    DeltaHedger,
    OptionsExecutor,
)


# ---------------------------------------------------------------------------
# Fake adapters
# ---------------------------------------------------------------------------


@dataclass
class _FakeOrder:
    venue: str
    method: str
    asset: str
    amount: float
    extra: dict = field(default_factory=dict)


class FakeThalex(ExchangeAdapter):
    venue = "thalex"

    def __init__(self):
        self.calls: list[_FakeOrder] = []
        self.instruments_by_intent: dict[tuple, str] = {}
        self.positions: list[PositionSnapshot] = []
        self.delta_per_position: dict[str, float] = {}
        self.greeks_by_instrument: dict[str, dict] = {}  # used by get_greeks fallthrough
        self.risk_caps = RiskCaps(
            max_contracts_per_trade=0.1,
            max_open_positions=3,
            allowed_underlyings=["BTC"],
        )

    async def get_greeks(self, instrument_name):
        return self.greeks_by_instrument.get(instrument_name, {})

    async def resolve_intent(self, intent):
        key = (intent.underlying, intent.kind, intent.tenor_days, intent.target_strike)
        return self.instruments_by_intent.get(key, "BTC-10MAY26-65000-C")

    def preflight(self, underlying, contracts, open_positions_count):
        from src.backend.trading.options import validate_options_order
        return validate_options_order(underlying, contracts, open_positions_count, self.risk_caps)

    async def place_buy_order(self, asset, amount, slippage=0.01):
        self.calls.append(_FakeOrder(self.venue, "buy", asset, amount))
        return OrderResult(venue=self.venue, order_id="t1", asset=asset, side="buy", amount=amount, status="ok")

    async def place_sell_order(self, asset, amount, slippage=0.01):
        self.calls.append(_FakeOrder(self.venue, "sell", asset, amount))
        return OrderResult(venue=self.venue, order_id="t2", asset=asset, side="sell", amount=amount, status="ok")

    async def place_take_profit(self, asset, is_buy, amount, tp_price): return OrderResult(venue=self.venue, order_id="", asset=asset, side="tp", amount=amount, status="not_supported")
    async def place_stop_loss(self, asset, is_buy, amount, sl_price): return OrderResult(venue=self.venue, order_id="", asset=asset, side="sl", amount=amount, status="not_supported")
    async def cancel_order(self, asset, order_id): return {"status": "ok"}
    async def cancel_all_orders(self, asset): return {"status": "ok"}
    async def get_open_orders(self): return []
    async def get_recent_fills(self, limit=50): return []

    async def get_user_state(self):
        return AccountState(venue=self.venue, balance=10000, total_value=10000, positions=list(self.positions))

    async def get_current_price(self, asset):
        return 1250.0


class FakeHyperliquid(ExchangeAdapter):
    venue = "hyperliquid"

    def __init__(self):
        self.calls: list[_FakeOrder] = []
        self.perp_position_size: float = 0.0  # signed: positive=long, negative=short
        self.btc_price = 60000.0

    async def place_buy_order(self, asset, amount, slippage=0.01):
        self.calls.append(_FakeOrder(self.venue, "buy", asset, amount))
        self.perp_position_size += amount
        return OrderResult(venue=self.venue, order_id="h1", asset=asset, side="buy", amount=amount, status="ok")

    async def place_sell_order(self, asset, amount, slippage=0.01):
        self.calls.append(_FakeOrder(self.venue, "sell", asset, amount))
        self.perp_position_size -= amount
        return OrderResult(venue=self.venue, order_id="h2", asset=asset, side="sell", amount=amount, status="ok")

    async def place_take_profit(self, asset, is_buy, amount, tp_price): return OrderResult(venue=self.venue, order_id="", asset=asset, side="tp", amount=amount, status="ok")
    async def place_stop_loss(self, asset, is_buy, amount, sl_price): return OrderResult(venue=self.venue, order_id="", asset=asset, side="sl", amount=amount, status="ok")
    async def cancel_order(self, asset, order_id): return {"status": "ok"}
    async def cancel_all_orders(self, asset): return {"status": "ok"}
    async def get_open_orders(self): return []
    async def get_recent_fills(self, limit=50): return []
    async def get_user_state(self):
        return AccountState(
            venue="hyperliquid",
            balance=10000,
            total_value=10000,
            positions=[
                PositionSnapshot(
                    venue="hyperliquid",
                    asset="BTC",
                    side="long" if self.perp_position_size >= 0 else "short",
                    size=abs(self.perp_position_size),
                    entry_price=self.btc_price,
                    current_price=self.btc_price,
                    unrealized_pnl=0.0,
                )
            ] if self.perp_position_size != 0 else [],
        )

    async def get_current_price(self, asset):
        return self.btc_price


# ---------------------------------------------------------------------------
# Strategy execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_credit_put_executes_a_single_sell():
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "sell",
        "strategy": "credit_put",
        "underlying": "BTC",
        "kind": "put",
        "tenor_days": 14,
        "target_strike": 55000,
        "contracts": 0.05,
        "rationale": "premium sale",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    assert len(thalex.calls) == 1
    assert thalex.calls[0].method == "sell"
    assert thalex.calls[0].amount == 0.05
    assert hl.calls == []  # no perp hedge for credit put


@pytest.mark.asyncio
async def test_long_call_delta_hedged_places_thalex_buy_and_hyperliquid_short():
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    # Pretend the option has delta 0.5 → contracts 0.05 → hedge size = 0.025 BTC short
    thalex.delta_per_position["BTC-10MAY26-65000-C"] = 0.5

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.05,
        "rationale": "vol cheap",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    assert len(thalex.calls) == 1 and thalex.calls[0].method == "buy"
    assert len(hl.calls) == 1 and hl.calls[0].method == "sell"
    # Hedge size = contracts × delta = 0.05 × 0.5 = 0.025
    assert hl.calls[0].amount == pytest.approx(0.025)


@pytest.mark.asyncio
async def test_long_put_delta_hedged_places_thalex_buy_and_hyperliquid_long():
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    thalex.delta_per_position["BTC-10MAY26-55000-P"] = -0.4

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_put_delta_hedged",
        "underlying": "BTC",
        "kind": "put",
        "tenor_days": 30,
        "target_strike": 55000,
        "contracts": 0.05,
        "rationale": "downside hedge",
    })
    thalex.instruments_by_intent[("BTC", "put", 30, 55000.0)] = "BTC-10MAY26-55000-P"
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    assert len(hl.calls) == 1
    assert hl.calls[0].method == "buy"
    # Hedge size = contracts × |delta| = 0.05 × 0.4 = 0.02
    assert hl.calls[0].amount == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_lookup_delta_falls_through_to_get_greeks_when_cache_empty():
    """When delta_per_position has no entry, the executor must call get_greeks."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    # No delta_per_position seed; get_greeks returns the live delta instead.
    thalex.greeks_by_instrument["BTC-10MAY26-65000-C"] = {"delta": 0.42}

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.05,
        "rationale": "use real greeks",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    # Hedge size = 0.05 × 0.42 = 0.021
    assert hl.calls[0].amount == pytest.approx(0.021)


@pytest.mark.asyncio
async def test_credit_spread_places_two_legs_in_correct_directions():
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "sell",
        "strategy": "credit_spread",
        "underlying": "BTC",
        "tenor_days": 14,
        "legs": [
            {"kind": "put", "side": "sell", "target_strike": 55000, "contracts": 0.1},
            {"kind": "put", "side": "buy", "target_strike": 50000, "contracts": 0.1},
        ],
        "rationale": "defined risk premium sale",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    methods = [c.method for c in thalex.calls]
    assert methods == ["sell", "buy"]


@pytest.mark.asyncio
async def test_executor_rejects_oversized_order_via_risk_caps():
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.5,  # > 0.1 cap
        "rationale": "oversized",
    })
    result = await executor.execute(decision, open_positions_count=0)
    assert result.ok is False
    assert "max_contracts_per_trade" in result.reason
    assert thalex.calls == []
    assert hl.calls == []


@pytest.mark.asyncio
async def test_executor_rejects_when_max_open_positions_reached():
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.05,
        "rationale": "x",
    })
    result = await executor.execute(decision, open_positions_count=3)
    assert result.ok is False
    assert "max_open_positions" in result.reason


# ---------------------------------------------------------------------------
# Delta hedger
# ---------------------------------------------------------------------------


def test_delta_hedger_no_action_when_drift_below_threshold():
    hedger = DeltaHedger(threshold=0.05)
    action = hedger.compute_rebalance(
        target_delta=-0.025,  # short 0.025 BTC equivalent
        current_perp_delta=-0.02,  # currently short 0.02
    )
    # |drift| = 0.005 < 0.05 → no action
    assert action.contracts_to_trade == pytest.approx(0.0)


def test_delta_hedger_acts_when_drift_above_threshold():
    hedger = DeltaHedger(threshold=0.05)
    action = hedger.compute_rebalance(
        target_delta=-0.10,
        current_perp_delta=-0.02,
    )
    # drift = -0.10 - (-0.02) = -0.08 → need to short 0.08 more BTC
    assert action.side == "sell"
    assert action.contracts_to_trade == pytest.approx(0.08)


def test_delta_hedger_handles_long_drift():
    hedger = DeltaHedger(threshold=0.05)
    action = hedger.compute_rebalance(
        target_delta=0.0,
        current_perp_delta=-0.10,
    )
    # Need to buy 0.10 BTC to flatten the short
    assert action.side == "buy"
    assert action.contracts_to_trade == pytest.approx(0.10)


def test_delta_hedger_default_threshold_is_002_btc():
    """Default threshold is 0.02 BTC drift — small enough to capture pullbacks
    but loose enough that quiet markets don't churn perp fees."""
    assert DeltaHedger().threshold == pytest.approx(0.02)


def test_delta_hedger_default_threshold_fires_at_0025_drift():
    """A 0.025 BTC drift breaches the new default and must trigger a rebalance."""
    hedger = DeltaHedger()  # uses default 0.02
    action = hedger.compute_rebalance(target_delta=-0.025, current_perp_delta=0.0)
    assert action.side == "sell"
    assert action.contracts_to_trade == pytest.approx(0.025)


def test_delta_hedger_default_threshold_holds_at_0015_drift():
    """A 0.015 BTC drift sits below the new default and must be a no-op."""
    hedger = DeltaHedger()  # uses default 0.02
    action = hedger.compute_rebalance(target_delta=-0.015, current_perp_delta=0.0)
    assert action.side == "noop"
    assert action.contracts_to_trade == pytest.approx(0.0)
