"""Tests for the DeltaHedgeManager: subscribe-on-open, rebalance-on-tick,
unsubscribe-on-close lifecycle.

These tests use fake adapters that record subscriptions, fire synthetic ticker
notifications, and observe perp orders. No event-loop hacks needed — the
manager's ``_on_ticker`` handler is awaited directly to simulate a single
push from the receive loop."""

import asyncio
from dataclasses import dataclass, field

import pytest

from src.backend.trading.exchange_adapter import (
    AccountState,
    ExchangeAdapter,
    OrderResult,
    PositionSnapshot,
)
from src.backend.trading.delta_hedge_manager import DeltaHedgeManager
from src.backend.trading.options_strategies import DeltaHedger


@dataclass
class _PerpOrder:
    side: str
    amount: float


class FakeThalexForHedge(ExchangeAdapter):
    venue = "thalex"

    def __init__(self):
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []
        self.callbacks: dict[str, callable] = {}

    async def subscribe_ticker(self, instrument_name, callback):
        self.subscribed.append(instrument_name)
        self.callbacks[instrument_name] = callback

    async def unsubscribe_ticker(self, instrument_name):
        self.unsubscribed.append(instrument_name)
        self.callbacks.pop(instrument_name, None)

    # ExchangeAdapter abstract method stubs (unused in these tests)
    async def place_buy_order(self, asset, amount, slippage=0.01):
        return OrderResult(venue=self.venue, order_id="x", asset=asset, side="buy", amount=amount, status="ok")
    async def place_sell_order(self, asset, amount, slippage=0.01):
        return OrderResult(venue=self.venue, order_id="x", asset=asset, side="sell", amount=amount, status="ok")
    async def place_take_profit(self, asset, is_buy, amount, tp_price):
        return OrderResult(venue=self.venue, order_id="", asset=asset, side="tp", amount=amount, status="not_supported")
    async def place_stop_loss(self, asset, is_buy, amount, sl_price):
        return OrderResult(venue=self.venue, order_id="", asset=asset, side="sl", amount=amount, status="not_supported")
    async def cancel_order(self, asset, order_id): return {"status": "ok"}
    async def cancel_all_orders(self, asset): return {"status": "ok"}
    async def get_open_orders(self): return []
    async def get_recent_fills(self, limit=50): return []
    async def get_user_state(self):
        return AccountState(venue=self.venue, balance=0.0, total_value=0.0, positions=[])
    async def get_current_price(self, asset): return 0.0


class FakeHyperliquidForHedge(ExchangeAdapter):
    venue = "hyperliquid"

    def __init__(self, perp_size: float = 0.0):
        self.calls: list[_PerpOrder] = []
        self.perp_size = perp_size  # signed: positive=long, negative=short
        self.btc_price = 60000.0

    async def place_buy_order(self, asset, amount, slippage=0.01):
        self.calls.append(_PerpOrder("buy", amount))
        self.perp_size += amount
        return OrderResult(venue=self.venue, order_id="h", asset=asset, side="buy", amount=amount, status="ok")

    async def place_sell_order(self, asset, amount, slippage=0.01):
        self.calls.append(_PerpOrder("sell", amount))
        self.perp_size -= amount
        return OrderResult(venue=self.venue, order_id="h", asset=asset, side="sell", amount=amount, status="ok")

    async def place_take_profit(self, asset, is_buy, amount, tp_price):
        return OrderResult(venue=self.venue, order_id="", asset=asset, side="tp", amount=amount, status="ok")
    async def place_stop_loss(self, asset, is_buy, amount, sl_price):
        return OrderResult(venue=self.venue, order_id="", asset=asset, side="sl", amount=amount, status="ok")
    async def cancel_order(self, asset, order_id): return {"status": "ok"}
    async def cancel_all_orders(self, asset): return {"status": "ok"}
    async def get_open_orders(self): return []
    async def get_recent_fills(self, limit=50): return []

    async def get_user_state(self):
        positions = []
        if self.perp_size != 0:
            positions.append(PositionSnapshot(
                venue="hyperliquid",
                asset="BTC",
                side="long" if self.perp_size > 0 else "short",
                size=abs(self.perp_size),
                entry_price=self.btc_price,
                current_price=self.btc_price,
                unrealized_pnl=0.0,
            ))
        return AccountState(venue="hyperliquid", balance=10000, total_value=10000, positions=positions)

    async def get_current_price(self, asset): return self.btc_price


# ---------------------------------------------------------------------------
# Lifecycle: add / remove
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_position_subscribes_to_ticker():
    thalex = FakeThalexForHedge()
    hl = FakeHyperliquidForHedge(perp_size=-0.025)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger())

    await manager.add_position(
        instrument_name="BTC-10MAY26-65000-C",
        contracts=0.05,
        kind="call",
        underlying="BTC",
    )

    assert "BTC-10MAY26-65000-C" in thalex.subscribed
    assert "BTC-10MAY26-65000-C" in thalex.callbacks


@pytest.mark.asyncio
async def test_remove_position_unsubscribes():
    thalex = FakeThalexForHedge()
    hl = FakeHyperliquidForHedge()
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger())

    await manager.add_position("BTC-10MAY26-65000-C", 0.05, "call", "BTC")
    await manager.remove_position("BTC-10MAY26-65000-C")

    assert "BTC-10MAY26-65000-C" in thalex.unsubscribed
    assert "BTC-10MAY26-65000-C" not in thalex.callbacks


# ---------------------------------------------------------------------------
# Rebalance triggers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ticker_push_above_threshold_fires_perp_rebalance():
    """A delta change that breaches 0.02 BTC drift must trigger a perp trade."""
    thalex = FakeThalexForHedge()
    # Initial perp hedge: short 0.025 BTC (matching 0.05 contracts × 0.5 ATM delta)
    hl = FakeHyperliquidForHedge(perp_size=-0.025)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.add_position("BTC-10MAY26-65000-C", contracts=0.05, kind="call", underlying="BTC")

    # Underlying rallied → option delta jumped from 0.5 to 0.8.
    # Target perp = -contracts × delta = -0.05 × 0.8 = -0.04
    # Drift = -0.04 - (-0.025) = -0.015 → still below threshold
    # Push another tick where delta = 0.9 → target = -0.045, drift = -0.020 → at threshold (no fire)
    # Push delta = 1.0 → target = -0.05, drift = -0.025 → breach
    callback = thalex.callbacks["BTC-10MAY26-65000-C"]
    await callback({"greeks": {"delta": 1.0}})

    assert len(hl.calls) == 1
    assert hl.calls[0].side == "sell"  # need MORE short to match deeper delta
    assert hl.calls[0].amount == pytest.approx(0.025)


@pytest.mark.asyncio
async def test_ticker_push_below_threshold_is_a_noop():
    """A small delta wiggle (< 0.02 BTC) must NOT trigger a perp trade."""
    thalex = FakeThalexForHedge()
    hl = FakeHyperliquidForHedge(perp_size=-0.025)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.add_position("BTC-10MAY26-65000-C", contracts=0.05, kind="call", underlying="BTC")

    # delta moved from 0.5 to 0.6 → target = -0.03, drift = -0.005 → below threshold
    callback = thalex.callbacks["BTC-10MAY26-65000-C"]
    await callback({"greeks": {"delta": 0.6}})

    assert hl.calls == []


@pytest.mark.asyncio
async def test_pullback_unwinds_excess_hedge():
    """After a delta surge fires a rebalance, a delta reversion must close the excess."""
    thalex = FakeThalexForHedge()
    hl = FakeHyperliquidForHedge(perp_size=-0.025)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.add_position("BTC-10MAY26-65000-C", contracts=0.05, kind="call", underlying="BTC")
    callback = thalex.callbacks["BTC-10MAY26-65000-C"]

    # Surge: delta → 1.0, fire short to bring perp from -0.025 to -0.05
    await callback({"greeks": {"delta": 1.0}})
    assert hl.perp_size == pytest.approx(-0.05)

    # Reversion: delta → 0.5, target = -0.025, drift = +0.025 → fire BUY 0.025
    await callback({"greeks": {"delta": 0.5}})
    assert len(hl.calls) == 2
    assert hl.calls[1].side == "buy"
    assert hl.calls[1].amount == pytest.approx(0.025)
    # Net: shorted 0.025 high, bought 0.025 low — pullback captured.
    assert hl.perp_size == pytest.approx(-0.025)


@pytest.mark.asyncio
async def test_callback_for_unknown_instrument_is_ignored():
    """Manager must not crash if a stale notification arrives for a removed position."""
    thalex = FakeThalexForHedge()
    hl = FakeHyperliquidForHedge()
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger())

    # Direct invocation of _on_ticker with no registered position
    await manager._on_ticker("BTC-NONEXISTENT", {"greeks": {"delta": 0.5}})
    assert hl.calls == []
