"""Tests for the portfolio-driven DeltaHedgeManager."""

from dataclasses import dataclass

import pytest

from src.backend.trading.delta_hedge_manager import DeltaHedgeManager
from src.backend.trading.exchange_adapter import AccountState, ExchangeAdapter, OrderResult, PositionSnapshot
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
        self.positions: list[PositionSnapshot] = []
        self.greeks_by_instrument: dict[str, dict] = {}

    async def subscribe_ticker(self, instrument_name, callback):
        self.subscribed.append(instrument_name)
        self.callbacks[instrument_name] = callback

    async def unsubscribe_ticker(self, instrument_name):
        self.unsubscribed.append(instrument_name)
        self.callbacks.pop(instrument_name, None)

    def cache_greeks_snapshot(self, instrument_name, payload):
        greeks = payload.get("greeks") or payload.get("greek") or payload.get("data") or payload
        if isinstance(greeks, dict) and greeks.get("delta") is not None:
            self.greeks_by_instrument[instrument_name] = {"delta": float(greeks["delta"])}

    async def get_greeks(self, instrument_name):
        return self.greeks_by_instrument.get(instrument_name, {})

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
        return AccountState(venue=self.venue, balance=0.0, total_value=0.0, positions=list(self.positions))

    async def get_current_price(self, asset): return 0.0


class FakeHyperliquidForHedge(ExchangeAdapter):
    venue = "hyperliquid"

    def __init__(self, perp_size: float = 0.0):
        self.calls: list[_PerpOrder] = []
        self.perp_size = perp_size
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
            positions.append(
                PositionSnapshot(
                    venue="hyperliquid",
                    asset="BTC",
                    side="long" if self.perp_size > 0 else "short",
                    size=abs(self.perp_size),
                    entry_price=self.btc_price,
                    current_price=self.btc_price,
                    unrealized_pnl=0.0,
                )
            )
        return AccountState(venue="hyperliquid", balance=10000, total_value=10000, positions=positions)

    async def get_current_price(self, asset): return self.btc_price


def _option_position(*, instrument_name: str, side: str = "long", size: float = 0.05, delta=None):
    return PositionSnapshot(
        venue="thalex",
        asset="BTC",
        instrument_name=instrument_name,
        side=side,
        size=size,
        entry_price=1000.0,
        current_price=1000.0,
        unrealized_pnl=0.0,
        delta=delta,
    )


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


@pytest.mark.asyncio
async def test_reconcile_subscribes_live_positions_and_hedges_net_delta():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=0.5)]
    hl = FakeHyperliquidForHedge(perp_size=0.0)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()

    assert thalex.subscribed == ["BTC-10MAY26-65000-C"]
    assert len(hl.calls) == 1
    assert hl.calls[0].side == "sell"
    assert hl.calls[0].amount == pytest.approx(0.025)


@pytest.mark.asyncio
async def test_ticker_push_above_threshold_fires_perp_rebalance():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=None)]
    thalex.greeks_by_instrument["BTC-10MAY26-65000-C"] = {"delta": 0.5}
    hl = FakeHyperliquidForHedge(perp_size=-0.025)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()
    callback = thalex.callbacks["BTC-10MAY26-65000-C"]
    await callback({"greeks": {"delta": 1.0}})

    assert len(hl.calls) == 1
    assert hl.calls[0].side == "sell"
    assert hl.calls[0].amount == pytest.approx(0.025)


@pytest.mark.asyncio
async def test_ticker_push_below_threshold_is_a_noop():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=None)]
    thalex.greeks_by_instrument["BTC-10MAY26-65000-C"] = {"delta": 0.5}
    hl = FakeHyperliquidForHedge(perp_size=-0.025)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()
    callback = thalex.callbacks["BTC-10MAY26-65000-C"]
    await callback({"greeks": {"delta": 0.6}})

    assert hl.calls == []


@pytest.mark.asyncio
async def test_pullback_unwinds_excess_hedge():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=None)]
    thalex.greeks_by_instrument["BTC-10MAY26-65000-C"] = {"delta": 0.5}
    hl = FakeHyperliquidForHedge(perp_size=-0.025)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()
    callback = thalex.callbacks["BTC-10MAY26-65000-C"]

    await callback({"greeks": {"delta": 1.0}})
    assert hl.perp_size == pytest.approx(-0.05)

    await callback({"greeks": {"delta": 0.5}})
    assert len(hl.calls) == 2
    assert hl.calls[1].side == "buy"
    assert hl.calls[1].amount == pytest.approx(0.025)
    assert hl.perp_size == pytest.approx(-0.025)


@pytest.mark.asyncio
async def test_spread_rebalances_only_residual_net_delta():
    thalex = FakeThalexForHedge()
    thalex.positions = [
        _option_position(instrument_name="BTC-10MAY26-65000-C", side="long", size=0.1, delta=0.6),
        _option_position(instrument_name="BTC-10MAY26-70000-C", side="short", size=0.05, delta=0.4),
    ]
    hl = FakeHyperliquidForHedge(perp_size=0.0)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()

    assert set(thalex.subscribed) == {"BTC-10MAY26-65000-C", "BTC-10MAY26-70000-C"}
    assert len(hl.calls) == 1
    assert hl.calls[0].side == "sell"
    assert hl.calls[0].amount == pytest.approx(0.04)


@pytest.mark.asyncio
async def test_multi_tenor_book_subscribes_all_legs_and_hedges_once_net():
    thalex = FakeThalexForHedge()
    thalex.positions = [
        _option_position(instrument_name="BTC-17APR26-60000-C", delta=0.5),
        _option_position(instrument_name="BTC-24APR26-60000-C", delta=0.4),
        _option_position(instrument_name="BTC-10MAY26-60000-C", delta=0.3),
        _option_position(instrument_name="BTC-09JUN26-60000-C", delta=0.2),
    ]
    hl = FakeHyperliquidForHedge(perp_size=0.0)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()

    assert set(thalex.subscribed) == {
        "BTC-17APR26-60000-C",
        "BTC-24APR26-60000-C",
        "BTC-10MAY26-60000-C",
        "BTC-09JUN26-60000-C",
    }
    assert len(hl.calls) == 1
    assert hl.calls[0].side == "sell"
    assert hl.calls[0].amount == pytest.approx(0.07)


@pytest.mark.asyncio
async def test_missing_greeks_degrades_underlying_and_skips_rebalance():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=None)]
    hl = FakeHyperliquidForHedge(perp_size=0.0)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()

    assert hl.calls == []
    assert "BTC" in manager.degraded_underlyings


@pytest.mark.asyncio
async def test_status_snapshot_exposes_underlying_metrics():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=0.5)]
    hl = FakeHyperliquidForHedge(perp_size=0.0)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()
    snapshot = manager.get_status_snapshot()

    assert snapshot["health"] == "healthy"
    assert snapshot["tracked_underlyings"] == 1
    assert snapshot["metrics"][0]["underlying"] == "BTC"
    assert snapshot["metrics"][0]["target_perp_delta"] == pytest.approx(-0.025)
    assert snapshot["metrics"][0]["current_perp_delta"] == pytest.approx(-0.025)


@pytest.mark.asyncio
async def test_reconcile_unsubscribes_closed_positions_and_flattens_hedge():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=0.5)]
    hl = FakeHyperliquidForHedge(perp_size=0.0)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()
    assert hl.perp_size == pytest.approx(-0.025)

    thalex.positions = []
    await manager.reconcile()

    assert "BTC-10MAY26-65000-C" in thalex.unsubscribed
    assert len(hl.calls) == 2
    assert hl.calls[1].side == "buy"
    assert hl.calls[1].amount == pytest.approx(0.025)


@pytest.mark.asyncio
async def test_callback_for_unknown_instrument_is_ignored():
    thalex = FakeThalexForHedge()
    hl = FakeHyperliquidForHedge()
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger())

    await manager._on_ticker("BTC-NONEXISTENT", {"greeks": {"delta": 0.5}})

    assert hl.calls == []
