"""Tests for the portfolio-driven DeltaHedgeManager."""

from dataclasses import dataclass
import json
import logging

import pytest

import src.backend.trading.delta_hedge_manager as dhm
from src.backend.config_loader import CONFIG
from src.backend.trading.delta_hedge_manager import DeltaHedgeManager, _executed_amount
from src.backend.trading.exchange_adapter import AccountState, ExchangeAdapter, OrderResult, PositionSnapshot
from src.backend.trading.options_strategies import DeltaHedger


@pytest.fixture(autouse=True)
def _hedge_env(monkeypatch, tmp_path):
    monkeypatch.setattr(dhm, "_DEFAULT_LEDGER_PATH", tmp_path / "hedge_ledger.json")
    monkeypatch.setitem(CONFIG, "hedge_cooldown_seconds", 0)


def _seed_ledger(tmp_path, entries):
    (tmp_path / "hedge_ledger.json").write_text(json.dumps(entries), encoding="utf-8")


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
        self.fail_user_state = False

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
        if self.fail_user_state:
            raise RuntimeError("temporary thalex outage")
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
async def test_ticker_push_above_threshold_fires_perp_rebalance(tmp_path):
    _seed_ledger(tmp_path, {"BTC": -0.025})
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
async def test_ticker_push_below_threshold_is_a_noop(tmp_path):
    _seed_ledger(tmp_path, {"BTC": -0.025})
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
async def test_pullback_unwinds_excess_hedge(tmp_path):
    _seed_ledger(tmp_path, {"BTC": -0.025})
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
async def test_status_snapshot_exposes_unavailable_state_error():
    thalex = FakeThalexForHedge()
    thalex.fail_user_state = True
    hl = FakeHyperliquidForHedge(perp_size=0.0)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()
    snapshot = manager.get_status_snapshot()

    assert snapshot["health"] == "unavailable"
    assert snapshot["state_error"] == "temporary thalex outage"


@pytest.mark.asyncio
async def test_disabled_manager_skips_reconcile_and_reports_disabled_status():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=0.5)]
    hl = FakeHyperliquidForHedge(perp_size=0.0)
    manager = DeltaHedgeManager(
        thalex=thalex,
        hyperliquid=hl,
        hedger=DeltaHedger(threshold=0.02),
        enabled=False,
    )

    orders = await manager.reconcile()
    snapshot = manager.get_status_snapshot()

    assert orders == []
    assert thalex.subscribed == []
    assert hl.calls == []
    assert snapshot["health"] == "disabled"
    assert snapshot["enabled"] is False


@pytest.mark.asyncio
async def test_disabling_manager_unsubscribes_live_tickers():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=0.5)]
    hl = FakeHyperliquidForHedge(perp_size=0.0)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()
    await manager.set_enabled(False)

    assert "BTC-10MAY26-65000-C" in thalex.unsubscribed
    assert manager.get_status_snapshot()["health"] == "disabled"


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
async def test_reconcile_keeps_subscriptions_and_hedge_when_position_state_is_unknown():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=0.5)]
    hl = FakeHyperliquidForHedge(perp_size=0.0)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()
    assert hl.perp_size == pytest.approx(-0.025)

    thalex.fail_user_state = True
    await manager.reconcile()

    assert thalex.unsubscribed == []
    assert hl.perp_size == pytest.approx(-0.025)
    assert len(hl.calls) == 1


@pytest.mark.asyncio
async def test_reconcile_logs_unavailable_state_only_once_until_recovered(caplog):
    thalex = FakeThalexForHedge()
    thalex.fail_user_state = True
    hl = FakeHyperliquidForHedge(perp_size=0.0)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    with caplog.at_level(logging.WARNING):
        await manager.reconcile()
        await manager.reconcile()

    messages = [record.message for record in caplog.records]
    assert messages.count("delta hedge reconcile: thalex get_user_state failed: temporary thalex outage") == 1
    assert messages.count("delta hedge reconcile: skipping all underlyings because Thalex position state is unknown") == 1

    thalex.fail_user_state = False
    await manager.reconcile()
    thalex.fail_user_state = True

    with caplog.at_level(logging.WARNING):
        await manager.reconcile()

    messages = [record.message for record in caplog.records]
    assert messages.count("delta hedge reconcile: thalex get_user_state failed: temporary thalex outage") == 2


@pytest.mark.asyncio
async def test_callback_for_unknown_instrument_is_ignored():
    thalex = FakeThalexForHedge()
    hl = FakeHyperliquidForHedge()
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger())

    await manager._on_ticker("BTC-NONEXISTENT", {"greeks": {"delta": 0.5}})

    assert hl.calls == []


class FakeRawHyperliquid(FakeHyperliquidForHedge):
    """Fake that mimics HyperliquidAPI.get_user_state's raw coin/szi shape."""

    def __init__(self, perp_size: float = 0.0):
        super().__init__(perp_size)
        self.fail_user_state = False

    async def get_user_state(self):
        if self.fail_user_state:
            raise RuntimeError("hl outage")
        positions = []
        if self.perp_size != 0:
            positions.append(
                {
                    "coin": "BTC",
                    "szi": str(self.perp_size),
                    "entryPx": "60000.0",
                    "leverage": {"type": "cross", "value": 20},
                    "unrealizedPnl": "0.0",
                    "unrealized_pnl": 0.0,
                    "pnl": 0.0,
                    "notional_entry": abs(self.perp_size) * 60000.0,
                }
            )
        return {"balance": 10000.0, "total_value": 10000.0, "positions": positions}


class RejectingHyperliquid(FakeHyperliquidForHedge):
    async def place_sell_order(self, asset, amount, slippage=0.01):
        self.calls.append(_PerpOrder("sell", amount))
        return OrderResult(
            venue=self.venue, order_id="", asset=asset, side="sell",
            amount=amount, status="rejected", error="Insufficient margin",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("szi,expected", [(-0.05, -0.05), (0.03, 0.03)])
async def test_current_perp_delta_parses_raw_hl_coin_szi_payload(szi, expected):
    thalex = FakeThalexForHedge()
    hl = FakeRawHyperliquid(perp_size=szi)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger())

    assert await manager._current_perp_delta("BTC") == pytest.approx(expected)
    assert await manager._current_perp_delta("ETH") == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_perp_state_fetch_failure_skips_rebalance():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=0.5)]
    hl = FakeRawHyperliquid()
    hl.fail_user_state = True
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    orders = await manager.reconcile()

    assert orders == []
    assert hl.calls == []
    assert manager._hedge_ledger == {}
    assert "BTC" in manager.degraded_underlyings


@pytest.mark.asyncio
async def test_ledger_residual_leaves_directional_perp_position_untouched():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=0.5)]
    hl = FakeRawHyperliquid(perp_size=0.5)
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()

    assert len(hl.calls) == 1
    assert hl.calls[0].side == "sell"
    assert hl.calls[0].amount == pytest.approx(0.025)
    assert manager._hedge_ledger["BTC"] == pytest.approx(-0.025)
    assert hl.perp_size == pytest.approx(0.475)

    await manager.reconcile()

    assert len(hl.calls) == 1
    assert hl.perp_size == pytest.approx(0.475)


@pytest.mark.asyncio
async def test_failed_hedge_order_leaves_ledger_unchanged():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=0.5)]
    hl = RejectingHyperliquid()
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    orders = await manager.reconcile()

    assert orders == []
    assert len(hl.calls) == 1
    assert manager._hedge_ledger.get("BTC", 0.0) == pytest.approx(0.0)
    snapshot = manager.get_status_snapshot()
    assert snapshot["metrics"][0]["status"] == "rejected"


@pytest.mark.asyncio
async def test_cooldown_suppresses_immediate_second_hedge(monkeypatch):
    monkeypatch.setitem(CONFIG, "hedge_cooldown_seconds", 300)
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=None)]
    thalex.greeks_by_instrument["BTC-10MAY26-65000-C"] = {"delta": 0.5}
    hl = FakeHyperliquidForHedge()
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()
    assert len(hl.calls) == 1

    thalex.greeks_by_instrument["BTC-10MAY26-65000-C"] = {"delta": 1.0}
    await manager.reconcile()

    assert len(hl.calls) == 1
    assert manager._hedge_ledger["BTC"] == pytest.approx(-0.025)


@pytest.mark.asyncio
async def test_residual_below_min_notional_is_acceptable_drift():
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=0.5)]
    hl = FakeHyperliquidForHedge()
    hl.btc_price = 100.0
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    orders = await manager.reconcile()

    assert orders == []
    assert hl.calls == []
    assert manager._hedge_ledger == {}


@pytest.mark.asyncio
async def test_hedge_order_notional_is_capped(monkeypatch):
    monkeypatch.setitem(CONFIG, "hedge_max_order_notional_usd", 600.0)
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=0.5)]
    hl = FakeHyperliquidForHedge()
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()

    assert len(hl.calls) == 1
    assert hl.calls[0].amount == pytest.approx(0.01)
    assert manager._hedge_ledger["BTC"] == pytest.approx(-0.01)


@pytest.mark.asyncio
async def test_hedge_ledger_persists_across_restart(tmp_path):
    thalex = FakeThalexForHedge()
    thalex.positions = [_option_position(instrument_name="BTC-10MAY26-65000-C", delta=0.5)]
    hl = FakeHyperliquidForHedge()
    manager = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))

    await manager.reconcile()
    assert manager._hedge_ledger["BTC"] == pytest.approx(-0.025)
    assert json.loads((tmp_path / "hedge_ledger.json").read_text())["BTC"] == pytest.approx(-0.025)

    manager2 = DeltaHedgeManager(thalex=thalex, hyperliquid=hl, hedger=DeltaHedger(threshold=0.02))
    assert manager2._hedge_ledger["BTC"] == pytest.approx(-0.025)

    orders = await manager2.reconcile()
    assert orders == []
    assert len(hl.calls) == 1


def test_executed_amount_prefers_filled_size_from_raw_response():
    raw = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalSz": "0.02", "avgPx": "60000", "oid": 1}}]}},
    }
    assert _executed_amount(raw, 0.025) == pytest.approx(0.02)
    assert _executed_amount({"status": "ok"}, 0.025) == pytest.approx(0.025)
    result = OrderResult(venue="hyperliquid", order_id="h", asset="BTC", side="sell", amount=0.02, status="ok")
    assert _executed_amount(result, 0.025) == pytest.approx(0.02)
