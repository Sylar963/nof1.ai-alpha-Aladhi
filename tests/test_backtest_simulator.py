from datetime import datetime, timedelta

import pytest

from src.backtest.simulator import Simulator

T0 = datetime(2026, 7, 1, 12, 0, 0)
T1 = T0 + timedelta(minutes=5)
T2 = T0 + timedelta(minutes=10)


def make_sim(**kwargs):
    params = {
        "starting_equity": 10_000.0,
        "slippage_bps": 0.0,
        "fee_bps": 10.0,
        "max_risk_per_trade_pct": 1.0,
        "max_gross_leverage": 3.0,
    }
    params.update(kwargs)
    return Simulator(**params)


def buy(asset="BTC", allocation=1000.0, tp=110.0, sl=95.0):
    return {
        "asset": asset,
        "action": "buy",
        "allocation_usd": allocation,
        "tp_price": tp,
        "sl_price": sl,
    }


def sell(asset="BTC", allocation=1000.0, tp=90.0, sl=105.0):
    return {
        "asset": asset,
        "action": "sell",
        "allocation_usd": allocation,
        "tp_price": tp,
        "sl_price": sl,
    }


def test_buy_then_tp_nets_pnl_minus_both_fees():
    sim = make_sim()
    sim.apply_decision(buy(), {"BTC": 100.0}, T0)

    pos = sim.positions["BTC"]
    assert pos.size == pytest.approx(10.0)
    assert pos.entry_price == pytest.approx(100.0)
    assert sim.fees_paid == pytest.approx(1.0)

    sim.check_exits({"BTC": 112.0}, T1)

    assert "BTC" not in sim.positions
    trade = sim.closed_trades[0]
    assert trade.reason == "tp"
    assert trade.exit_price == pytest.approx(110.0)
    assert trade.fees == pytest.approx(1.0 + 1.1)
    assert trade.pnl == pytest.approx(100.0 - 2.1)
    assert sim.equity({}) == pytest.approx(10_097.9)


def test_long_sl_gap_through_fills_at_worse_price():
    sim = make_sim()
    sim.apply_decision(buy(sl=95.0), {"BTC": 100.0}, T0)
    sim.check_exits({"BTC": 90.0}, T1)

    trade = sim.closed_trades[0]
    assert trade.reason == "sl"
    assert trade.exit_price == pytest.approx(90.0)
    assert trade.pnl == pytest.approx(-100.0 - 1.0 - 0.9)


def test_short_sl_gap_through_fills_at_worse_price():
    sim = make_sim()
    sim.apply_decision(sell(sl=105.0, tp=90.0), {"BTC": 100.0}, T0)
    sim.check_exits({"BTC": 112.0}, T1)

    trade = sim.closed_trades[0]
    assert trade.reason == "sl"
    assert trade.exit_price == pytest.approx(112.0)


def test_short_tp_fills_at_tp_level_only_when_crossed():
    sim = make_sim()
    sim.apply_decision(sell(sl=105.0, tp=90.0), {"BTC": 100.0}, T0)
    sim.check_exits({"BTC": 91.0}, T1)
    assert "BTC" in sim.positions

    sim.check_exits({"BTC": 85.0}, T2)
    trade = sim.closed_trades[0]
    assert trade.reason == "tp"
    assert trade.exit_price == pytest.approx(90.0)


def test_flip_closes_then_opens_with_fees_on_both_legs():
    sim = make_sim()
    sim.apply_decision(buy(allocation=1000.0, tp=120.0, sl=95.0), {"BTC": 100.0}, T0)
    sim.apply_decision(sell(allocation=500.0, tp=90.0, sl=105.0), {"BTC": 100.0}, T1)

    assert len(sim.closed_trades) == 1
    flip = sim.closed_trades[0]
    assert flip.reason == "flip"
    assert flip.side == "long"
    assert flip.pnl == pytest.approx(-2.0)

    pos = sim.positions["BTC"]
    assert pos.size == pytest.approx(-5.0)
    assert pos.entry_price == pytest.approx(100.0)
    assert sim.fees_paid == pytest.approx(1.0 + 1.0 + 0.5)


def test_risk_cap_clamps_oversized_allocation():
    sim = make_sim()
    sim.apply_decision(
        buy(allocation=20_000.0, tp=110.0, sl=99.0), {"BTC": 100.0}, T0
    )

    pos = sim.positions["BTC"]
    assert pos.size == pytest.approx(100.0)
    assert len(sim.clamp_notes) == 1
    assert "risk cap" in sim.clamp_notes[0]["note"]


def test_gross_leverage_cap_clamps_allocation():
    sim = make_sim(max_gross_leverage=1.0, max_risk_per_trade_pct=100.0)
    sim.apply_decision(
        buy(allocation=20_000.0, tp=200.0, sl=50.0), {"BTC": 100.0}, T0
    )

    pos = sim.positions["BTC"]
    assert pos.size == pytest.approx(100.0)
    assert any("gross leverage cap" in n["note"] for n in sim.clamp_notes)


def test_missing_sl_decision_is_skipped():
    sim = make_sim()
    decision = {"asset": "BTC", "action": "buy", "allocation_usd": 1000.0, "tp_price": 110.0}
    sim.apply_decision(decision, {"BTC": 100.0}, T0)

    assert sim.positions == {}
    assert len(sim.skipped) == 1
    assert "missing sl_price" in sim.skipped[0]["reason"]
    assert sim.fees_paid == 0.0


def test_inverted_sl_for_buy_is_skipped():
    sim = make_sim()
    sim.apply_decision(buy(sl=105.0), {"BTC": 100.0}, T0)

    assert sim.positions == {}
    assert "must be below entry" in sim.skipped[0]["reason"]


def test_zero_allocation_opposite_direction_is_pure_close():
    sim = make_sim()
    sim.apply_decision(buy(), {"BTC": 100.0}, T0)
    sim.apply_decision({"asset": "BTC", "action": "sell", "allocation_usd": 0}, {"BTC": 105.0}, T1)

    assert sim.positions == {}
    trade = sim.closed_trades[0]
    assert trade.reason == "close"
    assert trade.exit_price == pytest.approx(105.0)


def test_zero_allocation_same_direction_is_noop():
    sim = make_sim()
    sim.apply_decision(buy(), {"BTC": 100.0}, T0)
    fees_before = sim.fees_paid
    sim.apply_decision({"asset": "BTC", "action": "buy", "allocation_usd": 0}, {"BTC": 100.0}, T1)

    assert sim.positions["BTC"].size == pytest.approx(10.0)
    assert sim.fees_paid == fees_before
    assert sim.closed_trades == []


def test_same_direction_add_uses_volume_weighted_entry():
    sim = make_sim(max_risk_per_trade_pct=100.0)
    sim.apply_decision(buy(allocation=1000.0), {"BTC": 100.0}, T0)
    sim.apply_decision(
        buy(allocation=1000.0, tp=220.0, sl=190.0), {"BTC": 200.0}, T1
    )

    pos = sim.positions["BTC"]
    assert pos.size == pytest.approx(15.0)
    assert pos.entry_price == pytest.approx((10 * 100 + 5 * 200) / 15)
    assert pos.tp_price == pytest.approx(220.0)
    assert pos.sl_price == pytest.approx(190.0)


def test_slippage_is_adverse_on_both_sides():
    sim = make_sim(slippage_bps=100.0)
    sim.apply_decision(buy(), {"BTC": 100.0}, T0)
    assert sim.positions["BTC"].entry_price == pytest.approx(101.0)

    sim2 = make_sim(slippage_bps=100.0)
    sim2.apply_decision(sell(), {"BTC": 100.0}, T0)
    assert sim2.positions["BTC"].entry_price == pytest.approx(99.0)
