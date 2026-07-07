import json
from datetime import datetime, timedelta

import pytest

from src.backtest.replay import HoldAgent, build_context, replay
from src.backtest.simulator import Simulator

T0 = datetime(2026, 7, 1, 12, 0, 0)


def make_cycle(ts, prices):
    return {
        "timestamp": ts,
        "sections": {
            asset: {"asset": asset, "current_price": price}
            for asset, price in prices.items()
        },
    }


class ScriptedAgent:
    def __init__(self, script):
        self.script = list(script)
        self.contexts = []

    def decide_trade(self, assets, context):
        self.contexts.append(context)
        decisions = self.script.pop(0) if self.script else []
        return {"reasoning": "scripted", "trade_decisions": decisions}


def test_replay_buy_then_tp_produces_report():
    cycles = [
        make_cycle(T0, {"BTC": 100.0}),
        make_cycle(T0 + timedelta(minutes=5), {"BTC": 112.0}),
        make_cycle(T0 + timedelta(minutes=10), {"BTC": 112.0}),
    ]
    agent = ScriptedAgent([
        [{"asset": "BTC", "action": "buy", "allocation_usd": 1000.0,
          "tp_price": 110.0, "sl_price": 95.0}],
        [],
        [],
    ])
    report = replay(cycles, agent, starting_equity=10_000.0, slippage_bps=0.0, fee_bps=10.0)

    assert report["cycle_count"] == 3
    assert report["trade_count"] == 1
    assert report["win_rate"] == pytest.approx(1.0)
    assert report["total_fees"] == pytest.approx(2.1)
    assert report["final_equity"] == pytest.approx(10_097.9)
    assert report["total_return_pct"] == pytest.approx(0.979)
    assert len(agent.contexts) == 3


def test_context_mirrors_live_engine_shape():
    sim = Simulator(starting_equity=10_000.0, slippage_bps=0.0, fee_bps=10.0)
    sim.apply_decision(
        {"asset": "BTC", "action": "buy", "allocation_usd": 1000.0,
         "tp_price": 110.0, "sl_price": 95.0},
        {"BTC": 100.0}, T0,
    )
    cycle = make_cycle(T0 + timedelta(minutes=5), {"BTC": 105.0, "ETH": 3000.0})
    context = build_context(cycle, sim, invocation_count=2)
    payload = json.loads(context)

    assert list(payload.keys()) == ["invocation", "account", "market_data", "instructions"]
    assert payload["invocation"]["count"] == 2
    account = payload["account"]
    for key in (
        "total_return_pct", "balance", "account_value", "buying_power",
        "sharpe_ratio", "positions", "active_trades", "open_orders",
        "recent_diary", "recent_trade_events", "recent_fills",
    ):
        assert key in account
    assert account["positions"][0]["symbol"] == "BTC"
    assert account["positions"][0]["quantity"] == pytest.approx(10.0)
    assert account["active_trades"][0]["sl_price"] == pytest.approx(95.0)
    assert [s["asset"] for s in payload["market_data"]] == ["BTC", "ETH"]
    assert payload["instructions"]["assets"] == ["BTC", "ETH"]


def test_hold_agent_plumbing_produces_flat_report():
    cycles = [
        make_cycle(T0 + timedelta(minutes=5 * i), {"BTC": 100.0 + i})
        for i in range(4)
    ]
    report = replay(cycles, HoldAgent(), starting_equity=5_000.0)

    assert report["trade_count"] == 0
    assert report["total_return_pct"] == pytest.approx(0.0)
    assert report["total_fees"] == pytest.approx(0.0)
    assert report["cycle_count"] == 4


def test_replay_skips_assets_without_price():
    cycles = [
        make_cycle(T0, {"BTC": 100.0, "ETH": None}),
        make_cycle(T0 + timedelta(minutes=5), {"BTC": 101.0}),
    ]
    agent = ScriptedAgent([
        [{"asset": "ETH", "action": "buy", "allocation_usd": 1000.0,
          "tp_price": 110.0, "sl_price": 95.0}],
        [],
    ])
    report = replay(cycles, agent, starting_equity=10_000.0)
    assert report["trade_count"] == 0
    assert report["total_fees"] == pytest.approx(0.0)
    first_payload = json.loads(agent.contexts[0])
    assert first_payload["instructions"]["assets"] == ["BTC"]
