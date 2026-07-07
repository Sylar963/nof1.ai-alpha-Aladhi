"""Replay orchestrator: feed stored cycles to an agent and simulate the fills."""

import argparse
import json
import sys
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List

from src.backend.utils.prompt_utils import json_default
from src.backtest.metrics import compute_metrics, format_report
from src.backtest.simulator import Simulator
from src.backtest.store import DEFAULT_DB_PATH, load_cycles


class HoldAgent:
    """No-op agent that holds every asset; validates replay plumbing."""

    def decide_trade(self, assets, context):
        return {
            "reasoning": "hold-only replay agent",
            "trade_decisions": [
                {"asset": a, "action": "hold", "rationale": "hold agent"}
                for a in assets
            ],
        }


def _simulated_account(sim: Simulator, prices: Dict[str, float]) -> Dict:
    equity = sim.equity(prices)
    positions = []
    active_trades = []
    for asset, pos in sim.positions.items():
        px = prices.get(asset) or pos.entry_price
        positions.append({
            "symbol": asset,
            "quantity": pos.size,
            "entry_price": pos.entry_price,
            "current_price": px,
            "unrealized_pnl": (px - pos.entry_price) * pos.size,
            "venue": "hyperliquid",
        })
        active_trades.append({
            "venue": "hyperliquid",
            "asset": asset,
            "instrument_name": asset,
            "is_long": pos.size > 0,
            "amount": abs(pos.size),
            "entry_price": pos.entry_price,
            "tp_price": pos.tp_price,
            "sl_price": pos.sl_price,
            "opened_at": pos.opened_at.isoformat(),
        })
    return {
        "total_return_pct": (
            (equity - sim.starting_equity) / sim.starting_equity * 100.0
            if sim.starting_equity else 0.0
        ),
        "balance": sim.cash,
        "account_value": equity,
        "buying_power": {
            "account_value": equity,
            "max_gross_leverage": sim.max_gross_leverage,
        },
        "sharpe_ratio": None,
        "positions": positions,
        "active_trades": active_trades,
        "open_orders": [],
        "recent_diary": [],
        "recent_trade_events": [],
        "recent_fills": [],
    }


def build_context(cycle: Dict, sim: Simulator, invocation_count: int) -> str:
    """Assemble the LLM context JSON in the live engine's OrderedDict shape."""
    sections = {
        a: s for a, s in cycle["sections"].items()
        if s.get("current_price") and float(s["current_price"]) > 0
    }
    assets = sorted(sections.keys())
    prices = {a: float(s["current_price"]) for a, s in sections.items()}
    ts = cycle["timestamp"]
    payload = OrderedDict([
        ("invocation", {
            "count": invocation_count,
            "current_time": ts.isoformat() if isinstance(ts, datetime) else str(ts),
        }),
        ("account", _simulated_account(sim, prices)),
        ("market_data", [sections[a] for a in assets]),
        ("instructions", {
            "assets": assets,
            "note": "Follow the system prompt guidelines strictly",
        }),
    ])
    return json.dumps(payload, default=json_default, indent=2)


def replay(
    cycles: List[Dict],
    agent,
    starting_equity: float = 10_000.0,
    **sim_params,
) -> Dict:
    """Run the agent over stored cycles through the fill simulator.

    ``agent`` is any object exposing
    ``decide_trade(assets, context) -> {"reasoning", "trade_decisions"}``.
    Returns the metrics report dict.
    """
    sim = Simulator(starting_equity=starting_equity, **sim_params)
    for idx, cycle in enumerate(cycles):
        prices = {
            asset: section.get("current_price")
            for asset, section in cycle["sections"].items()
            if section.get("current_price") and float(section["current_price"]) > 0
        }
        timestamp = cycle["timestamp"]
        sim.check_exits(prices, timestamp)
        if not prices:
            sim.mark(prices, timestamp)
            continue
        context = build_context(cycle, sim, idx + 1)
        assets = sorted(prices.keys())
        decisions = agent.decide_trade(assets, context)
        if isinstance(decisions, dict):
            for decision in decisions.get("trade_decisions", []) or []:
                if isinstance(decision, dict) and decision.get("asset") in prices:
                    sim.apply_decision(decision, prices, timestamp)
        sim.mark(prices, timestamp)
    report = compute_metrics(sim.result())
    report["cycle_count"] = len(cycles)
    return report


def _parse_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}: {exc}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.backtest.replay",
        description="Replay stored market snapshots through a trading agent.",
    )
    parser.add_argument("--from", dest="date_from", type=_parse_date, default=None)
    parser.add_argument("--to", dest="date_to", type=_parse_date, default=None)
    parser.add_argument("--db", dest="db_path", default=DEFAULT_DB_PATH)
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--agent", choices=["hold", "llm"], default="hold")
    args = parser.parse_args(argv)

    cycles = load_cycles(args.db_path, start=args.date_from, end=args.date_to)
    assets = sorted({a for c in cycles for a in c["sections"]})
    if not cycles:
        print("No cycles found for the requested range.")
        return 1
    print(
        f"Loaded {len(cycles)} cycle(s) from {cycles[0]['timestamp']} "
        f"to {cycles[-1]['timestamp']} covering {assets}"
    )

    if args.agent == "llm":
        print(
            f"WARNING: replaying {len(cycles)} cycle(s) through the live "
            f"TradingAgent will spend OpenRouter tokens on every cycle."
        )
        from src.backend.agent.decision_maker import TradingAgent
        agent = TradingAgent()
    else:
        agent = HoldAgent()

    report = replay(cycles, agent, starting_equity=args.equity)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
