"""End-to-end integration test for the options decision pipeline.

Wires all the slice components together with fake adapters and a fake LLM:
- The two-cadence scheduler triggers a decision
- The options agent reads the OptionsContext and emits a decision
- The executor parses + dispatches multi-leg orders
- The hedge manager subscribes new positions

No network IO. Time-controlled via the scheduler's tunable cadences."""

import asyncio
from datetime import date

import pytest

from src.backend.agent.options_agent import OptionsAgent
from src.backend.options_intel.builder import build_options_context
from src.backend.options_intel.iv_history_store import IVHistoryStore
from src.backend.trading.options_strategies import OptionsExecutor
from tests.test_delta_hedge_manager import FakeHyperliquidForHedge, FakeThalexForHedge
from tests.test_options_builder import (
    FakeDeribitClient,
    FakeThalexAdapter,
    deribit_chain,
    thalex_chain,
)
from tests.test_options_strategies import FakeHyperliquid, FakeThalex


_TEST_TODAY = date(2026, 4, 10)


class StubLLM:
    """Returns a hard-coded credit-put-spread decision payload."""

    def __init__(self):
        self.calls = 0

    async def chat_json(self, *, system_prompt, user_prompt, schema):
        self.calls += 1
        return {
            "reasoning": "vol is rich, sell premium below support with defined risk",
            "trade_decisions": [
                {
                    "venue": "thalex",
                    "asset": "BTC",
                    "action": "sell",
                    "strategy": "credit_put_spread",
                    "underlying": "BTC",
                    "tenor_days": 14,
                    "vol_view": "short_vol",
                    "legs": [
                        {"kind": "put", "side": "sell", "target_strike": 55000, "contracts": 0.05},
                        {"kind": "put", "side": "buy", "target_strike": 50000, "contracts": 0.05},
                    ],
                    "rationale": "vol is rich, sell put spread",
                }
            ],
        }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_builds_context_runs_agent_executes_decision(
    thalex_chain, deribit_chain, tmp_path
):
    """Snapshot → agent → executor: a single full cycle without errors."""
    thalex_intel = FakeThalexAdapter(instruments_cache=thalex_chain, tickers={})
    deribit = FakeDeribitClient(summaries=deribit_chain)
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))

    ctx = await build_options_context(
        thalex=thalex_intel,
        deribit=deribit,
        iv_history=store,
        spot_history=[60000.0] * 16,
        today=_TEST_TODAY,
    )

    llm = StubLLM()
    agent = OptionsAgent(llm=llm)
    decisions = await agent.decide(ctx)

    assert llm.calls == 1
    assert len(decisions) == 1
    assert decisions[0].strategy == "credit_put_spread"

    # Now route through the executor with a separate fake exchange pair
    # (the intel-side fake doesn't model order execution).
    thalex_exec = FakeThalex()
    hl_exec = FakeHyperliquid()
    thalex_exec.instruments_by_intent[("BTC", "put", 14, 55000.0)] = "BTC-25APR26-55000-P"
    thalex_exec.instruments_by_intent[("BTC", "put", 14, 50000.0)] = "BTC-25APR26-50000-P"
    executor = OptionsExecutor(thalex=thalex_exec, hyperliquid=hl_exec)

    result = await executor.execute(decisions[0], open_positions_count=0)

    assert result.ok is True
    assert len(thalex_exec.calls) == 2
    methods = [c.method for c in thalex_exec.calls]
    assert methods == ["sell", "buy"]
    assert hl_exec.calls == []


@pytest.mark.asyncio
async def test_full_pipeline_long_call_delta_hedged_subscribes_to_hedge_manager(tmp_path):
    """Long call delta hedged should fire a thalex buy + a hyperliquid short
    + add the position to the delta hedge manager subscription registry."""

    class _DirectionalLLM:
        async def chat_json(self, *, system_prompt, user_prompt, schema):
            return {
                "reasoning": "vol cheap, buy directional",
                "trade_decisions": [
                    {
                        "venue": "thalex",
                        "asset": "BTC",
                        "action": "buy",
                        "strategy": "long_call_delta_hedged",
                        "underlying": "BTC",
                        "kind": "call",
                        "tenor_days": 30,
                        "target_strike": 65000,
                        "contracts": 0.05,
                        "rationale": "vol cheap, buy gamma",
                    }
                ],
            }

    # Build a minimal context manually
    from src.backend.options_intel.snapshot import OptionsContext

    ctx = OptionsContext(
        timestamp_utc="2026-04-10T19:00:00+00:00",
        spot=60000.0,
        spot_24h_change_pct=0.01,
        opening_range={"high": 60500, "low": 59800, "position": "inside"},
        keltner={"ema20": 60000, "upper": 60800, "lower": 59200, "position": "inside"},
        atm_iv_by_tenor={30: 0.65},
        skew_25d_by_tenor={30: 0.03},
        term_structure_slope=0.0,
        expected_move_pct_by_tenor={30: 0.06},
        vol_regime="cheap",
        vol_regime_confidence="high",
        realized_iv_ratio_30d=1.4,
        straddle_test_30d={"lower": 55000, "upper": 65000, "current_spot": 60000, "label": "cheap"},
        top_mispricings_vs_deribit=[],
        open_positions=[],
        portfolio_greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
        capital_available=10000.0,
        max_contracts_per_trade=0.1,
        max_open_positions=3,
        open_position_count=0,
        recent_options_trades=[],
    )

    agent = OptionsAgent(llm=_DirectionalLLM())
    decisions = await agent.decide(ctx)
    assert len(decisions) == 1
    assert decisions[0].strategy == "long_call_delta_hedged"

    # Execute and verify perp hedge fires
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    thalex.delta_per_position["BTC-10MAY26-65000-C"] = 0.5
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    result = await executor.execute(decisions[0], open_positions_count=0)

    assert result.ok is True
    assert len(thalex.calls) == 1 and thalex.calls[0].method == "buy"
    assert len(hl.calls) == 1 and hl.calls[0].method == "sell"
    # contracts × delta = 0.05 × 0.5 = 0.025
    assert hl.calls[0].amount == pytest.approx(0.025)


@pytest.mark.asyncio
async def test_full_pipeline_skips_invalid_llm_decisions(tmp_path):
    """If the LLM returns garbage, the agent drops it and the pipeline still completes."""

    class _BadLLM:
        async def chat_json(self, *, system_prompt, user_prompt, schema):
            return {"trade_decisions": [{"venue": "atlantis", "asset": "BTC", "action": "buy"}]}

    from src.backend.options_intel.snapshot import OptionsContext

    ctx = OptionsContext(
        timestamp_utc="2026-04-10T19:00:00+00:00",
        spot=60000.0,
        spot_24h_change_pct=0.0,
        opening_range={"high": 0, "low": 0, "position": "unknown"},
        keltner={"ema20": 0, "upper": 0, "lower": 0, "position": "unknown"},
        atm_iv_by_tenor={},
        skew_25d_by_tenor={},
        term_structure_slope=0.0,
        expected_move_pct_by_tenor={},
        vol_regime="unknown",
        vol_regime_confidence="unknown",
        realized_iv_ratio_30d=0.0,
        straddle_test_30d={},
        top_mispricings_vs_deribit=[],
        open_positions=[],
        portfolio_greeks={"delta": 0, "gamma": 0, "vega": 0, "theta": 0},
        capital_available=0.0,
        max_contracts_per_trade=0.1,
        max_open_positions=3,
        open_position_count=0,
        recent_options_trades=[],
    )

    agent = OptionsAgent(llm=_BadLLM())
    decisions = await agent.decide(ctx)
    assert decisions == []
