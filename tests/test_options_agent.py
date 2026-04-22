"""Tests for the OptionsAgent — separate LLM pipeline for vol/options reasoning.

The agent is the bridge between the OptionsContext digest (built by
options_intel.builder) and the executor. Tests use a fake LLM client and a
fake context builder so the full pipeline runs without network IO."""

import json

import pytest

from src.backend.agent.options_agent import OptionsAgent
from src.backend.agent.decision_schema import TradeDecision
from src.backend.options_intel.snapshot import OptionsContext


class FakeLLMClient:
    """Stand-in that returns whatever JSON the test seeds, recording the prompts."""

    def __init__(self, response_payload: dict):
        self.response_payload = response_payload
        self.calls: list[dict] = []

    async def chat_json(self, *, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": schema,
        })
        return self.response_payload


def _basic_context() -> OptionsContext:
    return OptionsContext(
        timestamp_utc="2026-04-10T19:00:00+00:00",
        spot=60000.0,
        spot_24h_change_pct=0.01,
        opening_range={"high": 60500, "low": 59800, "position": "inside"},
        keltner={"ema20": 60000, "upper": 60800, "lower": 59200, "position": "inside"},
        atm_iv_by_tenor={7: 0.65, 14: 0.66, 30: 0.68},
        skew_25d_by_tenor={7: 0.05, 14: 0.04, 30: 0.03},
        term_structure_slope=0.0001,
        expected_move_pct_by_tenor={7: 0.045, 14: 0.06, 30: 0.085},
        vol_regime="rich",
        vol_regime_confidence="high",
        realized_iv_ratio_30d=0.6,
        straddle_test_30d={"lower": 55000, "upper": 65000, "current_spot": 60000, "label": "fair"},
        top_mispricings_vs_deribit=[],
        open_positions=[],
        portfolio_greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
        capital_available=10000.0,
        max_contracts_per_trade=0.1,
        max_open_positions=3,
        open_position_count=0,
        recent_options_trades=[],
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_agent_constructs_with_llm_client():
    llm = FakeLLMClient({"trade_decisions": []})
    agent = OptionsAgent(llm=llm)
    assert agent.llm is llm


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_passes_options_context_to_llm():
    """The user prompt must contain the serialized OptionsContext digest."""
    llm = FakeLLMClient({"trade_decisions": []})
    agent = OptionsAgent(llm=llm)
    ctx = _basic_context()

    await agent.decide(ctx)

    assert len(llm.calls) == 1
    user_prompt = llm.calls[0]["user_prompt"]
    payload = json.loads(user_prompt)
    assert payload["spot"] == 60000.0
    assert payload["vol_regime"] == "rich"


@pytest.mark.asyncio
async def test_agent_system_prompt_emphasizes_vol_and_strategies():
    """The system prompt must reference the new strategy enum and vol concepts."""
    llm = FakeLLMClient({"trade_decisions": []})
    agent = OptionsAgent(llm=llm)
    await agent.decide(_basic_context())

    system_prompt = llm.calls[0]["system_prompt"]
    # Must mention all four defined-risk options strategies
    assert "credit_put_spread" in system_prompt
    assert "credit_call_spread" in system_prompt
    assert "iron_condor" in system_prompt
    assert "long_call_delta_hedged" in system_prompt
    # Naked legs must be explicitly forbidden if mentioned at all.
    lower = system_prompt.lower()
    if "naked" in lower:
        assert "forbidden" in lower or "no naked" in lower or "must be wrapped" in lower
    # Must reference vol regime / greeks language
    assert "vol_regime" in system_prompt or "volatility" in lower


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_returns_parsed_trade_decisions():
    """Valid LLM output must parse into a list of TradeDecision objects."""
    llm = FakeLLMClient({
        "reasoning": "vol is rich, sell premium via iron condor",
        "trade_decisions": [
            {
                "venue": "thalex",
                "asset": "BTC",
                "action": "sell",
                "strategy": "iron_condor",
                "underlying": "BTC",
                "tenor_days": 14,
                "vol_view": "short_vol",
                "legs": [
                    {"kind": "put", "side": "sell", "target_strike": 55000, "contracts": 0.05},
                    {"kind": "put", "side": "buy", "target_strike": 50000, "contracts": 0.05},
                    {"kind": "call", "side": "sell", "target_strike": 65000, "contracts": 0.05},
                    {"kind": "call", "side": "buy", "target_strike": 70000, "contracts": 0.05},
                ],
                "rationale": "vol crush",
            }
        ],
    })
    agent = OptionsAgent(llm=llm)
    decisions = await agent.decide(_basic_context())

    assert isinstance(decisions, list)
    assert len(decisions) == 1
    assert isinstance(decisions[0], TradeDecision)
    assert decisions[0].strategy == "iron_condor"
    assert len(decisions[0].legs) == 4


@pytest.mark.asyncio
async def test_agent_drops_invalid_decisions_without_raising():
    """Malformed decisions must be skipped, not crash the agent."""
    llm = FakeLLMClient({
        "reasoning": "mix of good and bad",
        "trade_decisions": [
            {  # bad — unknown strategy
                "venue": "thalex",
                "asset": "BTC",
                "action": "sell",
                "strategy": "moonshot",
                "underlying": "BTC",
                "rationale": "x",
            },
            {  # good
                "venue": "thalex",
                "asset": "BTC",
                "action": "sell",
                "strategy": "credit_put_spread",
                "underlying": "BTC",
                "tenor_days": 14,
                "legs": [
                    {"kind": "put", "side": "sell", "target_strike": 55000, "contracts": 0.05},
                    {"kind": "put", "side": "buy", "target_strike": 50000, "contracts": 0.05},
                ],
                "rationale": "x",
            },
        ],
    })
    agent = OptionsAgent(llm=llm)
    decisions = await agent.decide(_basic_context())
    assert len(decisions) == 1
    assert decisions[0].strategy == "credit_put_spread"


@pytest.mark.asyncio
async def test_agent_drops_sizeless_buy_sell_decisions():
    """A buy/sell decision with no contracts, target_gamma_btc, or legs
    cannot be sized or executed — drop it upstream so downstream proposal
    creation doesn't raise."""
    llm = FakeLLMClient({
        "reasoning": "close the 80k put",
        "trade_decisions": [
            {  # sizeless — LLM forgot to set contracts
                "venue": "thalex",
                "asset": "BTC-01MAY26-80000-P",
                "action": "buy",
                "strategy": "long_put_delta_hedged",
                "underlying": "BTC",
                "tenor_days": 14,
                "rationale": "close existing short put",
            },
            {  # valid
                "venue": "thalex",
                "asset": "BTC",
                "action": "buy",
                "strategy": "long_call_delta_hedged",
                "underlying": "BTC",
                "tenor_days": 21,
                "kind": "call",
                "contracts": 0.02,
                "rationale": "buy gamma",
            },
        ],
    })
    agent = OptionsAgent(llm=llm)
    decisions = await agent.decide(_basic_context())
    assert len(decisions) == 1
    assert decisions[0].strategy == "long_call_delta_hedged"


@pytest.mark.asyncio
async def test_agent_keeps_hold_decisions_even_when_sizeless():
    """Hold decisions are informational — they must flow through so the
    operator sees the reasoning, even without a size."""
    llm = FakeLLMClient({
        "reasoning": "no edge right now",
        "trade_decisions": [
            {
                "venue": "thalex",
                "asset": "BTC",
                "action": "hold",
                "rationale": "wait for IV regime clarity",
            },
        ],
    })
    agent = OptionsAgent(llm=llm)
    decisions = await agent.decide(_basic_context())
    assert len(decisions) == 1
    assert decisions[0].action == "hold"


@pytest.mark.asyncio
async def test_agent_returns_empty_list_when_llm_returns_no_decisions():
    llm = FakeLLMClient({"trade_decisions": []})
    agent = OptionsAgent(llm=llm)
    assert await agent.decide(_basic_context()) == []


# ---------------------------------------------------------------------------
# Prompt knowledge sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_prompt_contains_strategy_selection():
    """The prompt must contain strategy selection decision logic."""
    llm = FakeLLMClient({"trade_decisions": []})
    agent = OptionsAgent(llm=llm)
    await agent.decide(_basic_context())

    prompt = llm.calls[0]["system_prompt"]
    lower = prompt.lower()
    # Decision tree branches by regime
    assert "skew" in lower
    assert "realized_iv_ratio" in prompt or "realized" in lower
    # Tenor guidance — match the exact "7-21 DTE" string from the prompt
    # so this doesn't silently pass on any other occurrence of "7" or "21".
    assert "7-21 DTE" in prompt
    assert "tenor" in lower


@pytest.mark.asyncio
async def test_system_prompt_contains_position_management():
    """The prompt must contain rolling, profit-taking, and cut-loss rules."""
    llm = FakeLLMClient({"trade_decisions": []})
    agent = OptionsAgent(llm=llm)
    await agent.decide(_basic_context())

    prompt = llm.calls[0]["system_prompt"]
    lower = prompt.lower()
    assert "roll" in lower
    assert "profit" in lower
    assert "cut" in lower or "close" in lower


@pytest.mark.asyncio
async def test_system_prompt_contains_regime_playbook():
    """The prompt must contain crypto-specific BTC vol knowledge."""
    llm = FakeLLMClient({"trade_decisions": []})
    agent = OptionsAgent(llm=llm)
    await agent.decide(_basic_context())

    prompt = llm.calls[0]["system_prompt"]
    lower = prompt.lower()
    assert "btc" in lower
    assert "backwardation" in lower or "contango" in lower
    assert "expiry" in lower or "gamma" in lower


@pytest.mark.asyncio
async def test_system_prompt_contains_risk_framework():
    """The prompt must contain sizing and portfolio constraint rules."""
    llm = FakeLLMClient({"trade_decisions": []})
    agent = OptionsAgent(llm=llm)
    await agent.decide(_basic_context())

    prompt = llm.calls[0]["system_prompt"]
    lower = prompt.lower()
    assert "delta" in lower
    assert "vega" in lower
    assert "0.02" in prompt or "sizing" in lower
