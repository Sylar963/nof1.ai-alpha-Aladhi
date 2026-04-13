"""Regression tests for the scheduled options decision loop in manual mode."""

from types import SimpleNamespace

import pytest

from src.backend.agent.decision_schema import TradeDecision
from src.backend.bot_engine import TradingBotEngine


@pytest.mark.asyncio
async def test_run_options_decision_cycle_creates_proposals_in_manual_mode(monkeypatch):
    import src.backend.agent.options_agent as options_agent_module

    decision = TradeDecision(
        asset="BTC",
        action="buy",
        rationale="cheap vol",
        venue="thalex",
        strategy="long_call_delta_hedged",
        underlying="BTC",
        kind="call",
        tenor_days=30,
        target_strike=65000.0,
        contracts=0.05,
    )

    class FakeOptionsAgent:
        def __init__(self, llm):
            self.llm = llm

        async def decide(self, context):
            return [decision]

    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    engine.trading_mode = "manual"
    engine.pending_proposals = []
    engine.state = SimpleNamespace(pending_proposals=[])
    engine._latest_options_context = object()
    engine._options_llm_adapter = lambda: object()

    executed_payloads = []

    async def _fake_execute(payload):
        executed_payloads.append(payload)

    engine._execute_thalex_decision = _fake_execute
    monkeypatch.setattr(options_agent_module, "OptionsAgent", FakeOptionsAgent)

    await TradingBotEngine._run_options_decision_cycle(engine)

    assert executed_payloads == []
    assert len(engine.pending_proposals) == 1
    proposal = engine.pending_proposals[0]
    assert proposal.venue == "thalex"
    assert proposal.market_conditions["decision_payload"]["strategy"] == "long_call_delta_hedged"
