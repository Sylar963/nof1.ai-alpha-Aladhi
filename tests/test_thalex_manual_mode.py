"""Manual-mode regression tests for Thalex proposal routing."""

from types import SimpleNamespace

import pytest

from src.backend.bot_engine import BotState, TradingBotEngine
from src.backend.models.trade_proposal import TradeProposal
from src.gui.services.bot_service import BotService


@pytest.mark.asyncio
async def test_execute_proposal_routes_thalex_proposals_through_thalex_executor():
    proposal = TradeProposal(
        venue="thalex",
        asset="BTC",
        action="buy",
        size=0.05,
        rationale="cheap vol",
        market_conditions={
            "venue": "thalex",
            "strategy": "long_call_delta_hedged",
            "contracts": 0.05,
            "decision_payload": {
                "venue": "thalex",
                "asset": "BTC",
                "action": "buy",
                "strategy": "long_call_delta_hedged",
                "underlying": "BTC",
                "kind": "call",
                "tenor_days": 30,
                "target_strike": 65000,
                "contracts": 0.05,
                "rationale": "cheap vol",
            },
        },
    )
    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    engine.state = SimpleNamespace(pending_proposals=[])
    engine.pending_proposals = [proposal]
    engine.on_trade_executed = None
    engine.on_error = None
    engine._notify_state_update = lambda: None
    diary_entries = []
    engine._write_diary_entry = diary_entries.append

    calls = []

    async def _fake_execute(decision_payload):
        calls.append(decision_payload)
        return True, ""

    engine._execute_thalex_decision = _fake_execute

    await engine._execute_proposal(proposal)

    assert calls == [proposal.market_conditions["decision_payload"]]
    assert proposal.status == "executed"
    assert diary_entries[0]["venue"] == "thalex"


@pytest.mark.asyncio
async def test_execute_proposal_marks_thalex_proposal_failed_on_rejection():
    proposal = TradeProposal(
        venue="thalex",
        asset="BTC",
        action="sell",
        market_conditions={
            "venue": "thalex",
            "decision_payload": {
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
                "rationale": "short vol",
            },
        },
    )
    errors = []
    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    engine.state = SimpleNamespace(pending_proposals=[])
    engine.pending_proposals = [proposal]
    engine.on_trade_executed = None
    engine.on_error = errors.append
    engine._notify_state_update = lambda: None
    engine._write_diary_entry = lambda *a, **k: None

    async def _fake_execute(decision_payload):
        return False, "risk cap"

    engine._execute_thalex_decision = _fake_execute

    await engine._execute_proposal(proposal)

    assert proposal.status == "failed"
    assert proposal.execution_error == "risk cap"
    assert errors == ["Failed to execute trade: risk cap"]


def test_bot_service_approve_proposal_calls_engine_synchronously():
    service = BotService()
    service._add_event = lambda *a, **k: None
    calls = []
    service.bot_engine = SimpleNamespace(
        is_running=True,
        approve_proposal=lambda proposal_id: calls.append(proposal_id) or True,
    )

    assert service.approve_proposal("abc123") is True
    assert calls == ["abc123"]


def test_bot_service_emits_hedge_degraded_and_recovered_events():
    service = BotService()

    degraded_state = BotState(
        hedge_status={
            "health": "degraded",
            "degraded_underlyings": {"BTC": "missing live delta"},
            "tracked_underlyings": 1,
            "active_underlyings": 1,
        }
    )
    recovered_state = BotState(
        hedge_status={
            "health": "healthy",
            "degraded_underlyings": {},
            "tracked_underlyings": 1,
            "active_underlyings": 1,
        }
    )

    service._on_state_update(degraded_state)
    service._on_state_update(recovered_state)

    messages = [event["message"] for event in service.get_recent_events(limit=10)]
    assert any("Delta hedge health degraded" in message for message in messages)
    assert any("Hedge degraded for BTC" in message for message in messages)
    assert any("Hedge recovered for BTC" in message for message in messages)
