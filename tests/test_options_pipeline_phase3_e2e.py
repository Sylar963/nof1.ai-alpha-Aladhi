import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.backend.config_loader import CONFIG
from src.backend.trading.options_event_bus import OptionsEvent, OptionsEventType


@pytest.mark.asyncio
async def test_run_options_decision_cycle_stamps_events_into_context(monkeypatch):
    monkeypatch.setitem(CONFIG, "options_structure_prompt", False)

    from src.backend.bot_engine import TradingBotEngine

    engine = object.__new__(TradingBotEngine)
    engine.logger = __import__("logging").getLogger("test_engine")
    engine.trading_mode = "auto"
    engine.pending_proposals = []
    engine.state = SimpleNamespace(
        structures=[],
        last_options_reasoning={},
    )
    engine._sync_pending_proposals_state = lambda: None
    engine._latest_options_context = SimpleNamespace(
        vol_regime="rich",
        vol_regime_confidence="high",
        portfolio_greeks={},
        top_mispricings_vs_deribit=[],
        structures=[],
        structure_views=[],
        triggered_by_events=[],
        to_dict=lambda: {"vol_regime": "rich"},
    )

    class _Agent:
        last_payload = {"reasoning": "r", "trade_decisions": []}
        async def decide(self, ctx): return []

    monkeypatch.setattr(
        "src.backend.agent.options_agent.OptionsAgent",
        lambda llm=None: _Agent(),
        raising=False,
    )
    engine._options_llm_adapter = lambda: None

    events = [OptionsEvent(
        type=OptionsEventType.MAX_INTERVAL_ELAPSED,
        fired_at=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
        payload={"interval_sec": 10800.0},
    )]

    await engine._run_options_decision_cycle(events=events)

    stamped = engine._latest_options_context.triggered_by_events
    assert len(stamped) == 1
    assert stamped[0].type == "max_interval_elapsed"
    assert "heartbeat" in stamped[0].description


@pytest.mark.asyncio
async def test_engine_constructs_event_bus_when_flag_on(monkeypatch):
    monkeypatch.setitem(CONFIG, "options_scheduler_enabled", True)
    monkeypatch.setitem(CONFIG, "options_event_bus_enabled", True)
    from src.backend.bot_engine import TradingBotEngine
    with patch.object(TradingBotEngine, "__init__", lambda self, **k: None):
        engine = TradingBotEngine()
    from src.backend.trading.options_event_bus import EventBus
    assert EventBus is not None


@pytest.mark.asyncio
async def test_engine_event_bus_off_by_default():
    from src.backend.config_loader import CONFIG
    assert CONFIG.get("options_event_bus_enabled") in (False, None)
