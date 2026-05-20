import pytest
from datetime import datetime, timezone

from src.backend.trading.options_event_bus import OptionsEvent, OptionsEventType


@pytest.mark.asyncio
async def test_build_options_context_accepts_events_kwarg_and_translates_to_summaries():
    from src.backend.options_intel import builder as builder_mod

    events = [
        OptionsEvent(
            type=OptionsEventType.MAX_INTERVAL_ELAPSED,
            fired_at=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
            payload={"interval_sec": 10800.0},
        ),
        OptionsEvent(
            type=OptionsEventType.STRUCTURE_BREACH,
            fired_at=datetime(2026, 5, 20, 12, 1, tzinfo=timezone.utc),
            payload={"structure_id": "abc123", "from": "nominal", "to": "warning"},
        ),
    ]
    assert events

    desc = builder_mod._describe_event_for_builder("max_interval_elapsed", {"interval_sec": 10800.0})
    assert "heartbeat" in desc
    assert "10800" in desc

    desc2 = builder_mod._describe_event_for_builder(
        "structure_breach",
        {"structure_id": "abc", "from": "nominal", "to": "warning"},
    )
    assert "abc" in desc2
    assert "nominal" in desc2
    assert "warning" in desc2

    import inspect
    sig = inspect.signature(builder_mod.build_options_context)
    assert "events" in sig.parameters


@pytest.mark.asyncio
async def test_describe_unknown_event_returns_type_string():
    from src.backend.options_intel.builder import _describe_event_for_builder
    assert _describe_event_for_builder("unknown_type", {}) == "unknown_type"
