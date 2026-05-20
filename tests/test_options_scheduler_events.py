"""Tests for OptionsScheduler event-driven mode (Phase 3 Task 7)."""

import asyncio
from datetime import datetime, timezone

import pytest

from src.backend.trading.options_event_bus import (
    EventBus,
    OptionsEvent,
    OptionsEventType,
)
from src.backend.trading.options_event_sources import HeartbeatSource
from src.backend.trading.options_scheduler import (
    OptionsScheduler,
    OptionsSchedulerConfig,
)


def _ev(t: OptionsEventType = OptionsEventType.REGIME_FLIP, **payload) -> OptionsEvent:
    return OptionsEvent(
        type=t,
        fired_at=datetime.now(timezone.utc),
        payload=payload or {},
    )


class _StubSource:
    def __init__(self, batches: list[list[OptionsEvent]]) -> None:
        self._batches = list(batches)
        self.poll_count = 0
        self.mark_calls: list[datetime] = []

    def poll(self, bot_state) -> list[OptionsEvent]:
        self.poll_count += 1
        if not self._batches:
            return []
        return self._batches.pop(0)

    def mark_cycle_ran(self, when: datetime) -> None:
        self.mark_calls.append(when)


class _SilentSource:
    def __init__(self) -> None:
        self.poll_count = 0

    def poll(self, bot_state) -> list[OptionsEvent]:
        self.poll_count += 1
        return []


@pytest.mark.asyncio
async def test_scheduler_default_behavior_unchanged_with_no_event_bus():
    surface_calls = 0
    decision_calls: list[tuple] = []

    async def surface():
        nonlocal surface_calls
        surface_calls += 1

    async def decision(*args, **kwargs):
        decision_calls.append((args, kwargs))

    sched = OptionsScheduler(
        OptionsSchedulerConfig(
            vol_surface_interval_seconds=0.05,
            options_decision_interval_seconds=0.05,
        ),
        refresh_vol_surface=surface,
        run_options_decision=decision,
    )
    await sched.start()
    await asyncio.sleep(0.25)
    await sched.stop()

    assert surface_calls >= 1
    assert len(decision_calls) >= 1
    assert all(args == () for args, _ in decision_calls)
    assert all(kwargs == {} for _, kwargs in decision_calls)


@pytest.mark.asyncio
async def test_scheduler_event_mode_drains_bus():
    decision_calls: list[tuple] = []

    async def surface():
        return None

    async def decision(*args, **kwargs):
        decision_calls.append((args, kwargs))

    bus = EventBus(dedup_window_sec=0.0)
    event = _ev()
    source = _StubSource(batches=[[], [event]])

    sched = OptionsScheduler(
        OptionsSchedulerConfig(
            vol_surface_interval_seconds=0,
            options_decision_interval_seconds=0,
        ),
        refresh_vol_surface=surface,
        run_options_decision=decision,
        event_bus=bus,
        event_sources=[source],
        event_poll_seconds=0.05,
    )

    await sched.start()
    await asyncio.sleep(0.30)
    await sched.stop()

    event_mode_calls = [c for c in decision_calls if c[1].get("events")]
    assert len(event_mode_calls) >= 1
    first_args, first_kwargs = event_mode_calls[0]
    assert first_args == ()
    assert "events" in first_kwargs
    assert any(e is event for e in first_kwargs["events"])


@pytest.mark.asyncio
async def test_scheduler_event_mode_no_events_no_call():
    decision_calls: list[tuple] = []

    async def surface():
        return None

    async def decision(*args, **kwargs):
        decision_calls.append((args, kwargs))

    bus = EventBus(dedup_window_sec=0.0)
    source = _SilentSource()

    sched = OptionsScheduler(
        OptionsSchedulerConfig(
            vol_surface_interval_seconds=0,
            options_decision_interval_seconds=0,
        ),
        refresh_vol_surface=surface,
        run_options_decision=decision,
        event_bus=bus,
        event_sources=[source],
        event_poll_seconds=0.04,
    )

    await sched.start()
    await asyncio.sleep(0.25)
    await sched.stop()

    assert source.poll_count >= 2
    assert decision_calls == []


@pytest.mark.asyncio
async def test_scheduler_event_mode_collapses_multiple_events_into_single_call():
    decision_calls: list[tuple] = []

    async def surface():
        return None

    async def decision(*args, **kwargs):
        decision_calls.append((args, kwargs))

    bus = EventBus(dedup_window_sec=0.0)
    triple = [
        _ev(OptionsEventType.REGIME_FLIP),
        _ev(OptionsEventType.DELTA_BAND_BREACH),
        _ev(OptionsEventType.MISPRICING_ACTIONABLE),
    ]
    source = _StubSource(batches=[triple])

    sched = OptionsScheduler(
        OptionsSchedulerConfig(
            vol_surface_interval_seconds=0,
            options_decision_interval_seconds=0,
        ),
        refresh_vol_surface=surface,
        run_options_decision=decision,
        event_bus=bus,
        event_sources=[source],
        event_poll_seconds=0.05,
    )

    await sched.start()
    await asyncio.sleep(0.20)
    await sched.stop()

    event_mode_calls = [c for c in decision_calls if c[1].get("events")]
    assert len(event_mode_calls) == 1
    _, kwargs = event_mode_calls[0]
    assert len(kwargs["events"]) == 3


@pytest.mark.asyncio
async def test_scheduler_event_mode_calls_mark_cycle_ran_after_success():
    decision_calls: list[tuple] = []

    async def surface():
        return None

    async def decision(*args, **kwargs):
        decision_calls.append((args, kwargs))

    bus = EventBus(dedup_window_sec=0.0)
    event = _ev(OptionsEventType.MAX_INTERVAL_ELAPSED)
    source = _StubSource(batches=[[event]])

    sched = OptionsScheduler(
        OptionsSchedulerConfig(
            vol_surface_interval_seconds=0,
            options_decision_interval_seconds=0,
        ),
        refresh_vol_surface=surface,
        run_options_decision=decision,
        event_bus=bus,
        event_sources=[source],
        event_poll_seconds=0.05,
    )

    await sched.start()
    await asyncio.sleep(0.25)
    await sched.stop()

    assert len(decision_calls) >= 1
    assert len(source.mark_calls) >= 1
    assert all(isinstance(t, datetime) for t in source.mark_calls)
    assert all(t.tzinfo is not None for t in source.mark_calls)


@pytest.mark.asyncio
async def test_scheduler_event_mode_uses_latest_state_provider():
    decision_calls: list[tuple] = []
    seen_states: list = []
    sentinel_state = object()

    async def surface():
        return None

    async def decision(*args, **kwargs):
        decision_calls.append((args, kwargs))

    class _SpySource:
        def poll(self, bot_state):
            seen_states.append(bot_state)
            return []

    bus = EventBus(dedup_window_sec=0.0)
    source = _SpySource()

    sched = OptionsScheduler(
        OptionsSchedulerConfig(
            vol_surface_interval_seconds=0,
            options_decision_interval_seconds=0,
        ),
        refresh_vol_surface=surface,
        run_options_decision=decision,
        event_bus=bus,
        event_sources=[source],
        event_poll_seconds=0.04,
        latest_state_provider=lambda: sentinel_state,
    )

    await sched.start()
    await asyncio.sleep(0.15)
    await sched.stop()

    assert seen_states, "spy source was never polled"
    assert all(s is sentinel_state for s in seen_states)


@pytest.mark.asyncio
async def test_scheduler_event_mode_heartbeat_source_integration():
    decision_calls: list[tuple] = []

    async def surface():
        return None

    async def decision(*args, **kwargs):
        decision_calls.append((args, kwargs))

    fake_now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)

    class _FakeClock:
        def __init__(self) -> None:
            self.now = fake_now

        def __call__(self) -> datetime:
            return self.now

    clock = _FakeClock()
    hb = HeartbeatSource(interval_sec=10.0, clock=clock)
    hb.poll(bot_state=None)
    clock.now = datetime(2026, 5, 20, 12, 0, 30, tzinfo=timezone.utc)

    bus = EventBus(dedup_window_sec=0.0)

    sched = OptionsScheduler(
        OptionsSchedulerConfig(
            vol_surface_interval_seconds=0,
            options_decision_interval_seconds=0,
        ),
        refresh_vol_surface=surface,
        run_options_decision=decision,
        event_bus=bus,
        event_sources=[hb],
        event_poll_seconds=0.05,
    )

    await sched.start()
    await asyncio.sleep(0.15)
    await sched.stop()

    event_mode_calls = [c for c in decision_calls if c[1].get("events")]
    assert len(event_mode_calls) >= 1
    _, kwargs = event_mode_calls[0]
    assert any(
        e.type == OptionsEventType.MAX_INTERVAL_ELAPSED for e in kwargs["events"]
    )
