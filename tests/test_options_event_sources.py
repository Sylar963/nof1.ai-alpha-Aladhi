from datetime import datetime, timedelta, timezone

from src.backend.trading.options_event_bus import OptionsEventType
from src.backend.trading.options_event_sources import HeartbeatSource


class _FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start
    def __call__(self) -> datetime:
        return self.now
    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def test_heartbeat_first_poll_returns_empty_and_initializes_anchor():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = HeartbeatSource(interval_sec=10800.0, clock=clock)
    assert src.poll(bot_state=None) == []


def test_heartbeat_silent_within_interval():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = HeartbeatSource(interval_sec=10800.0, clock=clock)
    src.poll(bot_state=None)
    clock.advance(100)
    assert src.poll(bot_state=None) == []


def test_heartbeat_fires_when_interval_elapsed():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = HeartbeatSource(interval_sec=10800.0, clock=clock)
    src.poll(bot_state=None)
    clock.advance(10801)
    events = src.poll(bot_state=None)
    assert len(events) == 1
    assert events[0].type == OptionsEventType.MAX_INTERVAL_ELAPSED


def test_heartbeat_re_fires_after_next_interval():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = HeartbeatSource(interval_sec=100.0, clock=clock)
    src.poll(bot_state=None)
    clock.advance(101)
    first = src.poll(bot_state=None)
    clock.advance(101)
    second = src.poll(bot_state=None)
    assert len(first) == 1
    assert len(second) == 1


def test_mark_cycle_ran_resets_timer():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = HeartbeatSource(interval_sec=100.0, clock=clock)
    src.poll(bot_state=None)
    clock.advance(50)
    src.mark_cycle_ran(clock())
    clock.advance(60)
    assert src.poll(bot_state=None) == []
    clock.advance(50)
    assert len(src.poll(bot_state=None)) == 1
