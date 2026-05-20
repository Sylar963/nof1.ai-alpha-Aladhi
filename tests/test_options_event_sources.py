from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.backend.trading.options_event_bus import OptionsEventType
from src.backend.trading.options_event_sources import (
    DeltaBandSource,
    HeartbeatSource,
    RegimeSource,
)


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


def test_regime_source_first_poll_anchors_silently():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = RegimeSource(clock=clock)
    state = SimpleNamespace(vol_regime="neutral")
    assert src.poll(state) == []


def test_regime_source_fires_on_transition():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = RegimeSource(clock=clock)
    src.poll(SimpleNamespace(vol_regime="neutral"))
    events = src.poll(SimpleNamespace(vol_regime="rich"))
    assert len(events) == 1
    assert events[0].type == OptionsEventType.REGIME_FLIP
    assert events[0].payload == {"from": "neutral", "to": "rich"}


def test_regime_source_silent_when_unchanged():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = RegimeSource(clock=clock)
    src.poll(SimpleNamespace(vol_regime="rich"))
    assert src.poll(SimpleNamespace(vol_regime="rich")) == []


def test_regime_source_cooldown_blocks_subsequent_flip():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = RegimeSource(cooldown_sec=1800.0, clock=clock)
    src.poll(SimpleNamespace(vol_regime="neutral"))
    first = src.poll(SimpleNamespace(vol_regime="rich"))
    assert len(first) == 1
    clock.advance(1000)
    blocked = src.poll(SimpleNamespace(vol_regime="cheap"))
    assert blocked == []
    clock.advance(1000)
    after = src.poll(SimpleNamespace(vol_regime="fair"))
    assert len(after) == 1
    assert after[0].payload == {"from": "cheap", "to": "fair"}


def test_regime_source_missing_field_returns_empty():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = RegimeSource(clock=clock)
    assert src.poll(SimpleNamespace()) == []


def test_delta_band_fires_when_abs_delta_exceeds():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = DeltaBandSource(threshold_btc=0.10, clock=clock)
    state = SimpleNamespace(portfolio_greeks={"delta": 0.12})
    events = src.poll(state)
    assert len(events) == 1
    assert events[0].type == OptionsEventType.DELTA_BAND_BREACH
    assert events[0].payload == {"delta_btc": 0.12, "threshold_btc": 0.10}


def test_delta_band_fires_for_negative_delta():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = DeltaBandSource(threshold_btc=0.10, clock=clock)
    state = SimpleNamespace(portfolio_greeks={"delta": -0.15})
    events = src.poll(state)
    assert len(events) == 1
    assert events[0].payload["delta_btc"] == -0.15


def test_delta_band_silent_within_band():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = DeltaBandSource(threshold_btc=0.10, clock=clock)
    state = SimpleNamespace(portfolio_greeks={"delta": 0.05})
    assert src.poll(state) == []


def test_delta_band_cooldown():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = DeltaBandSource(threshold_btc=0.10, cooldown_sec=900.0, clock=clock)
    state = SimpleNamespace(portfolio_greeks={"delta": 0.20})
    first = src.poll(state)
    assert len(first) == 1
    clock.advance(500)
    assert src.poll(state) == []
    clock.advance(500)
    after = src.poll(state)
    assert len(after) == 1


def test_delta_band_missing_greeks_returns_empty():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = DeltaBandSource(clock=clock)
    assert src.poll(SimpleNamespace()) == []
    assert src.poll(SimpleNamespace(portfolio_greeks={})) == []
    assert src.poll(SimpleNamespace(portfolio_greeks={"delta": None})) == []
