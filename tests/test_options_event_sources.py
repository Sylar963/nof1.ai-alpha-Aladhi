from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.backend.trading.options_event_bus import OptionsEventType
from src.backend.trading.options_event_sources import (
    DeltaBandSource,
    DTESource,
    HeartbeatSource,
    MispricingSource,
    RegimeSource,
    StructureSource,
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


def test_structure_source_fires_on_breach_transition():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = StructureSource(clock=clock)
    state1 = SimpleNamespace(structures=[{"structure_id": "s1", "breach_state": "nominal"}])
    assert src.poll(state1) == []
    state2 = SimpleNamespace(structures=[{"structure_id": "s1", "breach_state": "warning"}])
    events = src.poll(state2)
    assert len(events) == 1
    assert events[0].type == OptionsEventType.STRUCTURE_BREACH
    assert events[0].payload["structure_id"] == "s1"
    assert events[0].payload["from"] == "nominal"
    assert events[0].payload["to"] == "warning"


def test_structure_source_independent_per_structure_id():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = StructureSource(clock=clock)
    state1 = SimpleNamespace(structures=[
        {"structure_id": "s1", "breach_state": "nominal"},
        {"structure_id": "s2", "breach_state": "nominal"},
    ])
    src.poll(state1)
    state2 = SimpleNamespace(structures=[
        {"structure_id": "s1", "breach_state": "breached"},
        {"structure_id": "s2", "breach_state": "nominal"},
    ])
    events = src.poll(state2)
    assert len(events) == 1
    assert events[0].payload["structure_id"] == "s1"


def test_structure_source_per_structure_cooldown():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = StructureSource(cooldown_sec=3600.0, clock=clock)
    src.poll(SimpleNamespace(structures=[{"structure_id": "s1", "breach_state": "nominal"}]))
    first = src.poll(SimpleNamespace(structures=[{"structure_id": "s1", "breach_state": "warning"}]))
    assert len(first) == 1
    src.poll(SimpleNamespace(structures=[{"structure_id": "s1", "breach_state": "nominal"}]))
    clock.advance(100)
    second = src.poll(SimpleNamespace(structures=[{"structure_id": "s1", "breach_state": "warning"}]))
    assert second == []
    src.poll(SimpleNamespace(structures=[]))
    src.poll(SimpleNamespace(structures=[{"structure_id": "s1", "breach_state": "nominal"}]))
    after = src.poll(SimpleNamespace(structures=[{"structure_id": "s1", "breach_state": "warning"}]))
    assert len(after) == 1


def test_structure_source_warning_to_breached_fires():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = StructureSource(clock=clock)
    src.poll(SimpleNamespace(structures=[{"structure_id": "s1", "breach_state": "warning"}]))
    events = src.poll(SimpleNamespace(structures=[{"structure_id": "s1", "breach_state": "breached"}]))
    assert len(events) == 1
    assert events[0].payload["from"] == "warning"
    assert events[0].payload["to"] == "breached"


def test_structure_source_missing_field_returns_empty():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = StructureSource(clock=clock)
    assert src.poll(SimpleNamespace()) == []
    assert src.poll(SimpleNamespace(structures=[{"structure_id": "s1"}])) == []
    assert src.poll(SimpleNamespace(structures=[{"breach_state": "warning"}])) == []


def test_dte_source_fires_when_min_dte_crosses_threshold():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = DTESource(threshold_days=2, clock=clock)
    state1 = SimpleNamespace(structures=[{"structure_id": "s1", "tenor_days_min": 3}])
    assert src.poll(state1) == []
    state2 = SimpleNamespace(structures=[{"structure_id": "s1", "tenor_days_min": 2}])
    events = src.poll(state2)
    assert len(events) == 1
    assert events[0].type == OptionsEventType.DTE_THRESHOLD
    assert events[0].payload["structure_id"] == "s1"
    assert events[0].payload["tenor_days_min"] == 2
    assert events[0].payload["threshold_days"] == 2


def test_dte_source_per_structure_once_per_crossing():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = DTESource(threshold_days=2, clock=clock)
    state = SimpleNamespace(structures=[{"structure_id": "s1", "tenor_days_min": 2}])
    first = src.poll(state)
    assert len(first) == 1
    second = src.poll(state)
    assert second == []


def test_dte_source_re_fires_after_structure_disappears_and_reappears():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = DTESource(threshold_days=2, clock=clock)
    first = src.poll(SimpleNamespace(structures=[{"structure_id": "s1", "tenor_days_min": 2}]))
    assert len(first) == 1
    src.poll(SimpleNamespace(structures=[]))
    after = src.poll(SimpleNamespace(structures=[{"structure_id": "s1", "tenor_days_min": 2}]))
    assert len(after) == 1


def test_mispricing_source_fires_when_normalized_score_exceeds():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = MispricingSource(score_threshold=0.85, clock=clock)
    state = SimpleNamespace(top_mispricings_vs_deribit=[
        {"instrument_name": "BTC-30MAY26-70000-C", "edge_bps": 90.0}
    ])
    events = src.poll(state)
    assert len(events) == 1
    assert events[0].type == OptionsEventType.MISPRICING_ACTIONABLE
    assert events[0].payload["instrument_name"] == "BTC-30MAY26-70000-C"
    assert events[0].payload["edge_bps"] == 90.0
    assert events[0].payload["score"] == 0.90
    assert events[0].payload["threshold"] == 0.85


def test_mispricing_source_silent_below_threshold():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = MispricingSource(score_threshold=0.85, clock=clock)
    state = SimpleNamespace(top_mispricings_vs_deribit=[
        {"instrument_name": "BTC-30MAY26-70000-C", "edge_bps": 50.0}
    ])
    assert src.poll(state) == []


def test_mispricing_source_fires_for_negative_edge():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = MispricingSource(score_threshold=0.85, clock=clock)
    state = SimpleNamespace(top_mispricings_vs_deribit=[
        {"instrument_name": "BTC-30MAY26-70000-P", "edge_bps": -95.0}
    ])
    events = src.poll(state)
    assert len(events) == 1
    assert events[0].payload["score"] == 0.95
    assert events[0].payload["edge_bps"] == -95.0


def test_mispricing_source_cooldown():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = MispricingSource(score_threshold=0.85, cooldown_sec=3600.0, clock=clock)
    state = SimpleNamespace(top_mispricings_vs_deribit=[
        {"instrument_name": "BTC-30MAY26-70000-C", "edge_bps": 90.0}
    ])
    first = src.poll(state)
    assert len(first) == 1
    clock.advance(1000)
    assert src.poll(state) == []
    clock.advance(3000)
    after = src.poll(state)
    assert len(after) == 1


def test_mispricing_source_empty_list_returns_empty():
    clock = _FakeClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    src = MispricingSource(clock=clock)
    assert src.poll(SimpleNamespace()) == []
    assert src.poll(SimpleNamespace(top_mispricings_vs_deribit=[])) == []
    assert src.poll(SimpleNamespace(top_mispricings_vs_deribit=None)) == []
