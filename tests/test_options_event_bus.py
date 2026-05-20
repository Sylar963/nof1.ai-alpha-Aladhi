from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import datetime, timedelta, timezone

import pytest

from src.backend.trading.options_event_bus import (
    EventBus,
    OptionsEvent,
    OptionsEventType,
)


def test_event_type_enum_complete():
    expected = {
        "REGIME_FLIP": "regime_flip",
        "DELTA_BAND_BREACH": "delta_band_breach",
        "STRUCTURE_BREACH": "structure_breach",
        "DTE_THRESHOLD": "dte_threshold",
        "MISPRICING_ACTIONABLE": "mispricing_actionable",
        "MAX_INTERVAL_ELAPSED": "max_interval_elapsed",
        "MANUAL": "manual",
    }
    assert {m.name: m.value for m in OptionsEventType} == expected
    assert len(OptionsEventType) == 7


def test_event_dataclass_fields():
    assert is_dataclass(OptionsEvent)
    field_names = {f.name for f in fields(OptionsEvent)}
    assert field_names == {"type", "fired_at", "payload"}

    fired_at = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    event = OptionsEvent(
        type=OptionsEventType.REGIME_FLIP,
        fired_at=fired_at,
        payload={"reason": "iv_jump"},
    )

    with pytest.raises(FrozenInstanceError):
        event.type = OptionsEventType.MANUAL  # type: ignore[misc]

    assert event.to_dict() == {
        "type": "regime_flip",
        "fired_at": "2026-05-20T12:00:00+00:00",
        "payload": {"reason": "iv_jump"},
    }


def test_event_payload_default_is_empty_dict():
    fired_at = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    event = OptionsEvent(type=OptionsEventType.MANUAL, fired_at=fired_at)
    assert event.payload == {}
    assert event.to_dict()["payload"] == {}


def test_emit_then_drain_yields_event():
    bus = EventBus(dedup_window_sec=300.0)
    fired_at = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    event = OptionsEvent(type=OptionsEventType.REGIME_FLIP, fired_at=fired_at)

    assert bus.emit(event) is True
    assert bus.pending_count() == 1

    drained = bus.drain()
    assert drained == [event]
    assert bus.pending_count() == 0


def test_drain_returns_in_fifo_order():
    bus = EventBus(dedup_window_sec=300.0)
    base = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    e1 = OptionsEvent(
        type=OptionsEventType.REGIME_FLIP,
        fired_at=base,
    )
    e2 = OptionsEvent(
        type=OptionsEventType.DELTA_BAND_BREACH,
        fired_at=base + timedelta(seconds=1),
    )
    e3 = OptionsEvent(
        type=OptionsEventType.DTE_THRESHOLD,
        fired_at=base + timedelta(seconds=2),
    )

    assert bus.emit(e1) is True
    assert bus.emit(e2) is True
    assert bus.emit(e3) is True

    drained = bus.drain()
    assert drained == [e1, e2, e3]


def test_dedup_within_window_drops_duplicates():
    bus = EventBus(dedup_window_sec=300.0)
    base = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    e1 = OptionsEvent(type=OptionsEventType.REGIME_FLIP, fired_at=base)
    e2 = OptionsEvent(
        type=OptionsEventType.REGIME_FLIP,
        fired_at=base + timedelta(seconds=60),
    )

    assert bus.emit(e1) is True
    assert bus.emit(e2) is False
    assert bus.pending_count() == 1
    assert bus.drain() == [e1]


def test_dedup_keyed_by_type_and_structure_id():
    bus = EventBus(dedup_window_sec=300.0)
    base = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    e_a = OptionsEvent(
        type=OptionsEventType.STRUCTURE_BREACH,
        fired_at=base,
        payload={"structure_id": "a"},
    )
    e_b = OptionsEvent(
        type=OptionsEventType.STRUCTURE_BREACH,
        fired_at=base + timedelta(seconds=30),
        payload={"structure_id": "b"},
    )

    assert bus.emit(e_a) is True
    assert bus.emit(e_b) is True
    assert bus.pending_count() == 2
    assert bus.drain() == [e_a, e_b]


def test_dedup_window_allows_re_emission_after_expiry():
    bus = EventBus(dedup_window_sec=300.0)
    base = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    e1 = OptionsEvent(type=OptionsEventType.REGIME_FLIP, fired_at=base)
    e2 = OptionsEvent(
        type=OptionsEventType.REGIME_FLIP,
        fired_at=base + timedelta(seconds=301),
    )

    assert bus.emit(e1) is True
    assert bus.emit(e2) is True
    assert bus.pending_count() == 2


def test_pending_count_excludes_deduped():
    bus = EventBus(dedup_window_sec=300.0)
    base = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    e1 = OptionsEvent(type=OptionsEventType.REGIME_FLIP, fired_at=base)
    e2 = OptionsEvent(
        type=OptionsEventType.REGIME_FLIP,
        fired_at=base + timedelta(seconds=10),
    )
    e3 = OptionsEvent(
        type=OptionsEventType.REGIME_FLIP,
        fired_at=base + timedelta(seconds=20),
    )

    bus.emit(e1)
    bus.emit(e2)
    bus.emit(e3)

    assert bus.pending_count() == 1


def test_drain_clears_queue():
    bus = EventBus(dedup_window_sec=300.0)
    base = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    bus.emit(OptionsEvent(type=OptionsEventType.MANUAL, fired_at=base))

    assert bus.pending_count() == 1
    bus.drain()
    assert bus.pending_count() == 0
    assert bus.drain() == []
