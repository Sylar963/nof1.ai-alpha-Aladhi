from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from src.backend.trading.options_event_bus import OptionsEvent, OptionsEventType


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EventSource(Protocol):
    def poll(self, bot_state: Any) -> list[OptionsEvent]: ...


class HeartbeatSource:
    def __init__(
        self,
        interval_sec: float,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._interval_sec = interval_sec
        self._clock = clock
        self._last_cycle_at: datetime | None = None

    def mark_cycle_ran(self, when: datetime) -> None:
        self._last_cycle_at = when

    def poll(self, bot_state: Any) -> list[OptionsEvent]:
        now = self._clock()
        if self._last_cycle_at is None:
            self._last_cycle_at = now
            return []
        if (now - self._last_cycle_at).total_seconds() < self._interval_sec:
            return []
        event = OptionsEvent(
            type=OptionsEventType.MAX_INTERVAL_ELAPSED,
            fired_at=now,
            payload={"interval_sec": self._interval_sec},
        )
        self._last_cycle_at = now
        return [event]
