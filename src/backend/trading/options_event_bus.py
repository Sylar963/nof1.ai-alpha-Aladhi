from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OptionsEventType(str, Enum):
    REGIME_FLIP = "regime_flip"
    DELTA_BAND_BREACH = "delta_band_breach"
    STRUCTURE_BREACH = "structure_breach"
    DTE_THRESHOLD = "dte_threshold"
    MISPRICING_ACTIONABLE = "mispricing_actionable"
    MAX_INTERVAL_ELAPSED = "max_interval_elapsed"
    MANUAL = "manual"


@dataclass(frozen=True)
class OptionsEvent:
    type: OptionsEventType
    fired_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "fired_at": self.fired_at.isoformat(),
            "payload": dict(self.payload),
        }


class EventBus:
    def __init__(self, dedup_window_sec: float = 300.0) -> None:
        self._dedup_window_sec = dedup_window_sec
        self._queue: list[OptionsEvent] = []
        self._last_emit_at: dict[tuple, datetime] = {}

    def emit(self, event: OptionsEvent) -> bool:
        key = (event.type, event.payload.get("structure_id"))
        last = self._last_emit_at.get(key)
        if last is not None:
            delta = (event.fired_at - last).total_seconds()
            if delta < self._dedup_window_sec:
                return False
        self._last_emit_at[key] = event.fired_at
        self._queue.append(event)
        return True

    def drain(self) -> list[OptionsEvent]:
        drained = list(self._queue)
        self._queue.clear()
        return drained

    def pending_count(self) -> int:
        return len(self._queue)
