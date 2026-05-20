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


class RegimeSource:
    def __init__(
        self,
        cooldown_sec: float = 1800.0,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._cooldown_sec = cooldown_sec
        self._clock = clock
        self._prior_regime: str | None = None
        self._last_emit_at: datetime | None = None

    def poll(self, bot_state: Any) -> list[OptionsEvent]:
        regime = getattr(bot_state, "vol_regime", None)
        if not isinstance(regime, str) or not regime:
            return []
        if self._prior_regime is None:
            self._prior_regime = regime
            return []
        if regime == self._prior_regime:
            return []
        now = self._clock()
        if self._last_emit_at is not None:
            if (now - self._last_emit_at).total_seconds() < self._cooldown_sec:
                self._prior_regime = regime
                return []
        event = OptionsEvent(
            type=OptionsEventType.REGIME_FLIP,
            fired_at=now,
            payload={"from": self._prior_regime, "to": regime},
        )
        self._prior_regime = regime
        self._last_emit_at = now
        return [event]


class DeltaBandSource:
    def __init__(
        self,
        threshold_btc: float = 0.10,
        cooldown_sec: float = 900.0,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._threshold_btc = threshold_btc
        self._cooldown_sec = cooldown_sec
        self._clock = clock
        self._last_emit_at: datetime | None = None

    def poll(self, bot_state: Any) -> list[OptionsEvent]:
        greeks = getattr(bot_state, "portfolio_greeks", None)
        if not isinstance(greeks, dict):
            return []
        delta_raw = greeks.get("delta")
        if delta_raw is None:
            return []
        try:
            delta = float(delta_raw)
        except (TypeError, ValueError):
            return []
        if abs(delta) < self._threshold_btc:
            return []
        now = self._clock()
        if self._last_emit_at is not None:
            if (now - self._last_emit_at).total_seconds() < self._cooldown_sec:
                return []
        event = OptionsEvent(
            type=OptionsEventType.DELTA_BAND_BREACH,
            fired_at=now,
            payload={"delta_btc": delta, "threshold_btc": self._threshold_btc},
        )
        self._last_emit_at = now
        return [event]
