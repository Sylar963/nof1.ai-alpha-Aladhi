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


class StructureSource:
    def __init__(
        self,
        cooldown_sec: float = 3600.0,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._cooldown_sec = cooldown_sec
        self._clock = clock
        self._prior_states: dict[str, str] = {}
        self._last_emit_at: dict[str, datetime] = {}

    def poll(self, bot_state: Any) -> list[OptionsEvent]:
        structures = getattr(bot_state, "structures", None)
        if not isinstance(structures, list):
            return []
        events: list[OptionsEvent] = []
        now = self._clock()
        current_ids: set[str] = set()
        for s in structures:
            if not isinstance(s, dict):
                continue
            sid = s.get("structure_id")
            current_breach = s.get("breach_state")
            if not isinstance(sid, str) or not isinstance(current_breach, str):
                continue
            current_ids.add(sid)
            prior = self._prior_states.get(sid)
            self._prior_states[sid] = current_breach
            if prior is None:
                continue
            if prior == "nominal" and current_breach in ("warning", "breached"):
                last = self._last_emit_at.get(sid)
                if last is not None and (now - last).total_seconds() < self._cooldown_sec:
                    continue
                events.append(OptionsEvent(
                    type=OptionsEventType.STRUCTURE_BREACH,
                    fired_at=now,
                    payload={
                        "structure_id": sid,
                        "from": prior,
                        "to": current_breach,
                    },
                ))
                self._last_emit_at[sid] = now
            elif prior == "warning" and current_breach == "breached":
                last = self._last_emit_at.get(sid)
                if last is not None and (now - last).total_seconds() < self._cooldown_sec:
                    continue
                events.append(OptionsEvent(
                    type=OptionsEventType.STRUCTURE_BREACH,
                    fired_at=now,
                    payload={
                        "structure_id": sid,
                        "from": prior,
                        "to": current_breach,
                    },
                ))
                self._last_emit_at[sid] = now
        for sid in list(self._prior_states.keys()):
            if sid not in current_ids:
                self._prior_states.pop(sid, None)
                self._last_emit_at.pop(sid, None)
        return events


class DTESource:
    def __init__(
        self,
        threshold_days: int = 2,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._threshold_days = threshold_days
        self._clock = clock
        self._crossed: set[str] = set()

    def poll(self, bot_state: Any) -> list[OptionsEvent]:
        structures = getattr(bot_state, "structures", None)
        if not isinstance(structures, list):
            return []
        events: list[OptionsEvent] = []
        now = self._clock()
        current_ids: set[str] = set()
        for s in structures:
            if not isinstance(s, dict):
                continue
            sid = s.get("structure_id")
            tenor_min = s.get("tenor_days_min")
            if not isinstance(sid, str) or not isinstance(tenor_min, int):
                continue
            current_ids.add(sid)
            if tenor_min <= self._threshold_days and sid not in self._crossed:
                events.append(OptionsEvent(
                    type=OptionsEventType.DTE_THRESHOLD,
                    fired_at=now,
                    payload={
                        "structure_id": sid,
                        "tenor_days_min": tenor_min,
                        "threshold_days": self._threshold_days,
                    },
                ))
                self._crossed.add(sid)
        self._crossed = {sid for sid in self._crossed if sid in current_ids}
        return events


class MispricingSource:
    def __init__(
        self,
        score_threshold: float = 0.85,
        cooldown_sec: float = 3600.0,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._score_threshold = score_threshold
        self._cooldown_sec = cooldown_sec
        self._clock = clock
        self._last_emit_at: datetime | None = None

    def poll(self, bot_state: Any) -> list[OptionsEvent]:
        mispricings = getattr(bot_state, "top_mispricings_vs_deribit", None)
        if not isinstance(mispricings, list) or not mispricings:
            return []
        top = mispricings[0]
        if not isinstance(top, dict):
            return []
        edge_bps = top.get("edge_bps")
        if edge_bps is None:
            return []
        try:
            score = abs(float(edge_bps)) / 100.0
        except (TypeError, ValueError):
            return []
        if score < self._score_threshold:
            return []
        now = self._clock()
        if self._last_emit_at is not None:
            if (now - self._last_emit_at).total_seconds() < self._cooldown_sec:
                return []
        event = OptionsEvent(
            type=OptionsEventType.MISPRICING_ACTIONABLE,
            fired_at=now,
            payload={
                "instrument_name": top.get("instrument_name"),
                "edge_bps": float(edge_bps),
                "score": score,
                "threshold": self._score_threshold,
            },
        )
        self._last_emit_at = now
        return [event]
