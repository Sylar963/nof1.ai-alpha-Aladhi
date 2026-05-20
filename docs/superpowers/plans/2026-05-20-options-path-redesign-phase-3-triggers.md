# Phase 3 — Event-Driven Trigger Layer

**Date:** 2026-05-20
**Branch:** `aladhi-options-triggers-phase3`
**Predecessors:**
- Phase 1 (`OPTIONS_STRUCTURE_LAYER`) — landed via PR #18
- Phase 2 (`OPTIONS_STRUCTURE_PROMPT`) — landed via PR #19
**Spec:** `docs/superpowers/specs/2026-05-20-options-path-redesign-design.md` — Section 2 (Trigger Layer)
**Flag:** `OPTIONS_EVENT_BUS_ENABLED`

## Invariant

With `OPTIONS_EVENT_BUS_ENABLED=0` (default), the options pipeline behaves byte-identically to the post-Phase-2 baseline:
- `OptionsScheduler` runs its existing cron (15-min vol-surface, 3-hour LLM cycle).
- No event sources poll, no `EventBus` instance is created.
- `OptionsContext.triggered_by_events` stays empty (Phase 2 already handles this — `to_dict()` only emits it when `OPTIONS_STRUCTURE_PROMPT=1`).

Enforced by: `test_scheduler_default_behavior_unchanged_with_flag_off` (end-to-end) and per-task negative tests.

## Architecture summary

```
EventSources (poll BotState every OPTIONS_EVENT_POLL_SECONDS)
   ↓
EventBus (FIFO queue + per-type dedup + cooldown)
   ↓
OptionsScheduler.drain_events_and_decide()
   ↓
_run_options_decision_cycle(events=[...])
   ↓ (events stamped into latest_options_context.triggered_by_events)
OptionsAgent.decide(context)
```

When the bus is enabled:
- Scheduler's 3-hour cron loop becomes a `HeartbeatSource` event, **not** a separate cron.
- Vol-surface 15-min cron is unchanged (independent of triggers).
- LLM cycle fires only when `event_bus.pending_count() > 0` at drain time.

## Module map

| File | Status | Purpose |
|------|--------|---------|
| `src/backend/trading/options_event_bus.py` | NEW | `OptionsEventType`, `OptionsEvent`, `EventBus` |
| `src/backend/trading/options_event_sources.py` | NEW | 6 polling sources |
| `src/backend/trading/options_scheduler.py` | MODIFIED | Event-driven mode behind flag |
| `src/backend/bot_engine.py` | MODIFIED | Construct bus + sources when flag on; thread events into context |
| `src/backend/config_loader.py` | MODIFIED | 6 new keys |
| `src/backend/options_intel/builder.py` | MODIFIED | Accept `events` arg, stamp `triggered_by_events` |
| `.env.example` | MODIFIED | Document 6 new vars |
| `tests/test_options_event_bus.py` | NEW | EventBus dedup/cooldown unit tests |
| `tests/test_options_event_sources.py` | NEW | Per-source state-transition tests |
| `tests/test_options_scheduler_events.py` | NEW | Scheduler behavior with bus on/off |
| `tests/test_options_pipeline_phase3_e2e.py` | NEW | Full pipeline with flag on |

## Task list (TDD)

Each task: write failing tests first, then implement, then verify. Each subagent must:
1. Run `git rev-parse --abbrev-ref HEAD` and confirm `aladhi-options-triggers-phase3`.
2. **Never** switch branches, never touch main.
3. **No `Co-Authored-By` trailers** (project policy in `CLAUDE.md`).
4. Run the full suite at the end of each task (`pytest`) and report counts.

### Task 1 — Config keys

**Tests:** `tests/test_config_loader.py` — extend to assert defaults for the 6 new keys.

**Impl:** `src/backend/config_loader.py` — add via `_get_bool`/`_get_float`/`_get_int`:
- `options_event_bus_enabled` (bool, default `False`)
- `options_heartbeat_sec` (float, default `10800.0`)
- `options_delta_band_btc` (float, default `0.10`)
- `options_dte_trigger_days` (int, default `2`)
- `options_mispricing_trigger_score` (float, default `0.85`)
- `options_event_poll_seconds` (float, default `30.0`)

**`.env.example`:** add the 6 lines from the spec (Section 2 — `.env` additions).

### Task 2 — `OptionsEventType` + `OptionsEvent` dataclass

**Tests:** `tests/test_options_event_bus.py::test_event_type_enum_complete` — assert all 7 event types exist; `test_event_dataclass_fields` — assert frozen, has `type`, `fired_at`, `payload`, `to_dict()` round-trips.

**Impl:** `src/backend/trading/options_event_bus.py`:
- `class OptionsEventType(str, Enum)` with: `REGIME_FLIP`, `DELTA_BAND_BREACH`, `STRUCTURE_BREACH`, `DTE_THRESHOLD`, `MISPRICING_ACTIONABLE`, `MAX_INTERVAL_ELAPSED`, `MANUAL`.
- `@dataclass(frozen=True) class OptionsEvent` with `type: OptionsEventType`, `fired_at: datetime` (timezone-aware UTC), `payload: dict`, and a `to_dict()` method returning JSON-safe shape.

### Task 3 — `EventBus`

**Tests:** `tests/test_options_event_bus.py`:
- `test_emit_then_drain_yields_event`
- `test_drain_returns_in_fifo_order`
- `test_dedup_within_window_drops_duplicates` — emit two `REGIME_FLIP` within `dedup_window_sec`; drain returns only the first.
- `test_dedup_keyed_by_type_and_structure_id` — same type with different `payload["structure_id"]` are NOT deduped.
- `test_cooldown_blocks_re_emission` — after a successful drain, subsequent emits of same key within cooldown are silently dropped.
- `test_pending_count_excludes_deduped`
- `test_drain_clears_queue`

**Impl:** `src/backend/trading/options_event_bus.py`:

```python
class EventBus:
    def __init__(self, dedup_window_sec: float = 300.0, cooldown_sec: float = 0.0) -> None
    def emit(self, event: OptionsEvent) -> bool         # returns False if deduped
    def drain(self) -> list[OptionsEvent]               # FIFO, clears queue, records cooldown
    def pending_count(self) -> int
```

Key = `(event.type, payload.get("structure_id"))` for dedup/cooldown. Cooldown is enforced *per source* (sources pass their own `cooldown_sec` via the event's payload metadata) — see Task 4 for source-side cooldown logic; the bus enforces only the global dedup window.

### Task 4 — `HeartbeatSource` (simplest source, establishes the pattern)

**Tests:** `tests/test_options_event_sources.py`:
- `test_heartbeat_fires_when_interval_elapsed` — set last-cycle = now − 11000s, poll → emits `MAX_INTERVAL_ELAPSED`.
- `test_heartbeat_silent_within_interval` — last-cycle = now − 100s, poll → no event.
- `test_heartbeat_re_fires_after_next_interval` — emit, advance time past next interval, emit again.

**Impl:** `src/backend/trading/options_event_sources.py`:

```python
class EventSource(Protocol):
    def poll(self, bot_state) -> list[OptionsEvent]: ...

class HeartbeatSource:
    def __init__(self, interval_sec: float, clock: Callable[[], datetime] = ...) -> None
    def mark_cycle_ran(self, when: datetime) -> None
    def poll(self, bot_state) -> list[OptionsEvent]
```

`bot_state` is the engine's `BotState` dataclass. `HeartbeatSource` reads no fields from it — it tracks `_last_cycle_at` internally, updated by `mark_cycle_ran` (the engine calls this after a successful options decision cycle).

### Task 5 — `RegimeSource`, `DeltaBandSource`

**Tests:** `tests/test_options_event_sources.py`:
- `test_regime_source_fires_on_transition` — two polls: regime "neutral" then "rich" → one `REGIME_FLIP` event with payload `{from, to}`.
- `test_regime_source_silent_when_unchanged`
- `test_regime_source_cooldown` — after fire, advance < 30min, change regime → no event; advance > 30min → event.
- `test_delta_band_fires_when_abs_delta_exceeds` — portfolio delta 0.12 BTC, threshold 0.10 → fires.
- `test_delta_band_silent_within_band`
- `test_delta_band_cooldown` — 15min cooldown.

**Impl:** Sources read `bot_state.options` (the existing `BotState.options` field — `OptionsState`-like substructure). Add `_last_emitted_at` per source for cooldown enforcement. Use the same `clock` injection pattern as `HeartbeatSource` for test determinism.

### Task 6 — `StructureSource`, `DTESource`, `MispricingSource`

**Tests:** `tests/test_options_event_sources.py`:
- `test_structure_source_fires_on_breach_transition` — structure breach_state "nominal" → "warning" emits one `STRUCTURE_BREACH` per structure with `payload["structure_id"]`.
- `test_structure_source_per_structure_cooldown` — independent cooldown per `structure_id`.
- `test_dte_source_fires_when_min_dte_crosses_threshold` — structure tenor 3 → 2 emits one `DTE_THRESHOLD`; subsequent polls at tenor ≤2 don't re-emit.
- `test_dte_source_per_structure_once_per_crossing`
- `test_mispricing_source_fires_when_top_score_exceeds` — top mispricing score 0.90 → fires; 0.80 → silent.
- `test_mispricing_source_cooldown` — 1 hour.

**Impl:** All three follow the same `clock`-injected, stateful-dedup-keys pattern. `StructureSource` reads `bot_state.structures` (from Phase 1). `DTESource` reads `min(structure.tenor_days_min for structure in structures)` per structure (`structure_id` is the dedup key). `MispricingSource` reads `bot_state.options.top_mispricings_vs_deribit[0]["score"]`.

### Task 7 — Scheduler integration (event-driven mode)

**Tests:** `tests/test_options_scheduler_events.py`:
- `test_scheduler_event_mode_drains_bus` — feed bus 1 event, run one scheduler tick, assert decision callback called once with the event list.
- `test_scheduler_event_mode_no_events_no_call` — empty bus, scheduler ticks 3 times, decision callback never called.
- `test_scheduler_event_mode_collapses_multiple_events_into_single_call` — feed bus 3 events, one drain, decision called once with all 3.
- `test_scheduler_default_behavior_unchanged_with_flag_off` — flag off, scheduler behaves exactly as today (cron-based, no bus instance).
- `test_scheduler_event_mode_calls_mark_cycle_ran_on_heartbeat` — after a decision cycle completes, the heartbeat source is informed.

**Impl:** `src/backend/trading/options_scheduler.py`:
- Add optional `event_bus: EventBus | None = None` and `event_sources: list | None = None` to `OptionsSchedulerConfig`.
- Add a new background loop `_event_loop` that runs only when `event_bus is not None`:
  1. Sleep `OPTIONS_EVENT_POLL_SECONDS`.
  2. For each source: `event_bus.emit(...)` for each emitted event.
  3. If `event_bus.pending_count() > 0`: drain, call `run_options_decision(events=...)`, then call `heartbeat_source.mark_cycle_ran(now)` if present.
- When event mode is active, the legacy `options_decision_interval_seconds` cron is **disabled** (set to 0). Vol-surface cron stays.
- `run_options_decision` callback signature becomes `Callable[[list[OptionsEvent] | None], Awaitable[None]]`; backward-compatible by accepting `None` (cron-mode call).

### Task 8 — Engine wiring

**Tests:** `tests/test_options_pipeline_phase3_e2e.py`:
- `test_engine_constructs_bus_and_sources_when_flag_on`
- `test_engine_constructs_no_bus_when_flag_off`
- `test_run_options_decision_cycle_stamps_events_into_context` — pass `events=[OptionsEvent(REGIME_FLIP, ...)]`, assert `self._latest_options_context.triggered_by_events == [EventSummary(...)]` before `OptionsAgent.decide` is called.

**Impl:** `src/backend/bot_engine.py`:
- In the same block that constructs `OptionsScheduler` (~line 224): when `CONFIG.get("options_event_bus_enabled")` is on:
  - Build `EventBus(dedup_window_sec=300.0)`.
  - Build the 6 sources with config values from `CONFIG`.
  - Pass them into `OptionsSchedulerConfig`. Set `options_decision_interval_seconds=0` so the legacy decision cron is disabled.
- Change `_run_options_decision_cycle` signature to `async def _run_options_decision_cycle(self, events: list[OptionsEvent] | None = None)`. When `events` is non-empty and `self._latest_options_context` is not None, build `EventSummary` list (see `src/backend/options_intel/snapshot.py` for the existing dataclass) and assign to `self._latest_options_context.triggered_by_events`.
- Defensive: structure is **frozen** (`OptionsContext` is not — verify; if frozen, replace with `dataclasses.replace(...)`). Audit before writing.

### Task 9 — Builder accepts events at build time (optional path)

**Tests:** `tests/test_options_pipeline_phase3_e2e.py`:
- `test_build_options_context_accepts_events_kwarg` — `build_options_context(..., events=[...])` returns a context whose `triggered_by_events` matches.

**Impl:** `src/backend/options_intel/builder.py`:
- Add `events: list[OptionsEvent] | None = None` kwarg to `build_options_context`.
- Convert to `EventSummary` instances. Set on `OptionsContext.triggered_by_events`.
- Task 8's engine-side stamping is the fallback; this kwarg makes the e2e test path cleaner.

### Task 10 — Full pipeline e2e

**Tests:** `tests/test_options_pipeline_phase3_e2e.py`:
- `test_full_pipeline_phase3_e2e` — enable all 3 flags (`OPTIONS_STRUCTURE_LAYER`, `OPTIONS_STRUCTURE_PROMPT`, `OPTIONS_EVENT_BUS_ENABLED`). Mock Thalex returning credit-put-spread legs; mock vol surface; pre-seed bus with one `MAX_INTERVAL_ELAPSED`. Drive scheduler one event-loop tick. Assert:
  - `OptionsAgent.decide` called exactly once.
  - The context it received had `triggered_by_events` containing the heartbeat event.
  - `OptionsStructureSnapshot` row written (from Phase 1).
  - `OptionsReasoningEntry` row written (from Phase 2) with `triggered_by_events` JSON containing the heartbeat event.

## Review fix loop

When the subagent for each task completes, run `pytest -k "options" -v` plus the new test files. If anything red, fix-forward on the same branch. Do not commit unless all targeted tests pass and the full suite has no new regressions (1 pre-existing FakeThalexAPI failure is the baseline).

## Out-of-scope (defer to Phase 4 or later)

- Renaming `MAX_INTERVAL_ELAPSED` → `HEARTBEAT` (spec uses the former; keep it).
- Cross-venue events (perps regime change triggering options re-evaluation).
- LLM-side handling differences when triggered_by_events is non-empty vs heartbeat-only (the Phase 2 prompt already references events; no prompt edits needed in Phase 3).
- GUI surfacing of pending events (`pending_count()` on the dashboard).
- Persisting events themselves to DB (only persisted indirectly via `OptionsReasoningEntry.triggered_by_events`).

## Risks tracked from spec

- **Event flapping**: dedup window (300s) + per-source cooldown (15-60min). Tested in Tasks 3, 5, 6.
- **Missed heartbeat**: `HeartbeatSource` is treated like any other source; it always fires when interval elapsed regardless of other event volume. Tested in Task 4.
- **Frozen-dataclass mutation**: Task 8 audits before writing. If `OptionsContext` is frozen, use `dataclasses.replace`.
