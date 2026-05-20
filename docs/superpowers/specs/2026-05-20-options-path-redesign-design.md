# Options Path Redesign — Design Spec

**Date:** 2026-05-20
**Branch:** `aladhi-multi-tenor-hardening`
**Predecessor:** `2026-04-15-options-trading-brain-design.md` (fat system prompt + perps context leak fix — landed)
**Scope:** Replace the leg-bag view of options with a structure-centric pipeline: recognition (positions become typed structures), trigger logic (event-driven, not cron), and reasoning (structure-centric LLM context, persisted to DB)

## Problem

The options pipeline today represents positions as a flat list of legs. The LLM re-infers structure (credit put spread, iron condor, etc.) from leg pairs every cycle. There are three concrete consequences:

1. **No structural recognition.** Positions have no `strategy_type` field. `VALID_STRATEGIES` (`src/backend/agent/decision_schema.py:31-38`) omits debit spreads entirely — debit-spread intent has to be smuggled in via `entry_kind="vertical"` + leg construction. The system cannot answer "is this a credit put spread or a debit call spread?" without LLM inference.
2. **Dumb trigger loop.** `src/backend/trading/options_scheduler.py:41-43` runs the LLM on a fixed 3-hour cron. Mid-cycle events (vol regime flip, delta band breach, structure approaching DTE, an actionable mispricing) are invisible until the next tick.
3. **Thin context, no history.** The LLM sees vol surface + `open_positions[*]` + `portfolio_greeks` totals, but no per-structure rollups (P&L, breakevens, breach state). Reasoning is stored only in memory (`state.last_options_reasoning`) and lost on restart — no timeline, no diff, no post-execution outcome record.

## Approach

**Structural integration: introduce a structure layer that the trigger and reasoning layers both consume.**

The structure layer is the keystone. Once positions know what they are, triggers become "events on structures" instead of "events on legs", and the prompt naturally tightens because we can pass `structures[*]` as the primary entity with legs as supporting detail. Debit spreads cost a single enum entry rather than a separate code path because credit/debit is derived from `net_premium` sign — not stored as a distinct strategy.

Three phases, each behind its own feature flag, each independently shippable and reversible:

| Phase | Flag | Scope |
|------|------|-------|
| 1 | `OPTIONS_STRUCTURE_LAYER=1` | Classifier + DB persistence + GUI grouping. Prompt still leg-based. |
| 2 | `OPTIONS_STRUCTURE_PROMPT=1` | Prompt switches to structure-centric. Reasoning persisted to DB. |
| 3 | `OPTIONS_EVENT_BUS_ENABLED=1` | Event bus replaces the 3-hour cron. Cron kept as safety-net heartbeat. |

`OPTIONS_SCHEDULER_ENABLED=1` continues to gate the whole pipeline; the new flags are sub-gates.

## Architecture

### Module map

```
src/backend/options_intel/
├── structure.py             # NEW — OptionStructure, StructureKind, classify()
├── portfolio.py             # MODIFIED — emits structures alongside open_positions
└── snapshot.py              # MODIFIED — OptionsContext gains structures: list[StructureView]

src/backend/trading/
├── options_event_bus.py     # NEW — OptionsEvent enum + EventBus
├── options_event_sources.py # NEW — Regime/DeltaBand/Structure/DTE/Mispricing/Heartbeat sources
└── options_scheduler.py     # MODIFIED — drains event bus instead of pure-cron LLM trigger

src/backend/agent/
├── decision_schema.py       # MODIFIED — VALID_STRATEGIES adds debit_put_spread, debit_call_spread, long_straddle, iron_butterfly
└── options_agent.py         # MODIFIED — prompt regions expanded for debit spreads + structure-keyed management

src/database/
└── models.py                # MODIFIED — add OptionStructureSnapshot, OptionsReasoningEntry

src/gui/
├── pages/positions.py       # MODIFIED — group legs by structure_id; structure card view
├── pages/reasoning.py       # MODIFIED — history panel reading OptionsReasoningEntry table
└── pages/dashboard.py       # MODIFIED — active-structures tile
```

### Data flow

```
Thalex API
   ↓ (raw positions)
portfolio.py  ─────────────────────────────────────────┐
   ↓ (open_positions + greeks)                          │
structure.classify(legs) ──→ list[OptionStructure]      │
   ↓                                                    │
OptionStructureSnapshot (DB: update-in-place)           │
   ↓                                                    │
BotState.options.structures + open_positions ───────────┤
                                                        │
EventSources (poll BotState every ~30s)                 │
   ↓                                                    │
EventBus (queue)                                        │
   ↓                                                    │
options_scheduler drains queue                          │
   ↓ (events + structures)                              │
snapshot.build(...) → OptionsContext                    │
   ↓                                                    │
OptionsAgent.decide(context)  ←─────────────────────────┘
   ↓
TradeDecisions
   ↓
OptionsReasoningEntry (DB: append-per-cycle)
   ↓
Strategy executor → ExchangeAdapter
```

## Section 1 — Recognition Layer

### `OptionStructure` dataclass

`src/backend/options_intel/structure.py`:

```python
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

class StructureKind(str, Enum):
    CREDIT_PUT_SPREAD   = "credit_put_spread"
    CREDIT_CALL_SPREAD  = "credit_call_spread"
    DEBIT_PUT_SPREAD    = "debit_put_spread"
    DEBIT_CALL_SPREAD   = "debit_call_spread"
    IRON_CONDOR         = "iron_condor"
    IRON_BUTTERFLY      = "iron_butterfly"
    LONG_CALL           = "long_call"
    LONG_PUT            = "long_put"
    LONG_STRADDLE       = "long_straddle"
    LONG_STRANGLE       = "long_strangle"
    CALENDAR_PUT        = "calendar_put"
    CALENDAR_CALL       = "calendar_call"
    DIAGONAL_PUT        = "diagonal_put"
    DIAGONAL_CALL       = "diagonal_call"
    UNKNOWN             = "unknown"

class BreachState(str, Enum):
    NOMINAL  = "nominal"   # short leg delta < warning threshold, DTE comfortable
    WARNING  = "warning"   # short leg delta crossed warning threshold OR DTE < 5
    BREACHED = "breached"  # short leg delta > breach threshold OR DTE < 2

@dataclass(frozen=True)
class OptionStructure:
    structure_id: str           # sha1(sorted (instrument_name, side) tuples)
    kind: StructureKind
    underlying: str
    legs: tuple[OptionLeg, ...]
    tenor_days_min: int
    tenor_days_max: int
    net_premium: Decimal        # +credit, -debit, sign-carries
    is_credit: bool
    max_loss: Decimal | None    # None when undefined
    max_profit: Decimal | None
    breakevens: tuple[Decimal, ...]
    short_leg_delta: Decimal | None
    breach_state: BreachState
    pnl_abs: Decimal
    pnl_pct: Decimal
    aggregate_greeks: dict[str, Decimal]  # {"delta", "gamma", "vega", "theta"}; matches portfolio.py shape
    confidence: float           # 1.0 clean match, 0.5 ambiguous (still emitted), <0.5 forces UNKNOWN
```

### Classifier algorithm

Pure function `classify(legs: list[OptionLeg], mark_prices: dict[str, Decimal]) -> OptionStructure`:

1. **Bucket** legs by tenor, then by kind (call/put) within each tenor bucket.
2. **Match templates** in priority order:
   - 4 legs, 2 calls + 2 puts, same tenor, all distinct strikes → `IRON_CONDOR` (different short strikes) or `IRON_BUTTERFLY` (shared short strike)
   - 2 legs, same kind, same tenor, opposite sides (one long, one short) → vertical spread; `is_credit` from `net_premium > 0`; map to `CREDIT_*_SPREAD` or `DEBIT_*_SPREAD`
   - 2 legs, same kind, different tenor, same strike → `CALENDAR_PUT`/`CALENDAR_CALL`
   - 2 legs, same kind, different tenor, different strikes → `DIAGONAL_PUT`/`DIAGONAL_CALL`
   - 2 legs, different kinds, same tenor, both long, same strike → `LONG_STRADDLE`
   - 2 legs, different kinds, same tenor, both long, different strikes → `LONG_STRANGLE`
   - 1 leg, long, call → `LONG_CALL`; 1 leg, long, put → `LONG_PUT`
   - Otherwise → `UNKNOWN` (legs preserved)
3. **Compute properties** from legs + greeks:
   - `net_premium` = Σ(side_sign × contracts × mark_price) where side_sign is +1 long, −1 short
   - `max_loss`/`max_profit` from strike width × contract multiplier × contracts, signed per structure shape
   - `breakevens` from strikes and net_premium
   - `entry_net_premium`: the `net_premium` from the **first** time `structure_id` was classified. Persisted to `OptionStructureSnapshot` on first sight and never overwritten while the structure stays open.
   - `pnl_abs` = `entry_net_premium − current_net_premium` for credit structures (premium received decays favorably), and `current_net_premium − entry_net_premium` for debit structures.
   - `pnl_pct` = `pnl_abs / max(|entry_net_premium|, ε)`; ε = 1e-9.
4. **Compute `breach_state`** against min DTE and a structure-specific delta metric:
   - Delta metric: `short_leg_|delta|` for spreads / ICs / IBs (use the more in-the-money short leg if multiple); aggregate `|delta|` for naked longs and straddles/strangles.
   - `nominal`: delta metric < 0.25 AND DTE ≥ 5
   - `warning`: 0.25 ≤ delta metric < 0.40 OR 2 ≤ DTE < 5
   - `breached`: delta metric ≥ 0.40 OR DTE < 2
   - Calendars/diagonals use the short-tenor leg's |delta| as the delta metric.
5. **Confidence**:
   - `1.0` when a template matched cleanly (unambiguous leg topology).
   - `0.5` when ambiguous (e.g., mixed-tenor pair of opposite-side same-kind legs that could be calendar or diagonal); the better fit is emitted with the structure kind set and confidence 0.5.
   - `0.0` and `kind = UNKNOWN` on any structural anomaly (mismatched contract counts on what should be a balanced spread, mixed underlyings, sentinel/invalid leg, etc.) — legs are preserved in the structure for LLM visibility.

### DB persistence

New SQLAlchemy model `OptionStructureSnapshot` in `src/database/models.py`:

| Column | Type | Notes |
|------|------|-------|
| `structure_id` | str | PK, sha1 of sorted (instrument_name, side) tuples |
| `underlying` | str | "BTC" |
| `kind` | str | StructureKind value |
| `legs_json` | JSON | snapshot of legs at first sight |
| `opened_at` | datetime | first time classifier saw this id |
| `last_seen_at` | datetime | updated every cycle the id appears |
| `closed_at` | datetime | nullable; set when id stops appearing |
| `last_pnl_abs` | Numeric | most recent computation |
| `last_pnl_pct` | Numeric | |
| `last_breach_state` | str | |
| `entry_net_premium` | Numeric | for P&L baseline |
| `metadata_json` | JSON | LLM decision_id that opened it, if known |

Identity by `structure_id`. Update-in-place each cycle (no append). Legs disappearing → `closed_at` set on next cycle that confirms absence.

### Taxonomy update

`src/backend/agent/decision_schema.py` — `VALID_STRATEGIES` becomes:

```python
VALID_STRATEGIES = {
    "credit_put_spread",
    "credit_call_spread",
    "debit_put_spread",     # NEW
    "debit_call_spread",    # NEW
    "iron_condor",
    "iron_butterfly",       # NEW
    "long_call_delta_hedged",
    "long_put_delta_hedged",
    "long_straddle",        # NEW
    "vol_arb",
}
```

`parse_decision()` validates the new strategies the same way as the existing ones — no new code path. `entry_kind="vertical"` continues to mean "two legs, same kind, same tenor"; whether it's credit or debit is `strategy`-determined.

Prompt regions in `_STRATEGY_SELECTION` (`options_agent.py`) gain symmetric guidance:

- **`debit_put_spread`** when IV cheap + bearish directional thesis (mirror of credit_call_spread on the rich/bullish side)
- **`debit_call_spread`** when IV cheap + bullish directional thesis (mirror of credit_put_spread on the rich/bearish side)
- **`long_straddle`** when IV cheap and conviction high vol expansion ahead (existing `vol_arb` covered this implicitly; making it explicit)
- **`iron_butterfly`** when IV rich and high-conviction pin near a strike (tighter alternative to iron_condor)

## Section 2 — Trigger Layer

### Event types

`src/backend/trading/options_event_bus.py`:

```python
class OptionsEventType(str, Enum):
    REGIME_FLIP              = "regime_flip"
    DELTA_BAND_BREACH        = "delta_band_breach"
    STRUCTURE_BREACH         = "structure_breach"
    DTE_THRESHOLD            = "dte_threshold"
    MISPRICING_ACTIONABLE    = "mispricing_actionable"
    MAX_INTERVAL_ELAPSED     = "max_interval_elapsed"
    MANUAL                   = "manual"

@dataclass
class OptionsEvent:
    type: OptionsEventType
    fired_at: datetime
    payload: dict             # type-specific: structure_id, regime, delta, mispricing_id, ...

class EventBus:
    def emit(self, event: OptionsEvent) -> None
    def drain(self) -> list[OptionsEvent]   # also applies dedup + cooldown
    def pending_count(self) -> int
```

### Event sources

`src/backend/trading/options_event_sources.py`. Each source has a `poll(bot_state) -> list[OptionsEvent]` method; the engine loop calls them on a ~30s cadence (configurable). Sources are stateful only for deduplication keys.

| Source | Fires when | Cooldown |
|------|-----------|----------|
| `RegimeSource` | `vol_regime` changed from prior poll | 30 min per direction |
| `DeltaBandSource` | `|portfolio_delta_btc| > OPTIONS_DELTA_BAND_BTC` (default 0.10) | 15 min |
| `StructureSource` | any structure's `breach_state` transitioned to `warning` or `breached` since prior poll | per-structure, 1 hour |
| `DTESource` | any structure crossed `DTE ≤ OPTIONS_DTE_TRIGGER_DAYS` (default 2) | per-structure, fires once per crossing |
| `MispricingSource` | top mispricing for BTC has `score > OPTIONS_MISPRICING_TRIGGER_SCORE` (default 0.85) and wasn't acted on last cycle | 1 hour |
| `HeartbeatSource` | `OPTIONS_HEARTBEAT_SEC` elapsed since last LLM cycle (default 10800 = 3h) | n/a, this IS the safety net |

### Scheduler integration

`src/backend/trading/options_scheduler.py` becomes event-driven:

1. Vol surface refresh keeps its own 15-min cron (unchanged).
2. Event sources poll every 30s on the engine event loop.
3. LLM cycle fires when `event_bus.pending_count() > 0`. Multi-event windows collapse into a single LLM call; the event list rides in `OptionsContext.triggered_by_events`.
4. Per-event-type dedup window of 5 min prevents flapping.
5. If `OPTIONS_EVENT_BUS_ENABLED=0`, the scheduler falls back to the legacy 3-hour cron exactly as today — no event sources poll, `HeartbeatSource` not registered.

### `.env` additions

```env
# --- Options Event Bus ---
# Enable event-driven triggers (Phase 3). Falls back to 3-hour cron when 0.
OPTIONS_EVENT_BUS_ENABLED=0
# Safety-net heartbeat interval (seconds, default 10800 = 3h)
OPTIONS_HEARTBEAT_SEC=10800
# Portfolio delta-band trigger (absolute BTC, default 0.10)
OPTIONS_DELTA_BAND_BTC=0.10
# DTE threshold trigger (days, default 2)
OPTIONS_DTE_TRIGGER_DAYS=2
# Mispricing score trigger (0.0–1.0, default 0.85)
OPTIONS_MISPRICING_TRIGGER_SCORE=0.85
# Event source poll interval (seconds, default 30)
OPTIONS_EVENT_POLL_SECONDS=30

# --- Structure Layer ---
# Enable structure classifier + DB persistence + GUI grouping (Phase 1)
OPTIONS_STRUCTURE_LAYER=0
# Enable structure-centric LLM prompt (Phase 2)
OPTIONS_STRUCTURE_PROMPT=0
```

## Section 3 — Reasoning Layer

### `OptionsContext` rebuild

`src/backend/options_intel/snapshot.py`. `OptionsContext` gains:

```python
@dataclass
class OptionsContext:
    # ...existing fields...
    structures: list[StructureView]          # NEW — primary entity in the prompt
    triggered_by_events: list[EventSummary]  # NEW — what woke the LLM
    # open_positions: list[dict]             # kept as supporting detail
```

`StructureView` is the LLM-facing projection of `OptionStructure`:

```python
@dataclass
class StructureView:
    structure_id: str
    kind: str                  # "credit_put_spread", etc.
    underlying: str
    tenor_days: int            # min(tenor_days_min, tenor_days_max) for spreads, exact for calendars
    days_open: int             # from OptionStructureSnapshot.opened_at
    legs: list[dict]           # abbreviated: kind, side, strike, contracts, |delta|
    net_premium: Decimal
    is_credit: bool
    max_loss: Decimal | None
    max_profit: Decimal | None
    breakevens: list[Decimal]
    short_leg_delta: Decimal | None
    breach_state: str          # "nominal" | "warning" | "breached"
    pnl_abs: Decimal
    pnl_pct: Decimal
    aggregate_greeks: dict
```

`EventSummary`:
```python
@dataclass
class EventSummary:
    type: str                  # OptionsEventType value
    fired_at: datetime
    description: str           # human-readable, e.g. "structure_abc breached: short put |delta|=0.42"
    structure_id: str | None
```

### Prompt edits

Targeted, not full rewrite. In `src/backend/agent/options_agent.py`:

1. **Event prelude** (new, near top of user prompt): `Triggered by: STRUCTURE_BREACH on structure_abc; DELTA_BAND_BREACH (portfolio delta +0.18 BTC). Address these before exploring new entries.`
2. **`_STRATEGY_SELECTION`** extended for debit spread cases (cheap IV + directional thesis) and `iron_butterfly` (high-conviction pin).
3. **`_POSITION_MANAGEMENT`** rekeyed on structures: per-structure checklist — breach_state, DTE, pnl_pct vs management rules. Instead of "look at legs, infer the spread, check rolling rules", the LLM gets "for each structure where `breach_state != nominal` OR `dte_min ≤ 2` OR `|pnl_pct| ≥ 0.65`, decide: hold / take-profit / cut-loss / roll".
4. **Mispricing ranking** against held structures: "M1: 25Δ call rich on JUN 26. You hold short JUN 26 calls in structure_xyz; consider closing structure_xyz instead of adding new exposure."

### Reasoning persistence

New SQLAlchemy model `OptionsReasoningEntry` in `src/database/models.py`:

| Column | Type | Notes |
|------|------|-------|
| `id` | int | PK, autoincrement |
| `created_at` | datetime | LLM call timestamp |
| `triggered_by_events` | JSON | list of EventSummary at call time |
| `context_snapshot` | JSON | full OptionsContext sent to LLM. OptionsContext is already secret-free (vol surface, positions, mispricings, free margin) — no redaction required, but any future additions must be reviewed before persisting. |
| `llm_reasoning` | text | the `reasoning` string from LLM payload |
| `llm_decisions` | JSON | the `trade_decisions` array |
| `outcome` | JSON | nullable; populated post-execution with order_results, errors, skipped reasons |

`bot_engine.py` writes the row after each options LLM call. The `outcome` field is back-filled in the same cycle once the strategy executor returns its order results (synchronous within the options decision cycle). If the cycle is interrupted before back-fill, `outcome` stays `NULL` and is interpretable as "execution did not complete" — a follow-up cycle does not retroactively update older rows.

`state.last_options_reasoning` stays as the in-memory mirror for fast GUI access; on engine boot it hydrates from the most-recent DB row.

### GUI updates

- **Positions page** (`src/gui/pages/positions.py`): when `OPTIONS_STRUCTURE_LAYER=1`, group Thalex legs by `structure_id`. Each group renders as a card showing kind (e.g., "Credit Put Spread"), net_premium, max_loss, breakevens, breach_state badge, P&L. Expanding the card shows individual legs. Hyperliquid perps stay in their existing table — cross-venue linkage (e.g., the perps leg of a `long_call_delta_hedged`) is out of scope for this redesign and remains rendered alongside other perps positions as today. Legacy flat-leg view for Thalex preserved behind the flag.
- **Reasoning page** (`src/gui/pages/reasoning.py`): add an "Options reasoning history" panel reading `OptionsReasoningEntry`. Default to last 20 entries; show triggering events per entry; click to expand context_snapshot + llm_reasoning + outcome.
- **Dashboard** (`src/gui/pages/dashboard.py`): new tile "Active structures" — count by kind, total margin used, sum of unrealized P&L.

## Section 4 — Testing

### Classifier

`tests/test_option_structure_classifier.py` (new):

- **Golden table tests**: fixture of leg combinations → expected `StructureKind` + key derived properties. Cover every kind in the enum.
- **Ambiguous cases**: same-kind different-tenor with strikes that could be calendar or diagonal → assert confidence < 1.0 and correct primary classification.
- **`UNKNOWN` fallback**: mismatched contract counts, mixed underlyings, single-side with no opposite → `UNKNOWN` with legs preserved.
- **Property tests** (hypothesis): `net_premium` sign matches `is_credit`; `breakevens` math against strikes/premium; `max_loss` ≤ 0 and `max_profit` ≥ 0 invariants for credit structures (reversed for debits).
- **Identity stability**: same leg set in different orders → same `structure_id`.

### Event sources

`tests/test_options_event_sources.py` (new):

- Per source: inject two consecutive bot states across the trigger threshold → assert exactly one event emitted with expected payload.
- Cooldown: emit event, advance time < cooldown, second poll → no event; advance past cooldown → event re-emits.
- HeartbeatSource: assert it fires after `OPTIONS_HEARTBEAT_SEC` elapsed since last LLM cycle.

### Integration

`tests/test_options_pipeline_e2e.py` (new):

- Mock Thalex API returning a credit-put-spread leg set; mock LLM returning a `hold`.
- Drive `engine.run_options_cycle()` end-to-end with `OPTIONS_STRUCTURE_LAYER=1` + `OPTIONS_STRUCTURE_PROMPT=1`.
- Assert: structures appear in BotState; `OptionStructureSnapshot` row written; `OptionsReasoningEntry` row written with triggered_by_events containing `MAX_INTERVAL_ELAPSED`; GUI state mirror updated.

### Backwards compatibility

- Existing tests in `tests/test_options_agent.py` continue to pass with flags off (default).
- Legacy positions (no structure_id in DB) classify cleanly into `UNKNOWN` and surface as legacy leg-list in GUI — no migration script required for the structure snapshot table (start empty).

## Rollout & risk

### Phased rollout

Each phase ships independently. Each phase has its own flag. Defaults to off. Operator flips flag on after smoke test passes.

1. **Phase 1** (`OPTIONS_STRUCTURE_LAYER=1`): classifier runs every cycle, structures written to DB, GUI groups legs by structure_id. `OptionsContext` is **not** changed — structures stay out of the LLM prompt. LLM behavior is byte-identical to today (same prompt, same context). Risk floor: GUI rendering bug worst case.
2. **Phase 2** (`OPTIONS_STRUCTURE_PROMPT=1`, requires Phase 1 on): `OptionsContext.structures` and `triggered_by_events` are populated and `OptionsAgent` prompt edits go live. Reasoning written to `OptionsReasoningEntry`. Risk: LLM regresses on edge cases that were leg-shaped in its prior context. Mitigation: A/B by toggling the flag; reasoning DB makes regressions auditable.
3. **Phase 3** (`OPTIONS_EVENT_BUS_ENABLED=1`, requires Phase 2 on so `triggered_by_events` is consumed): event bus replaces cron as the trigger source; cron is now the heartbeat safety net only. Risk: event flapping, missed events. Mitigation: dedup + cooldown + `HeartbeatSource` backstop guarantees the LLM still runs at least every `OPTIONS_HEARTBEAT_SEC`.

### Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Misclassification (calendar read as vertical) | `confidence < 1.0` falls back to `UNKNOWN`; legs always remain in context so LLM can still reason |
| Event flapping (regime borderline) | 5-min dedup window + 30-min per-type cooldown |
| DB bloat | Structure snapshots updated in-place (one row per structure_id, ever). Reasoning entries are ≤8/day in practice (event-driven, capped by cooldowns) |
| Prompt regression | Phase 2 is independently togglable; reasoning persistence makes A/B comparison straightforward |
| Heartbeat starvation if events drown out cron | HeartbeatSource is a regular event source, not a cron — it always queues an event when the interval elapses regardless of other event volume |
| Pre-classifier persisted state | First boot after deploy: structures table empty; classifier rebuilds on first cycle. No migration required. |

## Out of scope

- Cross-venue structures (e.g., Thalex put spread + Hyperliquid delta hedge as a single structure). The hedge stays on its own pipeline.
- Outcome attribution beyond raw order_results (no Sharpe/drawdown/edge tracking on structures).
- Multi-underlying structures (ETH options not on Thalex roadmap here).
- Replacing the OpenRouter model selection for options vs perps.
- Migrating in-flight reasoning history from in-memory `state.last_options_reasoning` to the new DB table.
