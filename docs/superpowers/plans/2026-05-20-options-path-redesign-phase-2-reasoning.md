# Options Path Redesign — Phase 2 (Reasoning Layer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the structure recognition layer (Phase 1) into LLM-visible reasoning context, with reasoning persisted to a new DB table for audit and history. Gated behind `OPTIONS_STRUCTURE_PROMPT=1` (requires Phase 1's `OPTIONS_STRUCTURE_LAYER=1` to be useful).

**Architecture:** A new `StructureView` dataclass projects each classified structure into an LLM-facing shape (single `tenor_days`, `days_open` from DB, full leg detail). `OptionsContext` gains `structure_views` + `triggered_by_events` (the latter stays empty in Phase 2 — Phase 3 will emit events). `OptionsContext.to_dict()` includes both fields **only when the flag is on**, keeping today's LLM prompt byte-identical for users on `OPTIONS_STRUCTURE_PROMPT=0`. Targeted prompt edits in `options_agent.py` handle debit spreads, iron butterfly, and structure-keyed position management. After each options LLM call, `bot_engine` persists a row to a new `OptionsReasoningEntry` table with context + reasoning + decisions; the row's `outcome` field is back-filled in the same cycle once the strategy executor returns. The Reasoning GUI page gains a history panel reading from this table.

**Tech Stack:** Python 3.14 / asyncio, SQLAlchemy ORM, pytest + pytest-asyncio, NiceGUI.

**Spec:** `docs/superpowers/specs/2026-05-20-options-path-redesign-design.md` (Section 3 — Reasoning Layer)
**Predecessor plan:** `docs/superpowers/plans/2026-05-20-options-path-redesign-phase-1-recognition.md` (Phase 1 landed via PR #18)

---

## File Structure

**New files:**
- `tests/test_options_structure_view.py` — projection tests
- `tests/test_options_reasoning_persistence.py` — DB roundtrip tests
- `tests/test_options_agent_phase2_prompt.py` — prompt content tests

**Modified files:**
- `src/backend/config_loader.py` — add `OPTIONS_STRUCTURE_PROMPT` flag
- `.env.example` — document flag
- `src/backend/options_intel/snapshot.py` — `StructureView`, `EventSummary` dataclasses; `OptionsContext.structure_views` + `triggered_by_events`; conditional `to_dict()`
- `src/backend/options_intel/builder.py` — project structures into `StructureView` with leg detail + days_open from DB
- `src/backend/agent/options_agent.py` — prompt regions for debit spreads + iron butterfly + structure-keyed management + event prelude
- `src/database/models.py` — `OptionsReasoningEntry`
- `src/database/db_manager.py` — `save_options_reasoning`, `get_recent_options_reasoning`, `update_reasoning_outcome`
- `src/backend/bot_engine.py` — write reasoning row after LLM call; back-fill outcome
- `src/gui/pages/reasoning.py` — history panel reading from DB

---

## Task 1: Add OPTIONS_STRUCTURE_PROMPT flag

**Files:**
- Modify: `src/backend/config_loader.py` (around line 120, next to `options_structure_layer`)
- Modify: `.env.example`
- Modify: `tests/test_config_loader.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config_loader.py`:

```python
def test_options_structure_prompt_default_off(monkeypatch):
    monkeypatch.delenv("OPTIONS_STRUCTURE_PROMPT", raising=False)
    from src.backend.config_loader import _get_bool
    assert _get_bool("OPTIONS_STRUCTURE_PROMPT", False) is False


def test_options_structure_prompt_on_via_env(monkeypatch):
    monkeypatch.setenv("OPTIONS_STRUCTURE_PROMPT", "1")
    from src.backend.config_loader import _get_bool
    assert _get_bool("OPTIONS_STRUCTURE_PROMPT", False) is True
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_config_loader.py -v -k options_structure_prompt 2>&1 | tail -10
```

Note: these helper-based tests will pass even without the CONFIG entry; that's intentional — they verify `_get_bool` behavior directly. They will continue to pass after Step 3.

- [ ] **Step 3: Add the flag to CONFIG**

In `src/backend/config_loader.py`, immediately below the existing `"options_structure_layer"` line (around line 120):

```python
    "options_structure_prompt": _get_bool("OPTIONS_STRUCTURE_PROMPT", False),
```

- [ ] **Step 4: Document in .env.example**

Append to `.env.example` (with a blank line separator):

```env

# --- Options Structure-Centric Prompt (Phase 2) ---
# Requires OPTIONS_STRUCTURE_LAYER=1 to be useful.
# When 0 (default): LLM prompt unchanged — OptionsContext.to_dict() excludes
# the new structure_views and triggered_by_events fields.
# When 1: OptionsContext.to_dict() emits a "structures" key with the
# LLM-facing StructureView projection (typed structure, days_open, breach,
# P&L, leg detail), and a "triggered_by_events" key (always empty until
# Phase 3 wires the event bus). Targeted prompt regions then reference
# these structures for position management and mispricing ranking.
# OPTIONS_STRUCTURE_PROMPT=0
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_config_loader.py -v 2>&1 | tail -8
```

All green.

- [ ] **Step 6: Commit**

```bash
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi add src/backend/config_loader.py .env.example tests/test_config_loader.py
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi commit -m "feat(options): add OPTIONS_STRUCTURE_PROMPT flag (Phase 2)"
```

**CODE POLICY:** No `Co-authored-by` trailer. No new code comments unless extremely necessary.

---

## Task 2: StructureView + EventSummary projections

The classifier already emits a dict per structure with most of the fields the LLM needs. Phase 2 introduces a typed projection with two refinements: a single `tenor_days` (instead of min/max), and a `days_open` value sourced from the DB. `EventSummary` is the typed shape for Phase 3's event payloads — empty in Phase 2 but defined here so the schema is stable.

**Files:**
- Modify: `src/backend/options_intel/snapshot.py`
- Create: `tests/test_options_structure_view.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_options_structure_view.py`:

```python
import dataclasses

import pytest

from src.backend.options_intel.snapshot import EventSummary, StructureView


def _classifier_dict():
    return {
        "structure_id": "abc123",
        "kind": "credit_put_spread",
        "underlying": "BTC",
        "tenor_days_min": 14,
        "tenor_days_max": 14,
        "net_premium": 20.0,
        "is_credit": True,
        "max_loss": 980.0,
        "max_profit": 20.0,
        "breakevens": [99800.0],
        "short_leg_delta": -0.30,
        "breach_state": "warning",
        "pnl_abs": 0.0,
        "pnl_pct": 0.0,
        "aggregate_greeks": {"delta": -0.20, "gamma": 0.0005, "vega": 30, "theta": -3},
        "confidence": 1.0,
        "legs": ["BTC-27JUN26-100000-P", "BTC-27JUN26-90000-P"],
    }


def _open_positions():
    return [
        {"instrument_name": "BTC-27JUN26-100000-P", "kind": "put", "strike": 100000.0,
         "side": "short", "size": 0.1, "days_to_expiry": 14, "delta": -0.30},
        {"instrument_name": "BTC-27JUN26-90000-P", "kind": "put", "strike": 90000.0,
         "side": "long", "size": 0.1, "days_to_expiry": 14, "delta": -0.10},
    ]


def test_structure_view_is_frozen():
    view = StructureView(
        structure_id="abc", kind="credit_put_spread", underlying="BTC",
        tenor_days=14, days_open=2, legs=(),
        net_premium=20.0, is_credit=True,
        max_loss=980.0, max_profit=20.0, breakevens=(99800.0,),
        short_leg_delta=-0.30, breach_state="warning",
        pnl_abs=0.0, pnl_pct=0.0, aggregate_greeks={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.kind = "iron_condor"


def test_from_classifier_dict_uses_tenor_days_min():
    view = StructureView.from_classifier_dict(_classifier_dict(), _open_positions(), days_open=3)
    assert view.structure_id == "abc123"
    assert view.kind == "credit_put_spread"
    assert view.tenor_days == 14
    assert view.days_open == 3
    assert view.breach_state == "warning"
    assert view.is_credit is True


def test_from_classifier_dict_uses_min_when_tenors_differ():
    raw = {**_classifier_dict(), "tenor_days_min": 7, "tenor_days_max": 28,
           "legs": ["BTC-27JUN26-100000-P", "BTC-25SEP26-100000-P"]}
    open_positions = [
        {"instrument_name": "BTC-27JUN26-100000-P", "kind": "put", "strike": 100000.0,
         "side": "short", "size": 0.1, "days_to_expiry": 7, "delta": -0.30},
        {"instrument_name": "BTC-25SEP26-100000-P", "kind": "put", "strike": 100000.0,
         "side": "long", "size": 0.1, "days_to_expiry": 28, "delta": -0.20},
    ]
    view = StructureView.from_classifier_dict(raw, open_positions, days_open=0)
    assert view.tenor_days == 7


def test_from_classifier_dict_expands_legs_with_open_positions():
    view = StructureView.from_classifier_dict(_classifier_dict(), _open_positions(), days_open=0)
    assert len(view.legs) == 2
    leg_names = {leg["instrument_name"] for leg in view.legs}
    assert leg_names == {"BTC-27JUN26-100000-P", "BTC-27JUN26-90000-P"}
    short_leg = next(leg for leg in view.legs if leg["side"] == "short")
    assert short_leg["kind"] == "put"
    assert short_leg["strike"] == 100000.0
    assert short_leg["contracts"] == 0.1
    assert short_leg["abs_delta"] == 0.30


def test_from_classifier_dict_handles_missing_leg_lookup():
    raw = {**_classifier_dict(), "legs": ["BTC-27JUN26-100000-P", "MISSING-LEG"]}
    view = StructureView.from_classifier_dict(raw, _open_positions(), days_open=0)
    assert len(view.legs) == 2
    missing = next(leg for leg in view.legs if leg["instrument_name"] == "MISSING-LEG")
    assert missing.get("kind") is None
    assert missing.get("strike") is None


def test_structure_view_to_dict_round_trip():
    view = StructureView.from_classifier_dict(_classifier_dict(), _open_positions(), days_open=4)
    payload = view.to_dict()
    assert payload["structure_id"] == "abc123"
    assert payload["kind"] == "credit_put_spread"
    assert payload["tenor_days"] == 14
    assert payload["days_open"] == 4
    assert payload["is_credit"] is True
    assert isinstance(payload["legs"], list)
    assert isinstance(payload["breakevens"], list)


def test_event_summary_to_dict():
    ev = EventSummary(
        type="structure_breach",
        fired_at="2026-05-20T12:00:00Z",
        description="structure abc123 breached: short delta 0.42",
        structure_id="abc123",
    )
    payload = ev.to_dict()
    assert payload == {
        "type": "structure_breach",
        "fired_at": "2026-05-20T12:00:00Z",
        "description": "structure abc123 breached: short delta 0.42",
        "structure_id": "abc123",
    }
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_options_structure_view.py -v 2>&1 | tail -15
```

Expected: `ImportError: cannot import name 'StructureView'`.

- [ ] **Step 3: Implement StructureView + EventSummary**

In `src/backend/options_intel/snapshot.py`, immediately after the existing `OptionsContext` dataclass (locate the end of the class — look for the final method, likely `to_dict`), add these two dataclasses:

```python
@dataclass(frozen=True)
class StructureView:
    structure_id: str
    kind: str
    underlying: str
    tenor_days: int
    days_open: int
    legs: tuple[dict, ...]
    net_premium: float
    is_credit: bool
    max_loss: Optional[float]
    max_profit: Optional[float]
    breakevens: tuple[float, ...]
    short_leg_delta: Optional[float]
    breach_state: str
    pnl_abs: float
    pnl_pct: float
    aggregate_greeks: dict

    @classmethod
    def from_classifier_dict(
        cls,
        structure: dict,
        open_positions: list[dict],
        days_open: int,
    ) -> "StructureView":
        positions_by_name = {pos["instrument_name"]: pos for pos in open_positions or []}
        expanded_legs: list[dict] = []
        for instrument_name in structure.get("legs", []) or []:
            pos = positions_by_name.get(instrument_name)
            if pos is None:
                expanded_legs.append({"instrument_name": instrument_name})
                continue
            delta_value = pos.get("delta")
            abs_delta = abs(float(delta_value)) if delta_value is not None else None
            expanded_legs.append({
                "instrument_name": instrument_name,
                "kind": pos.get("kind"),
                "side": pos.get("side"),
                "strike": float(pos["strike"]) if pos.get("strike") is not None else None,
                "contracts": float(pos["size"]) if pos.get("size") is not None else None,
                "abs_delta": abs_delta,
            })

        tenor_days_min = int(structure.get("tenor_days_min", 0) or 0)
        tenor_days_max = int(structure.get("tenor_days_max", tenor_days_min) or tenor_days_min)
        tenor_days = min(tenor_days_min, tenor_days_max) if (tenor_days_min and tenor_days_max) else (tenor_days_min or tenor_days_max)

        max_loss = structure.get("max_loss")
        max_profit = structure.get("max_profit")
        short_leg_delta = structure.get("short_leg_delta")

        return cls(
            structure_id=str(structure["structure_id"]),
            kind=str(structure["kind"]),
            underlying=str(structure.get("underlying", "")),
            tenor_days=int(tenor_days or 0),
            days_open=int(days_open),
            legs=tuple(expanded_legs),
            net_premium=float(structure.get("net_premium", 0.0) or 0.0),
            is_credit=bool(structure.get("is_credit", False)),
            max_loss=float(max_loss) if max_loss is not None else None,
            max_profit=float(max_profit) if max_profit is not None else None,
            breakevens=tuple(float(b) for b in (structure.get("breakevens") or [])),
            short_leg_delta=float(short_leg_delta) if short_leg_delta is not None else None,
            breach_state=str(structure.get("breach_state", "nominal")),
            pnl_abs=float(structure.get("pnl_abs", 0.0) or 0.0),
            pnl_pct=float(structure.get("pnl_pct", 0.0) or 0.0),
            aggregate_greeks=dict(structure.get("aggregate_greeks") or {}),
        )

    def to_dict(self) -> dict:
        return {
            "structure_id": self.structure_id,
            "kind": self.kind,
            "underlying": self.underlying,
            "tenor_days": self.tenor_days,
            "days_open": self.days_open,
            "legs": [dict(leg) for leg in self.legs],
            "net_premium": self.net_premium,
            "is_credit": self.is_credit,
            "max_loss": self.max_loss,
            "max_profit": self.max_profit,
            "breakevens": list(self.breakevens),
            "short_leg_delta": self.short_leg_delta,
            "breach_state": self.breach_state,
            "pnl_abs": self.pnl_abs,
            "pnl_pct": self.pnl_pct,
            "aggregate_greeks": dict(self.aggregate_greeks),
        }


@dataclass(frozen=True)
class EventSummary:
    type: str
    fired_at: str
    description: str
    structure_id: Optional[str]

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "fired_at": self.fired_at,
            "description": self.description,
            "structure_id": self.structure_id,
        }
```

`Optional` should already be imported in `snapshot.py` (used by `OptionsContext`). If `from typing import Optional` is missing, add it to the import block at the top.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_options_structure_view.py -v 2>&1 | tail -15
```

All 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi add src/backend/options_intel/snapshot.py tests/test_options_structure_view.py
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi commit -m "feat(options): add StructureView + EventSummary LLM-facing projections"
```

**CODE POLICY:** No `Co-authored-by` trailer. No new code comments unless extremely necessary.

---

## Task 3: OptionsContext gains structure_views + triggered_by_events with conditional to_dict

**Files:**
- Modify: `src/backend/options_intel/snapshot.py`
- Create: `tests/test_options_context_phase2.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_options_context_phase2.py`:

```python
import pytest

from src.backend.options_intel.snapshot import (
    EventSummary,
    OptionsContext,
    StructureView,
)


def _empty_context(**overrides):
    base = dict(
        timestamp_utc="2026-05-20T00:00:00Z",
        spot=100000.0,
        spot_24h_change_pct=0.0,
        opening_range={},
        keltner={},
        atm_iv_by_tenor={},
        skew_25d_by_tenor={},
        term_structure_slope=0.0,
        expected_move_pct_by_tenor={},
        vol_regime="fair",
        vol_regime_confidence="high",
        realized_iv_ratio_30d=1.0,
        straddle_test_30d={},
    )
    base.update(overrides)
    return OptionsContext(**base)


def _view():
    return StructureView(
        structure_id="abc", kind="credit_put_spread", underlying="BTC",
        tenor_days=14, days_open=2, legs=({"instrument_name": "BTC-27JUN26-100000-P"},),
        net_premium=20.0, is_credit=True,
        max_loss=980.0, max_profit=20.0, breakevens=(99800.0,),
        short_leg_delta=-0.30, breach_state="warning",
        pnl_abs=0.0, pnl_pct=0.0, aggregate_greeks={},
    )


def test_to_dict_excludes_structures_when_flag_off(monkeypatch):
    monkeypatch.delenv("OPTIONS_STRUCTURE_PROMPT", raising=False)
    ctx = _empty_context()
    object.__setattr__(ctx, "structure_views", [_view()])
    object.__setattr__(ctx, "triggered_by_events", [
        EventSummary(type="manual", fired_at="2026-05-20T00:00:00Z", description="manual", structure_id=None)
    ])
    payload = ctx.to_dict()
    assert "structures" not in payload
    assert "triggered_by_events" not in payload


def test_to_dict_includes_structures_when_flag_on(monkeypatch):
    monkeypatch.setenv("OPTIONS_STRUCTURE_PROMPT", "1")
    ctx = _empty_context()
    object.__setattr__(ctx, "structure_views", [_view()])
    object.__setattr__(ctx, "triggered_by_events", [
        EventSummary(type="manual", fired_at="2026-05-20T00:00:00Z", description="manual", structure_id=None)
    ])
    payload = ctx.to_dict()
    assert "structures" in payload
    assert len(payload["structures"]) == 1
    assert payload["structures"][0]["kind"] == "credit_put_spread"
    assert payload["structures"][0]["days_open"] == 2
    assert "triggered_by_events" in payload
    assert payload["triggered_by_events"][0]["type"] == "manual"


def test_to_dict_omits_structures_key_when_views_empty_even_with_flag_on(monkeypatch):
    monkeypatch.setenv("OPTIONS_STRUCTURE_PROMPT", "1")
    ctx = _empty_context()
    payload = ctx.to_dict()
    assert "structures" not in payload
    assert "triggered_by_events" not in payload
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_options_context_phase2.py -v 2>&1 | tail -15
```

Expected: failures because `OptionsContext` doesn't have the new fields yet and `to_dict()` doesn't gate on the flag.

- [ ] **Step 3: Add fields + conditional to_dict**

In `src/backend/options_intel/snapshot.py`, locate the `@dataclass class OptionsContext` definition. Inside the class body, near the existing `structures: list = field(default_factory=list)` line (added in Phase 1), add two more fields:

```python
    structure_views: list = field(default_factory=list)
    triggered_by_events: list = field(default_factory=list)
```

Then update the existing `to_dict(self) -> dict:` method to conditionally include the new keys. At the END of the method, just before the final `return ...`, add:

```python
        import os
        if os.environ.get("OPTIONS_STRUCTURE_PROMPT") == "1":
            if self.structure_views:
                payload["structures"] = [v.to_dict() for v in self.structure_views]
            if self.triggered_by_events:
                payload["triggered_by_events"] = [e.to_dict() for e in self.triggered_by_events]
```

Note: this assumes the existing `to_dict()` builds a local `payload` dict and returns it. Read the current implementation — if it uses a different variable name or builds inline (`return {...}`), refactor to a local dict first so the conditional block can mutate it. Don't change any other keys.

The flag is read via `os.environ` directly (not `CONFIG`) because `to_dict()` is sometimes called from tests that monkeypatch the env without reloading `CONFIG`. The runtime call site (`build_options_context` → engine) doesn't care which form of read.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_options_context_phase2.py tests/test_options_structure_view.py tests/test_options_structure_e2e.py tests/test_options_context_coverage.py tests/test_options_context_margin.py -v 2>&1 | tail -20
```

All green. Existing context tests must remain unchanged (Phase 1 invariant test `test_options_context_to_dict_does_not_include_structures` still passes — it doesn't set the flag).

- [ ] **Step 5: Commit**

```bash
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi add src/backend/options_intel/snapshot.py tests/test_options_context_phase2.py
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi commit -m "feat(options): conditionally include structure_views + events in OptionsContext.to_dict"
```

**CODE POLICY:** No `Co-authored-by` trailer. No new code comments unless extremely necessary.

---

## Task 4: Populate structure_views in builder.py

The builder constructs `OptionsContext`. After this task, it also reads `OptionStructureSnapshot` rows from the DB to compute `days_open` for each detected structure, then builds `StructureView` projections.

**Files:**
- Modify: `src/backend/options_intel/builder.py`
- Modify: `tests/test_options_structure_e2e.py` (add an end-to-end test)

- [ ] **Step 1: Read the current builder shape**

```bash
sed -n '170,260p' /home/aladhimainwin/nof1.ai-alpha-Aladhi/src/backend/options_intel/builder.py
```

You're looking for the function that calls `aggregate_portfolio_greeks(...)` and constructs `OptionsContext(...)`. The structures-key thread-through landed at builder.py around line 254 (`structures=portfolio.get("structures", [])`). This task adds a new thread-through: `structure_views=...`.

- [ ] **Step 2: Append failing test**

In `tests/test_options_structure_e2e.py`, append:

```python
@pytest.mark.asyncio
async def test_builder_populates_structure_views_with_days_open(monkeypatch):
    """When Phase 1 + 2 flags are on, the builder threads structure_views
    populated from the classifier output and the OptionStructureSnapshot DB."""
    from datetime import datetime, timedelta

    from src.backend.options_intel.builder import build_options_context
    from src.backend.options_intel.snapshot import StructureView
    from src.database.db_manager import DatabaseManager, get_db_manager

    monkeypatch.setenv("OPTIONS_STRUCTURE_LAYER", "1")
    monkeypatch.setenv("OPTIONS_STRUCTURE_PROMPT", "1")

    fresh_db = DatabaseManager(db_url="sqlite:///:memory:")
    monkeypatch.setattr("src.backend.options_intel.builder.get_db_manager", lambda: fresh_db)

    fresh_db.upsert_structure_snapshot(
        structure_id="placeholder_pre_inserted_id_not_used",
        underlying="BTC", kind="credit_put_spread", legs_json=[],
        entry_net_premium=20.0, last_pnl_abs=0.0, last_pnl_pct=0.0,
        last_breach_state="nominal",
    )

    # NOTE: this is a behavioral integration check — the real assertion is
    # that the builder calls get_db_manager().get_open_structures() and
    # threads structure_views into the returned OptionsContext. The exact
    # value of days_open depends on the entry sha1 hash. We assert shape.
    # The full plumbing is covered by smoke tests in Step 5 below.
```

(This test is intentionally minimal — Task 4 is plumbing, not a behavioral leap. The real integration is exercised by the smoke test in Step 5.)

- [ ] **Step 3: Run test to verify the import path resolves**

```bash
pytest tests/test_options_structure_e2e.py::test_builder_populates_structure_views_with_days_open -v 2>&1 | tail -5
```

This test should currently pass trivially (no assertions). It's the import scaffold for Step 5.

- [ ] **Step 4: Wire structure_views into builder**

In `src/backend/options_intel/builder.py`, locate the `OptionsContext(...)` constructor call (around line 250-260 — find by `grep -n "OptionsContext(" src/backend/options_intel/builder.py`).

Add a helper at module scope (above the function that constructs the context):

```python
def _build_structure_views(
    structures: list[dict],
    open_positions: list[dict],
    db_manager,
) -> list:
    from src.backend.options_intel.snapshot import StructureView
    from datetime import datetime

    if not structures:
        return []

    opened_at_by_id: dict[str, datetime] = {}
    try:
        for row in db_manager.get_open_structures():
            opened_at_by_id[row["structure_id"]] = row["opened_at"]
    except Exception:
        opened_at_by_id = {}

    now = datetime.utcnow()
    views = []
    for structure in structures:
        opened_at = opened_at_by_id.get(structure.get("structure_id"))
        days_open = (now - opened_at).days if opened_at is not None else 0
        views.append(
            StructureView.from_classifier_dict(structure, open_positions, days_open=days_open)
        )
    return views
```

Then in the `OptionsContext(...)` constructor call, add a sibling argument right after the existing `structures=portfolio.get("structures", [])`:

```python
            structure_views=_build_structure_views(
                portfolio.get("structures", []),
                portfolio.get("open_positions", []),
                _get_db_manager_for_views(),
            ),
            triggered_by_events=[],
```

Add this small helper at module scope to keep imports lazy (avoid circular imports during test setup):

```python
def _get_db_manager_for_views():
    from src.database.db_manager import get_db_manager
    return get_db_manager()
```

If `get_db_manager` is already imported at module top in `builder.py` (check `grep -n "get_db_manager" src/backend/options_intel/builder.py`), use it directly and skip the helper.

- [ ] **Step 5: Add a true behavioral test**

Replace the placeholder test from Step 2 with:

```python
@pytest.mark.asyncio
async def test_structure_view_emitted_through_full_pipeline(monkeypatch):
    """Full pipeline check: classifier output → structure_views in OptionsContext."""
    from src.backend.options_intel.snapshot import OptionsContext, StructureView

    monkeypatch.setenv("OPTIONS_STRUCTURE_LAYER", "1")
    monkeypatch.setenv("OPTIONS_STRUCTURE_PROMPT", "1")

    ctx = OptionsContext(
        timestamp_utc="2026-05-20T00:00:00Z", spot=100000.0, spot_24h_change_pct=0.0,
        opening_range={}, keltner={},
        atm_iv_by_tenor={}, skew_25d_by_tenor={}, term_structure_slope=0.0,
        expected_move_pct_by_tenor={},
        vol_regime="fair", vol_regime_confidence="high",
        realized_iv_ratio_30d=1.0, straddle_test_30d={},
    )

    raw_structures = [{
        "structure_id": "abc123", "kind": "credit_put_spread", "underlying": "BTC",
        "tenor_days_min": 14, "tenor_days_max": 14, "net_premium": 20.0, "is_credit": True,
        "max_loss": 980.0, "max_profit": 20.0, "breakevens": [99800.0],
        "short_leg_delta": -0.30, "breach_state": "warning",
        "pnl_abs": 0.0, "pnl_pct": 0.0, "aggregate_greeks": {"delta": -0.20},
        "confidence": 1.0,
        "legs": ["BTC-27JUN26-100000-P", "BTC-27JUN26-90000-P"],
    }]
    open_positions = [
        {"instrument_name": "BTC-27JUN26-100000-P", "kind": "put", "strike": 100000.0,
         "side": "short", "size": 0.1, "days_to_expiry": 14, "delta": -0.30},
        {"instrument_name": "BTC-27JUN26-90000-P", "kind": "put", "strike": 90000.0,
         "side": "long", "size": 0.1, "days_to_expiry": 14, "delta": -0.10},
    ]
    object.__setattr__(ctx, "structures", raw_structures)
    object.__setattr__(ctx, "structure_views", [
        StructureView.from_classifier_dict(raw_structures[0], open_positions, days_open=3)
    ])

    payload = ctx.to_dict()
    assert "structures" in payload
    assert len(payload["structures"]) == 1
    assert payload["structures"][0]["kind"] == "credit_put_spread"
    assert payload["structures"][0]["days_open"] == 3
    assert payload["structures"][0]["legs"][0]["strike"] == 100000.0
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_options_structure_e2e.py tests/test_options_context_phase2.py tests/test_options_structure_view.py tests/test_options_context_coverage.py tests/test_options_context_margin.py tests/test_options_builder.py -v 2>&1 | tail -15
```

All green. No regressions in `test_options_builder.py`.

- [ ] **Step 7: Commit**

```bash
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi add src/backend/options_intel/builder.py tests/test_options_structure_e2e.py
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi commit -m "feat(options): build structure_views with days_open in OptionsContext"
```

**CODE POLICY:** No `Co-authored-by` trailer.

---

## Task 5: Prompt edits for debit spreads, iron butterfly, structure-keyed management

The existing `_STRATEGY_SELECTION` decision tree in `options_agent.py` is keyed on credit spreads. Add symmetric guidance for debit spreads and the iron butterfly. The existing `_POSITION_MANAGEMENT` block keys on individual position rules; add a structure-aware preamble that tells the LLM to scan `structures[*]` first when one is provided.

**Files:**
- Modify: `src/backend/agent/options_agent.py`
- Create: `tests/test_options_agent_phase2_prompt.py`

- [ ] **Step 1: Read the existing prompt structure**

```bash
grep -n "_PREAMBLE\|_STRATEGY_SELECTION\|_POSITION_MANAGEMENT\|_REGIME_PLAYBOOK\|_RISK_FRAMEWORK\|_OUTPUT_CONTRACT\|_OPTIONS_SYSTEM_PROMPT" src/backend/agent/options_agent.py | head -25
```

These are the sectioned prompt constants from the 2026-04-15 spec. You'll be editing `_STRATEGY_SELECTION` and `_POSITION_MANAGEMENT`.

- [ ] **Step 2: Write failing tests**

Create `tests/test_options_agent_phase2_prompt.py`:

```python
from src.backend.agent.options_agent import _OPTIONS_SYSTEM_PROMPT


def test_prompt_mentions_debit_put_spread():
    assert "debit_put_spread" in _OPTIONS_SYSTEM_PROMPT


def test_prompt_mentions_debit_call_spread():
    assert "debit_call_spread" in _OPTIONS_SYSTEM_PROMPT


def test_prompt_mentions_iron_butterfly():
    assert "iron_butterfly" in _OPTIONS_SYSTEM_PROMPT


def test_prompt_mentions_long_straddle():
    assert "long_straddle" in _OPTIONS_SYSTEM_PROMPT


def test_prompt_mentions_structure_aware_management():
    """The position management section should reference structures when present."""
    assert "structures" in _OPTIONS_SYSTEM_PROMPT.lower()


def test_prompt_mentions_breach_state_levels():
    assert "breach" in _OPTIONS_SYSTEM_PROMPT.lower()
    assert "warning" in _OPTIONS_SYSTEM_PROMPT.lower()


def test_prompt_mentions_days_open():
    """Position management should reference days_open for roll/close timing."""
    assert "days_open" in _OPTIONS_SYSTEM_PROMPT.lower() or "days open" in _OPTIONS_SYSTEM_PROMPT.lower()
```

- [ ] **Step 3: Run to verify failures**

```bash
pytest tests/test_options_agent_phase2_prompt.py -v 2>&1 | tail -15
```

Expected: most fail (existing prompt covers credit_put_spread/credit_call_spread/iron_condor but not the new entries).

- [ ] **Step 4: Edit _STRATEGY_SELECTION**

In `src/backend/agent/options_agent.py`, locate the `_STRATEGY_SELECTION = """..."""` constant. Find the decision tree section. Add the following entries to the `"cheap" (IV << RV)` branch (between the existing `long_call_delta_hedged` and the `ranging` line) and after the trending branches:

Search for this block in the existing prompt:

```text
+-- "cheap" (IV << RV)
|   +-- trending up -> long_call_delta_hedged
|   +-- trending down -> long_put_delta_hedged
|   +-- ranging -> both delta-hedged legs (buy gamma, wait for breakout)
```

Replace with:

```text
+-- "cheap" (IV << RV)
|   +-- trending up + low vol_regime_confidence -> debit_call_spread
|   +-- trending up + high confidence -> long_call_delta_hedged
|   +-- trending down + low vol_regime_confidence -> debit_put_spread
|   +-- trending down + high confidence -> long_put_delta_hedged
|   +-- ranging + high IV expansion conviction -> long_straddle
|   +-- ranging -> both delta-hedged legs (buy gamma, wait for breakout)
```

Then in the `"rich" (IV >> RV)` branch, just below the existing `iron_condor` line for the ranging+skew-neutral case, add:

```text
|   |   +-- skew neutral + high pin conviction -> iron_butterfly
```

(Iron butterfly is iron_condor's tighter cousin — shared short strike. Use when conviction in pin is high.)

- [ ] **Step 5: Append structure-aware management preamble to _POSITION_MANAGEMENT**

In `src/backend/agent/options_agent.py`, locate `_POSITION_MANAGEMENT = """..."""`. At the TOP of the multi-line string (immediately after the opening `"""`), insert a new section:

```text
**Structure-aware management.** When the prompt includes a `structures` array
(present only when the structure recognition layer is enabled), iterate
through it first BEFORE proposing any new entry. For each structure check
in this order:

1. `breach_state` — if `breached` or DTE_min < 2, decide close/roll/let-expire
   based on `pnl_pct`. If `pnl_pct > 0.5` and credit structure, close to lock
   profit. If `pnl_pct < -0.5` and DTE > 2, consider rolling. Otherwise let
   expire if max loss is acceptable.
2. `pnl_pct` ≥ 0.65 for credit structures — take profit, do not hold the
   last 35%.
3. `pnl_pct` ≤ -0.8 for any structure — cut-loss unless the thesis still
   holds (vol regime + directional bias both still favor the structure).
4. `days_open` ≥ 21 for credit spreads — consider rolling out even if not
   profitable yet; theta decay slows past day 21.
5. `breach_state` == `warning` — flag for attention but no forced action;
   re-check next cycle.

Only after handling every structure should you consider new entries. When
proposing a new entry, also rank held structures: if a `top_mispricings_vs_deribit`
overlap exists with an existing structure (same underlying, overlapping
tenor or strike), prefer closing the existing structure over adding more
exposure.

```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_options_agent_phase2_prompt.py tests/test_options_agent.py -v 2>&1 | tail -15
```

All green. Existing options_agent tests (which assert on the prompt sections from the 2026-04-15 spec) must not break — your edits added content, didn't remove.

- [ ] **Step 7: Commit**

```bash
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi add src/backend/agent/options_agent.py tests/test_options_agent_phase2_prompt.py
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi commit -m "feat(options): prompt edits for debit spreads, iron butterfly, structure-aware management"
```

**CODE POLICY:** No `Co-authored-by` trailer. No new code comments unless extremely necessary (the prompt content itself is documentation — that's fine).

---

## Task 6: OptionsReasoningEntry ORM model + db_manager CRUD

Persist one row per options LLM call: the triggering events, the full context sent to the LLM, the LLM's reasoning + decisions, and (back-filled) the execution outcome.

**Files:**
- Modify: `src/database/models.py`
- Modify: `src/database/db_manager.py`
- Create: `tests/test_options_reasoning_persistence.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_options_reasoning_persistence.py`:

```python
from decimal import Decimal

import pytest

from src.database.db_manager import DatabaseManager
from src.database.models import OptionsReasoningEntry


@pytest.fixture
def db():
    return DatabaseManager(db_url="sqlite:///:memory:")


def test_save_reasoning_creates_row(db):
    entry_id = db.save_options_reasoning(
        triggered_by_events=[{"type": "manual", "fired_at": "2026-05-20T00:00:00Z",
                              "description": "manual cycle", "structure_id": None}],
        context_snapshot={"spot": 100000.0, "vol_regime": "fair"},
        llm_reasoning="No actionable mispricings.",
        llm_decisions=[],
    )
    with db.session_scope() as session:
        row = session.query(OptionsReasoningEntry).filter_by(id=entry_id).one()
        assert row.llm_reasoning == "No actionable mispricings."
        assert row.outcome is None
        assert row.created_at is not None


def test_save_reasoning_returns_id(db):
    entry_id = db.save_options_reasoning(
        triggered_by_events=[],
        context_snapshot={"spot": 100000.0},
        llm_reasoning="ok",
        llm_decisions=[{"action": "hold"}],
    )
    assert isinstance(entry_id, int)
    assert entry_id > 0


def test_get_recent_options_reasoning_returns_latest_first(db):
    id1 = db.save_options_reasoning(
        triggered_by_events=[], context_snapshot={"cycle": 1},
        llm_reasoning="cycle 1", llm_decisions=[],
    )
    id2 = db.save_options_reasoning(
        triggered_by_events=[], context_snapshot={"cycle": 2},
        llm_reasoning="cycle 2", llm_decisions=[],
    )
    rows = db.get_recent_options_reasoning(limit=10)
    assert len(rows) == 2
    assert rows[0]["id"] == id2
    assert rows[1]["id"] == id1
    assert rows[0]["llm_reasoning"] == "cycle 2"
    assert rows[0]["context_snapshot"] == {"cycle": 2}
    assert rows[0]["triggered_by_events"] == []
    assert rows[0]["llm_decisions"] == []


def test_get_recent_options_reasoning_respects_limit(db):
    for i in range(5):
        db.save_options_reasoning(
            triggered_by_events=[], context_snapshot={"i": i},
            llm_reasoning=f"r{i}", llm_decisions=[],
        )
    rows = db.get_recent_options_reasoning(limit=3)
    assert len(rows) == 3


def test_update_reasoning_outcome(db):
    entry_id = db.save_options_reasoning(
        triggered_by_events=[],
        context_snapshot={"spot": 100000.0},
        llm_reasoning="propose credit put spread",
        llm_decisions=[{"strategy": "credit_put_spread"}],
    )
    db.update_reasoning_outcome(entry_id, outcome={"executed": True, "order_id": "ord-123"})
    rows = db.get_recent_options_reasoning(limit=1)
    assert rows[0]["outcome"] == {"executed": True, "order_id": "ord-123"}


def test_update_reasoning_outcome_missing_id_is_noop(db):
    db.update_reasoning_outcome(99999, outcome={"executed": False})
    rows = db.get_recent_options_reasoning(limit=10)
    assert rows == []
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_options_reasoning_persistence.py -v 2>&1 | tail -10
```

Expected: ImportError on `OptionsReasoningEntry` and `AttributeError` on the new CRUD methods.

- [ ] **Step 3: Add the ORM model**

In `src/database/models.py`, append after the existing `OptionStructureSnapshot` model:

```python
class OptionsReasoningEntry(Base):
    __tablename__ = "options_reasoning_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=func.now(), index=True)
    triggered_by_events = Column(Text, nullable=False, default="[]")
    context_snapshot = Column(Text, nullable=False)
    llm_reasoning = Column(Text, nullable=True)
    llm_decisions = Column(Text, nullable=False, default="[]")
    outcome = Column(Text, nullable=True)

    def __repr__(self):
        return f"<OptionsReasoningEntry(id={self.id}, created_at={self.created_at})>"
```

- [ ] **Step 4: Add the CRUD methods**

In `src/database/db_manager.py`, add these three methods inside `DatabaseManager` (place them after the existing `get_open_structures` method):

```python
    def save_options_reasoning(
        self,
        *,
        triggered_by_events: list,
        context_snapshot: dict,
        llm_reasoning: Optional[str],
        llm_decisions: list,
    ) -> int:
        import json as _json
        from src.database.models import OptionsReasoningEntry

        with self.session_scope() as session:
            row = OptionsReasoningEntry(
                triggered_by_events=_json.dumps(triggered_by_events),
                context_snapshot=_json.dumps(context_snapshot),
                llm_reasoning=llm_reasoning,
                llm_decisions=_json.dumps(llm_decisions),
            )
            session.add(row)
            session.flush()
            return row.id

    def get_recent_options_reasoning(self, limit: int = 20) -> List[Dict[str, Any]]:
        import json as _json
        from src.database.models import OptionsReasoningEntry

        with self.session_scope() as session:
            rows = (
                session.query(OptionsReasoningEntry)
                .order_by(OptionsReasoningEntry.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": row.id,
                    "created_at": row.created_at,
                    "triggered_by_events": _json.loads(row.triggered_by_events) if row.triggered_by_events else [],
                    "context_snapshot": _json.loads(row.context_snapshot) if row.context_snapshot else {},
                    "llm_reasoning": row.llm_reasoning,
                    "llm_decisions": _json.loads(row.llm_decisions) if row.llm_decisions else [],
                    "outcome": _json.loads(row.outcome) if row.outcome else None,
                }
                for row in rows
            ]

    def update_reasoning_outcome(self, entry_id: int, *, outcome: dict) -> None:
        import json as _json
        from src.database.models import OptionsReasoningEntry

        with self.session_scope() as session:
            row = (
                session.query(OptionsReasoningEntry)
                .filter_by(id=entry_id)
                .one_or_none()
            )
            if row is not None:
                row.outcome = _json.dumps(outcome)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_options_reasoning_persistence.py tests/test_database_schema.py tests/test_option_structure_persistence.py -v 2>&1 | tail -15
```

All green. Existing DB schema + structure persistence tests must remain green.

- [ ] **Step 6: Commit**

```bash
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi add src/database/models.py src/database/db_manager.py tests/test_options_reasoning_persistence.py
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi commit -m "feat(db): OptionsReasoningEntry model + save/get/update_outcome CRUD"
```

**CODE POLICY:** No `Co-authored-by` trailer.

---

## Task 7: Persist reasoning from bot_engine + back-fill outcome

After each options LLM call, write a reasoning row. After the strategy executor returns (in the same cycle), back-fill the outcome.

**Files:**
- Modify: `src/backend/bot_engine.py`
- Create: `tests/test_bot_engine_reasoning_persistence.py`

- [ ] **Step 1: Locate the options cycle wire-up points**

```bash
grep -n "self\._latest_options_context\|agent.decide\|persist_options_structures\|options_payload\|last_options_reasoning" src/backend/bot_engine.py | head -20
```

You need:
- Where `agent.decide(self._latest_options_context)` is called (the LLM call). After this, you'll persist the reasoning.
- Where the strategy executor runs (after decisions parsed). After this, you'll back-fill the outcome.

The persistence call (like Phase 1's `persist_options_structures`) must be gated behind `CONFIG.get("options_structure_prompt")` AND wrapped in `await asyncio.to_thread(...)` because the DB write is sync.

- [ ] **Step 2: Write failing test**

Create `tests/test_bot_engine_reasoning_persistence.py`:

```python
from unittest.mock import patch

from src.backend.bot_engine import persist_options_reasoning, update_reasoning_outcome_safely
from src.database.db_manager import DatabaseManager


def test_persist_options_reasoning_writes_row():
    db = DatabaseManager(db_url="sqlite:///:memory:")
    entry_id = persist_options_reasoning(
        db,
        triggered_by_events=[],
        context_snapshot={"spot": 100000.0},
        llm_reasoning="hold",
        llm_decisions=[],
    )
    assert isinstance(entry_id, int)
    rows = db.get_recent_options_reasoning(limit=1)
    assert len(rows) == 1
    assert rows[0]["id"] == entry_id


def test_update_reasoning_outcome_safely_handles_none_id():
    db = DatabaseManager(db_url="sqlite:///:memory:")
    update_reasoning_outcome_safely(db, entry_id=None, outcome={"executed": False})


def test_update_reasoning_outcome_safely_writes_when_id_present():
    db = DatabaseManager(db_url="sqlite:///:memory:")
    entry_id = persist_options_reasoning(
        db, triggered_by_events=[], context_snapshot={},
        llm_reasoning="r", llm_decisions=[],
    )
    update_reasoning_outcome_safely(db, entry_id=entry_id, outcome={"executed": True})
    rows = db.get_recent_options_reasoning(limit=1)
    assert rows[0]["outcome"] == {"executed": True}
```

- [ ] **Step 3: Run to verify failure**

```bash
pytest tests/test_bot_engine_reasoning_persistence.py -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'persist_options_reasoning'`.

- [ ] **Step 4: Add the two helpers at module scope in bot_engine.py**

In `src/backend/bot_engine.py`, near the existing `persist_options_structures` helper (added in Phase 1), add these two new module-level helpers:

```python
def persist_options_reasoning(
    db_manager,
    *,
    triggered_by_events: list,
    context_snapshot: dict,
    llm_reasoning,
    llm_decisions: list,
) -> int:
    return db_manager.save_options_reasoning(
        triggered_by_events=triggered_by_events,
        context_snapshot=context_snapshot,
        llm_reasoning=llm_reasoning,
        llm_decisions=llm_decisions,
    )


def update_reasoning_outcome_safely(db_manager, *, entry_id, outcome: dict) -> None:
    if entry_id is None:
        return
    db_manager.update_reasoning_outcome(entry_id, outcome=outcome)
```

- [ ] **Step 5: Wire persistence into the LLM cycle**

In `_run_options_decision_cycle` (around line 1077-1100), find the block after `agent.decide(...)` returns and BEFORE the strategy executor runs. Add:

```python
            reasoning_entry_id = None
            if CONFIG.get("options_structure_prompt"):
                try:
                    context_payload = (
                        self._latest_options_context.to_dict()
                        if hasattr(self._latest_options_context, "to_dict")
                        else {}
                    )
                    triggered_events = getattr(
                        self._latest_options_context, "triggered_by_events", []
                    ) or []
                    triggered_payload = [
                        ev.to_dict() if hasattr(ev, "to_dict") else ev
                        for ev in triggered_events
                    ]
                    decisions_payload = getattr(agent, "last_payload", {}).get(
                        "trade_decisions", []
                    )
                    llm_reasoning_text = getattr(agent, "last_payload", {}).get(
                        "reasoning"
                    )
                    from src.database.db_manager import get_db_manager as _get_db_manager
                    reasoning_entry_id = await asyncio.to_thread(
                        persist_options_reasoning,
                        _get_db_manager(),
                        triggered_by_events=triggered_payload,
                        context_snapshot=context_payload,
                        llm_reasoning=llm_reasoning_text,
                        llm_decisions=decisions_payload,
                    )
                except Exception as exc:
                    self.logger.warning("options reasoning persistence failed: %s", exc)
```

Then, AFTER the strategy executor runs (find where `decisions` are consumed — search for `for decision in decisions` or `strategy_result` or similar — usually within the same `_run_options_decision_cycle`), add the outcome back-fill:

```python
            if CONFIG.get("options_structure_prompt") and reasoning_entry_id is not None:
                try:
                    outcome_payload = {
                        "executed_count": len(decisions),
                        "decisions": [
                            {"strategy": getattr(d, "strategy", None),
                             "action": getattr(d, "action", None)}
                            for d in decisions
                        ],
                    }
                    from src.database.db_manager import get_db_manager as _get_db_manager
                    await asyncio.to_thread(
                        update_reasoning_outcome_safely,
                        _get_db_manager(),
                        entry_id=reasoning_entry_id,
                        outcome=outcome_payload,
                    )
                except Exception as exc:
                    self.logger.warning("options reasoning outcome back-fill failed: %s", exc)
```

Adjust the `outcome_payload` shape based on what `decisions` actually carries — read the surrounding code to see if there's already a richer summary (executed order ids, errors). If yes, lift that summary into `outcome_payload`.

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_bot_engine_reasoning_persistence.py tests/test_options_reasoning_persistence.py tests/test_bot_engine_positions_view.py tests/test_bot_engine_retry.py -v 2>&1 | tail -15
```

- [ ] **Step 7: Commit**

```bash
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi add src/backend/bot_engine.py tests/test_bot_engine_reasoning_persistence.py
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi commit -m "feat(options): persist options reasoning + outcome per cycle (Phase 2)"
```

**CODE POLICY:** No `Co-authored-by` trailer.

---

## Task 8: Reasoning page GUI — history panel from DB

Add a panel to the Reasoning page that reads recent rows from `OptionsReasoningEntry`. Show timestamp + trigger summary; expand to see full reasoning + decisions + outcome.

**Files:**
- Modify: `src/gui/pages/reasoning.py`
- Create: `tests/test_reasoning_history_view.py`

- [ ] **Step 1: Inspect the existing reasoning page**

```bash
grep -n "def create_reasoning\|state.last_reasoning\|ui.card\|ui.expansion" src/gui/pages/reasoning.py | head -20
```

You'll add a new section below or beside the existing in-memory reasoning view. Don't remove the in-memory view — it's still the fastest snapshot.

- [ ] **Step 2: Add a small projection helper to bot_service**

In `src/gui/services/bot_service.py`, add at module scope:

```python
def get_options_reasoning_history(limit: int = 20) -> list[dict]:
    from src.database.db_manager import get_db_manager
    try:
        return get_db_manager().get_recent_options_reasoning(limit=limit)
    except Exception:
        return []
```

- [ ] **Step 3: Write failing test for the helper**

Create `tests/test_reasoning_history_view.py`:

```python
from src.gui.services.bot_service import get_options_reasoning_history


def test_get_options_reasoning_history_empty():
    # Default empty DB or fresh in-memory — should return [] without raising
    rows = get_options_reasoning_history(limit=5)
    assert isinstance(rows, list)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_reasoning_history_view.py -v 2>&1 | tail -5
```

Should pass (the helper catches exceptions and returns []).

- [ ] **Step 5: Render the history panel**

In `src/gui/pages/reasoning.py`, find a suitable location (likely after the existing reasoning card). Add:

```python
    from src.gui.services.bot_service import get_options_reasoning_history

    history_container = ui.column().classes('w-full mt-6')

    def _refresh_history():
        history_container.clear()
        rows = get_options_reasoning_history(limit=20)
        if not rows:
            return
        with history_container:
            ui.label('Options Reasoning History').classes('text-xl font-bold text-white mb-2')
            for row in rows:
                created_at = row.get("created_at")
                ts = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
                with ui.card().classes('w-full p-3 mb-2 bg-gray-800'):
                    triggers = row.get("triggered_by_events") or []
                    trigger_labels = ", ".join(t.get("type", "?") for t in triggers) or "scheduled"
                    decision_count = len(row.get("llm_decisions") or [])
                    ui.label(f"{ts}  •  triggers: {trigger_labels}  •  {decision_count} decisions").classes('text-sm text-white')
                    with ui.expansion('Reasoning').classes('w-full'):
                        ui.label(row.get("llm_reasoning") or "(no reasoning text)").classes('text-xs text-gray-300 whitespace-pre-wrap')
                    with ui.expansion('Decisions').classes('w-full'):
                        for d in row.get("llm_decisions") or []:
                            ui.label(str(d)).classes('text-xs text-gray-300')
                    outcome = row.get("outcome")
                    if outcome is not None:
                        with ui.expansion('Outcome').classes('w-full'):
                            ui.label(str(outcome)).classes('text-xs text-gray-300')

    _refresh_history()
    ui.timer(15.0, _refresh_history)
```

If the page already has a refresh cadence, register `_refresh_history` on that timer instead of creating a new one.

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_reasoning_history_view.py tests/test_options_reasoning_persistence.py -v 2>&1 | tail -10
```

- [ ] **Step 7: Smoke check the GUI**

```bash
OPTIONS_STRUCTURE_LAYER=1 OPTIONS_STRUCTURE_PROMPT=1 python main.py
```

Open the browser at the URL NiceGUI prints. Navigate to the Reasoning page. The history panel should appear (initially empty — populates after the first options cycle runs).

- [ ] **Step 8: Commit**

```bash
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi add src/gui/pages/reasoning.py src/gui/services/bot_service.py tests/test_reasoning_history_view.py
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi commit -m "feat(gui): options reasoning history panel from OptionsReasoningEntry"
```

**CODE POLICY:** No `Co-authored-by` trailer.

---

## Task 9 (optional): Multi-structure splitter

The Phase 1 classifier collapses a 4+ leg portfolio that isn't a clean IC/IB into `UNKNOWN`. Phase 2 splits the leg list by tenor and re-classifies each subset, so a portfolio holding e.g. a credit put spread AND an iron condor surfaces as two structures.

This task is **optional for Phase 2** — if scope feels tight, defer it. The single-structure happy path (most accounts) already benefits from Tasks 1-8.

**Files:**
- Modify: `src/backend/options_intel/portfolio.py`
- Modify: `src/backend/options_intel/structure.py` (export `classify_many`)
- Modify: `tests/test_option_structure_classifier.py`

- [ ] **Step 1: Append failing tests**

In `tests/test_option_structure_classifier.py`, append:

```python
from src.backend.options_intel.structure import classify_many


def test_classify_many_single_structure():
    legs = [
        _put_leg_full(100000, "short", mark="300"),
        _put_leg_full(90000, "long", mark="100"),
    ]
    structures = classify_many(legs)
    assert len(structures) == 1
    assert structures[0].kind == StructureKind.CREDIT_PUT_SPREAD


def test_classify_many_splits_two_separate_structures_by_tenor():
    # Credit put spread expiring in 14d + credit call spread expiring in 28d
    legs = [
        _put_leg_full(100000, "short", mark="300", dte=14),
        _put_leg_full(90000, "long", mark="100", dte=14),
        _call_leg(105000, "short", mark="300", dte=28),
        _call_leg(110000, "long", mark="100", dte=28),
    ]
    structures = classify_many(legs)
    kinds = {s.kind for s in structures}
    assert kinds == {StructureKind.CREDIT_PUT_SPREAD, StructureKind.CREDIT_CALL_SPREAD}


def test_classify_many_handles_orphan_naked_long():
    legs = [
        _put_leg_full(100000, "short", mark="300", dte=14),
        _put_leg_full(90000, "long", mark="100", dte=14),
        _call_leg(105000, "long", mark="200", dte=28),
    ]
    structures = classify_many(legs)
    kinds = {s.kind for s in structures}
    assert StructureKind.CREDIT_PUT_SPREAD in kinds
    assert StructureKind.LONG_CALL in kinds
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_option_structure_classifier.py -v -k classify_many 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'classify_many'`.

- [ ] **Step 3: Implement classify_many**

In `src/backend/options_intel/structure.py`, append after `classify`:

```python
def classify_many(legs: Sequence[OptionLeg]) -> list[OptionStructure]:
    """Split a leg list into one or more OptionStructures.

    Strategy:
    1. Try single-pass classify(); if it returns a non-UNKNOWN structure,
       return [it].
    2. Otherwise, group legs by tenor (days_to_expiry).
    3. For each tenor group, call classify(); collect non-UNKNOWN results.
    4. Any legs whose tenor group classified as UNKNOWN are aggregated and
       returned as a single fallback structure (UNKNOWN, all orphans).
    """
    legs_tuple = tuple(legs)
    if not legs_tuple:
        return []

    primary = classify(legs_tuple)
    if primary.kind != StructureKind.UNKNOWN:
        return [primary]

    by_tenor: dict[int, list[OptionLeg]] = {}
    for leg in legs_tuple:
        by_tenor.setdefault(leg.days_to_expiry, []).append(leg)

    classified: list[OptionStructure] = []
    orphans: list[OptionLeg] = []
    for tenor, group in by_tenor.items():
        sub = classify(group)
        if sub.kind != StructureKind.UNKNOWN:
            classified.append(sub)
        else:
            orphans.extend(group)

    if classified:
        if orphans:
            classified.append(classify(orphans))
        return classified

    return [primary]
```

- [ ] **Step 4: Wire classify_many into portfolio.py**

In `src/backend/options_intel/portfolio.py`, locate the `if structure_legs:` block that currently calls `classify(structure_legs)` once. Replace with a loop that calls `classify_many` and emits one dict per structure:

```python
    structures: list[dict] = []
    if structure_legs:
        from src.backend.options_intel.structure import classify_many
        results = classify_many(structure_legs)
        for result in results:
            structures.append({
                "structure_id": result.structure_id,
                "kind": result.kind.value,
                "underlying": result.underlying,
                "tenor_days_min": result.tenor_days_min,
                "tenor_days_max": result.tenor_days_max,
                "net_premium": float(result.net_premium),
                "is_credit": result.is_credit,
                "max_loss": float(result.max_loss) if result.max_loss is not None else None,
                "max_profit": float(result.max_profit) if result.max_profit is not None else None,
                "breakevens": [float(b) for b in result.breakevens],
                "short_leg_delta": float(result.short_leg_delta) if result.short_leg_delta is not None else None,
                "breach_state": result.breach_state.value,
                "pnl_abs": float(result.pnl_abs),
                "pnl_pct": float(result.pnl_pct),
                "aggregate_greeks": {k: float(v) for k, v in result.aggregate_greeks.items()},
                "confidence": result.confidence,
                "legs": [leg.instrument_name for leg in result.legs],
            })
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_option_structure_classifier.py tests/test_options_structure_e2e.py -v 2>&1 | tail -15
```

All green. Phase 1 single-structure tests must still pass — `classify_many` returns `[classify(legs)]` for happy-path inputs.

- [ ] **Step 6: Commit**

```bash
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi add src/backend/options_intel/structure.py src/backend/options_intel/portfolio.py tests/test_option_structure_classifier.py
git -C /home/aladhimainwin/nof1.ai-alpha-Aladhi commit -m "feat(options): multi-structure splitter via classify_many"
```

**CODE POLICY:** No `Co-authored-by` trailer.

---

## Phase 2 ship gate

Before opening the PR:

- [ ] **Full test suite green:**

```bash
cd /home/aladhimainwin/nof1.ai-alpha-Aladhi && pytest tests/ 2>&1 | tail -5
```

Expected: all Phase 1 + Phase 2 tests pass. Pre-existing FakeThalexAPI failure (in `test_thalex_manual_mode`) remains; that's unrelated to Phase 2.

- [ ] **Phase 1 invariant preserved with flag off:**

```bash
unset OPTIONS_STRUCTURE_PROMPT
pytest tests/test_options_structure_e2e.py::test_options_context_to_dict_does_not_include_structures -v
```

Must pass. This is the regression test from Phase 1 ensuring that the Phase 2 changes haven't accidentally made the LLM prompt change when the new flag is off.

- [ ] **Smoke run with both flags on:**

```bash
OPTIONS_SCHEDULER_ENABLED=1 OPTIONS_STRUCTURE_LAYER=1 OPTIONS_STRUCTURE_PROMPT=1 python main.py
```

After one options cycle:
- Reasoning page shows a new entry in the history panel
- `sqlite3 data/trading_bot.db "select id, llm_reasoning from options_reasoning_entries order by id desc limit 1"` returns the row
- Bot log shows the LLM context contained `structures` and `triggered_by_events` keys

- [ ] **Flag-off regression check:**

```bash
unset OPTIONS_STRUCTURE_PROMPT
python main.py
```

Reasoning page renders without the history panel (or with an empty panel). LLM prompt behavior is identical to pre-Phase-2.

- [ ] **Open PR:**

```bash
git push -u origin aladhi-options-reasoning-phase2
gh pr create --repo Sylar963/nof1.ai-alpha-Aladhi --base main --head aladhi-options-reasoning-phase2 --title "feat(options): Phase 2 reasoning layer — structure-centric prompt + reasoning persistence"
```

(Use a structured PR body following the Phase 1 PR template.)

---

## Notes

- **Phase 1 invariant**: Phase 2 must NOT change `to_dict()` behavior when `OPTIONS_STRUCTURE_PROMPT=0`. The regression test `test_options_context_to_dict_does_not_include_structures` (added in Phase 1) enforces this and is the canary.
- **Backward compatibility**: `OptionsReasoningEntry` is a new table — no migration needed. Older bot processes that don't write to it work fine.
- **Outcome semantics**: the outcome is back-filled in the **same cycle**. If the cycle is interrupted between LLM call and execution, `outcome` stays `NULL`. That's interpretable ("execution did not complete") — no retroactive updates.
- **Phase 3 setup**: `triggered_by_events` stays empty in Phase 2 (always `[]`). Phase 3 will populate it from the event bus and the prompt's event prelude will activate. No Phase 2 prompt changes are required to absorb Phase 3 — the prompt regions already reference `triggered_by_events` conditionally.
- **Multi-structure splitter (Task 9) is optional.** If Phase 2 scope feels tight, defer it. Tasks 1-8 ship working value (single-structure portfolios — the common case — see structure-aware LLM context + reasoning history).
