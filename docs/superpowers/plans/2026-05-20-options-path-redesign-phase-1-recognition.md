# Options Path Redesign — Phase 1 (Recognition Layer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a structure-aware recognition layer that classifies a leg list into a typed `OptionStructure` (credit/debit spreads, ICs, calendars, etc.), persists structures to the database with stable identity, and groups Thalex positions in the GUI by structure. LLM context and prompt remain leg-based in this phase — Phase 2 wires structures into the LLM.

**Architecture:** New module `src/backend/options_intel/structure.py` exposes `OptionStructure`, `StructureKind`, and a pure `classify()` function. `aggregate_portfolio_greeks` is extended to emit a `structures` list alongside `open_positions`. A new `OptionStructureSnapshot` ORM table records structure identity, lifecycle (`opened_at`/`closed_at`), and rolling P&L baselines via update-in-place keyed by a sha1 hash of sorted `(instrument_name, side)` tuples. GUI positions page groups Thalex legs by `structure_id` when the `OPTIONS_STRUCTURE_LAYER` flag is enabled.

**Tech Stack:** Python 3.14 / asyncio, SQLAlchemy ORM (declarative_base in `src/database/models.py`), pytest + pytest-asyncio, NiceGUI for the GUI layer.

**Spec:** `docs/superpowers/specs/2026-05-20-options-path-redesign-design.md` (Section 1 — Recognition Layer)

---

## File Structure

**New files:**
- `src/backend/options_intel/structure.py` — `OptionLeg`, `StructureKind`, `BreachState`, `OptionStructure`, `classify()`
- `tests/test_option_structure_classifier.py` — golden + property tests for the classifier
- `tests/test_option_structure_persistence.py` — DB roundtrip tests
- `tests/test_options_structure_e2e.py` — portfolio.py + bot_engine.py integration

**Modified files:**
- `src/backend/agent/decision_schema.py` — add `debit_put_spread`, `debit_call_spread`, `iron_butterfly`, `long_straddle` to `VALID_STRATEGIES`
- `src/backend/options_intel/portfolio.py` — `aggregate_portfolio_greeks` returns `structures` in its result dict
- `src/database/models.py` — add `OptionStructureSnapshot` model
- `src/database/db_manager.py` — add `upsert_structure_snapshot`, `mark_structure_closed`, `get_open_structures`
- `src/backend/bot_engine.py` — call structure persistence after each options cycle
- `src/backend/config_loader.py` — load `OPTIONS_STRUCTURE_LAYER` flag
- `src/gui/pages/positions.py` — group Thalex legs by structure when flag on
- `.env.example` — document `OPTIONS_STRUCTURE_LAYER`

---

## Task 1: Extend VALID_STRATEGIES

Add the four new strategies the prompt will reference. This unblocks every subsequent task that mentions debit spreads / straddle / iron butterfly in tests.

**Files:**
- Modify: `src/backend/agent/decision_schema.py` (around line 31)
- Test: `tests/test_decision_schema.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_decision_schema.py`:

```python
from src.backend.agent.decision_schema import VALID_STRATEGIES, parse_decision

def test_valid_strategies_includes_phase1_additions():
    assert "debit_put_spread" in VALID_STRATEGIES
    assert "debit_call_spread" in VALID_STRATEGIES
    assert "iron_butterfly" in VALID_STRATEGIES
    assert "long_straddle" in VALID_STRATEGIES

def test_parse_decision_accepts_debit_put_spread():
    payload = {
        "reasoning": "IV cheap, bearish bias",
        "trade_decisions": [
            {
                "asset": "BTC",
                "action": "buy",
                "venue": "thalex",
                "strategy": "debit_put_spread",
                "underlying": "BTC",
                "tenor_days": 14,
                "entry_kind": "vertical",
                "rationale": "buy 95k put, sell 90k put",
                "legs": [
                    {"kind": "put", "side": "buy", "contracts": 0.1, "target_strike": 95000},
                    {"kind": "put", "side": "sell", "contracts": 0.1, "target_strike": 90000},
                ],
            }
        ],
    }
    result = parse_decision(payload)
    assert result.trade_decisions[0].strategy == "debit_put_spread"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_decision_schema.py::test_valid_strategies_includes_phase1_additions tests/test_decision_schema.py::test_parse_decision_accepts_debit_put_spread -v
```

Expected: both FAIL — `assert "debit_put_spread" in VALID_STRATEGIES` raises AssertionError; the parse test fails with `DecisionParseError: strategy must be one of {...}`.

- [ ] **Step 3: Add new strategies to VALID_STRATEGIES**

Edit `src/backend/agent/decision_schema.py`. Replace the existing `VALID_STRATEGIES = {...}` block with:

```python
VALID_STRATEGIES = {
    "credit_put_spread",
    "credit_call_spread",
    "debit_put_spread",
    "debit_call_spread",
    "iron_condor",
    "iron_butterfly",
    "long_call_delta_hedged",
    "long_put_delta_hedged",
    "long_straddle",
    "vol_arb",
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_decision_schema.py -v
```

Expected: all existing decision_schema tests still pass + the two new ones pass.

- [ ] **Step 5: Commit**

```bash
git add src/backend/agent/decision_schema.py tests/test_decision_schema.py
git commit -m "feat(options): add debit spreads, iron butterfly, long straddle to VALID_STRATEGIES"
```

---

## Task 2: Structure module skeleton — types only

Create the new module with the public dataclasses and enums. No classifier logic yet — that's tasks 3–8.

**Files:**
- Create: `src/backend/options_intel/structure.py`
- Test: `tests/test_option_structure_classifier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_option_structure_classifier.py`:

```python
from decimal import Decimal

from src.backend.options_intel.structure import (
    BreachState,
    OptionLeg,
    OptionStructure,
    StructureKind,
)


def test_structure_kind_enum_has_all_phase1_kinds():
    expected = {
        "credit_put_spread", "credit_call_spread",
        "debit_put_spread", "debit_call_spread",
        "iron_condor", "iron_butterfly",
        "long_call", "long_put", "long_straddle", "long_strangle",
        "calendar_put", "calendar_call",
        "diagonal_put", "diagonal_call",
        "unknown",
    }
    actual = {k.value for k in StructureKind}
    assert actual == expected


def test_breach_state_enum_values():
    assert {b.value for b in BreachState} == {"nominal", "warning", "breached"}


def test_option_leg_is_frozen_dataclass():
    leg = OptionLeg(
        instrument_name="BTC-27JUN26-100000-P",
        kind="put",
        strike=Decimal("100000"),
        side="long",
        contracts=Decimal("0.1"),
        days_to_expiry=14,
        mark_price=Decimal("1500"),
        delta=Decimal("-0.30"),
        gamma=None,
        vega=None,
        theta=None,
    )
    import pytest
    with pytest.raises(Exception):
        leg.kind = "call"  # frozen


def test_option_structure_is_frozen_dataclass():
    leg = OptionLeg(
        instrument_name="BTC-27JUN26-100000-P",
        kind="put",
        strike=Decimal("100000"),
        side="long",
        contracts=Decimal("0.1"),
        days_to_expiry=14,
        mark_price=Decimal("1500"),
        delta=Decimal("-0.30"),
        gamma=None,
        vega=None,
        theta=None,
    )
    structure = OptionStructure(
        structure_id="abc123",
        kind=StructureKind.LONG_PUT,
        underlying="BTC",
        legs=(leg,),
        tenor_days_min=14,
        tenor_days_max=14,
        net_premium=Decimal("-150"),
        is_credit=False,
        max_loss=Decimal("150"),
        max_profit=None,
        breakevens=(Decimal("98500"),),
        short_leg_delta=None,
        breach_state=BreachState.NOMINAL,
        pnl_abs=Decimal("0"),
        pnl_pct=Decimal("0"),
        aggregate_greeks={"delta": Decimal("-0.30"), "gamma": Decimal("0"), "vega": Decimal("0"), "theta": Decimal("0")},
        confidence=1.0,
    )
    import pytest
    with pytest.raises(Exception):
        structure.kind = StructureKind.LONG_CALL  # frozen
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_option_structure_classifier.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.backend.options_intel.structure'`.

- [ ] **Step 3: Create the structure module**

Create `src/backend/options_intel/structure.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Literal, Optional


class StructureKind(str, Enum):
    CREDIT_PUT_SPREAD = "credit_put_spread"
    CREDIT_CALL_SPREAD = "credit_call_spread"
    DEBIT_PUT_SPREAD = "debit_put_spread"
    DEBIT_CALL_SPREAD = "debit_call_spread"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    LONG_STRADDLE = "long_straddle"
    LONG_STRANGLE = "long_strangle"
    CALENDAR_PUT = "calendar_put"
    CALENDAR_CALL = "calendar_call"
    DIAGONAL_PUT = "diagonal_put"
    DIAGONAL_CALL = "diagonal_call"
    UNKNOWN = "unknown"


class BreachState(str, Enum):
    NOMINAL = "nominal"
    WARNING = "warning"
    BREACHED = "breached"


@dataclass(frozen=True)
class OptionLeg:
    instrument_name: str
    kind: Literal["call", "put"]
    strike: Decimal
    side: Literal["long", "short"]
    contracts: Decimal
    days_to_expiry: int
    mark_price: Decimal
    delta: Optional[Decimal]
    gamma: Optional[Decimal]
    vega: Optional[Decimal]
    theta: Optional[Decimal]


@dataclass(frozen=True)
class OptionStructure:
    structure_id: str
    kind: StructureKind
    underlying: str
    legs: tuple[OptionLeg, ...]
    tenor_days_min: int
    tenor_days_max: int
    net_premium: Decimal
    is_credit: bool
    max_loss: Optional[Decimal]
    max_profit: Optional[Decimal]
    breakevens: tuple[Decimal, ...]
    short_leg_delta: Optional[Decimal]
    breach_state: BreachState
    pnl_abs: Decimal
    pnl_pct: Decimal
    aggregate_greeks: dict[str, Decimal]
    confidence: float
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_option_structure_classifier.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/backend/options_intel/structure.py tests/test_option_structure_classifier.py
git commit -m "feat(options): add OptionStructure/StructureKind/BreachState/OptionLeg types"
```

---

## Task 3: structure_id hash with stability under leg reordering

`structure_id` is the keystone for DB identity. It must be stable across cycles regardless of leg ordering and case differences in instrument names.

**Files:**
- Modify: `src/backend/options_intel/structure.py`
- Test: `tests/test_option_structure_classifier.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_option_structure_classifier.py`:

```python
from src.backend.options_intel.structure import compute_structure_id


def _put_leg(strike: int, side: str, contracts: str = "0.1") -> OptionLeg:
    return OptionLeg(
        instrument_name=f"BTC-27JUN26-{strike}-P",
        kind="put",
        strike=Decimal(strike),
        side=side,
        contracts=Decimal(contracts),
        days_to_expiry=14,
        mark_price=Decimal("100"),
        delta=Decimal("-0.20"),
        gamma=None,
        vega=None,
        theta=None,
    )


def test_structure_id_stable_across_leg_order():
    leg_a = _put_leg(100000, "long")
    leg_b = _put_leg(95000, "short")
    id1 = compute_structure_id([leg_a, leg_b])
    id2 = compute_structure_id([leg_b, leg_a])
    assert id1 == id2
    assert len(id1) == 40  # sha1 hex


def test_structure_id_differs_when_sides_swap():
    leg_a = _put_leg(100000, "long")
    leg_b = _put_leg(95000, "short")
    leg_a_short = _put_leg(100000, "short")
    leg_b_long = _put_leg(95000, "long")
    assert compute_structure_id([leg_a, leg_b]) != compute_structure_id([leg_a_short, leg_b_long])


def test_structure_id_case_insensitive_on_instrument_name():
    leg_lower = OptionLeg(
        instrument_name="btc-27jun26-100000-p",
        kind="put", strike=Decimal("100000"), side="long",
        contracts=Decimal("0.1"), days_to_expiry=14, mark_price=Decimal("100"),
        delta=None, gamma=None, vega=None, theta=None,
    )
    leg_upper = OptionLeg(
        instrument_name="BTC-27JUN26-100000-P",
        kind="put", strike=Decimal("100000"), side="long",
        contracts=Decimal("0.1"), days_to_expiry=14, mark_price=Decimal("100"),
        delta=None, gamma=None, vega=None, theta=None,
    )
    assert compute_structure_id([leg_lower]) == compute_structure_id([leg_upper])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_option_structure_classifier.py::test_structure_id_stable_across_leg_order -v
```

Expected: FAIL — `ImportError: cannot import name 'compute_structure_id'`.

- [ ] **Step 3: Implement compute_structure_id**

Append to `src/backend/options_intel/structure.py`:

```python
import hashlib
from typing import Iterable


def compute_structure_id(legs: Iterable[OptionLeg]) -> str:
    """Stable sha1 hash of sorted (instrument_name_upper, side) tuples.

    Stable across leg ordering and case-insensitive on instrument_name so the
    same set of legs yields the same id across cycles regardless of how
    Thalex returns them.
    """
    tuples = sorted(
        (leg.instrument_name.upper(), leg.side) for leg in legs
    )
    payload = "|".join(f"{name}:{side}" for name, side in tuples)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_option_structure_classifier.py -v
```

Expected: all 7 tests in the file pass.

- [ ] **Step 5: Commit**

```bash
git add src/backend/options_intel/structure.py tests/test_option_structure_classifier.py
git commit -m "feat(options): add stable structure_id hash"
```

---

## Task 4: Classifier — single-leg and vertical spreads

Implement the first slice of `classify()`: naked longs and 2-leg same-kind same-tenor spreads (credit and debit).

**Files:**
- Modify: `src/backend/options_intel/structure.py`
- Test: `tests/test_option_structure_classifier.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_option_structure_classifier.py`:

```python
from src.backend.options_intel.structure import classify


def _call_leg(strike: int, side: str, mark: str = "200", dte: int = 14, delta: str = "0.30") -> OptionLeg:
    return OptionLeg(
        instrument_name=f"BTC-27JUN26-{strike}-C",
        kind="call",
        strike=Decimal(strike),
        side=side,
        contracts=Decimal("0.1"),
        days_to_expiry=dte,
        mark_price=Decimal(mark),
        delta=Decimal(delta),
        gamma=None, vega=None, theta=None,
    )


def _put_leg_full(strike: int, side: str, mark: str = "200", dte: int = 14, delta: str = "-0.30") -> OptionLeg:
    return OptionLeg(
        instrument_name=f"BTC-27JUN26-{strike}-P",
        kind="put",
        strike=Decimal(strike),
        side=side,
        contracts=Decimal("0.1"),
        days_to_expiry=dte,
        mark_price=Decimal(mark),
        delta=Decimal(delta),
        gamma=None, vega=None, theta=None,
    )


def test_classify_long_call():
    legs = [_call_leg(100000, "long")]
    s = classify(legs)
    assert s.kind == StructureKind.LONG_CALL
    assert s.is_credit is False
    assert s.confidence == 1.0


def test_classify_long_put():
    legs = [_put_leg_full(100000, "long")]
    s = classify(legs)
    assert s.kind == StructureKind.LONG_PUT
    assert s.is_credit is False


def test_classify_credit_put_spread():
    # Sell 100k put @ 300, buy 90k put @ 100 → net credit +200
    legs = [
        _put_leg_full(100000, "short", mark="300", delta="-0.30"),
        _put_leg_full(90000, "long", mark="100", delta="-0.10"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.CREDIT_PUT_SPREAD
    assert s.is_credit is True
    assert s.confidence == 1.0


def test_classify_debit_put_spread():
    # Buy 100k put @ 300, sell 90k put @ 100 → net debit -200
    legs = [
        _put_leg_full(100000, "long", mark="300", delta="-0.30"),
        _put_leg_full(90000, "short", mark="100", delta="-0.10"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.DEBIT_PUT_SPREAD
    assert s.is_credit is False


def test_classify_credit_call_spread():
    legs = [
        _call_leg(100000, "short", mark="300", delta="0.30"),
        _call_leg(110000, "long", mark="100", delta="0.10"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.CREDIT_CALL_SPREAD
    assert s.is_credit is True


def test_classify_debit_call_spread():
    legs = [
        _call_leg(100000, "long", mark="300", delta="0.30"),
        _call_leg(110000, "short", mark="100", delta="0.10"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.DEBIT_CALL_SPREAD
    assert s.is_credit is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_option_structure_classifier.py::test_classify_long_call -v
```

Expected: FAIL — `ImportError: cannot import name 'classify'`.

- [ ] **Step 3: Implement classifier core (single-leg + vertical only)**

Append to `src/backend/options_intel/structure.py`:

```python
from typing import Sequence


def _net_premium(legs: Sequence[OptionLeg]) -> Decimal:
    """Net premium: positive credit (received), negative debit (paid).

    side_sign: short = +1 (we received), long = -1 (we paid).
    """
    total = Decimal("0")
    for leg in legs:
        side_sign = Decimal("1") if leg.side == "short" else Decimal("-1")
        total += side_sign * leg.contracts * leg.mark_price
    return total


def _tenor_minmax(legs: Sequence[OptionLeg]) -> tuple[int, int]:
    dtes = [leg.days_to_expiry for leg in legs]
    return min(dtes), max(dtes)


def _aggregate_greeks(legs: Sequence[OptionLeg]) -> dict[str, Decimal]:
    out = {"delta": Decimal("0"), "gamma": Decimal("0"), "vega": Decimal("0"), "theta": Decimal("0")}
    for leg in legs:
        side_sign = Decimal("1") if leg.side == "long" else Decimal("-1")
        for key in ("delta", "gamma", "vega", "theta"):
            value = getattr(leg, key)
            if value is None:
                continue
            out[key] += side_sign * leg.contracts * value
    return out


def classify(legs: Sequence[OptionLeg]) -> OptionStructure:
    """Classify a leg list into an OptionStructure.

    Templates are matched in priority order. Naked longs and 2-leg
    same-kind same-tenor spreads (credit + debit) are covered here.
    More templates added in subsequent tasks.
    """
    legs_tuple = tuple(legs)
    underlying = _underlying_from_legs(legs_tuple)
    structure_id = compute_structure_id(legs_tuple)
    tenor_min, tenor_max = _tenor_minmax(legs_tuple)
    net_premium = _net_premium(legs_tuple)
    aggregate_greeks = _aggregate_greeks(legs_tuple)

    kind, confidence = _match_template(legs_tuple)

    base = dict(
        structure_id=structure_id,
        underlying=underlying,
        legs=legs_tuple,
        tenor_days_min=tenor_min,
        tenor_days_max=tenor_max,
        net_premium=net_premium,
        is_credit=net_premium > 0,
        aggregate_greeks=aggregate_greeks,
        confidence=confidence,
    )
    # Derived properties (max_loss, max_profit, breakevens, breach_state, pnl)
    # are stubbed here and filled in later tasks. Phase 1 task 8 wires them
    # in; until then we emit safe defaults.
    return OptionStructure(
        kind=kind,
        max_loss=None,
        max_profit=None,
        breakevens=(),
        short_leg_delta=None,
        breach_state=BreachState.NOMINAL,
        pnl_abs=Decimal("0"),
        pnl_pct=Decimal("0"),
        **base,
    )


def _underlying_from_legs(legs: Sequence[OptionLeg]) -> str:
    """Extract underlying from the first leg's instrument_name (e.g., BTC-...)."""
    if not legs:
        return ""
    name = legs[0].instrument_name.upper()
    return name.split("-", 1)[0] if "-" in name else name


def _match_template(legs: Sequence[OptionLeg]) -> tuple[StructureKind, float]:
    """Match leg topology to a StructureKind. Returns (kind, confidence)."""
    n = len(legs)
    if n == 0:
        return StructureKind.UNKNOWN, 0.0

    if n == 1:
        leg = legs[0]
        if leg.side != "long":
            return StructureKind.UNKNOWN, 0.0
        if leg.kind == "call":
            return StructureKind.LONG_CALL, 1.0
        if leg.kind == "put":
            return StructureKind.LONG_PUT, 1.0
        return StructureKind.UNKNOWN, 0.0

    if n == 2:
        kinds = {leg.kind for leg in legs}
        sides = {leg.side for leg in legs}
        tenors = {leg.days_to_expiry for leg in legs}

        if len(kinds) == 1 and len(sides) == 2 and len(tenors) == 1:
            kind = next(iter(kinds))
            net_prem = _net_premium(legs)
            if kind == "put":
                return (
                    (StructureKind.CREDIT_PUT_SPREAD, 1.0)
                    if net_prem > 0
                    else (StructureKind.DEBIT_PUT_SPREAD, 1.0)
                )
            if kind == "call":
                return (
                    (StructureKind.CREDIT_CALL_SPREAD, 1.0)
                    if net_prem > 0
                    else (StructureKind.DEBIT_CALL_SPREAD, 1.0)
                )

    return StructureKind.UNKNOWN, 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_option_structure_classifier.py -v
```

Expected: all classifier tests pass (the 6 new ones plus the earlier 7 in this file).

- [ ] **Step 5: Commit**

```bash
git add src/backend/options_intel/structure.py tests/test_option_structure_classifier.py
git commit -m "feat(options): classify single-leg longs and vertical spreads"
```

---

## Task 5: Classifier — iron condor and iron butterfly

Extend `_match_template` for 4-leg structures: 2 calls + 2 puts, same tenor.

**Files:**
- Modify: `src/backend/options_intel/structure.py`
- Test: `tests/test_option_structure_classifier.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_option_structure_classifier.py`:

```python
def test_classify_iron_condor():
    # Short 95k put + long 90k put + short 105k call + long 110k call
    legs = [
        _put_leg_full(95000, "short", mark="300"),
        _put_leg_full(90000, "long", mark="100"),
        _call_leg(105000, "short", mark="300"),
        _call_leg(110000, "long", mark="100"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.IRON_CONDOR
    assert s.is_credit is True
    assert s.confidence == 1.0


def test_classify_iron_butterfly():
    # Short 100k put + short 100k call (body) + long 90k put + long 110k call (wings)
    legs = [
        _put_leg_full(100000, "short", mark="600"),
        _call_leg(100000, "short", mark="600"),
        _put_leg_full(90000, "long", mark="100"),
        _call_leg(110000, "long", mark="100"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.IRON_BUTTERFLY
    assert s.is_credit is True
    assert s.confidence == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_option_structure_classifier.py::test_classify_iron_condor tests/test_option_structure_classifier.py::test_classify_iron_butterfly -v
```

Expected: FAIL — classifier returns `UNKNOWN` for both.

- [ ] **Step 3: Extend _match_template for 4-leg cases**

In `src/backend/options_intel/structure.py`, locate `_match_template`. Insert the following block **between** the `n == 2` branch and the final `return StructureKind.UNKNOWN, 0.0`:

```python
    if n == 4:
        calls = [leg for leg in legs if leg.kind == "call"]
        puts = [leg for leg in legs if leg.kind == "put"]
        tenors = {leg.days_to_expiry for leg in legs}
        if len(calls) == 2 and len(puts) == 2 and len(tenors) == 1:
            # Iron condor: each kind has one short and one long, distinct strikes
            calls_by_side = {leg.side: leg for leg in calls}
            puts_by_side = {leg.side: leg for leg in puts}
            if (
                set(calls_by_side.keys()) == {"long", "short"}
                and set(puts_by_side.keys()) == {"long", "short"}
            ):
                if calls_by_side["short"].strike == puts_by_side["short"].strike:
                    return StructureKind.IRON_BUTTERFLY, 1.0
                return StructureKind.IRON_CONDOR, 1.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_option_structure_classifier.py -v
```

Expected: all classifier tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/backend/options_intel/structure.py tests/test_option_structure_classifier.py
git commit -m "feat(options): classify iron condor and iron butterfly"
```

---

## Task 6: Classifier — calendars, diagonals, straddle, strangle

Extend `_match_template` for 2-leg different-tenor (calendar/diagonal) and 2-leg different-kind same-tenor (straddle/strangle).

**Files:**
- Modify: `src/backend/options_intel/structure.py`
- Test: `tests/test_option_structure_classifier.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_option_structure_classifier.py`:

```python
def test_classify_calendar_put():
    # Same strike, different tenors, opposite sides — short near, long far
    near = _put_leg_full(100000, "short", mark="200", dte=7)
    far = _put_leg_full(100000, "long", mark="500", dte=28)
    s = classify([near, far])
    assert s.kind == StructureKind.CALENDAR_PUT


def test_classify_calendar_call():
    near = _call_leg(100000, "short", mark="200", dte=7)
    far = _call_leg(100000, "long", mark="500", dte=28)
    s = classify([near, far])
    assert s.kind == StructureKind.CALENDAR_CALL


def test_classify_diagonal_put():
    near = _put_leg_full(95000, "short", mark="150", dte=7)
    far = _put_leg_full(100000, "long", mark="500", dte=28)
    s = classify([near, far])
    assert s.kind == StructureKind.DIAGONAL_PUT


def test_classify_diagonal_call():
    near = _call_leg(105000, "short", mark="150", dte=7)
    far = _call_leg(100000, "long", mark="500", dte=28)
    s = classify([near, far])
    assert s.kind == StructureKind.DIAGONAL_CALL


def test_classify_long_straddle():
    # Long call + long put, same strike, same tenor
    legs = [
        _call_leg(100000, "long", mark="500"),
        _put_leg_full(100000, "long", mark="500"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.LONG_STRADDLE


def test_classify_long_strangle():
    # Long call + long put, different strikes (wings of strangle), same tenor
    legs = [
        _call_leg(105000, "long", mark="200"),
        _put_leg_full(95000, "long", mark="200"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.LONG_STRANGLE
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_option_structure_classifier.py -v -k calendar or diagonal or straddle or strangle
```

Expected: all 6 new tests FAIL — classifier returns `UNKNOWN`.

- [ ] **Step 3: Extend _match_template for 2-leg cases beyond verticals**

In `src/backend/options_intel/structure.py`, locate the `if n == 2:` block in `_match_template`. After the existing same-kind same-tenor (vertical spread) check, add the following cases before falling through to `UNKNOWN`:

```python
        # Calendar / diagonal: same kind, different tenors
        if len(kinds) == 1 and len(tenors) == 2:
            kind = next(iter(kinds))
            strikes = {leg.strike for leg in legs}
            same_strike = len(strikes) == 1
            if kind == "put":
                return (
                    (StructureKind.CALENDAR_PUT, 1.0)
                    if same_strike
                    else (StructureKind.DIAGONAL_PUT, 1.0)
                )
            if kind == "call":
                return (
                    (StructureKind.CALENDAR_CALL, 1.0)
                    if same_strike
                    else (StructureKind.DIAGONAL_CALL, 1.0)
                )

        # Long straddle / strangle: different kinds, same tenor, both long
        if len(kinds) == 2 and sides == {"long"} and len(tenors) == 1:
            strikes = {leg.strike for leg in legs}
            if len(strikes) == 1:
                return StructureKind.LONG_STRADDLE, 1.0
            return StructureKind.LONG_STRANGLE, 1.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_option_structure_classifier.py -v
```

Expected: all tests in the file pass.

- [ ] **Step 5: Commit**

```bash
git add src/backend/options_intel/structure.py tests/test_option_structure_classifier.py
git commit -m "feat(options): classify calendars, diagonals, straddle, strangle"
```

---

## Task 7: Classifier — UNKNOWN fallback and confidence scoring

Cover the anomaly cases — mismatched contract counts on a vertical, mixed underlyings, ambiguous topology (returns reduced confidence rather than UNKNOWN where a best-fit exists).

**Files:**
- Modify: `src/backend/options_intel/structure.py`
- Test: `tests/test_option_structure_classifier.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_option_structure_classifier.py`:

```python
def test_classify_unknown_for_mismatched_contracts_vertical():
    # Same kind, same tenor, opposite sides, BUT contracts differ — not a clean vertical
    leg_a = _put_leg_full(100000, "short")
    leg_b = OptionLeg(
        instrument_name="BTC-27JUN26-90000-P",
        kind="put",
        strike=Decimal("90000"),
        side="long",
        contracts=Decimal("0.2"),  # mismatched
        days_to_expiry=14,
        mark_price=Decimal("100"),
        delta=Decimal("-0.10"),
        gamma=None, vega=None, theta=None,
    )
    s = classify([leg_a, leg_b])
    assert s.kind == StructureKind.UNKNOWN
    assert s.confidence == 0.0


def test_classify_unknown_for_mixed_underlyings():
    btc_leg = _put_leg_full(100000, "long")
    eth_leg = OptionLeg(
        instrument_name="ETH-27JUN26-3000-P",
        kind="put",
        strike=Decimal("3000"),
        side="short",
        contracts=Decimal("0.1"),
        days_to_expiry=14,
        mark_price=Decimal("50"),
        delta=Decimal("-0.20"),
        gamma=None, vega=None, theta=None,
    )
    s = classify([btc_leg, eth_leg])
    assert s.kind == StructureKind.UNKNOWN
    assert s.confidence == 0.0


def test_classify_unknown_preserves_legs():
    legs = [_put_leg_full(100000, "short"), _call_leg(105000, "long")]  # mixed kinds, both sides
    s = classify(legs)
    # Two legs, mixed kinds, mixed sides, same tenor → not straddle (sides differ), not vertical (kinds differ)
    assert s.kind == StructureKind.UNKNOWN
    assert len(s.legs) == 2  # legs preserved for LLM visibility


def test_classify_naked_short_is_unknown():
    # Naked shorts are forbidden by the strategy enum but might appear from manual desk activity
    legs = [_put_leg_full(100000, "short")]
    s = classify(legs)
    assert s.kind == StructureKind.UNKNOWN
    assert s.confidence == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_option_structure_classifier.py -v -k unknown
```

Expected: `test_classify_unknown_for_mismatched_contracts_vertical` FAILS — the current vertical-spread branch ignores contract equality. The others should already pass given Task 4–6 logic, but verify.

- [ ] **Step 3: Add contract-equality + underlying-equality guards**

In `src/backend/options_intel/structure.py`, modify the `n == 2` same-kind same-tenor block. Before deciding credit/debit, add a contract-equality check that falls through to `UNKNOWN` when contracts don't match. Replace:

```python
        if len(kinds) == 1 and len(sides) == 2 and len(tenors) == 1:
            kind = next(iter(kinds))
```

with:

```python
        if (
            len(kinds) == 1
            and len(sides) == 2
            and len(tenors) == 1
            and legs[0].contracts == legs[1].contracts
        ):
            kind = next(iter(kinds))
```

Then at the top of `_match_template`, before the `n == 0` check, add a mixed-underlyings guard:

```python
    underlyings = {_underlying_from_legs([leg]) for leg in legs}
    if len(underlyings) > 1:
        return StructureKind.UNKNOWN, 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_option_structure_classifier.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/backend/options_intel/structure.py tests/test_option_structure_classifier.py
git commit -m "feat(options): UNKNOWN fallback for mismatched contracts, mixed underlyings, naked shorts"
```

---

## Task 8: Classifier — derived properties (premium, breakevens, P&L, breach state)

Fill in the derived fields (`max_loss`, `max_profit`, `breakevens`, `short_leg_delta`, `breach_state`, `pnl_abs`, `pnl_pct`). P&L baseline accepts an optional `entry_net_premium` parameter from the DB; if absent, the current `net_premium` becomes the baseline and P&L is zero.

**Files:**
- Modify: `src/backend/options_intel/structure.py`
- Test: `tests/test_option_structure_classifier.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_option_structure_classifier.py`:

```python
def test_credit_put_spread_max_loss_and_breakeven():
    # Short 100k put @ 300, long 90k put @ 100, contracts 0.1
    # Width = 10,000; credit = 200; max_loss = (10,000 - 200) * 0.1 = 980; max_profit = 200 * 0.1 = 20
    legs = [
        _put_leg_full(100000, "short", mark="300", delta="-0.30"),
        _put_leg_full(90000, "long", mark="100", delta="-0.10"),
    ]
    s = classify(legs)
    assert s.max_loss == Decimal("980")
    assert s.max_profit == Decimal("20")
    # Breakeven for short put spread: short_strike - net_credit_per_contract
    # net_credit_per_contract = 200; breakeven = 100,000 - 200 = 99,800
    assert s.breakevens == (Decimal("99800"),)
    assert s.short_leg_delta == Decimal("-0.30")
    assert s.breach_state == BreachState.WARNING  # short |delta|=0.30 falls in 0.25–0.40 band


def test_long_call_max_profit_unbounded_max_loss_premium():
    legs = [_call_leg(100000, "long", mark="500", delta="0.40")]
    s = classify(legs)
    assert s.max_loss == Decimal("50")  # 500 * 0.1 contracts
    assert s.max_profit is None  # unbounded for naked long
    assert s.breakevens == (Decimal("100500"),)  # strike + premium_per_contract


def test_pnl_baseline_uses_current_when_no_prior():
    legs = [
        _put_leg_full(100000, "short", mark="300", delta="-0.30"),
        _put_leg_full(90000, "long", mark="100", delta="-0.10"),
    ]
    s = classify(legs)
    assert s.pnl_abs == Decimal("0")
    assert s.pnl_pct == Decimal("0")


def test_pnl_with_explicit_entry_baseline():
    legs = [
        _put_leg_full(100000, "short", mark="200", delta="-0.20"),
        _put_leg_full(90000, "long", mark="50", delta="-0.05"),
    ]
    # current net_premium = (200 - 50) * 0.1 = 15.0  (credit shrunk → favorable for credit spread)
    # entry was 20 (from prior test scenario) → pnl_abs = 20 - 15 = 5 (favorable)
    s = classify(legs, entry_net_premium=Decimal("20"))
    assert s.pnl_abs == Decimal("5")
    assert s.pnl_pct == Decimal("0.25")  # 5 / 20


def test_breach_state_breached_when_dte_lt_2():
    legs = [
        _put_leg_full(100000, "short", mark="300", delta="-0.20", dte=1),
        _put_leg_full(90000, "long", mark="100", delta="-0.05", dte=1),
    ]
    s = classify(legs)
    assert s.breach_state == BreachState.BREACHED


def test_breach_state_nominal_when_delta_low_dte_high():
    legs = [
        _put_leg_full(100000, "short", mark="300", delta="-0.15", dte=20),
        _put_leg_full(90000, "long", mark="100", delta="-0.05", dte=20),
    ]
    s = classify(legs)
    assert s.breach_state == BreachState.NOMINAL
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_option_structure_classifier.py -v -k "max_loss or max_profit or breakeven or pnl or breach_state"
```

Expected: FAILs — current classifier emits `max_loss=None`, `breakevens=()`, `breach_state=NOMINAL` always; `classify` doesn't accept `entry_net_premium`.

- [ ] **Step 3: Add derivation helpers and rewire classify()**

In `src/backend/options_intel/structure.py`, add the following helpers and rewrite `classify` to use them:

```python
def _short_leg(legs: Sequence[OptionLeg]) -> Optional[OptionLeg]:
    shorts = [leg for leg in legs if leg.side == "short"]
    if not shorts:
        return None
    # For ICs/IBs, prefer the leg with the most in-the-money short delta (largest |delta|)
    shorts_with_delta = [leg for leg in shorts if leg.delta is not None]
    if shorts_with_delta:
        return max(shorts_with_delta, key=lambda leg: abs(leg.delta))
    return shorts[0]


def _short_leg_delta(legs: Sequence[OptionLeg]) -> Optional[Decimal]:
    short = _short_leg(legs)
    if short is None or short.delta is None:
        return None
    return short.delta


def _max_loss_max_profit(kind: StructureKind, legs: Sequence[OptionLeg], net_premium: Decimal) -> tuple[Optional[Decimal], Optional[Decimal]]:
    """Compute (max_loss_abs, max_profit_abs) for the structure. Returns (None, None) when undefined."""
    contracts = legs[0].contracts if legs else Decimal("0")

    if kind in (StructureKind.CREDIT_PUT_SPREAD, StructureKind.CREDIT_CALL_SPREAD):
        # Width × contracts − credit_received
        strikes = sorted({leg.strike for leg in legs})
        width = strikes[1] - strikes[0]
        return (width * contracts - net_premium, net_premium)

    if kind in (StructureKind.DEBIT_PUT_SPREAD, StructureKind.DEBIT_CALL_SPREAD):
        # Debit paid is max loss; width × contracts − debit_paid is max profit
        strikes = sorted({leg.strike for leg in legs})
        width = strikes[1] - strikes[0]
        debit_paid = -net_premium  # net_premium is negative for debit
        return (debit_paid, width * contracts - debit_paid)

    if kind in (StructureKind.IRON_CONDOR, StructureKind.IRON_BUTTERFLY):
        # Max loss = max wing width × contracts − total credit; max profit = total credit
        calls = sorted([leg.strike for leg in legs if leg.kind == "call"])
        puts = sorted([leg.strike for leg in legs if leg.kind == "put"])
        call_width = calls[1] - calls[0] if len(calls) == 2 else Decimal("0")
        put_width = puts[1] - puts[0] if len(puts) == 2 else Decimal("0")
        max_width = max(call_width, put_width)
        return (max_width * contracts - net_premium, net_premium)

    if kind in (StructureKind.LONG_CALL, StructureKind.LONG_PUT,
                StructureKind.LONG_STRADDLE, StructureKind.LONG_STRANGLE):
        # Long-only: max loss = premium paid (−net_premium since debit); profit unbounded
        return (-net_premium, None)

    # Calendars/diagonals/UNKNOWN: undefined under simple closed-form
    return (None, None)


def _breakevens(kind: StructureKind, legs: Sequence[OptionLeg], net_premium: Decimal) -> tuple[Decimal, ...]:
    contracts = legs[0].contracts if legs else Decimal("0")
    if contracts == 0:
        return ()
    prem_per_contract = net_premium / contracts

    if kind == StructureKind.CREDIT_PUT_SPREAD:
        short_strike = max(leg.strike for leg in legs if leg.side == "short")
        return (short_strike - prem_per_contract,)
    if kind == StructureKind.CREDIT_CALL_SPREAD:
        short_strike = min(leg.strike for leg in legs if leg.side == "short")
        return (short_strike + prem_per_contract,)
    if kind == StructureKind.DEBIT_PUT_SPREAD:
        long_strike = max(leg.strike for leg in legs if leg.side == "long")
        return (long_strike - abs(prem_per_contract),)
    if kind == StructureKind.DEBIT_CALL_SPREAD:
        long_strike = min(leg.strike for leg in legs if leg.side == "long")
        return (long_strike + abs(prem_per_contract),)
    if kind == StructureKind.LONG_CALL:
        return (legs[0].strike + abs(prem_per_contract),)
    if kind == StructureKind.LONG_PUT:
        return (legs[0].strike - abs(prem_per_contract),)
    if kind in (StructureKind.IRON_CONDOR, StructureKind.IRON_BUTTERFLY):
        short_put = max((leg.strike for leg in legs if leg.kind == "put" and leg.side == "short"), default=None)
        short_call = min((leg.strike for leg in legs if leg.kind == "call" and leg.side == "short"), default=None)
        if short_put is None or short_call is None:
            return ()
        return (short_put - prem_per_contract, short_call + prem_per_contract)
    return ()


def _delta_metric_for_breach(kind: StructureKind, legs: Sequence[OptionLeg]) -> Optional[Decimal]:
    if kind in (StructureKind.LONG_CALL, StructureKind.LONG_PUT,
                StructureKind.LONG_STRADDLE, StructureKind.LONG_STRANGLE):
        aggregate = _aggregate_greeks(legs).get("delta", Decimal("0"))
        return abs(aggregate)
    if kind in (StructureKind.CALENDAR_PUT, StructureKind.CALENDAR_CALL,
                StructureKind.DIAGONAL_PUT, StructureKind.DIAGONAL_CALL):
        near_legs = [leg for leg in legs if leg.days_to_expiry == min(l.days_to_expiry for l in legs)]
        if near_legs and near_legs[0].delta is not None:
            return abs(near_legs[0].delta)
        return None
    short_d = _short_leg_delta(legs)
    return abs(short_d) if short_d is not None else None


def _breach_state(kind: StructureKind, legs: Sequence[OptionLeg]) -> BreachState:
    if not legs:
        return BreachState.NOMINAL
    dte_min = min(leg.days_to_expiry for leg in legs)
    delta_metric = _delta_metric_for_breach(kind, legs)

    breached_delta = delta_metric is not None and delta_metric >= Decimal("0.40")
    breached_dte = dte_min < 2
    if breached_delta or breached_dte:
        return BreachState.BREACHED

    warning_delta = delta_metric is not None and delta_metric >= Decimal("0.25")
    warning_dte = dte_min < 5
    if warning_delta or warning_dte:
        return BreachState.WARNING

    return BreachState.NOMINAL


def _pnl(net_premium: Decimal, entry_net_premium: Optional[Decimal], is_credit: bool) -> tuple[Decimal, Decimal]:
    """Return (pnl_abs, pnl_pct). For credit structures, favorable means premium shrinks
    (we'd close cheaper). For debit, favorable means premium grows.
    """
    if entry_net_premium is None or entry_net_premium == 0:
        return Decimal("0"), Decimal("0")
    if is_credit:
        pnl_abs = entry_net_premium - net_premium
    else:
        pnl_abs = net_premium - entry_net_premium
    pnl_pct = pnl_abs / abs(entry_net_premium)
    return pnl_abs, pnl_pct
```

Now rewrite `classify` to accept `entry_net_premium` and use the helpers:

```python
def classify(
    legs: Sequence[OptionLeg],
    *,
    entry_net_premium: Optional[Decimal] = None,
) -> OptionStructure:
    legs_tuple = tuple(legs)
    underlying = _underlying_from_legs(legs_tuple)
    structure_id = compute_structure_id(legs_tuple)
    tenor_min, tenor_max = _tenor_minmax(legs_tuple) if legs_tuple else (0, 0)
    net_premium = _net_premium(legs_tuple)
    aggregate_greeks = _aggregate_greeks(legs_tuple)

    kind, confidence = _match_template(legs_tuple)
    is_credit = net_premium > 0

    max_loss, max_profit = _max_loss_max_profit(kind, legs_tuple, net_premium)
    breakevens = _breakevens(kind, legs_tuple, net_premium)
    short_leg_delta = _short_leg_delta(legs_tuple)
    breach_state = _breach_state(kind, legs_tuple)
    pnl_abs, pnl_pct = _pnl(net_premium, entry_net_premium, is_credit)

    return OptionStructure(
        structure_id=structure_id,
        kind=kind,
        underlying=underlying,
        legs=legs_tuple,
        tenor_days_min=tenor_min,
        tenor_days_max=tenor_max,
        net_premium=net_premium,
        is_credit=is_credit,
        max_loss=max_loss,
        max_profit=max_profit,
        breakevens=breakevens,
        short_leg_delta=short_leg_delta,
        breach_state=breach_state,
        pnl_abs=pnl_abs,
        pnl_pct=pnl_pct,
        aggregate_greeks=aggregate_greeks,
        confidence=confidence,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_option_structure_classifier.py -v
```

Expected: all tests in the file pass.

- [ ] **Step 5: Commit**

```bash
git add src/backend/options_intel/structure.py tests/test_option_structure_classifier.py
git commit -m "feat(options): structure derived properties — premium, breakevens, P&L, breach state"
```

---

## Task 9: OptionStructureSnapshot ORM model + db_manager CRUD

Persist structure identity and lifecycle. Update-in-place on `structure_id` so the table reflects current state without bloating.

**Files:**
- Modify: `src/database/models.py`
- Modify: `src/database/db_manager.py`
- Test: `tests/test_option_structure_persistence.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_option_structure_persistence.py`:

```python
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.database.db_manager import DatabaseManager
from src.database.models import OptionStructureSnapshot


@pytest.fixture
def db():
    db = DatabaseManager(db_url="sqlite:///:memory:")
    return db


def test_upsert_creates_new_structure(db):
    db.upsert_structure_snapshot(
        structure_id="abc123",
        underlying="BTC",
        kind="credit_put_spread",
        legs_json=[{"instrument_name": "BTC-27JUN26-100000-P", "side": "short"}],
        entry_net_premium=Decimal("20"),
        last_pnl_abs=Decimal("0"),
        last_pnl_pct=Decimal("0"),
        last_breach_state="nominal",
    )
    with db.session_scope() as session:
        row = session.query(OptionStructureSnapshot).filter_by(structure_id="abc123").one()
        assert row.underlying == "BTC"
        assert row.kind == "credit_put_spread"
        assert row.closed_at is None
        assert row.opened_at is not None
        assert row.last_seen_at is not None


def test_upsert_existing_structure_updates_fields_but_not_opened_at(db):
    db.upsert_structure_snapshot(
        structure_id="abc123",
        underlying="BTC",
        kind="credit_put_spread",
        legs_json=[],
        entry_net_premium=Decimal("20"),
        last_pnl_abs=Decimal("0"),
        last_pnl_pct=Decimal("0"),
        last_breach_state="nominal",
    )
    with db.session_scope() as session:
        first_opened = session.query(OptionStructureSnapshot).filter_by(structure_id="abc123").one().opened_at

    # Update with new pnl/breach values
    db.upsert_structure_snapshot(
        structure_id="abc123",
        underlying="BTC",
        kind="credit_put_spread",
        legs_json=[],
        entry_net_premium=Decimal("20"),
        last_pnl_abs=Decimal("5"),
        last_pnl_pct=Decimal("0.25"),
        last_breach_state="warning",
    )
    with db.session_scope() as session:
        row = session.query(OptionStructureSnapshot).filter_by(structure_id="abc123").one()
        assert row.opened_at == first_opened
        assert row.last_pnl_abs == 5.0  # Float column in SQLite
        assert row.last_breach_state == "warning"


def test_mark_structure_closed_sets_closed_at(db):
    db.upsert_structure_snapshot(
        structure_id="abc123",
        underlying="BTC",
        kind="credit_put_spread",
        legs_json=[],
        entry_net_premium=Decimal("20"),
        last_pnl_abs=Decimal("0"),
        last_pnl_pct=Decimal("0"),
        last_breach_state="nominal",
    )
    db.mark_structure_closed("abc123")
    with db.session_scope() as session:
        row = session.query(OptionStructureSnapshot).filter_by(structure_id="abc123").one()
        assert row.closed_at is not None


def test_get_open_structures_excludes_closed(db):
    db.upsert_structure_snapshot(
        structure_id="open_1", underlying="BTC", kind="credit_put_spread",
        legs_json=[], entry_net_premium=Decimal("10"),
        last_pnl_abs=Decimal("0"), last_pnl_pct=Decimal("0"), last_breach_state="nominal",
    )
    db.upsert_structure_snapshot(
        structure_id="closed_1", underlying="BTC", kind="iron_condor",
        legs_json=[], entry_net_premium=Decimal("30"),
        last_pnl_abs=Decimal("0"), last_pnl_pct=Decimal("0"), last_breach_state="nominal",
    )
    db.mark_structure_closed("closed_1")
    open_ids = {row["structure_id"] for row in db.get_open_structures()}
    assert open_ids == {"open_1"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_option_structure_persistence.py -v
```

Expected: FAIL — `ImportError: cannot import name 'OptionStructureSnapshot'` or `AttributeError: 'DatabaseManager' object has no attribute 'upsert_structure_snapshot'`.

- [ ] **Step 3: Add the ORM model**

Append to `src/database/models.py` (after the last existing model class):

```python
class OptionStructureSnapshot(Base):
    """Persisted identity + lifecycle of an option structure.

    Update-in-place on structure_id. opened_at is set on first insert and
    never overwritten. closed_at is set when the structure leaves the
    portfolio. entry_net_premium is the P&L baseline; never overwritten
    while closed_at is null.
    """
    __tablename__ = "option_structure_snapshots"

    structure_id = Column(String(40), primary_key=True)  # sha1 hex = 40 chars
    underlying = Column(String(20), nullable=False, index=True)
    kind = Column(String(32), nullable=False, index=True)
    legs_json = Column(Text, nullable=False)  # JSON-serialized snapshot of legs at first sight

    opened_at = Column(DateTime, nullable=False, default=func.now())
    last_seen_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    closed_at = Column(DateTime, nullable=True, index=True)

    entry_net_premium = Column(Float, nullable=False)
    last_pnl_abs = Column(Float, nullable=False, default=0.0)
    last_pnl_pct = Column(Float, nullable=False, default=0.0)
    last_breach_state = Column(String(16), nullable=False, default="nominal")

    metadata_json = Column(Text, nullable=True)  # opening decision id, etc.

    def __repr__(self):
        return f"<OptionStructureSnapshot(id={self.structure_id[:8]}, kind={self.kind}, closed={self.closed_at is not None})>"
```

- [ ] **Step 4: Add db_manager CRUD methods**

Append to `src/database/db_manager.py` (inside the `DatabaseManager` class, before `get_db_manager`):

```python
    def upsert_structure_snapshot(
        self,
        *,
        structure_id: str,
        underlying: str,
        kind: str,
        legs_json: list,
        entry_net_premium,
        last_pnl_abs,
        last_pnl_pct,
        last_breach_state: str,
        metadata_json: Optional[dict] = None,
    ) -> None:
        import json as _json
        from src.database.models import OptionStructureSnapshot

        with self.session_scope() as session:
            existing = (
                session.query(OptionStructureSnapshot)
                .filter_by(structure_id=structure_id)
                .one_or_none()
            )
            if existing is None:
                row = OptionStructureSnapshot(
                    structure_id=structure_id,
                    underlying=underlying,
                    kind=kind,
                    legs_json=_json.dumps(legs_json),
                    entry_net_premium=float(entry_net_premium),
                    last_pnl_abs=float(last_pnl_abs),
                    last_pnl_pct=float(last_pnl_pct),
                    last_breach_state=last_breach_state,
                    metadata_json=_json.dumps(metadata_json) if metadata_json else None,
                )
                session.add(row)
            else:
                # opened_at and entry_net_premium never change while structure is open
                existing.kind = kind
                existing.last_pnl_abs = float(last_pnl_abs)
                existing.last_pnl_pct = float(last_pnl_pct)
                existing.last_breach_state = last_breach_state
                if existing.closed_at is not None:
                    existing.closed_at = None  # Reopened — clear the close marker

    def mark_structure_closed(self, structure_id: str) -> None:
        from datetime import datetime
        from src.database.models import OptionStructureSnapshot

        with self.session_scope() as session:
            row = (
                session.query(OptionStructureSnapshot)
                .filter_by(structure_id=structure_id)
                .one_or_none()
            )
            if row is not None and row.closed_at is None:
                row.closed_at = datetime.utcnow()

    def get_open_structures(self) -> List[Dict[str, Any]]:
        from src.database.models import OptionStructureSnapshot

        with self.session_scope() as session:
            rows = (
                session.query(OptionStructureSnapshot)
                .filter(OptionStructureSnapshot.closed_at.is_(None))
                .all()
            )
            return [
                {
                    "structure_id": row.structure_id,
                    "underlying": row.underlying,
                    "kind": row.kind,
                    "opened_at": row.opened_at,
                    "last_seen_at": row.last_seen_at,
                    "entry_net_premium": row.entry_net_premium,
                    "last_pnl_abs": row.last_pnl_abs,
                    "last_pnl_pct": row.last_pnl_pct,
                    "last_breach_state": row.last_breach_state,
                }
                for row in rows
            ]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_option_structure_persistence.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/database/models.py src/database/db_manager.py tests/test_option_structure_persistence.py
git commit -m "feat(db): OptionStructureSnapshot model + upsert/close/get_open CRUD"
```

---

## Task 10: Wire portfolio.py to emit structures

`aggregate_portfolio_greeks` now returns a `structures` list. Callers (snapshot.py) are unaffected because the returned dict gains a new key without changing existing ones.

**Files:**
- Modify: `src/backend/options_intel/portfolio.py`
- Test: `tests/test_options_structure_e2e.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_options_structure_e2e.py`:

```python
from datetime import date
from decimal import Decimal

import pytest

from src.backend.options_intel.portfolio import aggregate_portfolio_greeks


class FakeGreeksSource:
    def __init__(self, table):
        self.table = table

    async def get_greeks(self, instrument_name):
        return self.table.get(instrument_name, {})


@pytest.mark.asyncio
async def test_aggregate_emits_structures_for_credit_put_spread():
    positions = [
        {"instrument_name": "BTC-27JUN26-100000-P", "size": 0.1, "side": "short"},
        {"instrument_name": "BTC-27JUN26-90000-P", "size": 0.1, "side": "long"},
    ]
    greeks = {
        "BTC-27JUN26-100000-P": {"delta": -0.30, "gamma": 0.001, "vega": 50, "theta": -5, "mark_iv": 0.60},
        "BTC-27JUN26-90000-P": {"delta": -0.10, "gamma": 0.0005, "vega": 20, "theta": -2, "mark_iv": 0.65},
    }
    result = await aggregate_portfolio_greeks(
        positions=positions,
        greeks_source=FakeGreeksSource(greeks),
        today=date(2026, 6, 13),  # 14 DTE
        spot=100000.0,
    )
    assert "structures" in result
    assert len(result["structures"]) == 1
    structure = result["structures"][0]
    assert structure["kind"] == "credit_put_spread"
    assert structure["underlying"] == "BTC"
    assert structure["is_credit"] is True


@pytest.mark.asyncio
async def test_aggregate_empty_positions_emits_no_structures():
    result = await aggregate_portfolio_greeks(
        positions=[],
        greeks_source=FakeGreeksSource({}),
        today=date(2026, 6, 13),
    )
    assert result["structures"] == []


@pytest.mark.asyncio
async def test_aggregate_emits_unknown_for_single_short_leg():
    positions = [
        {"instrument_name": "BTC-27JUN26-100000-P", "size": 0.1, "side": "short"},
    ]
    greeks = {
        "BTC-27JUN26-100000-P": {"delta": -0.30, "gamma": 0.001, "vega": 50, "theta": -5, "mark_iv": 0.60},
    }
    result = await aggregate_portfolio_greeks(
        positions=positions,
        greeks_source=FakeGreeksSource(greeks),
        today=date(2026, 6, 13),
        spot=100000.0,
    )
    assert len(result["structures"]) == 1
    assert result["structures"][0]["kind"] == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_options_structure_e2e.py -v
```

Expected: FAIL — `KeyError: 'structures'`.

- [ ] **Step 3: Wire classify() into aggregate_portfolio_greeks**

Modify `src/backend/options_intel/portfolio.py`. At the top, add imports:

```python
from src.backend.options_intel.structure import (
    OptionLeg,
    classify,
)
from decimal import Decimal
```

At the end of `aggregate_portfolio_greeks` (just before the existing `return` statement at line 173), insert:

```python
    # Build OptionLeg objects for the structure classifier.
    structure_legs: list[OptionLeg] = []
    for position_view in open_positions:
        try:
            structure_legs.append(
                OptionLeg(
                    instrument_name=position_view["instrument_name"],
                    kind=position_view["kind"],
                    strike=Decimal(str(position_view["strike"])),
                    side=position_view["side"],
                    contracts=Decimal(str(position_view["size"])),
                    days_to_expiry=int(position_view["days_to_expiry"]),
                    mark_price=Decimal("0"),  # mark not surfaced here; structure premium uses 0
                    delta=Decimal(str(position_view["delta"])) if "delta" in position_view else None,
                    gamma=Decimal(str(position_view["gamma"])) if "gamma" in position_view else None,
                    vega=Decimal(str(position_view["vega"])) if "vega" in position_view else None,
                    theta=Decimal(str(position_view["theta"])) if "theta" in position_view else None,
                )
            )
        except (KeyError, ValueError) as exc:
            logger.debug("portfolio aggregator: skipping leg for classifier: %s", exc)
            continue

    structures: list[dict] = []
    if structure_legs:
        result = classify(structure_legs)
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

Then change the final return to include the new key:

```python
    return {
        "open_positions": open_positions,
        "portfolio_greeks": totals,
        "structures": structures,
    }
```

Note: Phase 1 emits a single classification covering all legs. Multi-structure portfolios (e.g., two separate credit put spreads) collapse into one `UNKNOWN` in Phase 1; multi-structure splitting is a Phase 2 task.

- [ ] **Step 4: Thread `structures` through OptionsContext (non-prompted)**

`OptionsContext` is the dataclass at `src/backend/options_intel/snapshot.py:27-83`. Add a `structures` field that the bot_engine will read, but exclude it from `to_dict()` so the LLM prompt is unchanged.

In `src/backend/options_intel/snapshot.py`, add a field next to `open_positions` (around line 53):

```python
    # Portfolio
    open_positions: list = field(default_factory=list)
    portfolio_greeks: dict = field(default_factory=dict)
    # Phase 1: classifier output. NOT included in to_dict() — the LLM prompt
    # is unchanged until OPTIONS_STRUCTURE_PROMPT=1 (Phase 2). The bot_engine
    # reads this for persistence and GUI exposure.
    structures: list = field(default_factory=list)
```

Then find the function in `snapshot.py` that calls `aggregate_portfolio_greeks` (search: `grep -n "aggregate_portfolio_greeks" src/backend/options_intel/snapshot.py`). Update the call site to thread the new `structures` key into the `OptionsContext` constructor:

```python
portfolio_result = await aggregate_portfolio_greeks(...)
# existing fields...
open_positions=portfolio_result["open_positions"],
portfolio_greeks=portfolio_result["portfolio_greeks"],
structures=portfolio_result.get("structures", []),
```

Verify `to_dict()` does **not** include `structures` (it should not — leave the function alone).

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_options_structure_e2e.py -v
```

Expected: all 3 tests pass. Also run the broader suite to confirm no regression in existing portfolio + context tests:

```bash
pytest tests/test_options_context_coverage.py tests/test_options_context_margin.py tests/test_options_builder.py -v
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/backend/options_intel/portfolio.py src/backend/options_intel/snapshot.py tests/test_options_structure_e2e.py
git commit -m "feat(options): emit structures from aggregate_portfolio_greeks + thread through OptionsContext"
```

---

## Task 11: Persist structures from bot_engine after each options cycle

After every options decision cycle, upsert each currently-classified structure and mark previously-open structures that have disappeared as closed.

**Files:**
- Modify: `src/backend/bot_engine.py`
- Test: `tests/test_options_structure_e2e.py`

- [ ] **Step 1: Reference points in bot_engine.py**

The options decision cycle handler is `_run_options_decision_cycle` at `src/backend/bot_engine.py:1053`. It reads `self._latest_options_context` (populated by the surface-refresh task around line 963) and calls `agent.decide(...)` around line 1074. Persistence runs **before** the `agent.decide` call so structures are recorded whether or not the LLM cycle succeeds.

`bot_engine.py` does not hold a `self.db_manager` attribute — it imports `get_db_manager()` lazily inside the method (see existing pattern at line 946-948).

- [ ] **Step 2: Write the failing test**

Append to `tests/test_options_structure_e2e.py`:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.database.db_manager import DatabaseManager


@pytest.mark.asyncio
async def test_persist_structures_upserts_open_marks_closed(monkeypatch):
    """Persistence helper writes new structures, updates existing ones, and
    closes structures that disappeared since the previous cycle."""
    from src.backend.bot_engine import persist_options_structures

    db = DatabaseManager(db_url="sqlite:///:memory:")

    # Initial cycle: one credit put spread observed
    structures_cycle_1 = [{
        "structure_id": "abc123",
        "underlying": "BTC",
        "kind": "credit_put_spread",
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
    }]

    persist_options_structures(db, structures_cycle_1)
    open_ids = {row["structure_id"] for row in db.get_open_structures()}
    assert open_ids == {"abc123"}

    # Second cycle: same structure still open (update last_pnl)
    structures_cycle_2 = [{**structures_cycle_1[0], "pnl_abs": 5.0, "pnl_pct": 0.25, "breach_state": "warning"}]
    persist_options_structures(db, structures_cycle_2)
    rows = db.get_open_structures()
    assert len(rows) == 1
    assert rows[0]["last_pnl_abs"] == 5.0

    # Third cycle: structure closed
    persist_options_structures(db, [])
    open_ids = {row["structure_id"] for row in db.get_open_structures()}
    assert open_ids == set()
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_options_structure_e2e.py::test_persist_structures_upserts_open_marks_closed -v
```

Expected: FAIL — `ImportError: cannot import name 'persist_options_structures' from 'src.backend.bot_engine'`.

- [ ] **Step 4: Add persist_options_structures helper**

Add the following function to `src/backend/bot_engine.py` at module scope (above the engine class):

```python
def persist_options_structures(db_manager, current_structures: list[dict]) -> None:
    """Upsert each current structure and close any previously-open structure
    not in the current set.

    Phase 1 gates calls to this from feature flag OPTIONS_STRUCTURE_LAYER.
    """
    from decimal import Decimal as _Decimal

    current_ids = {s["structure_id"] for s in current_structures}

    for s in current_structures:
        db_manager.upsert_structure_snapshot(
            structure_id=s["structure_id"],
            underlying=s["underlying"],
            kind=s["kind"],
            legs_json=s.get("legs", []),
            entry_net_premium=_Decimal(str(s["net_premium"])),
            last_pnl_abs=_Decimal(str(s["pnl_abs"])),
            last_pnl_pct=_Decimal(str(s["pnl_pct"])),
            last_breach_state=s["breach_state"],
        )

    previously_open = db_manager.get_open_structures()
    for row in previously_open:
        if row["structure_id"] not in current_ids:
            db_manager.mark_structure_closed(row["structure_id"])
```

- [ ] **Step 5: Wire the helper into the options cycle**

In `src/backend/bot_engine.py`, inside `_run_options_decision_cycle` (around line 1053), add the following block **after** the `self._latest_options_context is None` early-return check and **before** the `agent.decide(...)` call (around line 1074):

```python
        if os.environ.get("OPTIONS_STRUCTURE_LAYER") == "1":
            structures = getattr(self._latest_options_context, "structures", []) or []
            try:
                from src.database.db_manager import get_db_manager as _get_db_manager
                persist_options_structures(_get_db_manager(), structures)
            except Exception as exc:
                self.logger.warning("options structure persistence failed: %s", exc)
```

Note: `import os` is already at the top of bot_engine.py (verify with `grep -n "^import os" src/backend/bot_engine.py`). If missing, add it. The block is fail-soft — persistence errors are logged and the LLM cycle proceeds.

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_options_structure_e2e.py -v
```

Expected: all tests pass.

Also run the full bot_engine smoke tests:

```bash
pytest tests/test_bot_engine_positions_view.py tests/test_bot_engine_retry.py -v
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/backend/bot_engine.py tests/test_options_structure_e2e.py
git commit -m "feat(options): persist option structures across cycles + close-on-disappear"
```

---

## Task 12: Wire OPTIONS_STRUCTURE_LAYER flag + .env.example

The persistence path already gates on the env var; document it and expose it through `config_loader` for consistency.

**Files:**
- Modify: `src/backend/config_loader.py`
- Modify: `.env.example`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config_loader.py`:

```python
def test_options_structure_layer_flag_default_off(monkeypatch):
    monkeypatch.delenv("OPTIONS_STRUCTURE_LAYER", raising=False)
    from importlib import reload
    from src.backend import config_loader
    reload(config_loader)
    assert config_loader.CONFIG.get("OPTIONS_STRUCTURE_LAYER") in (None, "0", False)


def test_options_structure_layer_flag_on(monkeypatch):
    monkeypatch.setenv("OPTIONS_STRUCTURE_LAYER", "1")
    from importlib import reload
    from src.backend import config_loader
    reload(config_loader)
    assert str(config_loader.CONFIG.get("OPTIONS_STRUCTURE_LAYER")) == "1"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_config_loader.py -v -k options_structure_layer
```

Expected: FAIL — `CONFIG` does not contain `OPTIONS_STRUCTURE_LAYER`.

- [ ] **Step 3: Add the flag to config_loader**

In `src/backend/config_loader.py`, locate the dict that populates `CONFIG`. Add a line that reads the env var:

```python
CONFIG["OPTIONS_STRUCTURE_LAYER"] = os.environ.get("OPTIONS_STRUCTURE_LAYER", "0")
```

(Match the style of nearby entries — if other flags use `.lower() == "1"` or similar, follow that pattern.)

- [ ] **Step 4: Update .env.example**

Append to `.env.example`:

```env

# --- Options Structure Layer (Phase 1) ---
# Enable structure classifier + DB persistence + GUI grouping.
# When 0: positions are flat leg lists as before; classifier still runs but
# nothing is persisted and GUI shows the legacy view.
# When 1: each options cycle classifies positions into typed structures
# (credit/debit spreads, ICs, etc.), persists to option_structure_snapshots,
# and the GUI groups Thalex legs by structure_id.
# OPTIONS_STRUCTURE_LAYER=0
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_config_loader.py -v
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/backend/config_loader.py .env.example tests/test_config_loader.py
git commit -m "feat(options): document and wire OPTIONS_STRUCTURE_LAYER flag"
```

---

## Task 13: GUI structure grouping in positions.py

When the flag is on, group Thalex options legs into structure cards. Hyperliquid perps continue to render in the legacy table — cross-venue linkage is out of scope for Phase 1.

**Files:**
- Modify: `src/gui/pages/positions.py`
- Modify: `src/gui/services/bot_service.py`
- Test: `tests/test_options_structure_e2e.py`

- [ ] **Step 1: Reference points in positions.py**

The positions page entry is `create_positions(bot_service, state_manager)` at `src/gui/pages/positions.py:11`. The Thalex/Hyperliquid table is built around lines 58-83 (one unified `ui.table` rendering all venue positions). The structure card section will be inserted **above** that table when `view["thalex_structures"]` is non-empty; the existing table continues to render the leg list underneath (legs already appear in the table — duplication is intentional in Phase 1 so the operator can cross-check).

- [ ] **Step 2: Write the failing test**

Append to `tests/test_options_structure_e2e.py`:

```python
def test_bot_service_exposes_structures_when_flag_on(monkeypatch):
    """When OPTIONS_STRUCTURE_LAYER=1, bot_service.get_positions_view() returns
    a 'structures' key with the current cycle's classification."""
    from src.gui.services.bot_service import build_positions_view

    monkeypatch.setenv("OPTIONS_STRUCTURE_LAYER", "1")
    state_payload = {
        "thalex_positions": [
            {"instrument_name": "BTC-27JUN26-100000-P", "size": 0.1, "side": "short", "kind": "put", "strike": 100000, "days_to_expiry": 14, "delta": -0.30},
            {"instrument_name": "BTC-27JUN26-90000-P", "size": 0.1, "side": "long", "kind": "put", "strike": 90000, "days_to_expiry": 14, "delta": -0.10},
        ],
        "structures": [
            {"structure_id": "abc123", "kind": "credit_put_spread", "underlying": "BTC",
             "tenor_days_min": 14, "tenor_days_max": 14, "net_premium": 20.0, "is_credit": True,
             "max_loss": 980.0, "max_profit": 20.0, "breakevens": [99800.0],
             "short_leg_delta": -0.30, "breach_state": "warning",
             "pnl_abs": 0.0, "pnl_pct": 0.0, "aggregate_greeks": {"delta": -0.20},
             "confidence": 1.0, "legs": ["BTC-27JUN26-100000-P", "BTC-27JUN26-90000-P"]},
        ],
    }
    view = build_positions_view(state_payload)
    assert "thalex_structures" in view
    assert len(view["thalex_structures"]) == 1
    assert view["thalex_structures"][0]["kind"] == "credit_put_spread"


def test_bot_service_omits_structures_when_flag_off(monkeypatch):
    from src.gui.services.bot_service import build_positions_view

    monkeypatch.delenv("OPTIONS_STRUCTURE_LAYER", raising=False)
    state_payload = {
        "thalex_positions": [],
        "structures": [{"structure_id": "x", "kind": "credit_put_spread"}],
    }
    view = build_positions_view(state_payload)
    assert view.get("thalex_structures", []) == []
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_options_structure_e2e.py -v -k positions_view
```

Expected: FAIL — `ImportError: cannot import name 'build_positions_view'`.

- [ ] **Step 4: Add build_positions_view helper to bot_service.py**

In `src/gui/services/bot_service.py`, add at module scope:

```python
import os as _os


def build_positions_view(state_payload: dict) -> dict:
    """Project the engine state into the shape the positions page expects.

    Returns:
        {
            "thalex_positions": [...],   # flat leg list (legacy)
            "thalex_structures": [...],  # populated only when OPTIONS_STRUCTURE_LAYER=1
            "hyperliquid_positions": [...],
        }
    """
    flag_on = _os.environ.get("OPTIONS_STRUCTURE_LAYER") == "1"
    return {
        "thalex_positions": state_payload.get("thalex_positions", []),
        "thalex_structures": state_payload.get("structures", []) if flag_on else [],
        "hyperliquid_positions": state_payload.get("hyperliquid_positions", []),
    }
```

- [ ] **Step 5: Render structure cards above the existing table**

In `src/gui/pages/positions.py`, insert a structure-cards section between the summary tiles (ending around line 55) and the `Position Details` card (starting around line 58). The existing positions table is unchanged — Phase 1 layers the card view on top.

Add near the top of the file (with other imports):

```python
from src.gui.services.bot_service import build_positions_view
```

Then between the summary tiles and the position details card:

```python
    structures_container = ui.column().classes('w-full mb-6')

    def _render_structures():
        structures_container.clear()
        state_payload = bot_service.get_latest_state_payload() if hasattr(bot_service, 'get_latest_state_payload') else {}
        view = build_positions_view(state_payload)
        if not view["thalex_structures"]:
            return
        with structures_container:
            ui.label('Thalex Structures').classes('text-xl font-bold text-white mb-2')
            for structure in view["thalex_structures"]:
                with ui.card().classes('w-full p-3 mb-2 bg-gray-800'):
                    with ui.row().classes('items-center justify-between w-full'):
                        kind_label = structure["kind"].replace("_", " ").title()
                        ui.label(kind_label).classes('text-md font-semibold text-white')
                        badge_color = {
                            'nominal': 'green',
                            'warning': 'orange',
                            'breached': 'red',
                        }.get(structure["breach_state"], 'grey')
                        ui.badge(structure["breach_state"], color=badge_color)
                    ui.label(
                        f"Premium: {structure['net_premium']:+.2f} "
                        f"({'credit' if structure['is_credit'] else 'debit'})  "
                        f"DTE: {structure['tenor_days_min']}-{structure['tenor_days_max']}  "
                        f"P&L: {structure['pnl_abs']:+.2f} ({structure['pnl_pct']*100:+.2f}%)"
                    ).classes('text-sm text-gray-300')
                    if structure["max_loss"] is not None:
                        max_profit_str = (
                            f"{structure['max_profit']:.2f}"
                            if structure["max_profit"] is not None
                            else 'unbounded'
                        )
                        ui.label(
                            f"Max loss: {structure['max_loss']:.2f}  Max profit: {max_profit_str}"
                        ).classes('text-xs text-gray-400')
                    if structure["breakevens"]:
                        ui.label(
                            "Breakevens: " + ", ".join(f"{b:.0f}" for b in structure["breakevens"])
                        ).classes('text-xs text-gray-400')
                    with ui.expansion('Legs').classes('w-full'):
                        for leg in structure["legs"]:
                            ui.label(f"• {leg}").classes('text-xs text-gray-300')

    _render_structures()
```

The `bot_service.get_latest_state_payload()` call shape: if the service exposes positions/state via a different method name, grep:

```bash
grep -n "def get_\|state_payload\|self\._state" src/gui/services/bot_service.py | head -20
```

Use the existing method that already returns the state dict the positions table consumes (look at what's used to populate the existing `ui.table` rows higher up in `positions.py`). Wrap that same call in `build_positions_view(...)`.

If the existing table relies on a callback that runs on a timer to refresh, also register `_render_structures` on the same timer (the existing pattern in `positions.py` will show the cadence).

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_options_structure_e2e.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Manual smoke test**

```bash
OPTIONS_STRUCTURE_LAYER=1 python main.py
```

Open the browser at the URL NiceGUI prints. Navigate to the Positions page. Confirm:
- If Thalex has open positions, a "Thalex structures" section appears with cards
- Each card shows kind, premium, P&L, breach badge
- Expanding "Legs" reveals individual instrument names
- Hyperliquid positions render in their existing table beneath/alongside

Set `OPTIONS_STRUCTURE_LAYER=0` (or unset) and restart; confirm legacy flat-leg view returns.

- [ ] **Step 8: Commit**

```bash
git add src/gui/services/bot_service.py src/gui/pages/positions.py tests/test_options_structure_e2e.py
git commit -m "feat(gui): group Thalex legs by structure when OPTIONS_STRUCTURE_LAYER=1"
```

---

## Phase 1 ship gate

Before declaring Phase 1 done:

- [ ] **Run the full test suite:**

```bash
pytest -v
```

Expected: green. New tests added in this phase:
- `tests/test_decision_schema.py` (extended)
- `tests/test_option_structure_classifier.py` (new — ~25 tests)
- `tests/test_option_structure_persistence.py` (new — 4 tests)
- `tests/test_options_structure_e2e.py` (new — 6 tests)
- `tests/test_config_loader.py` (extended)

- [ ] **End-to-end smoke with the flag on, against a populated Thalex account:**

```bash
OPTIONS_SCHEDULER_ENABLED=1 OPTIONS_STRUCTURE_LAYER=1 python main.py
```

Verify in the Reasoning page or `bot.log`:
- Options cycle ran
- `option_structure_snapshots` table has rows (`sqlite3 data/trading_bot.db "select structure_id, kind, last_breach_state from option_structure_snapshots"`)
- Positions page shows structure cards

- [ ] **Flag-off regression check:**

```bash
unset OPTIONS_STRUCTURE_LAYER
python main.py
```

Verify Positions page renders the legacy flat-leg view; no DB writes to `option_structure_snapshots`; LLM behavior is identical to pre-Phase-1.

---

## Notes

- **Phase 2 prerequisites**: Phase 2 reads `OptionStructureSnapshot.entry_net_premium` as the P&L baseline for prompts. Phase 1 sets that baseline on first sight of a structure; if a structure was opened pre-Phase-1, the first Phase-1 cycle that sees it records the then-current `net_premium` as its baseline. That's not a true entry price but it's the best we can do without backfilling — and P&L will be approximately zero on first sight of pre-existing structures, which is honest.
- **Multi-structure portfolios**: Phase 1 collapses multi-structure portfolios into one classifier call. If the account holds both a credit put spread and an iron condor simultaneously, the classifier sees 6 legs and returns `UNKNOWN`. Phase 2 task list includes a structure-splitter (graph match by tenor + kind groupings) that calls `classify()` on each subset.
- **Mark price for structure premium**: `aggregate_portfolio_greeks` doesn't currently surface mark price per leg; the classifier currently sees `mark_price=Decimal("0")` and so `net_premium` is zero. Phase 1 still ships P&L UI (always zero) and Phase 2 wires mark price into the leg builder. The DB schema is unchanged between phases.
