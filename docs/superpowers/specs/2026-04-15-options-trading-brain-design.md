# Options Trading Brain — Design Spec

**Date:** 2026-04-15
**Branch:** `aladhi-multi-tenor-hardening`
**Scope:** Expand the OptionsAgent system prompt into a comprehensive BTC options trading brain + fix perps LLM context leaks

## Problem

The OptionsAgent pipeline is fully built and tested but:
1. The system prompt (`options_agent.py:49-106`) is a ~60-line cheat sheet — too thin for real trading decisions
2. It has no position management logic (rolling, profit-taking, cut-loss)
3. It has no crypto-specific regime knowledge (BTC vol structure, event patterns)
4. The options scheduler is gated behind `OPTIONS_SCHEDULER_ENABLED=1` (undocumented in `.env.example`)
5. The perps LLM sees UI-only fields (`closable`, `row_id`) in position context and parrots "not closable" in reasoning
6. The perps LLM has zero visibility into the options book state

## Approach

**Fat system prompt (Approach 2 — sectioned constants):** Expand the prompt from ~60 to ~250 lines, organized as named Python constants in `options_agent.py` that are joined into the final prompt. No external files, no knowledge base, no new modules.

**Token budget:** ~3,000 tokens for the system prompt + ~500 tokens for the OptionsContext user prompt = ~3,500 total input. Well within the 128K+ context windows of all configured models.

## Architecture

### Prompt Structure

Six named constants in `options_agent.py`, concatenated into `_OPTIONS_SYSTEM_PROMPT`:

```python
_PREAMBLE = """..."""           # Identity, venue rules (~30 lines, refactored from existing)
_STRATEGY_SELECTION = """..."""  # Decision trees by regime/skew/direction (~60 lines, new)
_POSITION_MANAGEMENT = """...""" # Rolling, profit-taking, cut-loss rules (~50 lines, new)
_REGIME_PLAYBOOK = """..."""     # Crypto-specific BTC vol patterns (~50 lines, new)
_RISK_FRAMEWORK = """..."""      # Sizing, portfolio constraints, strike widths (~40 lines, new)
_OUTPUT_CONTRACT = """..."""     # JSON schema, field requirements (~20 lines, extracted from existing)

_OPTIONS_SYSTEM_PROMPT = "\n\n".join([
    _PREAMBLE,
    _STRATEGY_SELECTION,
    _POSITION_MANAGEMENT,
    _REGIME_PLAYBOOK,
    _RISK_FRAMEWORK,
    _OUTPUT_CONTRACT,
])
```

### Files Changed

| File | Change |
|------|--------|
| `src/backend/agent/options_agent.py` | Replace `_OPTIONS_SYSTEM_PROMPT` with 6 sectioned constants |
| `src/backend/bot_engine.py` | Strip `closable`/`row_id` from LLM context; inject options summary into perps context |
| `src/backend/config_loader.py` | No changes (scheduler config already exists) |
| `.env.example` | Add `OPTIONS_SCHEDULER_ENABLED` and related vars |
| `tests/test_options_agent.py` | Extend assertions for new prompt sections |

## Prompt Content

### 1. Preamble (`_PREAMBLE`)

Refactored from the existing prompt. Covers:
- Agent identity: hedge-fund-grade BTC options trader on Thalex, hedged on Hyperliquid perps
- Venue routing: all options to `venue="thalex"`, delta hedges fire automatically
- Strategy enum: the 6 defined-risk strategies (credit_put_spread, credit_call_spread, iron_condor, long_call_delta_hedged, long_put_delta_hedged, vol_arb)
- Naked legs forbidden

### 2. Strategy Selection (`_STRATEGY_SELECTION`)

Decision tree keyed on `vol_regime`, `skew_25d_by_tenor`, and directional signals:

```
vol_regime?
+-- "rich" (IV >> RV)
|   +-- ranging (spot inside Keltner)
|   |   +-- skew negative (puts expensive) -> credit_put_spread
|   |   +-- skew positive (calls expensive) -> credit_call_spread
|   |   +-- skew neutral -> iron_condor
|   +-- trending -> credit spread OPPOSITE to trend direction
|       (sell calls in uptrend, sell puts in downtrend)
+-- "cheap" (IV << RV)
|   +-- trending up -> long_call_delta_hedged
|   +-- trending down -> long_put_delta_hedged
|   +-- ranging -> both delta-hedged legs (buy gamma, wait for breakout)
+-- "fair" (IV ~ RV)
    +-- mispricings > 200bps -> vol_arb
    +-- term_structure_slope steep -> calendar-style vol_arb across tenors
    +-- directional conviction from keltner/opening_range -> delta-hedged long in that direction
```

No "unknown -> hold" fallback. Instead: *"If vol_regime_confidence is low, weight mispricings and portfolio greeks more heavily. Never use regime uncertainty as an excuse to skip analysis — reason from the data you have."*

**Tenor selection:**
- Premium selling (credit spreads, condors): 7-21 DTE, sweet spot 14 DTE
- Directional gamma (delta-hedged): 21-45 DTE
- Vol arb: match the mispriced tenor exactly

**Strike selection:**
- Credit put spread: short at or below 15-delta put, long 5-10% further OTM
- Credit call spread: short at or above 15-delta call, long 5-10% further OTM
- Iron condor: both wings equidistant from spot
- Delta-hedged longs: ATM or slight OTM (25-40 delta)

### 3. Position Management (`_POSITION_MANAGEMENT`)

**Rolling:**
- Short spread < 3 DTE and profitable: roll to next 14 DTE at same or wider strikes
- Short spread < 3 DTE and underwater: let expire if max loss priced in, don't roll losers
- Delta-hedged long < 7 DTE and hasn't hit target: roll out if vol_regime still supports thesis, else close

**Profit-taking:**
- Credit spreads: close at 50-65% of max profit. Don't hold to expiry for the last 35%
- Delta-hedged longs: take profits when realized move exceeds expected_move_pct for that tenor
- Iron condors: close the tested side if spot breaks through short strike, keep untested side

**Cut-loss:**
- Credit spread: close if spread value doubles from entry (2x premium received)
- Delta-hedged long: close if IV drops further after entry (thesis was "vol cheap" but it got cheaper)
- Any position: close immediately if portfolio net delta exceeds 0.15 BTC and the position is the cause

**Action field mapping:**
- Roll: emit close action + open action as two decisions in the same array
- Profit/loss: `action="buy"` to close a short position, `action="sell"` to close a long
- Reference the existing position by instrument name in `rationale`

### 4. Regime Playbook (`_REGIME_PLAYBOOK`)

**BTC vol structure:**
- BTC IV typically 60-80% annualized calm, 100-150%+ during events
- `realized_iv_ratio_15d` below 0.8 = IV overpriced (sell premium), above 1.2 = IV underpriced (buy gamma)
- BTC skew structurally negative (puts richer). Skew flip to positive = euphoric positioning, fade with credit call spreads
- Term structure normally contango. Backwardation = near-term event/panic, short near-term + long back-month via vol_arb

**Event patterns:**
- Pre-major-expiry (Friday quarterlies): spot pins near max-pain, short vol via iron condors 2-3 days before
- Post-expiry: gamma unwind releases spot, wait for move then buy gamma if vol reprices cheap
- Weekend: avoid opening new short vol Friday afternoon, wait for Monday
- Macro events (FOMC, CPI): IV bids up 2-3 days before, crushes after. Sell premium AFTER event when IV still elevated but catalyst gone

**When NOT to trade:**
- Portfolio at `max_open_positions`
- `realized_iv_ratio_15d` between 0.9 and 1.1 AND no mispricings > 150bps — manage existing only
- Net portfolio vega > 500 USD/vol-point — rebalance before adding

### 5. Risk Framework (`_RISK_FRAMEWORK`)

**Position sizing:**
- Base: 0.02-0.05 contracts per leg
- Scale to 0.1 (hard cap) only when vol_regime_confidence "high" AND realized_iv_ratio confirms
- Same-direction exposure already exists: halve the size
- `capital_available` < 500 USD: minimum size (0.01) only

**Portfolio constraints:**
- Net delta: keep between -0.10 and +0.10 BTC
- Net vega: max 300 USD/vol-point short, 500 USD/vol-point long
- Theta: target positive when vol_regime "rich", accept negative when "cheap"
- No contradictory positions: don't run credit_put_spread + long_put_delta_hedged simultaneously

**Strike widths:**
- Minimum 5% spread between short and long strikes on any vertical
- Maximum 15% spread

### 6. Output Contract (`_OUTPUT_CONTRACT`)

Extracted from existing prompt, unchanged:
- Return strict JSON: `{"reasoning": str, "trade_decisions": [...]}`
- Each decision: asset, action, rationale, venue="thalex", strategy, underlying, tenor_days, legs[]
- No markdown, no extra properties
- Empty trade_decisions if no trade warranted

## Perps LLM Context Fix

### Strip UI-only fields (`bot_engine.py`)

In Phase 9 context building, filter `closable` and `row_id` from each position dict before serialization:

```python
# Strip UI-only fields before LLM context
llm_positions = [
    {k: v for k, v in pos.items() if k not in ("closable", "row_id")}
    for pos in combined_positions
]
```

Use `llm_positions` in the `dashboard` dict instead of `combined_positions`.

### Inject options summary into perps context (`bot_engine.py`)

When `self._latest_options_context` is available, add a compact summary to the perps LLM context:

```python
if self._latest_options_context is not None:
    ctx = self._latest_options_context
    context_payload["options_book_summary"] = {
        "vol_regime": ctx.vol_regime,
        "portfolio_greeks": ctx.portfolio_greeks,
        "open_position_count": ctx.open_position_count,
    }
```

~50 tokens. Enough for the perps agent to factor in "the options desk is net long 0.05 delta" without doing vol reasoning.

## `.env.example` additions

```env
# --- Options Scheduler ---
# Set to 1 to enable the two-cadence options loop (vol surface refresh + decision cycle)
# OPTIONS_SCHEDULER_ENABLED=0
# Vol surface refresh interval in seconds (default: 900 = 15 min)
# OPTIONS_VOL_SURFACE_INTERVAL_SECONDS=900
# Options decision cycle interval in seconds (default: 10800 = 3 hours)
# OPTIONS_DECISION_INTERVAL_SECONDS=10800
```

## Test Changes

Extend `tests/test_options_agent.py`:

- `test_agent_system_prompt_contains_strategy_selection_section`: assert key phrases from the decision tree (e.g., "credit_put_spread", "skew negative", "realized_iv_ratio")
- `test_agent_system_prompt_contains_position_management_section`: assert rolling/profit-taking language
- `test_agent_system_prompt_contains_regime_playbook`: assert BTC-specific terms (e.g., "backwardation", "max-pain", "gamma unwind")
- `test_agent_system_prompt_contains_risk_framework`: assert sizing rules (e.g., "0.02", "net delta", "vega")

Existing tests remain unchanged — they test the pipeline mechanics (parsing, error handling), not prompt content.

## Out of Scope

- Knowledge base / external file loading (future — Approach B)
- Persistent trade outcome memory (future)
- Options model change / separate model config for options vs perps
- Thalex positions closable via GUI (separate decision)
