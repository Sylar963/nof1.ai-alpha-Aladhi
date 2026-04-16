# Options Trading Brain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the OptionsAgent system prompt into a comprehensive BTC options trading brain and fix perps LLM context leaks.

**Architecture:** Replace the monolithic `_OPTIONS_SYSTEM_PROMPT` string in `options_agent.py` with 6 named section constants joined at module level. Strip UI-only fields from perps LLM context in `bot_engine.py` and inject a compact options book summary. Document the options scheduler env vars in `.env.example`.

**Tech Stack:** Python, pytest, pytest-asyncio. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-15-options-trading-brain-design.md`

---

### Task 1: Add prompt section content tests

**Files:**
- Modify: `tests/test_options_agent.py`

These tests assert that the expanded prompt contains the knowledge sections we're about to write. They fail now because the sections don't exist yet.

- [ ] **Step 1: Add 4 new test functions to `tests/test_options_agent.py`**

Append after the existing `test_agent_returns_empty_list_when_llm_returns_no_decisions` test (line 189):

```python


# ---------------------------------------------------------------------------
# Prompt knowledge sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_prompt_contains_strategy_selection():
    """The prompt must contain strategy selection decision logic."""
    llm = FakeLLMClient({"trade_decisions": []})
    agent = OptionsAgent(llm=llm)
    await agent.decide(_basic_context())

    prompt = llm.calls[0]["system_prompt"]
    lower = prompt.lower()
    # Decision tree branches by regime
    assert "skew" in lower
    assert "realized_iv_ratio" in prompt or "realized" in lower
    # Tenor guidance
    assert "7" in prompt and "21" in prompt  # premium selling DTE range
    assert "tenor" in lower


@pytest.mark.asyncio
async def test_system_prompt_contains_position_management():
    """The prompt must contain rolling, profit-taking, and cut-loss rules."""
    llm = FakeLLMClient({"trade_decisions": []})
    agent = OptionsAgent(llm=llm)
    await agent.decide(_basic_context())

    prompt = llm.calls[0]["system_prompt"]
    lower = prompt.lower()
    assert "roll" in lower
    assert "profit" in lower
    assert "cut" in lower or "close" in lower


@pytest.mark.asyncio
async def test_system_prompt_contains_regime_playbook():
    """The prompt must contain crypto-specific BTC vol knowledge."""
    llm = FakeLLMClient({"trade_decisions": []})
    agent = OptionsAgent(llm=llm)
    await agent.decide(_basic_context())

    prompt = llm.calls[0]["system_prompt"]
    lower = prompt.lower()
    assert "btc" in lower
    assert "backwardation" in lower or "contango" in lower
    assert "expiry" in lower or "gamma" in lower


@pytest.mark.asyncio
async def test_system_prompt_contains_risk_framework():
    """The prompt must contain sizing and portfolio constraint rules."""
    llm = FakeLLMClient({"trade_decisions": []})
    agent = OptionsAgent(llm=llm)
    await agent.decide(_basic_context())

    prompt = llm.calls[0]["system_prompt"]
    lower = prompt.lower()
    assert "delta" in lower
    assert "vega" in lower
    assert "0.02" in prompt or "sizing" in lower
```

- [ ] **Step 2: Run the new tests to confirm they fail**

Run: `pytest tests/test_options_agent.py::test_system_prompt_contains_strategy_selection tests/test_options_agent.py::test_system_prompt_contains_position_management tests/test_options_agent.py::test_system_prompt_contains_regime_playbook tests/test_options_agent.py::test_system_prompt_contains_risk_framework -v`

Expected: at least 2 FAIL (position_management and regime_playbook sections don't exist yet). strategy_selection may partially pass because the old prompt has some terms. risk_framework may partially pass because the old prompt mentions vega/delta.

- [ ] **Step 3: Commit**

```bash
git add tests/test_options_agent.py
git commit -m "test: add prompt knowledge section assertions for options brain"
```

---

### Task 2: Refactor existing prompt into `_PREAMBLE` and `_OUTPUT_CONTRACT`

**Files:**
- Modify: `src/backend/agent/options_agent.py:49-106`

Extract the existing prompt into two constants. The final `_OPTIONS_SYSTEM_PROMPT` is now built by joining sections. This is a pure refactor — no content changes.

- [ ] **Step 1: Replace the `_OPTIONS_SYSTEM_PROMPT` definition (lines 49-106) with the sectioned structure**

Replace everything from line 49 (`_OPTIONS_SYSTEM_PROMPT = """\`) through line 106 (`"""`) with:

```python
# ---------------------------------------------------------------------------
# System prompt — assembled from named sections
# ---------------------------------------------------------------------------

_PREAMBLE = """\
You are a hedge-fund-grade BTC OPTIONS trader running on the Thalex venue,
hedged on Hyperliquid perps. You reason about volatility surface, greeks,
expected moves, and term structure — NOT about RSI, MACD, or moving averages.

Your job is to read the compact OptionsContext digest below and emit
zero or more trade decisions in strict JSON form.

VENUE
- All options decisions go to Thalex (venue="thalex").
- Delta hedges fire automatically on Hyperliquid for the *_delta_hedged
  strategies — you do NOT need to size them yourself.

DEFINED-RISK STRATEGIES (the only allowed values for `strategy`)
- credit_put_spread        — sell premium below support, defined risk
- credit_call_spread       — sell premium above resistance, defined risk
- iron_condor              — sell both spreads simultaneously (vol crush)
- long_call_delta_hedged   — directional long, gamma-scalped on perps
- long_put_delta_hedged    — directional short, gamma-scalped on perps
- vol_arb                  — exploit Thalex IV mispricing vs Deribit

NAKED LEGS ARE FORBIDDEN. Every short premium position MUST be wrapped as a
vertical (credit_*_spread) or as both legs of an iron condor."""


_OUTPUT_CONTRACT = """\
OUTPUT CONTRACT
Return a strict JSON object with two keys, in order:
  - "reasoning": long-form analysis (string, be verbose).
  - "trade_decisions": array of decision objects.

Each decision object MUST contain:
  - asset, action (buy|sell|hold), rationale (string)
  - venue (must be "thalex")
  - strategy (one of the defined-risk values above)
  - underlying ("BTC")
  - tenor_days (int) OR per-leg tenor_days
  - For multi-leg strategies: legs[] with kind/side/contracts/target_strike

OPTIONAL METADATA (encouraged)
- entry_kind: outright | vertical | calendar | diagonal | iron_condor | vol_arb
- vol_view:   short_vol | long_vol | neutral

SIZING
- Always express size in `contracts` (float, BTC equivalent). Min 0.001,
  max 0.1 per trade.
- For multi-tenor gamma builds, set `target_gamma_btc` instead of `contracts`
  and the system will distribute across tenors automatically.

Do not emit Markdown. Do not include extra properties. If no trade is
warranted right now, return an empty trade_decisions array."""


_OPTIONS_SYSTEM_PROMPT = "\n\n".join([
    _PREAMBLE,
    _OUTPUT_CONTRACT,
])
```

- [ ] **Step 2: Run ALL existing tests to confirm nothing broke**

Run: `pytest tests/test_options_agent.py tests/test_options_pipeline_integration.py -v`

Expected: all 9 original tests PASS, the 4 new knowledge tests still fail (we haven't added the new sections yet).

- [ ] **Step 3: Commit**

```bash
git add src/backend/agent/options_agent.py
git commit -m "refactor: extract options prompt into _PREAMBLE and _OUTPUT_CONTRACT sections"
```

---

### Task 3: Add `_STRATEGY_SELECTION` section

**Files:**
- Modify: `src/backend/agent/options_agent.py`

- [ ] **Step 1: Add the `_STRATEGY_SELECTION` constant after `_PREAMBLE`**

Insert between the `_PREAMBLE` definition and `_OUTPUT_CONTRACT`:

```python
_STRATEGY_SELECTION = """\
STRATEGY SELECTION — decision tree

Read vol_regime, skew_25d_by_tenor, keltner position, and opening_range to
pick the right strategy. Walk the tree top-down; stop at the first matching
leaf.

WHEN vol_regime == "rich" (IV >> RV, realized_iv_ratio_15d < 0.8):
  Spot ranging (keltner position "inside", opening_range position "inside"):
    - skew_25d negative (puts expensive)  → credit_put_spread
    - skew_25d positive (calls expensive) → credit_call_spread
    - skew_25d near zero                  → iron_condor (sell both wings)
  Spot trending (keltner "above" or "below", or opening_range breakout):
    - Uptrend  → credit_call_spread (fade the overpriced calls into strength)
    - Downtrend → credit_put_spread (fade the overpriced puts into weakness)

WHEN vol_regime == "cheap" (IV << RV, realized_iv_ratio_15d > 1.2):
  Spot trending up   → long_call_delta_hedged (buy gamma, scalp delta on perps)
  Spot trending down → long_put_delta_hedged
  Spot ranging       → long_call_delta_hedged + long_put_delta_hedged
                        (synthetic straddle via gamma — wait for breakout)

WHEN vol_regime == "fair" (realized_iv_ratio_15d between 0.8 and 1.2):
  top_mispricings_vs_deribit with edge > 200 bps → vol_arb
  term_structure_slope steep (|slope| > 0.002)   → vol_arb across tenors
                        (short the expensive tenor, long the cheap one)
  Directional conviction from keltner/opening_range → delta-hedged long
                        in that direction (use direction when vol gives no edge)

If vol_regime_confidence is low, weight mispricings and portfolio greeks more
heavily. Never use regime uncertainty as an excuse to skip analysis — reason
from the data you have.

TENOR SELECTION
- Premium selling (credit spreads, iron condors): 7-21 DTE, sweet spot 14 DTE.
  Theta decay accelerates inside 21 DTE — this is where short premium earns.
- Directional gamma (delta-hedged longs): 21-45 DTE. Enough time for the move
  to develop, slower theta bleed while you wait.
- Vol arb: match the mispriced tenor exactly from top_mispricings_vs_deribit.

STRIKE SELECTION
- credit_put_spread:  short strike at or below 15-delta put (below support),
                      long strike 5-10% further OTM for defined risk.
- credit_call_spread: short strike at or above 15-delta call (above resistance),
                      long strike 5-10% further OTM.
- iron_condor:        both wings equidistant from spot for balanced risk.
- delta-hedged longs: ATM or slight OTM (25-40 delta) for best gamma/premium."""
```

- [ ] **Step 2: Update the `_OPTIONS_SYSTEM_PROMPT` join to include the new section**

Change:

```python
_OPTIONS_SYSTEM_PROMPT = "\n\n".join([
    _PREAMBLE,
    _OUTPUT_CONTRACT,
])
```

To:

```python
_OPTIONS_SYSTEM_PROMPT = "\n\n".join([
    _PREAMBLE,
    _STRATEGY_SELECTION,
    _OUTPUT_CONTRACT,
])
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_options_agent.py -v`

Expected: `test_system_prompt_contains_strategy_selection` now PASSES. All original tests still PASS.

- [ ] **Step 4: Commit**

```bash
git add src/backend/agent/options_agent.py
git commit -m "feat(options-brain): add strategy selection decision tree to prompt"
```

---

### Task 4: Add `_POSITION_MANAGEMENT` section

**Files:**
- Modify: `src/backend/agent/options_agent.py`

- [ ] **Step 1: Add the `_POSITION_MANAGEMENT` constant after `_STRATEGY_SELECTION`**

Insert between `_STRATEGY_SELECTION` and `_OUTPUT_CONTRACT`:

```python
_POSITION_MANAGEMENT = """\
POSITION MANAGEMENT — rolling, profit-taking, and cut-loss

You are responsible for managing existing positions, not just opening new ones.
Check open_positions and portfolio_greeks BEFORE proposing new trades.

ROLLING
- Short spread < 3 DTE and profitable: roll to next 14 DTE cycle at same or
  wider strikes to keep collecting theta. Emit two decisions: one to close the
  existing spread (buy back), one to open the new spread (sell).
- Short spread < 3 DTE and underwater: let it expire if max loss is already
  priced in. Do NOT roll losers into fresh premium — that compounds risk.
- Delta-hedged long < 7 DTE and hasn't hit target: roll out to next tenor if
  vol_regime still supports the thesis (still "cheap"). Otherwise close.

PROFIT-TAKING
- Credit spreads / iron condors: close at 50-65% of max profit (buy back the
  spread cheap). Do NOT hold to expiry chasing the last 35% — gamma risk
  accelerates and a single adverse move can wipe the remaining edge.
- Delta-hedged longs: take profits when the realized spot move exceeds the
  expected_move_pct_by_tenor for that tenor. The gamma scalp on perps has
  already banked edge — don't get greedy.
- Iron condor one side tested: if spot breaks through the short strike on one
  side, close THAT side immediately. Keep the untested side running — it still
  has positive expected value.

CUT-LOSS
- Credit spread: close if spread mark-to-market doubles from entry premium
  received (2x loss). The thesis is wrong — cut and reassess.
- Delta-hedged long: close if IV drops further after entry. Your thesis was
  "vol is cheap" but it got cheaper — you are wrong, exit.
- Any position: close immediately if portfolio net delta (from portfolio_greeks)
  exceeds |0.15| BTC and this position is the primary contributor. The hedge
  is failing.

ACTION FIELD MAPPING
- To roll: emit the close decision + the open decision as two separate entries
  in the same trade_decisions array.
- To take profit or cut loss on a short position: action="buy" (buy back).
- To take profit or cut loss on a long position: action="sell".
- Always reference the existing position instrument name in the rationale field
  so the operator can trace the logic."""
```

- [ ] **Step 2: Update the `_OPTIONS_SYSTEM_PROMPT` join**

```python
_OPTIONS_SYSTEM_PROMPT = "\n\n".join([
    _PREAMBLE,
    _STRATEGY_SELECTION,
    _POSITION_MANAGEMENT,
    _OUTPUT_CONTRACT,
])
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_options_agent.py -v`

Expected: `test_system_prompt_contains_position_management` now PASSES. All original tests still PASS.

- [ ] **Step 4: Commit**

```bash
git add src/backend/agent/options_agent.py
git commit -m "feat(options-brain): add position management rules to prompt"
```

---

### Task 5: Add `_REGIME_PLAYBOOK` section

**Files:**
- Modify: `src/backend/agent/options_agent.py`

- [ ] **Step 1: Add the `_REGIME_PLAYBOOK` constant after `_POSITION_MANAGEMENT`**

```python
_REGIME_PLAYBOOK = """\
BTC VOL REGIME PLAYBOOK — crypto-specific knowledge

BTC VOLATILITY STRUCTURE
- BTC implied volatility typically trades 60-80% annualized in calm markets,
  100-150%+ during events or liquidation cascades.
- The key signal is realized_iv_ratio_15d: below 0.8 means IV is overpriced
  relative to actual moves (sell premium). Above 1.2 means IV is underpriced
  relative to actual moves (buy gamma).
- BTC skew is structurally negative — puts trade richer than equidistant calls
  because of persistent crash-hedging demand. A skew flip to positive (calls
  more expensive than puts) signals euphoric positioning — fade it with
  credit_call_spread.
- Term structure is normally in contango (longer tenors carry higher IV).
  Backwardation (near-term IV > long-term) signals a near-term event or panic.
  When you see backwardation: short near-term premium, long back-month via
  vol_arb to capture the term structure normalization.

EVENT-DRIVEN PATTERNS
- Pre-major-expiry (Friday quarterlies): gamma exposure concentrates near
  popular strikes, spot tends to pin near max-pain. Short vol via iron_condor
  placed around the max-pain strike 2-3 days before expiry.
- Post-expiry: gamma unwind releases spot from pinning — expect a directional
  move. Wait for the move to develop, then buy gamma via delta-hedged longs
  if vol reprices cheap after the gamma flush.
- Weekend: theta decays but spot can gap on low liquidity. Avoid opening new
  short vol positions on Friday afternoon. Wait for Monday when the weekend
  gap risk has resolved and fresh premium is available.
- Macro events (FOMC, CPI, major crypto catalysts): IV typically gets bid
  2-3 days before the event, then crushes after. Sell premium AFTER the event
  when IV is still elevated but the catalyst has passed — not before.

WHEN NOT TO TRADE
- Portfolio already at max_open_positions: focus on managing existing book.
  Emit management decisions (rolls, profit-takes) but no new openings.
- realized_iv_ratio_15d between 0.9 and 1.1 AND no mispricings above 150 bps:
  vol is fairly priced everywhere, there is genuinely no edge. Manage existing
  positions only.
- Net portfolio vega exceeds |500| USD/vol-point in either direction: the book
  is overexposed. Rebalance existing positions before adding new ones."""
```

- [ ] **Step 2: Update the `_OPTIONS_SYSTEM_PROMPT` join**

```python
_OPTIONS_SYSTEM_PROMPT = "\n\n".join([
    _PREAMBLE,
    _STRATEGY_SELECTION,
    _POSITION_MANAGEMENT,
    _REGIME_PLAYBOOK,
    _OUTPUT_CONTRACT,
])
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_options_agent.py -v`

Expected: `test_system_prompt_contains_regime_playbook` now PASSES. All original tests still PASS.

- [ ] **Step 4: Commit**

```bash
git add src/backend/agent/options_agent.py
git commit -m "feat(options-brain): add BTC vol regime playbook to prompt"
```

---

### Task 6: Add `_RISK_FRAMEWORK` section

**Files:**
- Modify: `src/backend/agent/options_agent.py`

- [ ] **Step 1: Add the `_RISK_FRAMEWORK` constant after `_REGIME_PLAYBOOK`**

```python
_RISK_FRAMEWORK = """\
RISK FRAMEWORK — sizing, portfolio constraints, strike widths

POSITION SIZING
- Base size: 0.02-0.05 contracts per leg. This is the default range.
- Scale up to max_contracts_per_trade (0.1) ONLY when vol_regime_confidence is
  "high" AND realized_iv_ratio_15d confirms the regime (< 0.8 for rich,
  > 1.2 for cheap).
- When the portfolio already has same-direction exposure (e.g., existing short
  vol position and you propose another credit spread): HALVE the base size to
  avoid concentration.
- When capital_available < 500 USD: use minimum size (0.01 contracts) only.
  Preserve capital for margin and hedging needs.

PORTFOLIO CONSTRAINTS
- Net delta: keep portfolio delta between -0.10 and +0.10 BTC. Read the
  portfolio_greeks.delta field. If adding a trade would push delta outside
  this band, either size down or pick a more delta-neutral strategy.
- Net vega: do not exceed 300 USD/vol-point short or 500 USD/vol-point long.
  Short vega blows up faster than long vega decays — the asymmetry demands
  tighter limits on the short side.
- Theta targeting: aim for positive theta (collecting time decay) when
  vol_regime is "rich". Accept negative theta when "cheap" — you are paying
  for gamma and the expected move should cover the theta cost.
- No contradictory positions: do NOT hold credit_put_spread and
  long_put_delta_hedged simultaneously — they are opposite vol bets on the
  same side. If you hold one and want the other, close the first.

STRIKE WIDTH (defined risk verticals)
- Minimum 5% distance between short and long strikes on any vertical spread.
  Tighter spreads have poor fill quality on Thalex and excessive pin risk.
- Maximum 15% distance — wider than that, the long protective leg is too far
  OTM to provide meaningful risk definition at reasonable cost."""
```

- [ ] **Step 2: Update the `_OPTIONS_SYSTEM_PROMPT` join**

```python
_OPTIONS_SYSTEM_PROMPT = "\n\n".join([
    _PREAMBLE,
    _STRATEGY_SELECTION,
    _POSITION_MANAGEMENT,
    _REGIME_PLAYBOOK,
    _RISK_FRAMEWORK,
    _OUTPUT_CONTRACT,
])
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_options_agent.py -v`

Expected: ALL 13 tests PASS (9 original + 4 new knowledge section tests).

- [ ] **Step 4: Run the full pipeline integration tests too**

Run: `pytest tests/test_options_agent.py tests/test_options_pipeline_integration.py -v`

Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backend/agent/options_agent.py
git commit -m "feat(options-brain): add risk framework to prompt"
```

---

### Task 7: Strip UI-only fields from perps LLM context

**Files:**
- Modify: `src/backend/bot_engine.py:1135-1145`

The perps LLM currently receives `closable` and `row_id` in position dicts. These are UI concerns that pollute the reasoning.

- [ ] **Step 1: In `bot_engine.py`, modify Phase 7 (Build Dashboard) to strip UI fields**

Find the dashboard construction at line ~1136:

```python
                    dashboard = {
                        'total_return_pct': total_return_pct,
                        'balance': balance,
                        'account_value': total_value,
                        'sharpe_ratio': sharpe_ratio,
                        'positions': combined_positions,
```

Replace `'positions': combined_positions,` with:

```python
                        'positions': [
                            {k: v for k, v in pos.items() if k not in ("closable", "row_id")}
                            for pos in combined_positions
                        ],
```

- [ ] **Step 2: Verify the bot still starts and produces prompts without `closable`**

Run: `pytest tests/ -k "test_" --co -q | head -5` (quick sanity that imports work)

Then check the code is syntactically correct:

Run: `python -c "from src.backend.bot_engine import TradingBotEngine; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/backend/bot_engine.py
git commit -m "fix: strip UI-only fields (closable, row_id) from perps LLM context"
```

---

### Task 8: Inject options book summary into perps context

**Files:**
- Modify: `src/backend/bot_engine.py:1276-1288`

When the options scheduler has populated `_latest_options_context`, add a compact summary to the perps LLM context payload so the perps agent knows the state of the options book.

- [ ] **Step 1: In Phase 9 (Build LLM Context), add `options_book_summary` to the payload**

Find the context_payload construction at line ~1277:

```python
                    context_payload = OrderedDict([
                        ("invocation", {
                            "count": self.invocation_count,
                            "current_time": datetime.now(UTC).isoformat()
                        }),
                        ("account", dashboard),
                        ("market_data", market_sections),
                        ("instructions", {
                            "assets": self.assets,
                            "note": "Follow the system prompt guidelines strictly"
                        })
                    ])
```

Replace with:

```python
                    context_payload = OrderedDict([
                        ("invocation", {
                            "count": self.invocation_count,
                            "current_time": datetime.now(UTC).isoformat()
                        }),
                        ("account", dashboard),
                        ("market_data", market_sections),
                        ("instructions", {
                            "assets": self.assets,
                            "note": "Follow the system prompt guidelines strictly"
                        })
                    ])

                    # Inject compact options book summary when available so the
                    # perps agent can factor in the options desk's net exposure.
                    if self._latest_options_context is not None:
                        ctx = self._latest_options_context
                        context_payload["options_book_summary"] = {
                            "vol_regime": ctx.vol_regime,
                            "portfolio_greeks": ctx.portfolio_greeks,
                            "open_position_count": ctx.open_position_count,
                        }
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "from src.backend.bot_engine import TradingBotEngine; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/backend/bot_engine.py
git commit -m "feat: inject options book summary into perps LLM context"
```

---

### Task 9: Document options scheduler in `.env.example`

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Add the options scheduler section to `.env.example`**

Append after the `THALEX_DELTA_THRESHOLD` line (line 60) and before the `# Runtime` section (line 62):

```env

# --- Options Scheduler ---
# Set to 1 to enable the two-cadence options loop alongside the 5m perps loop.
# When enabled, the bot runs a 15m vol surface refresh and a 3h options
# decision cycle via the OptionsAgent (separate LLM pipeline).
# OPTIONS_SCHEDULER_ENABLED=0
# Vol surface refresh interval in seconds (default: 900 = 15 min)
# OPTIONS_VOL_SURFACE_INTERVAL_SECONDS=900
# Options decision cycle interval in seconds (default: 10800 = 3 hours)
# OPTIONS_DECISION_INTERVAL_SECONDS=10800
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add OPTIONS_SCHEDULER_ENABLED and cadence vars to .env.example"
```

---

### Task 10: Final verification — run full test suite

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `pytest tests/ -v --tb=short 2>&1 | tail -40`

Expected: all tests PASS, no regressions.

- [ ] **Step 2: Verify the expanded prompt token budget**

Run a quick token estimate:

```bash
python -c "
from src.backend.agent.options_agent import _OPTIONS_SYSTEM_PROMPT
words = len(_OPTIONS_SYSTEM_PROMPT.split())
chars = len(_OPTIONS_SYSTEM_PROMPT)
lines = _OPTIONS_SYSTEM_PROMPT.count(chr(10))
est_tokens = int(words * 1.3)
print(f'Lines: {lines}, Words: {words}, Chars: {chars}, Est tokens: ~{est_tokens}')
assert est_tokens < 5000, f'Prompt too large: ~{est_tokens} tokens (budget: 5000)'
print('Token budget OK')
"
```

Expected: ~2500-3500 estimated tokens, well under 5000 budget.

- [ ] **Step 3: Spot-check the prompts.log won't contain closable anymore**

```bash
python -c "
positions = [
    {'symbol': 'BTC', 'venue': 'hyperliquid', 'closable': True, 'row_id': 'x'},
    {'symbol': 'BTC-25APR26-55000-P', 'venue': 'thalex', 'closable': False, 'row_id': 'y'},
]
cleaned = [{k: v for k, v in p.items() if k not in ('closable', 'row_id')} for p in positions]
assert 'closable' not in str(cleaned)
assert 'row_id' not in str(cleaned)
print('Cleaned positions:', cleaned)
print('UI field stripping OK')
"
```

Expected: positions printed without `closable` or `row_id` keys.
