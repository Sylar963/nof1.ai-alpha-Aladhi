# Keltner-Volatility Position Sizing & Scaling Execution Framework

> **Status:** design / deferred. Not yet implemented. See "Architecture & Files" for the integration map when this gets picked up.

## Context

The bot currently delegates **100% of position-sizing to the LLM**, bounded only by a hard margin cap (`max_new_notional_by_asset` in `src/backend/bot_engine.py:336`). This produces noisy, regime-blind sizing: when BTC's Keltner channel is wide (post-news, high realized vol), the LLM has no deterministic guide rail to trim risk, and when channels compress, it lacks an explicit signal to lean in.

Likewise, entries and exits today are **single-shot** — `TradeProposal` only has `pending | approved | rejected | executed | failed` states (`src/backend/models/trade_proposal.py:44`), with no notion of multi-stage scaling. The LLM either takes the full position or doesn't, and exits are equally binary.

This design introduces:

1. A **deterministic risk engine** that turns Keltner-channel width + ATR into a regime-aware dollar-risk-normalized position size. The LLM still proposes direction and intent, but the engine enforces a maximum size envelope.
2. A **multi-stage scaling executor** that breaks an entry into 2–3 tranches triggered by price-to-Keltner-envelope interactions, and exits into 2 tranches at envelope extremes.

Both are gated behind feature flags so the existing single-shot, LLM-only path keeps working. Perps only; options multi-tenor logic (commit `52ed70d`) is orthogonal and untouched.

---

## Part 1 — Mathematical Framework

### 1.1 Volatility metric

Keltner width is already a linear function of ATR (`upper = EMA + 4·ATR`, `lower = EMA − 4·ATR`, so `width = 8·ATR`). To make it usable across BTC/ETH/SOL we work in **width-percent**:

```
width_pct = (upper - lower) / middle
```

Computed in `compute_keltner_width()` (new helper in `indicator_engine.py`) on the **5-min** interval, using the same Keltner series already built at `indicator_engine.py:142`.

### 1.2 Regime classification

Rolling percentile of `width_pct` over `RISK_WIDTH_LOOKBACK` bars (default 200 ≈ 16h on 5m):

| Regime       | Condition                       | Intent           |
|--------------|---------------------------------|------------------|
| `compressed` | `width_pct < p25`               | Lean in          |
| `normal`     | `p25 ≤ width_pct ≤ p75`         | Baseline         |
| `expanded`   | `width_pct > p75`               | Trim             |

The regime label is also surfaced to the LLM (item 4 below) so its qualitative reasoning stays aligned.

### 1.3 Base sizing (Van Tharp R-multiple)

```
risk_dollars     = equity × RISK_PER_TRADE_PCT          # default 0.5%
stop_distance    = RISK_STOP_K × ATR                    # default K = 2.0
base_units       = risk_dollars / stop_distance
base_notional    = base_units × price
```

ATR appears in the denominator → identical dollar-risk on a K·ATR move regardless of asset. BTC at $60k with ATR=$600 and SOL at $200 with ATR=$8 both consume the same `risk_dollars` for a stop-out.

### 1.4 Volatility multiplier

```
mult(width_pct) = clip( (width_p50 / width_pct) ** α,  m_min,  m_max )
```

- α = `RISK_VOL_ALPHA` (default 1.0; α<1 gentler, α>1 more aggressive)
- `m_min` = `RISK_VOL_MULT_MIN` (default 0.5)
- `m_max` = `RISK_VOL_MULT_MAX` (default 1.5)

Width *above* median → mult < 1 → trim. Width *below* median → mult > 1 → lean. Symmetric and bounded.

### 1.5 Final notional

```
target_notional  = base_notional × mult(width_pct)
final_notional   = min( target_notional,
                        max_new_notional_by_asset,         # existing margin cap
                        llm_allocation_usd )               # LLM still gets a veto on max
```

The LLM-proposed `allocation_usd` becomes a ceiling, not the source of truth. This lets the LLM signal "I'm not confident, go small" without letting it accidentally oversize.

---

## Part 2 — Multi-Stage Scaling Execution

### 2.1 State machine (per asset, per direction)

```
IDLE ──► STAGE_1_FILLED ──► STAGE_2_FILLED ──► STAGE_3_FILLED ──► PARTIAL_EXIT_1 ──► PARTIAL_EXIT_2 ──► EXITED
                                                                                   ▲                  ▲
                                                  stop / opposite envelope break ──┴──────────────────┘
```

State is persisted in a new SQLAlchemy model `ScalingState` (`src/database/models.py`) keyed by `(venue, asset, direction)`, with columns: `state`, `entry_price`, `entry_atr`, `entry_width_pct`, `total_target_notional`, `filled_notional`, `parent_proposal_id`, `updated_at`.

### 2.2 Entry stages (long; short is mirrored)

Let `N = final_notional` from Part 1, `m` = Keltner middle, `u`/`l` = upper/lower, `ATR` snapshot at entry.

| Stage | Trigger condition                                                                    | Size       |
|-------|--------------------------------------------------------------------------------------|------------|
| 1     | LLM emits a long decision, regime ≠ `expanded` (or override flag set)                | 50% of N   |
| 2     | Price within `SCALING_MID_TOL_ATR × ATR` of middle **and** close > middle in last 3 bars (trend intact) | 30% of N   |
| 3     | Price retests lower-half region (between middle and `middle − 0.5·(middle − l)`) without breaching `l` on close | 20% of N   |

If price closes **below `l`** at any point before stage 3 completes, abort remaining stages — the trend thesis is broken.

### 2.3 Exit stages

| Stage   | Trigger                                          | Size of remaining position |
|---------|--------------------------------------------------|----------------------------|
| TP1     | High touches `u`                                 | 40%                        |
| TP2     | Close > `u + 0.5·ATR` (overextension)            | 30%                        |
| Trail   | Close < middle for 2 consecutive bars            | remainder                  |
| Stop    | Close < `entry_price − RISK_STOP_K × entry_ATR`  | full remainder (hard stop) |

### 2.4 Pseudocode

```python
def scaling_tick(asset, snapshot, state):
    p, mid, up, lo, atr = snapshot.close, snapshot.kelt_mid, snapshot.kelt_up, snapshot.kelt_lo, snapshot.atr

    # Entry side
    if state.state == "STAGE_1_FILLED" and trend_intact(state, snapshot):
        if abs(p - mid) <= cfg.MID_TOL_ATR * atr and last_n_closes_above(snapshot, mid, 3):
            emit_child_proposal(state, pct=0.30, reason="stage_2_midline_pullback")
            state.state = "STAGE_2_FILLED"

    elif state.state == "STAGE_2_FILLED" and trend_intact(state, snapshot):
        retest_floor = mid - 0.5 * (mid - lo)
        if retest_floor <= p < mid and snapshot.low > lo:
            emit_child_proposal(state, pct=0.20, reason="stage_3_lower_retest")
            state.state = "STAGE_3_FILLED"

    # Exit side
    if state.state in ENTRY_FILLED_STATES:
        if snapshot.high >= up and not state.tp1_done:
            emit_partial_close(state, pct=0.40, reason="tp1_upper_envelope")
            state.tp1_done = True
        if p > up + 0.5 * atr and not state.tp2_done:
            emit_partial_close(state, pct=0.30, reason="tp2_overextension")
            state.tp2_done = True
        if closes_below(snapshot, mid, 2):
            emit_full_close(state, reason="trail_midline_break")
            state.state = "EXITED"
        if p < state.entry_price - cfg.STOP_K * state.entry_atr:
            emit_full_close(state, reason="hard_stop")
            state.state = "EXITED"

    # Abort on opposite envelope break
    if state.state in PARTIAL_ENTRY_STATES and snapshot.close < lo:
        cancel_pending_entry_stages(state)
        state.state = "STAGE_1_FILLED"  # freeze additions; exit side still active
```

`trend_intact(state, snapshot)` = `snapshot.close > state.entry_price - 1.0 × state.entry_atr` AND `snapshot.kelt_mid_slope >= 0` for longs.

`emit_child_proposal` / `emit_partial_close` create new `TradeProposal` rows with `parent_proposal_id` set and `stage` populated — these flow through the existing approval+execution path, so the margin preflight at `bot_engine.py:347` still gates every tranche.

---

## Part 3 — Architecture & Files

### 3.1 New files

| Path                                              | Purpose                                                                |
|---------------------------------------------------|------------------------------------------------------------------------|
| `src/backend/trading/risk_engine.py`              | Pure functions: `compute_width_pct`, `regime`, `volatility_multiplier`, `suggest_notional`. No I/O. |
| `src/backend/trading/scaling_executor.py`         | `ScalingExecutor` class; owns the per-asset state machine; called once per perps cycle from `bot_engine._main_loop`. |
| `tests/test_risk_engine.py`                       | Property tests for sizing math (monotonicity in width, ATR, equity).   |
| `tests/test_scaling_executor.py`                  | State-transition tests with synthetic snapshots driving the FSM through all paths. |

### 3.2 Modified files

| Path                                              | Change                                                                 |
|---------------------------------------------------|------------------------------------------------------------------------|
| `src/backend/indicators/indicator_engine.py`      | Add `compute_keltner_width(series)` → list of widths; surface `width`, `width_pct`, `width_p25/p50/p75`, `vol_regime` in `build_indicator_bundle` (around line 285). |
| `src/backend/bot_engine.py`                       | (a) In `_keltner_snapshot` (line 1666) include width fields. (b) After LLM decision and before existing margin preflight (line 347), call `risk_engine.suggest_notional(...)` and clamp `allocation_usd`. (c) Each cycle, call `scaling_executor.tick(asset, snapshot, state)` to emit child proposals. |
| `src/backend/agent/decision_schema.py`            | Add optional `scaling_intent: Literal["single_shot", "staged_entry"]` on `TradeDecision` (default `"single_shot"`). |
| `src/backend/agent/decision_maker.py`             | System prompt: surface `vol_regime` and `width_pct`; instruct the LLM that `allocation_usd` is a *ceiling*, not the executed amount, and that staged entries fill across multiple bars. |
| `src/backend/models/trade_proposal.py`            | Add `stage: int = 0`, `total_stages: int = 1`, `parent_proposal_id: Optional[UUID] = None`, `trigger_reason: Optional[str] = None`. |
| `src/database/models.py`                          | New `ScalingState` table (columns listed in §2.1). New columns on `TradeProposal` ORM to mirror dataclass. |
| `src/backend/config_loader.py` + `.env.example`   | New keys (defaults below).                                             |

### 3.3 Config keys

```
RISK_ENGINE_ENABLED=1
RISK_PER_TRADE_PCT=0.5
RISK_STOP_K=2.0
RISK_VOL_ALPHA=1.0
RISK_VOL_MULT_MIN=0.5
RISK_VOL_MULT_MAX=1.5
RISK_WIDTH_LOOKBACK=200

SCALING_ENABLED=1
SCALING_STAGE_1_PCT=50
SCALING_STAGE_2_PCT=30
SCALING_STAGE_3_PCT=20
SCALING_MID_TOL_ATR=0.5
SCALING_TP1_PCT=40
SCALING_TP2_PCT=30
SCALING_STOP_K=2.0
SCALING_TRAIL_BARS=2
```

Both `*_ENABLED` flags default-on once code lands, but every cycle short-circuits to current behavior if either is `0`. That guarantees the existing path is reachable for A/B.

### 3.4 Integration order (each cycle)

```
1. fetch snapshot                          (existing)
2. compute indicators incl. width/regime   (modified)
3. LLM decision                            (existing)
4. risk_engine.suggest_notional → clamp    (NEW, gated by RISK_ENGINE_ENABLED)
5. scaling_executor.plan_entry             (NEW, gated by SCALING_ENABLED) → produces 1..3 staged proposals
6. margin preflight                        (existing — runs per proposal)
7. submit                                  (existing)
8. scaling_executor.tick                   (NEW) — advance state, emit follow-on proposals
```

---

## Part 4 — Verification

1. **Unit:**
   - `test_risk_engine.py`: assert `suggest_notional` is monotonic decreasing in `width_pct` (held `equity`/`atr`/`price` fixed), monotonic increasing in `equity`, dollar-risk constant across asset prices when ATR is scaled.
   - `test_scaling_executor.py`: walk a synthetic OHLC sequence (pullback to mid → bounce → tag upper → overextension → trail break) and assert exactly the right child proposals are emitted in the right order, with size totals = 100%.
2. **Integration:** existing `tests/test_decision_schema.py` + `test_bot_engine_retry.py` continue to pass with both flags off; new fixtures with flags on cover the staged path.
3. **Backfill replay:** `python -m scripts.replay_risk_engine --asset BTC --days 30` (small new script) runs the risk engine over historical 5m bars from the DB and prints sized-vs-LLM-only notionals — sanity check that the multiplier band stays in [0.5, 1.5].
4. **Paper-mode smoke:** flip `RISK_ENGINE_ENABLED=1` only (scaling off) for 24h, confirm sizing tracks regime; then enable scaling for another 24h, confirm child proposals appear in `recommendations` page with correct `stage` labels.

---

## Out of scope

- Options-side sizing changes (Thalex 3-hour cycle untouched).
- Cross-asset correlation budgeting.
- Per-asset α tuning — single global α to start.
- Backtest harness beyond the replay script above.
- LLM-driven dynamic α (LLM sees regime + width_pct but does not pick coefficients).

---

## Alternatives considered (briefly)

- **Pure-LLM sizing with regime as a hint only** — rejected: leaves a deterministic safety floor on the table; the whole point is a guard rail.
- **Single-stage with regime-only sizing** — rejected: solves Part 1 but not Part 2; user explicitly asked for envelope-driven scaling.
- **New `ScalingProposal` parallel model** — rejected in favor of extending `TradeProposal` with `parent_proposal_id`; keeps the GUI/history/approval flow uniform.
