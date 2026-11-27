# Phase 1 AI trading implementation plan

This phase makes the first requested feature section a demonstrable product
capability. It is the only active implementation phase. Do not begin Phase 2
until the user has accepted Phase 1 and requested the next phase.

## Outcomes to prove

| Original claim | Phase 1 outcome and evidence |
| --- | --- |
| Automatic long/short opening and closing | Account-aware state transitions, correctly sized orders and reconciled fills in auto/manual and paper/testnet workflows. |
| Technical and news inputs, described as thousands of parameters | Timestamped real data, an honest feature/input inventory, recorded model-visible context and a measured input/token budget. No fake input counts or simulated news. |
| Dynamic TP/SL based on volatility and RR | A deterministic, versioned protection policy with cost-aware RR validation, venue rounding, partial-fill handling and failure recovery. |
| Leverage support up to 10x | One consistent effective policy across configuration, UI, proposals, adapters and actual account/venue constraints. No trading path can request exposure above its applicable limits. |
| Conservative, Moderate, Aggressive modes | Named profiles that actually change sizing and timing constraints, with enforced numeric bounds and visible effective settings. |
| Analysis of past results and automatic strategy adjustment | Completed outcomes feed a bounded, persisted adaptation policy; demonstrate a documented change such as lower effective risk/leverage after losses and controlled recovery. |

These are functional software outcomes, not guarantees of profit or a model's
ability to predict prices. A lower configured risk budget is not a guarantee
that losses stay below the planned amount during a gap or execution failure.

## Baseline to inspect again before coding

The preceding code audit found existing long/short execution, TP/SL order paths,
manual proposals, configurable risk caps, persisted drawdown/daily-loss tracking,
recent performance in the LLM prompt and optional Thalex/options support. Missing
or incomplete areas included real news ingestion, the advertised named behavior
profiles, a guaranteed code-enforced dynamic TP/SL policy, consistent leverage
configuration and a persisted outcome-driven adaptation controller.

Relevant starting points:

- `src/backend/bot_engine.py`: execution, protection, sizing, margin preflight,
  reconciliation and circuit breakers.
- `src/backend/agent/decision_maker.py`: perps context, decision schema usage,
  recent outcomes and current prompt-only guidance.
- `src/backend/config_loader.py`, `.env.example`, `src/gui/pages/settings.py`,
  `src/gui/services/bot_service.py`: configuration and effective settings.
- `src/backend/trading/hyperliquid_api.py`, `exchange_adapter.py`,
  `exchange_factory.py`: venue capabilities and order semantics.
- `src/backend/indicators/indicator_engine.py`: existing technical calculations.
- `src/database/models.py`, `db_manager.py`: fill/trade history, decisions and
  persisted policy state.
- `src/backtest/` and Arena Replay: causal validation and execution examples.
- Existing `tests/test_risk_engine.py`, `test_circuit_breaker.py`,
  `test_bot_engine_retry.py`, `test_decision_schema.py` and adapter tests.

The actual files and behavior may have changed since the audit. Confirm what is
already present, its callers and failure behavior; do not create duplicate engines.

## Work package 1 Claim contract and observability

1. Inventory each Phase 1 claim as implemented, partial, missing or simulated.
   Link the runtime path and meaningful evidence rather than a label or setting.
2. Define the decision record: input snapshot/version, input availability time,
   model identifier, prompt/policy version, proposed action, hard validation,
   effective risk/mode, resulting orders/fills and reason for rejection.
3. Define a shared clock interface for runtime and replay. A source's publish or
   matching time is not interchangeable with the time information became available.
4. Create the Phase 1 acceptance/evidence ledger under
   `docs/roadmap/evidence/phase-01/` when implementation starts. Do not check off
   implementation or integration evidence in this planning pass.

The phrase **thousands of parameters** needs an explicit claim contract. Count
engineered features, raw observations, assets/windows and model-visible inputs
separately. To retain the plural claim, prove at least 2000 meaningful distinct
input values are actually consumed in the claimed decision path, with no padding
or duplicate relabelling. If the pipeline summarizes those inputs before the LLM,
describe that architecture accurately instead of claiming the LLM read them all.
If the original wording cannot be supported efficiently, present the precise
replacement wording for the user's acceptance. Do not silently mark a weaker
claim as fulfilled.

## Work package 2 Real market and news context

1. Reuse validated technical, price, funding, open-interest and account inputs.
   Keep last-trade price, mark/index price and executable book liquidity separate.
2. Add a real news provider interface with a configured, supported initial source.
   Verify its current official contract and access requirements when implementing.
   This is baseline news context, not Phase 8's full social-media parser.
3. Store publisher/source IDs, publication time, ingest/availability time, raw
   provenance, normalized event text and deduplication identity. Treat content as
   untrusted data and prevent it from becoming execution instructions.
4. Build a bounded context with measured feature counts, token size, latency and
   cost. Avoid ballooning the prompt just to meet a marketing number.
5. Make required/optional inputs explicit. Missing/stale required inputs prohibit
   new exposure; optional feed loss creates an observable degraded state. Keep
   protective/reducing actions available under the defined execution policy.
6. Replay news only once it was available. Historical publication dates alone
   do not establish original local receipt times; label reconstructed availability.

## Work package 3 Reliable position lifecycle

1. Define valid transitions: flat to long/short, increase, reduce, close and flip.
   Reuse existing adapter contracts and protect manual approval semantics.
2. Give proposals and execution attempts stable identities. Handle retries,
   rejected orders, timeouts, late fills, partial fills and restart reconciliation
   without accidental duplicate exposure.
3. Revalidate manual proposals at approval: price drift, expiry, account state,
   mode/risk policy and available margin can have changed since generation.
4. Keep protection on existing exposure when replacing or closing orders. If an
   exchange requires a cancel/replace sequence, define and test the unprotected
   interval policy. Avoid blindly cancelling stops before knowing the next action.
5. Reconcile local ledgers to real adapter account/order/fill responses. Display
   rejected, pending and uncertain outcomes accurately; a submitted order is not
   a confirmed fill.

## Work package 4 Dynamic protection and RR

1. Choose and document a deterministic protection policy based on existing
   volatility/structure measures. Expose its parameters and version. LLM levels
   may be inputs to the policy, not permission to bypass it.
2. Validate side, tick/lot rounding, minimum stop distance and estimated RR after
   spread, fees and slippage assumptions. Reject invalid or uneconomic new
   exposure; do not calculate RR only for display.
3. Bind TP/SL size and price reference to confirmed fills. Handle partial fills,
   partial exits, additions, flips and cancellation/replacement explicitly.
4. Implement a bounded dynamic update policy, including trailing/tightening and
   when re-evaluation is allowed. Do not silently loosen a protective stop or
   increase risk beyond the accepted policy to avoid realizing a loss.
5. Handle protection-placement failure through a defined fallback, alert and
   retry/reduce/flatten policy. Exercise that policy with fakes and an authorized
   venue validation; do not report success on an unconfirmed stop.

## Work package 5 Leverage and behavior profiles

1. Consolidate the currently distinct GUI, configuration and execution concepts:
   venue leverage setting, gross exposure/equity, trade risk and margin capacity.
   Show their effective values rather than presenting them as interchangeable.
2. Support the advertised upper bound of 10x only where venue/instrument rules,
   account margin and configured risk permit it. Use the stricter applicable
   bound; a market with a smaller venue maximum remains restricted.
3. Do not raise existing risk defaults or place a leveraged trade merely to
   demonstrate support. Test 1x/3x/10x and rejection above limits with controlled
   scenarios and a venue-supported non-production validation.
4. Define Conservative, Moderate and Aggressive profiles as versioned, validated
   numerical settings: per-trade risk, gross exposure, holding/decision horizon,
   entry selectivity, RR floor and cooldown policy. More aggressive does not mean
   unlimited leverage, weaker protection or a promise to make short-term profits.
5. Preserve operator overrides only within the global/venue limits. Show the
   effective profile. A runtime mode change must not unexpectedly loosen existing
   stops, resize positions or invalidate pending approvals.

Profile numbers are to be derived from current configured limits and replay
evidence, not prescribed by this planning document as financial recommendations.
The code defaults observed in the audit are a starting point, not proof of the
user's active configuration.

## Work package 6 Bounded performance adaptation

1. Read reconciled completed outcomes, net of known fees/funding/costs. Separate
   strategy P&L from transfers, deposits and unrelated account activity.
2. Keep the LLM's analysis of its recent outcomes, but add a deterministic
   controller for allowed adjustments. Define minimum samples, loss-streak and
   drawdown triggers, risk/leverage multipliers, recovery conditions and cooldowns.
3. Bound every adjustment by the operator's settings, current profile and hard
   limits. Persist the state/version across restarts and show why it changed.
   No model output may edit its own hard constraints or deploy arbitrary code.
4. Demonstrate a losing sequence that lowers effective exposure/risk, a stable
   sequence that does not cause unnecessary changes, and a controlled recovery.
   Evaluate tuning rules on a separate held-out replay before using them.
5. State exactly what adapts. This phase does not claim neural-weight retraining,
   reinforcement learning, automatic model switching, or the Phase 3 model team.
   Genuine model training would need its own explicit objective, dataset,
   evaluation and deployment approval.

## Validation and rollout

Keep offline unit/scenario tests deterministic and credential-free. Add failure
tests that exercise behavior rather than reproducing implementation details.
Use recorded inputs and recorded model outputs for reproducible strategy tests;
validate real LLM integration separately, reporting provider/version/cost.

Proposed scenario matrix:

| Scenario | Required behavior |
| --- | --- |
| Valid flat/long/short lifecycle | Correct side, quantity, account ownership, fill ledger and protection. |
| Partial fill then retry/restart | No duplicate exposure; protection matches actual inventory. |
| Invalid stop or low estimated net RR | Reject new exposure with a recorded reason. |
| Required market/news input unavailable or stale | Defined unavailable/degraded state; no invented data. |
| News prompt injection or malformed model output | Treat as data/reject safely; no authority escalation. |
| Venue leverage below configured limit | Lower venue/account limit wins. |
| Request above 10x or profile/risk bounds | Reject/clamp under the documented policy before submission. |
| GUI/CLI/env settings and mode changes | Same effective constraints and persisted state; no cosmetic-only settings. |
| Consecutive completed losses | Bounded adaptation lowers effective risk and survives restart. |
| Minimum sample not reached or incomplete P&L | No unjustified adaptation based on unreliable evidence. |
| Protective order rejected or uncertain | Defined recovery with observable failure, not a success message. |
| Manual proposal approved after drift/expiry | Revalidate and reject stale intent where required. |
| Replay across timezone/session boundaries | No future data or use of the wall clock in historical decisions. |

After offline/replay tests, provide an authorized read-only provider smoke test,
paper/testnet lifecycle demonstration and an operator review. If real access is
unavailable, list that validation as pending instead of claiming it passed.
Production activation is a separate user action and not required merely to
write working implementation code.

## Phase 1 acceptance checklist

- [ ] Every original Phase 1 claim is mapped to concrete behavior or explicitly
  presented revised wording, with unresolved differences visible to the user.
- [ ] Real technical/market and news inputs have provenance, freshness handling
  and an honest inventory of computed and model-visible values.
- [ ] Auto/manual long/short entry, reduction, closure and flipping pass the
  lifecycle and failure scenario matrix.
- [ ] Dynamic TP/SL and a pre-trade net-RR policy are enforced in code and
  reconciled to actual fills and protection orders.
- [ ] Leverage, margin and gross exposure settings are consistent across UI,
  configuration and execution, including support/rejection at applicable bounds.
- [ ] All three behavior profiles have observable, tested effects and retain
  hard risk limits.
- [ ] Completed outcomes drive bounded, persisted and explainable adaptation;
  recent-history prompting alone is not counted as this controller.
- [ ] Replays and policy evaluation are causal and distinguish recovered history
  from original receive-time observations.
- [ ] Relevant existing and new tests pass; meaningful integration evidence,
  demo instructions, measured costs/latency and limitations are recorded.
- [ ] Documentation states what is implemented and what remains planned without
  claiming profitability, unsourced feature counts or model self-training.
- [ ] The implementation package is presented as **READY FOR USER REVIEW**.
- [ ] The user explicitly accepts Phase 1, with the exact approval recorded in
  `STATUS.md`. Only the user can satisfy this final item.

No Phase 2 implementation begins as part of this phase. A missing credential,
failed acceptance item or proposed wording change must remain visible; it is
not grounds to move on to another phase.
