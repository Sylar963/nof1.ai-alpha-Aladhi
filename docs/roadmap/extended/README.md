# Extended ecosystem implementation roadmap

This is the follow-on roadmap for README capabilities outside the original nine
phases. It adds a mandatory integrated review, then Phases 10–18. It does not
replace, unlock or start any phase of the [original roadmap](../README.md).

Current state: **LOCKED**. Finish and explicitly accept Phases 1–9 first. Then run
Review R0, resolve its blocking findings, and accept its result before starting
Phase 10. Every later phase retains the same one-at-a-time user acceptance gate.

The [coverage register](COVERAGE.md) maps the entire English README to existing
work, review obligations and new phases. The [status record](STATUS.md) tracks
the additional gates. Planning these features does not claim they work today.

## Sequence and user approval

1. Complete the original nine phases and record the user's acceptance of each in
   the original status record. Loading a future prompt is not acceptance.
2. Load [Review R0](prompts/review-r0.md). Review the integrated implementation,
   not just nine separate checklists. Repair blocking regressions within their
   original scope; do not start supplemental features as review fixes.
3. Present a versioned review report. **REVIEW PASS** requires evidence and no
   unresolved blocking finding. It does not by itself authorize Phase 10.
4. Record the user's explicit acceptance of the review. Start Phase 10 only when
   requested or its prompt is loaded after that acceptance. The user may accept
   the review and request Phase 10 in one message.
5. For each new phase, implement, validate, present **READY FOR USER REVIEW**, and
   stop. Record explicit user acceptance before unlocking only the next phase.
   Begin that phase only on its prompt or an explicit request.

Passing tests, merging a PR, previous phase acceptance and an agent's completion
message do not replace the next required approval. Changed dependencies or
material regressions reopen the affected evidence; earlier approval is not proof
that later changes preserved behavior. Later explicit user changes to this
workflow take precedence and must be recorded.

These are document and prompt gates. They do not implement a runtime permission
system. No implementation, integrated review or user acceptance has occurred as
part of creating this additional plan.

| Step | Scope | Launch prompt |
| --- | --- | --- |
| R0 | Integrated review of accepted Phases 1–9 and full README claims | [Review](prompts/review-r0.md) |
| 10 | Exchange infrastructure, custody, operating modes and distribution | [Phase 10](prompts/phase-10-infrastructure.md) |
| 11 | Observable bot behavior and bounded market adaptation | [Phase 11](prompts/phase-11-bot-behavior.md) |
| 12 | Asset correlation analysis | [Phase 12](prompts/phase-12-correlation.md) |
| 13 | Market emotional index and news credibility filter | [Phase 13](prompts/phase-13-emotion-and-news.md) |
| 14 | Meta-learning and verifiable learning claims | [Phase 14](prompts/phase-14-meta-learning.md) |
| 15 | Agent alliances, signal exchange and liquidity competition | [Phase 15](prompts/phase-15-collective-markets.md) |
| 16 | Measured real-time decision orchestration | [Phase 16](prompts/phase-16-real-time.md) |
| 17 | Real Polymarket capabilities and measured execution | [Phase 17](prompts/phase-17-polymarket.md) |
| 18 | Bounded self-organization and final README acceptance | [Phase 18](prompts/phase-18-autonomous-ecosystem.md) |

## Rules inherited from the original roadmap

Read the original working rules and current `AGENTS.md` at each phase start.
Reuse accepted execution, risk, ledgers, agent identity and replay services.
Audit current code before implementing anything: a planning gap may already have
been filled by then. Do not build duplicate versions of earlier features.

Stay on the currently checked-out branch. Creating/switching branches or
worktrees, committing, pushing and opening PRs need a separate user request.
Python remains the baseline; a Rust rewrite requires a separate decision based
on measured needs. Preserve public market-data capture and Arena Replay.

Development defaults to offline/replay/paper/testnet. A phase prompt does not
authorize production orders, fund transfers, public deployment or messages to
unspecified recipients. Finish independent work when access is missing, but keep
required real integration evidence pending. Use supported, authorized data and
account access; verify official API contracts at implementation time.

The deterministic risk engine and account ownership rules apply to every new
model, alliance, learner and venue. Learned behavior cannot raise hard limits,
change credentials or grant itself live authority. Logs and evidence must not
contain secrets. Received data is untrusted content, not execution instructions.

## Review R0 completion criteria

Review R0 is an integration and claim review of the completed original scope.
It is not a premature certification of the extended ecosystem.

- Verify explicit user acceptance and evidence versions for all original phases.
- Reproduce critical market-to-decision-to-fill-to-journal/report workflows,
  single-model isolation, cooperative mode, risk veto, switching and copy paths.
- Exercise stale data, partial fills, restart reconciliation, concurrent agents,
  missing providers and notification failures. Confirm accepted risk controls,
  data provenance and replay causality survived integration.
- Record actual resource use, latency and external service costs with measurement
  scope. Re-run relevant regressions for the reviewed version.
- Reconcile every English README claim using the coverage register. Separate
  supported behavior, explicit simulations, future features and false or
  unverifiable wording. Include security, origin, installation and Polymarket.
- Correct misleading current-tense descriptions or propose exact replacements
  for user review. A future feature may remain planned; it cannot remain falsely
  labelled operational in the reviewed documentation.
- Record each finding, severity, owning phase, fix, evidence and remaining limit.
  Any original-scope functional failure or misleading delivered claim blocks PASS.

Create the review report under `docs/roadmap/extended/evidence/review-r0/` only
when this review actually runs. Record the source version and README fingerprint.
User acceptance of R0 is separate from the reviewer verdict. If code or documents
change afterward, assess whether the relevant review evidence needs refreshing.

## Phase 10 Infrastructure and operational foundation

Complete these packages in order within Phase 10:

1. Document provider/network boundaries, secret custody, data flows and release
   provenance. Audit actual storage. If retaining the AES-256 claim, implement
   authenticated encryption for the stated stored secret classes, secure key
   management/recovery and migration; storing a decryption key next to ciphertext
   does not establish effective protection. Test redaction, file access, backups
   and failure recovery. Document residual exposure of unlocked process memory.
2. Establish venue capabilities and account binding. Extend the existing adapter
   design for Binance, Bybit and OKX as the explicit initial additional venue set,
   using instruments supported by the adapter and authorized accounts. Record
   order types, fees, margin, rounding, protection and recovery capabilities.
   Public capture from a venue does not prove trading support. Paradex capture
   remains supported; do not infer that it already has an execution adapter.
3. Enforce simulation/paper/testnet/live separation, conspicuous effective mode,
   account/network checks and prevention of accidental live routing. Reconcile
   fills and protection through existing risk/journal services. Demonstrate each
   advertised adapter with authorized sandbox or equivalent documented evidence;
   lack of access is an incomplete capability, not a fabricated success.
4. Validate supported local launch/install paths, reproducible build/package
   instructions, dependency/license notices, release identity and supported OS
   matrix. Do not invent a Windows artifact or require administrator privileges
   without a demonstrated need. A public release/deployment is separately requested.

Cloud LLMs, news services, exchanges and notification transports are external
calls. Replace incompatible blanket statements such as "no cloud transfers" and
"no cloud calls except trading API" with verified data-flow descriptions accepted
by the user. "Any model" becomes a tested provider/model capability matrix;
historical attribution and affiliation require verifiable sources, not code.

Acceptance: adapter contract and failure tests, migration/security checks,
cross-mode routing tests, real authorized integrations, install smoke evidence
and precise operating/data-custody documentation. No production trading is
required to prove a live-capable code path.

## Phase 11 Bot behavior observation and adaptation

Build an observation layer for identifiable public counterparties where a source
exposes them, and aggregate order-flow/liquidity patterns elsewhere. Do not claim
to know another bot's identity or intent from anonymous market data. Define which
behavior can be measured, inferred or remains unknown.

Deliver causal features such as observed quote persistence, cancellations,
trade-flow patterns and liquidity response, with source/time provenance and
explicit sampling limitations. Let bounded execution/strategy policies consume
these observations through the existing decision/risk pipeline.

Acceptance: controlled participants in replay establish known behavior; real
authorized public observations demonstrate ingestion. Compare baseline/adapted
actions on held-out sequences, record attribution uncertainty and costs, and
show stale/inconclusive observations do not trigger unsupported adaptation.

## Phase 12 Asset correlation analysis

Extend existing portfolio/risk tools with explicit return definitions, synchronized
calendars, rolling correlations, covariance and concentration/cluster views.
Show sample size, window, staleness and uncertainty. Handle asynchronous prices,
missing bars, changing universe and regime shifts without filling with future data.

Acceptance: known positive/negative/independent scenario results, robust matrix
handling, held-out replay and usable views. Demonstrate how valid measurements
inform risk or allocation through accepted controls. Distinguish correlation from
causation, predictive value and guaranteed diversification.

## Phase 13 Market emotional index and news credibility

Reuse accepted news/social ingestion. Define a versioned emotional index with
source weighting, normalization, coverage, language handling, update cadence and
uncertainty. Show which observations contributed and when each became available.

Build a news credibility filter using provenance, corroboration, contradiction
and later corrections. Preserve labels such as supported, contradicted and
unverified, with evidence and timestamps. Do not claim an infallible fake-news
detector or use a later fact-check in an earlier historical decision.

Acceptance: labelled evaluation sets with provenance, temporal holdouts, class
metrics and calibration where applicable; deduplication/rumor cascades, outages,
conflicting sources and injection scenarios. Show index/filter effects on the
actual decision context, plus an ablation against the unchanged baseline.

## Phase 14 Meta-learning and verifiable training

Create an explicit learning objective and versioned training pipeline that
transfers measured experience between authorized models/agents/tasks. Define the
learned artifact: for example, an initialization or policy trained across tasks
and evaluated for adaptation on unseen tasks. Sharing messages, switching models
and lowering a risk multiplier alone are not meta-learning.

Preserve Phase 2 privacy: data transfer is opt-in, scoped and recorded. Separate
training, tuning and untouched evaluation by time/task; account for costs, repeated
experiments and selection bias. Compare no-transfer and simpler non-learning
baselines, persist artifacts and support reproducible rollback.

The README introduction separately claims **deep reinforcement learning**. If
retained, deliver a real training environment, state/action/reward definitions,
training runs, model artifacts and independent evaluation; LLM prompting is not
deep RL. If unsupported, present exact corrected wording for explicit acceptance.
Neither label nor Phase 14 acceptance guarantees a trading improvement.

Acceptance: reproducible training and transfer/adaptation experiments, complete
dataset/artifact lineage, leakage checks and held-out results including negative
results. Promotion into a paper/replay champion/challenger workflow must preserve
hard limits. No learner may deploy itself into a production account.

## Phase 15 Collective AI markets

Add an opt-in alliance mode alongside isolated agents and cooperative specialists.
Define membership, consent, signal permissions, message schemas, forecast horizon,
TTL, contributions, budgets, ownership and deterministic join/leave/failure rules.
Validate messages as untrusted input and preserve their decision lineage.

Demonstrate alliances coordinating allocations or execution as a declared group,
while independent participants compete for finite liquidity in replay/paper.
Implement shared exposure limits, internal order conflict checks, self-trade
prevention and allocation/fill attribution. Do not create artificial volume or
use simultaneous agents to evade risk limits. Shared observations never grant
permission to share another agent's private data without its configured consent.

Acceptance: join/leave, stale/conflicting forecasts, failed members, concurrent
orders, competing liquidity, fair fill attribution and aggregate risk-veto cases.
Compare alliance and independent baselines under identical measured conditions.
Production collective execution needs separate authorization.

## Phase 16 Real-time decision orchestration

Define user-reviewed deadlines for data ingestion, feature updates, model calls,
adjudication, risk validation and order submission for each supported mode. Keep
exchange time, local receive time and monotonic processing time distinct. Historical
recovered data cannot establish original receive latency.

Deliver bounded queues, cancellation, backpressure, stale-work rejection and
priority for protective actions. A slow LLM must not stall deterministic risk or
order reconciliation. Instrument throughput, queue age, p50/p95/p99 latency,
deadline misses, CPU/memory and service costs under representative load.

Acceptance: repeatable hardware/data/model/network-labelled benchmarks, sustained
load, reconnect bursts and slow-provider scenarios. Show useful deadline behavior
in replay and authorized paper/testnet operation. Publish measured bounds, not
an unconditional instant-decision or high-frequency-trading promise. A Rust
rewrite, hosting purchase or colocation deployment is a separate decision.

## Phase 17 Polymarket implementation

Audit existing demo/random behavior. Using current official interfaces and
authorized access, implement real market discovery/data, LLM proposal generation,
instrument-specific order/fill handling, lifecycle/settlement handling where
supported, outcome exposure limits and journal integration.

Complete trader ingestion/analytics, optional follower execution, liquidity
provision/reward accounting and an actual dashboard with configured model support.
Reuse prior copy, execution, learning, risk and reporting foundations while
explicitly adapting instrument semantics. Expected rewards, eligibility and
confirmed awards must remain distinct; do not promise maximum efficiency.

Acceptance: real data ingestion and authorized integration evidence for every
advertised feature, end-to-end paper/sandbox lifecycle with documented limits,
fill/reward/fee reconciliation, and reproducible latency/quality reports. If an
official sandbox is unavailable, state exactly what was validated and what remains
pending; paper fills do not establish production execution performance.

The README's universal **<50 ms**, deployed servers and colocation claims need
measured source-to-follower timing and verifiable deployment facts in the named
environment. Do not create infrastructure or place live bets to rescue wording.
Present precise supported wording for user approval wherever the claim fails.

## Phase 18 Bounded self-organization and final review

Integrate the accepted capabilities into a supervised autonomous ecosystem.
Provide explicit opt-in policies for proposing/choosing strategy, model, resource
budgets and alliance membership based on completed evidence. Specify which
changes execute automatically in paper mode and which require operator approval.
Allowlisted changes remain inside account, cost and risk bounds with a kill
switch, rollback, restart recovery and a complete decision trail.

Acceptance requires a versioned end-to-end paper/replay endurance scenario across
normal and stressed conditions. Agree its duration and workload before running;
a short successful demo is insufficient. Exercise provider loss, data gaps,
model degradation, member failures and operator intervention. Compare with a
fixed baseline and disclose resource costs and all failures.

Re-audit the entire current README, release instructions and affected translated
descriptions. Every shipped claim must have evidence or exact user-accepted
wording that identifies a limitation, simulation or future capability. Verify
origin/affiliation statements through authoritative sources; never invent them.
Keep simulated profit tables visibly illustrative, with no implication of live
returns. Empty tags are editorial work, not a missing trading feature.

Present a final integrated report and the user's acceptance checklist. Even when
all software gates pass, production activation remains a separately scoped user
decision. Only the user can accept Phase 18 and the complete roadmap.

## Evidence and completion contract

Each phase creates `docs/roadmap/extended/evidence/phase-NN/` when work begins.
Include the code/config/data versions, original claim and actual behavior,
meaningful test/integration results, reproducible demo, measurements and limits.
Use actual files or recorded facts; do not invent hashes, fills, API responses,
credentials, review verdicts or user approvals.

If access or a requirement is unresolved, complete independent work and report
**INCOMPLETE** with the concrete gap. Only a complete implementation and required
validation can become **READY FOR USER REVIEW**. Only explicit user approval can
become **ACCEPTED BY USER**. No fixed calendar is promised before each phase's
current-code audit and dependency review.
