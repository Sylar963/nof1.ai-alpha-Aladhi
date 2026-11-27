# Phase 4 intelligent risk management launch prompt

Implement only Phase 4, Intelligent Risk Management.

Read `AGENTS.md`, `requirements.txt`, the stack/testing skill,
`docs/roadmap/README.md` and `docs/roadmap/STATUS.md`. Verify explicit user acceptance
for every preceding phase and that Phase 4 is unlocked. Do not treat this prompt
as approval of incomplete prior work.

Audit the risk controls accepted in Phase 1 and the agent/cooperative paths added
in Phases 2 and 3. Refine and execute a concrete implementation plan. Extend the
existing risk engine instead of adding disconnected checks or a second source of
configuration truth.

Deliver code-enforced net RR, drawdown and daily-loss protection, volatility-based
sizing and trading/session cooldowns across the supported paths. Evaluate proposals
against current available capital, venue margin, account ownership, net/gross and
correlated exposures, options Greeks and hedges. Define timezones, session/day
boundaries, reset rules, pause/resume behavior and the effect of deposits/withdrawals.

Persist thresholds and baselines across restarts. Put deterministic limits before
order submission and revalidate manual approvals. Resolve concurrent proposals
against a consistent portfolio state. Risk pauses must suppress new exposure
without disabling the documented protective/reducing path. The model may explain
or propose risk but cannot bypass limits. Estimated stop loss is not a guarantee
of realized maximum loss through gaps or execution failures.

Prove the controls with simultaneous-agent scenarios, correlated instruments,
partial/late fills, stale data, unavailable margin, breached limits, transfers,
restart recovery and manual approval after drift. Show actual state transitions
and rejection reasons. A prompt telling the model to reduce size or wait is not
an acceptance result.

Use deterministic tests, causal replay and authorized paper/testnet integration.
Preserve earlier accepted behavior, stay on the current branch and do not begin
Phase 5 or later work. No production activation, branch changes, commits, pushes
or PRs are implied by this prompt.

Record the claim ledger, acceptance matrix and evidence in
`docs/roadmap/evidence/phase-04/`; update `STATUS.md`. Present READY FOR USER REVIEW
when all implementation and required validation items are complete, or clearly
report incomplete items. Only I may accept this phase. Start Phase 5 only after
that explicit acceptance and a request/load of its prompt.
