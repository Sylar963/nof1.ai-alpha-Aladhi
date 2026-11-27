# Phase 1 AI trading launch prompt

Implement Phase 1, AI Trading, of this repository's nine-phase roadmap. This is
an implementation request, not a request for another high-level plan.

Read `AGENTS.md`, `requirements.txt`,
`.agents/skills/nof1-stack-and-testing/SKILL.md`, `docs/roadmap/README.md`,
`docs/roadmap/STATUS.md` and `docs/roadmap/phase-01-ai-trading.md` first.
Verify that Phase 1 is the active phase. Refresh the baseline audit against
current code, then implement its work packages in order. Reuse working code.

Deliver:

1. A traceable, causal technical/market/news-to-decision pipeline using real
   sources, explicit freshness and degraded states, and an honest inventory of
   input values and what actually reaches the model. Do not pad features or
   fabricate news to claim thousands of parameters. Any required claim rewording
   must be explicit in the user review.
2. Reliable auto/manual long/short entry, increase, reduction, close and flip
   behavior, including partial fills, idempotent retries, stale proposals and
   restart reconciliation.
3. Code-enforced dynamic TP/SL and fee/slippage-aware RR policy, with protection
   bound to confirmed fills and a tested response to protection failures.
4. Consistent venue/account leverage, gross exposure and risk settings across
   UI/configuration/execution. Support the advertised upper bound of 10x only
   where all applicable limits permit it; do not raise defaults or activate live
   leverage to prove support.
5. Conservative, Moderate and Aggressive profiles with observable numerical
   effects and hard bounds. A profile label or different prompt is insufficient.
6. Bounded, persisted performance adaptation from reconciled completed outcomes.
   Demonstrate a loss sequence reducing effective risk/leverage, no change under
   insufficient evidence, and controlled recovery. Recent-history context is not
   model training or automatic model switching.

Keep all existing protection and venue safety checks. Use deterministic tests,
causal replay and authorized paper/testnet/read-only integration evidence. Do not
start production orders, read secrets into outputs, rewrite the application in
Rust, or implement Phases 2 through 9. Necessary Phase 1 foundations are allowed;
optional later product capabilities are not.

Stay on the currently checked-out branch. Do not create/switch branches or
worktrees. Commit, push or open a PR only if I request that in this task.
Ask only for material missing credentials or unresolved constraints after checking
the repository, and continue independent authorized work while waiting.

Maintain `docs/roadmap/evidence/phase-01/` with a claim ledger, acceptance checklist,
test/integration results, demo steps, measured latency/cost and known limitations.
Update `STATUS.md` as work progresses. Do not stop at an audit if implementation
can proceed. Do not claim missing integration evidence has passed.

When all Phase 1 acceptance items except user approval are satisfied, mark it
READY FOR USER REVIEW and present the evidence. Otherwise report precisely what
remains incomplete. Do not mark Phase 1 accepted. Do not start Phase 2 until I
explicitly say Phase 1 is done and load/request Phase 2. Passing tests or merging
a PR does not grant that approval.
