# Phase 12 asset correlation launch prompt

Implement only Phase 12, Asset Correlation Analysis.

Read `AGENTS.md`, `requirements.txt`, the stack/testing skill, both roadmap
READMEs and status files, and extended `COVERAGE.md`. Require recorded explicit
user acceptance of original Phases 1–9, R0 and Phases 10–11, R0 PASS and unlocked
Phase 12. This prompt does not bypass missing approvals.

Audit accepted portfolio, return, market-data and risk services. Define return
types, aligned observation calendars, supported universe, windows and uncertainty.
Implement rolling correlations, covariance and usable concentration/cluster views
with sample counts, freshness and missing-data states. Reuse existing allocation
and risk interfaces rather than adding independent exposure controls.

Handle asynchronous prices, gaps, changing universe, low samples, singular or
unstable matrices and regime shifts. Do not fill from the future or treat stale
prices as independent evidence. Distinguish descriptive correlation from causal
or predictive claims.

Validate known positive/negative/independent scenarios and held-out causal replay.
Show an actual risk/allocation use with a documented unavailable-data policy,
plus UI/CLI evidence. Do not promise guaranteed diversification or improved returns.

Preserve all accepted controls, stay on the current branch and use authorized
offline/replay/paper/testnet validation. No live policy activation, branch/worktree
change, commit, push, PR or Phase 13 work is implied.

Write the claim ledger, metric definitions, tests, measurement outputs and demo
under `docs/roadmap/extended/evidence/phase-12/`; update extended `STATUS.md`.
Use READY FOR USER REVIEW only when all required work/evidence is complete;
otherwise report remaining gaps. Wait for my explicit Phase 12 acceptance and
request/load of Phase 13. Never accept this phase yourself.
