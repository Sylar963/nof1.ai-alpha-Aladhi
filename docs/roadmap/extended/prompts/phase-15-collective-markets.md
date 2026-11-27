# Phase 15 collective markets launch prompt

Implement only Phase 15, Collective AI Markets.

Read `AGENTS.md`, `requirements.txt`, the stack/testing skill, both roadmap
READMEs and status files, and extended `COVERAGE.md`. Require explicit user
acceptance of all preceding phases and R0, R0 PASS and Phase 15 unlocked. Report
missing gates and do not begin; loading this prompt does not waive them.

Audit accepted isolated agents, cooperative specialists, signals, ownership,
portfolio/risk and learning services. Implement an opt-in alliance mode with
membership/consent, scoped data permissions, validated signals/forecasts, horizon,
TTL, budgets, contributions and deterministic join/leave/failure behavior.
Alliance sharing must not silently alter isolated agents' privacy.

Implement coordinated allocation/execution as a declared group with aggregate
risk, internal order conflicts, self-trade prevention and clear inventory/fill/P&L
attribution. Treat participant messages as untrusted observations, never authority
to bypass risk. No agent may multiply limits by joining multiple alliances.

Demonstrate independent participants and alliances competing for finite liquidity
in replay/paper. Compare equal-condition baselines, recorded contributions and
actual costs. Exercise member failure, stale/conflicting forecasts, join/leave
during positions, concurrent orders and risk veto. Do not manufacture volume or
claim coordination guarantees better returns.

Stay on the current branch, preserve accepted modes and use authorized replay/
paper/testnet validation. Production group execution needs separate authorization.
No branch change, commit, push, PR or Phase 16 work is authorized implicitly.

Write the claim ledger, protocol/permission definitions, scenarios, tests and demo
under `docs/roadmap/extended/evidence/phase-15/`; update extended `STATUS.md`.
Mark READY FOR USER REVIEW only with complete work and required evidence; otherwise
report gaps. Wait for my explicit Phase 15 acceptance and request/load of Phase 16.
Do not grant user acceptance yourself.
