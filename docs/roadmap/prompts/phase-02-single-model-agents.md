# Phase 2 isolated single model agents launch prompt

Implement only Phase 2, Isolated Single-Model Trading Agents.

First read `AGENTS.md`, `requirements.txt`, the repository's stack/testing skill,
`docs/roadmap/README.md` and `docs/roadmap/STATUS.md`. Phase 1 must have explicit
user acceptance recorded and Phase 2 must be unlocked. Loading this prompt does
not accept Phase 1. If that approval is missing, report the gate and do not begin.

Audit the current engine, account binding, global configuration, database paths,
LLM clients and UI lifecycle. Refine a concrete Phase 2 implementation plan from
the roadmap, then implement it rather than stopping at the plan.

Deliver per-agent identity, primary model/provider, settings, lifecycle,
account/subaccount ownership, isolated private context and journals, risk budget,
and independent performance views. Permit two configured agents to run at the
same time and compare models on the same frozen public replay using independent
accounts. Do not claim isolation from agent labels on a shared netted account.
Use exactly one configured AI model per agent, including model-assisted output
repair. Replace or disable any different-model sanitizer/fallback in this mode;
deterministic schema validation and retries to the same model are allowed.
Do not assume a model brand inherently performs news, liquidity
or technical analysis: provide its configured role and required inputs.

Prove that starting/stopping one agent, editing its model or risk settings,
restarting it or receiving an error cannot alter another agent's orders,
positions, private history, credentials or policy. Verify account ownership,
data isolation, resource cleanup, persistence and concurrent execution with
meaningful tests and paper/testnet evidence. Preserve all accepted Phase 1 controls.

Do not implement the Phase 3 cooperative team or any later phase. Remain on the
current branch; do not create/switch branches, commit, push or open a PR unless
I separately request that action. Use the roadmap's offline/paper/testnet and
secret-handling boundaries; no production activation is implied.

Maintain a claim ledger and review evidence under
`docs/roadmap/evidence/phase-02/`, and update `STATUS.md`. Once the phase's
requirements and required evidence are complete, mark READY FOR USER REVIEW and
show a two-agent demonstration. Otherwise list concrete gaps. Never mark the
phase accepted yourself, and do not start Phase 3 until I explicitly accept
Phase 2 and request/load the next phase.
