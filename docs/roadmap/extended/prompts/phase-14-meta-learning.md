# Phase 14 meta learning launch prompt

Implement only Phase 14, Meta-Learning and Verifiable Training.

Read `AGENTS.md`, `requirements.txt`, the stack/testing skill, both roadmap
READMEs and status files, and extended `COVERAGE.md`. Require explicit user
acceptance of all preceding phases and R0, R0 PASS and unlocked Phase 14. Missing
approval stops this phase; this prompt does not accept previous work.

Audit existing adaptation/switching and approved training data. Define the actual
meta-learning objective and artifact: measured experience across authorized tasks
must enable evaluated adaptation on unseen tasks. Message exchange, model
switching and a loss multiplier alone do not satisfy this requirement.

Implement a reproducible pipeline with versioned datasets, training/tuning/test
separation by time/task, artifacts, configurations, costs and rollback. Sharing
private agent experience must be opt-in and recorded; preserve isolated mode.
Compare no-transfer and simpler non-learning baselines, disclose selection bias
and include negative held-out results without manufacturing improvement.

Separately audit the README deep reinforcement learning claim. If retaining it,
provide a real training environment, state/action/reward definition, training runs,
artifacts and independent evaluation. Otherwise present exact supported wording
for my explicit acceptance; never count prompting as deep RL or a wording removal
as an implemented learner. Do not silently reduce agreed feature scope.

Integrate trained candidates into the accepted paper/replay champion/challenger
workflow with bounded promotion, rollback and deterministic risk veto. A learner
cannot change hard limits, credentials or its production authority. Report the
compute plan before any work requiring an unapproved paid training service.

Stay on the current branch. Preserve accepted behavior and follow offline/replay/
paper/testnet boundaries. No live deployment, unapproved data export, branch change,
commit, push or PR; no Phase 15 implementation.

Maintain claim/evidence, dataset/artifact lineage, reproducible experiments and
operator instructions in `docs/roadmap/extended/evidence/phase-14/`; update extended
`STATUS.md`. Present READY FOR USER REVIEW only when required work and evidence
are complete; otherwise identify gaps. Do not accept Phase 14 yourself or start
Phase 15 until I explicitly accept and request/load that phase.
