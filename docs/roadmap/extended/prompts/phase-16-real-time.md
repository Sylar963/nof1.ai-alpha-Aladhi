# Phase 16 real time decisions launch prompt

Implement only Phase 16, Measured Real-Time Decision Orchestration.

Read `AGENTS.md`, `requirements.txt`, the stack/testing skill, both roadmap
READMEs and status files, and extended `COVERAGE.md`. Require explicit user
acceptance of all preceding phases and R0, R0 PASS and Phase 16 unlocked. Stop
before implementation if any approval is missing.

Audit accepted queues, model calls, risk/execution priorities and capture/calibration
timestamps. Define user-reviewed deadlines per mode and pipeline stage. Distinguish
exchange event time, local receive time and monotonic duration; recovered history
cannot prove original receive latency.

Implement bounded queues, cancellation, backpressure, stale-work rejection and
priority for protective/reconciliation work. Slow models must not block hard risk
or protection management. Instrument throughput, queue age, p50/p95/p99 latencies,
deadline misses, CPU/memory and external service costs.

Benchmark representative sustained load, burst/reconnect, provider delays and
failure scenarios with explicit hardware, data, network, model versions and sample
sizes. Demonstrate deadline behavior in replay and authorized paper/testnet. Use
measured bounds; do not promise instant decisions, universal low latency or HFT.
Present profiling evidence before any proposed Rust rewrite; this prompt does not
authorize rewriting, purchasing hosting or deploying colocated servers.

Preserve accepted capabilities, remain on the current branch and follow roadmap
offline/paper/testnet boundaries. No live activation, branch/worktree change,
commit, push, PR or Phase 17 implementation is implied.

Maintain the claim ledger, benchmark harness/configuration/results, failure evidence
and operator instructions under `docs/roadmap/extended/evidence/phase-16/`; update
extended `STATUS.md`. Present READY FOR USER REVIEW only after implementation and
required validation are complete; otherwise state gaps. Only my explicit Phase 16
acceptance and request/load of Phase 17 permit advancement.
