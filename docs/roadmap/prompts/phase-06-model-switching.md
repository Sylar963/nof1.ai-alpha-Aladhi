# Phase 6 automatic model switching launch prompt

Implement only Phase 6, Automatic Model Switching.

Read `AGENTS.md`, `requirements.txt`, the stack/testing skill,
`docs/roadmap/README.md` and `docs/roadmap/STATUS.md`. Verify every earlier phase is
explicitly accepted by the user and Phase 6 is unlocked. Loading this prompt does
not accept an earlier phase or bypass its missing evidence.

Audit the accepted model registry, agent ownership, performance ledger and regime
inputs. Design and implement a bounded, versioned policy that can actually change
the active trading model when completed-outcome performance, regime or qualified
candidate evidence meets the policy's conditions.

Define eligible candidates, minimum samples, evaluation windows, fees/slippage and
LLM costs, switching thresholds, cooldowns, ties and unavailable-data behavior.
Evaluate tuning on held-out/walk-forward data, separate from the data that selected
the policy. Do not let future results, incomplete trades or model self-reported
confidence stand in for measured evidence. Distinguish provider-error fallback
from a performance/regime-based switch.

Persist every selection, evidence snapshot, policy version and reason. Implement
a controlled handover and rollback without duplicate orders, lost protection or
accidental changes to existing position ownership and exit policy. All accepted
risk limits remain external to the selected model and cannot be raised by it.

Demonstrate automatic switch and no-switch cases, insufficient samples, stale
metrics, regime transitions, provider errors, restart recovery and switching while
a position exists. Report costs, switching frequency and risk-adjusted evaluation
without promising future profit.

Stay on the current branch, preserve prior phases, and use authorized replay,
paper/testnet/read-only validation. Do not start copy trading or Phase 7. No live
model handover, branch change, commit, push or PR unless separately requested.

Maintain evidence and the claim ledger in `docs/roadmap/evidence/phase-06/` and
update `STATUS.md`. Present READY FOR USER REVIEW only with complete implementation
and required validation. Do not mark accepted yourself. Wait for my explicit
Phase 6 acceptance and request/load of Phase 7 before advancing.
