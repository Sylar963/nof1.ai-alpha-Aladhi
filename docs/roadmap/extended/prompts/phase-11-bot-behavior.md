# Phase 11 bot behavior launch prompt

Implement only Phase 11, Observable Bot Behavior and Bounded Adaptation.

Read `AGENTS.md`, `requirements.txt`, the stack/testing skill, both roadmap
READMEs and status files, and extended `COVERAGE.md`. Require original Phases 1–9,
R0 and Phase 10 to have recorded explicit user acceptance, R0 PASS and Phase 11
unlocked. If not, report the gate and stop before implementation.

Audit accepted market/account data and adaptation policies. Define a concrete
observation contract that distinguishes identifiable public participants,
aggregate order-flow behavior and unknown identity/intent. Verify sources and
supported fields; never label anonymous flow as a known bot without evidence.

Implement causal liquidity/flow/quote-response features with provenance,
availability times, sampling limits and uncertainty. Integrate bounded policy
responses through the existing decision/risk engine rather than giving observed
participants direct execution authority. No private account access is implied.

Use known participants in controlled replay for ground truth and actual authorized
public data for ingestion evidence. Evaluate baseline/adapted decisions on held-out
sequences, report costs and uncertainty, and exercise gaps, stale observations,
regime shifts and inconclusive evidence. Demonstrate behavior changes without
promising improved profit or inferred knowledge of hidden trading intentions.

Preserve accepted functionality and stay on the current branch. Follow the
roadmap's offline/replay/paper/testnet boundaries. Do not activate live policies,
change branches, commit, push, open a PR or implement Phase 12 unless separately
requested through the applicable gates.

Maintain the claim ledger, evaluation, failure scenarios and demo evidence under
`docs/roadmap/extended/evidence/phase-11/`; update extended `STATUS.md`. Present
READY FOR USER REVIEW only with complete implementation and required validation,
otherwise list gaps. Only I may accept Phase 11. Wait for that explicit acceptance
and a request/load of Phase 12 before advancing.
