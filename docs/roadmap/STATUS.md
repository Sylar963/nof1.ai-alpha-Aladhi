# Trading roadmap status

Updated: 2026-11-26, America/Los_Angeles.

## Current authorization

- Active phase: **1, AI Trading**.
- Current stage: **planning complete; implementation not started by this request**.
- The user requested the nine-phase roadmap and reusable phase prompts.
- The user explicitly requires one phase at a time, with no advancement until
  they say the current phase is done.
- Next action: load [the Phase 1 prompt](prompts/phase-01-ai-trading.md).
- No phase has been accepted. Future phase prompts are prepared but locked.
- The [extended roadmap](extended/README.md) and its prompts are prepared. Its
  integrated review waits for acceptance of all nine phases; Phases 10–18 wait
  for that review to pass and receive user acceptance. Phase 1 remains active.

## Phase register

| Phase | Name | State | User acceptance |
| --- | --- | --- | --- |
| 1 | AI trading | Active; ready to begin implementation | Pending |
| 2 | Isolated single-model agents | Locked | Not requested |
| 3 | Cooperative multi-model environment | Locked | Not requested |
| 4 | Intelligent risk management | Locked | Not requested |
| 5 | Journal and analytics | Locked | Not requested |
| 6 | Automatic model switching | Locked | Not requested |
| 7 | Copy trading | Locked | Not requested |
| 8 | Additional trader tools | Locked | Not requested |
| 9 | Notifications and reports | Locked | Not requested |

## How to maintain this file

During the active phase, record completed work, current blockers and evidence
links. Use **IN PROGRESS**, **INCOMPLETE**, or **READY FOR USER REVIEW** as
appropriate. Record **ACCEPTED BY USER** only after an unambiguous user approval
of that phase; include the exact message, date and evidence version reviewed.
Never infer acceptance from test results, a commit, a merged PR or an agent's
completion report.

After acceptance, unlock only the next phase. Acceptance alone does not start
it: wait for its launch prompt or an explicit request to begin. Record any
later user-authorized change to the sequence or acceptance process here.

After Phase 9 acceptance, check all nine acceptance records and unlock only the
extended roadmap's Review R0. Run it only on request/prompt. Supplemental Phase 10
still requires R0 PASS, explicit user review acceptance and a start request/prompt.

## Phase 1 progress

- [x] Master roadmap created.
- [x] Phase 1 implementation plan created.
- [x] Launch prompts for all nine phases created.
- [ ] Refresh the baseline claim-to-code audit at implementation start.
- [ ] Implement the Phase 1 work packages.
- [ ] Complete acceptance tests and integration evidence.
- [ ] Present the Phase 1 review package.
- [ ] Receive and record the user's Phase 1 acceptance.

## Acceptance history

None. Completing this planning request does not complete Phase 1.
