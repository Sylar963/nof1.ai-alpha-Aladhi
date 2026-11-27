# Phase 3 cooperative multi model environment launch prompt

Implement only Phase 3, Cooperative Multi-Model Decisions.

Read `AGENTS.md`, `requirements.txt`, the repository's stack/testing skill,
`docs/roadmap/README.md` and `docs/roadmap/STATUS.md`. All preceding phases must
have explicit user acceptance recorded and Phase 3 must be unlocked. This prompt
does not waive an earlier gate; stop before implementation if it is locked.

Audit the accepted single-agent framework, real data adapters and risk engine.
Create a concrete design against current code and then implement a cooperative
mode that is explicitly distinct from Phase 2's isolated mode.

Deliver specialists for news/sentiment, technical indicators,
on-chain/liquidity and risk/capital allocation. Give them real supported inputs,
configured models/providers, structured outputs and bounded resources. Define
their communication and a final adjudicator or explicit aggregation policy.
Preserve the input snapshot, availability timestamps, model/prompt/policy versions,
specialist contributions, disagreements and final rationale. Do not describe
identical API calls with role labels as a verified multi-model system without
showing the actual configuration and contribution chain.

Handle invalid/stale data, conflicting recommendations, missing specialists,
timeouts and provider failures deterministically. Untrusted news/on-chain content
cannot issue execution instructions. Models and votes cannot override the
deterministic risk veto or account permissions.

Validate a full authorized model/API integration and a reproducible replay using
recorded model outputs. Show a case where specialist information changes the
decision, a disagreement case and a risk-veto case. Report latency, token/cost
budgets and behavior when a required integration is unavailable. Preserve the
isolation and operating behavior of the previously accepted modes.

Do not implement Phase 4's advanced portfolio risk expansion or later features
except the minimal existing-risk integration this phase requires. Stay on the
current branch and follow all roadmap execution/privacy boundaries. No commit,
push, PR, branch change or production activation unless separately requested.

Write the claim ledger, tests, integration evidence and demo instructions under
`docs/roadmap/evidence/phase-03/`; maintain `STATUS.md`. Mark READY FOR USER REVIEW
only when the phase is implemented and its required evidence exists. Otherwise
state gaps. Do not mark it accepted or begin Phase 4 until I explicitly accept
Phase 3 and request/load Phase 4.
