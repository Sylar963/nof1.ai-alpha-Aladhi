# Phase 13 emotion and news credibility launch prompt

Implement only Phase 13, Market Emotional Index and News Credibility Filter.

Read `AGENTS.md`, `requirements.txt`, the stack/testing skill, both roadmap
READMEs and status files, and extended `COVERAGE.md`. Require explicit user
acceptance of all preceding phases and R0, R0 PASS and Phase 13 unlocked. Stop
before implementation if any gate is missing; loading this prompt is not approval.

Reuse accepted news and authorized social ingestion. Define and implement a
versioned emotional index with explicit weighting, normalization, coverage,
language handling, cadence, contributing observations and uncertainty.

Implement credibility assessment from source provenance, corroboration,
contradiction and corrections. Preserve evidence and supported/contradicted/
unverified states. Do not equate sentiment with factual truth or advertise an
infallible fake-news detector. Respect historical availability: later fact-checks
cannot inform an earlier decision.

Build labelled evaluation data with provenance, temporal holdouts, class metrics
and calibration where meaningful. Exercise duplicate rumor cascades, conflicting
sources, missing feeds, uncertain labels, language differences and prompt injection.
Demonstrate both components in actual decision context and an ablation against
the accepted baseline. Missing source access stays pending rather than simulated.

Preserve prior features, remain on the current branch and use offline/replay/
paper/testnet and authorized data. No access to unapproved private channels,
unspecified messages, production activation, branch change, commit, push or PR.
Do not begin Phase 14.

Maintain the claim ledger, evaluation set lineage, metrics and runnable demos in
`docs/roadmap/extended/evidence/phase-13/`; update extended `STATUS.md`. Present
READY FOR USER REVIEW only when implementation and required evidence are complete;
otherwise state gaps. Only my explicit acceptance and request/load of Phase 14
permit advancement.
