# Phase 7 Hyperliquid copy trading launch prompt

Implement only Phase 7, Hyperliquid Copy Trading.

Read `AGENTS.md`, `requirements.txt`, the stack/testing skill,
`docs/roadmap/README.md` and `docs/roadmap/STATUS.md`. All preceding phases must
have explicit user acceptance recorded and Phase 7 must be unlocked. If not,
report the gate and do not begin implementation.

Audit existing adapters and simulated copy-trading examples. Build a real
Hyperliquid public source-trade listener and normalized event ledger using the
current official contract, then integrate an authorized follower execution path.
Do not count the random-latency Polymarket printing demo as a starting integration.

Deliver source-wallet selection, stable event identity, cursor/reconnect recovery,
deduplication and a trace from every eligible source event to follower proposals,
orders, fills or recorded skips. Implement proportional sizing, rounding,
instrument/account mapping, optional AI filtering and the previously accepted
risk/margin/cooldown limits. Handle partial fills, late events, source closes,
reduce-only follower exits, unowned positions and process restarts.

Copying must never grant the source trader control over the follower's hard
limits, keys or account ownership. Record the effect of filtering and limits;
do not advertise full identical copying when a trade is intentionally skipped
or scaled. Measure observed source-to-follower latency and distinguish it from
unknown exchange/network delays; never insert random latency or promise 50 ms.

Prove real source ingestion and an end-to-end paper/testnet follower lifecycle,
including duplicate/reconnect/close cases. Real-money follower activation requires
a separate explicit user instruction with the intended account/source/limits.
Do not start production mirroring from this prompt alone.

Preserve earlier phases and stay on the current branch. Do not begin Phase 8,
switch branches, commit, push or open a PR unless separately requested. Maintain
the claim ledger, event/fill evidence, measured latency and demo instructions in
`docs/roadmap/evidence/phase-07/`; update `STATUS.md`.

When requirements and required validation are complete, mark READY FOR USER
REVIEW and present the result, or clearly report remaining gaps. Only I may
accept Phase 7. Start Phase 8 only after that acceptance and a request/load of
its launch prompt.
