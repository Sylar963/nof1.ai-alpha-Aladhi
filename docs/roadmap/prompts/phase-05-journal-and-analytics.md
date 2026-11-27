# Phase 5 trader journal and analytics launch prompt

Implement only Phase 5, Trader Journal and Analytics.

Read `AGENTS.md`, `requirements.txt`, the stack/testing skill,
`docs/roadmap/README.md` and `docs/roadmap/STATUS.md`. Require recorded explicit user
acceptance of all preceding phases and an unlocked Phase 5. If the gate is not
satisfied, report it and do not begin implementation.

Audit existing database models, event/fill persistence, performance calculations,
GUI charts and CSV export. Define and implement one reconciled reporting model
that reuses accepted account/agent identities and existing data where possible.

Deliver a trade journal containing timestamps, instruments, side/size, execution
events/fills, entry/exit reasons, model signal/version, strategy, effective risk
policy and final outcomes. Preserve multi-leg/hedge attribution and distinguish
realized/unrealized P&L, fees, funding and external capital flows. Unknown costs
or incomplete histories must be visible rather than treated as zero.

Implement P&L/metrics by day, week, agent, model, instrument and strategy, with
well-defined timezone/calendar boundaries. Provide usable GUI performance charts,
actual terminal visualizations, CSV/JSON exports and generated PDF/HTML reports
containing computed analytics and charts. A print statement, empty report template
or model-name column alone does not satisfy the corresponding capability.

Reconcile report totals and exports against controlled fill/fee/funding scenarios
and adapter account statements where access is authorized. Test partial closes,
flips, transfers, options expiry, hedge allocation, missing data and open positions.
Open/parse the produced PDF/HTML files and verify their contents and charts.

Preserve all prior phases. Do not implement automatic model switching, scheduled
external delivery or later tools here. Remain on the current branch and follow
the roadmap's offline/paper/testnet, privacy and publishing rules. No production
trading, notifications, branch change, commit/push or PR is authorized implicitly.

Maintain the claim ledger, test outputs, sample reports and demo instructions in
`docs/roadmap/evidence/phase-05/`; update `STATUS.md`. Mark READY FOR USER REVIEW
only once implementation and required evidence are complete. Never infer user
acceptance from tests or a PR. Do not begin Phase 6 until I explicitly accept
Phase 5 and request/load the next phase.
