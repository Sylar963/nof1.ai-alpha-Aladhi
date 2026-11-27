# Phase 8 additional trader tools launch prompt

Implement only Phase 8, Additional Tools for Traders.

Read `AGENTS.md`, `requirements.txt`, the stack/testing skill,
`docs/roadmap/README.md` and `docs/roadmap/STATUS.md`. Require explicit user
acceptance of every earlier phase and an unlocked Phase 8. Loading this prompt
does not accept missing prior work. If locked, report the gate before implementation.

Audit existing replay/backtest, strategy, signal, portfolio and analytics code.
Refine the implementation design and execute the following packages in order:

1. AI strategy builder: translate text into a restricted, versioned strategy
   specification, show the actual rules and validate it in replay. Never execute
   arbitrary generated code or deploy it silently.
2. Historical backtesting: run that specification on frozen inputs with causal
   information availability, explicit fees/slippage/fill assumptions, held-out
   evaluation and reproducible results. Preserve known limitations of depth and
   recovered histories. A touched limit price does not prove a passive fill.
3. Signal mixer: normalize and combine model/strategy signals through configurable
   rules/weights, report conflicts/contributions and retain deterministic risk veto.
4. Portfolio manager: provide model/agent/asset capital budgets, exposure views
   and controlled rebalancing proposals using the accepted account ownership and
   portfolio risk system.
5. Performance heatmap: derive instrument, strategy and model cells from the
   reconciled ledger, with sample counts, metric definitions and missing-data states.
6. Social sentiment parser: real authorized X/Twitter, Reddit and Telegram sources,
   with provenance, timestamps, deduplication, classification and measured quality.
   Do not substitute simulated feeds or access unapproved private channels.
7. Dynamic fee optimizer: use verified account/venue fees and compare execution
   alternatives including spread, missed fills, adverse selection and inventory
   risk. Demonstrate net execution cost against a documented baseline, not just a
   lower nominal fee or an invented maker fill.

Finish and verify each package before beginning the next package. These are
internal Phase 8 work packages; they do not start Phase 9. Reuse prior accepted
services instead of creating parallel ledgers or risk policies. Missing access
must be stated; a UI placeholder does not complete a tool.

Use deterministic scenario tests, held-out replay and authorized integration
evidence. Preserve accepted functionality, remain on the current branch and
follow all roadmap execution/privacy constraints. No production activation,
private-channel access, branch changes, commits, pushes or PRs unless separately
requested by the user.

Maintain a per-tool claim ledger and acceptance results under
`docs/roadmap/evidence/phase-08/`; update `STATUS.md` with package progress.
Present the whole phase as READY FOR USER REVIEW only after all seven tools and
their required evidence are complete. Otherwise report unfinished packages.
Do not mark Phase 8 accepted or begin Phase 9 until I explicitly accept Phase 8
and request/load Phase 9.
