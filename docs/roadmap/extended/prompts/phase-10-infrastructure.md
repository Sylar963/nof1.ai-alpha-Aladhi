# Phase 10 infrastructure launch prompt

Implement only Phase 10 of the extended roadmap.

Read `AGENTS.md`, `requirements.txt`, the stack/testing skill, the entire current
`README.md`, both roadmap READMEs, both `STATUS.md` records and the extended
`COVERAGE.md`. Require explicit acceptance of original Phases 1–9, R0 REVIEW PASS,
user acceptance of that review and an unlocked Phase 10. Missing gates stop this
phase; loading this prompt does not waive them.

Audit current adapters, secrets, configuration, account routing and packaging.
Refine the concrete design and implement Phase 10's packages in order:

1. Verified data-flow/custody documentation and actual secret storage protection.
   If retaining AES-256 wording, provide authenticated encryption with appropriate
   key management, migration/recovery, redaction and meaningful failure tests.
   Describe external LLM/news/exchange/notification transfers accurately.
2. Explicit Binance, Bybit and OKX adapter capabilities and authorized account
   bindings through existing execution/risk/journal contracts. Verify official
   interfaces; public capture is not evidence of order execution support.
3. Consistent simulation/paper/testnet/live routing with account/network checks,
   visible effective mode and protection against unintended live submission.
4. Supported local installation/launch and build/package paths, release provenance,
   provider/model capability matrix and accurate origin/affiliation statements.
   Test supported platforms; do not invent release artifacts or deployments.

Reuse existing Hyperliquid/Thalex/capture/replay services. Preserve all accepted
functionality. Prove meaningful adapter/mode/security/installation scenarios and
real authorized integrations. Missing access remains an explicit validation gap.
Do not trade real money merely to prove a live-capable path. Propose exact
user-reviewed corrections for incompatible blanket README claims.

Stay on the current branch. Follow the roadmap's offline/paper/testnet and secret
boundaries; no live activation, transfer, public release, branch/worktree change,
commit, push or PR is authorized by this prompt.

Maintain claim/evidence records under
`docs/roadmap/extended/evidence/phase-10/` and update the extended `STATUS.md`.
Mark READY FOR USER REVIEW only when implementation and required validation are
complete; otherwise list concrete gaps. Do not mark accepted yourself. Wait for
my explicit Phase 10 acceptance and request/load of Phase 11 before advancing.
