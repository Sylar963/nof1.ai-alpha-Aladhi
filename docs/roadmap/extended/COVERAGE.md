# Full README capability coverage

Source read in full: [README.md](../../../README.md), 292 lines on 2026-11-26.
SHA-256: `a73dcd2e890d3f94625b8bbef0781e855f46bc944cabb0ebafdcea4871b7df19`.
Line references identify this snapshot; refresh them and its fingerprint at R0.
This is a scope mapping, not a fresh implementation audit or passing review.

The original [Phases 1–9](../README.md) remain authoritative for their features.
Supplemental phases extend accepted services instead of reimplementing overlaps.
R0 checks the integrated original scope; final Phase 18 review checks the full
then-current README. No unsupported original claim is silently counted as done.

## Main README sections

| README section or claim | Original coverage | Additional obligation |
| --- | --- | --- |
| Introduction: autonomous trading and adaptive optimization, lines 1–3 | Phase 1 bounded adaptation; Phase 6 switching | Phase 14 validates genuine learning/deep RL or exact revised wording; Phase 18 bounds autonomy. |
| Introduction, concept and footer: origin, authorship, Hyperliquid API attribution, inspiration | Not a feature deliverable | R0 flags unsupported wording; Phase 10 verifies provenance/affiliation and distinguishes inspiration from ownership. |
| Language links and release/download banner, lines 4–16 | None | Phase 10 checks actual artifacts, links and supported platforms; Phase 18 aligns affected translations. |
| Project Concept: local PC and no required server/control, lines 18–26 | Existing local engine/replay; original phases add services | Phase 10 verifies local orchestration, external-service dependence and any optional hosted component. |
| Any model support | Phase 2 configurable single-model agents; Phase 3 specialists | Phase 10 documents tested providers and capabilities; do not promise literally every model. |
| All data/keys only local, no cloud transfers or leaks | Not established by nine-phase plan | Phase 10 audits storage/transfers and encryption; cloud model/news/notification use requires accurate user-approved wording. Absolute no-leaks guarantees are not testable deliverables. |
| Main Features 1 AI Trading, lines 31–44 | Phase 1 | R0 checks integrated evidence; later work must preserve controls. |
| Main Features 2 Single-Model Agents, lines 48–56 | Phase 2 | Exactly one AI model per isolated agent, including repair; model-brand skills need empirical evidence or clearly assigned roles. |
| Main Features 3 Multi-Model Environment, lines 60–70 | Phase 3 | R0 checks actual data/tools and contribution chain; Phase 15 alliances are a separate mode. |
| Main Features 4 Intelligent Risk, lines 74–78 | Phases 1 and 4 | Later venues/learners/alliances extend the same hard controls. |
| Main Features 5 Journal/Analytics, lines 82–86 | Phase 5 | Reuse one reconciled ledger throughout supplemental phases. |
| Main Features 6 Auto Switching, lines 90–96 | Phase 6 | Switching is distinct from Phase 14 learning and Phase 18 self-organization. |
| Main Features 7 Hyperliquid Copy Trading, lines 100–102 | Phase 7 | Phase 17 separately implements Polymarket semantics; skipped/scaled trades cannot be called identical copies. |
| Main Features 8 Trader Tools, lines 106–113 | Phase 8, all seven packages | Later correlation/index/filter/learning features reuse these tools. |
| Main Features 9 Notifications/Reports, lines 117–121 | Phase 9 | Preserve explicit destinations, sending authority and delivery evidence. |
| Successful Session table, Profit Visualization and simulated-values note, lines 125–150 | Phase 5 computes real analytics; these tables are illustrations | R0 and Phase 18 retain explicit simulation labels or replace with reproducible attributed results; never present sample numbers as real performance. |
| Polymarket LLM betting, liquidity farming, trader parsing, copy, dashboard, models, lines 154–164 | Not covered by Hyperliquid copy or a generic UI | Phase 17 real integrations with instrument-specific accounting, execution and dashboard. |
| Polymarket <50 ms and deployed colocated servers | No such guarantee from replay benchmarks | Phase 17 measures deployment-specific timing and verifies actual infrastructure; unsupported wording must change explicitly. |
| Security: AES-256 keys, no cloud calls except exchange, sandbox, daily limits, lines 166–173 | Phases 1/4 daily limits; original testing modes | Phase 10 verifies encrypted storage and isolation and corrects incompatible external-call claims. |
| Installation: release executable, admin launch, onboarding, lines 177–183 | No release certification in original plan | Phase 10 verifies actual packaging, provenance, supported OS, least-required privilege and setup paths. |
| Linux/WSL launcher/environment/backend behavior, lines 185–197 | Existing launcher instructions | Phase 10 exercises supported launch paths; Phase 18 final instructions review. |
| Extended Concept learning/evolution, lines 201–204 | Partial bounded adaptation/cooperation | Phases 14–18 give explicit measurable meaning; no claim of unbounded or profitable evolution. |
| Keywords/Tags, lines 257–259 | No capability requirement | Phase 18 removes the empty placeholder or supplies accurate editorial tags. |
| Public capture, raw timestamps/gaps/depth, lines 270–280 | Existing capture plus original causal validation | R0 validates actual capture behavior; Phase 10 preserves it; Phase 16 uses valid measured timing. |
| Arena Replay, checkpoints, hidden comparison, lines 282–286 | Existing lightweight replay and original strategy tests | R0 exercises workflow and checkpoint causality; supplemental features integrate without replacing replay. |
| Calibration settings/session/override rules, lines 288–292 | Existing calibration research | R0 checks configuration behavior; Phase 16 separates research deadlines from live guarantees. |

## Extended Concept mapping

| Requested capability | Existing roadmap coverage | Supplemental scope |
| --- | --- | --- |
| Infrastructure: Binance, Bybit, OKX and other exchange APIs | Existing adapters; original phases validate their supported paths | Phase 10 adds explicit initial Binance/Bybit/OKX scope and capability matrix. "Etc." is not an unlimited venue promise. Further venues need scoped requirements. |
| Infrastructure: simulation or live mode | Phases 1/4/7 validate paper/testnet paths and controls | Phase 10 unifies account/network/mode routing; no automatic activation of real accounts. |
| Infrastructure: analyze other bots and adapt | Phase 1 adapts to own completed outcomes | Phase 11 observes supported public behavior and bounded response; no invented identity/intent. |
| Single-model bots and independent strategies/timeframes | Phase 2 | Reuse; R0 verifies isolation and per-agent configuration. |
| GPT-Trader example: Twitter/news/volume | Phase 1 real news/market context; Phase 8 authorized social ingestion | Reuse; brand examples are configured roles, not guaranteed innate model capabilities. |
| Multi-model news/technical/on-chain/risk specialists | Phase 3 | Reuse; Phase 15 adds alliances without rebuilding the specialist team. |
| Alliances and inter-agent signals/forecasts | Not delivered merely by Phase 3 voting or Phase 8 signal mixing | Phase 15 membership, consent, messages, budgets and joint attribution. |
| Compete for liquidity and act as a large participant | Portfolio management alone is insufficient | Phase 15 finite-liquidity comparison and coordinated group execution under shared risk bounds. |
| Strategy Generator | Phase 8 strategy builder and backtest | Reuse the accepted restricted strategy specification. |
| Asset Correlation Analysis | Phase 4 exposure checks/Phase 8 portfolio views are partial foundations | Phase 12 dedicated measurements, uncertainty and decision/risk integration. |
| Meta-Learning between models | Neither Phases 1/6 adaptation/switching nor message sharing proves learning | Phase 14 explicit transfer/training objective and unseen-task evaluation. |
| Market Emotional Index | Phase 8 sentiment parser supplies inputs | Phase 13 defined aggregate index with lineage and evaluation. |
| Fake News Filter | Phase 1 untrusted news handling is a foundation | Phase 13 provenance/corroboration/credibility with uncertainty and temporal evaluation. |
| Real-Time Decision Making | Existing replay/speed tools and Phase 1 latency evidence | Phase 16 bounded event orchestration, deadlines and representative measured operation. |
| Self-organizing autonomous market agents | Original nine provide components, not the full ecosystem | Phase 18 governed membership/resource/policy adaptation, endurance scenarios and final user acceptance. |

## Review treatment of claims

For each claim, record the exact wording, code/data/configuration path, evidence
version, limitations, owner and one of: **SUPPORTED**, **PARTIAL**, **PLANNED**,
**SIMULATED**, **UNVERIFIED**, or **INACCURATE**. Record user-accepted wording
changes alongside their rationale; do not convert a removed claim into an
implemented feature. Keep implementation status separate from user acceptance.

R0 may pass with explicitly planned supplemental work. It may not pass with a
broken original-phase requirement or an unsupported claim still presented as
delivered. Full feature completion and factual README reconciliation are checked
again in Phase 18; unresolved agreed requirements remain incomplete.
