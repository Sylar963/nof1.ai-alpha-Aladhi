# Speed and acceptance calibration

Approved scope: the user approved the six-step proposal, beginning with causal market features, a research dataset/report, and Arena Replay inspection. This implementation delivers that measurement milestone, not a live hedge policy or calibrated trading thresholds.

## Global constraints

- Public recorded data only; no exchange orders, private account access, or changes to existing trading decisions.
- Preserve receipt order and capture provenance. Future observations may be used only for explicitly separate outcome labels.
- Independent Binance spot/perpetual ranges, Paradex diagnostic range, timezone America/Tijuana, 20:00–20:15. T1 = midpoint ± 0.0033 × midpoint; later targets spaced by twice OR width. Six-minute context never extends the exact 15-minute cutoff.
- Fixed-time causal sampling, signed logarithmic speed at 1/5/30/120 seconds, acceleration and historical percentiles with sample counts and warmup. Quote and transaction features remain distinct.
- Acceptance measures observed behavior at levels, not a prediction or permission to hedge. Thresholds are explicit provisional research settings. Missing data, stale prices, incomplete ranges, and gaps must suppress confident states.
- Recovered trades are chart history with unknown original availability, excluded from receipt-time calibration and latency claims.
- No new dependencies. Keep Python 3.12 compatible. Preserve unrelated user changes. Work in the existing feature checkout; the user authorized checkpoint commits and continuation on the existing PR.

### Task 1: Causal feature engine

Create `src/backend/market_data/calibration.py` and `tests/test_speed_acceptance.py`. Use test-first development. Public interface: frozen `CalibrationConfig`; `SpeedAcceptanceEngine(config=None)`; `observe(record)` accepts the existing capture envelope; `snapshot(now_ns=None)` returns JSON-safe versioned dictionary with `sources` keyed by capture source, `basis`, and provenance/config. `snapshot` must be observational: request frequency cannot train the estimator or change its results. Provide source counters and explicit reasons for insufficient evidence.

Normalize Binance spot/perp BBO and aggTrade, Paradex API BBO/depth/trades. No depth deltas masquerading as full books. Keep price-discovery quote streams separate from trade streams; quote speed is valid but never a transaction OR. Build transaction ORs independently per source using source transaction/event timestamps only after receipt. Guard late/out-of-order trades, duplicate identities, session changes, gaps, stale intervals, and partial OR coverage. Reject invalid numeric values. Track speed at fixed one-second grid points before consuming later messages, so tick bursts do not overweight historical percentiles. Use bounded rolling histories and past-only baselines, expose sample counts/percentiles and acceleration. Require sufficient coverage/freshness for each horizon.

After a valid complete OR, expose ORH/ORL/ORM and upper/lower T1–T4. Track all level encounters, including skipped ones; price crossing a band between observations must count but flag an unobserved touch. For active encounters report signed distance, arrival speed, dwell/time-weighted occupancy outside a level band, efficiency, reclaims, transaction volume/aggressor flow if known. Research states distinguish arrival, persistent acceptance, rejection, insufficient evidence. No future candle, liquidation inference, or automatic action. Include age/sample counts and expose uncertainty, not confidence invented from speed percentile.

Tests should cover causality/prefix invariance, fixed-time weighting, horizon formula/direction, stale/gap resets, trade dedup and event-time handling, exact OR cutoff/timezone/DST, target formulas, separate sources and missing feeds, level crossing/occupancy/rejection vs acceptance, and JSON safety. Read AGENTS.md, CLAUDE.md, testing skill and relevant protocol/trade source before edits. Report exact API/schema and test results to the controller. Do not commit or dispatch agents.

### Task 2: Research dataset, report and Arena Replay

Create a disk-backed research index preserving all sources, BBO, actual receipt clocks, original sequence and discontinuities without altering the existing execution index semantics. It must validate source integrity and work on frozen checkpoints. Add a streaming CLI `scripts/calibrate_market_data.py` and report helpers. Produce versioned metadata/settings, per-source coverage, speed summaries/sample counts, eligible encounters, explicitly separate future outcome labels with horizon and censoring/gap flags, and chronological session split guidance. Never advertise trained thresholds from a short sample. Bound memory and report output; use NDJSON for potentially large encounter exports.

Connect the causal engine to Workbench through a separate research cursor/index, advancing only through the current reached replay clock. Include only reached features in state. Add a compact speed/acceptance UI and OR/target overlays, clearly source-labelled and separate from the existing generic OR-breakout baseline. Add structured user labels at cursor with captured causal feature state; persist/export through existing journals. No future outcomes exposed during practice. Keep existing chart, fills, bot baseline and orders unchanged. Tests must demonstrate playback-step invariance, future hiding, multi-source preservation, gap/recovered-data exclusions, journaling/export and regression compatibility. Add `docs/speed_acceptance.md` with commands, settings, meanings and data limitations; link it from replay/capture docs. Do not commit or dispatch agents.

### Task 3: Independent review and validation

Review correctness and causal boundaries independently, resolve substantive findings with regression tests, run feature/replay/capture checks on supported Python environments and a completed real capture. Report actual data coverage and benchmark measurements without promising live latency or profitability.
