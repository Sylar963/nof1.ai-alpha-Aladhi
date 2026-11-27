# Speed and acceptance research

This phase measures how price reaches and behaves at opening-range levels. It
does not submit hedge orders or replace portfolio risk controls. Arena Replay
provides the forward-only inspection and teaching interface; the feature engine
also works independently on recorded public data.

## What is measured

Signed speed uses logarithmic price displacement at 1, 5, 30 and 120 seconds:

`speed_bps_per_second = 10000 * log(current_price / earlier_price) / seconds`

Positive values mean rising prices. Acceleration compares consecutive speed
measurements. Historical percentiles compare observations with earlier samples
from the same stream; an extreme percentile is not a reversal probability.
Histories are bounded and sampled on a fixed time grid so a burst of messages
does not get hundreds of times the statistical weight of a quiet second.

Initial research settings are explicit and versioned in every report:

| Setting | Default |
| --- | --- |
| Price freshness | 3 seconds since original receipt |
| Percentile history | Last 3,600 valid one-second samples per stream/horizon |
| Percentile warmup | 30 earlier samples |
| OR coverage gap tolerance | 30 seconds |
| Level band | 1 bp on each side |
| Acceptance observation | At least 5 seconds, 3 prints and 70% observed time beyond the band |
| Encounter measurement window | 120 seconds |

These are provisional measurement settings, not optimized entry rules. A
late/malformed print or clock mismatch interrupts the evidence; it cannot
preserve an acceptance state through an unknown interval. Trusted prior prices
may remain visible while their speed evidence is unavailable.

Quotes measure price discovery. Transactions form the opening range. They are
kept separate: a midpoint jump is not proof of a transaction at that price.
Binance spot and perpetual ranges are independent; Paradex has its own
diagnostic range. Missing feeds stay missing.

The default range is exactly 20:00 inclusive to 20:15 exclusive in
`America/Tijuana`. Six-minute chart bars do not extend the cutoff to 20:18.
For high `H`, low `L`, midpoint `M = (H + L) / 2`, and width `W = H - L`:

- Upper/lower T1 are `M ± 0.0033 * M`.
- Each following upper/lower target extends by `2 * W` from the previous target.
- ORH, ORL and midpoint remain separate levels. T1 can lie inside the range.

Acceptance is a collection of observations rather than a candle pattern alone:
arrival speed, distance from a level, observed time beyond its band, persistence,
backtracking, reclaims and transaction flow where available. A slow advance can
be persistent; a fast burst can continue or reject. Neither speed nor a hammer
alone establishes forced liquidation, absorption or a profitable fade.

## Causal boundaries

Every feature uses only observations received by the current cursor. Exchange
timestamps identify the transaction's range/session, but do not move a late
message earlier in the receipt timeline. Source workers can submit a frame after
a later-received frame from another worker. The research cursor preserves the
original receipt timestamps and capture sequence, and uses the running maximum
receipt time as availability; it never sorts a delayed submission backward.
Asking for state, changing chart
timeframes or speeding up playback must not change the estimator.

Gaps, reconnects, stale observations and incomplete range coverage are explicit.
They must not create a completed range or a confident acceptance state. Historical
trades recovered by `repair_replay_prices.py` have unknown original receipt
times. They can support chart practice, but are excluded from receipt-time
calibration and cannot establish a 50 ms Binance execution lead.

The research index preserves all captured sources independently of the replay's
Paradex execution index. Original captures are never rewritten. An active capture
must first be frozen using the replay checkpoint mechanism. Indexes, checkpoints
and exported research files use additional disk space under `data/`.

## Calibration workflow

1. Record the selected markets across multiple complete Tijuana sessions and
   different volatility conditions. Start before 20:00. Keep spot/perpetual and
   Paradex feeds together when studying their relationship.
2. Inspect speed and level encounters in Arena Replay. Record your interpretation
   at the cursor, including deliberate waits/skips and hedge-release situations.
3. Study all eligible encounters, including those you did not trade. Keep future
   excursions/outcomes in a separate report from the entry-time features.
4. Develop thresholds on earlier complete sessions, freeze the settings and
   evaluate on later sessions. Group overlapping encounters by session/day;
   randomly splitting ticks would make neighboring observations look independent.
5. Validate the eventual hedge policy with actual options exposure, equity,
   fees, funding, margin headroom and executable depth. Market-state measurements
   alone do not establish an equity-protection strategy.

The initial settings are research assumptions, not calibrated trading limits.
Short samples and sessions with missing sources must be reported as insufficient
coverage. A useful next dataset contains complete ranges, many eligible level
encounters and later untouched sessions for validation.

See [capture and durability](market_data_capture.md) and
[Arena Replay](paradex_replay.md) for the existing recording and practice workflow.
The [validation record and controlled scenario](superpowers/reports/2026-11-26-speed-acceptance-validation.md)
describe the tested state transitions, source outages and available data coverage.

## Generate an offline measurement report

Run from the repository root with a healthy closed capture or an existing
frozen checkpoint and a new output directory.

```bash
venv/bin/python scripts/calibrate_market_data.py \
  --capture data/market_capture/CAPTURE_OR_CHECKPOINT \
  --output data/research/CAPTURE-speed \
  --outcome-seconds 60
```

The outcome horizon is optional and defaults to 60 seconds. The command prints
the path to `report.json`. An existing output directory causes an error. A
failure after reserving the directory records the input path and error in
`report.json` with status `failed`. A failed or interrupted run may contain
partial artifacts. Use a fresh directory when retrying.

- **`report.json`** records capture provenance, config and schema versions,
  source coverage and counters, sample counts, warnings and chronological
  split guidance. Success has status `measurement_only_uncalibrated`. Each
  source and stream reports actual settled one-second grid points and valid
  speed counts, signed mean, maximum absolute speed, and nearest-rank
  quantiles over the latest 4,096 valid absolute speeds per horizon.
- **`encounters.ndjson`** contains every observed level encounter with its
  stable `id`, first-observed `arrival` row and last observed `latest` state.
  Arrival features are available for entry-time analysis. The two rows retain
  their original capture sequence and clocks.
- **`outcomes.ndjson`** joins to encounters using `encounter_id`. Each row
  holds the horizon, status, reason, transaction sample count and source
  evidence sequences. It calculates direction-adjusted returns from the
  arrival price using subsequent same-source transactions through the horizon.
  MFE is at least zero, MAE is at most zero, and terminal return uses the last
  eligible print inside the horizon. A fresh print at or beyond the deadline
  witnesses continuity. A witness beyond the deadline does not contribute
  a price to the excursions. A censored row can contain partial prefix
  measurements. A covered window without a subsequent in-window transaction
  is unknown. Use complete outcomes for full-horizon comparisons.
- **`measurements.sqlite`** retains encountered rows, accepted transactions,
  continuity breaks and observed opening-range evidence. Sequence numbers
  refer to the original capture identified by `report.json`. The
  `capture-index/` directory contains the local receipt-order index.

Duplicate and recovered prints supply no outcome samples. A gap, capture or
exchange-session boundary, stale transaction stream or invalid or late trade
can censor a window. Quotes never supply transaction labels. Opening-range
completion establishes coverage of the range itself. Split selection also
requires coverage of the full study window, sufficient complete sessions and
cursor-time user labels.
