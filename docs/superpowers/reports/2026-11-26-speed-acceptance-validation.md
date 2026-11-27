# Speed and acceptance measurement validation

This milestone implements the measurement and inspection stage. Thresholds are
provisional research settings; no fitted trading policy or equity-protection
performance is claimed. The public recorder and offline replay submit no live
orders. Existing trading-engine behavior is outside this change.

## Checks

- 248 selected capture, configuration, replay and calibration tests passed on
  Python 3.12.3 and Python 3.14.0 with every warning category treated as an error.
- Six frontend tests passed using Node's built-in test runner. They cover queued
  timeframe changes, matching-response merges, mismatched responses, failed
  requests, exact cursor tokens and visible rejection/arrival context.
- Independent re-verification passed all 57 focused calibration tests on both
  Python versions with warnings treated as errors. Reviewed findings are closed.
- Real-data browser workflow saved a source/stream interpretation, switched
  timeframes rapidly and exported its evidence without future outcome labels.
- Controlled browser workflow displayed four six-minute candles and the complete
  opening range, saved three interpretations, and exported their visible context.
  Both browser workflows reported zero script exceptions and no horizontal overflow.
- Pine reference edits passed restricted expression/control-flow checks.
  A TradingView compile remains a separate check.

Focused commands:

```bash
venv/bin/python -m pytest -q -W error \
  tests/test_speed_acceptance.py tests/test_calibration_dataset.py \
  tests/test_calibration_replay.py tests/test_calibration_report.py \
  tests/test_speed_acceptance_scenarios.py
node tests/replay_web_timeframe.test.cjs
```

Repeat the Python command with `venv-gtk/bin/python` for Python 3.12.

## Controlled level encounters

The fixture is explicitly marked `synthetic_test_only`. Its opening-range low is
59,900, high is 60,100 and midpoint is 60,000. Upper T1 is 60,198. These prices are
synthetic test inputs.

| Seconds after range start | Observed behavior | Verified result |
| --- | --- | --- |
| 901.1 | Price rises from 60,180 to 60,220 in 0.1 seconds | Upper T1 arrival, about 66.45 bps/s segment speed |
| 907 | Price remains beyond the level band | Persistent acceptance with 5.9 observed seconds outside |
| 908 | Price returns to 60,160 | Upward encounter rejects; new downward arrival stays separately visible |
| 910.5 | Binance spot connection gap | Spot evidence closes as insufficient; other market streams remain independent |
| During OR | A separate fixture interrupts spot coverage | Spot publishes no completed range or eligible encounter; perp remains complete |

Arrival snapshots retain their original values after later states appear. The
report keeps eventual excursions in `outcomes.ndjson`, separate from arrival
features. Reversing a deadline witness and same-clock gap reverses whether the
outcome is complete or censored, using original capture sequence for the tie.

To generate a fresh local demonstration:

```bash
venv/bin/python - <<'PY'
from pathlib import Path
from tests.test_speed_acceptance_scenarios import scenario_capture
directory = Path('data/research/synthetic-speed-acceptance-demo')
directory.mkdir(parents=True, exist_ok=False)
scenario_capture(directory / 'capture')
PY
venv/bin/python scripts/replay_paradex.py \
  --capture data/research/synthetic-speed-acceptance-demo/capture
```

This fails rather than overwrite an existing demonstration directory. Use six-minute
candles and advance through 20:15 in the configured Tijuana session (the fixture's
range starts at 2026-11-25 04:00 UTC). The recording is labelled **SYNTHETIC** in Replay.

## Real data and limits

The bounded all-source smoke capture recorded 92,586 rows and 93,944,288 bytes over
approximately 180 seconds. It retained Binance spot, Binance perp book, Binance
perp trades and Paradex observations. Binance 120-second speeds produced valid
samples and past-only ranks. The run did not include a complete opening range,
and Paradex supplied no transactions in that short interval.

A frozen prefix of the existing Paradex recording contained 133,810 rows and
225 accepted transaction observations, with recorded interruptions and incomplete
range coverage. The original recording was not rewritten. It is not a multi-venue
calibration sample.

A separate all-source capture was launched at 19:51:40 on November 26 in
`America/Tijuana`, before the 20:00–20:15 opening range. It requests a stop at
20:30 with an 8 GiB record cap. Its process/log metadata is
`data/market_capture/validation-20261126T195140-process.json`. This validation
record does not claim results from that still-collecting sample; inspect its
final durable status, gaps and `run_end` before analyzing it.

Local, gitignored artifacts generated during verification:

- `data/research/20261126-180s-all-source-speed-v1/report.json`
- `data/research/20261126-paradex-speed-v1/report.json`
- `data/research/synthetic-speed-acceptance-v1/report/report.json`

These reports remain `measurement_only_uncalibrated`. Multiple complete real
sessions, representative level encounters and cursor-time user labels are still
needed before selecting and evaluating thresholds on later untouched sessions.
The independent replay-fill model still does not establish options P&L, margin,
funding, market impact or a tradable 50 ms cross-venue lead.
