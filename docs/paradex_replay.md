# Arena Replay

A small chart-first trading replay for our Paradex opening-range work. Python
runs the replay clock and execution simulation; an original canvas interface
renders candles and trade levels. The standalone app uses the existing aiohttp
dependency, SQLite and browser APIs. It loads no NiceGUI, Plotly, exchange
adapter, LLM, CDN, frontend framework or build tool.

## Open the replay

```bash
venv/bin/python scripts/replay_paradex.py
```

Open `http://127.0.0.1:8081`, choose a recording and click **Start practice**.
To preselect a capture:

```bash
venv/bin/python scripts/replay_paradex.py --capture data/market_capture/YOUR_RUN
```

Active captures are supported. The app creates a separate, labelled checkpoint
from exactly the bytes reported durable in `status.json`. It validates this
prefix and treats it as a closed replay sample. The original recorder keeps
running. Unflushed bytes and future events are excluded. The frozen copy is
stored in `data/replay_snapshots/`; restarting the same attempt selects that
same sample by default. Select the active recording again for a newer sample.

The first load validates the complete sample and builds a compact SQLite index
under `data/replay_cache/`. Raw frames are not sent to the browser. Subsequent
attempts reuse the index when the source files and durable metadata match.
Checkpoints and indexes use additional disk space and are separate from the
recorder's size budget. Keep original capture files for research provenance.

The main desktop app's Replay page embeds the standalone app and links to its
full-screen view. Start the replay service separately using the command above.

## Chart workflow

- Replay real observations with Play/Pause, step forward and adjustable speed.
  Chart timeframe and replay timestep are independent. Available chart intervals
  include 1s, 5s, 1m, 5m, 6m, 15m and 1h.
- Warm-up reveals a short initial history before practice begins. The bot also
  starts trading only after warm-up. Short samples cap warm-up at one fifth of
  the available recording so time remains for practice.
- Scroll to zoom, drag to pan and use Fit to return to the latest candles.
- Buy/Long and Sell/Short place market orders. Choose Limit or right-click a
  price to place a limit order. The first version allows one human trade or
  pending entry at a time.
- Stop and target levels appear on the chart. Drag them or edit their inputs.
  An unfilled limit entry can also be dragged. Price scaling stays fixed during
  a level drag. Default protective distances use only visible history and spread;
  they are simulation conveniences, not a trading recommendation.
- Protective orders are submitted after the entry actually fills. They use the
  same configured arrival latency as other orders. Reduce-only quantities track
  actual filled inventory. Stops and targets cannot reverse the position.
- Close position submits a simulated market close. Cancel orders cancels working
  orders and protection; inventory remains open.
- Draw horizontal levels or two-point trend lines. Horizontal levels can be
  dragged or removed by double-clicking. Drawings are saved with the attempt.
- Go To advances to a UTC time or the next configured opening range. It processes
  every intervening recorded event, including orders and exits; it does not jump
  over execution. A jump is bounded to seven days and can be paused between batches.
  A destination outside the recording is rejected. Restart creates a separate
  attempt for going back.
- Record your read, or use Wait/Skip to explain a decision. Entry comments become
  the order's rationale. Routine chart controls do not require filling a form.

Keyboard shortcuts: Space play/pause, Right Arrow step, B buy, S sell, X close,
H horizontal line, T trend line, F fit chart, Escape cursor, N write a note.
Shortcuts do not fire while typing into an input or while a dialog is open.

## Shared clock and comparison

The human account and the opening-range baseline share one forward-only replay
clock and have independent simulated liquidity and accounts. Only reached
observations contribute to visible candles, opening-range levels and decisions.
Switching timeframe cannot expose a completed future candle.

The initial deterministic baseline trades the first last-trade-price break of a complete
opening range, uses the opposite range edge as a stop and a configurable
range-width target. Default range: 20:00–20:15 `America/Tijuana`, DST aware.
Change settings before starting the next attempt. Partial price coverage or
recorded interruptions during the range disable that session's entry. Execution
always requires a fresh two-sided recorded book.
This baseline is not yet trained to imitate your discretionary decisions.

The **Speed & acceptance** panel adds an independent causal research cursor for
all captured markets. Select a source to inspect signed speed, earlier-sample
percentiles, transaction OR/target levels and level persistence. Save an
interpretation with your reason to journal the current features. These
measurements do not change the breakout baseline or simulated orders. See
[speed and acceptance calibration](speed_acceptance.md).

Bot activity stays hidden until the recording ends or you select **Finish &
review**. Finishing freezes the attempt at the current cursor, expires outstanding
orders, reveals both decision traces and saves the comparison. It does not run
through future data. Open inventory remains marked at the last observed price.

Decision/fill logs are appended and fsynced to
`data/replay_sessions/ATTEMPT_ID/journal.ndjson`. Export writes
`comparison.json` with settings, rule version, accounts, drawings and both traces.
The journal contains the bot's trace and is not an access-control boundary.
Interrupted attempts retain their notes; restoring an unfinished UI session is
not yet implemented. Use the examples to agree and implement the next rule
version, then test frozen rules on newly recorded sessions.

## Data and execution limits

The default chart is **one-minute last-trade-price OHLC**, built from real Paradex
FILL and RPI transactions using matching-engine timestamps. Transaction volume
is in base units. Off-book transfers, settlements and block prints are not used
to form these candles. Duplicate trade notifications are removed. Candle highs
cannot include transactions that the replay has not reached. Late prints update
their original candle without replacing a later candle's close.

The recorder now captures the public trade tape alongside BBO and up to 15
displayed API-book levels per side. Price signals and the opening range use
trades; order fills continue to use recorded API depth. RPI prints are visible
prices, but RPI-only liquidity is not assumed executable by an API bot.

Older book-only recordings remain labelled **legacy quotes**. Their midpoint
can jump when a distant quote changes or disappears; they are not transaction
candles and should not be compared with a BTCUSD/BTCUSDT last-price chart. No
price smoothing or invented intrabar path is used to hide those differences.

To make a separate price-repaired copy of an old closed capture/checkpoint:

```bash
venv/bin/python scripts/repair_replay_prices.py --capture data/replay_snapshots/YOUR_CHECKPOINT
```

This downloads production historical transactions, including six hours of prior
chart context, and preserves raw responses, pagination, a trade-sidecar hash,
the original depth files and recovery provenance under `data/replay_repaired/`.
Original recordings are not rewritten. Recovered prints become visible only
after a recorded book whose source timestamp has reached the matching time;
earlier context is revealed during warm-up. Original trade receipt times cannot
be recovered, so repaired samples cannot validate receive latency or lead/lag.
Fresh recordings preserve actual receive times. Offline loading never downloads
or silently substitutes market data. Different venues' BTCUSD/BTCUSDT trades
will not necessarily match Paradex's perpetual trades tick for tick.

Orders can fill only on a subsequent depth snapshot after their configured
arrival delay. Market orders walk available opposite depth and cancel the
unfilled remainder (IOC). Limits fill only against crossing opposite-side
liquidity; a touch or disappearance does not establish a passive queue fill.
Both actors get separate depth copies, consumed once per actor per snapshot.
Replacement snapshots refresh displayed quantities. Simulated orders do not
change historical prices or recover queue position and market impact.

Cancellation is immediate at the cursor. Gaps/reconnections and intervals over
the freshness threshold invalidate old depth and cancel pending orders; positions
remain. These intervals have unknown execution outcomes. Default latency 100 ms,
freshness 2000 ms and fees 5 bps are editable assumptions. Funding, margin and
liquidation are not modeled. A configurable inventory cap applies to both actors.

## Three-day collection

```bash
venv/bin/python scripts/record_market_data.py \
  --sources paradex --paradex-market BTC-USD-PERP \
  --duration 259200 --max-total-bytes 34359738368
```

This requests 72 hours with a 32 GiB record cap and rotating files. Coverage
depends on activity, disk space, connectivity and machine uptime. The cap can
stop recording early. Start before the desired range and keep the machine awake.
Use Ctrl-C to close a foreground capture cleanly. Read
[capture diagnostics and recovery](market_data_capture.md) for gaps and storage
integrity. All data, checkpoints, caches and logs stay gitignored under `data/`.

## Measure before changing languages

```bash
venv/bin/python scripts/benchmark_replay.py --capture data/replay_snapshots/YOUR_CHECKPOINT
```

The benchmark reports index build/reuse time, startup including warm-up, event
throughput and process peak RSS. It runs on real captures with bounded chart
buffers (2000 bars per timeframe) and batches execution independently of rendering.
The browser receives candle deltas, rather than a UI element update for every tick.
The static HTML/CSS/JavaScript is about 50 KB uncompressed with no downloaded assets.

On the development machine, a roughly 58 MB sample indexed in about 3.1 seconds;
a cached attempt including five minutes of warm-up loaded in about 0.4 seconds;
one recorded minute advanced in about 61 ms. These are sample measurements,
not guarantees for a three-day capture. Profile larger recordings and live browser
interaction before deciding whether to port the replay core to Rust. The UI/API
boundary keeps that option open without coupling it to a full bot rewrite.

## Tests

```bash
venv/bin/python -m pytest tests/test_replay_trade_prices.py tests/test_replay_workbench.py tests/test_paradex_replay.py -q
```

Tests cover capture checkpoints, indexing, candle causality, bounded chart memory,
delayed fills, brackets and level amendments, cancellation, range coverage,
comparison visibility, export, and HTTP mutation boundaries. Browser validation
also exercises chart dragging, notes, timeframe changes and position closure.
