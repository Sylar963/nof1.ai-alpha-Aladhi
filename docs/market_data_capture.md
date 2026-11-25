# Public market-data recorder

This standalone process records Binance BTCUSDT spot/perpetual and Paradex BTC-USD-PERP observations for speed calibration and prospective lead/lag research. It does not initialise the trading engine, authenticate an account, call an LLM, or submit orders. No API keys are required.

## Run

From the repository root, with the existing dependencies installed:

```bash
venv/bin/python scripts/record_market_data.py --duration 60
```

For continuous recording, omit `--duration`. Stop with Ctrl-C or SIGTERM; the process cancels receivers and drains accepted records. The default 1 GiB record cap also stops capture visibly. The command exits nonzero for failed storage, invalid frames, access restrictions, or a bounded run that received no market observations from a selected source.

```bash
venv/bin/python scripts/record_market_data.py --output data/market_capture
venv/bin/python scripts/record_market_data.py --duration 30 --sources paradex
venv/bin/python scripts/record_market_data.py --duration 30 --sources binance_spot,binance_perp_book,binance_perp_market
```

Source selection is for diagnostics and deliberate partial capture. A capture that excludes a required market cannot establish cross-venue lead/lag. Run in a location where all selected public endpoints are accessible. HTTP 400/401/403/404/451 stops the run; the recorder does not change locations or endpoints to bypass restrictions.

The settings are in `.env.example` under `CAPTURE_*`, with defaults in `src/backend/config_loader.py`. Existing `.env` values are not copied to capture files; only the recorder's public settings enter the manifest. CLI overrides include `--symbol`, `--paradex-market`, `--max-queue-bytes`, `--max-file-bytes`, and `--max-total-bytes`. `--help` lists them. Symbols on Binance and Paradex must be selected independently; no source symbol is interpreted as an execution mapping.

## Captured feeds

| Source group | Endpoint and observations |
| --- | --- |
| `binance_spot` | `stream.binance.com:9443/stream`, combined `bookTicker`, `aggTrade`, `depth@100ms`; initial REST depth snapshot |
| `binance_perp_book` | `fstream.binance.com/public/stream`, combined `bookTicker`, `depth@100ms`; initial REST depth snapshot |
| `binance_perp_market` | `fstream.binance.com/market/stream`, combined `aggTrade`, `markPrice@1s`, `forceOrder` |
| `paradex` | `ws.api.prod.paradex.trade/v1`, `bbo.BTC-USD-PERP` and `order_book.BTC-USD-PERP.snapshot@15@50ms`, SBE 1:1 |

Binance REST snapshots are fetched concurrently with WebSocket receiving. Messages received before snapshot alignment remain in the capture. `depth_alignment.buffered_update_statuses` maps their connection-local frame ordinals to `pre_snapshot`, `bridged`, or `continuous` classifications. Later depth records carry `depth_sequence_status`. A broken snapshot bridge or diff chain records a gap and reconnects with a new session/snapshot. Spot uses inclusive U/u coverage; USD-M uses pu after bridging. BBO sequence numbers are not assumed consecutive. Paradex's depth feed uses replacement snapshots, not incremental book reconstruction.

Paradex snapshots contain at most 15 levels per side and have a 50 ms refresh setting. Use the independent event-driven BBO observations for finer receive-time comparisons. The API-accessible book is selected; interactive/RPI-only liquidity is not assumed available. Binance aggregated perp trades are published at a 100 ms cadence and liquidation observations are one-second snapshots, not a complete trade/liquidation tape. Sparse liquidation observations are normal.

## Files and durability

Each invocation creates a fresh directory like `data/market_capture/20261125T160000Z-a1b2c3d4e5f6`:

- `manifest.json`: public source URLs, settings, run identity, capture version and schema hash.
- `paradex_1_0.xml`, `LICENSE.paradex`, `README.md`: archived upstream schema, licence and provenance.
- `records-00000000000000000001.ndjson`, subsequent segments: observations and lifecycle markers in acceptance order.
- `status.json`: last fsynced record/byte counts and writer `closed`/`failed` state.

The manifest archives the current upstream schema 1:3. The live client explicitly negotiates 1:1 and decodes only public BBO/book layouts and heartbeat controls defined for that version. It rejects an unexpected schema/version rather than silently guessing a layout. Schema provenance and SHA-256 are recorded in the archived README.

Every market frame preserves raw text or base64 binary, the decoded payload, source/session identity and frame ordinal. `capture_sequence` orders accepted records across source workers. Decimal prices/quantities remain strings and unavailable fields remain null. `received_at_ns` is local UTC; `received_monotonic_ns` is a local monotonic clock. Normalised source timestamps use nanoseconds, while `source_time_basis` identifies the original `unix_ms` or `unix_us` unit and the payload retains native fields. Some Binance spot BBO messages supply no exchange timestamp; the recorder does not fabricate one. Paradex binary `ts_us` is the feed publish timestamp, not a guaranteed matching-engine event timestamp.

An ordered writer batches work every 250 ms, with filesystem work off the receiving event loop. A record is durable only after fsync. A process or power failure can lose all pending records, including a batch undergoing fsync and records accepted during that write; their combined serialized size is bounded by the pending-byte limit. Counters/status advance after successful fsync. Default bounds are 8 MiB pending serialized records, 128 MiB per segment, and 1 GiB total record bytes. Schema/manifest/status files are outside the record byte cap. Rotation retains earlier segments. The recorder stops rather than overwrite or delete history. Queue capacity includes data currently undergoing fsync.

`status.json` describes the writer, not exchange health. A clean writer close can follow a transport/protocol failure: inspect the final `run_end.outcome`, `error`, and `source_message_counts` as well. Failure before a run_end record or a missing closed status means the run was interrupted. If the filesystem also prevents diagnostic writes, the process still reports failure to stderr and exits nonzero.

## Recovery and replay

Transient transport errors and 429/5xx responses create explicit gap/retry records, with bounded exponential backoff, jitter and Retry-After. Backoff resets after market data, not merely a successful socket open. Reconnects use new session IDs. There is no historical WebSocket replay: missing intervals remain missing. A newly installed snapshot establishes current depth; it does not recover lost observations.

Malformed frames are quarantined with original bytes/text and stop capture. An oversized REST response retains a labelled truncated prefix. If aiohttp rejects malformed UTF-8 or an oversized WebSocket frame before exposing its bytes, the quarantine record includes the close code and `raw_unavailable=true`; the recorder stops rather than retry that protocol failure indefinitely. Opening handshakes have a 15-second deadline, separately from live WebSocket heartbeat and closure timeouts. For disk-full, queue overflow or record cap, accepted records are drained where storage permits, and errors remain visible. Correct the problem and start a new run. Do not append to an old, potentially torn segment.

```python
from pathlib import Path
from src.backend.market_data.storage import iter_capture_records

for record in iter_capture_records(Path("data/market_capture/YOUR_RUN")):
    if record["kind"] == "bbo":
        print(record["source"], record["received_monotonic_ns"], record["payload"])
```

Replay rejects incomplete final lines, malformed JSON, and non-contiguous capture sequences instead of silently repairing them. It reconciles a closed run's record/byte totals with durable status, detecting removed complete lines, middle segments, and missing tails. For interrupted runs, complete contiguous records beyond a lagging durable status are allowed, but replay cannot fall below the known durable totals. Keep the original failed run unchanged; recovery or salvage should produce a separate, explicitly labelled artifact.

## Research limits

This is the collection foundation, not a calibrated speed classifier or a trading signal. Six-minute candles, Tijuana session ranges, acceptance/rejection features, portfolio joins and order execution are separate stages. Historical API downloads cannot recover past local receive timestamps. Capture from the intended deployment location with a synchronised system clock before assessing a proposed 50 ms Binance lead. Application receive timestamps include network and scheduling delay; they are not kernel packet timestamps. Source clocks and publish conventions differ, so subtracting exchange timestamps alone does not establish a tradable lead. Actual fees, fill latency and unavailable liquidity still need execution validation later.

## Validation

```bash
venv/bin/python -m pytest tests/test_market_data_protocol.py tests/test_market_data_storage.py tests/test_market_data_recorder.py tests/test_config_loader.py -q
```

Tests are offline and cover binary layouts/units/nulls, strict input validation, diff continuity, concurrent snapshot capture, reconnect/session gaps, cancellation, bounded storage, fsync errors, and torn replay. Public live smoke captures are supplementary and place no orders.

Official contracts reviewed 2026-11-25:

- [Binance spot streams](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/ws-streams/~)
- [Binance USD-M public streams](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/ws-streams/public)
- [Binance USD-M market streams](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/ws-streams/market)
- [Paradex SBE](https://docs.paradex.trade/ws/general-information/binary-encoding-sbe)
- [Paradex book feed](https://docs.paradex.trade/ws/web-socket-channels/order-book-market-symbol-feed-type-15-refresh-rate-price-tick/order-book-market-symbol-feed-type-15-refresh-rate-price-tick)
