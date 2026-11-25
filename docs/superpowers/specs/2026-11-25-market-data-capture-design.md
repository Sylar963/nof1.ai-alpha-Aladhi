# Market data capture design

Approved in conversation on 2026-11-25. The user requested implementation of the public WebSocket collection scope after reviewing Binance spot/perpetual and Paradex capture.

## Purpose and boundaries

Collect BTCUSDT Binance spot/perpetual and BTC-USD-PERP Paradex observations for statistical speed studies and prospective lead/lag measurement. This process has no order entry, account authentication, portfolio management, LLM calls, or imports of trading adapters. No claim is made that Binance leads Paradex by a fixed interval. Historical replay cannot reconstruct unobserved periods.

## Sources

- Binance spot combined WebSocket: `btcusdt@bookTicker`, `btcusdt@aggTrade`, `btcusdt@depth@100ms`.
- Binance USD-M public combined WebSocket: `btcusdt@bookTicker`, `btcusdt@depth@100ms`.
- Binance USD-M market combined WebSocket: `btcusdt@aggTrade`, `btcusdt@markPrice@1s`, `btcusdt@forceOrder`.
- Paradex public WebSocket: `bbo.BTC-USD-PERP` and `order_book.BTC-USD-PERP.snapshot@15@50ms` with SBE schema 1:1. Use API-accessible books, not interactive/RPI feeds.
- Fetch Binance public depth snapshots on each connection before interpreting diff continuity. Preserve the snapshots and all received diffs; old pre-snapshot updates are retained with explicit status. A sequence gap reconnects and obtains another snapshot.

Paradex BBO sequences are not assumed consecutive. Paradex depth is a limited replacement snapshot; it is not an unthrottled full book. The archived upstream XML currently describes schema 1:3; templates 2/3 have the same documented fields at 1:1. Negotiate 1:1 and reject unexpected schema/version or malformed frames. Decode only subscribed public BBO/book templates and heartbeat controls, preserving exact binary bytes in base64. Fixed-point prices retain decimal precision and nulls. SBE timestamps are microseconds; Binance native timestamps are milliseconds and some spot BBO records have no source timestamp.

## Capture envelope and persistence

Every record contains capture schema version, source, session UUID, frame ordinal, local UTC nanoseconds and local monotonic nanoseconds. Market records additionally contain original text/base64, encoding, decoded payload, source event/transaction timestamps in nanoseconds where present, channel, market, source sequence, and message kind. Timestamp basis remains explicit. Connection starts/ends, gaps, REST snapshots, control replies and quarantined malformed frames are recorded.

Create a unique run directory below `data/market_capture`. Archive manifest and SBE schema/license/provenance. One bounded ordered writer serializes NDJSON segments; blocking file writes/fsync run off the receiving event loop. Defaults: 8 MiB pending serialized data, 128 MiB per segment, 1 GiB total captured records, 250 ms batch interval. Rotation does not delete prior segments. Overflow, disk failure, malformed data and storage cap terminate capture visibly. Status and durable counters advance only after fsync. Graceful shutdown drains accepted records. Incomplete/crashed runs remain distinguishable from clean shutdowns. A new invocation creates a new run and never appends to a potentially torn old segment.

## Runtime and configuration

Use existing `aiohttp>=3.13.1,<4.0.0` and Python 3.12-compatible code; no SDK dependency. Public transport sends subscription/control messages only. Reconnect with bounded exponential backoff and jitter, reset after actual market data, renew sessions and resubscribe. Honor HTTP 429 Retry-After. Treat HTTP 400/403/451 as visible configuration/access failures. Missing/sparse liquidation activity is not itself an outage. Use a transport watchdog and explicit connection-gap records.

CLI supports output directory, symbol/market, source subset for diagnostics, bounded duration, queue/file/total byte limits. Defaults/configuration live in `config_loader.py` and `.env.example`; no secret values are read into the capture manifest. No changes to `.env` or existing trading execution.

## Verification

Offline tests cover SBE fixture decoding/nulls/version/truncation, Binance units/controls/sequences, writer ordering/rotation/caps/fsync failure/draining, HTTP snapshot capture, WebSocket subscription/reconnect/session gaps, quarantine, and cancellation. A bounded public-network smoke capture is supplementary; access restrictions are reported instead of bypassed. Existing targeted config/backtest tests verify no regression.
