# Market Data Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Record Binance BTCUSDT spot/perpetual and Paradex public data durably for speed and lead/lag research.

**Architecture:** Independent protocol, storage, and transport modules with an executable public-data-only CLI. Raw observations and clock metadata survive normalization. Tests use local fakes, never trading credentials.

**Tech Stack:** Python 3.12-compatible asyncio, aiohttp>=3.13.1,<4.0.0, standard-library struct/decimal/filesystem, pytest/pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-11-25-market-data-capture-design.md`

## Global Constraints

- No order entry, account authentication, portfolio management, LLM calls, or imports of trading adapters.
- No changes to `.env` or user-created `data.md`, skill files, or lock files.
- Preserve raw messages, source units, nulls and decimal precision; do not invent missing observations.
- Storage defaults: 8 MiB queue, 128 MiB segment, 1 GiB run cap, 250 ms batching.
- Offline tests only; a separate bounded public smoke capture may be attempted.
- Work on `feat/market-data-capture`, keeping the existing IDE checkout. No push or merge.

## Task 1: Protocol decoding and sequence validation

**Files:** Create `src/backend/market_data/__init__.py`, `protocol.py`, `schemas/paradex_1_0.xml`, `schemas/LICENSE.paradex`, `schemas/README.md`, and `tests/test_market_data_protocol.py`.

**Interfaces:** `decode_frame(source: str, raw: str | bytes) -> dict`; `ProtocolError(ValueError)`; `SequenceGap(ValueError)`; `BinanceDepthSequence(snapshot_id: int, futures: bool = False).observe(payload: dict) -> str`.

Decoded dictionaries contain `kind`, `channel`, `market`, `payload`, `encoding`, `source_event_time_ns`, `source_transaction_time_ns`, `source_sequence`, and `source_time_basis`. Preserve source payload fields. Binance sources are `binance_spot`, `binance_perp_book`, `binance_perp_market`; Paradex source is `paradex`. Market kinds are `bbo`, `trade`, `depth`, `funding`, `liquidation`, or `control`. SBE requests schema 1:1; public templates 2/3 and heartbeat 40 are supported. Unknown templates, wrong version, truncation, nonfinite prices and malformed documented fields raise ProtocolError.

- [x] Add offline tests demonstrating exact decimals and microsecond timestamps, Binance missing spot-BBO timestamps, subscription controls, bad JSON, nulls, variable block lengths, truncated groups, and incompatible schema.
- [x] Run `venv/bin/python -m pytest tests/test_market_data_protocol.py -q`, verify missing-module failure.
- [x] Implement decoder from archived upstream XML and sequence checks. Use `struct.unpack_from`, bounds checking, Decimal formatting, and group block lengths. Preserve raw snapshots rather than claiming full replay continuity.
- [x] Verify with fixed independently constructed binary frames and schema-layout assertions; rerun focused tests.

## Task 2: Durable bounded writer

**Files:** Create `src/backend/market_data/storage.py` and `tests/test_market_data_storage.py`.

**Interfaces:** `CaptureWriter(output_dir: Path, *, max_queue_bytes=8388608, max_file_bytes=134217728, max_total_bytes=1073741824, flush_interval=0.25)`; `await start(manifest: dict)`; synchronous `submit(record: dict)`; `await close()`; `await wait_failed()` (raises failure, otherwise waits); `CaptureStorageError(RuntimeError)`; `iter_capture_records(output_dir: Path)`.

`start` creates a fresh output directory, manifest and status; callers supply a unique run path. Do not reuse existing run directories. `submit` serializes with allow_nan=False, counts pending bytes until durable write, assigns increasing capture_sequence, and raises on capacity or stopped/failed writer. A dedicated task batches off-thread segment writes and fsyncs. It writes atomic status including durable_records/durable_bytes/closed/failed after successful flush. Segment names sort lexically. `close` drains accepted records and surfaces failures. A crashed final line is reported by replay, never silently accepted.

- [x] Add tests for order preservation, flush/drain, rotation, queue capacity, total cap, file error propagation and torn replay.
- [x] Verify expected missing-module failure, implement, then run `venv/bin/python -m pytest tests/test_market_data_storage.py -q`.

## Task 3: Transport, configuration, CLI and operator documentation

**Files:** Create `src/backend/market_data/recorder.py`, `scripts/record_market_data.py`, `docs/market_data_capture.md`, `tests/test_market_data_recorder.py`; modify `src/backend/config_loader.py`, `.env.example`, and `README.md`.

The public source definitions are isolated in `src/backend/market_data/feeds.py` so endpoint/subscription construction can be reviewed independently of lifecycle supervision.

**Interfaces:** `CaptureConfig.from_config(CONFIG)`; `build_sources(config) -> list[SourceSpec]`; `await record_market_data(config)`; CLI `venv/bin/python scripts/record_market_data.py --duration 30`.

- [x] Add tests for exact public subscriptions and Binance source separation, fresh-session reconnect and gap markers, REST snapshots before sequence interpretation, HTTP Retry-After, invalid frame quarantine, and cancellation/storage-failure shutdown.
- [x] Implement independent aiohttp workers, bounded message size, source metadata, jitter/backoff, graceful stop, and archive schema/provenance per run. Protocol or disk failures stop the capture; transport failures create explicit gaps and retry.
- [x] Add CLI and CONFIG defaults with positive-value validation and subset selection. Keep secrets out of manifests.
- [x] Run focused new tests and `tests/test_config_loader.py`, then backtest tests; run CLI help and a bounded public smoke capture if network access permits.
- [x] Review changes against the spec, resolve findings, update plan checkboxes, and report runnable command, validation and any live-source limitations.

## Pre-flight decisions

| Shared interface | Decision |
| --- | --- |
| Protocol -> recorder | Raw preservation happens in recorder; decode metadata is additive. |
| Writer -> recorder | submit is synchronous/nonblocking; wait_failed is monitored independently. |
| Recorder -> config | Only public capture settings enter manifest. |
| Schema -> decoder | Archive current upstream XML; negotiate stable 1:1 public layouts and reject mismatches. |
| Workspace | Approved implementation continues on a new feature branch in the user's IDE checkout; no additional worktree or dependency installation is needed. |
