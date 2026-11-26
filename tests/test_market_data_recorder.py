import asyncio
import json
import struct
from dataclasses import replace

import aiohttp
import pytest
from src.backend.market_data import recorder

from src.backend.market_data.recorder import (
    CaptureConfig,
    SourceAccessError,
    RetryableSourceError,
    _fetch_snapshot,
    _record_source,
    _DepthState,
    _connection,
    build_sources,
    record_market_data,
)
from src.backend.market_data.protocol import ProtocolError, SequenceGap
from src.backend.market_data.storage import CaptureStorageError, CaptureWriter, iter_capture_records


def bbo(price="100.00"):
    return json.dumps({"stream": "btcusdt@bookTicker", "data": {
        "u": 17, "s": "BTCUSDT", "b": price, "B": "1.0",
        "a": "101.00", "A": "2.0",
    }})


class MemoryWriter:
    def __init__(self):
        self.records = []

    def submit(self, record):
        self.records.append(record)


class FakeSocket:
    def __init__(self, frames, stop=None):
        self.frames = list(frames)
        self.stop = stop
        self.sent = []

    async def __aenter__(self):
        return self

    def __await__(self):
        return self.__aenter__().__await__()

    async def __aexit__(self, *args):
        pass

    async def send_json(self, payload):
        self.sent.append(payload)

    async def receive(self):
        await asyncio.sleep(0)
        if self.frames:
            return self.frames.pop(0)
        if self.stop:
            self.stop.set()
        return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)

    def exception(self):
        return None


class FakeContent:
    def __init__(self, raw):
        self.raw = raw.encode()

    async def read(self, size):
        result, self.raw = self.raw[:size], self.raw[size:]
        return result


class FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self.content = FakeContent(json.dumps(payload))
        self.status = status
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeSession:
    def __init__(self, sockets=(), responses=()):
        self.sockets = list(sockets)
        self.responses = list(responses)

    def ws_connect(self, *args, **kwargs):
        return self.sockets.pop(0)

    def get(self, *args, **kwargs):
        return self.responses.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def test_sources_use_separate_public_binance_routes_and_api_paradex_book():
    sources = {s.name: s for s in build_sources(CaptureConfig())}
    assert set(sources) == {"binance_spot", "binance_perp_book", "binance_perp_market", "paradex"}
    assert "/public/stream?" in sources["binance_perp_book"].url
    assert "/market/stream?" in sources["binance_perp_market"].url
    assert "@bookTicker" in sources["binance_spot"].url
    assert "@markPrice@1s" in sources["binance_perp_market"].url
    assert "@forceOrder" in sources["binance_perp_market"].url
    assert "sbeSchemaVersion=1" in sources["paradex"].url
    channels = [r["params"]["channel"] for r in sources["paradex"].subscribe_messages]
    assert channels == ["bbo.BTC-USD-PERP", "order_book.BTC-USD-PERP.snapshot@15@50ms", "trades.BTC-USD-PERP"]
    assert all(r["method"] == "subscribe" for r in sources["paradex"].subscribe_messages)
    assert "interactive" not in json.dumps(channels)


@pytest.mark.parametrize("changes", [
    {"sources": ()}, {"sources": ("private",)}, {"symbol": "BTCUSDT&secret=x"},
    {"max_queue_bytes": 0}, {"flush_interval": 0}, {"duration": -1},
])
def test_invalid_capture_config_fails_before_network(changes):
    with pytest.raises(ValueError):
        replace(CaptureConfig(), **changes)


@pytest.mark.asyncio
async def test_reconnect_records_gaps_and_new_sessions_without_assuming_bbo_sequence():
    stop = asyncio.Event()
    writer = MemoryWriter()
    config = replace(CaptureConfig(), sources=("binance_spot",), reconnect_base=0.001)
    source = replace(build_sources(config)[0], depth_url=None)
    sockets = [FakeSocket([aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, bbo(), None)]),
               FakeSocket([aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, bbo("100.01"), None)], stop)]
    await asyncio.wait_for(_record_source(FakeSession(sockets), source, writer, stop, config), 1)
    market = [r for r in writer.records if r.get("kind") == "bbo"]
    assert len(market) == 2
    assert market[0]["session_id"] != market[1]["session_id"]
    assert market[0]["raw"] == bbo()
    assert market[0]["received_at_ns"] > 0
    assert market[0]["received_monotonic_ns"] > 0
    assert market[0]["source_event_time_ns"] is None
    assert any(r["kind"] == "gap" for r in writer.records)


@pytest.mark.asyncio
async def test_malformed_frame_is_preserved_and_stops_worker():
    writer = MemoryWriter()
    config = replace(CaptureConfig(), sources=("binance_spot",))
    source = replace(build_sources(config)[0], depth_url=None)
    socket = FakeSocket([aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, "{broken", None)])
    with pytest.raises(ProtocolError):
        await _record_source(FakeSession([socket]), source, writer, asyncio.Event(), config)
    quarantine = next(r for r in writer.records if r["kind"] == "quarantine")
    assert quarantine["raw"] == "{broken"
    assert quarantine["error"]


@pytest.mark.asyncio
async def test_paradex_sends_only_public_subscriptions():
    stop = asyncio.Event()
    socket = FakeSocket([], stop)
    config = replace(CaptureConfig(), sources=("paradex",))
    source = build_sources(config)[0]
    await _record_source(FakeSession([socket]), source, MemoryWriter(), stop, config)
    assert socket.sent == list(source.subscribe_messages)


@pytest.mark.asyncio
async def test_snapshot_capture_preserves_native_units_and_watermark():
    snapshot = {"lastUpdateId": 42, "bids": [["100.0", "1.0"]], "asks": [["101.0", "2.0"]]}
    writer = MemoryWriter()
    source = build_sources(CaptureConfig())[0]
    result = await _fetch_snapshot(FakeSession(responses=[FakeResponse(snapshot)]), source, writer, "s1", CaptureConfig())
    assert result["lastUpdateId"] == 42
    record = writer.records[0]
    assert record["kind"] == "depth_snapshot"
    assert record["payload"]["asks"][0] == ["101.0", "2.0"]
    assert json.loads(record["raw"]) == snapshot
    assert record["source_event_time_ns"] is None


@pytest.mark.asyncio
async def test_futures_snapshot_preserves_exchange_clocks():
    source = build_sources(CaptureConfig())[1]
    snapshot = {"lastUpdateId": 42, "E": 1700000000004, "T": 1700000000001, "bids": [], "asks": []}
    writer = MemoryWriter()
    await _fetch_snapshot(FakeSession(responses=[FakeResponse(snapshot)]), source, writer, "s1", CaptureConfig())
    assert writer.records[0]["source_event_time_ns"] == 1700000000004000000
    assert writer.records[0]["source_transaction_time_ns"] == 1700000000001000000
    assert writer.records[0]["source_time_basis"] == "unix_ms"


@pytest.mark.asyncio
async def test_region_block_is_visible_and_not_retried():
    source = build_sources(CaptureConfig())[0]
    session = FakeSession(responses=[FakeResponse({"msg": "restricted"}, status=451)])
    with pytest.raises(SourceAccessError, match="451"):
        await _fetch_snapshot(session, source, MemoryWriter(), "s1", CaptureConfig())


@pytest.mark.asyncio
async def test_rate_limit_exposes_retry_after_without_fake_empty_book():
    source = build_sources(CaptureConfig())[0]
    writer = MemoryWriter()
    session = FakeSession(responses=[FakeResponse({}, status=429, headers={"Retry-After": "7"})])
    with pytest.raises(Exception) as caught:
        await _fetch_snapshot(session, source, writer, "s1", CaptureConfig())
    assert caught.value.retry_after == 7
    assert not any(r["kind"] == "depth_snapshot" for r in writer.records)


def depth(first, last):
    return json.dumps({"stream": "btcusdt@depth@100ms", "data": {
        "e": "depthUpdate", "E": 1700000000000, "s": "BTCUSDT", "U": first, "u": last,
        "b": [["100.0", "1.0"]], "a": [["101.0", "1.0"]],
    }})


def test_buffered_updates_get_explicit_alignment_classifications():
    state = _DepthState(build_sources(CaptureConfig())[0], 4096)
    assert state.observe(json.loads(depth(90, 99))["data"], 200, 1) == "awaiting_snapshot"
    state.observe(json.loads(depth(101, 102))["data"], 200, 3)
    result = state.install({"lastUpdateId": 100})
    assert result == {"snapshot_id": 100, "buffered_update_statuses": [
        {"frame_ordinal": 1, "status": "pre_snapshot"}, {"frame_ordinal": 3, "status": "bridged"},
    ]}


@pytest.mark.asyncio
async def test_snapshot_request_does_not_block_receiving_bbo():
    received = asyncio.Event()
    stop = asyncio.Event()

    class ObservingWriter(MemoryWriter):
        def submit(self, record):
            super().submit(record)
            if record["kind"] == "bbo":
                received.set()

    class DelayedContent(FakeContent):
        async def read(self, size):
            await received.wait()
            return await super().read(size)

    response = FakeResponse({"lastUpdateId": 100, "bids": [], "asks": []})
    response.content = DelayedContent(json.dumps({"lastUpdateId": 100, "bids": [], "asks": []}))
    socket = FakeSocket([aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, bbo(), None),
                         aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, depth(101, 101), None)], stop)
    writer = ObservingWriter()
    config = CaptureConfig(sources=("binance_spot",))
    await asyncio.wait_for(_record_source(FakeSession([socket], [response]), build_sources(config)[0], writer, stop, config), 1)
    kinds = [r["kind"] for r in writer.records]
    assert kinds.index("bbo") < kinds.index("depth_snapshot")
    assert "depth_alignment" in kinds


@pytest.mark.asyncio
async def test_depth_gap_reconnects_and_captures_another_snapshot():
    stop = asyncio.Event()
    config = CaptureConfig(sources=("binance_spot",), reconnect_base=0.001)
    source = build_sources(config)[0]
    frames = [aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, depth(101, 101), None),
              aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, depth(104, 104), None)]
    session = FakeSession([FakeSocket(frames), FakeSocket([
        aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, depth(201, 201), None)], stop)],
        [FakeResponse({"lastUpdateId": 100, "bids": [], "asks": []}),
         FakeResponse({"lastUpdateId": 200, "bids": [], "asks": []})])
    writer = MemoryWriter()
    await asyncio.wait_for(_record_source(session, source, writer, stop, config), 1)
    snapshots = [r for r in writer.records if r["kind"] == "depth_snapshot"]
    assert len(snapshots) == 2
    assert snapshots[0]["session_id"] != snapshots[1]["session_id"]
    assert any(r["kind"] == "gap" for r in writer.records)
    assert any(r.get("depth_sequence_status") == "gap" for r in writer.records)


@pytest.mark.asyncio
async def test_binance_control_error_is_fatal_not_a_healthy_subscription():
    config = CaptureConfig(sources=("binance_perp_market",))
    writer = MemoryWriter()
    raw = '{"code": 2, "msg": "invalid stream", "id": 1}'
    socket = FakeSocket([aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, raw, None)])
    with pytest.raises(SourceAccessError, match="invalid stream"):
        await _record_source(FakeSession([socket]), build_sources(config)[0], writer, asyncio.Event(), config)
    assert any(r.get("raw") == raw for r in writer.records)


@pytest.mark.asyncio
async def test_snapshot_invalid_utf8_is_quarantined():
    response = FakeResponse({})
    response.content.raw = b"\xff\xfe"
    writer = MemoryWriter()
    with pytest.raises(ProtocolError):
        await _fetch_snapshot(FakeSession(responses=[response]), build_sources(CaptureConfig())[0], writer, "s1", CaptureConfig())
    assert writer.records[0]["kind"] == "quarantine"
    assert writer.records[0]["raw_encoding"] == "base64"


def paradex_bbo():
    market = b"BTC-USD-PERP"
    body = struct.pack("<6q", 1700000000000000, 17, 10000000000, 1000000, 10100000000, 1000000)
    return struct.pack("<4H", len(body), 2, 1, 1) + body + bytes([len(market)]) + market


@pytest.mark.asyncio
async def test_bounded_capture_drains_files_and_records_run_outcome(tmp_path, monkeypatch):
    stop = asyncio.Event()
    socket = FakeSocket([aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, paradex_bbo(), None)], stop)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kwargs: FakeSession([socket]))
    config = CaptureConfig(output_dir=tmp_path, sources=("paradex",), duration=1)
    run_dir = await record_market_data(config, stop=stop)
    rows = list(iter_capture_records(run_dir))
    assert rows[-1]["kind"] == "run_end"
    assert rows[-1]["outcome"] == "completed"
    assert rows[-1]["source_message_counts"] == {"paradex": 1}
    assert (run_dir / "paradex_1_0.xml").exists()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["closed"] and not status["failed"]
    assert status["durable_records"] == len(rows)


@pytest.mark.asyncio
async def test_disk_failure_cancels_idle_receiver_and_surfaces(tmp_path, monkeypatch):
    class IdleAfterFrame(FakeSocket):
        async def receive(self):
            if self.frames:
                return self.frames.pop(0)
            await asyncio.Event().wait()

    def fail_write(self, rows, new_segment):
        raise OSError("simulated disk full")

    socket = IdleAfterFrame([aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, paradex_bbo(), None)])
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kwargs: FakeSession([socket]))
    monkeypatch.setattr(CaptureWriter, "_write_rows", fail_write)
    config = CaptureConfig(output_dir=tmp_path, sources=("paradex",), flush_interval=0.001)
    with pytest.raises(CaptureStorageError, match="disk full"):
        await asyncio.wait_for(record_market_data(config), 2)


@pytest.mark.asyncio
async def test_cancellation_drains_capture_and_closes_pending_receiver(tmp_path, monkeypatch):
    received = asyncio.Event()

    class IdleSocket(FakeSocket):
        async def receive(self):
            if self.frames:
                return self.frames.pop(0)
            received.set()
            await asyncio.Event().wait()

    socket = IdleSocket([aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, paradex_bbo(), None)])
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kwargs: FakeSession([socket]))
    task = asyncio.create_task(record_market_data(CaptureConfig(output_dir=tmp_path, sources=("paradex",))))
    await asyncio.wait_for(received.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    run_dir = next(tmp_path.iterdir())
    records = list(iter_capture_records(run_dir))
    assert any(r["kind"] == "bbo" for r in records)
    assert records[-1]["kind"] == "run_end"


def test_buffered_gap_retains_prior_classifications_and_offending_ordinal():
    state = _DepthState(build_sources(CaptureConfig())[0], 4096)
    for ordinal, interval in enumerate(((90, 99), (101, 101), (104, 104)), 1):
        state.observe(json.loads(depth(*interval))["data"], 200, ordinal)
    with pytest.raises(SequenceGap) as caught:
        state.install({"lastUpdateId": 100})
    assert caught.value.alignment == {
        "snapshot_id": 100, "gap_frame_ordinal": 3,
        "buffered_update_statuses": [
            {"frame_ordinal": 1, "status": "pre_snapshot"},
            {"frame_ordinal": 2, "status": "bridged"},
            {"frame_ordinal": 3, "status": "gap"},
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [1007, 1009])
async def test_websocket_protocol_failure_stops_and_records_unavailable_raw_bytes(code):
    config = CaptureConfig(sources=("paradex",))
    error = aiohttp.WebSocketError(code, "decoder rejected frame")
    socket = FakeSocket([aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, error, None)])
    writer = MemoryWriter()
    with pytest.raises(ProtocolError, match=str(code)):
        await asyncio.wait_for(_record_source(FakeSession([socket]), build_sources(config)[0], writer,
                                              asyncio.Event(), config), 1)
    quarantine = next(r for r in writer.records if r["kind"] == "quarantine")
    assert quarantine["close_code"] == code
    assert quarantine["raw_unavailable"] and quarantine["raw"] is None
    assert not any(r["kind"] == "gap" for r in writer.records)


@pytest.mark.asyncio
async def test_simultaneous_snapshot_access_failure_wins_over_socket_close():
    class ImmediateClose(FakeSocket):
        async def receive(self):
            return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)

    config = CaptureConfig(sources=("binance_spot",))
    source = build_sources(config)[0]
    session = FakeSession([ImmediateClose([])], [FakeResponse({}, status=451)])
    with pytest.raises(SourceAccessError, match="451"):
        await _connection(session, source, MemoryWriter(), asyncio.Event(), config, "s1", {"messages": 0})


@pytest.mark.asyncio
async def test_opening_websocket_handshake_has_a_deadline():
    cancelled = asyncio.Event()

    class StalledHandshake:
        def __await__(self):
            async def stall():
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return stall().__await__()

    config = CaptureConfig(sources=("paradex",), handshake_timeout=0.005)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(_connection(FakeSession([StalledHandshake()]), build_sources(config)[0],
                                            MemoryWriter(), asyncio.Event(), config, "s1", {"messages": 0}), 1)
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_fatal_worker_finishing_after_stop_is_not_reported_completed(tmp_path, monkeypatch):
    stop = asyncio.Event()
    release_failure = asyncio.Event()
    real_wait = asyncio.wait

    async def fail_after_stop(session, source, writer, stop, config, activity):
        activity[source.name] += 1
        stop.set()
        await release_failure.wait()
        raise ProtocolError("fatal snapshot during shutdown")

    async def finish_pending_before_returning(tasks, **kwargs):
        done, pending = await real_wait(tasks, **kwargs)
        release_failure.set()
        await asyncio.sleep(0)
        return done, pending

    monkeypatch.setattr(recorder, "_record_source", fail_after_stop)
    monkeypatch.setattr(asyncio, "wait", finish_pending_before_returning)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kwargs: FakeSession())
    config = CaptureConfig(output_dir=tmp_path, sources=("paradex",))
    with pytest.raises(ProtocolError, match="during shutdown"):
        await record_market_data(config, stop=stop)
    records = list(iter_capture_records(next(tmp_path.iterdir())))
    assert records[-1]["outcome"] == "failed"
    assert "fatal snapshot" in records[-1]["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected,kind", [
    ("{broken", ProtocolError, "quarantine"), (bbo(), RetryableSourceError, "bbo"),
])
async def test_simultaneous_snapshot_retry_preserves_received_frame_and_fatal_priority(raw, expected, kind):
    class ImmediateFrame(FakeSocket):
        async def receive(self):
            return self.frames.pop(0)

    config = CaptureConfig(sources=("binance_spot",))
    writer = MemoryWriter()
    socket = ImmediateFrame([aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, raw, None)])
    session = FakeSession([socket], [FakeResponse({}, status=429)])
    with pytest.raises(expected):
        await _connection(session, build_sources(config)[0], writer, asyncio.Event(), config, "s1", {"messages": 0})
    assert any(r["kind"] == kind and r.get("raw") == raw for r in writer.records)


@pytest.mark.asyncio
async def test_late_snapshot_retry_cannot_mask_active_fatal_frame_error():
    release = asyncio.Event()

    class DelayedRetry(FakeResponse):
        async def __aenter__(self):
            await release.wait()
            return self

    class ReleaseOnClose(FakeSocket):
        async def receive(self):
            return self.frames.pop(0)

        async def __aexit__(self, *args):
            release.set()
            await asyncio.sleep(0)

    config = CaptureConfig(sources=("binance_spot",))
    socket = ReleaseOnClose([aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, "{broken", None)])
    writer = MemoryWriter()
    with pytest.raises(ProtocolError):
        await _connection(FakeSession([socket], [DelayedRetry({}, status=429)]), build_sources(config)[0],
                          writer, asyncio.Event(), config, "s1", {"messages": 0})
    assert any(r["kind"] == "quarantine" for r in writer.records)


@pytest.mark.asyncio
async def test_snapshot_retry_after_survives_simultaneous_socket_close():
    class ImmediateClose(FakeSocket):
        async def receive(self):
            return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)

    config = CaptureConfig(sources=("binance_spot",))
    session = FakeSession([ImmediateClose([])], [FakeResponse({}, status=429, headers={"Retry-After": "120"})])
    with pytest.raises(RetryableSourceError) as caught:
        await _connection(session, build_sources(config)[0], MemoryWriter(), asyncio.Event(), config, "s1", {"messages": 0})
    assert caught.value.retry_after == 120
