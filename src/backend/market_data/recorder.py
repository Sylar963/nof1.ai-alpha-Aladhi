import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from pathlib import Path
from uuid import uuid4

import aiohttp

from src.backend.market_data.feeds import SOURCE_NAMES, SourceSpec, build_sources
from src.backend.market_data.protocol import BinanceDepthSequence, ProtocolError, SequenceGap, decode_frame
from src.backend.market_data.storage import CaptureStorageError, CaptureWriter


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureConfig:
    output_dir: Path = Path("data/market_capture")
    symbol: str = "BTCUSDT"
    paradex_market: str = "BTC-USD-PERP"
    sources: tuple[str, ...] = SOURCE_NAMES
    duration: float | None = None
    max_queue_bytes: int = 8 * 1024 * 1024
    max_file_bytes: int = 128 * 1024 * 1024
    max_total_bytes: int = 1024 * 1024 * 1024
    flush_interval: float = 0.25
    max_frame_bytes: int = 2 * 1024 * 1024
    reconnect_base: float = 1.0
    reconnect_max: float = 30.0
    handshake_timeout: float = 15.0

    def __post_init__(self):
        if not re.fullmatch(r"[A-Z0-9]{3,30}", self.symbol):
            raise ValueError("symbol must be an uppercase Binance symbol")
        if not re.fullmatch(r"[A-Z0-9-]{3,80}", self.paradex_market):
            raise ValueError("paradex_market must be an uppercase native market")
        if not self.sources or len(set(self.sources)) != len(self.sources) or set(self.sources) - set(SOURCE_NAMES):
            raise ValueError("sources must be a nonempty unique subset of public capture sources")
        for field in ("max_queue_bytes", "max_file_bytes", "max_total_bytes", "max_frame_bytes"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        for field in ("flush_interval", "reconnect_base", "reconnect_max", "handshake_timeout", "duration"):
            value = getattr(self, field)
            if field == "duration" and value is None:
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if self.reconnect_base > self.reconnect_max:
            raise ValueError("reconnect_base cannot exceed reconnect_max")

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            output_dir=Path(config["capture_output_dir"]),
            symbol=config["capture_symbol"], paradex_market=config["capture_paradex_market"],
            sources=tuple(s.strip() for s in config["capture_sources"].split(",")),
            max_queue_bytes=config["capture_max_queue_bytes"], max_file_bytes=config["capture_max_file_bytes"],
            max_total_bytes=config["capture_max_total_bytes"], flush_interval=config["capture_flush_interval"],
        )


class SourceAccessError(RuntimeError):
    pass


class RetryableSourceError(ConnectionError):
    def __init__(self, message: str, retry_after: float = 0):
        super().__init__(message)
        self.retry_after = retry_after


def _envelope(source: SourceSpec, session_id: str, kind: str, **fields) -> dict:
    return {
        "capture_schema_version": 1, "source": source.name, "session_id": session_id,
        "frame_ordinal": None,
        "received_at_ns": time.time_ns(), "received_monotonic_ns": time.monotonic_ns(),
        "kind": kind, **fields,
    }


def _retry_after(value: str | None) -> float:
    if value is None:
        return 0
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return 0
    return max(0, delay) if math.isfinite(delay) else 0


def _check_http(status: int, headers, source: str):
    if status in (400, 401, 403, 404, 451):
        raise SourceAccessError(f"{source}: HTTP {status}; check endpoint/access, no automatic bypass")
    if status != 200:
        raise RetryableSourceError(f"{source}: HTTP {status}", _retry_after(headers.get("Retry-After")))


async def _fetch_snapshot(session, source: SourceSpec, writer, session_id: str, config: CaptureConfig) -> dict:
    async with session.get(source.depth_url, timeout=aiohttp.ClientTimeout(total=15)) as response:
        _check_http(response.status, response.headers, source.name)
        chunks = bytearray()
        oversized = False
        while chunk := await response.content.read(65536):
            chunks.extend(chunk)
            if len(chunks) > config.max_frame_bytes:
                oversized = True
                break
        record = _envelope(source, session_id, "depth_snapshot", encoding="json",
                           raw=base64.b64encode(chunks).decode("ascii"), raw_encoding="base64",
                           source_event_time_ns=None, source_transaction_time_ns=None,
                           source_time_basis=None, market=source.market, request_url=source.depth_url)
    try:
        if oversized:
            record["raw_truncated"] = True
            raise ValueError("depth snapshot exceeds frame limit; prefix preserved")
        record.update(raw=bytes(chunks).decode("utf-8"), raw_encoding="utf-8")
        payload = json.loads(record["raw"])
        if not isinstance(payload, dict) or type(payload.get("lastUpdateId")) is not int or payload["lastUpdateId"] < 0:
            raise ValueError("missing or invalid lastUpdateId")
        for side in ("bids", "asks"):
            if not isinstance(payload.get(side), list):
                raise ValueError(f"missing {side}")
            for level in payload[side]:
                if not isinstance(level, list) or len(level) != 2 or not all(isinstance(v, str) for v in level):
                    raise ValueError("invalid price level")
                price, quantity = map(Decimal, level)
                if not price.is_finite() or not quantity.is_finite() or price <= 0 or quantity < 0:
                    raise ValueError("invalid price or quantity")
        record.update(payload=payload, source_sequence=payload["lastUpdateId"])
        for native, normalized in (("E", "source_event_time_ns"), ("T", "source_transaction_time_ns")):
            value = payload.get(native)
            if value is not None:
                if type(value) is not int or value < 0:
                    raise ValueError(f"invalid snapshot {native} timestamp")
                record[normalized] = value * 1_000_000
                record["source_time_basis"] = "unix_ms"
    except (ValueError, InvalidOperation, UnicodeError) as exc:
        writer.submit({**record, "kind": "quarantine", "error": str(exc)})
        raise ProtocolError(f"invalid {source.name} depth snapshot: {exc}") from exc
    writer.submit(record)
    return payload


class _BufferedSequenceGap(SequenceGap):
    def __init__(self, message, alignment):
        super().__init__(message)
        self.alignment = alignment


class _DepthState:
    def __init__(self, source: SourceSpec, limit: int):
        self.source = source
        self.limit = limit
        self.buffer = []
        self.buffer_bytes = 0
        self.tracker = None

    def observe(self, payload: dict, raw_bytes: int, frame_ordinal: int) -> str:
        if self.tracker is not None:
            return self.tracker.observe(payload)
        if self.buffer_bytes + raw_bytes > self.limit:
            raise SequenceGap("snapshot alignment buffer exceeded; obtain a fresh snapshot")
        self.buffer.append((payload, frame_ordinal))
        self.buffer_bytes += raw_bytes
        return "awaiting_snapshot"

    def install(self, snapshot: dict):
        self.tracker = BinanceDepthSequence(snapshot["lastUpdateId"], futures=self.source.futures)
        alignment = {"snapshot_id": snapshot["lastUpdateId"], "buffered_update_statuses": []}
        for payload, ordinal in self.buffer:
            try:
                status = self.tracker.observe(payload)
            except SequenceGap as exc:
                alignment["buffered_update_statuses"].append({"frame_ordinal": ordinal, "status": "gap"})
                alignment["gap_frame_ordinal"] = ordinal
                raise _BufferedSequenceGap(str(exc), alignment) from exc
            alignment["buffered_update_statuses"].append({"frame_ordinal": ordinal, "status": status})
        self.buffer.clear()
        self.buffer_bytes = 0
        return alignment


async def _receive(ws):
    message = await ws.receive()
    return message, time.time_ns(), time.monotonic_ns()


def _choose_failure(values):
    failures = [value for value in values if isinstance(value, BaseException)]
    fatal = next((value for value in failures if isinstance(value, (ProtocolError, SourceAccessError, CaptureStorageError))), None)
    delayed = max((value for value in failures if isinstance(value, RetryableSourceError)),
                  key=lambda value: value.retry_after, default=None)
    ordinary = next((value for value in failures if not isinstance(value, asyncio.CancelledError)), None)
    return fatal or delayed or ordinary or (failures[0] if failures else None)


async def _cancel_tasks(tasks):
    tasks = [task for task in tasks if task is not None]
    for task in tasks:
        if not task.done():
            task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [value for value in results if isinstance(value, BaseException)
                and not isinstance(value, asyncio.CancelledError)]
    return _choose_failure(failures)


def _check_ws_error(source, writer, session_id, code, error):
    if code in (1002, 1003, 1007, 1009):
        message = f"{source.name}: WebSocket protocol error {code}: {error}"
        writer.submit(_envelope(source, session_id, "quarantine", close_code=code,
                                error=message, raw=None, raw_unavailable=True))
        raise ProtocolError(message)


def _capture_frame(source, writer, session_id, ordinal, raw, received_at, received_monotonic, depth):
    record = _envelope(source, session_id, "frame", frame_ordinal=ordinal,
                       received_at_ns=received_at, received_monotonic_ns=received_monotonic,
                       raw=base64.b64encode(raw).decode("ascii") if isinstance(raw, bytes) else raw,
                       encoding="sbe-base64" if isinstance(raw, bytes) else "json")
    try:
        decoded = decode_frame(source.name, raw)
        if decoded["kind"] != "control" and decoded["market"] != source.market:
            raise ProtocolError("frame belongs to an unexpected market")
        if source.name != "paradex" and decoded["kind"] != "control" and decoded["channel"] not in source.channels:
            raise ProtocolError("frame belongs to an unexpected channel")
        record.update(decoded)
        record["raw_encoding"] = "base64" if isinstance(raw, bytes) else "utf-8"
        if source.name == "paradex" and decoded["kind"] == "depth" and decoded["payload"]["pkg_type"] != "SNAPSHOT":
            raise ProtocolError("expected Paradex replacement snapshot, received delta")
        if decoded["kind"] == "depth" and depth is not None:
            try:
                record["depth_sequence_status"] = depth.observe(decoded["payload"], len(raw), ordinal)
            except SequenceGap:
                writer.submit({**record, "depth_sequence_status": "gap"})
                raise
    except ProtocolError as exc:
        writer.submit({**record, "kind": "quarantine", "error": str(exc)})
        raise
    writer.submit(record)
    if decoded["kind"] == "control":
        payload = decoded["payload"]
        if isinstance(payload, dict) and payload.get("error") is not None:
            raise SourceAccessError(f"{source.name}: subscription rejected: {payload['error']}")
        if isinstance(payload, dict) and "code" in payload and "msg" in payload:
            raise SourceAccessError(f"{source.name}: subscription rejected: {payload['code']} {payload['msg']}")
        if isinstance(payload, dict) and (payload.get("e") == "serverShutdown" or "!serverShutdown" in str(decoded["channel"])):
            raise ConnectionError(f"{source.name}: planned server shutdown")
    return decoded


async def _connection(session, source, writer, stop, config, session_id, progress):
    snapshot_task = None
    receive_task = None
    depth = _DepthState(source, config.max_queue_bytes) if source.depth_url else None
    ordinal = 0
    active_error = None

    def harvest_completed():
        nonlocal receive_task, snapshot_task, ordinal
        errors = []
        if receive_task is not None and receive_task.done():
            task, receive_task = receive_task, None
            try:
                message, wall, monotonic = task.result()
                if message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    _check_ws_error(source, writer, session_id, message.data, message.extra)
                    if not stop.is_set():
                        raise ConnectionError(f"{source.name}: socket closed")
                elif message.type == aiohttp.WSMsgType.ERROR:
                    error = message.data if isinstance(message.data, Exception) else ws.exception()
                    _check_ws_error(source, writer, session_id, getattr(error, "code", None), str(error))
                    raise ConnectionError(f"{source.name}: {error}")
                elif message.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    ordinal += 1
                    decoded = _capture_frame(source, writer, session_id, ordinal, message.data, wall, monotonic, depth)
                    if decoded["kind"] != "control":
                        progress["messages"] += 1
            except BaseException as exc:
                errors.append(exc)
        if snapshot_task is not None and snapshot_task.done():
            task, snapshot_task = snapshot_task, None
            try:
                alignment = depth.install(task.result())
                writer.submit(_envelope(source, session_id, "depth_alignment", state="snapshot_installed", **alignment))
            except _BufferedSequenceGap as exc:
                errors.append(exc)
                try:
                    writer.submit(_envelope(source, session_id, "depth_alignment", state="gap", **exc.alignment))
                except BaseException as storage_error:
                    errors.append(storage_error)
            except BaseException as exc:
                errors.append(exc)
        return errors

    try:
        async with asyncio.timeout(config.handshake_timeout):
            ws = await session.ws_connect(source.url, heartbeat=20, autoping=True, max_msg_size=config.max_frame_bytes,
                                          timeout=aiohttp.ClientWSTimeout(ws_close=5))
        async with ws:
            writer.submit(_envelope(source, session_id, "connected", url=source.url))
            for message in source.subscribe_messages:
                await ws.send_json(message)
            if source.depth_url:
                snapshot_task = asyncio.create_task(_fetch_snapshot(session, source, writer, session_id, config))
            receive_task = asyncio.create_task(_receive(ws))
            while not stop.is_set():
                waiting = [receive_task] + ([snapshot_task] if snapshot_task is not None else [])
                await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                failure = _choose_failure(harvest_completed())
                if failure is not None:
                    raise failure
                if receive_task is None and not stop.is_set():
                    receive_task = asyncio.create_task(_receive(ws))
    except BaseException as exc:
        active_error = exc
    finally:
        late_errors = harvest_completed()
        cleanup_error = await _cancel_tasks((receive_task, snapshot_task))
    failure = _choose_failure([active_error, cleanup_error, *late_errors])
    if failure is not None:
        raise failure


async def _record_source(session, source, writer, stop, config, activity=None):
    failures = 0
    while not stop.is_set():
        session_id = str(uuid4())
        progress = {"messages": 0}
        writer.submit(_envelope(source, session_id, "connection_attempt", url=source.url))
        try:
            await _connection(session, source, writer, stop, config, session_id, progress)
        except (ConnectionError, aiohttp.ClientError, TimeoutError, SequenceGap) as exc:
            if isinstance(exc, aiohttp.WSServerHandshakeError):
                try:
                    _check_http(exc.status, exc.headers or {}, source.name)
                except RetryableSourceError as translated:
                    exc = translated
            writer.submit(_envelope(source, session_id, "gap", error=str(exc),
                                    history_recovered=False, frames_before_disconnect=progress["messages"]))
            if stop.is_set():
                return
            failures = 0 if progress["messages"] else min(failures + 1, 16)
            delay = min(config.reconnect_max, config.reconnect_base * 2 ** failures)
            delay = max(getattr(exc, "retry_after", 0), random.uniform(delay / 2, delay))
            logger.warning("%s disconnected: %s; retrying in %.2fs", source.name, exc, delay)
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except TimeoutError:
                pass
        finally:
            if activity is not None:
                activity[source.name] += progress["messages"]
            writer.submit(_envelope(source, session_id, "connection_end", market_messages=progress["messages"]))


def _archive_schema(run_dir: Path):
    schema_dir = Path(__file__).with_name("schemas")
    for name in ("paradex_1_0.xml", "LICENSE.paradex", "README.md"):
        with (run_dir / name).open("xb") as file:
            file.write((schema_dir / name).read_bytes())
            file.flush()
            os.fsync(file.fileno())


async def record_market_data(config: CaptureConfig, *, stop: asyncio.Event | None = None) -> Path:
    stop = stop or asyncio.Event()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:12]
    run_dir = config.output_dir / run_id
    sources = build_sources(config)
    settings = asdict(config)
    settings["output_dir"] = str(config.output_dir)
    schema = Path(__file__).with_name("schemas") / "paradex_1_0.xml"
    manifest = {"capture_schema_version": 1, "run_id": run_id, "settings": settings,
                "sources": [asdict(s) for s in sources], "sbe_schema_sha256": hashlib.sha256(schema.read_bytes()).hexdigest(),
                "sbe_negotiated": "1:1", "receive_clock": "application receive; UTC ns and monotonic ns",
                "historical_replay": False, "liquidations_complete": False}
    writer = CaptureWriter(run_dir, max_queue_bytes=config.max_queue_bytes, max_file_bytes=config.max_file_bytes,
                           max_total_bytes=config.max_total_bytes, flush_interval=config.flush_interval)
    await writer.start(manifest)
    logger.info("Recording public market data to %s", run_dir)
    timer = asyncio.get_running_loop().call_later(config.duration, stop.set) if config.duration else None
    activity = {source.name: 0 for source in sources}
    tasks = []
    error = None
    try:
        await asyncio.to_thread(_archive_schema, run_dir)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, connect=15, sock_read=15)) as session:
            tasks = [asyncio.create_task(_record_source(session, source, writer, stop, config, activity)) for source in sources]
            tasks += [asyncio.create_task(writer.wait_failed()), asyncio.create_task(stop.wait())]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if not stop.is_set():
                raise RuntimeError("capture worker ended unexpectedly")
    except BaseException as exc:
        error = exc
    finally:
        stop.set()
        if timer:
            timer.cancel()
        cleanup_error = await _cancel_tasks(tasks)
        error = _choose_failure([error, cleanup_error])
        missing = [name for name, count in activity.items() if not count]
        if error is None and missing:
            error = RuntimeError("no market data received from: " + ", ".join(missing))
        try:
            writer.submit({"capture_schema_version": 1, "source": None, "session_id": None,
                           "frame_ordinal": None, "kind": "run_end", "received_at_ns": time.time_ns(),
                           "received_monotonic_ns": time.monotonic_ns(), "outcome": "failed" if error else "completed",
                           "source_message_counts": activity, "error": str(error) if error else None})
        except BaseException as exc:
            error = error or exc
        try:
            await writer.close()
        except BaseException as exc:
            error = error or exc
    if error:
        raise error
    return run_dir
