"""Bounded NDJSON persistence and replay for a single capture run."""

import asyncio
import json
import math
import os
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn


class CaptureStorageError(RuntimeError):
    """Capture data could not be accepted, persisted, or replayed."""


def _json_line(value: dict) -> bytes:
    return (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode()


def _sync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic(path: Path, data: bytes) -> None:
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)
    _sync_directory(path.parent)


class CaptureWriter:
    """Serialize records in submission order and flush them off the event loop."""

    def __init__(
        self,
        output_dir: Path,
        *,
        max_queue_bytes: int = 8_388_608,
        max_file_bytes: int = 134_217_728,
        max_total_bytes: int = 1_073_741_824,
        flush_interval: float = 0.25,
    ) -> None:
        if min(max_queue_bytes, max_file_bytes, max_total_bytes) <= 0:
            raise ValueError("capture byte limits must be positive")
        if not math.isfinite(flush_interval) or flush_interval <= 0:
            raise ValueError("capture flush interval must be positive and finite")
        self.output_dir = Path(output_dir)
        self._max_queue_bytes = max_queue_bytes
        self._max_file_bytes = max_file_bytes
        self._max_total_bytes = max_total_bytes
        self._flush_interval = flush_interval
        self._pending: deque[bytes] = deque()
        self._pending_bytes = 0
        self._accepted_bytes = 0
        self._next_sequence = 1
        self._segment_number = 0
        self._segment_bytes = 0
        self._durable_records = 0
        self._durable_bytes = 0
        self._starting = False
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._failed = asyncio.Event()
        self._failure: Exception | None = None

    async def start(self, manifest: dict) -> None:
        self._raise_if_failed()
        if self._starting or self._task is not None:
            raise CaptureStorageError("capture writer already started")
        self._starting = True
        try:
            await asyncio.to_thread(self._initialize, manifest)
        except (OSError, TypeError, ValueError, RecursionError) as exc:
            self._fail(exc)
            raise CaptureStorageError(f"could not initialize capture: {exc}") from exc
        finally:
            self._starting = False
        self._task = asyncio.create_task(self._run())

    def submit(self, record: dict) -> None:
        self._raise_if_failed()
        if self._task is None or self._stop.is_set():
            raise CaptureStorageError("capture writer is not accepting records")

        try:
            line = _json_line(dict(record, capture_sequence=self._next_sequence))
        except (TypeError, ValueError, RecursionError) as exc:
            error = CaptureStorageError(f"could not serialize capture record: {exc}")
            self._fail(error)
            raise error from exc

        size = len(line)
        if size > self._max_file_bytes:
            self._reject("capture record exceeds the segment byte limit")
        if self._accepted_bytes + size > self._max_total_bytes:
            self._reject("capture total byte limit exceeded")
        if self._pending_bytes + size > self._max_queue_bytes:
            self._reject("capture queue byte limit exceeded")

        self._pending.append(line)
        self._pending_bytes += size
        self._accepted_bytes += size
        self._next_sequence += 1

    async def close(self) -> None:
        if self._task is not None:
            self._stop.set()
            await asyncio.shield(self._task)
        self._raise_if_failed()

    async def wait_failed(self) -> None:
        await self._failed.wait()
        self._raise_if_failed()

    def _fail(self, error: Exception) -> None:
        if self._failure is None:
            self._failure = error
            self._failed.set()
            self._stop.set()

    def _reject(self, message: str) -> NoReturn:
        error = CaptureStorageError(message)
        self._fail(error)
        raise error

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise CaptureStorageError(str(self._failure)) from self._failure

    def _initialize(self, manifest: dict) -> None:
        manifest_data = _json_line(manifest)
        parent = self.output_dir.parent
        directories_to_sync = [parent]
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
            directories_to_sync.append(parent)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for directory in directories_to_sync:
            _sync_directory(directory)
        _write_atomic(self.output_dir / "manifest.json", manifest_data)
        self._write_status()

    async def _run(self) -> None:
        try:
            while True:
                if not self._stop.is_set():
                    try:
                        await asyncio.wait_for(self._stop.wait(), self._flush_interval)
                    except TimeoutError:
                        pass
                await self._flush()
                if self._stop.is_set() and not self._pending:
                    break
            if self._failure is None:
                await asyncio.to_thread(self._write_status, closed=True)
        except Exception as exc:
            self._fail(exc)
        finally:
            self._pending.clear()
            self._pending_bytes = 0

        if self._failure is not None:
            try:
                await asyncio.to_thread(self._write_status, failed=True)
            except OSError:
                pass

    async def _flush(self) -> None:
        batch, self._pending = self._pending, deque()
        while batch:
            new_segment = (
                self._segment_number == 0
                or self._segment_bytes + len(batch[0]) > self._max_file_bytes
            )
            if new_segment:
                self._segment_number += 1
                self._segment_bytes = 0

            rows = []
            size = 0
            remaining = self._max_file_bytes - self._segment_bytes
            while batch and size + len(batch[0]) <= remaining:
                row = batch.popleft()
                rows.append(row)
                size += len(row)

            await asyncio.to_thread(self._write_rows, rows, new_segment)
            self._segment_bytes += size
            self._durable_records += len(rows)
            self._durable_bytes += size
            self._pending_bytes -= size
            await asyncio.to_thread(self._write_status, failed=self._failure is not None)

    def _write_rows(self, rows: list[bytes], new_segment: bool) -> None:
        path = self.output_dir / f"records-{self._segment_number:020d}.ndjson"
        with path.open("xb" if new_segment else "ab") as file:
            file.writelines(rows)
            file.flush()
            os.fsync(file.fileno())
        if new_segment:
            _sync_directory(self.output_dir)

    def _write_status(self, *, closed: bool = False, failed: bool = False) -> None:
        status = {
            "durable_records": self._durable_records,
            "durable_bytes": self._durable_bytes,
            "closed": closed,
            "failed": failed,
        }
        _write_atomic(self.output_dir / "status.json", _json_line(status))


def _read_replay_status(path: Path) -> tuple[bool, int, int] | None:
    try:
        status = json.loads(path.read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValueError, RecursionError) as exc:
        raise CaptureStorageError(f"could not read capture status {path}: {exc}") from exc

    if not isinstance(status, dict) or type(status.get("closed")) is not bool:
        raise CaptureStorageError(f"invalid capture status at {path}: expected boolean closed")
    durable_records = status.get("durable_records")
    durable_bytes = status.get("durable_bytes")
    if any(type(total) is not int or total < 0 for total in (durable_records, durable_bytes)):
        raise CaptureStorageError(f"invalid capture status at {path}: expected integer durable totals")
    return status["closed"], durable_records, durable_bytes


def iter_capture_records(output_dir: Path) -> Iterator[dict]:
    """Yield complete NDJSON records in order and verify durable totals on exhaustion."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise CaptureStorageError(f"capture directory does not exist: {output_dir}")
    status_path = output_dir / "status.json"
    status = _read_replay_status(status_path)
    next_sequence = 1
    replayed_bytes = 0
    for segment in sorted(output_dir.glob("records-*.ndjson")):
        try:
            with segment.open("rb") as file:
                for line_number, line in enumerate(file, start=1):
                    location = f"{segment}:{line_number}"
                    if not line.endswith(b"\n"):
                        raise CaptureStorageError(f"unterminated capture record at {location}")
                    try:
                        record = json.loads(line)
                    except (ValueError, RecursionError) as exc:
                        raise CaptureStorageError(f"invalid capture record at {location}") from exc
                    if not isinstance(record, dict):
                        raise CaptureStorageError(f"expected a capture object at {location}")
                    sequence = record.get("capture_sequence")
                    if type(sequence) is not int or sequence <= 0:
                        raise CaptureStorageError(
                            f"expected a positive integer capture_sequence at {location}"
                        )
                    if sequence != next_sequence:
                        raise CaptureStorageError(
                            f"capture sequence discontinuity at {location}: "
                            f"expected {next_sequence}, got {sequence}"
                        )
                    next_sequence += 1
                    replayed_bytes += len(line)
                    yield record
        except OSError as exc:
            raise CaptureStorageError(f"could not read capture segment: {exc}") from exc

    if status is not None:
        closed, durable_records, durable_bytes = status
        replayed_records = next_sequence - 1
        missing_durable_data = replayed_records < durable_records or replayed_bytes < durable_bytes
        closed_run_mismatch = closed and (replayed_records, replayed_bytes) != (
            durable_records,
            durable_bytes,
        )
        if missing_durable_data or closed_run_mismatch:
            raise CaptureStorageError(
                f"capture replay disagrees with durable totals in {status_path}: "
                f"found {replayed_records} records and {replayed_bytes} bytes, "
                f"status reports {durable_records} records and {durable_bytes} bytes"
            )
