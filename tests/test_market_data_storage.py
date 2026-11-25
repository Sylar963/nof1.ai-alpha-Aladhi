import asyncio
import json
import os
import threading

import pytest

from src.backend.market_data.storage import (
    CaptureStorageError,
    CaptureWriter,
    iter_capture_records,
)


def read_status(output_dir):
    return json.loads((output_dir / "status.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_close_drains_records_in_sequence_across_segments(tmp_path):
    output_dir = tmp_path / "run"
    writer = CaptureWriter(output_dir, max_file_bytes=100, flush_interval=60)
    manifest = {"schema_version": 1}
    await writer.start(manifest)
    records = [
        {"raw": "x" * 30, "capture_sequence": -1, "frame": frame}
        for frame in range(4)
    ]
    for record in records:
        writer.submit(record)
    assert read_status(output_dir)["durable_records"] == 0

    await writer.close()

    segments = sorted(output_dir.glob("records-*.ndjson"))
    assert len(segments) == 4
    assert all(segment.stat().st_size <= 100 for segment in segments)
    assert json.loads((output_dir / "manifest.json").read_text()) == manifest
    assert list(iter_capture_records(output_dir)) == [
        dict(record, capture_sequence=sequence)
        for sequence, record in enumerate(records, start=1)
    ]
    assert all(record["capture_sequence"] == -1 for record in records)
    assert read_status(output_dir) == {
        "durable_records": 4,
        "durable_bytes": sum(segment.stat().st_size for segment in segments),
        "closed": True,
        "failed": False,
    }


@pytest.mark.asyncio
async def test_start_keeps_an_existing_run_untouched(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    manifest = output_dir / "manifest.json"
    manifest.write_text('{"previous":true}', encoding="utf-8")
    writer = CaptureWriter(output_dir)

    with pytest.raises(CaptureStorageError, match="exists"):
        await writer.start({})

    assert list(output_dir.iterdir()) == [manifest]
    assert json.loads(manifest.read_text(encoding="utf-8")) == {"previous": True}


@pytest.mark.asyncio
async def test_periodic_flush_frees_capacity_for_an_append(tmp_path):
    output_dir = tmp_path / "run"
    writer = CaptureWriter(output_dir, max_queue_bytes=90, flush_interval=0.01)
    await writer.start({})
    writer.submit({"raw": "a" * 30})

    async with asyncio.timeout(5):
        while read_status(output_dir)["durable_records"] == 0:
            await asyncio.sleep(0.01)
    writer.submit({"raw": "b" * 30})
    await writer.close()

    assert len(list(output_dir.glob("records-*.ndjson"))) == 1
    assert list(iter_capture_records(output_dir)) == [
        {"raw": "a" * 30, "capture_sequence": 1},
        {"raw": "b" * 30, "capture_sequence": 2},
    ]


@pytest.mark.asyncio
async def test_queue_bytes_stay_reserved_until_fsync(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    writer = CaptureWriter(output_dir, max_queue_bytes=90, flush_interval=0.01)
    await writer.start({})
    loop = asyncio.get_running_loop()
    entered_fsync = asyncio.Event()
    release_fsync = threading.Event()
    original_fsync = os.fsync

    def block_record_fsync(fd):
        if not entered_fsync.is_set():
            loop.call_soon_threadsafe(entered_fsync.set)
            if not release_fsync.wait(timeout=5):
                raise OSError("test timed out waiting for fsync release")
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", block_record_fsync)
    writer.submit({"raw": "a" * 30})
    try:
        await asyncio.wait_for(entered_fsync.wait(), timeout=5)
        assert read_status(output_dir)["durable_records"] == 0
        with pytest.raises(CaptureStorageError, match="queue"):
            writer.submit({"raw": "b" * 30})
        with pytest.raises(CaptureStorageError, match="queue"):
            await asyncio.wait_for(writer.wait_failed(), timeout=5)
    finally:
        release_fsync.set()

    with pytest.raises(CaptureStorageError, match="queue"):
        await asyncio.wait_for(writer.close(), timeout=5)
    assert list(iter_capture_records(output_dir)) == [
        {"raw": "a" * 30, "capture_sequence": 1},
    ]
    assert read_status(output_dir)["failed"] is True
    assert read_status(output_dir)["closed"] is False


@pytest.mark.asyncio
async def test_total_cap_includes_queued_records_and_drains_accepted_records(tmp_path):
    output_dir = tmp_path / "run"
    writer = CaptureWriter(output_dir, max_total_bytes=90, flush_interval=60)
    await writer.start({})
    writer.submit({"raw": "a" * 30})

    with pytest.raises(CaptureStorageError, match="total"):
        writer.submit({"raw": "b" * 30})
    with pytest.raises(CaptureStorageError, match="total"):
        await asyncio.wait_for(writer.wait_failed(), timeout=5)
    with pytest.raises(CaptureStorageError, match="total"):
        await asyncio.wait_for(writer.close(), timeout=5)

    assert list(iter_capture_records(output_dir)) == [
        {"raw": "a" * 30, "capture_sequence": 1},
    ]
    assert read_status(output_dir)["failed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("record", "max_file_bytes", "message"),
    [
        ({"raw": "too big"}, 1, "segment"),
        ({"value": float("nan")}, 100, "serialize"),
    ],
)
async def test_invalid_serialized_records_fail_the_writer(
    tmp_path, record, max_file_bytes, message
):
    output_dir = tmp_path / "run"
    writer = CaptureWriter(output_dir, max_file_bytes=max_file_bytes)
    await writer.start({})

    with pytest.raises(CaptureStorageError, match=message):
        writer.submit(record)
    with pytest.raises(CaptureStorageError, match=message):
        await asyncio.wait_for(writer.close(), timeout=5)

    assert list(iter_capture_records(output_dir)) == []
    assert read_status(output_dir)["failed"] is True


@pytest.mark.asyncio
async def test_fsync_failure_reaches_producers_without_advancing_counters(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "run"
    writer = CaptureWriter(output_dir, flush_interval=0.01)
    await writer.start({})
    original_fsync = os.fsync
    failed_once = False

    def fail_record_fsync(fd):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("simulated disk failure")
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_record_fsync)
    writer.submit({"raw": "record"})

    with pytest.raises(CaptureStorageError, match="simulated disk failure"):
        await asyncio.wait_for(writer.wait_failed(), timeout=5)
    with pytest.raises(CaptureStorageError, match="simulated disk failure"):
        writer.submit({"raw": "later"})
    with pytest.raises(CaptureStorageError, match="simulated disk failure"):
        await asyncio.wait_for(writer.close(), timeout=5)

    assert read_status(output_dir) == {
        "durable_records": 0,
        "durable_bytes": 0,
        "closed": False,
        "failed": True,
    }


def test_replay_raises_on_an_unterminated_final_record(tmp_path):
    segment = tmp_path / "records-000001.ndjson"
    segment.write_bytes(b'{"capture_sequence":1}\n{"capture_sequence":2}')
    records = iter_capture_records(tmp_path)

    assert next(records) == {"capture_sequence": 1}
    with pytest.raises(CaptureStorageError, match="unterminated"):
        next(records)


@pytest.mark.parametrize(
    "segment_numbers", [(1, 3), (1, 1)], ids=["missing-segment", "missing-line"]
)
def test_replay_rejects_missing_middle_records(tmp_path, segment_numbers):
    for segment_number, sequence in zip(segment_numbers, (1, 3), strict=True):
        segment = tmp_path / f"records-{segment_number:06d}.ndjson"
        with segment.open("ab") as file:
            file.write(f'{{"capture_sequence":{sequence}}}\n'.encode())

    records = iter_capture_records(tmp_path)
    assert next(records) == {"capture_sequence": 1}
    with pytest.raises(CaptureStorageError, match="expected 2, got 3"):
        next(records)


@pytest.mark.parametrize("sequence", [0, -1, True, 1.0, "1", None])
def test_replay_rejects_invalid_capture_sequence(tmp_path, sequence):
    segment = tmp_path / "records-000001.ndjson"
    segment.write_text(json.dumps({"capture_sequence": sequence}) + "\n", encoding="utf-8")

    with pytest.raises(CaptureStorageError, match="positive integer capture_sequence"):
        next(iter_capture_records(tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize("retained_segments", [2, 0], ids=["missing-tail", "missing-all"])
async def test_closed_replay_rejects_missing_tail(tmp_path, retained_segments):
    output_dir = tmp_path / "run"
    writer = CaptureWriter(output_dir, max_file_bytes=70)
    await writer.start({})
    for _ in range(3):
        writer.submit({"raw": "a" * 30})
    await writer.close()
    segments = sorted(output_dir.glob("records-*.ndjson"))
    assert len(segments) == 3
    for segment in segments[retained_segments:]:
        segment.unlink()

    records = iter_capture_records(output_dir)
    assert [next(records)["capture_sequence"] for _ in range(retained_segments)] == list(
        range(1, retained_segments + 1)
    )
    with pytest.raises(CaptureStorageError, match="durable totals"):
        list(records)


@pytest.mark.parametrize(
    ("closed", "durable_records", "byte_adjustment"),
    [(True, 1, -1), (False, 1, 1), (False, 2, 0)],
    ids=["closed-extra-bytes", "interrupted-missing-bytes", "interrupted-missing-records"],
)
def test_replay_checks_durable_totals(tmp_path, closed, durable_records, byte_adjustment):
    segment = tmp_path / "records-000001.ndjson"
    line = b'{"capture_sequence":1}\n'
    segment.write_bytes(line)
    status = {
        "closed": closed,
        "durable_records": durable_records,
        "durable_bytes": len(line) + byte_adjustment,
    }
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(CaptureStorageError, match="durable totals"):
        list(iter_capture_records(tmp_path))


@pytest.mark.parametrize(
    ("field", "value"), [("closed", 1), ("durable_records", True), ("durable_bytes", -1)]
)
def test_replay_rejects_invalid_status_values(tmp_path, field, value):
    status = {"closed": False, "durable_records": 0, "durable_bytes": 0}
    status[field] = value
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(CaptureStorageError, match="invalid capture status"):
        list(iter_capture_records(tmp_path))


def test_interrupted_replay_allows_complete_rows_beyond_durable_status(tmp_path):
    segment = tmp_path / "records-000001.ndjson"
    first_row = b'{"capture_sequence":1}\n'
    segment.write_bytes(first_row + b'{"capture_sequence":2}\n')
    status = {"closed": False, "durable_records": 1, "durable_bytes": len(first_row)}
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")

    assert list(iter_capture_records(tmp_path)) == [
        {"capture_sequence": 1},
        {"capture_sequence": 2},
    ]
