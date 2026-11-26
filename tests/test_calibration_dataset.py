import json

import pytest

from src.backtest.calibration_dataset import ResearchCapture
from src.backend.market_data.storage import CaptureStorageError


def capture(path, records):
    path.mkdir()
    data = b''.join((json.dumps({**row, 'capture_sequence': n}) + '\n').encode()
                    for n, row in enumerate(records, 1))
    (path / 'records-00000000000000000001.ndjson').write_bytes(data)
    (path / 'manifest.json').write_text(json.dumps({'capture_schema_version': 1, 'run_id': 'research'}))
    (path / 'status.json').write_text(json.dumps({'closed': True, 'failed': False,
                                                'durable_records': len(records), 'durable_bytes': len(data)}))
    return path


def row(source, ns, kind='bbo'):
    return {'source': source, 'kind': kind, 'session_id': source + '-session',
            'market': 'BTCUSDT' if source.startswith('binance') else 'BTC-USD-PERP',
            'received_monotonic_ns': ns, 'received_at_ns': 1700000000000000000 + ns,
            'source_event_time_ns': 1700000000000000000 + ns - 100,
            'payload': {'b': '100', 'a': '101'}, 'raw_text': 'do not duplicate raw bytes'}


def test_research_keeps_all_sources_bbo_lifecycle_and_original_order(tmp_path):
    records = [row('binance_spot', 10), row('paradex', 10),
               row('binance_perp_book', 20), row('binance_perp_market', 30, 'trade'),
               row('binance_spot', 40, 'gap')]
    path = capture(tmp_path / 'capture', records)
    indexed = ResearchCapture(path, cache_dir=tmp_path / 'cache')
    actual = list(indexed.records())
    assert [r['source'] for r in actual] == [r['source'] for r in records]
    assert [r['capture_sequence'] for r in actual] == [1, 2, 3, 4, 5]
    assert actual[0]['source_event_time_ns'] == records[0]['source_event_time_ns']
    assert actual[-1]['kind'] == 'gap'
    assert not any('raw_text' in r for r in actual)
    assert set(indexed.summary['sources']) == {r['source'] for r in records}
    reused = ResearchCapture(path, cache_dir=tmp_path / 'cache')
    assert indexed.built and not reused.built
    assert list(reused.records()) == actual


def test_research_cursor_reads_only_reached_receipt_times(tmp_path):
    indexed = ResearchCapture(capture(tmp_path / 'capture', [row('paradex', 10), row('paradex', 20)]),
                              cache_dir=tmp_path / 'cache')
    assert [r['received_monotonic_ns'] for r in indexed.records(until_ns=10)] == [10]


def test_research_refuses_active_writer_and_integrity_loss(tmp_path):
    path = capture(tmp_path / 'capture', [row('paradex', 10), row('paradex', 20)])
    status_path = path / 'status.json'
    status = json.loads(status_path.read_text())
    status_path.write_text(json.dumps({**status, 'closed': False}))
    with pytest.raises(ValueError, match='frozen|closed|checkpoint'):
        ResearchCapture(path, cache_dir=tmp_path / 'cache')
    status_path.write_text(json.dumps(status))
    segment = next(path.glob('records-*'))
    segment.write_bytes(segment.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises((CaptureStorageError, ValueError)):
        ResearchCapture(path, cache_dir=tmp_path / 'cache')


def test_research_retains_delayed_submission_without_backdating_availability(tmp_path):
    path = capture(tmp_path / 'capture', [row('paradex', 20), row('binance_spot', 10)])
    indexed = ResearchCapture(path, cache_dir=tmp_path / 'cache')
    assert list(indexed.records(until_ns=15)) == []
    actual = list(indexed.records(until_ns=20))
    assert [r['received_monotonic_ns'] for r in actual] == [20, 10]
    assert [r['available_monotonic_ns'] for r in actual] == [20, 20]
    assert indexed.summary['delayed_submissions'] == 1
