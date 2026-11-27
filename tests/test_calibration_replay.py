import json

import pytest

from src.backtest.calibration_replay import CalibrationReplay
from src.backtest.paradex_replay import ReplayConfig
from src.backtest.replay_dataset import IndexedCapture
from src.backtest.replay_workbench import Workbench
from tests.test_calibration_dataset import capture


NS = 1_000_000_000
BASE = 1795651200000000000


def sample(tmp_path):
    records = []
    for second in range(10):
        records.extend([
            {'source': 'paradex', 'market': 'BTC-USD-PERP', 'kind': 'depth', 'session_id': 'p',
             'received_monotonic_ns': (10 + second) * NS, 'received_at_ns': BASE + second * NS,
             'payload': {'pkg_type': 'SNAPSHOT', 'bids': [['100', '1']], 'asks': [['102', '1']]}},
            {'source': 'binance_spot', 'market': 'BTCUSDT', 'kind': 'bbo', 'session_id': 'b',
             'received_monotonic_ns': (10 + second) * NS, 'received_at_ns': BASE + second * NS,
             'payload': {'b': str(100 + second), 'a': str(102 + second)}}])
    return capture(tmp_path / 'capture', records)


def test_feature_cursor_does_not_reveal_later_sequence_at_same_time(tmp_path):
    path = sample(tmp_path)
    cursor = CalibrationReplay(path, cache_dir=tmp_path / 'cache')
    cursor.advance(10 * NS, sequence_bound=1)
    assert cursor.snapshot(10 * NS)['sources']['binance_spot']['quote']['price'] is None
    cursor.advance(10 * NS, sequence_bound=2)
    assert cursor.snapshot(10 * NS)['sources']['binance_spot']['quote']['price'] == 101
    cursor.close()


def test_feature_cursor_is_independent_of_playback_step_and_state_polls(tmp_path):
    path = sample(tmp_path)
    small = CalibrationReplay(path, cache_dir=tmp_path / 'cache')
    large = CalibrationReplay(path, cache_dir=tmp_path / 'cache')
    for second in range(10, 20):
        small.advance(second * NS)
        small.snapshot(second * NS)
        small.snapshot(second * NS)
    large.advance(19 * NS)
    assert small.snapshot(19 * NS) == large.snapshot(19 * NS)
    small.close()
    large.close()


def test_workbench_labels_journal_server_computed_features_and_exports_them(tmp_path):
    path = sample(tmp_path)
    desk = Workbench(IndexedCapture(path, cache_dir=tmp_path / 'cache'), ReplayConfig(),
                     journal_dir=tmp_path / 'journals', warmup_seconds=0)
    desk.advance(12 * NS)
    visible = desk.state()
    before = visible['calibration']
    assert before['sources']['binance_spot']['quote']['price'] == 103
    with pytest.raises(ValueError):
        desk.label_speed({'label': 'invented', 'source': 'binance_spot', 'reason': 'test'})
    body = {'label': 'acceptance_no_add', 'source': 'binance_spot', 'stream': 'quote',
            'expected_cursor': visible['calibration_cursor'],
            'reason': 'slow persistence', 'features': {'future': True}}
    desk.label_speed(body)
    entry = desk.engine.events[-1]
    assert entry['action'] == 'calibration_label'
    assert entry['features']['metrics'] == before['sources']['binance_spot']['quote']
    assert 'sources' not in entry['features']
    assert entry['stream'] == 'quote'
    assert entry['reason'] == 'slow persistence'
    journal = [json.loads(line) for line in desk.engine.journal_path.read_text().splitlines()]
    assert journal[-1]['features'] == entry['features']
    desk.advance(13 * NS)
    with pytest.raises(ValueError, match='advanced|changed'):
        desk.label_speed(body)
    desk.finish()
    exported = json.loads(desk.save().read_text())
    assert exported['calibration']['config_version'] == before['config_version']
    assert any(e['action'] == 'calibration_label' for e in exported['events'])
    with pytest.raises(ValueError):
        desk.label_speed({'label': 'fast_add', 'source': 'binance_spot', 'reason': 'too late'})
    desk.close()


def test_calibration_or_is_independent_of_generic_baseline_settings(tmp_path):
    path = sample(tmp_path)
    desk = Workbench(IndexedCapture(path, cache_dir=tmp_path / 'cache'),
                     ReplayConfig(range_start='12:00', range_minutes=60),
                     journal_dir=None, warmup_seconds=0)
    config = desk.state()['calibration']['config']
    assert config['range_start'] == '20:00'
    assert config['range_minutes'] == 15
    desk.close()


def test_recovered_event_does_not_release_original_sequence_bound(tmp_path):
    path = sample(tmp_path)
    dataset = IndexedCapture(path, cache_dir=tmp_path / 'cache')
    first = next(dataset.records('BTC-USD-PERP'))
    first['original_capture_sequence'] = 1
    recovered = {**first, 'capture_sequence': 2, 'original_capture_sequence': None, 'kind': 'trade',
                 'payload': {'recovered': True}}
    third = {**first, 'capture_sequence': 3, 'original_capture_sequence': 3}
    early = {**first, 'capture_sequence': 0, 'original_capture_sequence': 0,
             'received_monotonic_ns': 9 * NS}
    dataset.records = lambda market: (r for r in [early, first, recovered, third])
    desk = Workbench(dataset, ReplayConfig(), journal_dir=None, warmup_seconds=0)
    desk.advance(10 * NS, max_events=1)
    assert desk.state()['calibration']['sources']['binance_spot']['quote']['price'] is None
    desk.advance(10 * NS, max_events=1)
    assert desk.state()['calibration']['sources']['binance_spot']['quote']['price'] is None
    desk.close()


def test_label_cursor_preserves_nanoseconds_above_javascript_integer_precision(tmp_path):
    path = sample(tmp_path)
    desk = Workbench(IndexedCapture(path, cache_dir=tmp_path / 'cache'), ReplayConfig(),
                     journal_dir=None, warmup_seconds=0)
    try:
        desk.advance(12 * NS)
        desk.engine.clock_ns = 2**53 + 1
        visible = desk.state()
        assert str(desk.engine.clock_ns) in visible['calibration_cursor']
        body = {'label': 'wait', 'source': 'binance_spot', 'stream': 'quote', 'reason': 'same paused view',
                'expected_cursor': visible['calibration_cursor']}
        desk.label_speed(body)
        assert desk.engine.events[-1]['features']['as_of_ns'] == 2**53 + 1
        desk.engine.clock_ns += 1
        with pytest.raises(ValueError, match='advanced'):
            desk.label_speed(body)
    finally:
        desk.close()
