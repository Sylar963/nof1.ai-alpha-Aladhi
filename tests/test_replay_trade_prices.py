import hashlib
import json
from decimal import Decimal

import pytest

from src.backtest.paradex_replay import ParadexReplay, ReplayConfig
from src.backtest.replay_dataset import IndexedCapture
from src.backtest.replay_workbench import Candles, Workbench
from src.backtest.trade_history import recover_trade_history


BASE = 1795632780000000000
NS = 1000000000


def book(second):
    return {'source': 'paradex', 'market': 'BTC-USD-PERP', 'session_id': 'one', 'kind': 'depth',
            'received_at_ns': BASE + int(second * NS), 'received_monotonic_ns': 10 * NS + int(second * NS),
            'source_event_time_ns': BASE + int(second * NS),
            'payload': {'pkg_type': 'SNAPSHOT', 'bids': [['100', '1']],
                        'asks': [['400' if int(second * 10) % 2 else '600', '1']]}}


def trade(receive, matched, price, identifier, size='.01'):
    record = book(receive)
    record.update(kind='trade', source_transaction_time_ns=BASE + int(matched * NS), payload={
        'market': 'BTC-USD-PERP', 'created_at_us': BASE // 1000 + int(matched * 1000000),
        'trade_id_low64': identifier, 'price': price, 'size': size, 'trade_type': 'RPI'})
    return record


def write_capture(path, records):
    path.mkdir()
    records = sorted(records, key=lambda row: row['received_monotonic_ns'])
    raw = b''.join((json.dumps({**row, 'capture_sequence': i}) + '\n').encode()
                   for i, row in enumerate(records, 1))
    (path / 'records-00000000000000000001.ndjson').write_bytes(raw)
    (path / 'manifest.json').write_text(json.dumps({'capture_schema_version': 1, 'run_id': path.name}))
    (path / 'status.json').write_text(json.dumps({'closed': True, 'failed': False,
                                                'durable_records': len(records), 'durable_bytes': len(raw)}))
    return path


def test_trade_candles_ignore_book_spikes_deduplicate_prints_and_cannot_show_future_highs(tmp_path):
    first = trade(.15, .1, '101', '1')
    duplicate = {**first, 'received_monotonic_ns': 10 * NS + 650000000, 'received_at_ns': BASE + 650000000}
    source = write_capture(tmp_path / 'capture', [book(i / 10) for i in range(20)] +
                           [first, trade(.45, .4, '102', '2', '.02'), duplicate])
    dataset = IndexedCapture(source, cache_dir=tmp_path / 'cache')
    desk = Workbench(dataset, ReplayConfig(fee_bps='0'), journal_dir=None, warmup_seconds=0)
    assert desk.engine.price_source == 'trades'
    assert not desk.state(frame=60)['candles']
    desk.advance(10 * NS + 200000000)
    current = desk.state(frame=60)['candles'][0]
    assert current['open'] == current['high'] == current['low'] == current['close'] == 101
    desk.advance(10 * NS + 700000000)
    bar = desk.state(frame=60)['candles'][0]
    assert bar['high'] == 102 and bar['low'] == 101 and bar['close'] == 102
    assert bar['volume'] == pytest.approx(.03)
    assert dataset.summary['trade_count'] == 2
    assert desk.engine.mark_price == 102
    assert desk.engine.midpoint > 200
    desk.engine.close()


def test_trade_chart_does_not_replace_executable_depth_prices(tmp_path):
    source = write_capture(tmp_path / 'capture', [book(i / 10) for i in range(20)] + [trade(.15, .1, '101', '1')])
    desk = Workbench(IndexedCapture(source, cache_dir=tmp_path / 'cache'), ReplayConfig(fee_bps='0'),
                     journal_dir=None, warmup_seconds=0)
    desk.advance(10 * NS + 200000000)
    desk.place({'side': 'buy', 'quantity': '.01'})
    desk.advance(10 * NS + 400000000)
    assert desk.engine.accounts['human'].average_entry >= 400
    assert desk.engine.mark_price == 101
    desk.engine.close()


def test_opening_range_and_breakout_use_trade_prices_instead_of_book_midpoint(tmp_path):
    source = write_capture(tmp_path / 'capture', [book(i) for i in range(65)] +
                           [trade(.15, .1, '101', '1'), trade(20.15, 20.1, '102', '2'),
                            trade(61.15, 61.1, '103', '3')])
    config = ReplayConfig(session_timezone='UTC', range_start='18:53', range_minutes=1)
    desk = Workbench(IndexedCapture(source, cache_dir=tmp_path / 'cache'), config,
                     journal_dir=None, warmup_seconds=0)
    desk.advance(10 * NS + 60500000000)
    assert desk.engine.range['high'] == 102 and desk.engine.range['low'] == 101
    assert desk.engine.range['complete']
    assert not desk.engine.orders
    desk.advance(10 * NS + 61200000000)
    assert len(desk.engine.orders) == 1
    assert desk.engine.orders[0].actor == 'bot' and desk.engine.orders[0].side == 'buy'
    assert desk.engine.accounts['bot'].position == 0
    desk.engine.close()


@pytest.mark.parametrize('delayed_price,expected_high,expected_low', [('110', '110', '101'), ('95', '102', '95')])
@pytest.mark.parametrize('after_cutoff', [False, True])
def test_delayed_range_trade_uses_its_own_price_without_replacing_latest_mark(
        tmp_path, delayed_price, expected_high, expected_low, after_cutoff):
    latest_time, latest_price = (61.1, '150') if after_cutoff else (20.1, '102')
    delayed_receive = 62.15 if after_cutoff else 21.15
    records = [book(i) for i in range(66)] + [
        trade(.15, .1, '101', '1'), trade(20.15, 20.1, '102', '2'),
        trade(delayed_receive, 10.1, delayed_price, '4')]
    if after_cutoff:
        records.append(trade(61.15, 61.1, latest_price, '3'))
    source = write_capture(tmp_path / 'capture', records)
    config = ReplayConfig(session_timezone='UTC', range_start='18:53', range_minutes=1)
    desk = ParadexReplay(source, config, journal_dir=None)
    desk.baseline_enabled = False
    try:
        desk.advance_to(10 * NS + int((delayed_receive - .05) * NS))
        assert desk.range['high'] == Decimal('102') and desk.range['low'] == Decimal('101')
        desk.advance_to(10 * NS + int((delayed_receive + .05) * NS))
        assert desk.range['high'] == Decimal(expected_high)
        assert desk.range['low'] == Decimal(expected_low)
        assert desk.last_trade == desk.mark_price == desk.signal_price == Decimal(latest_price)
        assert desk.last_trade_time_ns == BASE + int(latest_time * NS)
    finally:
        desk.close()


def test_late_print_updates_earlier_bar_without_changing_later_close_or_duplicating_volume():
    candles = Candles()
    candles.observe(10 * NS, Decimal('100'), Decimal('.1'))
    candles.observe(70 * NS, Decimal('105'), Decimal('.2'))
    candles.visible(60)
    candles.observe(20 * NS, Decimal('103'), Decimal('.3'))
    changed = candles.visible(60, since=60)
    assert [row['time'] for row in changed] == [0, 60]
    assert changed[0]['close'] == 103 and changed[0]['volume'] == pytest.approx(.4)
    assert changed[1]['close'] == 105


class Reply:
    def __init__(self, body):
        self.body = body
        self.content = json.dumps(body).encode()

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def test_recovered_tape_preserves_original_and_reveals_prints_only_after_known_book_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = write_capture(tmp_path / 'original', [book(i / 10) for i in range(20)])
    original = next(source.glob('records-*')).read_bytes()
    rows = [{'id': str(1000 + i), 'market': 'BTC-USD-PERP', 'created_at': BASE // 1000000 + offset,
             'price': value, 'size': '.01', 'side': 'BUY', 'trade_type': 'FILL'}
            for i, (offset, value) in enumerate([(-60000, '90'), (350, '104')])]
    requests = []

    def get(url, params, timeout):
        requests.append((url, params))
        return Reply({'results': list(reversed(rows)), 'next': None})

    recovered = recover_trade_history(source, tmp_path / 'repaired', warmup_hours=1, get=get)
    assert next(source.glob('records-*')).read_bytes() == original
    assert not (source / 'chart-trades.ndjson').exists()
    metadata = json.loads((recovered / 'manifest.json').read_text())['trade_history']
    assert metadata['original_receive_times_available'] is False
    assert metadata['sha256'] == hashlib.sha256((recovered / 'chart-trades.ndjson').read_bytes()).hexdigest()
    assert requests[0][0] == 'https://api.prod.paradex.trade/v1/trades'
    dataset = IndexedCapture(recovered, cache_dir=tmp_path / 'cache')
    desk = Workbench(dataset, ReplayConfig(), journal_dir=None, warmup_seconds=0)
    desk.advance(10 * NS + 300000000)
    assert desk.engine.mark_price == 90
    desk.advance(10 * NS + 400000000)
    assert desk.engine.mark_price == 104
    assert desk.state()['trade_history_provenance']['original_receive_times_available'] is False
    desk.finish()
    assert desk.engine.report()['price_source'] == 'trades'
    desk.engine.close()


def test_recovery_rejects_wrong_market_and_removes_only_its_temporary_output(tmp_path):
    source = write_capture(tmp_path / 'original', [book(0), book(1)])
    bad = {'id': '1', 'market': 'ETH-USD-PERP', 'created_at': BASE // 1000000,
           'price': '100', 'size': '1', 'trade_type': 'FILL'}
    with pytest.raises(ValueError, match='unexpected market'):
        recover_trade_history(source, tmp_path / 'repaired', get=lambda *args, **kwargs: Reply({'results': [bad]}))
    assert (source / 'manifest.json').exists()
    assert not list((tmp_path / 'repaired').iterdir())
