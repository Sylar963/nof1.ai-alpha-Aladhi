import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.backtest.paradex_replay import Account, ParadexReplay, ReplayConfig, inspect_capture
from src.backend.market_data.storage import CaptureStorageError


BASE = datetime(2026, 11, 25, 3, 59, 59, tzinfo=timezone.utc)


def depth(seconds, bids=None, asks=None, *, kind='depth', session='s1', market='BTC-USD-PERP'):
    return {'capture_schema_version': 1, 'source': 'paradex', 'session_id': session,
            'kind': kind, 'market': market,
            'received_at_ns': int((BASE + timedelta(seconds=seconds)).timestamp() * 1e9),
            'received_monotonic_ns': 10_000_000_000 + int(seconds * 1e9),
            'payload': {'pkg_type': 'SNAPSHOT', 'bids': bids or [['99', '1']],
                        'asks': asks or [['101', '1']]}}


def capture(tmp_path, records):
    path = tmp_path / 'capture'
    path.mkdir()
    lines = b''.join((json.dumps({**r, 'capture_sequence': i}) + '\n').encode()
                     for i, r in enumerate(records, start=1))
    (path / 'records-00000000000000000001.ndjson').write_bytes(lines)
    (path / 'manifest.json').write_text(json.dumps({'capture_schema_version': 1, 'run_id': 'fixture'}))
    (path / 'status.json').write_text(json.dumps({'closed': True, 'failed': False,
                                                'durable_records': len(records), 'durable_bytes': len(lines)}))
    return path


@pytest.fixture
def config():
    return ReplayConfig(session_timezone='UTC', range_start='04:00', range_minutes=1,
                        order_latency_ms=100, stale_book_ms=2000, fee_bps='0')


def test_market_fills_only_after_arrival_and_walks_depth_without_reusing_it(tmp_path, config):
    records = [depth(s, asks=[['101', '0.02'], ['102', '0.03']]) for s in [0, 0.05, 0.1, 0.2]]
    desk = ParadexReplay(capture(tmp_path, records), config)
    desk.step()
    first = desk.submit('human', 'buy', '0.04', rationale='test entry')
    second = desk.submit('human', 'buy', '0.04', rationale='second order')
    desk.step()
    assert desk.accounts['human'].position == 0
    desk.step()
    assert first.status == 'filled'
    assert second.status == 'partial_cancelled'
    assert second.remaining == Decimal('0.03')
    assert desk.accounts['human'].position == Decimal('0.05')
    assert desk.accounts['human'].average_entry == Decimal('101.6')
    desk.step()
    assert desk.accounts['human'].position == Decimal('0.05')


def test_zero_latency_still_cannot_fill_on_seen_snapshot_and_actor_depth_is_independent(tmp_path, config):
    desk = ParadexReplay(capture(tmp_path, [depth(0), depth(1)]), replace(config, order_latency_ms=0))
    desk.step()
    human = desk.submit('human', 'buy', '0.5', rationale='entry')
    bot = desk.submit('bot', 'buy', '0.5', rationale='baseline')
    assert human.id == bot.id == 1
    assert not desk.accounts['human'].position
    desk.step()
    assert desk.accounts['human'].position == desk.accounts['bot'].position == Decimal('0.5')


def test_passive_limit_never_fills_from_touch_or_disappearance(tmp_path, config):
    desk = ParadexReplay(capture(tmp_path, [depth(0), depth(1), depth(2, bids=[['100', '1']]),
                                          depth(3, bids=[['98', '1']], asks=[['99', '0.01']])]), config)
    desk.step()
    order = desk.submit('human', 'buy', '0.01', order_type='limit', price='99', rationale='bid range low')
    desk.step()
    desk.step()
    assert not desk.accounts['human'].position
    assert order.status == 'resting'
    desk.step()
    assert desk.accounts['human'].position == Decimal('0.01')
    assert order.status == 'filled'


@pytest.mark.parametrize('interruption', ['gap', 'connection_end', 'session_change', 'silent_stale'])
def test_gaps_reconnects_and_silent_staleness_cancel_pending_fills(tmp_path, config, interruption):
    records = [depth(0)]
    if interruption == 'session_change':
        records.append(depth(1, session='s2'))
    elif interruption == 'silent_stale':
        records.append(depth(5))
    else:
        records.extend([depth(0.5, kind=interruption), depth(1)])
    desk = ParadexReplay(capture(tmp_path, records), config)
    desk.step()
    order = desk.submit('human', 'buy', '.01', rationale='entry')
    while desk.step():
        pass
    assert not desk.accounts['human'].position
    assert order.status == 'cancelled'


def test_stop_and_reduce_only_never_reverse_position_and_flatten_cancels_other_orders(tmp_path, config):
    desk = ParadexReplay(capture(tmp_path, [depth(0), depth(1), depth(2, bids=[['90', '1']], asks=[['92', '1']]),
                                          depth(3), depth(4)]), config)
    desk.step()
    desk.submit('human', 'buy', '.02', rationale='entry')
    desk.step()
    stop = desk.submit('human', 'sell', '.1', order_type='stop', price='95', reduce_only=True, rationale='invalidation')
    desk.step()
    assert desk.accounts['human'].position == 0
    assert desk.accounts['human'].realized_pnl == Decimal('-.22')
    desk.step()
    assert stop.status == 'partial_cancelled'
    assert desk.accounts['human'].position == 0


def range_records():
    records = [depth(0)]
    for s in range(1, 61):
        mid = 100 if s % 2 else 102
        records.append(depth(s, bids=[[str(mid - 1), '1']], asks=[[str(mid + 1), '1']]))
    records.extend([depth(61, bids=[['103', '1']], asks=[['105', '1']]), depth(61.05),
                    depth(61.1, bids=[['103', '1']], asks=[['105', '1']]),
                    depth(62, bids=[['110', '1']], asks=[['112', '1']]),
                    depth(62.2, bids=[['110', '1']], asks=[['112', '1']])])
    return records


def test_baseline_builds_range_from_past_only_and_exits_on_target(tmp_path, config):
    desk = ParadexReplay(capture(tmp_path, range_records()), config)
    for _ in range(61):
        desk.step()
    assert not desk.range['complete']
    assert desk.range['high'] == 102
    assert desk.range['low'] == 100
    assert not desk.orders
    desk.step()
    assert desk.range['complete']
    assert len(desk.orders) == 1
    assert desk.orders[0].actor == 'bot'
    assert not desk.accounts['bot'].position
    desk.step()
    assert not desk.accounts['bot'].position
    desk.step()
    assert desk.accounts['bot'].position == Decimal('.01')
    desk.step()
    desk.step()
    assert desk.finished
    assert desk.accounts['bot'].position == 0
    assert desk.accounts['bot'].realized_pnl == Decimal('.05')


@pytest.mark.parametrize('bad_coverage', ['late_start', 'gap', 'stale', 'one_sided'])
def test_partial_or_interrupted_opening_range_cannot_produce_bot_entry(tmp_path, config, bad_coverage):
    records = range_records()
    if bad_coverage == 'late_start':
        records = records[10:]
    elif bad_coverage == 'gap':
        records.insert(20, depth(19.5, kind='gap'))
    elif bad_coverage == 'stale':
        del records[10:20]
    else:
        records[20]['payload']['asks'] = []
    desk = ParadexReplay(capture(tmp_path, records), config)
    while desk.step():
        pass
    assert not desk.range['complete']
    assert not desk.orders


def test_replay_clock_uses_selected_timezone_and_does_not_reveal_future_prices(tmp_path):
    records = range_records()
    desk = ParadexReplay(capture(tmp_path, records), ReplayConfig(range_minutes=1))
    desk.step()
    assert all(ns <= records[0]['received_at_ns'] for ns, _ in desk.history)
    assert not desk.range['complete']
    assert datetime.fromtimestamp(desk.range['start_ns'] / 1e9, timezone.utc).hour == 4
    with pytest.raises(ValueError, match='finish replay'):
        desk.report()


def test_journal_and_export_preserve_reasons_configuration_and_last_mark(tmp_path, config):
    records = [depth(0), depth(1), depth(2, kind='connection_end')]
    desk = ParadexReplay(capture(tmp_path, records), config, journal_dir=tmp_path / 'attempts')
    desk.step()
    desk.note('wait', 'Need acceptance above range')
    desk.submit('human', 'buy', '.01', rationale='Retest held')
    desk.step()
    desk.step()
    assert desk.midpoint is None
    path = desk.save_report(tmp_path / 'attempts')
    report = json.loads(path.read_text())
    journal = [json.loads(line) for line in desk.journal_path.read_text().splitlines()]
    assert any(event.get('rationale') == 'Retest held' for event in journal)
    assert report['config']['session_timezone'] == 'UTC'
    assert report['rule_version'] == config.rule_version
    assert report['accounts']['human']['position'] == '0.01'
    assert report['mark']['midpoint'] == '100'
    assert report['accounts']['human']['equity'] == '9999.99'


def test_validation_rejects_incomplete_capture_and_corrupt_tail_before_practice(tmp_path, config):
    path = capture(tmp_path, [depth(0), depth(1)])
    status = json.loads((path / 'status.json').read_text())
    (path / 'status.json').write_text(json.dumps({**status, 'closed': False}))
    with pytest.raises(ValueError, match='closed'):
        inspect_capture(path, config.market)
    (path / 'status.json').write_text(json.dumps(status))
    segment = next(path.glob('records-*'))
    segment.write_bytes(segment.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises(CaptureStorageError, match='durable totals'):
        ParadexReplay(path, config)


def test_wrong_market_crossed_book_and_non_finite_parameters_are_rejected(tmp_path, config):
    path = capture(tmp_path, [depth(0, market='ETH-USD-PERP')])
    with pytest.raises(ValueError, match='no Paradex depth'):
        ParadexReplay(path, config)
    with pytest.raises(ValueError):
        replace(config, fee_bps='NaN')
    with pytest.raises(ValueError):
        replace(config, order_latency_ms=-1)


def test_crossed_book_is_rejected_before_any_orders_can_be_placed(tmp_path, config):
    path = capture(tmp_path, [depth(0, bids=[['102', '1']], asks=[['101', '1']])])
    with pytest.raises(ValueError, match='crossed'):
        ParadexReplay(path, config)


def test_flatten_and_cancellation_reasons_and_position_cap(tmp_path, config):
    desk = ParadexReplay(capture(tmp_path, [depth(0), depth(1), depth(2), depth(3)]), config)
    desk.step()
    with pytest.raises(ValueError, match='reason'):
        desk.submit('human', 'buy', '.01')
    with pytest.raises(ValueError, match='cap'):
        desk.submit('human', 'buy', '2', rationale='oversize')
    desk.submit('human', 'buy', '.1', rationale='entry')
    desk.step()
    resting = desk.submit('human', 'buy', '.1', order_type='limit', price='90', rationale='scale')
    close = desk.flatten('human', 'failed breakout')
    assert resting.status == 'cancelled'
    assert close.reduce_only
    desk.step()
    assert desk.accounts['human'].position == 0


def test_account_flip_realizes_only_closed_quantity_and_charges_fees():
    account = Account()
    account.fill('buy', Decimal('2'), Decimal('100'), Decimal('5'))
    account.fill('sell', Decimal('3'), Decimal('110'), Decimal('5'))
    assert account.position == -1
    assert account.average_entry == 110
    assert account.realized_pnl == 20
    assert account.fees == Decimal('.265')


def test_advance_is_bounded_and_does_not_skip_events(tmp_path, config):
    desk = ParadexReplay(capture(tmp_path, [depth(0), depth(1), depth(2), depth(3)]), config)
    start = desk.clock_ns
    assert desk.advance_to(start + 10_000_000_000, max_events=2) == 2
    assert desk.sequence == 2
    assert desk.clock_ns == start + 1_000_000_000
    desk.advance_to(start + 10_000_000_000)
    assert desk.finished
    assert desk.sequence == 4
