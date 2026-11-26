import json
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.backtest.paradex_replay import ReplayConfig
from src.backtest.replay_dataset import IndexedCapture, snapshot_capture
from src.backtest.replay_workbench import Candles, Workbench
from src.gui.replay_server import create_server


@pytest.fixture
def capture(tmp_path):
    source = tmp_path / 'market'
    source.mkdir()
    records = []
    for i in range(100):
        records.append({'source': 'paradex', 'market': 'BTC-USD-PERP', 'session_id': 'one',
                        'kind': 'depth', 'capture_sequence': i + 1,
                        'received_at_ns': 1795632800000000000 + i * 100000000,
                        'received_monotonic_ns': 10000000000 + i * 100000000,
                        'payload': {'pkg_type': 'SNAPSHOT', 'bids': [[str(99 + i % 3), '1']],
                                    'asks': [[str(103 + i % 3), '1']]}})
    raw = ''.join(json.dumps(record) + '\n' for record in records).encode()
    (source / 'records-00000000000000000001.ndjson').write_bytes(raw)
    (source / 'manifest.json').write_text(json.dumps({'capture_schema_version': 1, 'run_id': 'test-run'}))
    (source / 'status.json').write_text(json.dumps({'closed': True, 'failed': False,
                                                   'durable_bytes': len(raw), 'durable_records': len(records)}))
    return source


@pytest.fixture
def workbench(capture, tmp_path):
    dataset = IndexedCapture(capture, cache_dir=tmp_path / 'cache')
    work = Workbench(dataset, ReplayConfig(order_latency_ms=100, fee_bps='0'),
                     journal_dir=tmp_path / 'journals', warmup_seconds=0)
    work.advance(work.engine.clock_ns)
    yield work
    work.engine.close()


def test_index_reuses_validated_data_and_caches_invalidate_on_source_change(capture, tmp_path):
    first = IndexedCapture(capture, cache_dir=tmp_path / 'cache')
    second = IndexedCapture(capture, cache_dir=tmp_path / 'cache')
    assert first.built and not second.built
    assert list(first.records('BTC-USD-PERP')) == list(second.records('BTC-USD-PERP'))
    assert first.summary['indexed_events'] == 100


def test_active_checkpoint_copies_only_durable_prefix_and_never_modifies_recording(capture, tmp_path):
    status_path = capture / 'status.json'
    status = json.loads(status_path.read_text())
    status_path.write_text(json.dumps({**status, 'closed': False}))
    segment = next(capture.glob('records-*'))
    with segment.open('ab') as file:
        file.write(b'{"incomplete":')
    frozen = snapshot_capture(capture, tmp_path / 'snapshots')
    index = IndexedCapture(frozen, cache_dir=tmp_path / 'cache')
    assert index.summary['depth_snapshots'] == 100
    assert json.loads(status_path.read_text())['closed'] is False
    assert segment.read_bytes().endswith(b'{"incomplete":')
    assert json.loads((frozen / 'manifest.json').read_text())['replay_snapshot_of'] == str(capture.resolve())


def test_candles_reveal_only_observed_ticks_and_timeframe_changes_cannot_leak_future(workbench):
    work = workbench
    assert len(work.state(frame=1)['candles']) == 1
    assert work.state(frame=60)['candles'][0]['high'] == 101
    work.advance(work.engine.clock_ns + 200000000)
    bar = work.state(frame=60)['candles'][0]
    assert bar['open'] == 101 and bar['high'] == 103 and bar['low'] == 101 and bar['close'] == 103
    assert work.state(frame=1)['utc'] < '2026-11-25T19:00'
    assert not any(event['actor'] == 'bot' for event in work.state()['events'])


def test_visual_trade_levels_place_delayed_brackets_and_amend_stop(workbench):
    work = workbench
    work.place({'side': 'buy', 'quantity': '.01', 'stop': '95', 'target': '115'})
    assert not work.engine.accounts['human'].position
    work.advance(work.engine.clock_ns + 100000000)
    assert work.engine.accounts['human'].position == Decimal('.01')
    assert {order.order_type for order in work._active()} == {'stop', 'limit'}
    old_stop = work.plan['stop_id']
    work.amend('stop', '96')
    assert work.plan['stop'] == 96
    assert work.plan['stop_id'] != old_stop
    assert any(event['action'] == 'move_level' for event in work.engine.events)
    with pytest.raises(ValueError, match='Stop'):
        work.amend('stop', '1000')
    assert work.plan['stop'] == 96


def test_limit_entry_can_be_dragged_before_fill_and_cancelled(workbench):
    work = workbench
    work.place({'side': 'buy', 'quantity': '.01', 'order_type': 'limit', 'price': '98', 'stop': '90', 'target': '110'})
    work.amend('entry', '97')
    active = work._active()
    assert len(active) == 1 and active[0].price == 97
    assert not work.engine.accounts['human'].position
    work.cancel_all()
    assert not work._active()


def test_finish_reveals_bot_saves_drawings_and_preserves_exact_replay_cursor(workbench):
    work = workbench
    work.engine._log('bot', 'order', rationale='hidden baseline decision')
    assert not any(event['actor'] == 'bot' for event in work.state()['events'])
    work.drawings = [{'kind': 'horizontal', 'price': 90}]
    cursor = work.engine.clock_ns
    path = work.finish()
    report = json.loads(path.read_text())
    assert work.engine.clock_ns == cursor
    assert report['drawings'] == work.drawings
    assert any(event['actor'] == 'bot' for event in work.state(event_offset=100)['events'])


def test_visible_candle_memory_is_bounded():
    candles = Candles()
    for i in range(5000):
        candles.observe(i * 1000000000, Decimal(i + 1))
    assert len(candles.frames[1]) == 2000
    assert len(candles.visible(1)) == 1200


@pytest.mark.asyncio
async def test_http_contract_blocks_cross_origin_actions_and_returns_only_visible_prices(capture, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_server(capture)
    async with TestClient(TestServer(app)) as client:
        home = await client.get('/')
        assert home.status == 200
        assert 'Arena Replay' in await home.text()
        loaded = await client.post('/api/sessions', json={'capture': str(capture), 'warmup_seconds': 0})
        assert loaded.status == 200
        state = await loaded.json()
        session = state['id']
        assert len(state['candles']) == 1
        blocked = await client.post(f'/api/sessions/{session}/finish', json={}, headers={'Origin': 'https://untrusted.test'})
        assert blocked.status == 403
        assert (await client.get(f'/api/sessions/{session}/finish')).status == 405
        invalid = await client.post(f'/api/sessions/{session}/advance', json={'seconds': 10, 'timeframe': 99})
        assert invalid.status == 400
        same = await (await client.get(f'/api/sessions/{session}/state')).json()
        assert same['clock_ns'] == state['clock_ns']
        outside = await client.post(f'/api/sessions/{session}/advance', json={'utc': '2026-11-28T00:00:00Z'})
        assert outside.status == 400
        unchanged = await (await client.get(f'/api/sessions/{session}/state')).json()
        assert unchanged['clock_ns'] == state['clock_ns']
        destination = (datetime.fromisoformat(state['utc']) + timedelta(seconds=1)).isoformat()
        progressed = await client.post(f'/api/sessions/{session}/advance', json={'utc': destination, 'timeframe': 1})
        visible = await progressed.json()
        assert len(visible['candles']) == 2
        assert visible['finished'] is False
        done = await client.post(f'/api/sessions/{session}/finish', json={})
        assert (await done.json())['finished']
        exported = await client.get(f'/api/sessions/{session}/export')
        assert exported.status == 200
        assert (await exported.json(content_type=None))['chart_kind'] == 'midpoint candles'
