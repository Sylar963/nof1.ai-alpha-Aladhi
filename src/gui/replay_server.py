"""Small local aiohttp API and static canvas UI. No GUI framework is loaded."""

import asyncio
import json
import time
from dataclasses import fields
from datetime import datetime, time as session_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from aiohttp import web

from src.backtest.paradex_replay import NS, ReplayConfig
from src.backtest.replay_dataset import IndexedCapture, snapshot_capture
from src.backtest.replay_workbench import Workbench


ASSETS = Path(__file__).with_name('replay_web')
STATE = web.AppKey('replay_state', dict)


def response(value, **kwargs):
    return web.Response(text=json.dumps(value, default=str, allow_nan=False), content_type='application/json', **kwargs)


@web.middleware
async def errors(request, handler):
    if request.method == 'POST':
        origin = request.headers.get('Origin')
        if origin and origin != f'{request.scheme}://{request.host}':
            return response({'error': 'cross-origin mutation rejected'}, status=403)
        if request.content_type != 'application/json':
            return response({'error': 'JSON request required'}, status=415)
    try:
        result = await handler(request)
        result.headers['Cache-Control'] = 'no-store' if request.path.startswith('/api/') else 'no-cache'
        return result
    except (ValueError, KeyError, OSError) as exc:
        return response({'error': str(exc)}, status=400)


def catalog(app):
    state = app[STATE]
    paths = set(state['paths'])
    for root in ('data/market_capture', 'data/replay_snapshots', 'data/replay_repaired'):
        paths.update(p.parent.resolve() for p in Path(root).glob('*/manifest.json'))
    result = []
    for path in paths:
        if path.name.endswith('.tmp'):
            continue
        try:
            status = json.loads((path / 'status.json').read_text())
            manifest = json.loads((path / 'manifest.json').read_text())
            if not status.get('durable_records') or status.get('failed'):
                continue
            result.append({'id': str(path), 'name': path.name, 'closed': status.get('closed'),
                           'bytes': status['durable_bytes'], 'checkpoint': bool(manifest.get('replay_snapshot_of')),
                           'price_source': 'recovered trades' if manifest.get('trade_history') else
                           'live trades' if any('trades.' in channel for source in manifest.get('sources', [])
                                                for channel in source.get('channels', [])) else 'legacy quotes'})
        except (OSError, ValueError, KeyError):
            continue
    return sorted(result, key=lambda row: row['name'], reverse=True)


async def list_captures(request):
    return response({'captures': catalog(request.app), 'default': request.app[STATE]['default']})


async def create_session(request):
    body = await request.json()
    state = request.app[STATE]
    available = {item['id'] for item in catalog(request.app)}
    chosen = body.get('capture') or state['default']
    if not chosen:
        raise ValueError('choose a recording')
    source = Path(chosen).resolve()
    if str(source) not in available:
        raise ValueError('choose a listed recording')
    config_body = body.get('config') or {}
    allowed = {field.name for field in fields(ReplayConfig)}
    if set(config_body) - allowed:
        raise ValueError('unknown replay setting')
    config = ReplayConfig(**config_body)
    async with state['load_lock']:
        for key, (old, lock) in list(state['sessions'].items()):
            async with lock:
                if old.engine.finished or time.monotonic() - old.last_used > 3600:
                    old.close()
                    state['sessions'].pop(key, None)
        if len(state['sessions']) >= 4:
            raise ValueError('Close another replay attempt before opening more')
        started = time.perf_counter()
        status = json.loads((source / 'status.json').read_text())
        if status.get('closed') is not True:
            source = await asyncio.to_thread(snapshot_capture, source, Path('data/replay_snapshots'))
        dataset = await asyncio.to_thread(IndexedCapture, source, config.market)
        desk = await asyncio.to_thread(Workbench, dataset, config, warmup_seconds=body.get('warmup_seconds', 60))
        state['sessions'][desk.engine.attempt_id] = (desk, asyncio.Lock())
        state['paths'].add(source)
        data = desk.state(int(body.get('timeframe', 5)))
        data['load_ms'] = round((time.perf_counter() - started) * 1000, 2)
        data['index_built'] = dataset.built
        return response(data)


async def session_action(request):
    desk, lock = request.app[STATE]['sessions'][request.match_info['session']]
    body = await request.json() if request.method == 'POST' else {}
    action = request.match_info['action']
    async with lock:
        started = time.perf_counter()
        desk.last_used = time.monotonic()
        frame = int(body.get('timeframe', request.query.get('timeframe', 5)))
        if frame not in desk.candles.frames:
            raise ValueError('unsupported chart timeframe')

        def execute():
            work = None
            if action == 'advance':
                seconds = float(body.get('seconds', 1))
                if not 0 <= seconds <= 86400:
                    raise ValueError('step must be between 0 and 86400 seconds')
                target = body.get('target_ns')
                moment = None
                if body.get('jump') == 'opening_range':
                    local = datetime.fromtimestamp(desk.engine.wall_ns / NS, ZoneInfo(desk.engine.config.session_timezone))
                    moment = datetime.combine(local.date(), session_time.fromisoformat(desk.engine.config.range_start), local.tzinfo)
                    if moment <= local:
                        moment += timedelta(days=1)
                elif body.get('utc'):
                    moment = datetime.fromisoformat(body['utc'])
                    if moment.tzinfo is None:
                        raise ValueError('Go to time must include its timezone')
                if moment is not None:
                    wall_target = int(moment.timestamp() * NS)
                    if wall_target > desk.dataset.summary['last_depth_ns']:
                        raise ValueError('That moment is beyond this recording; choose a time inside the sample')
                    target = desk.engine.clock_ns + wall_target - desk.engine.wall_ns
                if target is None:
                    target = desk.engine.clock_ns + int(seconds * NS)
                if type(target) is not int or target < desk.engine.clock_ns or target > desk.engine.clock_ns + 7 * 86400 * NS:
                    raise ValueError('invalid replay target')
                work = desk.advance(target)
            elif action == 'order':
                desk.place(body)
            elif action == 'level':
                desk.amend(body['field'], body['price'], body.get('reason') or 'Dragged chart level')
            elif action == 'flatten':
                desk.flatten(body.get('reason') or 'Closed trade from chart')
            elif action == 'cancel':
                desk.cancel_all()
            elif action == 'note':
                desk.engine.note(body.get('action', 'note'), body.get('reason', ''))
            elif action == 'calibration-label':
                desk.label_speed(body)
            elif action == 'drawing':
                drawings = body.get('drawings', [])
                if not isinstance(drawings, list) or len(drawings) > 100:
                    raise ValueError('at most 100 chart drawings are supported')
                desk.engine._log('human', 'drawing', drawings=drawings)
                desk.drawings = drawings
            elif action == 'finish':
                desk.finish()
            elif action == 'close':
                desk.close()
                return {'closed': True}
            elif action != 'state':
                raise ValueError('unknown replay action')
            since = int(body.get('since', request.query.get('since', 0)))
            offset = int(body.get('event_offset', request.query.get('event_offset', 0)))
            data = desk.state(frame, since, offset)
            data['work'] = work
            return data

        data = await asyncio.to_thread(execute)
        if action == 'close':
            request.app[STATE]['sessions'].pop(request.match_info['session'], None)
        data['server_ms'] = round((time.perf_counter() - started) * 1000, 2)
        return response(data)


async def export(request):
    desk, lock = request.app[STATE]['sessions'][request.match_info['session']]
    async with lock:
        path = await asyncio.to_thread(desk.save)
        return web.FileResponse(path, headers={'Content-Disposition': f'attachment; filename="replay-{desk.engine.attempt_id}.json"'})


async def asset(request):
    name = request.match_info.get('name', 'index.html')
    if name not in {'index.html', 'app.js', 'style.css'}:
        raise web.HTTPNotFound()
    return web.FileResponse(ASSETS / name)


def create_server(default_capture: Path | None = None):
    app = web.Application(middlewares=[errors], client_max_size=128 * 1024)
    initial = {Path(default_capture).resolve()} if default_capture else set()
    app[STATE] = {'sessions': {}, 'paths': initial, 'default': str(Path(default_capture).resolve()) if default_capture else None,
                  'load_lock': asyncio.Lock()}
    app.router.add_get('/', asset)
    app.router.add_get('/{name:app.js|style.css}', asset)
    app.router.add_get('/api/captures', list_captures)
    app.router.add_post('/api/sessions', create_session)
    app.router.add_get('/api/sessions/{session}/export', export)
    app.router.add_get('/api/sessions/{session}/{action:state}', session_action)
    app.router.add_post('/api/sessions/{session}/{action}', session_action)

    async def shutdown(application):
        for desk, lock in application[STATE]['sessions'].values():
            async with lock:
                desk.close()

    app.on_cleanup.append(shutdown)
    return app
