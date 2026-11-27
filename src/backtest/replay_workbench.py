"""Chart-first replay controller. The browser receives only revealed candles."""

import time
from collections import deque
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from src.backtest.paradex_replay import NS, ParadexReplay, ReplayConfig, decimal, utc
from src.backtest.replay_dataset import IndexedCapture
from src.backtest.calibration_replay import CalibrationReplay, LABELS
from src.backend.market_data.calibration import CalibrationConfig


TIMEFRAMES = (1, 5, 15, 60, 300, 360, 900, 3600)
ACTIVE = {'pending', 'resting', 'partial'}


class Candles:
    def __init__(self):
        self.frames = {seconds: deque(maxlen=2000) for seconds in TIMEFRAMES}
        self.dirty = {}

    def observe(self, wall_ns, price, size=0):
        seconds = wall_ns // NS
        value = float(price)
        for frame, rows in self.frames.items():
            bucket = seconds // frame * frame
            self.dirty[frame] = min(bucket, self.dirty.get(frame, bucket))
            bar = next((row for row in reversed(rows) if row['time'] == bucket), None) if rows and bucket < rows[-1]['time'] else (rows[-1] if rows and rows[-1]['time'] == bucket else None)
            if bar is None:
                new = {'time': bucket, 'open': value, 'high': value, 'low': value, 'close': value,
                       'volume': float(size), '_first': wall_ns, '_last': wall_ns}
                if rows and bucket < rows[-1]['time']:
                    self.frames[frame] = deque(sorted([*rows, new], key=lambda row: row['time'])[-2000:], maxlen=2000)
                else:
                    rows.append(new)
            else:
                bar.update(high=max(bar['high'], value), low=min(bar['low'], value), volume=bar['volume'] + float(size))
                if wall_ns < bar['_first']:
                    bar.update(open=value, _first=wall_ns)
                if wall_ns >= bar['_last']:
                    bar.update(close=value, _last=wall_ns)

    def visible(self, frame, since=0):
        if frame not in TIMEFRAMES:
            raise ValueError('unsupported chart timeframe')
        since = min(since, self.dirty.pop(frame, since))
        return [{key: value for key, value in bar.items() if not key.startswith('_')}
                for bar in self.frames[frame] if bar['time'] >= since][-1200:]


class Workbench:
    def __init__(self, dataset: IndexedCapture, config: ReplayConfig, journal_dir=Path('data/replay_sessions'),
                 warmup_seconds=60):
        self.engine = ParadexReplay(dataset.source, config, source=dataset, journal_dir=journal_dir)
        self.dataset = dataset
        self.calibration = CalibrationReplay(dataset.source, CalibrationConfig(), cache_dir=dataset.path.parent)
        self.candles = Candles()
        self.plan = None
        self.drawings = []
        self.last_used = time.monotonic()
        self.processed = 0
        self.engine.baseline_enabled = False
        available = max(0, dataset.summary['last_replay_ns'] - self.engine.clock_ns)
        target = self.engine.clock_ns + min(max(0, min(int(warmup_seconds), 3600)) * NS, available // 5)
        self.advance(target, max_events=200000)
        self.engine.baseline_enabled = True
        self.engine._log('system', 'practice_start', warmup_seconds=warmup_seconds)

    def advance(self, target_ns, max_events=10000):
        started = time.perf_counter()
        engine = self.engine
        count = 0
        while not engine.finished and engine.next_event_ns is not None and engine.next_event_ns <= target_ns and count < max_events:
            engine.step()
            self.calibration.advance(engine.clock_ns, getattr(engine, 'last_original_sequence', None))
            count += 1
            if engine.last_kind == 'depth' and engine.midpoint is not None:
                self._protect()
            if engine.price_updated:
                self.candles.observe(engine.history[-1][0], engine.history[-1][1],
                                     engine.last_trade_size if engine.price_source == 'trades' else 0)
        if not engine.finished and (engine.next_event_ns is None or engine.next_event_ns > target_ns):
            delta = max(0, target_ns - engine.clock_ns)
            engine.clock_ns += delta
            engine.wall_ns += delta
            self.calibration.advance(engine.clock_ns)
        self.processed += count
        self.last_used = time.monotonic()
        return {'events': count, 'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
                'target_ns': target_ns, 'more': not engine.finished and engine.clock_ns < target_ns}

    def _active(self, actor='human'):
        return [order for order in self.engine.orders if order.actor == actor and order.status in ACTIVE]

    def _protect(self):
        engine = self.engine
        if not self.plan or engine.finished or not engine.book_valid:
            return
        position = engine.accounts['human'].position
        if not position:
            return
        for field, kind in [('stop', 'stop'), ('target', 'limit')]:
            level = self.plan.get(field)
            if level is None:
                continue
            order_id = self.plan.get(field + '_id')
            current = next((o for o in self._active() if o.id == order_id), None)
            if current and current.remaining == abs(position) and current.price == level:
                continue
            if current:
                engine.cancel('human', current.id, 'Resize chart protection to actual filled quantity')
            order = engine.submit('human', 'sell' if position > 0 else 'buy', abs(position),
                                  order_type=kind, price=level, reduce_only=True,
                                  rationale='Chart protection: ' + field)
            self.plan[field + '_id'] = order.id

    def place(self, body):
        engine = self.engine
        if engine.accounts['human'].position or self._active():
            raise ValueError('Close or cancel the current trade before opening another')
        side = body.get('side')
        kind = body.get('order_type', 'market')
        if side not in {'buy', 'sell'} or kind not in {'market', 'limit'}:
            raise ValueError('choose buy/sell and market/limit')
        if not engine.book_valid:
            raise ValueError('Advance to fresh recorded depth before trading')
        reference = decimal(body.get('price'), 'entry') if kind == 'limit' else (
            engine.asks[0][0] if side == 'buy' else engine.bids[0][0])
        stop = decimal(body['stop'], 'stop') if body.get('stop') is not None else None
        target = decimal(body['target'], 'target') if body.get('target') is not None else None
        self._validate_levels(side, reference, stop, target)
        order = engine.submit('human', side, body.get('quantity'), order_type=kind,
                              price=reference if kind == 'limit' else None,
                              rationale=body.get('reason') or 'Chart ' + side + ' entry')
        self.plan = {'entry_id': order.id, 'side': side, 'stop': stop, 'target': target}
        return order

    @staticmethod
    def _validate_levels(side, reference, stop, target):
        if stop is not None and (stop >= reference if side == 'buy' else stop <= reference):
            raise ValueError('Stop must be below a long or above a short entry/current quote')
        if target is not None and (target <= reference if side == 'buy' else target >= reference):
            raise ValueError('Target must be above a long or below a short entry/current quote')

    def amend(self, field, value, reason='Moved trade level on chart'):
        engine = self.engine
        if not self.plan or field not in {'entry', 'stop', 'target'}:
            raise ValueError('no trade level to move')
        if engine.finished or not engine.book_valid:
            raise ValueError('Trade levels can be changed only during practice with fresh depth')
        value = decimal(value, field)
        position = engine.accounts['human'].position
        entry = next((o for o in self._active() if o.id == self.plan['entry_id']), None)
        if field == 'entry':
            if position or entry is None or entry.order_type != 'limit':
                raise ValueError('Only an unfilled limit entry can be moved')
            self._validate_levels(entry.side, value, self.plan.get('stop'), self.plan.get('target'))
            engine.cancel('human', entry.id, reason)
            new = engine.submit('human', entry.side, entry.remaining, order_type='limit', price=value, rationale=reason)
            self.plan['entry_id'] = new.id
        else:
            reference = (engine.bids[0][0] if position > 0 else engine.asks[0][0]) if position else (
                entry.price if entry and entry.price else engine.midpoint)
            levels = {name: self.plan.get(name) for name in ('stop', 'target')}
            levels[field] = value
            self._validate_levels(self.plan['side'], reference, **levels)
            engine._log('human', 'move_level', level=field, price=value, rationale=reason)
            self.plan[field] = value
            old_id = self.plan.pop(field + '_id', None)
            old = next((o for o in self._active() if o.id == old_id), None)
            if old:
                engine.cancel('human', old.id, reason)
            self._protect()

    def flatten(self, reason='Closed position from chart'):
        self.engine.flatten('human', reason)
        self.plan = None

    def cancel_all(self):
        for order in self._active():
            self.engine.cancel('human', order.id, 'Cancelled chart orders')
        self.plan = None

    def state(self, frame=5, since=0, event_offset=0):
        engine = self.engine
        visible_events = [e for e in engine.events if e['actor'] != 'bot' or engine.finished]
        mark = engine.mark_price
        account = engine.accounts['human'].summary(mark, engine.config.starting_equity)
        layers = []
        position = engine.accounts['human'].position
        if position:
            layers.append({'key': 'entry', 'price': float(engine.accounts['human'].average_entry),
                           'label': 'ENTRY', 'draggable': False})
        elif self.plan:
            entry = next((o for o in self._active() if o.id == self.plan['entry_id']), None)
            if entry and entry.price:
                layers.append({'key': 'entry', 'price': float(entry.price), 'label': 'LIMIT', 'draggable': True})
        if self.plan and (position or any(o.id == self.plan['entry_id'] for o in self._active())):
            for field, label in [('stop', 'STOP'), ('target', 'TARGET')]:
                if self.plan.get(field) is not None:
                    layers.append({'key': field, 'price': float(self.plan[field]), 'label': label, 'draggable': not engine.finished})
        start = self.dataset.summary['first_replay_ns']
        end = self.dataset.summary['last_replay_ns']
        calibration = self.calibration.snapshot(engine.clock_ns)
        return {'id': engine.attempt_id, 'market': engine.config.market, 'clock_ns': engine.clock_ns,
                'utc': utc(engine.wall_ns).isoformat(), 'finished': engine.finished, 'failure': engine.failure,
                'progress': min(1, max(0, (engine.clock_ns - start) / max(1, end - start))),
                'book_valid': engine.book_valid, 'midpoint': engine.midpoint, 'last_price': mark,
                'bid': engine.bids[0][0] if engine.bids else None, 'ask': engine.asks[0][0] if engine.asks else None,
                'opening_range': engine.range, 'candles': self.candles.visible(frame, since), 'timeframe': frame,
                'layers': layers, 'account': account,
                'orders': [asdict(o) for o in engine.orders if o.actor == 'human' or engine.finished][-100:],
                'events': visible_events[0 if engine.finished else max(0, event_offset):], 'event_count': len(visible_events),
                'drawings': self.drawings, 'rule_version': engine.config.rule_version,
                'comparison': engine.report()['accounts'] if engine.finished and not engine.failure else None,
                'capture': engine.capture['run_id'], 'capture_path': str(self.dataset.source), 'plan': self.plan,
                'price_source': engine.price_source,
                'trade_history_provenance': engine.capture.get('trade_history_provenance'),
                'calibration': calibration, 'calibration_cursor': self._calibration_cursor(calibration),
                'processed_events': self.processed}

    def _calibration_cursor(self, features):
        return f"{self.engine.attempt_id}:{self.engine.clock_ns}:{features['capture_sequence']}"

    def label_speed(self, body):
        if self.engine.finished or self.engine.failure:
            raise ValueError('labels must be recorded during practice')
        label = body.get('label')
        reason = body.get('reason', '')
        features = self.calibration.snapshot(self.engine.clock_ns)
        if body.get('expected_cursor') != self._calibration_cursor(features):
            raise ValueError('Replay advanced; inspect the current features and save your interpretation again')
        source = body.get('source')
        stream = body.get('stream')
        view = features['sources'].get(source)
        if (label not in LABELS or not view or stream not in {'quote', 'trade'} or view[stream]['price'] is None
                or not isinstance(reason, str) or not reason.strip()):
            raise ValueError('choose an observed source, a calibration label, and a reason')
        if len(reason) > 4000:
            raise ValueError('calibration reason is limited to 4000 characters')
        context = {key: features[key] for key in ('version', 'config_version', 'config', 'as_of_ns', 'capture_sequence')}
        encounters = sorted(view['encounters'], key=lambda row:
                            (-row['last_ns'], abs(row['signed_distance_bps'])))[:3]
        context.update(metrics=view[stream], opening_range=view['opening_range'], market=view['market'],
                       encounter=view['encounters'][-1] if view['encounters'] else None,
                       encounters=encounters, basis=features['basis'])
        self.engine._log('human', 'calibration_label', label=label, source=source, stream=stream,
                         reason=reason.strip(), features=context)

    def close(self):
        self.calibration.close()
        self.engine.close()

    def finish(self):
        self.engine.finish()
        return self.save()

    def save(self):
        return self.engine.save_report(extra={'drawings': self.drawings, 'chart_kind': self.engine.price_source + ' candles',
                                              'calibration': self.calibration.snapshot(self.engine.clock_ns)})
