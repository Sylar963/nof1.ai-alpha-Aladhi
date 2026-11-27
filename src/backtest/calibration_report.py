"""Offline, receipt-causal measurement exports for public market captures."""

import json
import math
import sqlite3
from collections import Counter, deque
from contextlib import closing
from pathlib import Path

from src.backend.market_data.calibration import HORIZONS, NS, SpeedAcceptanceEngine
from src.backend.market_data.feeds import SOURCE_NAMES
from src.backtest.calibration_dataset import ResearchCapture


REPORT_VERSION = 1
TAIL_SAMPLES = 4096
MAX_SQLITE_INTEGER = 2**63 - 1
STREAMS = {
    'binance_spot': ('quote', 'trade'),
    'binance_perp_book': ('quote',),
    'binance_perp_market': ('trade',),
    'paradex': ('quote', 'trade'),
}
TRADE_SOURCES = tuple(name for name, streams in STREAMS.items() if 'trade' in streams)

SCHEMA = '''
    CREATE TABLE encounters (
        id TEXT PRIMARY KEY, arrival_sequence INTEGER NOT NULL,
        arrival TEXT NOT NULL, latest TEXT NOT NULL
    );
    CREATE INDEX encounter_order ON encounters(arrival_sequence, id);
    CREATE TABLE transactions (
        sequence INTEGER PRIMARY KEY, source TEXT NOT NULL, market TEXT, session_id TEXT,
        session_date TEXT, revision INTEGER NOT NULL, available_ns INTEGER NOT NULL,
        original_receipt_ns INTEGER NOT NULL, received_at_ns INTEGER NOT NULL,
        event_ns INTEGER NOT NULL, price REAL NOT NULL, fresh INTEGER NOT NULL
    );
    CREATE INDEX transaction_window ON transactions(source, revision, available_ns, sequence)
        WHERE fresh = 1;
    CREATE TABLE continuity_breaks (
        source TEXT NOT NULL, revision INTEGER NOT NULL, sequence INTEGER NOT NULL,
        available_ns INTEGER NOT NULL, reason TEXT NOT NULL,
        PRIMARY KEY (source, revision)
    );
    CREATE TABLE opening_ranges (
        source TEXT NOT NULL, session_date TEXT NOT NULL, status TEXT NOT NULL,
        evidence TEXT NOT NULL, PRIMARY KEY (source, session_date)
    );
'''


def _json(value):
    return json.dumps(value, allow_nan=False)


def _write_report(path, report):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


class _SpeedStats:
    def __init__(self):
        self.samples = 0
        self.ranked_samples = 0
        self.mean = 0.0
        self.max_abs = None
        self.tail = deque(maxlen=TAIL_SAMPLES)

    def add(self, speed, rank):
        self.samples += 1
        self.ranked_samples += rank is not None
        self.mean += (speed - self.mean) / self.samples
        magnitude = abs(speed)
        self.max_abs = magnitude if self.max_abs is None else max(self.max_abs, magnitude)
        self.tail.append(magnitude)

    def summary(self):
        values = sorted(self.tail)
        return {
            'samples': self.samples,
            'ranked_samples': self.ranked_samples,
            'mean_bps_s': self.mean if self.samples else None,
            'max_abs_bps_s': self.max_abs,
            'tail_samples': len(values),
            'tail_abs_speed_bps_s': {
                str(p): values[math.ceil(p * len(values) / 100) - 1] if values else None
                for p in (50, 95, 99)
            },
        }


class _GridSummary:
    def __init__(self):
        self.stats = {name: {stream: {str(h): _SpeedStats() for h in HORIZONS}
                             for stream in ('quote', 'trade')} for name in SOURCE_NAMES}
        self.points = Counter()

    def add(self, source, stream, grid_ns, horizons):
        self.points[source, stream] += 1
        for horizon, row in horizons.items():
            if row['speed_bps_s'] is not None:
                self.stats[source][stream][horizon].add(row['speed_bps_s'], row['abs_percentile'])

    def summary(self):
        return {name: {stream: {
            'settled_grid_points': self.points[name, stream],
            'horizons': {key: item.summary() for key, item in horizons.items()},
        } for stream, horizons in streams.items()} for name, streams in self.stats.items()}


def _save_encounters(db, rows):
    for row in rows:
        encoded = _json(row)
        db.execute('''
            INSERT INTO encounters(id, arrival_sequence, arrival, latest) VALUES (?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET latest = excluded.latest
        ''', (row['id'], row['start_sequence'], encoded, encoded))


def _save_range(db, source, view):
    if view is None or view['session_date'] is None:
        return
    if view['status'] == 'forming':
        view = {**view, 'status': 'incomplete', 'reason': 'missing_cutoff_witness'}
    db.execute('INSERT OR REPLACE INTO opening_ranges VALUES (?, ?, ?, ?)',
               (source, view['session_date'], view['status'], _json(view)))


def _observe(db, capture, engine):
    ranges = {}
    revisions = engine.trade_revisions
    max_age_ns = engine.config.max_age_s * NS
    for index, record in enumerate(capture.records(), 1):
        changed = engine.observe(record)
        if changed:
            _save_encounters(db, changed)
        accepted = engine.accepted_trade
        breaks = engine.trade_breaks
        for name, reason in breaks.items():
            db.execute('INSERT INTO continuity_breaks VALUES (?, ?, ?, ?, ?)',
                       (name, revisions[name], record['capture_sequence'],
                        record['available_monotonic_ns'], reason))
        if breaks:
            revisions = engine.trade_revisions
        if accepted is not None:
            db.execute('INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                       (accepted['capture_sequence'], accepted['source'], accepted['market'],
                        accepted['session_id'], accepted['session_date'], accepted['continuity_revision'],
                        accepted['received_ns'], accepted['original_receipt_ns'], record['received_at_ns'],
                        accepted['event_ns'], accepted['price'],
                        accepted['received_ns'] - accepted['original_receipt_ns'] <= max_age_ns))
        source = record.get('source')
        if source in TRADE_SOURCES and (accepted is not None or breaks):
            view = engine.opening_range(source)
            if view['session_date'] is not None:
                previous = ranges.get(source)
                if previous and previous['session_date'] != view['session_date']:
                    _save_range(db, source, previous)
                ranges[source] = view
        if index % 2000 == 0:
            db.commit()
    snapshot = engine.snapshot()
    for source, view in snapshot['sources'].items():
        _save_encounters(db, (row for row in view['encounters'] if not row['complete']))
        current = view['opening_range']
        previous = ranges.get(source)
        if previous and previous['session_date'] != current['session_date']:
            _save_range(db, source, previous)
        _save_range(db, source, current)
    db.commit()
    return snapshot


def _horizon_ns(seconds):
    numerator, denominator = seconds.as_integer_ratio()
    whole, fraction = divmod(numerator * NS, denominator)
    return max(1, whole + (fraction * 2 >= denominator))


def _outcome_window(db, arrival, horizon_ns, capture_end, max_age_ns):
    sequence = arrival['start_sequence']
    start = arrival['start_ns']
    end = start + horizon_ns
    origin = db.execute('SELECT source, revision, original_receipt_ns FROM transactions '
                        'WHERE sequence = ? AND fresh = 1', (sequence,)).fetchone()
    result = dict(arrival_sequence=sequence, horizon_end_ns=end, status='unknown',
                  reason='missing_arrival_transaction', samples=0,
                  first_sample_sequence=None, last_sample_sequence=None, last_sample_ns=None,
                  witness_sequence=None, break_sequence=None)
    if origin is None:
        return result, None
    source, revision, original_receipt = origin
    params = source, revision
    samples, low, high, first, last = db.execute('''
        SELECT count(*), min(price), max(price), min(sequence), max(sequence)
        FROM transactions
        WHERE fresh = 1 AND source = ? AND revision = ? AND available_ns BETWEEN ? AND ?
            AND sequence > ?
    ''', (*params, start, min(end, MAX_SQLITE_INTEGER), sequence)).fetchone()
    last_ns, last_receipt, terminal = start, original_receipt, None
    if last is not None:
        last_ns, last_receipt, terminal = db.execute(
            'SELECT available_ns, original_receipt_ns, price FROM transactions WHERE sequence = ?',
            (last,)).fetchone()
    broken = db.execute('''
        SELECT available_ns, sequence, reason FROM continuity_breaks
        WHERE source = ? AND revision = ?
    ''', params).fetchone()
    witness = None
    if end <= MAX_SQLITE_INTEGER:
        witness = db.execute('''
            SELECT available_ns, sequence FROM transactions
            WHERE fresh = 1 AND source = ? AND revision = ? AND available_ns >= ?
                AND sequence > ?
            ORDER BY available_ns, sequence LIMIT 1
        ''', (*params, end, sequence)).fetchone()
    if witness and (not broken or witness < broken[:2]):
        status, reason = ('complete', None) if samples else ('unknown', 'no_in_horizon_transactions')
    elif broken:
        status, reason = 'censored', broken[2]
    else:
        status = 'censored'
        reason = 'stale' if capture_end - last_receipt > max_age_ns else 'capture_tail'
    result.update(status=status, reason=reason, samples=samples, first_sample_sequence=first,
                  last_sample_sequence=last, last_sample_ns=last_ns if samples else None,
                  witness_sequence=witness[1] if witness else None,
                  break_sequence=broken[1] if broken and status == 'censored' else None)
    return result, (low, high, terminal) if samples else None


def _export(db, directory, horizon_seconds, capture_end, max_age_ns):
    horizon_ns = _horizon_ns(horizon_seconds)
    counts = Counter()
    sequence = None
    with (directory / 'encounters.ndjson').open('w', encoding='utf-8') as encounters, (
        directory / 'outcomes.ndjson'
    ).open('w', encoding='utf-8') as outcomes:
        for identifier, initial, latest in db.execute('''
            SELECT id, arrival, latest FROM encounters ORDER BY arrival_sequence, id
        '''):
            arrival = json.loads(initial)
            encounters.write(_json(dict(id=identifier, arrival=arrival, latest=json.loads(latest))) + '\n')
            if sequence != arrival['start_sequence']:
                sequence = arrival['start_sequence']
                window, prices = _outcome_window(db, arrival, horizon_ns, capture_end, max_age_ns)
            mfe = mae = terminal = None
            if prices is not None:
                low, high, last = prices
                price, direction = arrival['arrival_price'], arrival['direction']
                lower = direction * 10000 * (low / price - 1)
                upper = direction * 10000 * (high / price - 1)
                mfe, mae = max(0.0, lower, upper), min(0.0, lower, upper)
                terminal = direction * 10000 * (last / price - 1)
            counts[window['status']] += 1
            outcomes.write(_json(dict(
                encounter_id=identifier, source=arrival['source'], session_date=arrival['session_date'],
                start_ns=arrival['start_ns'], horizon_seconds=horizon_seconds, **window,
                mfe_bps=mfe, mae_bps=mae, terminal_return_bps=terminal,
            )) + '\n')
    return {'encounters': sum(counts.values()), 'outcomes': {key: counts[key]
            for key in ('complete', 'censored', 'unknown')}}


def _coverage(db, snapshot, summary, speed, counts):
    warnings = []
    coverage = {}
    for name, view in snapshot['sources'].items():
        raw = summary['sources'].get(name)
        quote_samples = view['quote']['samples']
        trade_samples = view['trade']['samples']
        status = ('missing' if raw is None else 'observed' if quote_samples or trade_samples
                  else 'without_price_observations')
        range_counts = dict(db.execute('''
            SELECT status, count(*) FROM opening_ranges WHERE source = ? GROUP BY status
        ''', (name,)))
        coverage[name] = {
            'status': status, 'market': view['market'], 'session_id': view['session_id'],
            'capture_records': raw or {'records': 0, 'kinds': {}, 'first_ns': None, 'last_ns': None},
            'counters': view['counters'], 'accepted_quotes': quote_samples, 'accepted_trades': trade_samples,
            'opening_range': view['opening_range'], 'range_status_counts': range_counts,
        }
        if raw is None:
            warnings.append(dict(code='missing_source', source=name))
            continue
        for stream in STREAMS[name]:
            if not view[stream]['samples']:
                warnings.append(dict(code='missing_observations', source=name, stream=stream))
            elif not speed[name][stream]['horizons']['120']['ranked_samples']:
                warnings.append(dict(code='limited_speed_samples', source=name, stream=stream))
        if name in TRADE_SOURCES and (range_counts.get('incomplete') or not range_counts.get('complete')):
            warnings.append(dict(code='incomplete_opening_range_coverage', source=name))
    overlap, first, last = db.execute('''
        SELECT count(*), min(session_date), max(session_date) FROM (
            SELECT session_date FROM opening_ranges WHERE status = 'complete'
            GROUP BY session_date HAVING count(*) = ?
        )
    ''', (len(TRADE_SOURCES),)).fetchone()
    if overlap < 2:
        warnings.append(dict(code='short_sample', detail='Fewer than two dates with complete ranges across the trade sources.'))
    if not counts['encounters']:
        warnings.append(dict(code='no_level_encounters'))
    if counts['outcomes']['censored'] or counts['outcomes']['unknown']:
        warnings.append(dict(code='incomplete_outcomes', **counts['outcomes']))
    split = {
        'status': 'guidance_only',
        'session_timezone': snapshot['range_timezone'],
        'dates_with_complete_ranges_across_trade_sources': overlap,
        'first_candidate_date': first,
        'last_candidate_date': last,
        'guidance': [
            'Verify uninterrupted coverage throughout the study window, including complete ranges and outcomes.',
            'Keep all sources and overlapping encounters from a session date together.',
            'Develop settings on earlier complete sessions, freeze them, then evaluate on later untouched sessions.',
            'Choose split dates only after collecting sufficient complete sessions and cursor-time user labels.',
        ],
    }
    return coverage, warnings, split


def generate_report(capture, output, config=None, outcome_seconds=60) -> Path:
    """Measure one immutable capture and write a fresh, self-contained report directory."""
    source = Path(capture).resolve()
    directory = Path(output).resolve()
    if directory.is_relative_to(source):
        raise ValueError('report output must be outside the source capture')
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'report.json'
    report = {'report_version': REPORT_VERSION, 'status': 'running', 'capture': {'capture_path': str(source)}}
    try:
        _write_report(path, report)
        if isinstance(outcome_seconds, bool):
            raise ValueError('outcome_seconds must be finite and positive')
        outcome_seconds = float(outcome_seconds)
        if not math.isfinite(outcome_seconds) or outcome_seconds <= 0:
            raise ValueError('outcome_seconds must be finite and positive')
        grid = _GridSummary()
        engine = SpeedAcceptanceEngine(config, on_grid_sample=grid.add)
        dataset = ResearchCapture(source, cache_dir=directory / 'capture-index')
        report['capture'] = dataset.summary
        with closing(sqlite3.connect(directory / 'measurements.sqlite')) as db:
            db.execute('PRAGMA temp_store=FILE')
            db.executescript(SCHEMA)
            snapshot = _observe(db, dataset, engine)
            counts = _export(db, directory, outcome_seconds, dataset.summary['last_ns'],
                             engine.config.max_age_s * NS)
            speed = grid.summary()
            coverage, warnings, split = _coverage(db, snapshot, dataset.summary, speed, counts)
        report.update(
            status='measurement_only_uncalibrated', engine_version=snapshot['version'],
            config_version=snapshot['config_version'], config=snapshot['config'],
            receipt_clock=snapshot['receipt_clock'], range_clock=snapshot['range_clock'],
            outcome_seconds=outcome_seconds, coverage=coverage, counts=counts, speed=speed,
            speed_sampling={'grid_seconds': 1, 'tail_limit_per_horizon': TAIL_SAMPLES,
                            'tail_quantiles': 'nearest rank of last valid absolute speeds'},
            warnings=warnings, chronological_split=split,
            artifacts={'encounters': 'encounters.ndjson', 'outcomes': 'outcomes.ndjson',
                       'measurements': 'measurements.sqlite', 'capture_index': 'capture-index'},
        )
        _write_report(path, report)
    except BaseException as exc:
        report.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        _write_report(path, report)
        raise
    return path
