"""Disk-backed replay index and immutable checkpoints of active captures."""

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from uuid import uuid4

from src.backend.market_data.storage import iter_capture_records
from src.backtest.paradex_replay import inspect_capture, replay_records
from src.backend.market_data.trades import trade_print


INDEX_VERSION = 2


def snapshot_capture(source: Path, destination: Path) -> Path:
    source = Path(source).resolve()
    status = json.loads((source / 'status.json').read_text())
    if status.get('failed') or any(type(status.get(key)) is not int or status[key] <= 0
                                   for key in ('durable_bytes', 'durable_records')):
        raise ValueError('capture has no healthy durable checkpoint')
    manifest = json.loads((source / 'manifest.json').read_text())
    identity = json.dumps([str(source), manifest, status['durable_records'], status['durable_bytes']], sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    path = Path(destination) / f"{source.name}-checkpoint-{status['durable_records']}-{key}"
    manifest.update(run_id=path.name, replay_snapshot_of=str(source),
                    snapshot_durable_records=status['durable_records'])
    checkpoint_status = {**status, 'closed': True}
    if path.exists():
        try:
            stored_manifest = json.loads((path / 'manifest.json').read_text())
            stored_status = json.loads((path / 'status.json').read_text())
        except (OSError, ValueError) as exc:
            raise ValueError('existing checkpoint provenance is missing or invalid') from exc
        if stored_manifest != manifest or stored_status != checkpoint_status:
            raise ValueError('existing checkpoint provenance does not match the source durable prefix')
        return path
    temporary = path.with_name(path.name + '-' + uuid4().hex + '.tmp')
    temporary.mkdir(parents=True)
    try:
        remaining = status['durable_bytes']
        for segment in sorted(source.glob('records-*.ndjson')):
            if not remaining:
                break
            with segment.open('rb') as incoming, (temporary / segment.name).open('xb') as outgoing:
                while remaining:
                    block = incoming.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    outgoing.write(block)
                    remaining -= len(block)
        if remaining:
            raise ValueError('durable checkpoint exceeds available capture bytes')
        (temporary / 'manifest.json').write_text(json.dumps(manifest))
        (temporary / 'status.json').write_text(json.dumps(checkpoint_status))
        for _ in iter_capture_records(temporary):
            pass
        temporary.rename(path)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return path


class IndexedCapture:
    """Validate raw frames once; subsequent attempts read a compact SQLite cursor."""

    def __init__(self, source: Path, market='BTC-USD-PERP', cache_dir=Path('data/replay_cache')):
        self.source = Path(source).resolve()
        status = json.loads((self.source / 'status.json').read_text())
        if status.get('closed') is not True or status.get('failed'):
            raise ValueError('use a completed capture or a frozen checkpoint')
        segments = sorted(self.source.glob('records-*.ndjson'))
        manifest = (self.source / 'manifest.json').read_bytes()
        tape = self.source / 'chart-trades.ndjson'
        signature = json.dumps([INDEX_VERSION, str(self.source), market, status,
                                hashlib.sha256(manifest).hexdigest(),
                                [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in segments + ([tape] if tape.exists() else [])]])
        key = hashlib.sha256(signature.encode()).hexdigest()[:24]
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = (directory / f'{key}.sqlite').resolve()
        self.market = market
        self.built = not self.path.exists()
        if self.built:
            self._build()
        db = sqlite3.connect(self.path)
        try:
            self.summary = json.loads(db.execute('SELECT value FROM metadata').fetchone()[0])
        finally:
            db.close()

    def _build(self):
        summary = inspect_capture(self.source, self.market)
        temporary = self.path.with_name(self.path.name + '-' + uuid4().hex + '.tmp')
        db = sqlite3.connect(temporary)
        first = last = None
        count = trades = recovered_count = 0
        try:
            db.executescript('''
                PRAGMA journal_mode=OFF;
                CREATE TABLE events (sequence INTEGER PRIMARY KEY, replay_ns INTEGER, wall_ns INTEGER,
                                     session TEXT, kind TEXT, payload TEXT, event_ns INTEGER,
                                     transaction_ns INTEGER, original_sequence INTEGER);
                CREATE TABLE seen_trades (identity TEXT PRIMARY KEY);
                CREATE TABLE recovered (created_ns INTEGER, identity TEXT PRIMARY KEY, payload TEXT);
                CREATE TABLE metadata (value TEXT);
            ''')
            batch = []
            manifest = json.loads((self.source / 'manifest.json').read_text())
            provenance = manifest.get('trade_history')
            tape = self.source / 'chart-trades.ndjson'
            if provenance:
                digest = hashlib.sha256()
                with tape.open('rb') as file:
                    for line in file:
                        digest.update(line)
                        payload = json.loads(line)
                        parsed = trade_print(payload)
                        if parsed:
                            db.execute('INSERT OR IGNORE INTO recovered VALUES (?,?,?)',
                                       (parsed[0], parsed[3], json.dumps(payload)))
                if digest.hexdigest() != provenance['sha256']:
                    raise ValueError('historical trade sidecar hash does not match provenance')
            recovery = iter(db.execute('SELECT created_ns,payload FROM recovered ORDER BY created_ns,identity'))
            upcoming = next(recovery, None)

            def append(record, *, original=None):
                nonlocal count, first, last, trades
                if record['kind'] == 'trade':
                    parsed = trade_print(record['payload'])
                    if parsed is None:
                        return False
                    if not db.execute('INSERT OR IGNORE INTO seen_trades VALUES (?)', (parsed[3],)).rowcount:
                        return False
                    trades += 1
                count += 1
                replay_ns = record['received_monotonic_ns']
                batch.append((count, replay_ns, record['received_at_ns'], record['session_id'], record['kind'],
                              json.dumps(record.get('payload') or {}, separators=(',', ':')),
                              record.get('source_event_time_ns'), record.get('source_transaction_time_ns'), original))
                first = replay_ns if first is None else first
                last = replay_ns
                if len(batch) >= 2000:
                    db.executemany('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)', batch)
                    batch.clear()
                return True

            for record in replay_records(self.source, self.market):
                if record['kind'] == 'bbo':
                    continue
                append(record, original=record['capture_sequence'])
                source_time = record.get('source_event_time_ns') or record['received_at_ns']
                if record['kind'] == 'depth':
                    while upcoming is not None and upcoming[0] <= source_time:
                        payload = json.loads(upcoming[1])
                        payload['recovered'] = True
                        payload['availability_basis'] = 'next recorded book after trade; original trade receive time unknown'
                        recovered_count += append({**record, 'kind': 'trade', 'payload': payload,
                                                   'source_event_time_ns': None,
                                                   'source_transaction_time_ns': upcoming[0]})
                        upcoming = next(recovery, None)
            db.executemany('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)', batch)
            summary.update(index_version=INDEX_VERSION, indexed_events=count,
                           first_replay_ns=first, last_replay_ns=last, market=self.market,
                           trade_count=trades, recovered_trade_count=recovered_count,
                           price_basis='trades' if trades else 'legacy_midpoint',
                           trade_history_provenance=provenance,
                           chart_history_start_ns=provenance['start_at_ms'] * 1000000 if provenance else None)
            db.execute('INSERT INTO metadata VALUES (?)', (json.dumps(summary),))
            db.commit()
            db.close()
            temporary.replace(self.path)
        except BaseException:
            db.close()
            temporary.unlink(missing_ok=True)
            raise

    def inspect(self, market):
        if market != self.market:
            raise ValueError('indexed market does not match replay settings')
        return dict(self.summary)

    def records(self, market):
        self.inspect(market)
        db = sqlite3.connect(f'{self.path.as_uri()}?mode=ro', uri=True, check_same_thread=False)
        try:
            for seq, replay_ns, wall_ns, session, kind, payload, event_ns, trade_ns, original in db.execute('SELECT * FROM events ORDER BY sequence'):
                yield {'capture_sequence': seq, 'received_monotonic_ns': replay_ns,
                       'received_at_ns': wall_ns, 'session_id': session, 'kind': kind,
                       'source': 'paradex', 'market': market, 'payload': json.loads(payload),
                       'source_event_time_ns': event_ns, 'source_transaction_time_ns': trade_ns,
                       'original_capture_sequence': original}
        finally:
            db.close()
