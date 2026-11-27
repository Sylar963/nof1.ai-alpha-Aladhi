"""Receipt-ordered research data independent of the Paradex execution index."""

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from src.backend.market_data.storage import iter_capture_records


INDEX_VERSION = 1


class ResearchCapture:
    def __init__(self, source: Path, cache_dir=Path('data/replay_cache')):
        self.source = Path(source).resolve()
        manifest_bytes = (self.source / 'manifest.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        status = json.loads((self.source / 'status.json').read_text())
        if manifest.get('capture_schema_version') != 1:
            raise ValueError('unsupported capture schema')
        if status.get('closed') is not True or status.get('failed') is not False:
            raise ValueError('research requires a healthy closed capture or frozen checkpoint')
        segments = sorted(self.source.glob('records-*.ndjson'))
        signature = json.dumps([INDEX_VERSION, str(self.source), status,
                                hashlib.sha256(manifest_bytes).hexdigest(),
                                [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in segments]])
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / ('research-' + hashlib.sha256(signature.encode()).hexdigest()[:24] + '.sqlite')
        self.built = not self.path.exists()
        if self.built:
            self._build(manifest, status)
        with closing(sqlite3.connect(self.path)) as db:
            self.summary = json.loads(db.execute('SELECT value FROM metadata').fetchone()[0])

    def _build(self, manifest, status):
        temporary = self.path.with_name(self.path.name + '-' + uuid4().hex + '.tmp')
        db = sqlite3.connect(temporary)
        sources = {}
        first = last = None
        count = delayed = 0
        try:
            db.executescript('''
                PRAGMA journal_mode=OFF;
                CREATE TABLE events (sequence INTEGER PRIMARY KEY, receipt_ns INTEGER, record TEXT);
                CREATE INDEX receipt ON events(receipt_ns, sequence);
                CREATE TABLE metadata (value TEXT);
            ''')
            batch = []
            for record in iter_capture_records(self.source):
                ns = record.get('received_monotonic_ns')
                if type(ns) is not int or ns < 0:
                    raise ValueError('research receipt clock missing or invalid')
                available = max(ns, last if last is not None else ns)
                delayed += available != ns
                if type(record.get('received_at_ns')) is not int or record['received_at_ns'] < 0:
                    raise ValueError('research UTC receipt clock missing or invalid')
                source = record.get('source')
                if source:
                    summary = sources.setdefault(source, {'records': 0, 'kinds': {}, 'first_ns': ns, 'last_ns': ns})
                    summary['records'] += 1
                    summary['last_ns'] = ns
                    kind = record.get('kind', 'unknown')
                    summary['kinds'][kind] = summary['kinds'].get(kind, 0) + 1
                compact = {key: value for key, value in record.items()
                           if key not in {'raw_text', 'raw_base64', 'raw', 'raw_unavailable'}}
                compact['available_monotonic_ns'] = available
                batch.append((record['capture_sequence'], available, json.dumps(compact, allow_nan=False)))
                count += 1
                first = ns if first is None else first
                last = available
                if len(batch) >= 2000:
                    db.executemany('INSERT INTO events VALUES (?,?,?)', batch)
                    batch.clear()
            db.executemany('INSERT INTO events VALUES (?,?,?)', batch)
            summary = {'index_version': INDEX_VERSION, 'run_id': manifest.get('run_id', self.source.name),
                       'capture_path': str(self.source), 'capture_manifest': manifest,
                       'durable_records': status['durable_records'], 'durable_bytes': status['durable_bytes'],
                       'indexed_records': count, 'first_ns': first, 'last_ns': last, 'sources': sources,
                       'delayed_submissions': delayed,
                       'recovered_sidecar_excluded': bool(manifest.get('trade_history')),
                       'ordering': 'capture_sequence; actual receipt clocks; no exchange-time sort'}
            db.execute('INSERT INTO metadata VALUES (?)', (json.dumps(summary, allow_nan=False),))
            db.commit()
            db.close()
            temporary.replace(self.path)
        except BaseException:
            db.close()
            temporary.unlink(missing_ok=True)
            raise

    def records(self, until_ns=None):
        db = sqlite3.connect(f'{self.path.resolve().as_uri()}?mode=ro', uri=True, check_same_thread=False)
        try:
            query = 'SELECT record FROM events'
            params = ()
            if until_ns is not None:
                if type(until_ns) is not int or until_ns < 0:
                    raise ValueError('until_ns must be a non-negative receipt timestamp')
                query += ' WHERE receipt_ns <= ?'
                params = (until_ns,)
            query += ' ORDER BY sequence'
            for (record,) in db.execute(query, params):
                yield json.loads(record)
        finally:
            db.close()
