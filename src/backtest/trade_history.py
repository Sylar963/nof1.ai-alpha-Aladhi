"""Recover an explicitly labelled trade-price history without changing raw depth."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests

from src.backend.market_data.storage import iter_capture_records
from src.backend.market_data.trades import trade_print
from src.backtest.paradex_replay import inspect_capture


TRADE_URL = 'https://api.prod.paradex.trade/v1/trades'


def recover_trade_history(source: Path, output=Path('data/replay_repaired'), *, market='BTC-USD-PERP',
                          warmup_hours=6, max_pages=1000, get=None):
    source = Path(source).resolve()
    summary = inspect_capture(source, market)
    if summary['trade_count']:
        raise ValueError('capture already contains live trade prices; use it directly')
    if not 0 <= warmup_hours <= 24 or not 1 <= max_pages <= 1000:
        raise ValueError('warmup must be 0..24 hours and page limit 1..1000')
    first_ns = last_ns = None
    for record in iter_capture_records(source):
        if record.get('source') == 'paradex' and record.get('market') == market and record['kind'] == 'depth':
            clock = record.get('source_event_time_ns') or record['received_at_ns']
            first_ns = clock if first_ns is None else min(first_ns, clock)
            last_ns = clock if last_ns is None else max(last_ns, clock)
    start_ms = first_ns // 1000000 - int(warmup_hours * 3600000)
    end_ms = last_ns // 1000000
    run_id = source.name + '-trades-' + uuid4().hex[:8]
    destination = Path(output) / run_id
    temporary = destination.with_name(destination.name + '.tmp')
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)
    seen, cursors = {}, set()
    pages = count = total_bytes = 0
    digest = hashlib.sha256()
    get = get or requests.get
    try:
        with (temporary / 'chart-trades.ndjson').open('wb') as tape:
            params = {'market': market, 'start_at': start_ms, 'end_at': end_ms, 'page_size': 1000}
            while True:
                if pages >= max_pages:
                    raise ValueError('trade history page limit reached before pagination was exhausted')
                reply = get(TRADE_URL, params=params, timeout=30)
                reply.raise_for_status()
                raw = reply.content
                total_bytes += len(raw)
                if total_bytes > 256 * 1024 * 1024:
                    raise ValueError('trade history response budget exceeded')
                body = reply.json()
                if not isinstance(body, dict) or not isinstance(body.get('results'), list):
                    raise ValueError('invalid historical trade page')
                pages += 1
                (temporary / f'trade-history-page-{pages:04d}.json').write_bytes(raw)
                for row in body['results']:
                    if not isinstance(row, dict) or row.get('market') != market:
                        raise ValueError('historical trade belongs to an unexpected market')
                    timestamp = row.get('created_at')
                    if type(timestamp) is not int or not start_ms <= timestamp <= end_ms:
                        raise ValueError('historical trade falls outside the requested time interval')
                    identity = (row.get('trade_type'), row.get('id'))
                    if identity in seen:
                        if seen[identity] != row:
                            raise ValueError('historical trade identity has conflicting contents')
                        continue
                    seen[identity] = row
                    parsed = trade_print(row)
                    if parsed is None:
                        continue
                    line = (json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n').encode()
                    tape.write(line)
                    digest.update(line)
                    count += 1
                cursor = body.get('next')
                if not cursor:
                    break
                if not isinstance(cursor, str) or cursor in cursors:
                    raise ValueError('historical trade cursor repeated or is invalid')
                cursors.add(cursor)
                params = {'cursor': cursor}
        if not count:
            raise ValueError('production trade history returned no price-forming transactions')
        manifest = json.loads((temporary / 'manifest.json').read_text())
        manifest.update(run_id=run_id, repaired_from=str(source), trade_history={
            'endpoint': TRADE_URL, 'environment': 'production', 'market': market,
            'start_at_ms': start_ms, 'end_at_ms': end_ms, 'pages': pages, 'prints': count,
            'pagination_exhausted': True, 'original_receive_times_available': False,
            'availability': 'historical prices become visible at the next recorded book after matching time',
            'fetched_at_utc': datetime.now(timezone.utc).isoformat(), 'sha256': digest.hexdigest(),
        })
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        temporary.rename(destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary)
        raise
