"""Measure Python replay startup, throughput and memory on a real recording."""

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.paradex_replay import ReplayConfig
from src.backtest.replay_dataset import IndexedCapture
from src.backtest.replay_workbench import Workbench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    dataset = IndexedCapture(args.capture)
    index_seconds = time.perf_counter() - started
    started = time.perf_counter()
    work = Workbench(dataset, ReplayConfig(), journal_dir=None, warmup_seconds=300)
    startup_seconds = time.perf_counter() - started
    before_events, before_clock = work.processed, work.engine.clock_ns
    started = time.perf_counter()
    while not work.engine.finished:
        work.advance(dataset.summary['last_replay_ns'])
    elapsed = time.perf_counter() - started
    count = work.processed - before_events
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({
        'python': platform.python_version(), 'recording': str(dataset.source),
        'raw_bytes': dataset.summary['durable_bytes'], 'index_bytes': dataset.path.stat().st_size,
        'index_reused': not dataset.built, 'index_open_or_build_seconds': round(index_seconds, 3),
        'attempt_startup_ms_including_warmup': round(startup_seconds * 1000, 2),
        'processed_events': count, 'processing_seconds': round(elapsed, 3),
        'events_per_second': round(count / max(elapsed, 0.000001)),
        'replayed_seconds': round((work.engine.clock_ns - before_clock) / 1e9, 3),
        'peak_process_rss_mib': round(maximum_rss / (1048576 if sys.platform == 'darwin' else 1024), 2),
        'chart_candle_limit_per_timeframe': 2000,
    }, indent=2))
    work.close()


if __name__ == '__main__':
    main()
