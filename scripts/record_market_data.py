import argparse
import asyncio
import logging
import signal
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend.config_loader import CONFIG
from src.backend.market_data.recorder import CaptureConfig, record_market_data


async def _run(config):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    registered = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
            registered.append(sig)
        except NotImplementedError:
            pass
    try:
        run_dir = await record_market_data(config, stop=stop)
        print(f"Capture complete: {run_dir}")
    finally:
        for sig in registered:
            loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description="Record public Binance/Paradex market data; never submits orders.")
    parser.add_argument("--output", type=Path, help="Parent directory for a new capture run")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds; otherwise run until Ctrl-C")
    parser.add_argument("--sources", help="Comma-separated public source names; defaults to all four groups")
    parser.add_argument("--symbol", help="Binance spot/perpetual symbol, default BTCUSDT")
    parser.add_argument("--paradex-market", help="Paradex native market, default BTC-USD-PERP")
    parser.add_argument("--max-total-bytes", type=int, help="Capture record cap for this run; stop rather than delete")
    parser.add_argument("--max-file-bytes", type=int, help="Rotate NDJSON segments at this size")
    parser.add_argument("--max-queue-bytes", type=int, help="Maximum pending serialized data")
    args = parser.parse_args()
    try:
        config = CaptureConfig.from_config(CONFIG)
        changes = {field: value for field, value in {
            "output_dir": args.output, "duration": args.duration, "symbol": args.symbol,
            "paradex_market": args.paradex_market, "max_total_bytes": args.max_total_bytes,
            "max_file_bytes": args.max_file_bytes, "max_queue_bytes": args.max_queue_bytes,
        }.items() if value is not None}
        if args.sources is not None:
            changes["sources"] = tuple(s.strip() for s in args.sources.split(","))
        config = replace(config, **changes)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(_run(config))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.error("Capture stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
