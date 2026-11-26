"""Launch the lightweight chart replay without loading the live bot or NiceGUI."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description='Lightweight Paradex chart replay')
    parser.add_argument('--capture', type=Path, help='Recorder run; active runs load as frozen checkpoints')
    parser.add_argument('--port', type=int, default=8081)
    args = parser.parse_args()
    from aiohttp import web
    from src.gui.replay_server import create_server
    web.run_app(create_server(args.capture), host='127.0.0.1', port=args.port, print=print)


if __name__ in {'__main__', '__mp_main__'}:
    main()
