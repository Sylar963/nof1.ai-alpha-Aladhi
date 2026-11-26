"""Build a separate trade-price replay sample from an older depth-only capture."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.trade_history import recover_trade_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--warmup-hours', type=float, default=6)
    args = parser.parse_args()
    path = recover_trade_history(args.capture, warmup_hours=args.warmup_hours)
    print(f'Trade-price replay created: {path}')


if __name__ == '__main__':
    main()
