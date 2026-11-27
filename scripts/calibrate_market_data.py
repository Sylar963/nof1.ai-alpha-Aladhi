"""Measure a closed public-market capture or frozen checkpoint offline."""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.calibration_report import generate_report


def positive_seconds(value):
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected finite, positive seconds') from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError('expected finite, positive seconds')
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True, help='closed capture or frozen checkpoint')
    parser.add_argument('--output', type=Path, required=True, help='new report directory')
    parser.add_argument('--outcome-seconds', type=positive_seconds, default=60,
                        help='transaction outcome horizon in seconds (default: 60)')
    args = parser.parse_args()
    try:
        print(generate_report(args.capture, args.output, outcome_seconds=args.outcome_seconds))
    except Exception as exc:
        parser.exit(1, f'{parser.prog}: {exc}\n')


if __name__ == '__main__':
    main()
