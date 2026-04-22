#!/usr/bin/env python3
"""Backfill the IV history store with ~2 months of Deribit-derived anchors.

Without this, the regime classifier's Signal 1 ("is spot outside the
implied range we had 30 days ago?") returns ``unknown`` for the first
30 days of bot runtime because the ``iv_history`` table is empty.
Running this once at setup writes one synthetic anchor per day over the
lookback window so Signal 1 can fire on the first options cycle.

The anchors are built from:
  * Deribit DVOL (30-day BTC implied-vol index, in percent) as the IV proxy
  * BTC-PERPETUAL daily mark price as the spot reference

DVOL is Deribit's native 30-day constant-maturity index, so the target
tenor matches the data exactly. Live cycles keep adding real Thalex/
Deribit-sourced anchors alongside, and ``lookback()`` picks the row
whose age is closest to the 30-day target.

Run **once** at setup. The bot writes a fresh anchor every options
refresh cycle afterwards, so the store stays topped up on its own. If
the bot is offline for weeks, re-run with the gap length.

Usage:
    python scripts/backfill_iv_history.py --days 60
    python scripts/backfill_iv_history.py --days 90 --dry-run
    python scripts/backfill_iv_history.py --days 60 --overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backend.options_intel.deribit_client import DeribitPublicClient
from src.backend.options_intel.iv_history_store import IVHistoryRow, IVHistoryStore


_TARGET_TENOR_DAYS = 30
_YEARS_PER_DAY = 1.0 / 365.0


def _align_by_date(series) -> dict[str, float]:
    out: dict[str, float] = {}
    for ts_ms, val in series:
        day = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).date().isoformat()
        out[day] = float(val)
    return out


def _existing_days_for_tenor(store: IVHistoryStore, tenor_days: int) -> set[str]:
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT ts FROM iv_history WHERE tenor_days = ?",
            (int(tenor_days),),
        ).fetchall()
    out: set[str] = set()
    for (ts,) in rows:
        try:
            out.add(datetime.fromisoformat(str(ts)).date().isoformat())
        except ValueError:
            continue
    return out


async def _fetch_inputs(client: DeribitPublicClient, start_ms: int, end_ms: int):
    dvol_bars = await client.get_volatility_index_data(
        currency="BTC",
        start_timestamp_ms=start_ms,
        end_timestamp_ms=end_ms,
        resolution="1D",
    )
    spot_payload = await client._get(
        "/public/get_tradingview_chart_data",
        params={
            "instrument_name": "BTC-PERPETUAL",
            "start_timestamp": str(start_ms),
            "end_timestamp": str(end_ms),
            "resolution": "1D",
        },
    )
    spot_result = spot_payload.get("result") if isinstance(spot_payload, dict) else None
    spot_rows: list[tuple[int, float]] = []
    if isinstance(spot_result, dict):
        ticks = spot_result.get("ticks") or []
        closes = spot_result.get("close") or []
        for ts_ms, close in zip(ticks, closes):
            try:
                spot_rows.append((int(ts_ms), float(close)))
            except (TypeError, ValueError):
                continue

    dvol_rows = [(int(b[0]), float(b[4])) for b in dvol_bars if b]
    return dvol_rows, spot_rows


async def backfill(days: int, db_path: str, dry_run: bool, overwrite: bool) -> tuple[int, int]:
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    client = DeribitPublicClient()
    try:
        dvol_rows, spot_rows = await _fetch_inputs(client, start_ms, end_ms)
    finally:
        await client.close()

    if not dvol_rows:
        print("No DVOL data returned from Deribit — aborting.")
        return (0, 0)
    if not spot_rows:
        print("No BTC-PERPETUAL data returned from Deribit — aborting.")
        return (0, 0)

    dvol_by_day = _align_by_date(dvol_rows)
    spot_by_day = _align_by_date(spot_rows)
    common_days = sorted(set(dvol_by_day) & set(spot_by_day))
    if not common_days:
        print("DVOL and spot series have no overlapping days — aborting.")
        return (0, 0)

    store = IVHistoryStore(db_path=db_path)
    existing = set() if overwrite else _existing_days_for_tenor(store, _TARGET_TENOR_DAYS)

    written = 0
    skipped = 0
    for day in common_days:
        if day in existing:
            skipped += 1
            continue
        dvol_pct = dvol_by_day[day]
        spot_close = spot_by_day[day]
        if dvol_pct <= 0 or spot_close <= 0:
            skipped += 1
            continue
        atm_iv = dvol_pct / 100.0
        em = spot_close * atm_iv * math.sqrt(_TARGET_TENOR_DAYS * _YEARS_PER_DAY)
        row = IVHistoryRow(
            ts=datetime.fromisoformat(day + "T00:00:00+00:00"),
            tenor_days=_TARGET_TENOR_DAYS,
            atm_iv=atm_iv,
            atm_straddle_em=em / spot_close,
            spot_at_init=spot_close,
            lower_strike=spot_close - em,
            upper_strike=spot_close + em,
        )
        if dry_run:
            print(
                f"[dry-run] {day}: spot=${spot_close:>9,.0f} dvol={dvol_pct:5.1f}% "
                f"range=[${row.lower_strike:>9,.0f}, ${row.upper_strike:>9,.0f}]"
            )
            written += 1
            continue
        try:
            store.write(row)
            written += 1
        except Exception as exc:
            print(f"[warn] failed to write {day}: {exc}")
            skipped += 1

    return (written, skipped)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=60, help="Lookback window in days (default 60)")
    parser.add_argument("--db-path", type=str, default="data/iv_history.sqlite")
    parser.add_argument("--dry-run", action="store_true", help="Print rows without writing")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Write a new row even when one already exists for a given day (default: skip)",
    )
    args = parser.parse_args()

    written, skipped = asyncio.run(
        backfill(args.days, args.db_path, args.dry_run, args.overwrite),
    )
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb}: {written} anchors (tenor_days={_TARGET_TENOR_DAYS}), skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
