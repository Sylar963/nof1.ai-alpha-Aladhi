"""Backfill script: seeds the IV history store from Deribit DVOL + BTC spot.

Locks the contract that lets the regime classifier's Signal 1 fire on
day 1: write one row per day, bucket-tagged as ``tenor_days=30`` so
``lookback(tenor_days=30, target_age_days=30)`` actually returns a row.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKFILL_PATH = _PROJECT_ROOT / "scripts" / "backfill_iv_history.py"


def _load_backfill_module():
    spec = importlib.util.spec_from_file_location("backfill_iv_history", _BACKFILL_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["backfill_iv_history"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeDeribit:
    """Stand-in matching the two endpoints the backfill script hits."""

    def __init__(self, dvol_bars: list[list[float]], spot_ticks: list[int], spot_closes: list[float]):
        self.dvol_bars = dvol_bars
        self.spot_ticks = spot_ticks
        self.spot_closes = spot_closes
        self.closed = False

    async def get_volatility_index_data(self, **kwargs):
        return self.dvol_bars

    async def _get(self, path, params=None):
        if path.endswith("get_tradingview_chart_data"):
            return {"result": {"ticks": self.spot_ticks, "close": self.spot_closes}}
        raise AssertionError(f"unexpected path {path}")

    async def close(self):
        self.closed = True


def _day_ms(day: datetime) -> int:
    return int(day.replace(tzinfo=timezone.utc).timestamp() * 1000)


@pytest.mark.asyncio
async def test_backfill_writes_one_row_per_day(tmp_path, monkeypatch):
    mod = _load_backfill_module()

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    days = 3
    ticks = [_day_ms(today - timedelta(days=d)) for d in range(days, 0, -1)]
    dvol_bars = [[ts, 50.0, 52.0, 48.0, 50.0] for ts in ticks]
    spot_closes = [70000.0, 71000.0, 72000.0]

    fake = _FakeDeribit(dvol_bars, ticks, spot_closes)
    monkeypatch.setattr(mod, "DeribitPublicClient", lambda: fake)

    db_path = tmp_path / "iv.sqlite"
    written, skipped = await mod.backfill(
        days=days, db_path=str(db_path), dry_run=False, overwrite=False,
    )
    assert written == 3
    assert skipped == 0
    assert fake.closed is True

    from src.backend.options_intel.iv_history_store import IVHistoryStore
    store = IVHistoryStore(db_path=str(db_path))
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT tenor_days, atm_iv, spot_at_init, lower_strike, upper_strike FROM iv_history"
        ).fetchall()
    assert len(rows) == 3
    for tenor, atm_iv, spot, lower, upper in rows:
        assert tenor == 30
        assert atm_iv == pytest.approx(0.50)
        expected_em = spot * 0.50 * math.sqrt(30 / 365.0)
        assert lower == pytest.approx(spot - expected_em)
        assert upper == pytest.approx(spot + expected_em)


@pytest.mark.asyncio
async def test_backfill_skips_existing_days_by_default(tmp_path, monkeypatch):
    mod = _load_backfill_module()

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    ticks = [_day_ms(today - timedelta(days=1))]
    dvol_bars = [[ticks[0], 45.0, 46.0, 44.0, 45.0]]
    spot_closes = [80000.0]

    fake = _FakeDeribit(dvol_bars, ticks, spot_closes)
    monkeypatch.setattr(mod, "DeribitPublicClient", lambda: fake)

    db_path = tmp_path / "iv.sqlite"
    first_written, _ = await mod.backfill(days=1, db_path=str(db_path), dry_run=False, overwrite=False)
    second_written, second_skipped = await mod.backfill(
        days=1, db_path=str(db_path), dry_run=False, overwrite=False,
    )
    assert first_written == 1
    assert second_written == 0
    assert second_skipped == 1


@pytest.mark.asyncio
async def test_backfill_overwrite_flag_writes_duplicates(tmp_path, monkeypatch):
    mod = _load_backfill_module()

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    ticks = [_day_ms(today - timedelta(days=1))]
    dvol_bars = [[ticks[0], 45.0, 46.0, 44.0, 45.0]]
    spot_closes = [80000.0]

    fake = _FakeDeribit(dvol_bars, ticks, spot_closes)
    monkeypatch.setattr(mod, "DeribitPublicClient", lambda: fake)

    db_path = tmp_path / "iv.sqlite"
    await mod.backfill(days=1, db_path=str(db_path), dry_run=False, overwrite=False)
    written, _ = await mod.backfill(days=1, db_path=str(db_path), dry_run=False, overwrite=True)
    assert written == 1

    from src.backend.options_intel.iv_history_store import IVHistoryStore
    store = IVHistoryStore(db_path=str(db_path))
    with store._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM iv_history").fetchone()[0]
    assert count == 2


@pytest.mark.asyncio
async def test_backfilled_anchor_is_findable_by_classifier_lookback(tmp_path, monkeypatch):
    """Signal 1 calls ``lookback(tenor_days=30, target_age_days=30)``;
    verify a backfilled row is actually retrieved at that query."""
    mod = _load_backfill_module()

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    target_day = today - timedelta(days=30)
    ticks = [_day_ms(target_day)]
    dvol_bars = [[ticks[0], 50.0, 50.0, 50.0, 50.0]]
    spot_closes = [70000.0]

    fake = _FakeDeribit(dvol_bars, ticks, spot_closes)
    monkeypatch.setattr(mod, "DeribitPublicClient", lambda: fake)

    db_path = tmp_path / "iv.sqlite"
    await mod.backfill(days=40, db_path=str(db_path), dry_run=False, overwrite=False)

    from src.backend.options_intel.iv_history_store import IVHistoryStore
    store = IVHistoryStore(db_path=str(db_path))
    row = store.lookback(tenor_days=30, target_age_days=30.0)
    assert row is not None
    assert row.spot_at_init == pytest.approx(70000.0)
