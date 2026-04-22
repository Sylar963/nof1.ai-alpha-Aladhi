"""Tests for the IV history SQLite store.

The store persists per-tenor straddle anchors so the regime classifier can
look back ~15 days and compare current spot to where the implied range was
back then. Tests use an in-memory SQLite database for speed and isolation."""

from datetime import datetime, timedelta, timezone

import pytest

from src.backend.options_intel.iv_history_store import (
    IVHistoryRow,
    IVHistoryStore,
)


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "iv_history.sqlite"
    return IVHistoryStore(db_path=str(db_path))


def _row(ts, tenor=15, atm_iv=0.65, em=0.08, spot=60000.0, lower=55000.0, upper=65000.0):
    return IVHistoryRow(
        ts=ts,
        tenor_days=tenor,
        atm_iv=atm_iv,
        atm_straddle_em=em,
        spot_at_init=spot,
        lower_strike=lower,
        upper_strike=upper,
    )


# ---------------------------------------------------------------------------
# Write + read
# ---------------------------------------------------------------------------


def test_write_then_read_round_trips(store):
    now = datetime.now(timezone.utc)
    store.write(_row(now))
    rows = store.read_recent(tenor_days=15, limit=10)
    assert len(rows) == 1
    assert rows[0].atm_iv == pytest.approx(0.65)
    assert rows[0].lower_strike == pytest.approx(55000.0)
    assert rows[0].upper_strike == pytest.approx(65000.0)


def test_lookback_at_target_age_finds_closest_row(store):
    """Lookback returns the row whose age is closest to the target."""
    now = datetime.now(timezone.utc)
    store.write(_row(now - timedelta(days=14, hours=12), atm_iv=0.50, em=0.06, lower=53000, upper=57000))
    store.write(_row(now - timedelta(days=15, hours=2), atm_iv=0.55, em=0.07, lower=52000, upper=58000))
    store.write(_row(now - timedelta(days=20), atm_iv=0.60, em=0.08, lower=51000, upper=59000))

    found = store.lookback(tenor_days=15, target_age_days=15, now=now)
    assert found is not None
    # The 15d-2h row is closest to a 30-day lookback.
    assert found.atm_iv == pytest.approx(0.55)


def test_lookback_returns_none_when_history_empty(store):
    found = store.lookback(tenor_days=15, target_age_days=15)
    assert found is None


def test_prune_drops_rows_older_than_30_days(store):
    """The 30-day rolling window keeps the table tiny."""
    now = datetime.now(timezone.utc)
    store.write(_row(now - timedelta(days=10)))
    store.write(_row(now - timedelta(days=29)))
    store.write(_row(now - timedelta(days=31)))
    store.write(_row(now - timedelta(days=60)))

    deleted = store.prune(retention_days=30, now=now)
    assert deleted == 2

    remaining = store.read_recent(tenor_days=15, limit=100)
    assert len(remaining) == 2
