"""Persistent SQLite store for ATM straddle anchors and IV history.

The store backs the regime classifier's straddle expected-move test by
keeping a rolling 30-day window of per-tenor anchors. Each row records:

- ``ts``                   — UTC ISO timestamp the anchor was captured
- ``tenor_days``           — option tenor (e.g. 15 for the 15-day anchor)
- ``atm_iv``               — ATM IV at capture time
- ``atm_straddle_em``      — ATM straddle premium / spot (implied move %)
- ``spot_at_init``         — spot price when the anchor was captured
- ``lower_strike``         — spot − straddle premium
- ``upper_strike``         — spot + straddle premium

The store is intentionally tiny (one table, four methods). It uses an
out-of-band SQLite database file, separate from the bot's main ``bot.db``,
so the options-intel package can be exercised in isolation without touching
the trading database.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class IVHistoryRow:
    """One persisted snapshot of the per-tenor straddle anchor."""

    ts: datetime
    tenor_days: int
    atm_iv: float
    atm_straddle_em: float
    spot_at_init: float
    lower_strike: float
    upper_strike: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS iv_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    tenor_days INTEGER NOT NULL,
    atm_iv REAL NOT NULL,
    atm_straddle_em REAL NOT NULL,
    spot_at_init REAL NOT NULL,
    lower_strike REAL NOT NULL,
    upper_strike REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_iv_history_tenor_ts ON iv_history (tenor_days, ts);
"""


class IVHistoryStore:
    """Lightweight SQLite-backed history store for straddle anchors."""

    def __init__(self, db_path: str = "data/iv_history.sqlite") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def write(self, row: IVHistoryRow) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO iv_history
                    (ts, tenor_days, atm_iv, atm_straddle_em,
                     spot_at_init, lower_strike, upper_strike)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _ts_to_iso(row.ts),
                    int(row.tenor_days),
                    float(row.atm_iv),
                    float(row.atm_straddle_em),
                    float(row.spot_at_init),
                    float(row.lower_strike),
                    float(row.upper_strike),
                ),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def read_recent(self, tenor_days: int, limit: int = 100) -> list[IVHistoryRow]:
        """Return the most recent rows for a given tenor, newest first."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT ts, tenor_days, atm_iv, atm_straddle_em,
                       spot_at_init, lower_strike, upper_strike
                FROM iv_history
                WHERE tenor_days = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (int(tenor_days), int(limit)),
            )
            return [_row_from_tuple(row) for row in cursor.fetchall()]

    def lookback(
        self,
        tenor_days: int,
        target_age_days: float,
        now: Optional[datetime] = None,
    ) -> Optional[IVHistoryRow]:
        """Return the row whose age is closest to ``target_age_days``.

        Used by the regime classifier: "compare current spot to where the
        15-day straddle range was 15 days ago".
        """
        now = now or datetime.now(timezone.utc)
        target_ts = now - _days(target_age_days)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT ts, tenor_days, atm_iv, atm_straddle_em,
                       spot_at_init, lower_strike, upper_strike,
                       ABS(julianday(ts) - julianday(?)) AS distance
                FROM iv_history
                WHERE tenor_days = ?
                ORDER BY distance ASC
                LIMIT 1
                """,
                (_ts_to_iso(target_ts), int(tenor_days)),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _row_from_tuple(row[:7])

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune(self, retention_days: float = 30.0, now: Optional[datetime] = None) -> int:
        """Drop rows older than ``retention_days``. Returns the number deleted."""
        now = now or datetime.now(timezone.utc)
        cutoff = now - _days(retention_days)
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM iv_history WHERE ts < ?",
                (_ts_to_iso(cutoff),),
            )
            conn.commit()
            return cursor.rowcount or 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts_to_iso(ts: datetime) -> str:
    """Serialize datetimes as ISO-8601 with explicit UTC offset."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _ts_from_iso(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def _days(amount: float):
    from datetime import timedelta
    return timedelta(days=float(amount))


def _row_from_tuple(values) -> IVHistoryRow:
    return IVHistoryRow(
        ts=_ts_from_iso(values[0]),
        tenor_days=int(values[1]),
        atm_iv=float(values[2]),
        atm_straddle_em=float(values[3]),
        spot_at_init=float(values[4]),
        lower_strike=float(values[5]),
        upper_strike=float(values[6]),
    )
