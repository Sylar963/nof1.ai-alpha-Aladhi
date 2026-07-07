"""Read-only loader for MarketData cycle snapshots persisted by the live engine."""

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import MarketData

DEFAULT_DB_PATH = os.path.join('data', 'bot.db')
CYCLE_INTERVAL = 'cycle'
DEFAULT_TOLERANCE_SECONDS = 60.0


def open_readonly_session(db_path: Optional[str] = None):
    """Open a read-only SQLAlchemy session on the bot's SQLite database."""
    path = db_path or DEFAULT_DB_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"database not found: {path}")

    def _connect():
        return sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)

    engine = create_engine("sqlite://", creator=_connect)
    return sessionmaker(bind=engine)()


def row_to_section(row: MarketData) -> Dict:
    """Build the LLM-facing market section for one snapshot row.

    Prefers the persisted ``indicators`` JSON; falls back to the raw
    columns when the JSON is missing or unparseable.
    """
    section: Dict = {}
    if row.indicators:
        try:
            parsed = json.loads(row.indicators)
            if isinstance(parsed, dict):
                section = parsed
        except (json.JSONDecodeError, TypeError):
            section = {}
    section.setdefault('asset', row.asset)
    section.setdefault('current_price', row.close)
    section.setdefault('open_interest', row.open_interest)
    section.setdefault('funding_rate', row.funding_rate)
    section.setdefault('volume_24h', row.volume)
    return section


def group_rows(
    rows: List[MarketData],
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> List[Dict]:
    """Group time-ordered snapshot rows into cycles.

    Rows whose timestamps fall within ``tolerance_seconds`` of the first
    row of the current group belong to the same cycle. Returns a list of
    ``{"timestamp": datetime, "sections": {asset: section}}`` dicts.
    """
    cycles: List[Dict] = []
    anchor: Optional[datetime] = None
    current: Optional[Dict] = None
    for row in rows:
        ts = row.timestamp
        if anchor is None or (ts - anchor).total_seconds() > tolerance_seconds:
            anchor = ts
            current = {"timestamp": ts, "sections": {}}
            cycles.append(current)
        current["sections"][row.asset] = row_to_section(row)
    return cycles


def load_cycles_from_session(
    session,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval: str = CYCLE_INTERVAL,
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> List[Dict]:
    """Load and group cycle snapshots from an existing session."""
    query = session.query(MarketData).filter(MarketData.interval == interval)
    if start is not None:
        query = query.filter(MarketData.timestamp >= start)
    if end is not None:
        query = query.filter(MarketData.timestamp < end)
    rows = query.order_by(MarketData.timestamp.asc(), MarketData.id.asc()).all()
    return group_rows(rows, tolerance_seconds=tolerance_seconds)


def load_cycles(
    db_path: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval: str = CYCLE_INTERVAL,
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> List[Dict]:
    """Open the bot DB read-only and return time-ordered decision cycles."""
    session = open_readonly_session(db_path)
    try:
        return load_cycles_from_session(
            session,
            start=start,
            end=end,
            interval=interval,
            tolerance_seconds=tolerance_seconds,
        )
    finally:
        session.close()
        session.bind.dispose()
