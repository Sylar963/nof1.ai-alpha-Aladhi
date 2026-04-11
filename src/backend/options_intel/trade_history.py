"""Recent options trade history query for the OptionsContext snapshot.

Reads the existing :class:`Trade` table filtered to closed Thalex trades
and returns the last N (newest first) as compact dicts the LLM digest can
embed under ``recent_options_trades``.

Failure mode is "return an empty list, log a warning" — the snapshot is
allowed to ship without history if the database is briefly unavailable
(e.g. SQLite locked by another process). The bot keeps trading; the LLM
just loses one piece of context for one cycle.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import desc

from src.database.models import Trade


logger = logging.getLogger(__name__)


def fetch_recent_options_trades(session: Any, limit: int = 5) -> list[dict]:
    """Return the last ``limit`` closed Thalex trades, newest first.

    Args:
        session: a SQLAlchemy session bound to the bot's database.
        limit: maximum number of entries to return.

    Returns:
        List of dicts with keys ``instrument_name``, ``pnl_usd``, ``pnl_pct``,
        ``rationale``, and ``closed_at`` (ISO-8601 string for clean JSON).
        Empty list when there are no matching trades or the query fails.
    """
    try:
        rows = (
            session.query(Trade)
            .filter(Trade.venue == "thalex")
            .filter(Trade.status == "closed")
            .order_by(desc(Trade.exit_timestamp))
            .limit(int(limit))
            .all()
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("fetch_recent_options_trades failed: %s", exc)
        return []

    out: list[dict] = []
    for row in rows:
        closed_at = row.exit_timestamp.isoformat() if row.exit_timestamp else None
        out.append({
            "instrument_name": row.instrument_name or row.asset or "",
            "pnl_usd": float(row.realized_pnl) if row.realized_pnl is not None else None,
            "pnl_pct": float(row.realized_pnl_pct) if row.realized_pnl_pct is not None else None,
            "rationale": row.rationale or "",
            "closed_at": closed_at,
        })
    return out
