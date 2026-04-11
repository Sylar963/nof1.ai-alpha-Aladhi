"""Idempotent migration: add venue + instrument_name columns to an existing DB.

The project doesn't use Alembic — schema is created via
``Base.metadata.create_all``. For an existing populated SQLite database that
predates the multi-venue refactor, this script:

1. Adds ``venue`` and ``instrument_name`` columns to ``trades`` if missing.
2. Adds ``venue`` and ``instrument_name`` columns to ``positions`` if missing.
3. Backfills both with ``venue='hyperliquid'`` for existing rows.
4. Replaces the old single-column unique constraint on ``positions.asset``
   with the new composite ``(asset, venue, instrument_name)`` constraint via
   the SQLite recreate-table dance (since SQLite cannot ALTER constraints).

Run with: ``python scripts/migrate_add_venue.py``

This script is safe to run multiple times — every step checks current state
before mutating the schema.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine, inspect, text  # noqa: E402

from src.database.models import Base, create_tables  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DB_PATH = os.environ.get("BOT_DB_PATH") or os.path.join("data", "bot.db")
DB_URL = f"sqlite:///{DB_PATH}"


def _has_column(inspector, table: str, column: str) -> bool:
    return any(c["name"] == column for c in inspector.get_columns(table))


def _add_column_if_missing(engine, table: str, column: str, ddl: str) -> bool:
    inspector = inspect(engine)
    if _has_column(inspector, table, column):
        logger.info("  - %s.%s already present", table, column)
        return False
    logger.info("  + ALTER TABLE %s ADD COLUMN %s %s", table, column, ddl)
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
    return True


def _backfill_venue(engine, table: str) -> None:
    with engine.begin() as conn:
        result = conn.execute(
            text(f"UPDATE {table} SET venue = 'hyperliquid' WHERE venue IS NULL")
        )
        if result.rowcount:
            logger.info("  ~ Backfilled venue='hyperliquid' on %s rows of %s", result.rowcount, table)


def _rebuild_positions_table(engine) -> None:
    """Drop the old single-column unique constraint by recreating the table.

    SQLite cannot ALTER away a constraint, so we:
      1. Rename positions → positions_old
      2. Recreate positions with the new schema (via Base.metadata.create_all)
      3. Copy rows back, defaulting venue='hyperliquid' and instrument_name=asset
      4. Drop positions_old
    """
    inspector = inspect(engine)
    if "positions" not in inspector.get_table_names():
        return

    constraints = inspector.get_unique_constraints("positions")
    has_old_constraint = any(
        set(c["column_names"]) == {"asset"} for c in constraints
    )
    has_new_constraint = any(
        set(c["column_names"]) == {"asset", "venue", "instrument_name"} for c in constraints
    )
    if has_new_constraint and not has_old_constraint:
        logger.info("  - positions table already on the new constraint")
        return

    logger.info("  ~ Rebuilding positions table to swap unique constraint")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE positions RENAME TO positions_old"))

    create_tables(engine)  # rebuilds positions with the new schema

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO positions (
                    id, asset, venue, instrument_name, side, size, entry_price,
                    current_price, liquidation_price, unrealized_pnl,
                    unrealized_pnl_pct, leverage, margin, trade_id, opened_at, updated_at
                )
                SELECT
                    id, asset, COALESCE(venue, 'hyperliquid'),
                    COALESCE(instrument_name, asset), side, size, entry_price,
                    current_price, liquidation_price, unrealized_pnl,
                    unrealized_pnl_pct, leverage, margin, trade_id, opened_at, updated_at
                FROM positions_old
                """
            )
        )
        conn.execute(text("DROP TABLE positions_old"))
    logger.info("  + positions table rebuilt with composite unique constraint")


def main() -> None:
    if not os.path.exists(DB_PATH):
        logger.info("No existing database at %s — running create_tables() to bootstrap.", DB_PATH)
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        engine = create_engine(DB_URL)
        create_tables(engine)
        logger.info("Done.")
        return

    engine = create_engine(DB_URL)
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    logger.info("Migrating %s (tables: %s)", DB_PATH, existing_tables)

    if "trades" in existing_tables:
        _add_column_if_missing(engine, "trades", "venue", "VARCHAR(20) DEFAULT 'hyperliquid'")
        _add_column_if_missing(engine, "trades", "instrument_name", "VARCHAR(64)")
        _backfill_venue(engine, "trades")

    if "positions" in existing_tables:
        _add_column_if_missing(engine, "positions", "venue", "VARCHAR(20) DEFAULT 'hyperliquid'")
        _add_column_if_missing(engine, "positions", "instrument_name", "VARCHAR(64)")
        _backfill_venue(engine, "positions")
        _rebuild_positions_table(engine)

    Base.metadata.create_all(engine)
    logger.info("Migration complete.")


if __name__ == "__main__":
    main()
