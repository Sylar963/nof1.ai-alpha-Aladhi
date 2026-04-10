"""End-to-end test of the venue migration script against a populated SQLite DB."""

import importlib.util
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "migrate_add_venue",
        Path(__file__).resolve().parents[1] / "scripts" / "migrate_add_venue.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_legacy_db(db_path: str) -> None:
    """Create a database that matches the OLD schema (no venue, asset is unique)."""
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset VARCHAR(20) NOT NULL,
                action VARCHAR(10) NOT NULL,
                entry_timestamp DATETIME NOT NULL,
                entry_price FLOAT NOT NULL,
                entry_size FLOAT NOT NULL,
                entry_value FLOAT NOT NULL,
                status VARCHAR(20)
            )
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset VARCHAR(20) NOT NULL UNIQUE,
                side VARCHAR(10) NOT NULL,
                size FLOAT NOT NULL,
                entry_price FLOAT NOT NULL,
                current_price FLOAT NOT NULL,
                liquidation_price FLOAT,
                unrealized_pnl FLOAT NOT NULL DEFAULT 0,
                unrealized_pnl_pct FLOAT NOT NULL DEFAULT 0,
                leverage FLOAT DEFAULT 1.0,
                margin FLOAT NOT NULL,
                trade_id INTEGER,
                opened_at DATETIME NOT NULL,
                updated_at DATETIME
            )
            """
        ))
        conn.execute(text(
            """
            INSERT INTO trades (asset, action, entry_timestamp, entry_price, entry_size, entry_value, status)
            VALUES ('BTC', 'buy', '2026-04-01 12:00:00', 60000, 0.5, 30000, 'closed')
            """
        ))
        conn.execute(text(
            """
            INSERT INTO positions (asset, side, size, entry_price, current_price, margin, opened_at)
            VALUES ('BTC', 'long', 0.5, 60000, 61000, 30000, '2026-04-01 12:00:00')
            """
        ))


def test_migration_adds_columns_and_rebuilds_constraint(tmp_path, monkeypatch):
    db_path = str(tmp_path / "legacy.db")
    _create_legacy_db(db_path)

    monkeypatch.setenv("BOT_DB_PATH", db_path)
    monkeypatch.chdir(tmp_path)

    module = _load_migration_module()
    module.DB_PATH = db_path
    module.DB_URL = f"sqlite:///{db_path}"
    module.main()

    engine = create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)

    trade_cols = {c["name"] for c in inspector.get_columns("trades")}
    assert {"venue", "instrument_name"}.issubset(trade_cols)

    pos_cols = {c["name"] for c in inspector.get_columns("positions")}
    assert {"venue", "instrument_name"}.issubset(pos_cols)

    pos_unique = inspector.get_unique_constraints("positions")
    constraint_columns = [set(c["column_names"]) for c in pos_unique]
    assert {"asset", "venue", "instrument_name"} in constraint_columns
    assert {"asset"} not in constraint_columns

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT venue, instrument_name, asset FROM positions")).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "hyperliquid"
        assert rows[0][2] == "BTC"

        trade_rows = conn.execute(text("SELECT venue FROM trades")).fetchall()
        assert trade_rows[0][0] == "hyperliquid"


def test_migration_is_idempotent(tmp_path, monkeypatch):
    """Running migrate twice in a row must not break the schema or duplicate work."""
    db_path = str(tmp_path / "legacy.db")
    _create_legacy_db(db_path)

    monkeypatch.setenv("BOT_DB_PATH", db_path)
    monkeypatch.chdir(tmp_path)

    module = _load_migration_module()
    module.DB_PATH = db_path
    module.DB_URL = f"sqlite:///{db_path}"
    module.main()
    module.main()  # second pass must be a no-op

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT COUNT(*) FROM positions")).fetchone()
        assert rows[0] == 1
