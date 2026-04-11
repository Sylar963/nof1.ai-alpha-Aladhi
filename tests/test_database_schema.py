"""Tests for the multi-venue database schema additions."""

from sqlalchemy import UniqueConstraint, create_engine, inspect

from src.database.models import Base, Position, Trade, create_tables


def test_trade_has_venue_column():
    """Trade rows must record which venue executed the trade."""
    cols = {c.name: c for c in Trade.__table__.columns}
    assert "venue" in cols
    assert cols["venue"].nullable is False
    assert cols["venue"].default.arg == "hyperliquid"


def test_trade_has_instrument_name_column():
    """For options venues we need the full instrument name distinct from `asset`."""
    cols = {c.name: c for c in Trade.__table__.columns}
    assert "instrument_name" in cols
    assert cols["instrument_name"].nullable is True


def test_position_has_venue_column():
    cols = {c.name: c for c in Position.__table__.columns}
    assert "venue" in cols
    assert cols["venue"].nullable is False
    assert cols["venue"].default.arg == "hyperliquid"


def test_position_has_instrument_name_column():
    cols = {c.name: c for c in Position.__table__.columns}
    assert "instrument_name" in cols


def test_position_asset_no_longer_has_unique_constraint():
    """The single-asset unique constraint is incompatible with multi-venue."""
    cols = {c.name: c for c in Position.__table__.columns}
    assert cols["asset"].unique is not True


def test_position_has_composite_unique_constraint():
    """Positions are now unique by (asset, venue, instrument_name)."""
    constraints = [c for c in Position.__table__.constraints if isinstance(c, UniqueConstraint)]
    target_columns = {"asset", "venue", "instrument_name"}
    assert any(
        {col.name for col in c.columns} == target_columns for c in constraints
    ), f"No composite unique constraint on {target_columns}; got {[(type(c).__name__, [col.name for col in c.columns]) for c in constraints]}"


def test_create_tables_runs_against_fresh_sqlite_in_memory():
    """The schema must materialize cleanly on a fresh SQLite database."""
    engine = create_engine("sqlite:///:memory:")
    create_tables(engine)
    inspector = inspect(engine)

    trade_cols = {c["name"] for c in inspector.get_columns("trades")}
    assert {"venue", "instrument_name"}.issubset(trade_cols)

    pos_cols = {c["name"] for c in inspector.get_columns("positions")}
    assert {"venue", "instrument_name"}.issubset(pos_cols)
