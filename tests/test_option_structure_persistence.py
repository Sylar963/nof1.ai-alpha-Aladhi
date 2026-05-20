from decimal import Decimal

import pytest

from src.database.db_manager import DatabaseManager
from src.database.models import OptionStructureSnapshot


@pytest.fixture
def db():
    return DatabaseManager(db_url="sqlite:///:memory:")


def test_upsert_creates_new_structure(db):
    db.upsert_structure_snapshot(
        structure_id="abc123",
        underlying="BTC",
        kind="credit_put_spread",
        legs_json=[{"instrument_name": "BTC-27JUN26-100000-P", "side": "short"}],
        entry_net_premium=Decimal("20"),
        last_pnl_abs=Decimal("0"),
        last_pnl_pct=Decimal("0"),
        last_breach_state="nominal",
    )
    with db.session_scope() as session:
        row = session.query(OptionStructureSnapshot).filter_by(structure_id="abc123").one()
        assert row.underlying == "BTC"
        assert row.kind == "credit_put_spread"
        assert row.closed_at is None
        assert row.opened_at is not None
        assert row.last_seen_at is not None


def test_upsert_existing_structure_updates_fields_but_not_opened_at(db):
    db.upsert_structure_snapshot(
        structure_id="abc123",
        underlying="BTC",
        kind="credit_put_spread",
        legs_json=[],
        entry_net_premium=Decimal("20"),
        last_pnl_abs=Decimal("0"),
        last_pnl_pct=Decimal("0"),
        last_breach_state="nominal",
    )
    with db.session_scope() as session:
        first_opened = session.query(OptionStructureSnapshot).filter_by(structure_id="abc123").one().opened_at

    db.upsert_structure_snapshot(
        structure_id="abc123",
        underlying="BTC",
        kind="credit_put_spread",
        legs_json=[],
        entry_net_premium=Decimal("20"),
        last_pnl_abs=Decimal("5"),
        last_pnl_pct=Decimal("0.25"),
        last_breach_state="warning",
    )
    with db.session_scope() as session:
        row = session.query(OptionStructureSnapshot).filter_by(structure_id="abc123").one()
        assert row.opened_at == first_opened
        assert row.last_pnl_abs == 5.0
        assert row.last_breach_state == "warning"


def test_mark_structure_closed_sets_closed_at(db):
    db.upsert_structure_snapshot(
        structure_id="abc123",
        underlying="BTC",
        kind="credit_put_spread",
        legs_json=[],
        entry_net_premium=Decimal("20"),
        last_pnl_abs=Decimal("0"),
        last_pnl_pct=Decimal("0"),
        last_breach_state="nominal",
    )
    db.mark_structure_closed("abc123")
    with db.session_scope() as session:
        row = session.query(OptionStructureSnapshot).filter_by(structure_id="abc123").one()
        assert row.closed_at is not None


def test_get_open_structures_excludes_closed(db):
    db.upsert_structure_snapshot(
        structure_id="open_1", underlying="BTC", kind="credit_put_spread",
        legs_json=[], entry_net_premium=Decimal("10"),
        last_pnl_abs=Decimal("0"), last_pnl_pct=Decimal("0"), last_breach_state="nominal",
    )
    db.upsert_structure_snapshot(
        structure_id="closed_1", underlying="BTC", kind="iron_condor",
        legs_json=[], entry_net_premium=Decimal("30"),
        last_pnl_abs=Decimal("0"), last_pnl_pct=Decimal("0"), last_breach_state="nominal",
    )
    db.mark_structure_closed("closed_1")
    open_ids = {row["structure_id"] for row in db.get_open_structures()}
    assert open_ids == {"open_1"}
