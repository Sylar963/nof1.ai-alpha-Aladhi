from decimal import Decimal

import pytest

from src.database.db_manager import DatabaseManager
from src.database.models import OptionsReasoningEntry


@pytest.fixture
def db():
    return DatabaseManager(db_url="sqlite:///:memory:")


def test_save_reasoning_creates_row(db):
    entry_id = db.save_options_reasoning(
        triggered_by_events=[{"type": "manual", "fired_at": "2026-05-20T00:00:00Z",
                              "description": "manual cycle", "structure_id": None}],
        context_snapshot={"spot": 100000.0, "vol_regime": "fair"},
        llm_reasoning="No actionable mispricings.",
        llm_decisions=[],
    )
    with db.session_scope() as session:
        row = session.query(OptionsReasoningEntry).filter_by(id=entry_id).one()
        assert row.llm_reasoning == "No actionable mispricings."
        assert row.outcome is None
        assert row.created_at is not None


def test_save_reasoning_returns_id(db):
    entry_id = db.save_options_reasoning(
        triggered_by_events=[],
        context_snapshot={"spot": 100000.0},
        llm_reasoning="ok",
        llm_decisions=[{"action": "hold"}],
    )
    assert isinstance(entry_id, int)
    assert entry_id > 0


def test_get_recent_options_reasoning_returns_latest_first(db):
    id1 = db.save_options_reasoning(
        triggered_by_events=[], context_snapshot={"cycle": 1},
        llm_reasoning="cycle 1", llm_decisions=[],
    )
    id2 = db.save_options_reasoning(
        triggered_by_events=[], context_snapshot={"cycle": 2},
        llm_reasoning="cycle 2", llm_decisions=[],
    )
    rows = db.get_recent_options_reasoning(limit=10)
    assert len(rows) == 2
    assert rows[0]["id"] == id2
    assert rows[1]["id"] == id1
    assert rows[0]["llm_reasoning"] == "cycle 2"
    assert rows[0]["context_snapshot"] == {"cycle": 2}
    assert rows[0]["triggered_by_events"] == []
    assert rows[0]["llm_decisions"] == []


def test_get_recent_options_reasoning_respects_limit(db):
    for i in range(5):
        db.save_options_reasoning(
            triggered_by_events=[], context_snapshot={"i": i},
            llm_reasoning=f"r{i}", llm_decisions=[],
        )
    rows = db.get_recent_options_reasoning(limit=3)
    assert len(rows) == 3


def test_update_reasoning_outcome(db):
    entry_id = db.save_options_reasoning(
        triggered_by_events=[],
        context_snapshot={"spot": 100000.0},
        llm_reasoning="propose credit put spread",
        llm_decisions=[{"strategy": "credit_put_spread"}],
    )
    db.update_reasoning_outcome(entry_id, outcome={"executed": True, "order_id": "ord-123"})
    rows = db.get_recent_options_reasoning(limit=1)
    assert rows[0]["outcome"] == {"executed": True, "order_id": "ord-123"}


def test_update_reasoning_outcome_missing_id_is_noop(db):
    db.update_reasoning_outcome(99999, outcome={"executed": False})
    rows = db.get_recent_options_reasoning(limit=10)
    assert rows == []
