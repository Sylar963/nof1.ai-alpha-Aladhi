from src.backend.bot_engine import (
    persist_options_reasoning,
    update_reasoning_outcome_safely,
)
from src.database.db_manager import DatabaseManager


def test_persist_options_reasoning_writes_row():
    db = DatabaseManager(db_url="sqlite:///:memory:")
    entry_id = persist_options_reasoning(
        db,
        triggered_by_events=[],
        context_snapshot={"spot": 100000.0},
        llm_reasoning="hold",
        llm_decisions=[],
    )
    assert isinstance(entry_id, int)
    rows = db.get_recent_options_reasoning(limit=1)
    assert len(rows) == 1
    assert rows[0]["id"] == entry_id


def test_update_reasoning_outcome_safely_handles_none_id():
    db = DatabaseManager(db_url="sqlite:///:memory:")
    update_reasoning_outcome_safely(db, entry_id=None, outcome={"executed": False})


def test_update_reasoning_outcome_safely_writes_when_id_present():
    db = DatabaseManager(db_url="sqlite:///:memory:")
    entry_id = persist_options_reasoning(
        db, triggered_by_events=[], context_snapshot={},
        llm_reasoning="r", llm_decisions=[],
    )
    update_reasoning_outcome_safely(db, entry_id=entry_id, outcome={"executed": True})
    rows = db.get_recent_options_reasoning(limit=1)
    assert rows[0]["outcome"] == {"executed": True}
