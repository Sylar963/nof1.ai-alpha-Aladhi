from src.gui.services.bot_service import get_options_reasoning_history


def test_get_options_reasoning_history_returns_list():
    rows = get_options_reasoning_history(limit=5)
    assert isinstance(rows, list)


def test_get_options_reasoning_history_handles_db_unavailable(monkeypatch):
    def boom():
        raise RuntimeError("db is down")

    monkeypatch.setattr(
        "src.gui.services.bot_service._get_db_manager_for_reasoning",
        boom,
    )
    rows = get_options_reasoning_history(limit=5)
    assert rows == []
