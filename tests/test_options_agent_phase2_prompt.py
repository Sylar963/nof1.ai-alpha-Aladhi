from src.backend.agent.options_agent import _OPTIONS_SYSTEM_PROMPT


def test_prompt_mentions_debit_put_spread():
    assert "debit_put_spread" in _OPTIONS_SYSTEM_PROMPT


def test_prompt_mentions_debit_call_spread():
    assert "debit_call_spread" in _OPTIONS_SYSTEM_PROMPT


def test_prompt_mentions_iron_butterfly():
    assert "iron_butterfly" in _OPTIONS_SYSTEM_PROMPT


def test_prompt_mentions_long_straddle():
    assert "long_straddle" in _OPTIONS_SYSTEM_PROMPT


def test_prompt_mentions_structures_array():
    """The position management section should reference the structures array."""
    assert "structures" in _OPTIONS_SYSTEM_PROMPT.lower()


def test_prompt_mentions_breach_states():
    lower = _OPTIONS_SYSTEM_PROMPT.lower()
    assert "breach" in lower
    assert "warning" in lower


def test_prompt_mentions_days_open():
    lower = _OPTIONS_SYSTEM_PROMPT.lower()
    assert "days_open" in lower or "days open" in lower


def test_prompt_mentions_pnl_pct():
    """Position management should reference pnl_pct as the profit/loss metric."""
    assert "pnl_pct" in _OPTIONS_SYSTEM_PROMPT.lower() or "pnl pct" in _OPTIONS_SYSTEM_PROMPT.lower()
