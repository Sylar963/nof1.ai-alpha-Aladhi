"""Tests for the circuit breaker + peak tracking + kill switch flatten_all."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.bot_engine import BotState, TradingBotEngine


def _make_engine(assets=("BTC",)):
    engine = object.__new__(TradingBotEngine)
    engine.logger = logging.getLogger("test_circuit_breaker")
    engine.assets = list(assets)
    engine.state = BotState()
    engine.pending_proposals = []
    engine._max_leverage_cache = {}
    engine.is_paused = False
    engine.pause_reason = None
    engine.peak_account_value = 0.0
    engine._execution_failure_streak = 0
    engine._notify_state_update = MagicMock()
    engine._write_diary_entry = MagicMock()
    engine.hyperliquid = MagicMock()
    engine.thalex = None
    engine.close_position = AsyncMock(return_value=True)
    return engine


# --- streak tracking ---------------------------------------------------------


def test_streak_increments_and_trips_at_threshold(monkeypatch):
    monkeypatch.setattr("src.backend.bot_engine.CIRCUIT_BREAKER_CONSECUTIVE_FAILS", 3)
    engine = _make_engine()

    engine._record_execution_failure("err1")
    engine._record_execution_failure("err2")
    assert engine.is_paused is False
    assert engine._execution_failure_streak == 2

    engine._record_execution_failure("err3")
    assert engine.is_paused is True
    assert "3 consecutive" in (engine.pause_reason or "")
    # State broadcast once on trip
    assert engine._notify_state_update.called


def test_success_resets_streak():
    engine = _make_engine()
    engine._record_execution_failure("err1")
    engine._record_execution_failure("err2")
    assert engine._execution_failure_streak == 2

    engine._record_execution_success()
    assert engine._execution_failure_streak == 0
    assert engine.state.execution_failure_streak == 0
    assert engine.is_paused is False


def test_streak_no_op_when_threshold_zero(monkeypatch):
    """A 0 threshold disables the streak-based trip (opt-out config)."""
    monkeypatch.setattr("src.backend.bot_engine.CIRCUIT_BREAKER_CONSECUTIVE_FAILS", 0)
    engine = _make_engine()
    for _ in range(20):
        engine._record_execution_failure("err")
    assert engine.is_paused is False


# --- drawdown ----------------------------------------------------------------


def test_peak_seeds_on_first_reading():
    engine = _make_engine()
    engine._update_peak_and_check_drawdown(10_000.0)
    assert engine.peak_account_value == 10_000.0
    assert engine.state.drawdown_pct == 0.0
    assert engine.is_paused is False


def test_peak_advances_on_gains():
    engine = _make_engine()
    engine._update_peak_and_check_drawdown(10_000.0)
    engine._update_peak_and_check_drawdown(12_000.0)
    assert engine.peak_account_value == 12_000.0


def test_drawdown_trips_circuit_breaker(monkeypatch):
    monkeypatch.setattr("src.backend.bot_engine.CIRCUIT_BREAKER_DRAWDOWN_PCT", 5.0)
    engine = _make_engine()
    engine._update_peak_and_check_drawdown(10_000.0)  # seed
    engine._update_peak_and_check_drawdown(9_400.0)   # -6% from peak
    assert engine.is_paused is True
    assert "drawdown" in (engine.pause_reason or "").lower()


def test_drawdown_shallow_does_not_trip(monkeypatch):
    monkeypatch.setattr("src.backend.bot_engine.CIRCUIT_BREAKER_DRAWDOWN_PCT", 5.0)
    engine = _make_engine()
    engine._update_peak_and_check_drawdown(10_000.0)
    engine._update_peak_and_check_drawdown(9_700.0)   # -3%, under 5% limit
    assert engine.is_paused is False


# --- resume ------------------------------------------------------------------


def test_resume_clears_pause_and_streak():
    engine = _make_engine()
    engine._trip_circuit_breaker("test reason")
    engine._execution_failure_streak = 5
    engine.state.execution_failure_streak = 5

    resumed = engine.resume_trading()
    assert resumed is True
    assert engine.is_paused is False
    assert engine.pause_reason is None
    assert engine._execution_failure_streak == 0
    assert engine.state.execution_failure_streak == 0


def test_resume_noop_when_not_paused():
    engine = _make_engine()
    assert engine.resume_trading() is False


# --- flatten_all -------------------------------------------------------------


@pytest.mark.asyncio
async def test_flatten_all_happy_path():
    engine = _make_engine(assets=["BTC", "ETH"])
    engine.hyperliquid.cancel_all_orders = AsyncMock(return_value={"cancelled_count": 2})
    engine.state.positions = [
        {"venue": "hyperliquid", "symbol": "BTC", "quantity": 0.5},
        {"venue": "hyperliquid", "symbol": "ETH", "quantity": -1.0},
    ]

    result = await engine.flatten_all()

    # Both assets cancelled
    assert result["orders_cancelled"]["hyperliquid"] == {"BTC": 2, "ETH": 2}
    # Both positions closed
    assert {p["asset"] for p in result["positions_closed"]} == {"BTC", "ETH"}
    assert result["errors"] == []
    assert result["thalex_positions_remaining"] == []
    # CB tripped
    assert engine.is_paused is True
    assert "kill switch" in (engine.pause_reason or "").lower()


@pytest.mark.asyncio
async def test_flatten_all_reports_thalex_remaining():
    engine = _make_engine(assets=["BTC"])
    engine.hyperliquid.cancel_all_orders = AsyncMock(return_value={"cancelled_count": 0})
    engine.thalex = MagicMock()
    engine.thalex.cancel_all_orders = AsyncMock(return_value=None)
    engine.state.positions = [
        {"venue": "thalex", "instrument_name": "BTC-26APR26-50000-C", "quantity": 0.1},
    ]

    result = await engine.flatten_all()

    assert result["orders_cancelled"]["thalex"] is True
    assert len(result["thalex_positions_remaining"]) == 1
    assert result["thalex_positions_remaining"][0]["instrument_name"] == "BTC-26APR26-50000-C"
    assert engine.is_paused is True
    assert "Thalex" in (engine.pause_reason or "")


@pytest.mark.asyncio
async def test_flatten_all_partial_failure_still_pauses():
    """One venue failing must not abort the rest or leave the loop running."""
    engine = _make_engine(assets=["BTC"])
    engine.hyperliquid.cancel_all_orders = AsyncMock(side_effect=ConnectionError("rpc down"))
    engine.state.positions = []

    result = await engine.flatten_all()

    assert any("hl_cancel_BTC" in err for err in result["errors"])
    # Still paused — a broken kill switch is worse than a loud one.
    assert engine.is_paused is True


# --- _read_recent_options_skips ---------------------------------------------


def test_read_recent_options_skips_filters_correct_actions(tmp_path):
    engine = _make_engine()
    diary_path = tmp_path / "diary.jsonl"
    entries = [
        {"timestamp": "t1", "action": "buy", "asset": "BTC"},
        {"timestamp": "t2", "action": "options_proposal_skipped_insufficient_hedge_margin",
         "strategy": "long_call_delta_hedged", "reason": "need $500 have $0"},
        {"timestamp": "t3", "action": "manual_close", "asset": "BTC"},
        {"timestamp": "t4", "action": "options_execution_failed",
         "venue": "thalex", "reason": "leg rejected"},
    ]
    import json
    with open(diary_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    engine.diary_path = diary_path

    skips = engine._read_recent_options_skips(limit=5)
    assert len(skips) == 2
    assert skips[0]["action"] == "options_proposal_skipped_insufficient_hedge_margin"
    assert skips[1]["action"] == "options_execution_failed"
    # Only the whitelisted keys come through.
    assert set(skips[0].keys()).issubset({"timestamp", "action", "asset", "reason", "strategy"})
