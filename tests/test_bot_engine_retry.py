"""Tests for retry / dismiss / TTL pruning of failed proposals in TradingBotEngine.

Builds a minimal engine instance via ``object.__new__`` so these tests don't
need real exchange/LLM clients — they only exercise the proposal bookkeeping.
"""

import asyncio
import logging
from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock

import pytest

from src.backend.bot_engine import BotState, TradingBotEngine
from src.backend.models.trade_proposal import TradeProposal


def _make_engine() -> TradingBotEngine:
    engine = object.__new__(TradingBotEngine)
    engine.logger = logging.getLogger("test_bot_engine_retry")
    engine.pending_proposals = []
    engine.state = BotState()
    engine._notify_state_update = MagicMock()
    return engine


def _failed_proposal(error: str = "Insufficient margin") -> TradeProposal:
    p = TradeProposal(asset="BTC", action="buy", entry_price=50000.0, size=0.1)
    p.approve()
    p.mark_failed(error)
    return p


def test_retry_proposal_resets_status_and_schedules_execution(monkeypatch):
    engine = _make_engine()
    failed = _failed_proposal()
    engine.pending_proposals.append(failed)

    scheduled = []

    def fake_create_task(coro):
        # Close the coroutine so pytest doesn't warn about un-awaited coroutines.
        coro.close()
        scheduled.append(True)
        return MagicMock()

    monkeypatch.setattr("src.backend.bot_engine.asyncio.create_task", fake_create_task)
    engine._execute_proposal = MagicMock(return_value=asyncio.sleep(0))

    assert engine.retry_proposal(failed.id) is True
    assert failed.status == "pending"
    assert failed.executed_at is None
    assert scheduled == [True]
    # State was reflected to UI
    assert len(engine.state.pending_proposals) == 1
    engine._notify_state_update.assert_called_once()


def test_retry_proposal_returns_false_when_not_failed():
    engine = _make_engine()
    pending = TradeProposal(asset="BTC", action="buy", entry_price=50000.0, size=0.1)
    engine.pending_proposals.append(pending)

    assert engine.retry_proposal(pending.id) is False
    assert pending.status == "pending"
    engine._notify_state_update.assert_not_called()


def test_retry_proposal_returns_false_when_unknown_id():
    engine = _make_engine()
    assert engine.retry_proposal("nope") is False
    engine._notify_state_update.assert_not_called()


def test_dismiss_proposal_removes_failed_entry():
    engine = _make_engine()
    failed = _failed_proposal()
    engine.pending_proposals.append(failed)

    assert engine.dismiss_proposal(failed.id) is True
    assert engine.pending_proposals == []
    assert engine.state.pending_proposals == []
    engine._notify_state_update.assert_called_once()


def test_dismiss_proposal_returns_false_when_not_failed():
    engine = _make_engine()
    pending = TradeProposal(asset="BTC", action="buy", entry_price=50000.0, size=0.1)
    engine.pending_proposals.append(pending)

    assert engine.dismiss_proposal(pending.id) is False
    assert pending in engine.pending_proposals


def test_prune_stale_failed_proposals_drops_old_keeps_fresh_and_pending(monkeypatch):
    # Shorten TTL for the test
    monkeypatch.setattr("src.backend.bot_engine.FAILED_PROPOSAL_TTL_SECONDS", 60)

    engine = _make_engine()
    stale_failed = _failed_proposal("old failure")
    stale_failed.executed_at = datetime.now(UTC) - timedelta(seconds=120)

    fresh_failed = _failed_proposal("recent failure")
    # executed_at set to "now" by mark_failed

    pending = TradeProposal(asset="ETH", action="sell", entry_price=3000.0, size=1.0)

    engine.pending_proposals = [stale_failed, fresh_failed, pending]
    engine._prune_stale_failed_proposals()

    ids = {p.id for p in engine.pending_proposals}
    assert stale_failed.id not in ids
    assert fresh_failed.id in ids
    assert pending.id in ids


def test_sync_pending_proposals_state_emits_only_visible(monkeypatch):
    monkeypatch.setattr("src.backend.bot_engine.FAILED_PROPOSAL_TTL_SECONDS", 0)  # no TTL pruning

    engine = _make_engine()
    failed = _failed_proposal()
    pending = TradeProposal(asset="ETH", action="sell", entry_price=3000.0, size=1.0)
    executed = TradeProposal(asset="SOL", action="buy", entry_price=100.0, size=5.0)
    executed.approve()
    executed.mark_executed(101.0)

    engine.pending_proposals = [failed, pending, executed]
    engine._sync_pending_proposals_state()

    statuses = sorted(p["status"] for p in engine.state.pending_proposals)
    assert statuses == ["failed", "pending"]
