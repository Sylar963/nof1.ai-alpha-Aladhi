"""Tests for TradeProposal retry state transitions."""

from datetime import datetime, UTC

from src.backend.models.trade_proposal import TradeProposal


def _make_failed_proposal(error: str = "Insufficient margin") -> TradeProposal:
    p = TradeProposal(asset="BTC", action="buy", entry_price=50000.0, size=0.1)
    p.approve()
    p.mark_failed(error)
    return p


def test_reset_for_retry_transitions_failed_to_pending():
    p = _make_failed_proposal()
    assert p.status == "failed"
    assert p.is_retryable is True
    assert p.executed_at is not None

    assert p.reset_for_retry() is True
    assert p.status == "pending"
    assert p.is_pending is True
    assert p.is_retryable is False
    assert p.executed_at is None
    # Prior error preserved for UI display until retry produces a new outcome
    assert p.execution_error == "Insufficient margin"


def test_reset_for_retry_noop_when_pending():
    p = TradeProposal(asset="BTC", action="buy", entry_price=50000.0, size=0.1)
    assert p.status == "pending"
    assert p.reset_for_retry() is False
    assert p.status == "pending"


def test_reset_for_retry_noop_when_executed():
    p = TradeProposal(asset="BTC", action="buy", entry_price=50000.0, size=0.1)
    p.approve()
    p.mark_executed(50010.0)
    assert p.reset_for_retry() is False
    assert p.status == "executed"


def test_reset_for_retry_noop_when_rejected():
    p = TradeProposal(asset="BTC", action="buy", entry_price=50000.0, size=0.1)
    p.reject("not interested")
    assert p.reset_for_retry() is False
    assert p.status == "rejected"


def test_is_visible_to_ui_covers_pending_and_failed_only():
    p_pending = TradeProposal(asset="BTC", action="buy", entry_price=50000.0, size=0.1)
    p_failed = _make_failed_proposal()
    p_rejected = TradeProposal(asset="ETH", action="sell", entry_price=3000.0, size=1.0)
    p_rejected.reject()
    p_executed = TradeProposal(asset="SOL", action="buy", entry_price=100.0, size=5.0)
    p_executed.approve()
    p_executed.mark_executed(101.0)

    assert p_pending.is_visible_to_ui is True
    assert p_failed.is_visible_to_ui is True
    assert p_rejected.is_visible_to_ui is False
    assert p_executed.is_visible_to_ui is False


def test_double_retry_click_is_guarded():
    """Simulate two rapid Retry clicks: second must be a no-op."""
    p = _make_failed_proposal()
    assert p.reset_for_retry() is True
    # While retry is in flight, status is "pending" — a second click must fail.
    assert p.reset_for_retry() is False
