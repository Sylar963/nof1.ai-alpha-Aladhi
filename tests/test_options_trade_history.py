"""Tests for the recent-options-trades query that feeds the OptionsContext.

Returns the last N closed Thalex options trades from the existing Trade
table so the LLM can see what it most recently traded and how it fared."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backend.options_intel.trade_history import (
    fetch_recent_options_trades,
)
from src.database.models import Base, Trade


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def _trade(session, *, asset, venue, status, exit_at, pnl, instrument_name=None, strategy_rationale=""):
    trade = Trade(
        asset=asset,
        action="buy",
        entry_timestamp=exit_at - timedelta(days=2),
        entry_price=60000.0,
        entry_size=0.05,
        entry_value=3000.0,
        exit_timestamp=exit_at,
        exit_price=61000.0,
        exit_value=3050.0,
        realized_pnl=pnl,
        realized_pnl_pct=pnl / 3000.0 if pnl is not None else None,
        venue=venue,
        instrument_name=instrument_name,
        status=status,
        rationale=strategy_rationale,
    )
    session.add(trade)
    session.commit()
    return trade


# ---------------------------------------------------------------------------
# Empty / no Thalex history
# ---------------------------------------------------------------------------


def test_returns_empty_list_when_no_trades(session):
    assert fetch_recent_options_trades(session, limit=5) == []


def test_filters_out_hyperliquid_trades(session):
    """Only venue='thalex' trades should appear in the options-trade history."""
    now = datetime.now(timezone.utc)
    _trade(session, asset="BTC", venue="hyperliquid", status="closed", exit_at=now, pnl=100.0)
    assert fetch_recent_options_trades(session, limit=5) == []


def test_filters_out_open_trades(session):
    """Open positions are not 'recent' history — they show up in open_positions."""
    now = datetime.now(timezone.utc)
    _trade(
        session,
        asset="BTC",
        venue="thalex",
        status="open",
        exit_at=now,
        pnl=None,
        instrument_name="BTC-25APR26-65000-C",
    )
    assert fetch_recent_options_trades(session, limit=5) == []


# ---------------------------------------------------------------------------
# Returns recent closed Thalex trades
# ---------------------------------------------------------------------------


def test_returns_closed_thalex_trades_newest_first(session):
    now = datetime.now(timezone.utc)
    _trade(
        session,
        asset="BTC",
        venue="thalex",
        status="closed",
        exit_at=now - timedelta(days=3),
        pnl=120.0,
        instrument_name="BTC-25APR26-65000-C",
        strategy_rationale="vol cheap, gamma scalp",
    )
    _trade(
        session,
        asset="BTC",
        venue="thalex",
        status="closed",
        exit_at=now - timedelta(days=1),
        pnl=-50.0,
        instrument_name="BTC-25APR26-55000-P",
        strategy_rationale="iron condor",
    )
    _trade(
        session,
        asset="BTC",
        venue="thalex",
        status="closed",
        exit_at=now,
        pnl=200.0,
        instrument_name="BTC-10MAY26-65000-C",
        strategy_rationale="long_call_delta_hedged",
    )

    history = fetch_recent_options_trades(session, limit=5)

    assert len(history) == 3
    # Newest first
    assert history[0]["instrument_name"] == "BTC-10MAY26-65000-C"
    assert history[0]["pnl_usd"] == 200.0
    assert history[1]["instrument_name"] == "BTC-25APR26-55000-P"
    assert history[2]["instrument_name"] == "BTC-25APR26-65000-C"


def test_respects_limit(session):
    now = datetime.now(timezone.utc)
    for i in range(10):
        _trade(
            session,
            asset="BTC",
            venue="thalex",
            status="closed",
            exit_at=now - timedelta(hours=i),
            pnl=10.0 * i,
            instrument_name=f"BTC-25APR26-{60000 + i * 1000}-C",
        )
    history = fetch_recent_options_trades(session, limit=3)
    assert len(history) == 3


def test_history_entry_shape(session):
    now = datetime.now(timezone.utc)
    _trade(
        session,
        asset="BTC",
        venue="thalex",
        status="closed",
        exit_at=now,
        pnl=120.0,
        instrument_name="BTC-25APR26-65000-C",
        strategy_rationale="long_call_delta_hedged: gamma scalp",
    )
    entry = fetch_recent_options_trades(session, limit=1)[0]
    assert entry["instrument_name"] == "BTC-25APR26-65000-C"
    assert entry["pnl_usd"] == 120.0
    assert "rationale" in entry
    assert "closed_at" in entry
    assert isinstance(entry["closed_at"], str)  # serialized for JSON
