"""Tests for the ``buying_power`` snapshot fed to the perps LLM.

The snapshot is the feedback signal that tells the agent "you can/can't
afford this trade" — without it the LLM had no margin figures in its
context and would propose trades the account couldn't cover.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.bot_engine import TradingBotEngine


def _make_engine(assets):
    engine = object.__new__(TradingBotEngine)
    engine.logger = logging.getLogger("test_buying_power")
    engine.assets = list(assets)
    engine._max_leverage_cache = {}
    engine.hyperliquid = MagicMock()
    return engine


@pytest.mark.asyncio
async def test_snapshot_returns_zero_caps_when_account_empty():
    engine = _make_engine(["BTC", "ETH"])
    engine.hyperliquid.get_free_margin_info = AsyncMock(return_value={
        "withdrawable": 0.0,
        "free_margin": 0.0,
        "account_value": 0.0,
        "total_margin_used": 0.0,
    })
    engine.hyperliquid.get_max_leverage = AsyncMock(side_effect=lambda asset: 40 if asset == "BTC" else 25)

    snap = await engine._get_buying_power_snapshot()

    assert snap["withdrawable"] == 0.0
    assert snap["free_margin"] == 0.0
    # With $0 available, every asset's new-notional cap must be 0 — this is
    # what signals the LLM to output hold.
    assert snap["max_new_notional_by_asset"] == {"BTC": 0.0, "ETH": 0.0}
    assert snap["max_leverage_by_asset"] == {"BTC": 40, "ETH": 25}


@pytest.mark.asyncio
async def test_snapshot_computes_cap_with_5pct_buffer():
    engine = _make_engine(["BTC"])
    engine.hyperliquid.get_free_margin_info = AsyncMock(return_value={
        "withdrawable": 1000.0,
        "free_margin": 1200.0,
        "account_value": 1200.0,
        "total_margin_used": 0.0,
    })
    engine.hyperliquid.get_max_leverage = AsyncMock(return_value=40)

    snap = await engine._get_buying_power_snapshot()

    # Conservative available = min(1000, 1200) = 1000
    # Cap = 1000 * 40 / 1.05 ≈ 38095.24 (matches preflight math exactly)
    expected = round((1000.0 * 40) / 1.05, 2)
    assert snap["max_new_notional_by_asset"]["BTC"] == expected
    assert snap["withdrawable"] == 1000.0
    assert snap["free_margin"] == 1200.0


@pytest.mark.asyncio
async def test_snapshot_handles_lookup_failure_conservatively():
    engine = _make_engine(["BTC"])
    engine.hyperliquid.get_free_margin_info = AsyncMock(side_effect=ConnectionError("rpc down"))
    engine.hyperliquid.get_max_leverage = AsyncMock(return_value=40)

    snap = await engine._get_buying_power_snapshot()

    # Failure ≠ missing field. Returning zeros forces the LLM to hold rather
    # than propose blind.
    assert snap["withdrawable"] == 0.0
    assert snap["free_margin"] == 0.0
    assert snap["max_new_notional_by_asset"]["BTC"] == 0.0


@pytest.mark.asyncio
async def test_max_leverage_is_cached_across_calls():
    engine = _make_engine(["BTC", "ETH", "SOL"])
    engine.hyperliquid.get_free_margin_info = AsyncMock(return_value={
        "withdrawable": 500.0, "free_margin": 500.0, "account_value": 500.0, "total_margin_used": 0.0,
    })
    engine.hyperliquid.get_max_leverage = AsyncMock(side_effect=lambda asset: {"BTC": 40, "ETH": 25, "SOL": 20}[asset])

    await engine._get_buying_power_snapshot()
    await engine._get_buying_power_snapshot()

    # 3 assets x 2 calls = 6 potential hits; the cache must collapse this to 3.
    assert engine.hyperliquid.get_max_leverage.call_count == 3


@pytest.mark.asyncio
async def test_get_max_leverage_cached_defaults_to_1_on_error():
    engine = _make_engine(["BTC"])
    engine.hyperliquid.get_max_leverage = AsyncMock(side_effect=RuntimeError("meta hiccup"))

    lev = await engine._get_max_leverage_cached("BTC")
    assert lev == 1
    # Negative caching: second call must not retry the failing RPC.
    lev_again = await engine._get_max_leverage_cached("BTC")
    assert lev_again == 1
    assert engine.hyperliquid.get_max_leverage.call_count == 1
