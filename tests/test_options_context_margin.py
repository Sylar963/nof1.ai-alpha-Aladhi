"""Tests for the Hyperliquid hedge-budget signal injected into OptionsContext.

These are the options-side equivalent of test_buying_power_snapshot. They
verify the options LLM now sees HL free margin + the derived
``max_hedge_notional`` — without which delta-hedged proposals would keep
tripping the pre-trade guard in silence.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.options_intel.snapshot import OptionsContext


def test_options_context_fields_default_zero():
    ctx = OptionsContext(
        timestamp_utc="t",
        spot=50000.0,
        spot_24h_change_pct=0.0,
        opening_range={},
        keltner={},
        atm_iv_by_tenor={},
        skew_25d_by_tenor={},
        term_structure_slope=0.0,
        expected_move_pct_by_tenor={},
        vol_regime="neutral",
        vol_regime_confidence="low",
        realized_iv_ratio_30d=1.0,
        straddle_test_30d={},
    )
    assert ctx.hyperliquid_free_margin == 0.0
    assert ctx.hyperliquid_max_leverage == 1
    assert ctx.max_hedge_notional == 0.0
    assert ctx.recent_options_skips == []


def test_options_context_serializes_new_fields():
    ctx = OptionsContext(
        timestamp_utc="t",
        spot=50000.0,
        spot_24h_change_pct=0.0,
        opening_range={},
        keltner={},
        atm_iv_by_tenor={},
        skew_25d_by_tenor={},
        term_structure_slope=0.0,
        expected_move_pct_by_tenor={},
        vol_regime="rich",
        vol_regime_confidence="high",
        realized_iv_ratio_30d=0.7,
        straddle_test_30d={},
        hyperliquid_free_margin=1234.56,
        hyperliquid_max_leverage=40,
        max_hedge_notional=47000.0,
        recent_options_skips=[{"action": "options_proposal_skipped_insufficient_hedge_margin"}],
    )
    d = ctx.to_dict()
    assert d["hyperliquid_free_margin"] == 1234.56
    assert d["hyperliquid_max_leverage"] == 40
    assert d["max_hedge_notional"] == 47000.0
    assert len(d["recent_options_skips"]) == 1


# --- builder integration -----------------------------------------------------


class _FakeThalex:
    """Minimal ThalexAPI stand-in — only covers what build_options_context uses."""
    _instruments_cache = []

    async def get_user_state(self):
        return {"balance": 500.0, "positions": []}

    async def get_greeks(self, *args, **kwargs):
        return {}


class _FakeDeribit:
    async def get_index_price(self, *args, **kwargs):
        return 50000.0

    async def get_book_summary_by_currency(self, *args, **kwargs):
        return []


class _FakeIVStore:
    def write(self, *args, **kwargs):
        pass

    def read_between(self, *args, **kwargs):
        return []

    def lookback(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_builder_populates_hedge_budget_from_hyperliquid():
    from src.backend.options_intel.builder import build_options_context

    hl = MagicMock()
    hl.get_free_margin_info = AsyncMock(return_value={
        "withdrawable": 1000.0, "free_margin": 1200.0, "account_value": 1200.0, "total_margin_used": 0.0,
    })
    hl.get_max_leverage = AsyncMock(return_value=40)

    ctx = await build_options_context(
        thalex=_FakeThalex(),
        deribit=_FakeDeribit(),
        iv_history=_FakeIVStore(),
        spot_history=[50000.0] * 20,
        persist_anchor=False,
        hyperliquid=hl,
        recent_options_skips=[{"action": "options_execution_failed", "reason": "test"}],
    )
    assert ctx.hyperliquid_free_margin == 1000.0  # conservative min(withdrawable, free_margin)
    assert ctx.hyperliquid_max_leverage == 40
    # Same math as perps buying_power: available × leverage / 1.05
    expected_cap = round((1000.0 * 40) / 1.05, 2)
    assert ctx.max_hedge_notional == expected_cap
    assert len(ctx.recent_options_skips) == 1


@pytest.mark.asyncio
async def test_builder_zeros_hedge_budget_when_hyperliquid_missing():
    """No HL adapter = fail-closed zeros, not a missing field."""
    from src.backend.options_intel.builder import build_options_context

    ctx = await build_options_context(
        thalex=_FakeThalex(),
        deribit=_FakeDeribit(),
        iv_history=_FakeIVStore(),
        spot_history=[50000.0] * 20,
        persist_anchor=False,
        hyperliquid=None,
    )
    assert ctx.hyperliquid_free_margin == 0.0
    assert ctx.max_hedge_notional == 0.0


@pytest.mark.asyncio
async def test_builder_survives_hyperliquid_rpc_failure():
    from src.backend.options_intel.builder import build_options_context

    hl = MagicMock()
    hl.get_free_margin_info = AsyncMock(side_effect=ConnectionError("rpc down"))
    hl.get_max_leverage = AsyncMock(return_value=40)

    ctx = await build_options_context(
        thalex=_FakeThalex(),
        deribit=_FakeDeribit(),
        iv_history=_FakeIVStore(),
        spot_history=[50000.0] * 20,
        persist_anchor=False,
        hyperliquid=hl,
    )
    # Free margin lookup failed → zeros, cap stays at 0 so LLM sees "no hedge budget".
    assert ctx.hyperliquid_free_margin == 0.0
    assert ctx.max_hedge_notional == 0.0
