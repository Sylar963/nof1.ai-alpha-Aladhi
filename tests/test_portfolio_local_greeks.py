"""When Thalex ticker only publishes delta, the aggregator must backfill
gamma/vega/theta from a local Black-Scholes calc using spot + strike + iv.

Without this, portfolio_greeks silently under-reports vega/gamma/theta
to zero and the LLM sees an artificially 'flat' book.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.backend.options_intel.portfolio import aggregate_portfolio_greeks


class _PartialGreeksAdapter:
    """Simulates Thalex: returns only delta + mark_iv, no gamma/vega/theta."""

    async def get_greeks(self, instrument_name: str) -> dict:
        return {"delta": 0.52, "mark_iv": 0.62}


class _EmptyGreeksAdapter:
    async def get_greeks(self, instrument_name: str) -> dict:
        return {}


class _BadAdapter:
    async def get_greeks(self, instrument_name: str) -> dict:
        raise RuntimeError("ticker unreachable")


@pytest.mark.asyncio
async def test_local_bs_fills_gamma_vega_theta_when_ticker_returns_only_delta():
    positions = [{"instrument_name": "BTC-29MAY26-80000-C", "size": 1.0, "side": "long"}]
    result = await aggregate_portfolio_greeks(
        positions=positions,
        greeks_source=_PartialGreeksAdapter(),
        today=date(2026, 4, 22),
        spot=80000.0,
    )
    pos = result["open_positions"][0]
    assert pos["delta"] == pytest.approx(0.52)  # ticker value preserved
    assert pos["gamma"] > 0
    assert pos["vega"] > 0
    assert pos["theta"] < 0
    assert pos.get("greeks_source") == "bs_local"


@pytest.mark.asyncio
async def test_portfolio_totals_reflect_local_backfill():
    positions = [{"instrument_name": "BTC-29MAY26-80000-C", "size": 1.0, "side": "long"}]
    result = await aggregate_portfolio_greeks(
        positions=positions,
        greeks_source=_PartialGreeksAdapter(),
        today=date(2026, 4, 22),
        spot=80000.0,
    )
    assert result["portfolio_greeks"]["gamma"] > 0
    assert result["portfolio_greeks"]["vega"] > 0


@pytest.mark.asyncio
async def test_backfill_skipped_when_no_iv_available():
    """No iv → no BS → greeks stay empty (we don't invent data)."""
    positions = [{"instrument_name": "BTC-29MAY26-80000-C", "size": 1.0, "side": "long"}]
    result = await aggregate_portfolio_greeks(
        positions=positions,
        greeks_source=_EmptyGreeksAdapter(),
        today=date(2026, 4, 22),
        spot=80000.0,
    )
    pos = result["open_positions"][0]
    assert "gamma" not in pos
    assert "vega" not in pos


@pytest.mark.asyncio
async def test_short_position_contributes_negative_signed_greeks():
    positions = [{"instrument_name": "BTC-29MAY26-80000-C", "size": 1.0, "side": "short"}]
    result = await aggregate_portfolio_greeks(
        positions=positions,
        greeks_source=_PartialGreeksAdapter(),
        today=date(2026, 4, 22),
        spot=80000.0,
    )
    assert result["portfolio_greeks"]["delta"] < 0  # short call = negative delta
    assert result["portfolio_greeks"]["gamma"] < 0  # short gamma
