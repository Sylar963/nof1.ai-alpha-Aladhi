"""Tests for the options portfolio aggregator.

The aggregator walks the Thalex account's open positions, fetches per-leg
greeks via the adapter, parses days-to-expiry from each instrument name,
and returns the compact lists the OptionsContext snapshot needs."""

from datetime import date

import pytest

from src.backend.options_intel.portfolio import aggregate_portfolio_greeks


class _FakeGreeksSource:
    """Stand-in for ThalexAPI.get_greeks — returns pre-seeded greeks per instrument."""

    def __init__(self, greeks_by_instrument):
        self.greeks_by_instrument = greeks_by_instrument
        self.calls: list[str] = []

    async def get_greeks(self, instrument_name):
        self.calls.append(instrument_name)
        return self.greeks_by_instrument.get(instrument_name, {})


_TODAY = date(2026, 4, 11)


# ---------------------------------------------------------------------------
# Empty / no positions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregate_with_no_positions_returns_zeros():
    source = _FakeGreeksSource({})
    result = await aggregate_portfolio_greeks(positions=[], greeks_source=source, today=_TODAY)
    assert result["open_positions"] == []
    assert result["portfolio_greeks"] == {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    assert source.calls == []


# ---------------------------------------------------------------------------
# Single long position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregate_single_long_call_position():
    """A single +0.05 long call: portfolio greeks = 0.05 × per-contract greeks."""
    source = _FakeGreeksSource({
        "BTC-25APR26-65000-C": {
            "delta": 0.42,
            "gamma": 0.001,
            "vega": 8.0,
            "theta": -3.5,
        },
    })
    positions = [
        {"instrument_name": "BTC-25APR26-65000-C", "size": 0.05, "side": "long"},
    ]

    result = await aggregate_portfolio_greeks(positions, source, today=_TODAY)

    assert len(result["open_positions"]) == 1
    pos = result["open_positions"][0]
    assert pos["instrument_name"] == "BTC-25APR26-65000-C"
    assert pos["size"] == 0.05
    assert pos["delta"] == pytest.approx(0.42)
    assert pos["days_to_expiry"] == 14  # Apr 25 - Apr 11

    pg = result["portfolio_greeks"]
    assert pg["delta"] == pytest.approx(0.05 * 0.42)
    assert pg["gamma"] == pytest.approx(0.05 * 0.001)
    assert pg["vega"] == pytest.approx(0.05 * 8.0)
    assert pg["theta"] == pytest.approx(0.05 * -3.5)


# ---------------------------------------------------------------------------
# Multi-position aggregation with mixed sides
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregate_short_position_flips_signs():
    """A short position contributes its greeks with the opposite sign."""
    source = _FakeGreeksSource({
        "BTC-25APR26-55000-P": {
            "delta": -0.30,
            "gamma": 0.0008,
            "vega": 6.5,
            "theta": -2.0,
        },
    })
    positions = [
        {"instrument_name": "BTC-25APR26-55000-P", "size": 0.10, "side": "short"},
    ]

    result = await aggregate_portfolio_greeks(positions, source, today=_TODAY)
    pg = result["portfolio_greeks"]
    # Short flips sign: portfolio_delta = -0.10 * -0.30 = +0.03
    assert pg["delta"] == pytest.approx(0.03)
    assert pg["theta"] == pytest.approx(0.10 * 2.0)  # short is theta-positive


@pytest.mark.asyncio
async def test_aggregate_multi_position_sums_correctly():
    source = _FakeGreeksSource({
        "BTC-25APR26-65000-C": {"delta": 0.42, "gamma": 0.001, "vega": 8.0, "theta": -3.5},
        "BTC-25APR26-55000-P": {"delta": -0.30, "gamma": 0.0008, "vega": 6.5, "theta": -2.0},
    })
    positions = [
        {"instrument_name": "BTC-25APR26-65000-C", "size": 0.05, "side": "long"},
        {"instrument_name": "BTC-25APR26-55000-P", "size": 0.10, "side": "long"},
    ]
    result = await aggregate_portfolio_greeks(positions, source, today=_TODAY)
    pg = result["portfolio_greeks"]
    assert pg["delta"] == pytest.approx(0.05 * 0.42 + 0.10 * -0.30)
    assert pg["vega"] == pytest.approx(0.05 * 8.0 + 0.10 * 6.5)
    assert len(result["open_positions"]) == 2


@pytest.mark.asyncio
async def test_aggregate_handles_get_greeks_failure_gracefully():
    """If get_greeks raises for one instrument, that position keeps static data."""

    class _FlakyGreeks:
        def __init__(self):
            self.calls: list[str] = []

        async def get_greeks(self, name):
            self.calls.append(name)
            if name == "BTC-25APR26-50000-P":
                raise RuntimeError("simulated failure")
            return {"delta": 0.42, "gamma": 0.001, "vega": 8.0, "theta": -3.5}

    source = _FlakyGreeks()
    positions = [
        {"instrument_name": "BTC-25APR26-65000-C", "size": 0.05, "side": "long"},
        {"instrument_name": "BTC-25APR26-50000-P", "size": 0.10, "side": "long"},
    ]
    result = await aggregate_portfolio_greeks(positions, source, today=_TODAY)
    # The successful position contributes; the failing one keeps static data
    # so a single ticker hiccup doesn't blank the whole portfolio view.
    assert result["portfolio_greeks"]["delta"] == pytest.approx(0.05 * 0.42)
    assert len(result["open_positions"]) == 2


@pytest.mark.asyncio
async def test_aggregate_skips_unparseable_instrument_names():
    """Instruments that don't parse as options are dropped — no days_to_expiry available."""
    source = _FakeGreeksSource({
        "BTC-PERPETUAL": {"delta": 1.0, "gamma": 0, "vega": 0, "theta": 0},
        "BTC-25APR26-65000-C": {"delta": 0.42, "gamma": 0.001, "vega": 8.0, "theta": -3.5},
    })
    positions = [
        {"instrument_name": "BTC-PERPETUAL", "size": 0.5, "side": "long"},
        {"instrument_name": "BTC-25APR26-65000-C", "size": 0.05, "side": "long"},
    ]
    result = await aggregate_portfolio_greeks(positions, source, today=_TODAY)
    assert len(result["open_positions"]) == 1
    assert result["open_positions"][0]["instrument_name"] == "BTC-25APR26-65000-C"


@pytest.mark.asyncio
async def test_aggregate_ignores_positions_with_size_zero():
    source = _FakeGreeksSource({"BTC-25APR26-65000-C": {"delta": 0.42, "gamma": 0, "vega": 0, "theta": 0}})
    positions = [
        {"instrument_name": "BTC-25APR26-65000-C", "size": 0.0, "side": "long"},
    ]
    result = await aggregate_portfolio_greeks(positions, source, today=_TODAY)
    assert result["open_positions"] == []
    assert result["portfolio_greeks"]["delta"] == pytest.approx(0.0)
