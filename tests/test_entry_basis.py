"""Fill-based entry premium basis for open option structures."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.backend.options_intel.entry_basis import (
    compute_fill_entry_prices,
    fill_entry_price_map,
)
from src.backend.options_intel.structure import OptionLeg, classify_many


def leg(name, kind, strike, side, contracts=0.1, dte=14, mark=500.0):
    return OptionLeg(
        instrument_name=name,
        kind=kind,
        strike=Decimal(str(strike)),
        side=side,
        contracts=Decimal(str(contracts)),
        days_to_expiry=dte,
        mark_price=Decimal(str(mark)),
        delta=None,
        gamma=None,
        vega=None,
        theta=None,
    )


class TestComputeFillEntryPrices:
    def test_single_fill_exact_coverage(self):
        fills = [{"instrument_name": "BTC-X-100000-C", "direction": "sell",
                  "amount": 0.1, "price": 1200.0, "time": 100}]
        out = compute_fill_entry_prices(fills, {"BTC-X-100000-C": ("short", 0.1)})
        assert out["BTC-X-100000-C"] == Decimal("1200.0")

    def test_vwap_across_fills(self):
        fills = [
            {"instrument_name": "I", "direction": "buy", "amount": 0.05, "price": 1000.0, "time": 200},
            {"instrument_name": "I", "direction": "buy", "amount": 0.05, "price": 2000.0, "time": 100},
        ]
        out = compute_fill_entry_prices(fills, {"I": ("long", 0.1)})
        assert out["I"] == Decimal("1500.0")

    def test_lifo_uses_newest_fills_first(self):
        fills = [
            {"instrument_name": "I", "direction": "buy", "amount": 0.1, "price": 900.0, "time": 50},
            {"instrument_name": "I", "direction": "buy", "amount": 0.1, "price": 1100.0, "time": 300},
        ]
        out = compute_fill_entry_prices(fills, {"I": ("long", 0.1)})
        assert out["I"] == Decimal("1100.0")

    def test_opposite_direction_fills_ignored(self):
        fills = [
            {"instrument_name": "I", "direction": "buy", "amount": 0.05, "price": 400.0, "time": 300},
            {"instrument_name": "I", "direction": "sell", "amount": 0.1, "price": 1200.0, "time": 200},
        ]
        out = compute_fill_entry_prices(fills, {"I": ("short", 0.1)})
        assert out["I"] == Decimal("1200.0")

    def test_insufficient_coverage_omitted(self):
        fills = [{"instrument_name": "I", "direction": "sell",
                  "amount": 0.05, "price": 1200.0, "time": 100}]
        out = compute_fill_entry_prices(fills, {"I": ("short", 0.1)})
        assert out == {}

    def test_partial_fill_taken_up_to_position_size(self):
        fills = [{"instrument_name": "I", "direction": "sell",
                  "amount": 0.5, "price": 1200.0, "time": 100}]
        out = compute_fill_entry_prices(fills, {"I": ("short", 0.1)})
        assert out["I"] == Decimal("1200.0")


class TestFillEntryPriceMap:
    @pytest.mark.asyncio
    async def test_adapter_without_fills_method_is_noop(self):
        out = await fill_entry_price_map(SimpleNamespace(), {"I": ("short", 0.1)})
        assert out == {}

    @pytest.mark.asyncio
    async def test_fetch_failure_is_noop(self):
        async def boom(limit=200):
            raise RuntimeError("venue down")
        out = await fill_entry_price_map(
            SimpleNamespace(get_recent_fills=boom), {"I": ("short", 0.1)},
        )
        assert out == {}


class TestClassifyWithFillBasis:
    def _spread_legs(self):
        return [
            leg("BTC-X-95000-P", "put", 95000, "short", mark=900.0),
            leg("BTC-X-90000-P", "put", 90000, "long", mark=500.0),
        ]

    def test_fill_basis_beats_persisted_mark_entry(self):
        legs = self._spread_legs()
        structures = classify_many(
            legs,
            entry_net_premium_by_id={"ignored": 999.0},
            entry_price_by_instrument={
                "BTC-X-95000-P": Decimal("1200.0"),
                "BTC-X-90000-P": Decimal("400.0"),
            },
        )
        assert len(structures) == 1
        s = structures[0]
        assert s.entry_net_premium == Decimal("0.1") * (Decimal("1200") - Decimal("400"))
        current = Decimal("0.1") * (Decimal("900") - Decimal("500"))
        assert s.pnl_abs == s.entry_net_premium - current

    def test_missing_leg_basis_falls_back_to_persisted_entry(self):
        legs = self._spread_legs()
        from src.backend.options_intel.structure import compute_structure_id
        sid = compute_structure_id(legs)
        structures = classify_many(
            legs,
            entry_net_premium_by_id={sid: 70.0},
            entry_price_by_instrument={"BTC-X-95000-P": Decimal("1200.0")},
        )
        assert structures[0].entry_net_premium == Decimal("70")
