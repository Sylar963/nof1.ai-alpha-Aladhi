"""Tests for the ExchangeAdapter abstract base class."""

import pytest

from src.backend.trading.exchange_adapter import (
    ExchangeAdapter,
    OrderResult,
    PositionSnapshot,
    AccountState,
)


def test_exchange_adapter_cannot_be_instantiated_directly():
    """ABC must refuse instantiation — it's a contract, not a class."""
    with pytest.raises(TypeError):
        ExchangeAdapter()  # type: ignore[abstract]


def test_subclass_missing_methods_cannot_be_instantiated():
    """A subclass that does not implement every abstract method must fail at construction."""

    class Incomplete(ExchangeAdapter):
        venue = "incomplete"

        async def place_buy_order(self, asset, amount, slippage=0.01):
            return OrderResult(venue=self.venue, order_id="x", asset=asset, side="buy", amount=amount, status="ok")

        # Intentionally missing every other abstract method.

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_complete_subclass_can_be_instantiated():
    """A subclass that implements every abstract method must instantiate cleanly."""

    class Complete(ExchangeAdapter):
        venue = "complete"

        async def place_buy_order(self, asset, amount, slippage=0.01):
            return OrderResult(venue=self.venue, order_id="1", asset=asset, side="buy", amount=amount, status="ok")

        async def place_sell_order(self, asset, amount, slippage=0.01):
            return OrderResult(venue=self.venue, order_id="2", asset=asset, side="sell", amount=amount, status="ok")

        async def place_take_profit(self, asset, is_buy, amount, tp_price):
            return OrderResult(venue=self.venue, order_id="3", asset=asset, side="tp", amount=amount, status="ok")

        async def place_stop_loss(self, asset, is_buy, amount, sl_price):
            return OrderResult(venue=self.venue, order_id="4", asset=asset, side="sl", amount=amount, status="ok")

        async def cancel_order(self, asset, order_id):
            return {"status": "ok"}

        async def cancel_all_orders(self, asset):
            return {"status": "ok"}

        async def get_open_orders(self):
            return []

        async def get_recent_fills(self, limit=50):
            return []

        async def get_user_state(self):
            return AccountState(venue=self.venue, balance=0.0, total_value=0.0, positions=[])

        async def get_current_price(self, asset):
            return 0.0

    instance = Complete()
    assert isinstance(instance, ExchangeAdapter)
    assert instance.venue == "complete"


def test_order_result_dataclass_carries_venue_tag():
    """OrderResult must always carry a venue tag — that's how routing stays sane."""
    result = OrderResult(
        venue="thalex",
        order_id="abc-123",
        asset="BTC-27JUN25-100000-C",
        side="buy",
        amount=0.1,
        status="filled",
    )
    assert result.venue == "thalex"
    assert result.order_id == "abc-123"
    assert result.status == "filled"


def test_position_snapshot_dataclass_supports_options_instruments():
    """PositionSnapshot must distinguish underlying asset from full instrument name."""
    snap = PositionSnapshot(
        venue="thalex",
        asset="BTC",
        instrument_name="BTC-27JUN25-100000-C",
        side="long",
        size=0.1,
        entry_price=1250.0,
        current_price=1300.0,
        unrealized_pnl=5.0,
    )
    assert snap.asset == "BTC"
    assert snap.instrument_name == "BTC-27JUN25-100000-C"
    assert snap.size == 0.1


def test_position_snapshot_defaults_instrument_name_to_asset_for_spot_perp():
    """For spot/perp venues there is no separate instrument name — default to asset."""
    snap = PositionSnapshot(
        venue="hyperliquid",
        asset="BTC",
        side="long",
        size=0.5,
        entry_price=60000.0,
        current_price=61000.0,
        unrealized_pnl=500.0,
    )
    assert snap.instrument_name == "BTC"
