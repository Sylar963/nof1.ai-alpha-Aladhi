"""Tests for the ExchangeFactory venue dispatcher."""

import pytest

from src.backend.trading.exchange_adapter import ExchangeAdapter, AccountState, OrderResult
from src.backend.trading.exchange_factory import ExchangeFactory


class _StubAdapter(ExchangeAdapter):
    """Minimal in-memory adapter used to register and retrieve via the factory."""

    venue = "stub"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def place_buy_order(self, asset, amount, slippage=0.01):
        self.calls.append(f"buy {asset} {amount}")
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


@pytest.fixture
def fresh_factory():
    """Yield a factory with a clean registry, then restore."""
    saved = ExchangeFactory._registry.copy()
    ExchangeFactory._registry.clear()
    try:
        yield ExchangeFactory
    finally:
        ExchangeFactory._registry.clear()
        ExchangeFactory._registry.update(saved)


def test_factory_raises_on_unknown_venue(fresh_factory):
    """Asking for a venue that was never registered must fail loudly."""
    with pytest.raises(ValueError, match="Unknown venue"):
        fresh_factory.create("does-not-exist")


def test_factory_creates_registered_adapter(fresh_factory):
    """A registered venue string must return a fresh adapter instance."""
    fresh_factory.register("stub", _StubAdapter)
    instance = fresh_factory.create("stub")
    assert isinstance(instance, _StubAdapter)
    assert instance.venue == "stub"


def test_factory_register_is_case_insensitive(fresh_factory):
    """Venue lookups must tolerate casing differences from config files."""
    fresh_factory.register("Stub", _StubAdapter)
    assert isinstance(fresh_factory.create("STUB"), _StubAdapter)
    assert isinstance(fresh_factory.create("stub"), _StubAdapter)


def test_factory_lists_registered_venues(fresh_factory):
    """`available_venues()` should expose the registry for diagnostics."""
    fresh_factory.register("stub", _StubAdapter)
    assert "stub" in fresh_factory.available_venues()


def test_factory_default_registry_includes_hyperliquid():
    """The shipped factory must register Hyperliquid out of the box."""
    assert "hyperliquid" in ExchangeFactory.available_venues()
