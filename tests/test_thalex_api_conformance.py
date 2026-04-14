"""Structural conformance tests for ThalexAPI as an ExchangeAdapter.

These tests verify the class shape and config loading. They do NOT open a
WebSocket connection to Thalex — that lives in the integration suite."""

import pytest
from types import SimpleNamespace

from src.backend.trading.exchange_adapter import ExchangeAdapter
from src.backend.trading import thalex_api as thalex_module


def test_thalex_api_is_subclass_of_exchange_adapter():
    assert issubclass(thalex_module.ThalexAPI, ExchangeAdapter)


def test_thalex_api_has_venue_attribute():
    assert thalex_module.ThalexAPI.venue == "thalex"


def test_thalex_api_implements_every_abstract_method():
    abstract_methods = ExchangeAdapter.__abstractmethods__
    for name in abstract_methods:
        attr = getattr(thalex_module.ThalexAPI, name, None)
        assert attr is not None, f"ThalexAPI is missing {name}"
        assert callable(attr)


def test_thalex_api_factory_registered():
    from src.backend.trading.exchange_factory import ExchangeFactory
    assert "thalex" in ExchangeFactory.available_venues()


def test_thalex_api_constructs_in_test_mode_without_connecting(tmp_path, monkeypatch):
    """Construction must succeed even without a real PEM file path,
    as long as we're not asked to connect. The PEM is only loaded at connect time."""
    monkeypatch.setenv("THALEX_NETWORK", "test")
    monkeypatch.setenv("THALEX_KEY_ID", "test-kid")
    monkeypatch.setenv("THALEX_PRIVATE_KEY_PATH", str(tmp_path / "fake.pem"))
    instance = thalex_module.ThalexAPI()
    assert instance.venue == "thalex"
    assert instance.network_name == "test"
    assert instance.key_id == "test-kid"
    assert instance.connected is False


def test_thalex_api_rejects_unknown_network(monkeypatch):
    monkeypatch.setenv("THALEX_NETWORK", "wonderland")
    monkeypatch.setenv("THALEX_KEY_ID", "test-kid")
    monkeypatch.setenv("THALEX_PRIVATE_KEY_PATH", "/tmp/fake.pem")
    with pytest.raises(ValueError, match="THALEX_NETWORK"):
        thalex_module.ThalexAPI()


@pytest.mark.asyncio
async def test_place_buy_order_uses_ioc_limit_price_guard():
    from thalex import OrderType, TimeInForce

    adapter = thalex_module.ThalexAPI.__new__(thalex_module.ThalexAPI)
    adapter.venue = "thalex"
    adapter._client = SimpleNamespace(ticker=object(), insert=object())

    calls = []

    async def _fake_request_with_retry(sender, **kwargs):
        if sender is adapter._client.ticker:
            return {"best_ask": 100.0}
        if sender is adapter._client.insert:
            calls.append(kwargs)
            return {"status": "filled", "price": kwargs["price"], "order_id": "o1"}
        raise AssertionError(f"unexpected sender {sender!r}")

    adapter._request_with_retry = _fake_request_with_retry

    order = await thalex_module.ThalexAPI.place_buy_order(adapter, "BTC-10MAY26-65000-C", 0.05, slippage=0.02)

    assert calls[0]["order_type"] == OrderType.LIMIT
    assert calls[0]["time_in_force"] == TimeInForce.IOC
    assert calls[0]["price"] == pytest.approx(102.0)
    assert order.price == pytest.approx(102.0)


@pytest.mark.asyncio
async def test_place_take_profit_submits_reduce_only_gtc_limit():
    from thalex import Direction, OrderType, TimeInForce

    adapter = thalex_module.ThalexAPI.__new__(thalex_module.ThalexAPI)
    adapter.venue = "thalex"
    adapter._client = SimpleNamespace(insert=object())

    calls = []

    async def _fake_request_with_retry(sender, **kwargs):
        assert sender is adapter._client.insert
        calls.append(kwargs)
        return {"status": "open", "price": kwargs["price"], "order_id": "tp1"}

    adapter._request_with_retry = _fake_request_with_retry

    order = await thalex_module.ThalexAPI.place_take_profit(adapter, "BTC-10MAY26-65000-C", True, 0.05, 1400.0)

    assert calls[0]["direction"] == Direction.SELL
    assert calls[0]["order_type"] == OrderType.LIMIT
    assert calls[0]["time_in_force"] == TimeInForce.GTC
    assert calls[0]["reduce_only"] is True
    assert order.price == pytest.approx(1400.0)


@pytest.mark.asyncio
async def test_place_stop_loss_submits_native_conditional_order():
    from thalex import Direction, Target

    adapter = thalex_module.ThalexAPI.__new__(thalex_module.ThalexAPI)
    adapter.venue = "thalex"
    adapter._client = SimpleNamespace(create_conditional_order=object())

    calls = []

    async def _fake_request_with_retry(sender, **kwargs):
        assert sender is adapter._client.create_conditional_order
        calls.append(kwargs)
        return {"status": "open", "order_id": "sl1"}

    adapter._request_with_retry = _fake_request_with_retry

    order = await thalex_module.ThalexAPI.place_stop_loss(adapter, "BTC-10MAY26-65000-C", True, 0.05, 900.0)

    assert calls[0]["direction"] == Direction.SELL
    assert calls[0]["stop_price"] == pytest.approx(900.0)
    assert calls[0]["reduce_only"] is True
    assert calls[0]["target"] == Target.MARK
    assert order.order_id == "sl1"


@pytest.mark.asyncio
async def test_place_bracket_order_submits_native_bracket():
    from thalex import Direction

    adapter = thalex_module.ThalexAPI.__new__(thalex_module.ThalexAPI)
    adapter.venue = "thalex"
    adapter._client = SimpleNamespace(create_conditional_order=object())

    calls = []

    async def _fake_request_with_retry(sender, **kwargs):
        assert sender is adapter._client.create_conditional_order
        calls.append(kwargs)
        return {"status": "open", "order_id": "br1"}

    adapter._request_with_retry = _fake_request_with_retry

    order = await thalex_module.ThalexAPI.place_bracket_order(
        adapter,
        "BTC-10MAY26-65000-C",
        True,
        0.05,
        900.0,
        1400.0,
    )

    assert calls[0]["direction"] == Direction.SELL
    assert calls[0]["stop_price"] == pytest.approx(900.0)
    assert calls[0]["bracket_price"] == pytest.approx(1400.0)
    assert order.order_id == "br1"


@pytest.mark.asyncio
async def test_place_trailing_stop_submits_native_trailing_order():
    adapter = thalex_module.ThalexAPI.__new__(thalex_module.ThalexAPI)
    adapter.venue = "thalex"
    adapter._client = SimpleNamespace(create_conditional_order=object())

    calls = []

    async def _fake_request_with_retry(sender, **kwargs):
        assert sender is adapter._client.create_conditional_order
        calls.append(kwargs)
        return {"status": "open", "order_id": "tr1"}

    adapter._request_with_retry = _fake_request_with_retry

    order = await thalex_module.ThalexAPI.place_trailing_stop_order(
        adapter,
        "BTC-10MAY26-65000-C",
        True,
        0.05,
        950.0,
        0.02,
    )

    assert calls[0]["stop_price"] == pytest.approx(950.0)
    assert calls[0]["trailing_stop_callback_rate"] == pytest.approx(0.02)
    assert order.order_id == "tr1"
