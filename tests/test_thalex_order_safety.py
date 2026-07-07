"""Order-safety regression tests for ThalexAPI.

Covers the audit fixes: client_order_id on every insert, timeout landed-check
instead of blind retry, pending-future cleanup on timeout, mid-based entry
pricing capped at the touch, passive tick alignment, partial-fill status
preservation, and the fail-closed margin preflight.
"""

import asyncio
import itertools
from types import SimpleNamespace

import pytest

from src.backend.config_loader import CONFIG
from src.backend.trading import thalex_api as thalex_module


def _bare_adapter():
    adapter = thalex_module.ThalexAPI.__new__(thalex_module.ThalexAPI)
    adapter.venue = "thalex"
    return adapter


@pytest.mark.asyncio
async def test_order_insert_timeout_checks_venue_and_does_not_retry():
    """If an insert times out but the order landed, the retry loop must adopt
    the venue-side order instead of re-sending (double-fill guard)."""
    adapter = _bare_adapter()
    client = SimpleNamespace(
        ticker=object(), insert=object(), open_orders=object(), trade_history=object()
    )
    adapter._client = client

    insert_attempts = []

    async def _fake_request(sender, **kwargs):
        if sender is client.ticker:
            return {"best_bid": 90.0, "best_ask": 100.0}
        if sender is client.insert:
            insert_attempts.append(kwargs)
            raise asyncio.TimeoutError()
        if sender is client.open_orders:
            return [
                {
                    "order_id": "o-77",
                    "client_order_id": insert_attempts[-1]["client_order_id"],
                    "status": "open",
                }
            ]
        raise AssertionError(f"unexpected sender {sender!r}")

    adapter._request = _fake_request

    order = await adapter.place_buy_order("BTC-10MAY26-65000-C", 0.05)

    assert len(insert_attempts) == 1
    assert order.order_id == "o-77"
    assert order.status == "resting"


@pytest.mark.asyncio
async def test_order_insert_timeout_retries_when_order_did_not_land():
    adapter = _bare_adapter()
    client = SimpleNamespace(
        ticker=object(), insert=object(), open_orders=object(), trade_history=object()
    )
    adapter._client = client

    insert_attempts = []

    async def _fake_request(sender, **kwargs):
        if sender is client.ticker:
            return {"best_bid": 90.0, "best_ask": 100.0}
        if sender is client.insert:
            insert_attempts.append(kwargs)
            if len(insert_attempts) == 1:
                raise asyncio.TimeoutError()
            return {"status": "filled", "order_id": "o-88"}
        if sender is client.open_orders:
            return []
        if sender is client.trade_history:
            return []
        raise AssertionError(f"unexpected sender {sender!r}")

    adapter._request = _fake_request

    order = await adapter.place_buy_order("BTC-10MAY26-65000-C", 0.05)

    assert len(insert_attempts) == 2
    # Same client_order_id reused across retries so the venue can dedupe.
    assert insert_attempts[0]["client_order_id"] == insert_attempts[1]["client_order_id"]
    assert order.order_id == "o-88"
    assert order.status == "filled"


@pytest.mark.asyncio
async def test_request_timeout_pops_pending_future(monkeypatch):
    adapter = _bare_adapter()
    adapter.connected = True
    adapter._is_client_alive = lambda: True
    adapter._pending = {}
    adapter._id_counter = itertools.count(1)

    async def _fake_wait_for(fut, timeout):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(thalex_module.asyncio, "wait_for", _fake_wait_for)

    async def _sender(id, **kwargs):
        pass

    with pytest.raises(asyncio.TimeoutError):
        await adapter._request(_sender)

    assert adapter._pending == {}


@pytest.mark.asyncio
async def test_entry_limit_price_buys_at_mid_plus_fraction_of_half_spread(monkeypatch):
    monkeypatch.setitem(CONFIG, "thalex_cross_fraction", 0.25)
    adapter = _bare_adapter()
    adapter._client = SimpleNamespace(ticker=object())
    adapter._instruments_cache = [{"instrument_name": "X", "tick_size": 0.5}]

    async def _fake_request_with_retry(sender, **kwargs):
        return {"best_bid": 90.0, "best_ask": 100.0}

    adapter._request_with_retry = _fake_request_with_retry

    # mid=95, half-spread=5 → buy 96.25 aligned DOWN to 96.0
    buy_price = await adapter._entry_limit_price("X", "buy", 0.01)
    assert buy_price == pytest.approx(96.0)
    # sell 93.75 aligned UP to 94.0
    sell_price = await adapter._entry_limit_price("X", "sell", 0.01)
    assert sell_price == pytest.approx(94.0)


@pytest.mark.asyncio
async def test_entry_limit_price_never_crosses_the_touch(monkeypatch):
    monkeypatch.setitem(CONFIG, "thalex_cross_fraction", 3.0)
    adapter = _bare_adapter()
    adapter._client = SimpleNamespace(ticker=object())
    adapter._instruments_cache = []

    async def _fake_request_with_retry(sender, **kwargs):
        return {"best_bid": 90.0, "best_ask": 100.0}

    adapter._request_with_retry = _fake_request_with_retry

    assert await adapter._entry_limit_price("X", "buy", 0.01) <= 100.0
    assert await adapter._entry_limit_price("X", "sell", 0.01) >= 90.0


def test_align_price_rounds_toward_passive():
    align = thalex_module.ThalexAPI._align_price
    assert align(101.3, 0.5, "buy") == pytest.approx(101.0)
    assert align(101.3, 0.5, "sell") == pytest.approx(101.5)
    assert align(101.5, 0.5, "buy") == pytest.approx(101.5)
    assert align(101.5, 0.5, "sell") == pytest.approx(101.5)


def test_partially_filled_status_preserved_with_filled_amount():
    adapter = _bare_adapter()
    order = adapter._make_order_result(
        "BTC-10MAY26-65000-C",
        "buy",
        0.05,
        {"status": "partially_filled", "order_id": "o1", "filled_amount": 0.02},
    )
    assert order.status == "partially_filled"
    assert order.filled_amount == pytest.approx(0.02)

    full = adapter._make_order_result(
        "BTC-10MAY26-65000-C", "buy", 0.05, {"status": "filled", "order_id": "o2"}
    )
    assert full.status == "filled"
    assert full.filled_amount == pytest.approx(0.05)

    resting = adapter._make_order_result(
        "BTC-10MAY26-65000-C", "buy", 0.05, {"status": "open", "order_id": "o3"}
    )
    assert resting.status == "resting"
    assert resting.filled_amount == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_margin_preflight_fails_closed_on_rpc_error():
    adapter = _bare_adapter()

    async def _broken_user_state():
        raise RuntimeError("account_summary RPC failed")

    adapter.get_user_state = _broken_user_state

    ok, reason = await adapter.margin_preflight(100.0)
    assert ok is False
    assert "account_summary" in reason


def test_preflight_reducing_exempts_position_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("THALEX_NETWORK", "test")
    monkeypatch.setenv("THALEX_KEY_ID", "test-kid")
    monkeypatch.setenv("THALEX_PRIVATE_KEY_PATH", str(tmp_path / "fake.pem"))
    adapter = thalex_module.ThalexAPI()

    ok, reason = adapter.preflight("BTC", 0.05, open_positions_count=99)
    assert ok is False
    assert "max_open_positions" in reason

    ok, _ = adapter.preflight("BTC", 0.05, open_positions_count=99, reducing=True)
    assert ok is True
