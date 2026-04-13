"""Tests for ThalexAPI ticker subscription path.

These tests cover the subscribe / unsubscribe lifecycle and the ``_dispatch``
routing of channel notifications to registered subscriber callbacks. They
bypass the WebSocket entirely by patching the underlying client + ``_request``
on a constructed adapter instance."""

import asyncio
from types import SimpleNamespace

import pytest

from src.backend.trading.thalex_api import ThalexAPI


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("THALEX_NETWORK", "test")
    monkeypatch.setenv("THALEX_KEY_ID", "test-kid")
    monkeypatch.setenv("THALEX_PRIVATE_KEY_PATH", str(tmp_path / "fake.pem"))
    instance = ThalexAPI()
    instance.connected = True
    instance._client = SimpleNamespace(
        public_subscribe=lambda **kw: None,
        public_unsubscribe=lambda **kw: None,
    )
    return instance


def _record_request():
    """Async stub that records each invocation and returns ``None``."""
    sent: list[dict] = []

    async def _stub(sender=None, **kwargs):
        sent.append({"sender_name": getattr(sender, "__name__", repr(sender)), **kwargs})
        return {"status": "ok"}

    _stub.sent = sent
    return _stub


# ---------------------------------------------------------------------------
# subscribe_ticker / unsubscribe_ticker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_ticker_registers_callback_and_sends_rpc(adapter):
    """A subscribe call must store the callback and emit a public/subscribe RPC."""
    adapter._request = _record_request()
    received: list[dict] = []

    async def cb(payload):
        received.append(payload)

    await adapter.subscribe_ticker("BTC-10MAY26-65000-C", cb)

    assert "BTC-10MAY26-65000-C" in adapter._ticker_subscribers
    assert adapter._ticker_subscribers["BTC-10MAY26-65000-C"] is cb
    assert len(adapter._request.sent) == 1
    sent = adapter._request.sent[0]
    assert "channels" in sent
    channel_name = sent["channels"][0]
    assert "ticker" in channel_name
    assert "BTC-10MAY26-65000-C" in channel_name


@pytest.mark.asyncio
async def test_unsubscribe_ticker_removes_callback(adapter):
    """Unsubscribing must drop the callback and (best-effort) emit unsubscribe."""
    adapter._request = _record_request()

    async def cb(payload):
        pass

    await adapter.subscribe_ticker("BTC-10MAY26-65000-C", cb)
    await adapter.unsubscribe_ticker("BTC-10MAY26-65000-C")

    assert "BTC-10MAY26-65000-C" not in adapter._ticker_subscribers


@pytest.mark.asyncio
async def test_subscribe_replaces_existing_callback(adapter):
    """Subscribing twice for the same instrument replaces the callback."""
    adapter._request = _record_request()

    async def cb1(payload):
        pass

    async def cb2(payload):
        pass

    await adapter.subscribe_ticker("BTC-10MAY26-65000-C", cb1)
    await adapter.subscribe_ticker("BTC-10MAY26-65000-C", cb2)

    assert adapter._ticker_subscribers["BTC-10MAY26-65000-C"] is cb2


# ---------------------------------------------------------------------------
# _dispatch channel routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_routes_subscription_notification_to_callback(adapter):
    """A JSON-RPC notification with method='subscription' must invoke the callback."""
    received: list[dict] = []
    done = asyncio.Event()

    async def cb(payload):
        received.append(payload)
        done.set()

    adapter._ticker_subscribers["BTC-10MAY26-65000-C"] = cb

    notification = {
        "method": "subscription",
        "params": {
            "channel": "ticker.BTC-10MAY26-65000-C.raw",
            "notification": {"mark_price": 1250.0, "greeks": {"delta": 0.42}},
        },
    }
    adapter._dispatch(notification)

    await asyncio.wait_for(done.wait(), timeout=1.0)
    assert received == [{"mark_price": 1250.0, "greeks": {"delta": 0.42}}]


@pytest.mark.asyncio
async def test_dispatch_ignores_unknown_channel(adapter):
    """A subscription update for an instrument we never subscribed to is dropped silently."""
    adapter._dispatch({
        "method": "subscription",
        "params": {
            "channel": "ticker.ETH-10MAY26-3000-C.raw",
            "notification": {"mark_price": 99.0},
        },
    })
    # No exception, no state change — just verify the call returned cleanly.
    assert "ETH-10MAY26-3000-C" not in adapter._ticker_subscribers


@pytest.mark.asyncio
async def test_resubscribe_tickers_retries_transient_failures(adapter, monkeypatch):
    attempts = []

    async def _sleep(_seconds):
        return None

    async def _request(sender=None, **kwargs):
        attempts.append({"sender": sender, **kwargs})
        if len(attempts) == 1:
            raise RuntimeError("transient")
        return {"status": "ok"}

    adapter._ticker_subscribers["BTC-10MAY26-65000-C"] = lambda payload: None
    adapter._request = _request
    monkeypatch.setattr("src.backend.trading.thalex_api.asyncio.sleep", _sleep)

    await adapter._resubscribe_tickers()

    assert len(attempts) == 2
    assert attempts[-1]["channels"] == ["ticker.BTC-10MAY26-65000-C.raw"]
