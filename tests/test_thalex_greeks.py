"""Tests for ThalexAPI.get_greeks defensive parser + TTL cache.

These tests bypass the WebSocket entirely by monkey-patching the adapter's
``_request`` helper to return synthetic ticker payloads. The point is to lock
down the parser's tolerance for the multiple plausible field-name layouts the
Thalex API might return — the SDK source doesn't document the exact response
shape, so we have to be defensive."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from src.backend.trading.thalex_api import ThalexAPI


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    """Build a ThalexAPI without ever opening the WebSocket.

    Tests stub `_request` on the instance to inject synthetic ticker payloads.
    The fake client only needs a `ticker` attribute that's safe to reference;
    `_request` is mocked, so it never gets invoked.
    """
    monkeypatch.setenv("THALEX_NETWORK", "test")
    monkeypatch.setenv("THALEX_KEY_ID", "test-kid")
    monkeypatch.setenv("THALEX_PRIVATE_KEY_PATH", str(tmp_path / "fake.pem"))
    instance = ThalexAPI()
    instance.connected = True  # convince adapter not to try connecting
    instance._client = SimpleNamespace(ticker=lambda **kw: None)
    return instance


def _stub_request(payload):
    """Return an async stub that yields a fixed payload regardless of args."""
    call_count = {"n": 0}

    async def _stub(sender=None, **kwargs):
        call_count["n"] += 1
        return payload

    _stub.call_count = call_count
    return _stub


# ---------------------------------------------------------------------------
# Defensive parser
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_greeks_parses_top_level_delta(adapter):
    adapter._request = _stub_request({"delta": 0.42, "gamma": 0.001, "vega": 7.5, "theta": -3.1})
    result = await adapter.get_greeks("BTC-10MAY26-65000-C")
    assert result["delta"] == pytest.approx(0.42)
    assert result["gamma"] == pytest.approx(0.001)
    assert result["vega"] == pytest.approx(7.5)
    assert result["theta"] == pytest.approx(-3.1)


@pytest.mark.asyncio
async def test_get_greeks_parses_nested_greeks_dict(adapter):
    """Deribit-style nested greeks payload."""
    adapter._request = _stub_request({
        "mark_price": 1250.0,
        "greeks": {"delta": 0.55, "gamma": 0.002, "vega": 9.0, "theta": -2.0},
        "mark_iv": 0.65,
    })
    result = await adapter.get_greeks("BTC-10MAY26-65000-C")
    assert result["delta"] == pytest.approx(0.55)
    assert result["gamma"] == pytest.approx(0.002)
    assert result["mark_iv"] == pytest.approx(0.65)


@pytest.mark.asyncio
async def test_get_greeks_parses_singular_greek_dict(adapter):
    """Some venues use 'greek' singular instead of 'greeks'."""
    adapter._request = _stub_request({
        "mark_price": 900.0,
        "greek": {"delta": -0.3, "gamma": 0.001},
    })
    result = await adapter.get_greeks("BTC-10MAY26-50000-P")
    assert result["delta"] == pytest.approx(-0.3)


@pytest.mark.asyncio
async def test_get_greeks_returns_empty_dict_when_field_missing(adapter):
    """A response with no recognizable greeks must NOT raise — callers fall back."""
    adapter._request = _stub_request({"mark_price": 1250.0, "best_bid": 1240.0})
    result = await adapter.get_greeks("BTC-10MAY26-65000-C")
    assert result == {}


@pytest.mark.asyncio
async def test_get_greeks_returns_empty_dict_for_non_dict_payload(adapter):
    """Garbage payloads (None, list, str) must be tolerated."""
    adapter._request = _stub_request(None)
    assert await adapter.get_greeks("BTC-10MAY26-65000-C") == {}

    adapter._request = _stub_request(["unexpected"])
    assert await adapter.get_greeks("BTC-10MAY26-65000-C") == {}


# ---------------------------------------------------------------------------
# 5-second TTL cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_greeks_caches_for_five_seconds(adapter):
    """Two consecutive calls within the TTL must hit the wire only once."""
    payload = {"greeks": {"delta": 0.5}}
    stub = _stub_request(payload)
    adapter._request = stub

    first = await adapter.get_greeks("BTC-10MAY26-65000-C")
    second = await adapter.get_greeks("BTC-10MAY26-65000-C")

    assert first["delta"] == 0.5
    assert second["delta"] == 0.5
    assert stub.call_count["n"] == 1


@pytest.mark.asyncio
async def test_get_greeks_cache_expires(adapter, monkeypatch):
    """After the TTL expires, the next call must re-hit the wire."""
    payload_a = {"greeks": {"delta": 0.5}}
    payload_b = {"greeks": {"delta": 0.7}}

    stub_a = _stub_request(payload_a)
    adapter._request = stub_a
    await adapter.get_greeks("BTC-10MAY26-65000-C")

    # Fast-forward the cache TTL by rewriting the cached timestamp.
    cached_ts, cached_payload = adapter._greeks_cache["BTC-10MAY26-65000-C"]
    adapter._greeks_cache["BTC-10MAY26-65000-C"] = (cached_ts - 10.0, cached_payload)

    stub_b = _stub_request(payload_b)
    adapter._request = stub_b
    refreshed = await adapter.get_greeks("BTC-10MAY26-65000-C")

    assert refreshed["delta"] == 0.7
    assert stub_b.call_count["n"] == 1
