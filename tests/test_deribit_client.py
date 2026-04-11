"""Tests for the Deribit public REST client.

The client is a tiny aiohttp wrapper around the public JSON-RPC endpoints.
We test it without ever opening a real socket: a fake transport replaces
the underlying ``_get`` helper and records each call. The 15-minute TTL
cache is exercised by manipulating the cache timestamps directly."""

import pytest

from src.backend.options_intel.deribit_client import (
    DERIBIT_PROD_BASE_URL,
    DERIBIT_TEST_BASE_URL,
    DeribitPublicClient,
)


def _fake_transport(payload):
    """Build an async stub that records calls and returns a fixed payload."""
    sent: list[dict] = []

    async def _stub(path, params=None):
        sent.append({"path": path, "params": params or {}})
        return payload

    _stub.sent = sent
    return _stub


# ---------------------------------------------------------------------------
# URL / construction
# ---------------------------------------------------------------------------


def test_default_base_url_is_production():
    client = DeribitPublicClient()
    assert client.base_url == DERIBIT_PROD_BASE_URL


def test_test_network_uses_test_base_url():
    client = DeribitPublicClient(network="test")
    assert client.base_url == DERIBIT_TEST_BASE_URL


# ---------------------------------------------------------------------------
# Endpoint shapes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_instruments_filters_btc_options():
    client = DeribitPublicClient()
    fake = _fake_transport({
        "result": [
            {"instrument_name": "BTC-27JUN25-100000-C", "kind": "option"},
            {"instrument_name": "BTC-27JUN25-100000-P", "kind": "option"},
        ]
    })
    client._get = fake  # type: ignore[assignment]

    instruments = await client.get_instruments(currency="BTC", kind="option")

    assert len(instruments) == 2
    assert fake.sent[0]["path"] == "/public/get_instruments"
    assert fake.sent[0]["params"] == {"currency": "BTC", "kind": "option", "expired": "false"}


@pytest.mark.asyncio
async def test_get_book_summary_by_currency_returns_list():
    client = DeribitPublicClient()
    fake = _fake_transport({
        "result": [
            {
                "instrument_name": "BTC-27JUN25-100000-C",
                "mark_iv": 65.0,
                "mark_price": 0.025,
                "underlying_price": 60000.0,
            }
        ]
    })
    client._get = fake  # type: ignore[assignment]

    summaries = await client.get_book_summary_by_currency(currency="BTC", kind="option")

    assert isinstance(summaries, list)
    assert summaries[0]["instrument_name"] == "BTC-27JUN25-100000-C"
    assert fake.sent[0]["path"] == "/public/get_book_summary_by_currency"
    assert fake.sent[0]["params"] == {"currency": "BTC", "kind": "option"}


@pytest.mark.asyncio
async def test_get_index_price_returns_value():
    client = DeribitPublicClient()
    fake = _fake_transport({"result": {"index_price": 60123.45, "estimated_delivery_price": 60000.0}})
    client._get = fake  # type: ignore[assignment]

    index = await client.get_index_price(index_name="btc_usd")

    assert index == pytest.approx(60123.45)
    assert fake.sent[0]["path"] == "/public/get_index_price"
    assert fake.sent[0]["params"] == {"index_name": "btc_usd"}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_book_summary_cache_hit_within_ttl():
    """Two consecutive calls inside the TTL must hit the wire only once."""
    client = DeribitPublicClient()
    fake = _fake_transport({"result": [{"instrument_name": "BTC-X", "mark_iv": 70.0}]})
    client._get = fake  # type: ignore[assignment]

    first = await client.get_book_summary_by_currency("BTC", "option")
    second = await client.get_book_summary_by_currency("BTC", "option")

    assert first == second
    assert len(fake.sent) == 1


@pytest.mark.asyncio
async def test_book_summary_cache_expires():
    """After TTL the next call must re-hit the wire."""
    client = DeribitPublicClient(cache_ttl_seconds=900)
    fake = _fake_transport({"result": [{"instrument_name": "BTC-X", "mark_iv": 70.0}]})
    client._get = fake  # type: ignore[assignment]

    await client.get_book_summary_by_currency("BTC", "option")
    # Force the cached entry to look stale.
    cache_key = ("book_summary", "BTC", "option")
    ts, payload = client._cache[cache_key]
    client._cache[cache_key] = (ts - 10_000, payload)

    fake2 = _fake_transport({"result": [{"instrument_name": "BTC-Y", "mark_iv": 80.0}]})
    client._get = fake2  # type: ignore[assignment]

    refreshed = await client.get_book_summary_by_currency("BTC", "option")
    assert refreshed[0]["instrument_name"] == "BTC-Y"
    assert len(fake2.sent) == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_instruments_returns_empty_list_on_missing_result():
    """A malformed response (no 'result' key) must NOT crash — return [] and log."""
    client = DeribitPublicClient()

    async def _empty(path, params=None):
        return {}

    client._get = _empty  # type: ignore[assignment]
    assert await client.get_instruments(currency="BTC", kind="option") == []


# ---------------------------------------------------------------------------
# Mark price history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_mark_price_history_returns_close_series():
    """The mark price history endpoint returns [[ts_ms, price], ...].
    The client must extract the price column as a list of floats."""
    client = DeribitPublicClient()
    fake = _fake_transport({
        "result": [
            [1745107200000, 60000.0],
            [1745193600000, 60500.0],
            [1745280000000, 61000.0],
            [1745366400000, 60750.0],
        ]
    })
    client._get = fake  # type: ignore[assignment]

    closes = await client.get_mark_price_history(
        instrument_name="BTC-PERPETUAL",
        start_timestamp_ms=1745107200000,
        end_timestamp_ms=1745366400000,
        resolution_seconds=86400,
    )
    assert closes == [60000.0, 60500.0, 61000.0, 60750.0]
    assert fake.sent[0]["path"] == "/public/get_mark_price_history"
    assert fake.sent[0]["params"]["instrument_name"] == "BTC-PERPETUAL"
    assert fake.sent[0]["params"]["resolution"] == "86400"


@pytest.mark.asyncio
async def test_get_mark_price_history_handles_empty_result():
    client = DeribitPublicClient()
    fake = _fake_transport({"result": []})
    client._get = fake  # type: ignore[assignment]
    closes = await client.get_mark_price_history(
        instrument_name="BTC-PERPETUAL",
        start_timestamp_ms=0,
        end_timestamp_ms=0,
    )
    assert closes == []


@pytest.mark.asyncio
async def test_get_mark_price_history_caches_within_ttl():
    """Two consecutive calls with the same args inside the TTL hit the wire once."""
    client = DeribitPublicClient()
    fake = _fake_transport({"result": [[1, 60000.0]]})
    client._get = fake  # type: ignore[assignment]

    a = await client.get_mark_price_history("BTC-PERPETUAL", 0, 1, 86400)
    b = await client.get_mark_price_history("BTC-PERPETUAL", 0, 1, 86400)
    assert a == b
    assert len(fake.sent) == 1


@pytest.mark.asyncio
async def test_get_mark_price_history_returns_empty_on_malformed_payload():
    client = DeribitPublicClient()

    async def _bad(path, params=None):
        return {"result": "unexpected"}

    client._get = _bad  # type: ignore[assignment]
    closes = await client.get_mark_price_history("BTC-PERPETUAL", 0, 0)
    assert closes == []
