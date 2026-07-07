"""Tests for Hyperliquid client bootstrap behavior."""

import asyncio
from unittest.mock import AsyncMock

import pytest
import requests

from hyperliquid.utils import constants

META_AND_CTXS = [
    {"universe": [{"name": "BTC", "szDecimals": 5}, {"name": "COARSE", "szDecimals": 0}]},
    [{}, {}],
]


@pytest.fixture
def hl_module():
    import src.backend.trading.hyperliquid_api as module
    return module


@pytest.fixture
def make_adapter(monkeypatch, hl_module):
    def _make(info_cls=None, exchange_cls=None):
        class FakeWallet:
            address = "0x123"

        class DefaultInfo:
            def __init__(self, *args, **kwargs):
                pass

            def meta_and_asset_ctxs(self):
                return META_AND_CTXS

        class DefaultExchange:
            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setitem(hl_module.CONFIG, "hyperliquid_private_key", "0xabc")
        monkeypatch.setitem(hl_module.CONFIG, "mnemonic", None)
        monkeypatch.setitem(hl_module.CONFIG, "hyperliquid_base_url", None)
        monkeypatch.setitem(hl_module.CONFIG, "hyperliquid_network", "mainnet")
        monkeypatch.setattr(hl_module.Account, "from_key", lambda _: FakeWallet())
        monkeypatch.setattr(hl_module, "Info", info_cls or DefaultInfo)
        monkeypatch.setattr(hl_module, "Exchange", exchange_cls or DefaultExchange)
        return hl_module.HyperliquidAPI()

    return _make


def test_hyperliquid_api_build_clients_uses_safe_spot_meta(monkeypatch):
    import src.backend.trading.hyperliquid_api as hyperliquid_api_module

    info_calls = []
    exchange_calls = []

    class FakeWallet:
        address = "0x123"

    class FakeInfo:
        def __init__(self, base_url=None, skip_ws=False, meta=None, spot_meta=None, perp_dexs=None, timeout=None):
            info_calls.append(
                {
                    "base_url": base_url,
                    "skip_ws": skip_ws,
                    "spot_meta": spot_meta,
                }
            )

    class FakeExchange:
        def __init__(
            self,
            wallet,
            base_url=None,
            meta=None,
            vault_address=None,
            account_address=None,
            spot_meta=None,
            perp_dexs=None,
            timeout=None,
        ):
            exchange_calls.append(
                {
                    "wallet": wallet,
                    "base_url": base_url,
                    "spot_meta": spot_meta,
                }
            )

    monkeypatch.setitem(hyperliquid_api_module.CONFIG, "hyperliquid_private_key", "0xabc")
    monkeypatch.setitem(hyperliquid_api_module.CONFIG, "mnemonic", None)
    monkeypatch.setitem(hyperliquid_api_module.CONFIG, "hyperliquid_base_url", None)
    monkeypatch.setitem(hyperliquid_api_module.CONFIG, "hyperliquid_network", "mainnet")
    monkeypatch.setattr(hyperliquid_api_module.Account, "from_key", lambda _: FakeWallet())
    monkeypatch.setattr(hyperliquid_api_module, "Info", FakeInfo)
    monkeypatch.setattr(hyperliquid_api_module, "Exchange", FakeExchange)

    hyperliquid_api_module.HyperliquidAPI()

    assert info_calls == [
        {
            "base_url": constants.MAINNET_API_URL,
            "skip_ws": True,
            "spot_meta": {"universe": [], "tokens": []},
        }
    ]
    assert len(exchange_calls) == 1
    assert exchange_calls[0]["wallet"].address == "0x123"
    assert exchange_calls[0]["base_url"] == constants.MAINNET_API_URL
    assert exchange_calls[0]["spot_meta"] == {"universe": [], "tokens": []}


@pytest.mark.asyncio
async def test_hyperliquid_get_user_state_prefers_account_value_for_balance(monkeypatch):
    import src.backend.trading.hyperliquid_api as hyperliquid_api_module

    class FakeWallet:
        address = "0x123"

    class FakeInfo:
        def __init__(self, *args, **kwargs):
            pass

        def user_state(self, address):
            assert address == "0x123"
            return {
                "accountValue": 100.0,
                "withdrawable": 0.0,
                "assetPositions": [],
            }

    class FakeExchange:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setitem(hyperliquid_api_module.CONFIG, "hyperliquid_private_key", "0xabc")
    monkeypatch.setitem(hyperliquid_api_module.CONFIG, "mnemonic", None)
    monkeypatch.setitem(hyperliquid_api_module.CONFIG, "hyperliquid_base_url", None)
    monkeypatch.setitem(hyperliquid_api_module.CONFIG, "hyperliquid_network", "mainnet")
    monkeypatch.setattr(hyperliquid_api_module.Account, "from_key", lambda _: FakeWallet())
    monkeypatch.setattr(hyperliquid_api_module, "Info", FakeInfo)
    monkeypatch.setattr(hyperliquid_api_module, "Exchange", FakeExchange)

    adapter = hyperliquid_api_module.HyperliquidAPI()

    state = await adapter.get_user_state()

    assert state == {"balance": 100.0, "total_value": 100.0, "positions": []}


@pytest.mark.asyncio
async def test_hyperliquid_get_user_state_falls_back_to_margin_summary_account_value(monkeypatch):
    import src.backend.trading.hyperliquid_api as hyperliquid_api_module

    class FakeWallet:
        address = "0x123"

    class FakeInfo:
        def __init__(self, *args, **kwargs):
            pass

        def user_state(self, address):
            assert address == "0x123"
            return {
                "withdrawable": 0.0,
                "marginSummary": {"accountValue": "272.470437"},
                "crossMarginSummary": {"accountValue": "272.470437"},
                "assetPositions": [],
            }

    class FakeExchange:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setitem(hyperliquid_api_module.CONFIG, "hyperliquid_private_key", "0xabc")
    monkeypatch.setitem(hyperliquid_api_module.CONFIG, "mnemonic", None)
    monkeypatch.setitem(hyperliquid_api_module.CONFIG, "hyperliquid_base_url", None)
    monkeypatch.setitem(hyperliquid_api_module.CONFIG, "hyperliquid_network", "mainnet")
    monkeypatch.setattr(hyperliquid_api_module.Account, "from_key", lambda _: FakeWallet())
    monkeypatch.setattr(hyperliquid_api_module, "Info", FakeInfo)
    monkeypatch.setattr(hyperliquid_api_module, "Exchange", FakeExchange)

    adapter = hyperliquid_api_module.HyperliquidAPI()

    state = await adapter.get_user_state()

    assert state == {"balance": 272.470437, "total_value": 272.470437, "positions": []}


# ---------------------------------------------------------------------------
# parse_order_response — silent-failure guard against phantom trades
# ---------------------------------------------------------------------------


def test_parse_order_response_accepts_filled_status():
    from src.backend.trading.hyperliquid_api import HyperliquidAPI

    ok, reason = HyperliquidAPI.parse_order_response(
        {"status": "ok", "response": {"type": "order", "data": {"statuses": [
            {"filled": {"oid": 1, "totalSz": "0.01", "avgPx": "78000"}}
        ]}}}
    )
    assert ok is True
    assert reason == ""


def test_parse_order_response_accepts_resting_status():
    from src.backend.trading.hyperliquid_api import HyperliquidAPI

    ok, _ = HyperliquidAPI.parse_order_response(
        {"status": "ok", "response": {"type": "order", "data": {"statuses": [
            {"resting": {"oid": 42}}
        ]}}}
    )
    assert ok is True


def test_parse_order_response_flags_insufficient_margin():
    """The exact shape the venue returns when margin is insufficient —
    top-level ``ok`` but a per-order ``error`` entry. The naive ``if result``
    check passes this silently, which is the bug this helper exists to kill.
    """
    from src.backend.trading.hyperliquid_api import HyperliquidAPI

    ok, reason = HyperliquidAPI.parse_order_response(
        {"status": "ok", "response": {"type": "order", "data": {"statuses": [
            {"error": "Insufficient margin to place order."}
        ]}}}
    )
    assert ok is False
    assert "Insufficient margin" in reason


def test_parse_order_response_flags_top_level_err():
    from src.backend.trading.hyperliquid_api import HyperliquidAPI

    ok, reason = HyperliquidAPI.parse_order_response(
        {"status": "err", "response": "price outside tick"}
    )
    assert ok is False
    assert "tick" in reason


def test_parse_order_response_rejects_non_dict():
    from src.backend.trading.hyperliquid_api import HyperliquidAPI

    ok, _ = HyperliquidAPI.parse_order_response(None)
    assert ok is False
    ok, _ = HyperliquidAPI.parse_order_response("ok")
    assert ok is False


def test_parse_order_response_rejects_empty_statuses():
    from src.backend.trading.hyperliquid_api import HyperliquidAPI

    ok, _ = HyperliquidAPI.parse_order_response(
        {"status": "ok", "response": {"type": "order", "data": {"statuses": []}}}
    )
    assert ok is False


# ---------------------------------------------------------------------------
# get_free_margin_info — crossMarginSummary fallback and failure semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_free_margin_info_falls_back_to_cross_margin_summary(make_adapter):
    class FakeInfo:
        def __init__(self, *args, **kwargs):
            pass

        def user_state(self, address):
            return {
                "withdrawable": "1000.5",
                "crossMarginSummary": {"accountValue": "303000.0", "totalMarginUsed": "3000.0"},
                "assetPositions": [],
            }

    adapter = make_adapter(info_cls=FakeInfo)
    info = await adapter.get_free_margin_info()

    assert info["withdrawable"] == pytest.approx(1000.5)
    assert info["account_value"] == pytest.approx(303000.0)
    assert info["total_margin_used"] == pytest.approx(3000.0)
    assert info["free_margin"] == pytest.approx(300000.0)


@pytest.mark.asyncio
async def test_free_margin_info_prefers_margin_summary_when_present(make_adapter):
    class FakeInfo:
        def __init__(self, *args, **kwargs):
            pass

        def user_state(self, address):
            return {
                "withdrawable": "10.0",
                "marginSummary": {"accountValue": "100.0", "totalMarginUsed": "40.0"},
                "crossMarginSummary": {"accountValue": "999.0", "totalMarginUsed": "1.0"},
            }

    adapter = make_adapter(info_cls=FakeInfo)
    info = await adapter.get_free_margin_info()

    assert info["account_value"] == pytest.approx(100.0)
    assert info["total_margin_used"] == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_free_margin_info_raises_on_non_dict_payload(make_adapter):
    class FakeInfo:
        def __init__(self, *args, **kwargs):
            pass

        def user_state(self, address):
            return None

    adapter = make_adapter(info_cls=FakeInfo)
    with pytest.raises(RuntimeError):
        await adapter.get_free_margin_info()


@pytest.mark.asyncio
async def test_free_margin_info_propagates_rpc_failure(make_adapter, monkeypatch):
    class FakeInfo:
        def __init__(self, *args, **kwargs):
            pass

        def user_state(self, address):
            raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    adapter = make_adapter(info_cls=FakeInfo)
    with pytest.raises(requests.exceptions.ConnectionError):
        await adapter.get_free_margin_info()


# ---------------------------------------------------------------------------
# get_meta_and_ctxs — TTL cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_meta_cache_refetches_after_ttl(make_adapter, monkeypatch, hl_module):
    calls = []

    class FakeInfo:
        def __init__(self, *args, **kwargs):
            pass

        def meta_and_asset_ctxs(self):
            calls.append(1)
            return META_AND_CTXS

    clock = [1000.0]
    monkeypatch.setattr(hl_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setitem(hl_module.CONFIG, "hl_meta_cache_ttl_seconds", 60)

    adapter = make_adapter(info_cls=FakeInfo)
    await adapter.get_meta_and_ctxs()
    await adapter.get_meta_and_ctxs()
    assert len(calls) == 1

    clock[0] += 61
    await adapter.get_meta_and_ctxs()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_meta_cache_serves_stale_copy_when_refresh_fails(make_adapter, monkeypatch, hl_module):
    calls = []

    class FakeInfo:
        def __init__(self, *args, **kwargs):
            pass

        def meta_and_asset_ctxs(self):
            calls.append(1)
            if len(calls) > 1:
                raise requests.exceptions.ConnectionError("down")
            return META_AND_CTXS

    clock = [1000.0]
    monkeypatch.setattr(hl_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setitem(hl_module.CONFIG, "hl_meta_cache_ttl_seconds", 60)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    adapter = make_adapter(info_cls=FakeInfo)
    first = await adapter.get_meta_and_ctxs()
    clock[0] += 120
    stale = await adapter.get_meta_and_ctxs()
    assert stale == first


# ---------------------------------------------------------------------------
# get_recent_fills — most-recent-first ordering and field passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recent_fills_returns_newest_first_and_keeps_fields(make_adapter):
    class FakeInfo:
        def __init__(self, *args, **kwargs):
            pass

        def user_fills(self, address):
            return [
                {"time": 300, "coin": "BTC", "closedPnl": "3.0", "fee": "0.3", "dir": "Close Long"},
                {"time": 100, "coin": "BTC", "closedPnl": "1.0", "fee": "0.1", "dir": "Open Long"},
                {"time": 200, "coin": "ETH", "closedPnl": "2.0", "fee": "0.2", "dir": "Open Short"},
            ]

    adapter = make_adapter(info_cls=FakeInfo)
    fills = await adapter.get_recent_fills(limit=2)

    assert [f["time"] for f in fills] == [300, 200]
    assert fills[0]["closedPnl"] == "3.0"
    assert fills[0]["fee"] == "0.3"
    assert fills[0]["dir"] == "Close Long"


# ---------------------------------------------------------------------------
# round_price / round_size — HL tick and size precision rules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_round_price_caps_five_significant_figures(make_adapter):
    adapter = make_adapter()
    assert await adapter.round_price("BTC", 104321.77) == pytest.approx(104320.0)


@pytest.mark.asyncio
async def test_round_price_allows_integer_prices(make_adapter):
    adapter = make_adapter()
    assert await adapter.round_price("BTC", 104321.0) == pytest.approx(104321.0)


@pytest.mark.asyncio
async def test_round_price_respects_sz_decimals_cap(make_adapter):
    adapter = make_adapter()
    assert await adapter.round_price("BTC", 3.1234567) == pytest.approx(3.1)
    assert await adapter.round_price("COARSE", 0.00123456) == pytest.approx(0.001235)


@pytest.mark.asyncio
async def test_round_size_floors_instead_of_rounding_up(make_adapter):
    adapter = make_adapter()
    assert await adapter.round_size("COARSE", 1.999999) == pytest.approx(1.0)
    assert await adapter.round_size("BTC", 0.123456789) == pytest.approx(0.12345)
    assert await adapter.round_size("BTC", 0.29) == pytest.approx(0.29)


@pytest.mark.asyncio
async def test_round_size_fetches_meta_when_cache_empty(make_adapter):
    adapter = make_adapter()
    assert adapter._meta_cache is None
    assert await adapter.round_size("COARSE", 2.7) == pytest.approx(2.0)
    assert adapter._meta_cache == META_AND_CTXS


# ---------------------------------------------------------------------------
# slippage config, reduce-only close, and retry semantics
# ---------------------------------------------------------------------------


def _make_recording_exchange(record, response=None, exc=None):
    class FakeExchange:
        def __init__(self, *args, **kwargs):
            pass

        def market_open(self, name, is_buy, sz, px, slippage, *args, **kwargs):
            record.append({"name": name, "is_buy": is_buy, "sz": sz, "slippage": slippage})
            if exc:
                raise exc
            return response

        def market_close(self, coin, sz=None, px=None, slippage=None, *args, **kwargs):
            record.append({"coin": coin, "sz": sz, "slippage": slippage})
            if exc:
                raise exc
            return response

    return FakeExchange


FILLED_RESPONSE = {
    "status": "ok",
    "response": {"type": "order", "data": {"statuses": [{"filled": {"oid": 7, "totalSz": "0.1", "avgPx": "100"}}]}},
}


@pytest.mark.asyncio
async def test_place_buy_order_uses_config_slippage(make_adapter, monkeypatch, hl_module):
    record = []
    monkeypatch.setitem(hl_module.CONFIG, "hl_max_slippage", 0.007)
    adapter = make_adapter(exchange_cls=_make_recording_exchange(record, FILLED_RESPONSE))

    await adapter.place_buy_order("BTC", 0.1)
    assert record[0]["slippage"] == pytest.approx(0.007)


@pytest.mark.asyncio
async def test_place_buy_order_defaults_to_half_percent(make_adapter, monkeypatch, hl_module):
    record = []
    monkeypatch.delitem(hl_module.CONFIG, "hl_max_slippage", raising=False)
    adapter = make_adapter(exchange_cls=_make_recording_exchange(record, FILLED_RESPONSE))

    await adapter.place_buy_order("BTC", 0.1)
    assert record[0]["slippage"] == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_place_sell_order_keeps_explicit_slippage(make_adapter, monkeypatch, hl_module):
    record = []
    monkeypatch.setitem(hl_module.CONFIG, "hl_max_slippage", 0.007)
    adapter = make_adapter(exchange_cls=_make_recording_exchange(record, FILLED_RESPONSE))

    await adapter.place_sell_order("BTC", 0.1, slippage=0.02)
    assert record[0]["slippage"] == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_market_close_position_validates_response(make_adapter):
    record = []
    adapter = make_adapter(exchange_cls=_make_recording_exchange(record, FILLED_RESPONSE))

    result = await adapter.market_close_position("BTC")
    assert result["ok"] is True
    assert result["error"] == ""
    assert result["response"] == FILLED_RESPONSE
    assert record[0]["coin"] == "BTC"
    assert record[0]["sz"] is None


@pytest.mark.asyncio
async def test_market_close_position_flags_missing_position(make_adapter):
    record = []
    adapter = make_adapter(exchange_cls=_make_recording_exchange(record, response=None))

    result = await adapter.market_close_position("BTC")
    assert result["ok"] is False
    assert "no open position" in result["error"]


@pytest.mark.asyncio
async def test_order_placement_does_not_retry_read_timeout(make_adapter, monkeypatch):
    record = []
    adapter = make_adapter(
        exchange_cls=_make_recording_exchange(record, exc=requests.exceptions.ReadTimeout("ambiguous"))
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    with pytest.raises(requests.exceptions.ReadTimeout):
        await adapter.place_buy_order("BTC", 0.1)
    assert len(record) == 1


@pytest.mark.asyncio
async def test_retry_recovers_from_requests_connection_error(make_adapter, monkeypatch):
    adapter = make_adapter()
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise requests.exceptions.ConnectionError("refused")
        return "ok"

    result = await adapter._retry(flaky)
    assert result == "ok"
    assert len(calls) == 2
