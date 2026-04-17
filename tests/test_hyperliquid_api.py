"""Tests for Hyperliquid client bootstrap behavior."""

import pytest

from hyperliquid.utils import constants


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
