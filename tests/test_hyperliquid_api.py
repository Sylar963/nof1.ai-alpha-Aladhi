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
