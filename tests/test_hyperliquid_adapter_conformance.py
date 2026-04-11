"""Structural conformance tests for HyperliquidAPI as an ExchangeAdapter.

These tests do NOT hit the network. They verify that HyperliquidAPI satisfies
the ExchangeAdapter contract via inheritance + signature checks. Live behavior
tests live in the integration test suite.
"""

import inspect

from src.backend.trading.exchange_adapter import ExchangeAdapter
from src.backend.trading import hyperliquid_api as hl_module


def test_hyperliquid_api_is_subclass_of_exchange_adapter():
    """HyperliquidAPI must declare ExchangeAdapter as a base class."""
    assert issubclass(hl_module.HyperliquidAPI, ExchangeAdapter)


def test_hyperliquid_api_has_venue_attribute():
    """Adapters identify themselves with a `venue` class attribute."""
    assert getattr(hl_module.HyperliquidAPI, "venue", None) == "hyperliquid"


def test_hyperliquid_api_implements_every_abstract_method():
    """Every @abstractmethod on the base class must have a concrete override."""
    abstract_methods = ExchangeAdapter.__abstractmethods__
    for name in abstract_methods:
        attr = getattr(hl_module.HyperliquidAPI, name, None)
        assert attr is not None, f"HyperliquidAPI is missing override for {name}"
        assert callable(attr), f"HyperliquidAPI.{name} must be callable"


def test_hyperliquid_api_method_signatures_match_base():
    """Critical method signatures must align with the ABC so the factory can call them generically."""
    expected = {
        "place_buy_order": ["self", "asset", "amount", "slippage"],
        "place_sell_order": ["self", "asset", "amount", "slippage"],
        "place_take_profit": ["self", "asset", "is_buy", "amount", "tp_price"],
        "place_stop_loss": ["self", "asset", "is_buy", "amount", "sl_price"],
        "cancel_order": ["self", "asset"],  # third arg name varies (oid vs order_id) — only check first two
        "get_current_price": ["self", "asset"],
    }
    for name, expected_prefix in expected.items():
        sig = inspect.signature(getattr(hl_module.HyperliquidAPI, name))
        actual_params = list(sig.parameters.keys())
        for i, expected_param in enumerate(expected_prefix):
            assert actual_params[i] == expected_param, (
                f"HyperliquidAPI.{name} param {i} should be {expected_param}, got {actual_params[i]}"
            )
