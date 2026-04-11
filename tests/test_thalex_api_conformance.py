"""Structural conformance tests for ThalexAPI as an ExchangeAdapter.

These tests verify the class shape and config loading. They do NOT open a
WebSocket connection to Thalex — that lives in the integration suite."""

import pytest

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
