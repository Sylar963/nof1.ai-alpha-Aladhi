"""Tests for ThalexAPI account-state loading."""

from types import SimpleNamespace

import pytest

from src.backend.trading.thalex_api import ThalexAPI


@pytest.mark.asyncio
async def test_get_user_state_connects_before_accessing_client_methods(monkeypatch, tmp_path):
    monkeypatch.setenv("THALEX_NETWORK", "test")
    monkeypatch.setenv("THALEX_KEY_ID", "test-kid")
    monkeypatch.setenv("THALEX_PRIVATE_KEY_PATH", str(tmp_path / "fake.pem"))

    adapter = ThalexAPI()
    calls = []

    def _account_summary(**kwargs):
        return None

    def _portfolio(**kwargs):
        return None

    fake_client = SimpleNamespace(
        account_summary=_account_summary,
        portfolio=_portfolio,
    )

    async def _fake_connect():
        calls.append("connect")
        adapter.connected = True
        adapter._client = fake_client

    async def _fake_request(sender, **kwargs):
        if sender is fake_client.account_summary:
            calls.append("account_summary")
            return {"equity": 123.0, "portfolio_value": 130.0}
        if sender is fake_client.portfolio:
            calls.append("portfolio")
            return []
        raise AssertionError(f"unexpected sender: {sender!r}")

    adapter.connected = False
    adapter._client = None
    monkeypatch.setattr(adapter, "connect", _fake_connect)
    monkeypatch.setattr(adapter, "_request", _fake_request)

    state = await adapter.get_user_state()

    assert calls == ["connect", "account_summary", "portfolio"]
    assert state.balance == 123.0
    assert state.total_value == 130.0
    assert state.positions == []
