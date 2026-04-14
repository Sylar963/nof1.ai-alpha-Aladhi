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


@pytest.mark.asyncio
async def test_get_user_state_uses_margin_style_summary_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("THALEX_NETWORK", "test")
    monkeypatch.setenv("THALEX_KEY_ID", "test-kid")
    monkeypatch.setenv("THALEX_PRIVATE_KEY_PATH", str(tmp_path / "fake.pem"))

    adapter = ThalexAPI()

    fake_client = SimpleNamespace(
        account_summary=lambda **kwargs: None,
        portfolio=lambda **kwargs: None,
    )

    async def _fake_connect():
        adapter.connected = True
        adapter._client = fake_client

    async def _fake_request(sender, **kwargs):
        if sender is fake_client.account_summary:
            return {
                "cash_collateral": 294650.6325800734,
                "margin": 303182.24282869545,
                "unrealised_pnl": 8531.610248622088,
                "cash": [
                    {"currency": "USDC", "balance": 100000.0, "collateral_index_price": None},
                    {"currency": "BTC", "balance": 1.0, "collateral_index_price": 74629.48216666667},
                    {"currency": "ETH", "balance": 10.0, "collateral_index_price": 2374.478666666667},
                    {"currency": "USDT", "balance": 96276.36374674, "collateral_index_price": None},
                ],
            }
        if sender is fake_client.portfolio:
            return []
        raise AssertionError(f"unexpected sender: {sender!r}")

    adapter.connected = False
    adapter._client = None
    monkeypatch.setattr(adapter, "connect", _fake_connect)
    monkeypatch.setattr(adapter, "_request", _fake_request)

    state = await adapter.get_user_state()

    assert state.balance == pytest.approx(303182.24282869545)
    assert state.total_value == pytest.approx(303182.24282869545)
