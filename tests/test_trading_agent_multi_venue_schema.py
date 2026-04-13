"""Regression tests for the multi-venue TradingAgent prompt/schema contract."""

import json

from src.backend.agent.decision_maker import TradingAgent


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_trading_agent_uses_current_thalex_strategy_names(monkeypatch):
    sent_payloads = []

    def _fake_post(url, headers, json, timeout):
        sent_payloads.append(json)
        return _FakeResponse({
            "choices": [
                {
                    "message": {
                        "parsed": {
                            "reasoning": "flat",
                            "trade_decisions": [
                                {
                                    "asset": "BTC",
                                    "action": "hold",
                                    "venue": "hyperliquid",
                                    "allocation_usd": 0.0,
                                    "tp_price": None,
                                    "sl_price": None,
                                    "exit_plan": "",
                                    "strategy": None,
                                    "underlying": None,
                                    "kind": None,
                                    "tenor_days": None,
                                    "target_strike": None,
                                    "target_delta": None,
                                    "contracts": None,
                                    "legs": None,
                                    "rationale": "wait",
                                }
                            ],
                        }
                    }
                }
            ]
        })

    monkeypatch.setattr("requests.post", _fake_post)

    agent = TradingAgent()
    result = agent.decide_trade(["BTC"], "test context")

    assert result["trade_decisions"][0]["action"] == "hold"
    payload = sent_payloads[0]
    system_prompt = payload["messages"][0]["content"]
    strategy_enum = payload["response_format"]["json_schema"]["schema"]["properties"]["trade_decisions"]["items"]["properties"]["strategy"]["enum"]

    assert "credit_put_spread" in system_prompt
    assert "credit_call_spread" in system_prompt
    assert "iron_condor" in system_prompt
    assert "credit_put" not in strategy_enum
    assert "credit_spread" not in strategy_enum
    assert "credit_put_spread" in strategy_enum
    assert "credit_call_spread" in strategy_enum
    assert "iron_condor" in strategy_enum
    assert "vol_arb" in strategy_enum
