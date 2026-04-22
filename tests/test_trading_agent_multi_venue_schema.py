"""Regression tests for the perps TradingAgent contract.

The perps agent is PERPS-ONLY since the dual-agent separation. These
tests lock in that no options strategies, legs, or Thalex fields appear
in its prompt or its structured-output schema — cross-venue contamination
at the LLM layer (the perps LLM emitting an options decision without
``venue='thalex'``) was the root cause of options proposals silently
getting routed through the perps execution path and never surfacing as
actionable options proposals.
"""

import json

from src.backend.agent.decision_maker import TradingAgent


_OPTIONS_STRATEGY_NAMES = [
    "credit_put_spread",
    "credit_call_spread",
    "iron_condor",
    "long_call_delta_hedged",
    "long_put_delta_hedged",
    "vol_arb",
]


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _capture_payload(monkeypatch):
    """Drive the agent once with a stub LLM and return the first request payload."""
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
                                    "allocation_usd": 0.0,
                                    "tp_price": None,
                                    "sl_price": None,
                                    "exit_plan": "",
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
    return sent_payloads[0]


def test_perps_prompt_contains_no_options_strategies(monkeypatch):
    """The perps system prompt must not mention any options strategy by name.

    Cross-contamination happens because the LLM sees the strategy menu and
    emits an options decision without setting ``venue='thalex'``. Keeping the
    menu out of the perps prompt eliminates the temptation at the source.
    """
    payload = _capture_payload(monkeypatch)
    system_prompt = payload["messages"][0]["content"]
    for name in _OPTIONS_STRATEGY_NAMES:
        assert name not in system_prompt, (
            f"perps agent prompt leaked options strategy {name!r} — this agent "
            "is perps-only after the dual-agent separation"
        )
    # The prompt is allowed (in fact, required) to have a negative guardrail
    # telling the LLM NOT to emit options fields. What we check is that no
    # strategy *menu* appears — the menu is the thing that tempts the LLM
    # to emit an options decision in the first place.


def test_perps_schema_has_no_options_fields(monkeypatch):
    """Structured-output schema enforces the perps-only contract provider-side.

    Even if the prompt regresses, ``additionalProperties: false`` + a
    perps-only field list makes the provider reject any options response —
    a hard backstop so bad output can never reach the bot engine.
    """
    payload = _capture_payload(monkeypatch)
    schema = payload["response_format"]["json_schema"]["schema"]
    item_schema = schema["properties"]["trade_decisions"]["items"]
    props = item_schema["properties"]

    # No options fields present.
    forbidden = {"venue", "strategy", "legs", "contracts", "kind",
                 "tenor_days", "target_strike", "target_delta", "underlying"}
    leaked = forbidden.intersection(props.keys())
    assert not leaked, f"perps schema leaked options fields: {leaked}"

    # additionalProperties must be locked so the LLM can't sneak fields in.
    assert item_schema["additionalProperties"] is False

    # Core perps fields still required.
    required = set(item_schema["required"])
    assert {"asset", "action", "allocation_usd", "tp_price", "sl_price",
            "exit_plan", "rationale"}.issubset(required)
