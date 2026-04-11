"""Tests for the multi-venue Option B decision schema.

The decision schema is the contract between the LLM and the trading layer.
It must accept the legacy Hyperliquid spot/perp shape AND the new options
intent shape, with strict validation that catches malformed payloads before
they reach an exchange adapter."""

import pytest

from src.backend.agent.decision_schema import (
    DecisionParseError,
    OptionsLeg,
    TradeDecision,
    parse_decision,
)


def test_parse_legacy_hyperliquid_decision():
    """A pre-existing Hyperliquid decision payload must still parse cleanly."""
    payload = {
        "asset": "BTC",
        "action": "buy",
        "allocation_usd": 500.0,
        "tp_price": 65000.0,
        "sl_price": 58000.0,
        "exit_plan": "close on 4h close below EMA50",
        "rationale": "trend continuation",
    }
    decision = parse_decision(payload)
    assert decision.venue == "hyperliquid"  # default
    assert decision.action == "buy"
    assert decision.allocation_usd == 500.0
    assert decision.strategy is None
    assert decision.legs == []


def test_parse_thalex_long_call_intent():
    payload = {
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.05,
        "rationale": "implied vol cheap, want directional exposure with hedge",
    }
    decision = parse_decision(payload)
    assert decision.venue == "thalex"
    assert decision.strategy == "long_call_delta_hedged"
    assert decision.kind == "call"
    assert decision.tenor_days == 30
    assert decision.contracts == 0.05


def test_parse_thalex_credit_put_spread():
    """Defined-risk short put: sell near-money put + buy further OTM put."""
    payload = {
        "venue": "thalex",
        "asset": "BTC",
        "action": "sell",
        "strategy": "credit_put_spread",
        "underlying": "BTC",
        "tenor_days": 14,
        "vol_view": "short_vol",
        "legs": [
            {"kind": "put", "target_strike": 55000, "side": "sell", "contracts": 0.1},
            {"kind": "put", "target_strike": 50000, "side": "buy", "contracts": 0.1},
        ],
        "rationale": "selling premium below support, defined risk",
    }
    decision = parse_decision(payload)
    assert decision.strategy == "credit_put_spread"
    assert decision.action == "sell"
    assert decision.vol_view == "short_vol"
    assert len(decision.legs) == 2
    assert decision.legs[0].side == "sell"
    assert decision.legs[1].side == "buy"


def test_parse_thalex_credit_call_spread():
    """Defined-risk short call: sell near-money call + buy further OTM call."""
    payload = {
        "venue": "thalex",
        "asset": "BTC",
        "action": "sell",
        "strategy": "credit_call_spread",
        "underlying": "BTC",
        "tenor_days": 14,
        "legs": [
            {"kind": "call", "target_strike": 65000, "side": "sell", "contracts": 0.1},
            {"kind": "call", "target_strike": 70000, "side": "buy", "contracts": 0.1},
        ],
        "rationale": "selling upside premium",
    }
    decision = parse_decision(payload)
    assert decision.strategy == "credit_call_spread"
    assert all(leg.kind == "call" for leg in decision.legs)


def test_parse_thalex_iron_condor_with_four_legs():
    """The vol-crush play: sell put spread AND sell call spread simultaneously."""
    payload = {
        "venue": "thalex",
        "asset": "BTC",
        "action": "sell",
        "strategy": "iron_condor",
        "underlying": "BTC",
        "tenor_days": 14,
        "vol_view": "short_vol",
        "legs": [
            {"kind": "put", "target_strike": 55000, "side": "sell", "contracts": 0.1},
            {"kind": "put", "target_strike": 50000, "side": "buy", "contracts": 0.1},
            {"kind": "call", "target_strike": 65000, "side": "sell", "contracts": 0.1},
            {"kind": "call", "target_strike": 70000, "side": "buy", "contracts": 0.1},
        ],
        "rationale": "vol crush after event, defined risk both sides",
    }
    decision = parse_decision(payload)
    assert decision.strategy == "iron_condor"
    assert len(decision.legs) == 4
    sells = [leg for leg in decision.legs if leg.side == "sell"]
    buys = [leg for leg in decision.legs if leg.side == "buy"]
    assert len(sells) == 2
    assert len(buys) == 2


def test_parse_legacy_credit_put_strategy_is_rejected():
    """The single-leg `credit_put` is no longer valid — defined risk only."""
    with pytest.raises(DecisionParseError, match="strategy"):
        parse_decision({
            "venue": "thalex",
            "asset": "BTC",
            "action": "sell",
            "strategy": "credit_put",
            "underlying": "BTC",
            "kind": "put",
            "tenor_days": 14,
            "target_strike": 55000,
            "contracts": 0.1,
            "rationale": "x",
        })


def test_parse_per_leg_tenor_days_for_calendar_spread():
    """Calendar spreads use different expiries per leg via leg.tenor_days."""
    payload = {
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "vol_arb",
        "underlying": "BTC",
        "entry_kind": "calendar",
        "legs": [
            {"kind": "call", "side": "sell", "target_strike": 60000, "contracts": 0.05, "tenor_days": 7},
            {"kind": "call", "side": "buy", "target_strike": 60000, "contracts": 0.05, "tenor_days": 30},
        ],
        "rationale": "term structure dislocation",
    }
    decision = parse_decision(payload)
    assert decision.entry_kind == "calendar"
    assert decision.legs[0].tenor_days == 7
    assert decision.legs[1].tenor_days == 30


def test_parse_target_gamma_btc_field():
    """Multi-tenor sizing flag — LLM declares target gamma in BTC."""
    payload = {
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "target_gamma_btc": 0.005,
        "rationale": "build long gamma across tenors",
    }
    decision = parse_decision(payload)
    assert decision.target_gamma_btc == pytest.approx(0.005)


def test_parse_vol_view_must_be_valid_enum():
    with pytest.raises(DecisionParseError, match="vol_view"):
        parse_decision({
            "venue": "thalex",
            "asset": "BTC",
            "action": "sell",
            "strategy": "iron_condor",
            "underlying": "BTC",
            "tenor_days": 14,
            "vol_view": "moonshot",
            "legs": [
                {"kind": "put", "target_strike": 55000, "side": "sell", "contracts": 0.1},
                {"kind": "put", "target_strike": 50000, "side": "buy", "contracts": 0.1},
            ],
            "rationale": "x",
        })


def test_parse_entry_kind_must_be_valid_enum():
    with pytest.raises(DecisionParseError, match="entry_kind"):
        parse_decision({
            "venue": "thalex",
            "asset": "BTC",
            "action": "buy",
            "strategy": "long_call_delta_hedged",
            "underlying": "BTC",
            "kind": "call",
            "tenor_days": 30,
            "target_strike": 65000,
            "entry_kind": "warpdrive",
            "rationale": "x",
        })


def test_parse_rejects_unknown_venue():
    with pytest.raises(DecisionParseError, match="venue"):
        parse_decision({
            "venue": "atlantis",
            "asset": "BTC",
            "action": "buy",
            "rationale": "x",
        })


def test_parse_rejects_unknown_strategy():
    with pytest.raises(DecisionParseError, match="strategy"):
        parse_decision({
            "venue": "thalex",
            "asset": "BTC",
            "action": "buy",
            "strategy": "moonshot",
            "rationale": "x",
        })


def test_parse_thalex_intent_requires_underlying_when_strategy_set():
    with pytest.raises(DecisionParseError, match="underlying"):
        parse_decision({
            "venue": "thalex",
            "asset": "BTC",
            "action": "buy",
            "strategy": "long_call_delta_hedged",
            "kind": "call",
            "tenor_days": 30,
            "contracts": 0.05,
            "rationale": "x",
        })


def test_decision_to_option_intent_for_single_leg():
    """A single-leg options decision must produce a usable OptionIntent."""
    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.05,
        "rationale": "x",
    })
    intent = decision.to_option_intent()
    assert intent is not None
    assert intent.underlying == "BTC"
    assert intent.kind == "call"
    assert intent.tenor_days == 30
    assert intent.target_strike == 65000


def test_decision_to_option_intent_returns_none_for_hyperliquid():
    decision = parse_decision({
        "asset": "BTC",
        "action": "buy",
        "allocation_usd": 500.0,
        "rationale": "x",
    })
    assert decision.to_option_intent() is None


def test_action_must_be_buy_sell_or_hold():
    with pytest.raises(DecisionParseError, match="action"):
        parse_decision({"asset": "BTC", "action": "moon", "rationale": "x"})


def test_hold_decision_is_always_valid():
    decision = parse_decision({"asset": "BTC", "action": "hold", "rationale": "wait"})
    assert decision.action == "hold"
