"""Prompt-contract tests for the OptionsAgent hedge-failure / cut-loss split.

The screenshot that motivated this refactor showed the OptionsAgent cutting
a long_call_delta_hedged position because the perp hedge could not be sized
(HL free margin = 0). The rule wiring now says: hedge unavailability is a
HOLD + risk_flags=["hedge_unavailable"], not a cut-loss. These tests lock
that contract at the prompt level so future edits cannot silently revert
the behavior.
"""

from __future__ import annotations

import pytest

from src.backend.agent.options_agent import (
    OptionsAgent,
    _OPTIONS_SYSTEM_PROMPT,
)
from src.backend.options_intel.snapshot import OptionsContext


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_call: dict = {}

    async def chat_json(self, *, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        self.last_call = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": schema,
        }
        return self.payload


def _breached_context() -> OptionsContext:
    return OptionsContext(
        timestamp_utc="2026-04-22T14:00:00+00:00",
        spot=75000.0,
        spot_24h_change_pct=-0.01,
        opening_range={"high": 75500, "low": 74500, "position": "inside"},
        keltner={"ema20": 75000, "upper": 76000, "lower": 74000, "position": "inside"},
        atm_iv_by_tenor={7: 0.60, 14: 0.62, 30: 0.65, 60: 0.67},
        skew_25d_by_tenor={7: -0.02, 14: -0.02, 30: -0.015},
        term_structure_slope=0.0001,
        expected_move_pct_by_tenor={7: 0.04, 14: 0.06, 30: 0.08},
        vol_regime="cheap",
        vol_regime_confidence="medium",
        realized_iv_ratio_30d=1.3,
        straddle_test_30d={"lower": 70000, "upper": 80000, "current_spot": 75000, "label": "fair"},
        top_mispricings_vs_deribit=[],
        open_positions=[
            {
                "instrument_name": "BTC-29MAY26-90000-C",
                "size": 3.0,
                "side": "long",
                "days_to_expiry": 37,
                "kind": "call",
                "strike": 90000.0,
                "delta": 0.16,
                "gamma": 0.00001,
                "vega": 120.0,
                "theta": -30.0,
            }
        ],
        portfolio_greeks={"delta": 0.48, "gamma": 0.00003, "vega": 360.0, "theta": -90.0},
        capital_available=300000.0,
        max_contracts_per_trade=0.1,
        max_open_positions=3,
        open_position_count=1,
        hyperliquid_free_margin=0.0,
        hyperliquid_max_leverage=20,
        max_hedge_notional=0.0,
        recent_options_trades=[],
        recent_options_skips=[],
        vol_data_coverage={
            "covered_tenors_days": [7, 14, 30, 60],
            "missing_tenors_days": [],
            "positions_without_iv": [],
            "positions_without_greeks": [],
            "surface_age_seconds": 120,
            "surface_stale": False,
        },
    )


def test_system_prompt_distinguishes_hedge_failure_from_cut_loss():
    """The prompt must explicitly tell the LLM a delta breach alone is NOT a cut-loss."""
    prompt = _OPTIONS_SYSTEM_PROMPT
    assert "HEDGE FAILURE vs. CUT-LOSS" in prompt
    assert "hedge_unavailable" in prompt
    assert "not a cut-loss" in prompt.lower() or "NOT by itself a cut-loss" in prompt


def test_system_prompt_references_vol_data_coverage():
    """The prompt must tell the LLM to check vol_data_coverage before claiming 'no vol data'."""
    prompt = _OPTIONS_SYSTEM_PROMPT
    assert "VOL DATA COVERAGE" in prompt
    assert "vol_data_coverage" in prompt
    assert "surface_stale" in prompt


def test_system_prompt_documents_risk_flags_output_field():
    """risk_flags must be listed in the output contract."""
    prompt = _OPTIONS_SYSTEM_PROMPT
    assert "risk_flags" in prompt


def test_delta_band_wording_is_for_new_trades_not_forced_closure():
    """The portfolio delta band must be framed as a sizing constraint for new trades."""
    prompt = _OPTIONS_SYSTEM_PROMPT
    assert "for NEW" in prompt or "for new trades" in prompt.lower()


@pytest.mark.asyncio
async def test_user_prompt_includes_breached_state_and_coverage():
    """The breached-delta + hedge-unavailable state must reach the LLM intact."""
    llm = _FakeLLM({"reasoning": "hold: hedge unavailable", "trade_decisions": []})
    agent = OptionsAgent(llm=llm)
    ctx = _breached_context()
    await agent.decide(ctx)

    user_prompt = llm.last_call["user_prompt"]
    assert "0.48" in user_prompt or "\"delta\":0.48" in user_prompt
    assert "hyperliquid_free_margin" in user_prompt
    assert "BTC-29MAY26-90000-C" in user_prompt
    assert "vol_data_coverage" in user_prompt


@pytest.mark.asyncio
async def test_agent_parses_hold_with_hedge_unavailable_risk_flag():
    """When the LLM emits a hold + risk_flags=['hedge_unavailable'], the agent round-trips it."""
    llm = _FakeLLM({
        "reasoning": "hedge unavailable",
        "trade_decisions": [
            {
                "asset": "BTC-29MAY26-90000-C",
                "action": "hold",
                "venue": "thalex",
                "strategy": "long_call_delta_hedged",
                "underlying": "BTC",
                "kind": "call",
                "tenor_days": 37,
                "rationale": "BTC-29MAY26-90000-C hedge unavailable (HL free margin $0)",
                "risk_flags": ["hedge_unavailable"],
            }
        ],
    })
    agent = OptionsAgent(llm=llm)
    decisions = await agent.decide(_breached_context())

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.action == "hold"
    assert decision.risk_flags == ["hedge_unavailable"]
