"""Tests for the OptionsContext snapshot dataclass.

The snapshot is the compact digest the LLM consumes. It must:
- Serialize cleanly to JSON.
- Stay under a strict byte budget regardless of how many instruments Thalex
  has listed (no raw chains, no full surface — only highlights).
- Never accidentally embed an entire Thalex chain or Deribit chain even if
  callers pass big lists by mistake."""

import json

import pytest

from src.backend.options_intel.snapshot import (
    BYTE_BUDGET,
    OptionsContext,
)


@pytest.fixture
def example_context():
    return OptionsContext(
        timestamp_utc="2026-04-10T19:00:00+00:00",
        spot=60123.45,
        spot_24h_change_pct=0.012,
        opening_range={"high": 60500, "low": 59800, "position": "inside"},
        keltner={"ema20": 60000, "upper": 60800, "lower": 59200, "position": "inside"},
        atm_iv_by_tenor={7: 0.65, 14: 0.66, 30: 0.68, 60: 0.70, 90: 0.72},
        skew_25d_by_tenor={7: 0.05, 14: 0.04, 30: 0.03},
        term_structure_slope=0.0001,
        expected_move_pct_by_tenor={7: 0.045, 14: 0.06, 30: 0.085},
        vol_regime="cheap",
        vol_regime_confidence="high",
        realized_iv_ratio_15d=1.35,
        straddle_test_15d={
            "lower": 55000,
            "upper": 65000,
            "current_spot": 60123.45,
            "label": "fair",
        },
        top_mispricings_vs_deribit=[
            {"instrument_name": "BTC-10MAY26-65000-C", "iv_thalex": 0.72, "iv_deribit": 0.65, "edge_bps": 700},
            {"instrument_name": "BTC-10MAY26-60000-P", "iv_thalex": 0.66, "iv_deribit": 0.62, "edge_bps": 400},
        ],
        open_positions=[
            {"instrument_name": "BTC-10MAY26-65000-C", "contracts": 0.05, "delta": 0.42,
             "gamma": 0.001, "vega": 9.0, "theta": -3.5, "days_to_expiry": 28},
        ],
        portfolio_greeks={"delta": 0.021, "gamma": 0.00005, "vega": 0.45, "theta": -0.175},
        capital_available=10000.0,
        max_contracts_per_trade=0.1,
        max_open_positions=3,
        open_position_count=1,
        recent_options_trades=[
            {"strategy": "long_call_delta_hedged", "pnl_usd": 250, "closed_at": "2026-04-08T12:00:00+00:00"},
        ],
    )


def test_snapshot_serializes_to_json(example_context):
    serialized = example_context.to_json()
    payload = json.loads(serialized)
    assert payload["spot"] == pytest.approx(60123.45)
    assert payload["vol_regime"] == "cheap"
    assert payload["atm_iv_by_tenor"]["30"] == pytest.approx(0.68)


def test_snapshot_byte_budget_enforced(example_context):
    """The serialized digest must stay tight enough to fit comfortably in a prompt."""
    payload = example_context.to_json()
    assert len(payload.encode("utf-8")) < BYTE_BUDGET


def test_snapshot_does_not_leak_raw_chain():
    """Even if a caller passes a huge mispricings list by mistake, the
    serializer caps it so the prompt stays small."""
    huge_mispricings = [
        {"instrument_name": f"BTC-X-{i}-C", "iv_thalex": 0.6, "iv_deribit": 0.5, "edge_bps": 1000}
        for i in range(500)
    ]
    ctx = OptionsContext(
        timestamp_utc="2026-04-10T19:00:00+00:00",
        spot=60000.0,
        spot_24h_change_pct=0.0,
        opening_range={"high": 60500, "low": 59800, "position": "inside"},
        keltner={"ema20": 60000, "upper": 60800, "lower": 59200, "position": "inside"},
        atm_iv_by_tenor={},
        skew_25d_by_tenor={},
        term_structure_slope=0.0,
        expected_move_pct_by_tenor={},
        vol_regime="unknown",
        vol_regime_confidence="unknown",
        realized_iv_ratio_15d=0.0,
        straddle_test_15d={},
        top_mispricings_vs_deribit=huge_mispricings,
        open_positions=[],
        portfolio_greeks={"delta": 0, "gamma": 0, "vega": 0, "theta": 0},
        capital_available=0.0,
        max_contracts_per_trade=0.1,
        max_open_positions=3,
        open_position_count=0,
        recent_options_trades=[],
    )
    serialized = ctx.to_json()
    payload = json.loads(serialized)
    # Top mispricings list should be capped to a sane number (≤10).
    assert len(payload["top_mispricings_vs_deribit"]) <= 10
