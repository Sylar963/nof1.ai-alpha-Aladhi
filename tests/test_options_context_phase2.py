import pytest

from src.backend.config_loader import CONFIG
from src.backend.options_intel.snapshot import (
    EventSummary,
    OptionsContext,
    StructureView,
)


def _empty_context():
    return OptionsContext(
        timestamp_utc="2026-05-20T00:00:00Z",
        spot=100000.0,
        spot_24h_change_pct=0.0,
        opening_range={},
        keltner={},
        atm_iv_by_tenor={},
        skew_25d_by_tenor={},
        term_structure_slope=0.0,
        expected_move_pct_by_tenor={},
        vol_regime="fair",
        vol_regime_confidence="high",
        realized_iv_ratio_30d=1.0,
        straddle_test_30d={},
    )


def _view():
    return StructureView(
        structure_id="abc", kind="credit_put_spread", underlying="BTC",
        tenor_days=14, days_open=2, legs=({"instrument_name": "BTC-27JUN26-100000-P"},),
        net_premium=20.0, is_credit=True,
        max_loss=980.0, max_profit=20.0, breakevens=(99800.0,),
        short_leg_delta=-0.30, breach_state="warning",
        pnl_abs=0.0, pnl_pct=0.0, aggregate_greeks={},
    )


def test_to_dict_excludes_structures_when_flag_off(monkeypatch):
    monkeypatch.setitem(CONFIG, "options_structure_prompt", False)
    ctx = _empty_context()
    object.__setattr__(ctx, "structure_views", [_view()])
    object.__setattr__(ctx, "triggered_by_events", [
        EventSummary(type="manual", fired_at="2026-05-20T00:00:00Z",
                     description="manual", structure_id=None)
    ])
    payload = ctx.to_dict()
    assert "structures" not in payload
    assert "triggered_by_events" not in payload


def test_to_dict_includes_structures_when_flag_on(monkeypatch):
    monkeypatch.setitem(CONFIG, "options_structure_prompt", True)
    ctx = _empty_context()
    object.__setattr__(ctx, "structure_views", [_view()])
    object.__setattr__(ctx, "triggered_by_events", [
        EventSummary(type="manual", fired_at="2026-05-20T00:00:00Z",
                     description="manual", structure_id=None)
    ])
    payload = ctx.to_dict()
    assert "structures" in payload
    assert len(payload["structures"]) == 1
    assert payload["structures"][0]["kind"] == "credit_put_spread"
    assert payload["structures"][0]["days_open"] == 2
    assert "triggered_by_events" in payload
    assert payload["triggered_by_events"][0]["type"] == "manual"


def test_to_dict_omits_keys_when_views_empty_even_with_flag_on(monkeypatch):
    monkeypatch.setitem(CONFIG, "options_structure_prompt", True)
    ctx = _empty_context()
    payload = ctx.to_dict()
    assert "structures" not in payload
    assert "triggered_by_events" not in payload
