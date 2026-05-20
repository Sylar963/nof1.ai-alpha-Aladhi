import dataclasses

import pytest

from src.backend.options_intel.snapshot import EventSummary, StructureView


def _classifier_dict():
    return {
        "structure_id": "abc123",
        "kind": "credit_put_spread",
        "underlying": "BTC",
        "tenor_days_min": 14,
        "tenor_days_max": 14,
        "net_premium": 20.0,
        "is_credit": True,
        "max_loss": 980.0,
        "max_profit": 20.0,
        "breakevens": [99800.0],
        "short_leg_delta": -0.30,
        "breach_state": "warning",
        "pnl_abs": 0.0,
        "pnl_pct": 0.0,
        "aggregate_greeks": {"delta": -0.20, "gamma": 0.0005, "vega": 30, "theta": -3},
        "confidence": 1.0,
        "legs": ["BTC-27JUN26-100000-P", "BTC-27JUN26-90000-P"],
    }


def _open_positions():
    return [
        {"instrument_name": "BTC-27JUN26-100000-P", "kind": "put", "strike": 100000.0,
         "side": "short", "size": 0.1, "days_to_expiry": 14, "delta": -0.30},
        {"instrument_name": "BTC-27JUN26-90000-P", "kind": "put", "strike": 90000.0,
         "side": "long", "size": 0.1, "days_to_expiry": 14, "delta": -0.10},
    ]


def test_structure_view_is_frozen():
    view = StructureView(
        structure_id="abc", kind="credit_put_spread", underlying="BTC",
        tenor_days=14, days_open=2, legs=(),
        net_premium=20.0, is_credit=True,
        max_loss=980.0, max_profit=20.0, breakevens=(99800.0,),
        short_leg_delta=-0.30, breach_state="warning",
        pnl_abs=0.0, pnl_pct=0.0, aggregate_greeks={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.kind = "iron_condor"


def test_from_classifier_dict_uses_tenor_days_min():
    view = StructureView.from_classifier_dict(_classifier_dict(), _open_positions(), days_open=3)
    assert view.structure_id == "abc123"
    assert view.kind == "credit_put_spread"
    assert view.tenor_days == 14
    assert view.days_open == 3
    assert view.breach_state == "warning"
    assert view.is_credit is True


def test_from_classifier_dict_uses_min_when_tenors_differ():
    raw = {**_classifier_dict(), "tenor_days_min": 7, "tenor_days_max": 28,
           "legs": ["BTC-27JUN26-100000-P", "BTC-25SEP26-100000-P"]}
    open_positions = [
        {"instrument_name": "BTC-27JUN26-100000-P", "kind": "put", "strike": 100000.0,
         "side": "short", "size": 0.1, "days_to_expiry": 7, "delta": -0.30},
        {"instrument_name": "BTC-25SEP26-100000-P", "kind": "put", "strike": 100000.0,
         "side": "long", "size": 0.1, "days_to_expiry": 28, "delta": -0.20},
    ]
    view = StructureView.from_classifier_dict(raw, open_positions, days_open=0)
    assert view.tenor_days == 7


def test_from_classifier_dict_expands_legs_with_open_positions():
    view = StructureView.from_classifier_dict(_classifier_dict(), _open_positions(), days_open=0)
    assert len(view.legs) == 2
    leg_names = {leg["instrument_name"] for leg in view.legs}
    assert leg_names == {"BTC-27JUN26-100000-P", "BTC-27JUN26-90000-P"}
    short_leg = next(leg for leg in view.legs if leg["side"] == "short")
    assert short_leg["kind"] == "put"
    assert short_leg["strike"] == 100000.0
    assert short_leg["contracts"] == 0.1
    assert short_leg["abs_delta"] == 0.30


def test_from_classifier_dict_handles_missing_leg_lookup():
    raw = {**_classifier_dict(), "legs": ["BTC-27JUN26-100000-P", "MISSING-LEG"]}
    view = StructureView.from_classifier_dict(raw, _open_positions(), days_open=0)
    assert len(view.legs) == 2
    missing = next(leg for leg in view.legs if leg["instrument_name"] == "MISSING-LEG")
    assert missing["kind"] is None
    assert missing["side"] is None
    assert missing["strike"] is None
    assert missing["contracts"] is None
    assert missing["abs_delta"] is None


def test_structure_view_to_dict_round_trip():
    view = StructureView.from_classifier_dict(_classifier_dict(), _open_positions(), days_open=4)
    payload = view.to_dict()
    assert payload["structure_id"] == "abc123"
    assert payload["kind"] == "credit_put_spread"
    assert payload["tenor_days"] == 14
    assert payload["days_open"] == 4
    assert payload["is_credit"] is True
    assert isinstance(payload["legs"], list)
    assert isinstance(payload["breakevens"], list)


def test_event_summary_to_dict():
    ev = EventSummary(
        type="structure_breach",
        fired_at="2026-05-20T12:00:00Z",
        description="structure abc123 breached: short delta 0.42",
        structure_id="abc123",
    )
    payload = ev.to_dict()
    assert payload == {
        "type": "structure_breach",
        "fired_at": "2026-05-20T12:00:00Z",
        "description": "structure abc123 breached: short delta 0.42",
        "structure_id": "abc123",
    }
