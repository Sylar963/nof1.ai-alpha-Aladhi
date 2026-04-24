"""vol_data_coverage — map of which open positions have usable IV/greeks.

Built by options_intel.builder._build_vol_data_coverage and surfaced in
OptionsContext.to_dict() so the OptionsAgent can distinguish 'tenor missing
from chain' from 'I forgot to look' before writing 'no vol data' as a
rationale.
"""

from __future__ import annotations

from src.backend.options_intel.builder import _build_vol_data_coverage


def _pos(instrument: str, dte: int, *, with_greeks: bool = True) -> dict:
    out = {"instrument_name": instrument, "days_to_expiry": dte, "size": 1.0, "side": "long"}
    if with_greeks:
        out.update({"delta": 0.5, "gamma": 0.0001, "vega": 100.0, "theta": -10.0})
    return out


def test_coverage_reports_tenor_gaps_outside_covered_range():
    coverage = _build_vol_data_coverage(
        atm_iv_by_tenor={7: 0.6, 14: 0.62, 30: 0.65},
        open_positions=[_pos("BTC-29JUN26-100000-C", dte=60)],
        portfolio_greeks={"delta": 0.5, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
        surface_age_seconds=30.0,
        surface_stale_multiplier=2.0,
        vol_surface_interval_seconds=900.0,
    )
    assert coverage["covered_tenors_days"] == [7, 14, 30]
    assert coverage["missing_tenors_days"] == [60]
    assert len(coverage["positions_without_iv"]) == 1
    assert coverage["positions_without_iv"][0]["instrument"] == "BTC-29JUN26-100000-C"
    assert coverage["positions_without_iv"][0]["reason"] == "tenor_not_in_chain"


def test_coverage_treats_interpolatable_tenor_as_covered():
    coverage = _build_vol_data_coverage(
        atm_iv_by_tenor={7: 0.6, 30: 0.65, 60: 0.67},
        open_positions=[_pos("BTC-29MAY26-90000-C", dte=37)],
        portfolio_greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
        surface_age_seconds=30.0,
        surface_stale_multiplier=2.0,
        vol_surface_interval_seconds=900.0,
    )
    assert coverage["missing_tenors_days"] == []
    assert coverage["positions_without_iv"] == []


def test_coverage_lists_positions_with_missing_greeks():
    coverage = _build_vol_data_coverage(
        atm_iv_by_tenor={7: 0.6, 30: 0.65, 60: 0.67},
        open_positions=[_pos("BTC-29MAY26-90000-C", dte=37, with_greeks=False)],
        portfolio_greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
        surface_age_seconds=30.0,
        surface_stale_multiplier=2.0,
        vol_surface_interval_seconds=900.0,
    )
    assert len(coverage["positions_without_greeks"]) == 1
    assert coverage["positions_without_greeks"][0]["instrument"] == "BTC-29MAY26-90000-C"
    assert coverage["positions_without_greeks"][0]["reason"] == "greeks_unavailable"


def test_surface_stale_flips_true_past_multiplier():
    coverage = _build_vol_data_coverage(
        atm_iv_by_tenor={7: 0.6, 30: 0.65},
        open_positions=[],
        portfolio_greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
        surface_age_seconds=1801.0,
        surface_stale_multiplier=2.0,
        vol_surface_interval_seconds=900.0,
    )
    assert coverage["surface_stale"] is True
    assert coverage["surface_age_seconds"] == 1801.0


def test_surface_stale_false_within_threshold():
    coverage = _build_vol_data_coverage(
        atm_iv_by_tenor={7: 0.6, 30: 0.65},
        open_positions=[],
        portfolio_greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
        surface_age_seconds=1500.0,
        surface_stale_multiplier=2.0,
        vol_surface_interval_seconds=900.0,
    )
    assert coverage["surface_stale"] is False


def test_surface_stale_false_when_age_unknown():
    coverage = _build_vol_data_coverage(
        atm_iv_by_tenor={7: 0.6},
        open_positions=[],
        portfolio_greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
        surface_age_seconds=None,
        surface_stale_multiplier=2.0,
        vol_surface_interval_seconds=900.0,
    )
    assert coverage["surface_stale"] is False
    assert coverage["surface_age_seconds"] is None


def test_coverage_included_in_options_context_serialization():
    from src.backend.options_intel.snapshot import OptionsContext

    ctx = OptionsContext(
        timestamp_utc="2026-04-22T14:00:00+00:00",
        spot=75000.0,
        spot_24h_change_pct=0.0,
        opening_range={},
        keltner={},
        atm_iv_by_tenor={7: 0.6},
        skew_25d_by_tenor={},
        term_structure_slope=0.0,
        expected_move_pct_by_tenor={},
        vol_regime="cheap",
        vol_regime_confidence="medium",
        realized_iv_ratio_30d=1.2,
        straddle_test_30d={},
        vol_data_coverage={
            "covered_tenors_days": [7],
            "missing_tenors_days": [],
            "positions_without_iv": [],
            "positions_without_greeks": [],
            "surface_age_seconds": 60.0,
            "surface_stale": False,
        },
    )
    dumped = ctx.to_dict()
    assert "vol_data_coverage" in dumped
    assert dumped["vol_data_coverage"]["covered_tenors_days"] == [7]
    json_text = ctx.to_json()
    assert "vol_data_coverage" in json_text
