"""Aggregated vol surface — Thalex (primary) + Deribit (fallback).

The LLM sees a single merged surface. This test module locks the merge
policy: Thalex wins per-tenor where it has data; Deribit fills the gaps;
smiles are unioned so ATM lookups and 25d-skew searches see both venues.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.backend.options_intel.vol_surface import (
    build_vol_surface,
    merge_surfaces,
)


_TODAY = date(2026, 4, 22)


def _rec(name: str, *, strike: float, kind: str, iv: float, expiry_seconds: int, mark_price: float = 0.01, delta: float | None = None) -> dict:
    out = {
        "instrument_name": name,
        "type": "option",
        "option_type": kind,
        "strike_price": strike,
        "iv": iv,
        "mark_price": mark_price,
        "expiration_timestamp": expiry_seconds,
    }
    if delta is not None:
        out["delta"] = delta
    return out


def _tenor_ts(dte_days: int) -> int:
    import datetime as _dt
    return int(_dt.datetime(_TODAY.year, _TODAY.month, _TODAY.day, tzinfo=_dt.timezone.utc).timestamp()) + dte_days * 86400


def test_primary_surface_wins_per_tenor():
    thalex_chain = [
        _rec("BTC-T-7-80000-C", strike=80000, kind="call", iv=0.60, expiry_seconds=_tenor_ts(7)),
        _rec("BTC-T-7-80000-P", strike=80000, kind="put", iv=0.58, expiry_seconds=_tenor_ts(7)),
    ]
    deribit_chain = [
        _rec("BTC-D-7-80000-C", strike=80000, kind="call", iv=0.70, expiry_seconds=_tenor_ts(7)),
        _rec("BTC-D-7-80000-P", strike=80000, kind="put", iv=0.68, expiry_seconds=_tenor_ts(7)),
    ]
    primary = build_vol_surface(thalex_chain, spot=80000.0, today=_TODAY)
    fallback = build_vol_surface(deribit_chain, spot=80000.0, today=_TODAY)
    merged = merge_surfaces(primary, fallback)

    assert 7 in merged.atm_iv_by_tenor
    assert merged.atm_iv_by_tenor[7] == primary.atm_iv_by_tenor[7]
    assert merged.atm_iv_by_tenor[7] != fallback.atm_iv_by_tenor[7]


def test_fallback_fills_tenors_missing_from_primary():
    thalex_chain = [
        _rec("BTC-T-7-80000-C", strike=80000, kind="call", iv=0.60, expiry_seconds=_tenor_ts(7)),
        _rec("BTC-T-7-80000-P", strike=80000, kind="put", iv=0.58, expiry_seconds=_tenor_ts(7)),
    ]
    deribit_chain = [
        _rec("BTC-D-30-80000-C", strike=80000, kind="call", iv=0.65, expiry_seconds=_tenor_ts(30)),
        _rec("BTC-D-30-80000-P", strike=80000, kind="put", iv=0.67, expiry_seconds=_tenor_ts(30)),
    ]
    merged = merge_surfaces(
        build_vol_surface(thalex_chain, spot=80000.0, today=_TODAY),
        build_vol_surface(deribit_chain, spot=80000.0, today=_TODAY),
    )

    assert set(merged.atm_iv_by_tenor.keys()) == {7, 30}
    assert merged.atm_iv_by_tenor[30] == pytest.approx(0.66)


def test_smiles_are_unioned_across_venues():
    thalex_chain = [
        _rec("BTC-T-7-80000-C", strike=80000, kind="call", iv=0.60, expiry_seconds=_tenor_ts(7)),
    ]
    deribit_chain = [
        _rec("BTC-D-7-90000-C", strike=90000, kind="call", iv=0.70, expiry_seconds=_tenor_ts(7)),
    ]
    merged = merge_surfaces(
        build_vol_surface(thalex_chain, spot=80000.0, today=_TODAY),
        build_vol_surface(deribit_chain, spot=80000.0, today=_TODAY),
    )

    strikes_in_merged = {e.strike for e in merged.smiles[7]}
    assert strikes_in_merged == {80000.0, 90000.0}


def test_percent_iv_is_normalized_to_decimal_when_building():
    """Deribit publishes IV in percent (47.61 for 47.61%); surface must
    treat anything > 2 as percent and divide by 100."""
    chain = [
        _rec("BTC-D-7-80000-C", strike=80000, kind="call", iv=47.61, expiry_seconds=_tenor_ts(7)),
        _rec("BTC-D-7-80000-P", strike=80000, kind="put", iv=47.61, expiry_seconds=_tenor_ts(7)),
    ]
    surface = build_vol_surface(chain, spot=80000.0, today=_TODAY)
    assert surface.atm_iv_by_tenor[7] == pytest.approx(0.4761)


def test_surface_tolerates_strike_price_field_name():
    """Thalex uses ``strike_price`` not ``strike`` — the parser must
    accept both so a raw Thalex record flows through unchanged."""
    chain = [
        {
            "instrument_name": "BTC-T-7-80000-C",
            "type": "option",
            "option_type": "call",
            "strike_price": 80000.0,  # no "strike"
            "iv": 0.60,  # no "mark_iv"
            "expiration_timestamp": _tenor_ts(7),  # no "expiry_timestamp"
        },
        {
            "instrument_name": "BTC-T-7-80000-P",
            "type": "option",
            "option_type": "put",
            "strike_price": 80000.0,
            "iv": 0.58,
            "expiration_timestamp": _tenor_ts(7),
        },
    ]
    surface = build_vol_surface(chain, spot=80000.0, today=_TODAY)
    assert 7 in surface.atm_iv_by_tenor
    assert surface.atm_iv_by_tenor[7] == pytest.approx(0.59)
