"""Deribit ``book_summary_by_currency`` normalization.

Deribit book summaries carry ``mark_iv`` (percent) and ``mark_price`` but
NOT ``strike`` or ``option_type`` — those fields must be recovered by
parsing the instrument name. These tests lock that contract so the
merged vol surface always sees a consistent record shape regardless of
which venue it originated from.
"""

from __future__ import annotations

from src.backend.options_intel.deribit_chain import normalize_deribit_chain


def test_normalizer_parses_strike_and_kind_from_name():
    rec = [{
        "instrument_name": "BTC-25DEC26-65000-P",
        "mark_iv": 47.61,
        "mark_price": 0.063,
        "underlying_price": 80000.0,
    }]
    out = normalize_deribit_chain(rec)
    assert len(out) == 1
    parsed = out[0]
    assert parsed["option_type"] == "put"
    assert parsed["strike"] == 65000.0
    assert parsed["strike_price"] == 65000.0
    assert parsed["type"] == "option"


def test_normalizer_keeps_iv_as_published_percent():
    """The surface parser divides by 100 when iv > 2; we preserve the input."""
    rec = [{
        "instrument_name": "BTC-25DEC26-65000-P",
        "mark_iv": 47.61,
    }]
    out = normalize_deribit_chain(rec)
    assert out[0]["iv"] == 47.61
    assert out[0]["mark_iv"] == 47.61


def test_normalizer_emits_seconds_timestamps():
    rec = [{
        "instrument_name": "BTC-25DEC26-65000-P",
        "mark_iv": 47.61,
    }]
    out = normalize_deribit_chain(rec)
    ts = out[0]["expiration_timestamp"]
    assert ts < 1e11, "must be seconds, not milliseconds"
    assert out[0]["expiry_timestamp"] == ts


def test_normalizer_drops_records_with_unparseable_name():
    rec = [
        {"instrument_name": "GARBAGE", "mark_iv": 60.0},
        {"instrument_name": "BTC-25DEC26-65000-P", "mark_iv": 47.61},
    ]
    out = normalize_deribit_chain(rec)
    assert len(out) == 1
    assert out[0]["instrument_name"] == "BTC-25DEC26-65000-P"


def test_normalizer_drops_records_with_no_iv():
    rec = [
        {"instrument_name": "BTC-25DEC26-65000-P"},
        {"instrument_name": "BTC-25DEC26-65000-C", "mark_iv": 0.0},
    ]
    out = normalize_deribit_chain(rec)
    assert out == []


def test_normalizer_empty_input():
    assert normalize_deribit_chain([]) == []
