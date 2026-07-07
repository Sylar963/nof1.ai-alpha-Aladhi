"""Deribit ``book_summary_by_currency`` normalization.

Deribit book summaries carry ``mark_iv`` (percent) and ``mark_price`` but
NOT ``strike`` or ``option_type`` — those fields must be recovered by
parsing the instrument name. These tests lock that contract so the
merged vol surface always sees a consistent record shape regardless of
which venue it originated from.
"""

from __future__ import annotations

import pytest

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


def test_normalizer_converts_percent_iv_to_decimal():
    """Deribit publishes IV in percent; downstream consumers (mispricing,
    surface) work in decimal, so the normalizer divides by 100."""
    rec = [{
        "instrument_name": "BTC-25DEC26-65000-P",
        "mark_iv": 47.61,
    }]
    out = normalize_deribit_chain(rec)
    assert out[0]["iv"] == pytest.approx(0.4761)
    assert out[0]["mark_iv"] == pytest.approx(0.4761)


def test_normalizer_leaves_decimal_iv_unchanged():
    rec = [{
        "instrument_name": "BTC-25DEC26-65000-P",
        "mark_iv": 0.4761,
    }]
    out = normalize_deribit_chain(rec)
    assert out[0]["iv"] == pytest.approx(0.4761)


def test_surface_parser_does_not_double_convert_normalized_iv():
    """Decimal IV out of the normalizer must survive the surface parser's
    own >2.0 percent heuristic untouched."""
    from src.backend.options_intel.vol_surface import _parse_entry

    rec = [{
        "instrument_name": "BTC-25DEC26-65000-P",
        "mark_iv": 47.61,
        "mark_price": 0.063,
        "underlying_price": 80000.0,
    }]
    entry = _parse_entry(normalize_deribit_chain(rec)[0])
    assert entry is not None
    assert entry.iv == pytest.approx(0.4761)


def test_normalizer_converts_btc_mark_price_to_usd():
    rec = [{
        "instrument_name": "BTC-25DEC26-65000-P",
        "mark_iv": 47.61,
        "mark_price": 0.063,
        "underlying_price": 80000.0,
    }]
    out = normalize_deribit_chain(rec)
    assert out[0]["mark_price"] == pytest.approx(0.063 * 80000.0)
    assert out[0]["underlying_price"] == pytest.approx(80000.0)


def test_normalizer_falls_back_to_estimated_delivery_price():
    rec = [{
        "instrument_name": "BTC-25DEC26-65000-P",
        "mark_iv": 47.61,
        "mark_price": 0.063,
        "estimated_delivery_price": 79000.0,
    }]
    out = normalize_deribit_chain(rec)
    assert out[0]["mark_price"] == pytest.approx(0.063 * 79000.0)


def test_normalizer_drops_mark_price_without_underlying():
    """A BTC-denominated mark with no USD conversion basis must not leak
    through — it would poison USD expected-move metrics downstream."""
    rec = [{
        "instrument_name": "BTC-25DEC26-65000-P",
        "mark_iv": 47.61,
        "mark_price": 0.063,
    }]
    out = normalize_deribit_chain(rec)
    assert out[0]["mark_price"] is None


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
