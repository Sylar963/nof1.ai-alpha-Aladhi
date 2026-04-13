"""Tests for pure-logic options helpers used by the Thalex adapter.

These cover instrument-name parsing, intent → instrument resolution, and the
position-count / size risk caps. None of these tests touch the network."""

from datetime import date

import pytest

from src.backend.trading.options import (
    InstrumentSpec,
    OptionIntent,
    RiskCaps,
    find_best_instrument,
    parse_instrument_name,
    validate_options_order,
)


# ---------------------------------------------------------------------------
# parse_instrument_name
# ---------------------------------------------------------------------------


def test_parse_call_option():
    spec = parse_instrument_name("BTC-27JUN25-100000-C")
    assert spec.underlying == "BTC"
    assert spec.expiry == date(2025, 6, 27)
    assert spec.strike == 100000.0
    assert spec.kind == "call"


def test_parse_put_option():
    spec = parse_instrument_name("BTC-31DEC25-50000-P")
    assert spec.underlying == "BTC"
    assert spec.expiry == date(2025, 12, 31)
    assert spec.strike == 50000.0
    assert spec.kind == "put"


def test_parse_perpetual_returns_none():
    """Perpetuals don't fit the dated-option grammar — return None for parser."""
    assert parse_instrument_name("BTC-PERPETUAL") is None


def test_parse_invalid_string_returns_none():
    assert parse_instrument_name("nonsense") is None
    assert parse_instrument_name("") is None


# ---------------------------------------------------------------------------
# find_best_instrument
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_instruments():
    """Realistic-shape instrument list as Thalex public/instruments would return."""
    return [
        # 7-day expiry calls
        {"instrument_name": "BTC-20APR26-60000-C", "type": "option", "underlying": "BTCUSD",
         "option_type": "call", "expiry_timestamp": 1776672000},
        {"instrument_name": "BTC-20APR26-65000-C", "type": "option", "underlying": "BTCUSD",
         "option_type": "call", "expiry_timestamp": 1776672000},
        # 30-day expiry — what we want
        {"instrument_name": "BTC-10MAY26-60000-C", "type": "option", "underlying": "BTCUSD",
         "option_type": "call", "expiry_timestamp": 1778400000},
        {"instrument_name": "BTC-10MAY26-65000-C", "type": "option", "underlying": "BTCUSD",
         "option_type": "call", "expiry_timestamp": 1778400000},
        {"instrument_name": "BTC-10MAY26-70000-C", "type": "option", "underlying": "BTCUSD",
         "option_type": "call", "expiry_timestamp": 1778400000},
        # 30-day puts
        {"instrument_name": "BTC-10MAY26-60000-P", "type": "option", "underlying": "BTCUSD",
         "option_type": "put", "expiry_timestamp": 1778400000},
        {"instrument_name": "BTC-10MAY26-50000-P", "type": "option", "underlying": "BTCUSD",
         "option_type": "put", "expiry_timestamp": 1778400000},
        # Unrelated
        {"instrument_name": "ETH-10MAY26-3000-C", "type": "option", "underlying": "ETHUSD",
         "option_type": "call", "expiry_timestamp": 1778400000},
        {"instrument_name": "BTC-PERPETUAL", "type": "perpetual", "underlying": "BTCUSD"},
    ]


def test_find_best_call_by_strike(sample_instruments):
    """Intent: 30d BTC call, strike near 65000 → must pick the 65k call."""
    intent = OptionIntent(underlying="BTC", kind="call", tenor_days=30, target_strike=65000.0)
    pick = find_best_instrument(sample_instruments, intent, today=date(2026, 4, 10))
    assert pick == "BTC-10MAY26-65000-C"


def test_find_best_put_by_strike(sample_instruments):
    intent = OptionIntent(underlying="BTC", kind="put", tenor_days=30, target_strike=50000.0)
    pick = find_best_instrument(sample_instruments, intent, today=date(2026, 4, 10))
    assert pick == "BTC-10MAY26-50000-P"


def test_find_best_excludes_other_underlyings(sample_instruments):
    """ETH options must never be selected when underlying='BTC'."""
    intent = OptionIntent(underlying="BTC", kind="call", tenor_days=30, target_strike=3000.0)
    pick = find_best_instrument(sample_instruments, intent, today=date(2026, 4, 10))
    assert pick is not None
    assert pick.startswith("BTC-")


def test_find_best_picks_closest_tenor(sample_instruments):
    """Tenor=7 should pick the 20APR expiry, not 10MAY."""
    intent = OptionIntent(underlying="BTC", kind="call", tenor_days=10, target_strike=60000.0)
    pick = find_best_instrument(sample_instruments, intent, today=date(2026, 4, 10))
    assert pick == "BTC-20APR26-60000-C"


def test_find_best_uses_target_delta_when_strike_missing(sample_instruments):
    sample_instruments[2]["delta"] = 0.62
    sample_instruments[3]["delta"] = 0.49
    sample_instruments[4]["delta"] = 0.31
    intent = OptionIntent(underlying="BTC", kind="call", tenor_days=30, target_delta=0.50)
    pick = find_best_instrument(sample_instruments, intent, today=date(2026, 4, 10))
    assert pick == "BTC-10MAY26-65000-C"


def test_find_best_returns_none_when_target_delta_has_no_live_hints(sample_instruments):
    intent = OptionIntent(underlying="BTC", kind="call", tenor_days=30, target_delta=0.25)
    pick = find_best_instrument(sample_instruments, intent, today=date(2026, 4, 10))
    assert pick is None


def test_find_best_returns_none_when_no_match(sample_instruments):
    intent = OptionIntent(underlying="DOGE", kind="call", tenor_days=30, target_strike=1.0)
    pick = find_best_instrument(sample_instruments, intent, today=date(2026, 4, 10))
    assert pick is None


# ---------------------------------------------------------------------------
# validate_options_order — risk caps
# ---------------------------------------------------------------------------


def test_validate_rejects_oversized_order():
    caps = RiskCaps(max_contracts_per_trade=0.1, max_open_positions=3, allowed_underlyings=["BTC"])
    ok, reason = validate_options_order(
        underlying="BTC", contracts=0.2, open_positions_count=0, caps=caps
    )
    assert ok is False
    assert "max_contracts_per_trade" in reason


def test_validate_accepts_size_at_cap():
    caps = RiskCaps(max_contracts_per_trade=0.1, max_open_positions=3, allowed_underlyings=["BTC"])
    ok, _ = validate_options_order(
        underlying="BTC", contracts=0.1, open_positions_count=0, caps=caps
    )
    assert ok is True


def test_validate_rejects_when_position_cap_reached():
    caps = RiskCaps(max_contracts_per_trade=0.1, max_open_positions=3, allowed_underlyings=["BTC"])
    ok, reason = validate_options_order(
        underlying="BTC", contracts=0.05, open_positions_count=3, caps=caps
    )
    assert ok is False
    assert "max_open_positions" in reason


def test_validate_rejects_disallowed_underlying():
    caps = RiskCaps(max_contracts_per_trade=0.1, max_open_positions=3, allowed_underlyings=["BTC"])
    ok, reason = validate_options_order(
        underlying="ETH", contracts=0.05, open_positions_count=0, caps=caps
    )
    assert ok is False
    assert "underlying" in reason


def test_validate_rejects_zero_or_negative_contracts():
    caps = RiskCaps(max_contracts_per_trade=0.1, max_open_positions=3, allowed_underlyings=["BTC"])
    ok, _ = validate_options_order(
        underlying="BTC", contracts=0.0, open_positions_count=0, caps=caps
    )
    assert ok is False


def test_instrument_spec_round_trips_to_instrument_name():
    """An InstrumentSpec built from a parsed name should be able to render the original name."""
    original = "BTC-27JUN25-100000-C"
    spec = parse_instrument_name(original)
    assert spec is not None
    assert spec.to_instrument_name() == original
