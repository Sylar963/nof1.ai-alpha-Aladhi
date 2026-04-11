"""Tests for the Thalex × Deribit IV mispricing scanner.

The scanner aligns option instruments by exact (expiry, strike, kind) and
flags the largest IV gaps. Anything that doesn't have an exact Deribit
counterpart is **skipped** (PR A behavior — interpolation lives in PR C).
The skip count is exposed for visibility."""

from datetime import date

import pytest

from src.backend.options_intel.mispricing import (
    MispricingScanReport,
    interpolate_deribit_surface,
    scan_mispricings,
)


def _thalex(name, expiry, strike, kind, iv):
    return {
        "instrument_name": name,
        "type": "option",
        "underlying": "BTCUSD",
        "option_type": kind,
        "strike": strike,
        "mark_iv": iv,
        "expiry_timestamp": int(expiry),
    }


def _deribit(name, expiry_ts, strike, kind, iv):
    return {
        "instrument_name": name,
        "kind": "option",
        "option_type": kind,
        "strike": strike,
        "mark_iv": iv,
        "expiration_timestamp": int(expiry_ts * 1000),  # Deribit uses ms
    }


def _utc_ts(d: date) -> int:
    """Seconds since epoch for a date at 08:00 UTC (Deribit settlement time)."""
    from datetime import datetime, timezone
    return int(datetime(d.year, d.month, d.day, 8, 0, 0, tzinfo=timezone.utc).timestamp())


# ---------------------------------------------------------------------------
# Alignment + scoring
# ---------------------------------------------------------------------------


def test_scan_finds_top_mispricings_by_iv_diff_bps():
    expiry = _utc_ts(date(2026, 5, 10))

    thalex_chain = [
        _thalex("BTC-10MAY26-60000-C", expiry, 60000, "call", 0.65),
        _thalex("BTC-10MAY26-65000-C", expiry, 65000, "call", 0.70),
        _thalex("BTC-10MAY26-70000-C", expiry, 70000, "call", 0.75),
    ]
    deribit_chain = [
        _deribit("BTC-10MAY26-60000-C", expiry, 60000, "call", 0.62),  # Thalex 300 bps richer
        _deribit("BTC-10MAY26-65000-C", expiry, 65000, "call", 0.71),  # Thalex 100 bps cheaper
        _deribit("BTC-10MAY26-70000-C", expiry, 70000, "call", 0.65),  # Thalex 1000 bps richer
    ]

    report = scan_mispricings(thalex_chain, deribit_chain, top_n=3)

    assert isinstance(report, MispricingScanReport)
    assert len(report.top) == 3
    # Largest absolute IV diff first.
    assert report.top[0]["instrument_name"] == "BTC-10MAY26-70000-C"
    assert report.top[0]["edge_bps"] == pytest.approx(1000, abs=1)
    assert report.top[1]["instrument_name"] == "BTC-10MAY26-60000-C"
    assert report.top[1]["edge_bps"] == pytest.approx(300, abs=1)
    assert report.top[2]["instrument_name"] == "BTC-10MAY26-65000-C"
    assert report.top[2]["edge_bps"] == pytest.approx(-100, abs=1)


def test_scan_skips_unmatched_thalex_instruments():
    """Thalex instrument with no exact (expiry, strike, kind) match on Deribit is skipped."""
    expiry_a = _utc_ts(date(2026, 5, 10))
    expiry_b = _utc_ts(date(2026, 6, 15))  # not on Deribit

    thalex_chain = [
        _thalex("BTC-10MAY26-60000-C", expiry_a, 60000, "call", 0.65),
        _thalex("BTC-15JUN26-60000-C", expiry_b, 60000, "call", 0.68),
    ]
    deribit_chain = [
        _deribit("BTC-10MAY26-60000-C", expiry_a, 60000, "call", 0.60),
    ]

    report = scan_mispricings(thalex_chain, deribit_chain, top_n=5)

    assert len(report.top) == 1
    assert report.matched_count == 1
    assert report.skipped_count == 1


def test_scan_returns_empty_when_no_matches():
    expiry = _utc_ts(date(2026, 5, 10))
    thalex_chain = [_thalex("BTC-10MAY26-60000-C", expiry, 60000, "call", 0.65)]
    deribit_chain = [_deribit("BTC-10MAY26-65000-C", expiry, 65000, "call", 0.60)]

    report = scan_mispricings(thalex_chain, deribit_chain, top_n=5)
    assert report.top == []
    assert report.matched_count == 0
    assert report.skipped_count == 1


def test_scan_filters_below_threshold_bps():
    """A min_edge_bps filter drops trivial diffs from the top list."""
    expiry = _utc_ts(date(2026, 5, 10))
    thalex_chain = [
        _thalex("BTC-10MAY26-60000-C", expiry, 60000, "call", 0.6510),
        _thalex("BTC-10MAY26-65000-C", expiry, 65000, "call", 0.7000),
    ]
    deribit_chain = [
        _deribit("BTC-10MAY26-60000-C", expiry, 60000, "call", 0.6500),  # 10 bps diff
        _deribit("BTC-10MAY26-65000-C", expiry, 65000, "call", 0.6500),  # 500 bps diff
    ]
    report = scan_mispricings(thalex_chain, deribit_chain, top_n=5, min_edge_bps=200)
    assert len(report.top) == 1
    assert report.top[0]["instrument_name"] == "BTC-10MAY26-65000-C"


def test_interpolation_returns_empty_when_only_one_expiry_available():
    """Need at least two Deribit expiries on either side of the target to interpolate."""
    expiry = _utc_ts(date(2026, 5, 10))
    target = _utc_ts(date(2026, 5, 20))
    deribit_chain = [_deribit("BTC-10MAY26-60000-C", expiry, 60000, "call", 0.65)]
    interpolated = interpolate_deribit_surface(deribit_chain, target_expiry=target)
    assert interpolated == []


def test_interpolation_uses_linear_in_variance_for_target_tenor():
    """Standard practitioner interpolation: variance is linear in time, IV = sqrt(var)."""
    near_expiry = _utc_ts(date(2026, 5, 10))    # T1
    far_expiry = _utc_ts(date(2026, 5, 30))     # T2
    target = _utc_ts(date(2026, 5, 20))          # midpoint

    deribit_chain = [
        _deribit("BTC-10MAY26-60000-C", near_expiry, 60000, "call", 0.40),  # var(T1) = 0.16
        _deribit("BTC-30MAY26-60000-C", far_expiry, 60000, "call", 0.60),   # var(T2) = 0.36
    ]
    interpolated = interpolate_deribit_surface(deribit_chain, target_expiry=target)

    assert len(interpolated) == 1
    entry = interpolated[0]
    assert entry["strike"] == 60000
    assert entry["option_type"] == "call"
    # Linear-in-variance midpoint: var = (0.16 + 0.36) / 2 = 0.26 → IV ≈ 0.5099
    import math
    expected_iv = math.sqrt(0.5 * 0.16 + 0.5 * 0.36)
    assert entry["mark_iv"] == pytest.approx(expected_iv, abs=1e-4)


def test_interpolation_only_emits_strikes_present_in_both_bracketing_expiries():
    """A strike that exists at one expiry but not the other can't be interpolated."""
    near_expiry = _utc_ts(date(2026, 5, 10))
    far_expiry = _utc_ts(date(2026, 5, 30))
    target = _utc_ts(date(2026, 5, 20))

    deribit_chain = [
        _deribit("BTC-10MAY26-60000-C", near_expiry, 60000, "call", 0.40),
        _deribit("BTC-10MAY26-65000-C", near_expiry, 65000, "call", 0.50),
        _deribit("BTC-30MAY26-60000-C", far_expiry, 60000, "call", 0.60),
        # 65000 strike missing on far expiry → cannot interpolate
    ]
    interpolated = interpolate_deribit_surface(deribit_chain, target_expiry=target)
    strikes = {e["strike"] for e in interpolated}
    assert 60000 in strikes
    assert 65000 not in strikes


def test_scan_uses_interpolation_when_enabled_and_unmatched_thalex_tenor():
    """When use_interpolation=True, unmatched Thalex tenors get a synthetic Deribit IV."""
    near_expiry = _utc_ts(date(2026, 5, 10))
    far_expiry = _utc_ts(date(2026, 5, 30))
    target_expiry = _utc_ts(date(2026, 5, 20))  # not on Deribit

    thalex_chain = [
        _thalex("BTC-20MAY26-60000-C", target_expiry, 60000, "call", 0.55),
    ]
    deribit_chain = [
        _deribit("BTC-10MAY26-60000-C", near_expiry, 60000, "call", 0.40),
        _deribit("BTC-30MAY26-60000-C", far_expiry, 60000, "call", 0.60),
    ]

    report = scan_mispricings(
        thalex_chain,
        deribit_chain,
        top_n=5,
        use_interpolation=True,
    )
    assert report.matched_count == 1
    assert report.skipped_count == 0
    assert len(report.top) == 1
    # Interpolated IV ≈ sqrt((0.16 + 0.36)/2) ≈ 0.5099, Thalex 0.55 → ~390 bps richer
    edge = report.top[0]["edge_bps"]
    assert 350 < edge < 450
