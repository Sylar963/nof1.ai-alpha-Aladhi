import json
import math
from datetime import datetime, timezone

import pytest

from src.backend.market_data.calibration import CalibrationConfig, SpeedAcceptanceEngine


NS = 1_000_000_000
WINTER = int(datetime(2026, 11, 25, 4, tzinfo=timezone.utc).timestamp()) * NS
SUMMER = int(datetime(2026, 7, 25, 3, tzinfo=timezone.utc).timestamp()) * NS
SPOT = 'binance_spot'
PERP = 'binance_perp_market'


def record(t, kind, payload, source=SPOT, *, event=None, base=WINTER, session='a'):
    return {'source': source, 'session_id': session, 'kind': kind, 'payload': payload,
            'market': 'BTC-USD-PERP' if source == 'paradex' else 'BTCUSDT',
            'received_at_ns': base + round(t * NS),
            'received_monotonic_ns': round((t + 100) * NS),
            'source_transaction_time_ns': base + round((t if event is None else event) * NS)}


def quote(t, price, source=SPOT, **kwargs):
    return record(t, 'bbo', {'b': str(price), 'a': str(price)}, source, **kwargs)


def trade(t, price, source=SPOT, **kwargs):
    return record(t, 'trade', {'p': str(price), 'q': '1.5', 'm': False,
                               'a': round((t + 10) * 1000)}, source, **kwargs)


def src(engine, source=SPOT):
    return engine.snapshot()['sources'][source]


def opening_range(engine, base=WINTER, source=SPOT, skip=(), receipt_offset=0):
    for t in [-1, *range(0, 901)]:
        if t not in skip:
            point = trade(t, 99 if t == 0 else 101 if t == 899 else 100, source, base=base)
            point['received_monotonic_ns'] += receipt_offset
            engine.observe(point)


def test_grid_is_past_only_equal_weighted_and_snapshot_does_not_train():
    normal, burst = SpeedAcceptanceEngine(), SpeedAcceptanceEngine()
    for t in range(130):
        for engine in (normal, burst):
            engine.observe(quote(t + 0.1, 100 * math.exp(t * 0.0001)))
        for offset in (0.2, 0.3, 0.4):
            burst.observe(quote(t + offset, 100 * math.exp(t * 0.0001)))
        burst.snapshot(round((100 + t + 9) * NS))
    for engine in (normal, burst):
        engine.observe(quote(130.1, 10))
    horizons = src(normal)['quote']['horizons']
    assert horizons == src(burst)['quote']['horizons']
    for h in (1, 5, 30, 120):
        item = horizons[str(h)]
        assert item['speed_bps_s'] == pytest.approx(1)
        assert item['acceleration_bps_s2'] == pytest.approx(0, abs=1e-9)
        assert item['baseline_samples'] == 129 - h
    previous = normal.snapshot()
    normal.snapshot(400 * NS)
    assert normal.snapshot() == previous
    normal.observe(quote(131.1, 10))
    assert src(normal)['quote']['horizons']['1']['speed_bps_s'] < 0
    json.dumps(normal.snapshot(), allow_nan=False)


def test_missing_feeds_depth_delta_recovered_and_bad_numbers():
    e = SpeedAcceptanceEngine()
    e.observe(record(0, 'depth', {'b': [['100', '1']], 'a': [['102', '1']]}))
    e.observe(record(1, 'depth', {'pkg_type': 'DELTA', 'best_bid_api_price': '100',
                                  'best_ask_api_price': '101'}, 'paradex'))
    e.observe(record(2, 'trade', {'recovered': True}, 'paradex'))
    e.observe(quote(3, 'inf'))
    assert src(e)['quote']['price'] is None
    assert src(e, 'paradex')['trade']['price'] is None
    assert src(e, 'paradex')['quote']['price'] is None
    e.observe(record(4, 'depth', {'pkg_type': 'SNAPSHOT', 'best_bid_api_price': '100',
                                  'best_ask_api_price': '102', 'best_bid_price': '800'}, 'paradex'))
    assert src(e, 'paradex')['quote']['price'] == 101
    assert src(e)['opening_range']['levels'] == {}
    assert src(e, PERP)['trade']['horizons']['5']['reason'] == 'missing'
    assert e.snapshot()['basis']['perp_minus_spot']['bps'] is None
    json.dumps(e.snapshot(), allow_nan=False)


def test_quote_basis_and_stale_gap_and_session_resets():
    e = SpeedAcceptanceEngine()
    for t in range(10):
        e.observe(quote(t, 100 + t))
    e.observe(quote(9.1, 110, 'binance_perp_book'))
    basis = e.snapshot()['basis']['perp_minus_spot']
    assert basis['bps'] == pytest.approx(10000 * (110 / 109 - 1))
    assert e.snapshot(300 * NS)['sources'][SPOT]['quote']['horizons']['1']['reason'] == 'stale'
    e.observe(quote(40, 1000))
    assert src(e)['quote']['horizons']['5']['speed_bps_s'] is None
    for t in range(41, 50):
        e.observe(quote(t, 1000))
    e.observe(record(50, 'gap', {}))
    assert src(e)['quote']['price'] is None
    e.observe(quote(51, 1500, session='b'))
    assert src(e)['quote']['horizons']['1']['baseline_samples'] == 0


@pytest.mark.parametrize('base', [WINTER, SUMMER])
def test_exact_event_time_range_cutoff_independent_of_receipt_and_target_formulas(base):
    e = SpeedAcceptanceEngine()
    for t in [-1, *range(0, 900)]:
        e.observe(trade(t + 0.2, 99 if t == 0 else 101 if t == 899 else 100, event=t, base=base))
    assert src(e)['opening_range']['levels'] == {}
    e.observe(trade(900.2, 104, event=900, base=base))
    r = src(e)['opening_range']
    assert (r['status'], r['start_ns'], r['end_ns'], r['trade_count']) == (
        'complete', base, base + 900 * NS, 900)
    assert r['levels'] == pytest.approx({
        'ORH': 101, 'ORL': 99, 'ORM': 100,
        **{f'U{i}': 100.33 + 4 * (i - 1) for i in range(1, 5)},
        **{f'D{i}': 99.67 - 4 * (i - 1) for i in range(1, 5)}})
    assert src(e, PERP)['opening_range']['levels'] == {}


def test_partial_and_gapped_ranges_never_offer_levels():
    for skipped in (range(-1, 300), range(200, 600)):
        e = SpeedAcceptanceEngine()
        opening_range(e, skip=skipped)
        r = src(e)['opening_range']
        assert r['status'] == 'incomplete' and r['reason'] and r['levels'] == {}


def test_trade_id_dedup_and_late_trade_invalidates_or_without_retroactive_rewrite():
    e = SpeedAcceptanceEngine()
    opening_range(e)
    before = src(e)['opening_range']
    e.observe(dict(trade(899, 101), received_monotonic_ns=1001 * NS))
    assert src(e)['counters']['duplicate_trades'] == 1
    e.observe(trade(902, 112, event=880))
    assert src(e)['counters']['late_trades'] == 1
    assert before['status'] == 'complete'
    assert src(e)['opening_range']['status'] == 'incomplete'
    assert src(e)['trade']['price'] == 100


def test_paradex_transactions_require_price_forming_type_and_native_time():
    e = SpeedAcceptanceEngine()
    p = {'price': '105.20', 'size': '2', 'trade_type': 'FILL', 'trade_id_low64': '101',
         'created_at_us': WINTER // 1000, 'side': 'SELL'}
    e.observe(record(0, 'trade', p, 'paradex'))
    e.observe(record(1, 'trade', dict(p, trade_type='LIQUIDATION', price='99'), 'paradex'))
    e.observe(record(2, 'trade', p, 'paradex'))
    assert src(e, 'paradex')['trade']['price'] == 105.2
    assert src(e, 'paradex')['counters']['duplicate_trades'] == 1


def test_encounters_include_skipped_bands_time_weighting_persistence_and_rejection():
    e = SpeedAcceptanceEngine(CalibrationConfig(min_dwell_s=3, min_encounter_samples=3))
    opening_range(e)
    rows = e.observe(trade(901, 102))
    assert {r['level'] for r in rows} >= {'ORH', 'U1'}
    initial = next(r for r in rows if r['level'] == 'ORH')
    assert initial['state'] == 'arrival' and initial['skipped_touch']
    for t in (901.1, 901.2, 902, 903, 904, 905):
        rows = e.observe(trade(t, 102))
    accepted = next(r for r in rows if r['id'] == initial['id'])
    assert accepted['state'] == 'persistent_acceptance'
    assert accepted['outside_fraction'] == pytest.approx(1)
    assert accepted['outside_s'] == pytest.approx(4)
    assert accepted['signed_volume'] == pytest.approx(accepted['volume'])
    rows = e.observe(trade(906, 100.5))
    rejected = next(r for r in rows if r['id'] == initial['id'])
    assert rejected['state'] == 'rejection' and rejected['reclaims'] == 1
    assert rejected['complete'] and rejected['arrival_speed_bps_s'] > 0
    assert rejected['samples'] == 8
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize('offset,included', [(2, True), (2.000000001, False)])
@pytest.mark.parametrize('receipt_offset', [0, 2**53 + 1])
def test_encounter_deadline_includes_exact_cutoff_only(offset, included, receipt_offset):
    e = SpeedAcceptanceEngine(CalibrationConfig(encounter_timeout_s=2.0, min_dwell_s=1,
                                              min_encounter_samples=2))
    opening_range(e, receipt_offset=receipt_offset)
    arrival = trade(901, 102)
    arrival['received_monotonic_ns'] += receipt_offset
    initial = next(row for row in e.observe(arrival) if row['level'] == 'ORH')
    point = trade(901 + offset, 102.5)
    point['received_monotonic_ns'] += receipt_offset
    rows = e.observe(point)
    final = next(row for row in rows if row['id'] == initial['id'])
    assert final['complete']
    assert final['samples'] == (2 if included else 1)
    assert final['volume'] == (3 if included else 1.5)
    assert final['last_price'] == (102.5 if included else 102)
    expected_ns = point['received_monotonic_ns'] if included else initial['last_ns']
    assert final['last_ns'] == expected_ns
    assert final['last_receipt_ns'] == final['last_ns']
    assert final['state'] == ('persistent_acceptance' if included else initial['state'])


def test_calibration_defaults_use_runtime_settings_and_allow_explicit_overrides(monkeypatch):
    from src.backend.config_loader import CONFIG

    monkeypatch.setitem(CONFIG, 'calibration_session_timezone', 'UTC')
    monkeypatch.setitem(CONFIG, 'calibration_range_start', '03:00')
    monkeypatch.setitem(CONFIG, 'calibration_range_minutes', 5)
    monkeypatch.setitem(CONFIG, 'calibration_band_bps', 2.5)
    default = CalibrationConfig()
    assert (default.session_timezone, default.range_start, default.range_minutes, default.band_bps) == (
        'UTC', '03:00', 5, 2.5)
    explicit = CalibrationConfig(range_minutes=10, band_bps=4)
    assert explicit.range_minutes == 10 and explicit.band_bps == 4


def test_configured_window_restarts_on_dst_zone_and_cutoff_requires_observed_witness():
    cfg = CalibrationConfig(session_timezone='UTC', range_start='04:00', range_minutes=1)
    e = SpeedAcceptanceEngine(cfg)
    for t in range(60):
        e.observe(trade(t, 100 + t))
    r = e.snapshot(161 * NS)['sources'][SPOT]['opening_range']
    assert r['reason'] == 'missing_cutoff_coverage' and r['levels'] == {}
    e.observe(trade(60, 170))
    assert src(e)['opening_range']['levels']['ORH'] == 159
    assert src(e)['opening_range']['end_ns'] == WINTER + 60 * NS


def test_availability_clock_preserves_regressing_receipt_and_source_separation():
    e = SpeedAcceptanceEngine()
    e.observe(quote(0, 100))
    e.observe(quote(1.1, 111, 'binance_perp_book'))
    e.observe(dict(quote(1, 101), available_monotonic_ns=round(101.1 * NS)))
    assert src(e)['quote']['received_ns'] == round(101.1 * NS)
    assert src(e)['quote']['original_receipt_ns'] == 101 * NS
    assert src(e)['counters']['delayed_submissions'] == 1
    assert src(e)['trade']['price'] is None


def test_return_crossing_creates_distinct_opposite_encounter_and_observations_are_detached():
    e = SpeedAcceptanceEngine()
    opening_range(e)
    rows = e.observe(trade(901, 102))
    original = next(r for r in rows if r['level'] == 'ORH')
    original['arrival_horizons']['1']['reason'] = 'changed_by_caller'
    snap = next(r for r in src(e)['encounters'] if r['id'] == original['id'])
    assert snap['arrival_horizons']['1']['reason'] != 'changed_by_caller'
    rows = e.observe(trade(902, 100.5))
    reverse = next(r for r in rows if r['level'] == 'ORH' and r['direction'] == -1)
    assert reverse['state'] == 'arrival' and reverse['skipped_touch']
    assert reverse['id'] != original['id']


def test_unavailable_paradex_api_quote_invalidates_basis_until_valid_quote():
    e = SpeedAcceptanceEngine()
    e.observe(quote(0, 100))
    p = dict(pkg_type='SNAPSHOT', best_bid_api_price='99', best_ask_api_price='101')
    e.observe(record(0.1, 'depth', p, 'paradex'))
    assert e.snapshot()['basis']['paradex_minus_spot']['bps'] == 0
    e.observe(record(0.2, 'depth', dict(p, best_ask_api_price=None), 'paradex'))
    assert e.snapshot()['basis']['paradex_minus_spot']['bps'] is None


def test_delayed_submission_cannot_refresh_an_old_quote_or_bridge_a_stale_trade():
    e = SpeedAcceptanceEngine()
    e.observe(quote(0, 100))
    e.observe(quote(10, 200, 'binance_perp_book'))
    e.observe(quote(1, 101))
    assert src(e)['quote']['reason'] == 'stale'
    assert e.snapshot()['basis']['perp_minus_spot']['bps'] is None
    e.observe(trade(2, 10))
    assert src(e)['trade']['reason'] == 'stale'
    e.observe(trade(10.1, 100))
    assert src(e)['trade']['horizons']['1']['speed_bps_s'] is None


def test_basis_requires_matching_underlyings_and_declares_quote_currency_assumption():
    e = SpeedAcceptanceEngine()
    e.observe(quote(0, 100))
    p = dict(bid_price='101', ask_price='101')
    e.observe(dict(record(0.1, 'bbo', p, 'paradex'), market='ETH-USD-PERP'))
    assert e.snapshot()['basis']['paradex_minus_spot']['reason'] == 'incompatible_markets'
    e.observe(quote(0.2, 110, 'binance_perp_book'))
    assert e.snapshot()['basis']['perp_minus_spot']['quote_currencies'] == ['USDT', 'USDT']


def test_paradex_null_api_scalars_fall_back_to_populated_replacement_book():
    e = SpeedAcceptanceEngine()
    p = {'pkg_type': 'SNAPSHOT', 'best_bid_api_price': None, 'best_ask_api_price': None,
         'best_bid_price': '1000', 'best_ask_price': '1001',
         'bids': [['105', '0'], ['99', '0.5'], ['100', '2']],
         'asks': [['103', '1'], ['98', '0.0'], ['102', '1.5']]}
    e.observe(record(0, 'depth', p, 'paradex'))
    assert src(e, 'paradex')['quote']['price'] == 101
    e.observe(record(1, 'bbo', dict(bid_price=None, ask_price=None), 'paradex'))
    assert src(e, 'paradex')['quote']['price'] == 101
    assert src(e, 'paradex')['quote']['age_s'] == 1
    e.observe(record(2, 'depth', dict(p, bids=[['110', '1']]), 'paradex'))
    assert src(e, 'paradex')['quote']['price'] is None
    e.observe(record(3, 'depth', dict(p, pkg_type='DELTA'), 'paradex'))
    assert src(e, 'paradex')['quote']['price'] is None
    e.observe(record(4, 'depth', dict(p, asks=[['102', '-1']]), 'paradex'))
    assert src(e, 'paradex')['quote']['price'] is None


def test_invalid_first_opening_print_invalidates_session_after_prior_days_closed_range():
    e = SpeedAcceptanceEngine()
    e.observe(trade(-1, 100))
    e.observe(trade(0, 'nan'))
    for t in range(1, 901):
        e.observe(trade(t, 100))
    r = src(e)['opening_range']
    assert r['status'] == 'incomplete' and r['reason'] == 'invalid_range_trade'
    assert r['levels'] == {}


def test_invalid_transaction_severs_active_encounter_and_trade_evidence():
    e = SpeedAcceptanceEngine(CalibrationConfig(min_dwell_s=2))
    opening_range(e)
    for t in range(901, 905):
        rows = e.observe(trade(t, 102))
    active = next(r for r in rows if r['level'] == 'ORH')
    assert active['state'] == 'persistent_acceptance'
    invalid = trade(905, 102)
    invalid['payload']['q'] = '-1'
    rows = e.observe(invalid)
    closed = next(r for r in rows if r['id'] == active['id'])
    assert closed['complete'] and closed['state'] == 'insufficient_evidence'
    assert src(e)['trade']['price'] is None
    assert src(e)['opening_range']['status'] == 'complete'
    e.observe(trade(906, 103))
    assert src(e)['trade']['horizons']['1']['speed_bps_s'] is None


@pytest.mark.parametrize('end', [2.5, 200])
def test_quiet_clock_projection_matches_unrelated_control_without_training(end):
    e = SpeedAcceptanceEngine()
    e.observe(quote(0, 100))
    e.observe(quote(1.1, 110))
    before = e.snapshot()
    projected = e.snapshot(round((100 + end) * NS))['sources'][SPOT]['quote']
    assert e.snapshot() == before
    e.observe(record(end, 'control', {}, 'paradex'))
    assert src(e)['quote'] == projected
    if end == 2.5:
        assert projected['horizons']['1']['speed_bps_s'] == pytest.approx(10000 * math.log(1.1))


def test_report_accessor_and_fixed_grid_hook_follow_only_accepted_evidence():
    samples = []
    e = SpeedAcceptanceEngine(on_grid_sample=lambda *args: samples.append(args))
    e.observe(trade(0, 100))
    first = e.accepted_trade.copy()
    assert first['price'] == 100 and first['source'] == SPOT
    e.observe(trade(0, 100))
    assert e.accepted_trade is None
    e.observe(trade(1.1, 110))
    assert e.accepted_trade['price'] == 110
    assert samples and samples[-1][:3] == (SPOT, 'trade', 101 * NS)
    count = len(samples)
    e.snapshot(102_500_000_000)
    assert len(samples) == count
    revision = e.trade_revisions[SPOT]
    e.observe(trade(2, 'nan'))
    assert e.accepted_trade is None and e.trade_revisions[SPOT] > revision
    assert e.opening_range(SPOT)['status'] == 'incomplete'


@pytest.mark.parametrize('event,reason', [(905.5, 'late_trade'), (950, 'trade_clock_mismatch')])
def test_rejected_trade_cannot_preserve_acceptance_across_a_continuity_break(event, reason):
    e = SpeedAcceptanceEngine(CalibrationConfig(min_dwell_s=2))
    opening_range(e)
    for t in range(901, 907):
        rows = e.observe(trade(t, 102))
    prior = next(row for row in rows if row['level'] == 'ORH')
    assert prior['state'] == 'persistent_acceptance'
    rows = e.observe(trade(906.5, 90, event=event))
    assert e.accepted_trade is None and e.trade_breaks[SPOT] == reason
    closed = next(row for row in rows if row['id'] == prior['id'])
    assert closed['complete'] and closed['state'] == 'insufficient_evidence'
    view = src(e)['trade']
    assert view['price'] == 102 and view['reason'] == reason
    assert view['horizons']['1']['speed_bps_s'] is None
    e.observe(trade(907, 102))
    assert src(e)['trade']['horizons']['1']['speed_bps_s'] is None
    assert not any(not row['complete'] for row in src(e)['encounters'])


def test_pre_session_trade_silence_does_not_invalidate_an_exact_opening_print():
    plain, prefixed = SpeedAcceptanceEngine(), SpeedAcceptanceEngine()
    prefix = trade(-55, 100)
    prefix['payload']['a'] = 0
    prefixed.observe(prefix)
    for t in range(901):
        price = 99 if t == 0 else 101 if t == 899 else 100
        for engine in (plain, prefixed):
            engine.observe(trade(t, price))
    expected, actual = src(plain)['opening_range'], src(prefixed)['opening_range']
    assert expected['status'] == 'complete'
    assert actual == expected


@pytest.mark.parametrize('bad_event', [-2, 1000])
def test_rejected_out_of_range_trade_cannot_hide_a_hole_in_forming_range(bad_event):
    e = SpeedAcceptanceEngine()
    for t in range(-1, 101):
        e.observe(trade(t, 99 if t == 0 else 101 if t == 100 else 100))
    e.observe(trade(101, 90, event=bad_event))
    for t in range(200, 901):
        e.observe(trade(t, 100))
    assert src(e)['opening_range']['status'] == 'incomplete'
    assert not src(e)['opening_range']['levels']


def test_malformed_future_timestamp_cannot_move_range_to_tomorrow():
    e = SpeedAcceptanceEngine()
    for t in range(-1, 101):
        e.observe(trade(t, 99 if t == 0 else 101 if t == 100 else 100))
    start = src(e)['opening_range']['start_ns']
    invalid = trade(101, 90, event=86400)
    invalid['payload']['q'] = 'invalid'
    e.observe(invalid)
    assert src(e)['opening_range']['start_ns'] == start
    for t in range(102, 901):
        e.observe(trade(t, 100))
    assert src(e)['opening_range']['status'] == 'incomplete'
    assert src(e)['opening_range']['start_ns'] == start
