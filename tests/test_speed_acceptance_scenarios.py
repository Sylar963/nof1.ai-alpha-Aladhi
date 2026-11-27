import json
import math

import pytest

from src.backtest.calibration_report import generate_report
from src.backtest.paradex_replay import ReplayConfig
from src.backtest.replay_dataset import IndexedCapture
from src.backtest.replay_workbench import Workbench
from tests.test_calibration_dataset import capture
from tests.test_speed_acceptance import NS, PERP, SPOT, WINTER, quote, record, trade


def scenario_capture(path, *, forming_gap=False):
    points = [(-1, 60000), *[(t, 59900 if t == 0 else 60100 if t == 899 else 60000)
                              for t in range(901)],
              (901, 60180), (901.1, 60220), *[(t, 60220) for t in range(902, 908)],
              *[(t, 60160) for t in range(908, 921)]]
    rows = []
    for t, price in points:
        spot_session = 'b' if t >= 911 else 'forming-reconnected' if forming_gap and t >= 301 else 'a'
        spot_trade = trade(t, price, session=spot_session)
        spot_trade['payload']['q'] = '.01'
        perp_trade = trade(t, price + 20, PERP)
        perp_trade['payload']['q'] = '.01'
        rows.extend([quote(t, price, session=spot_session), spot_trade,
                     quote(t, price + 20, 'binance_perp_book'), perp_trade,
                     record(t, 'depth', {'pkg_type': 'SNAPSHOT',
                        'bids': [[str(price + 9), '1']], 'asks': [[str(price + 11), '1']]}, 'paradex'),
                     record(t, 'trade', {'trade_type': 'FILL', 'trade_id_low64': str(round((t + 10) * 1000)),
                        'created_at_us': (WINTER + round(t * NS)) // 1000,
                        'price': str(price + 10), 'size': '.01', 'side': 'BUY'}, 'paradex')])
    rows.append(record(910.5, 'gap', {}, session='forming-reconnected' if forming_gap else 'a'))
    if forming_gap:
        rows.append(record(300.5, 'gap', {}))
    source = capture(path, sorted(rows, key=lambda row: row['received_monotonic_ns']))
    manifest = json.loads((source / 'manifest.json').read_text())
    manifest.update(run_id='SYNTHETIC-speed-acceptance-validation', synthetic_test_only=True,
                    description='Controlled prices for testing; never production market history')
    (source / 'manifest.json').write_text(json.dumps(manifest))
    return source


def advance(work, seconds):
    work.advance(round((100 + seconds) * NS))
    return work.state(frame=360)['calibration']['sources'][SPOT]


def test_complete_range_fast_arrival_acceptance_rejection_and_gap_through_replay(tmp_path):
    path = scenario_capture(tmp_path / 'capture')
    work = Workbench(IndexedCapture(path, cache_dir=tmp_path / 'cache'), ReplayConfig(),
                     journal_dir=None, warmup_seconds=0)
    work.engine.baseline_enabled = False
    try:
        before = advance(work, 901.05)
        assert before['opening_range']['status'] == 'complete'
        assert before['opening_range']['levels']['U1'] == pytest.approx(60198)
        assert not any(row['level'] == 'U1' for row in before['encounters'])
        arrived = advance(work, 901.15)
        arrival = next(row for row in arrived['encounters'] if row['level'] == 'U1' and row['direction'] == 1)
        assert arrival['state'] == 'arrival'
        assert arrival['arrival_speed_bps_s'] == pytest.approx(10000 * math.log(60220 / 60180) / .1)
        accepted = advance(work, 907.1)
        acceptance = next(row for row in accepted['encounters'] if row['id'] == arrival['id'])
        assert acceptance['state'] == 'persistent_acceptance'
        assert acceptance['outside_fraction'] == 1
        assert accepted['trade']['horizons']['1']['speed_bps_s'] == 0
        rejected = advance(work, 908.1)
        rejection = next(row for row in rejected['encounters'] if row['id'] == arrival['id'])
        assert rejection['state'] == 'rejection' and rejection['complete']
        assert arrival['state'] == 'arrival' and acceptance['state'] == 'persistent_acceptance'
        visible = work.state()
        work.label_speed({'label': 'release_hedge', 'source': SPOT, 'stream': 'trade',
                          'expected_cursor': visible['calibration_cursor'], 'reason': 'Observed U1 rejection'})
        assert any(row['id'] == arrival['id'] and row['state'] == 'rejection'
                   for row in work.engine.events[-1]['features']['encounters'])
        broken = advance(work, 910.6)
        assert broken['trade']['price'] is None
        assert all(row['complete'] for row in broken['encounters'])
        assert work.state()['calibration']['sources'][PERP]['trade']['price'] is not None
        assert 'mfe_bps' not in json.dumps(work.state()['calibration'])
    finally:
        work.close()


def test_complete_scenario_report_keeps_entry_features_separate_from_outcomes(tmp_path):
    source = scenario_capture(tmp_path / 'capture')
    output = tmp_path / 'report'
    report = json.loads(generate_report(source, output, outcome_seconds=8).read_text())
    encounters = [json.loads(line) for line in (output / 'encounters.ndjson').read_text().splitlines()]
    outcomes = {row['encounter_id']: row for row in
                map(json.loads, (output / 'outcomes.ndjson').read_text().splitlines())}
    up = next(row for row in encounters if row['arrival']['source'] == SPOT
              and row['arrival']['level'] == 'U1' and row['arrival']['direction'] == 1)
    assert up['arrival']['state'] == 'arrival' and up['latest']['state'] == 'rejection'
    assert outcomes[up['id']]['status'] == 'complete'
    assert outcomes[up['id']]['mae_bps'] < 0
    down = next(row for row in encounters if row['arrival']['source'] == SPOT
                and row['arrival']['level'] == 'U1' and row['arrival']['direction'] == -1)
    assert outcomes[down['id']]['status'] == 'censored'
    assert outcomes[down['id']]['reason'] == 'capture_interruption'
    assert report['capture']['capture_manifest']['synthetic_test_only']
    assert report['status'] == 'measurement_only_uncalibrated'


def test_forming_range_interruption_blocks_only_the_affected_market(tmp_path):
    path = scenario_capture(tmp_path / 'capture', forming_gap=True)
    work = Workbench(IndexedCapture(path, cache_dir=tmp_path / 'cache'), ReplayConfig(),
                     journal_dir=None, warmup_seconds=0)
    work.engine.baseline_enabled = False
    try:
        spot = advance(work, 907.1)
        assert spot['opening_range']['status'] == 'incomplete'
        assert not spot['opening_range']['levels'] and not spot['encounters']
        assert work.state()['calibration']['sources'][PERP]['opening_range']['status'] == 'complete'
    finally:
        work.close()
