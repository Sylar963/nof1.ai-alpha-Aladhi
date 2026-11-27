import json
import math
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from src.backend.market_data.calibration import CalibrationConfig
from src.backtest.calibration_report import generate_report
from tests.test_calibration_dataset import capture
from tests.test_speed_acceptance import NS, PERP, SPOT, quote, record, trade


BOOK = 'binance_perp_book'
CFG = CalibrationConfig(session_timezone='UTC', range_start='04:00', range_minutes=1,
                        recent_encounters=1)


def opening():
    return [trade(t, 99 if t == 0 else 101 if t == 1 else 100)
            for t in range(-1, 61)]


def lines(output, name):
    return [json.loads(line) for line in (output / name).read_text().splitlines()]


def run_report(tmp_path, records, horizon=2):
    source = capture(tmp_path / 'capture', records)
    output = tmp_path / 'report'
    return json.loads(generate_report(source, output, CFG, horizon).read_text()), output


def test_full_encounter_export_preserves_arrival_and_separates_future_excursions(tmp_path):
    rows = [*opening(), trade(61, 102), trade(62, 104), trade(63, 100),
            trade(64, 98), trade(65, 105), trade(66, 105)]
    report, output = run_report(tmp_path, rows)
    encounters = lines(output, 'encounters.ndjson')
    outcomes = lines(output, 'outcomes.ndjson')
    assert len(encounters) == len(outcomes) > CFG.recent_encounters
    assert report['counts']['encounters'] == len(encounters)
    assert report['coverage'][SPOT]['opening_range']['levels']['ORH'] == 101
    up = next(row for row in encounters if row['arrival']['level'] == 'ORH'
              and row['arrival']['direction'] == 1)
    assert up['arrival']['samples'] == 1
    assert up['arrival']['arrival_price'] == 102
    assert up['arrival']['arrival_horizons'] == up['latest']['arrival_horizons']
    assert up['arrival']['last_sequence'] == 63 < up['latest']['last_sequence']
    assert 'mfe_bps' not in json.dumps(up)
    outcome = next(row for row in outcomes if row['encounter_id'] == up['id'])
    assert outcome['status'] == 'complete'
    assert outcome['samples'] == 2
    assert outcome['mfe_bps'] == pytest.approx(10000 * (104 / 102 - 1))
    assert outcome['mae_bps'] == outcome['terminal_return_bps'] == pytest.approx(10000 * (100 / 102 - 1))
    down = next(row for row in encounters if row['arrival']['level'] == 'ORH'
                and row['arrival']['direction'] == -1)
    outcome = next(row for row in outcomes if row['encounter_id'] == down['id'])
    assert outcome['mfe_bps'] == pytest.approx(200)
    assert outcome['mae_bps'] == outcome['terminal_return_bps'] == pytest.approx(-500)


@pytest.mark.parametrize(('extra', 'horizon', 'status', 'reason'), [
    ([record(62.5, 'gap', {}), trade(63, 100)], 2, 'censored', 'capture_interruption'),
    ([trade(63, 100, session='b')], 2, 'censored', 'capture_session_changed'),
    ([trade(66, 100)], 2, 'censored', 'stale'),
    ([trade(62.5, 'invalid'), trade(63, 100)], 2, 'censored', 'invalid_trade'),
    ([trade(62.5, 100, event=61.5), trade(63, 100)], 2, 'censored', 'late_trade'),
    ([trade(62.5, 100, event=1000), trade(63, 100)], 2, 'censored', 'trade_clock_mismatch'),
    ([], 2, 'censored', 'capture_tail'),
])
def test_outcomes_stop_at_first_break_and_keep_only_observed_prefix(tmp_path, extra, horizon, status, reason):
    _, output = run_report(tmp_path, [*opening(), trade(61, 102), trade(62, 103), *extra], horizon)
    outcome = next(row for row in lines(output, 'outcomes.ndjson') if row['arrival_sequence'] == 63)
    assert outcome['status'] == status and outcome['reason'] == reason
    assert outcome['samples'] == 1
    assert outcome['terminal_return_bps'] == pytest.approx(10000 * (103 / 102 - 1))


@pytest.mark.parametrize('witness_first', [True, False])
def test_capture_sequence_orders_witness_and_break_at_deadline(tmp_path, witness_first):
    witness = trade(63, 104)
    gap = record(63, 'gap', {})
    boundary = [witness, gap] if witness_first else [gap, witness]
    _, output = run_report(tmp_path, [*opening(), trade(61, 102), trade(62, 103), *boundary])
    outcome = next(row for row in lines(output, 'outcomes.ndjson') if row['arrival_sequence'] == 63)
    if witness_first:
        assert outcome['status'] == 'complete' and outcome['reason'] is None
        assert outcome['witness_sequence'] == outcome['last_sample_sequence'] == 65
        assert outcome['break_sequence'] is None
        assert outcome['samples'] == 2
    else:
        assert outcome['status'] == 'censored' and outcome['reason'] == 'capture_interruption'
        assert outcome['break_sequence'] == 65
        assert outcome['witness_sequence'] is None
        assert outcome['samples'] == 1


def test_duplicate_and_recovered_trades_cannot_supply_outcomes(tmp_path):
    first = trade(62, 103)
    duplicate = trade(62.1, 999)
    duplicate['payload']['a'] = first['payload']['a']
    recovered = {**trade(62.2, 1), 'recovered': True}
    report, output = run_report(tmp_path, [*opening(), trade(61, 102), first, duplicate,
                                           recovered, trade(63, 104)])
    outcome = next(row for row in lines(output, 'outcomes.ndjson') if row['arrival_sequence'] == 63)
    assert outcome['status'] == 'complete' and outcome['samples'] == 2
    assert outcome['mae_bps'] == 0
    assert outcome['mfe_bps'] == pytest.approx(10000 * (104 / 102 - 1))
    assert report['coverage'][SPOT]['counters']['duplicate_trades'] == 1
    assert report['coverage'][SPOT]['counters']['recovered_excluded'] == 1
    with closing(sqlite3.connect(output / 'measurements.sqlite')) as db:
        assert db.execute('SELECT COUNT(*) FROM transactions').fetchone() == (65,)


def test_witness_outside_horizon_cannot_supply_excursion_price(tmp_path):
    _, output = run_report(tmp_path, [*opening(), trade(61, 102), trade(63, 150)], 1)
    outcome = next(row for row in lines(output, 'outcomes.ndjson') if row['arrival_sequence'] == 63)
    assert outcome['status'] == 'unknown'
    assert outcome['reason'] == 'no_in_horizon_transactions'
    assert outcome['samples'] == 0
    assert outcome['terminal_return_bps'] is None


def test_each_venue_retains_its_own_range_and_transaction_clocks(tmp_path):
    rows = []
    for t in range(-1, 64):
        price = 99 if t == 0 else 101 if t == 1 else 102 if t >= 61 else 100
        native_ns = trade(t, price)['source_transaction_time_ns']
        rows.extend([
            trade(t, price),
            trade(t, price + 100, PERP),
            quote(t, price + 200, BOOK),
            record(t, 'trade', {
                'price': str(price + 200), 'size': '2', 'trade_type': 'FILL',
                'trade_id_low64': str(t + 2), 'created_at_us': native_ns // 1000, 'side': 'SELL',
            }, 'paradex'),
        ])
    report, output = run_report(tmp_path, rows)
    assert {source: view['status'] for source, view in report['coverage'].items()} == {
        SPOT: 'observed', BOOK: 'observed', PERP: 'observed', 'paradex': 'observed'}
    assert report['coverage'][BOOK]['accepted_trades'] == 0
    highs = {source: item['opening_range']['levels'].get('ORH')
             for source, item in report['coverage'].items()}
    assert highs == {SPOT: 101, BOOK: None, PERP: 201, 'paradex': 301}
    assert report['chronological_split']['dates_with_complete_ranges_across_trade_sources'] == 1
    assert {row['source'] for row in lines(output, 'outcomes.ndjson')} == set(highs) - {BOOK}
    with closing(sqlite3.connect(output / 'measurements.sqlite')) as db:
        event, wall, receipt, available = db.execute(
            'SELECT event_ns, received_at_ns, original_receipt_ns, available_ns '
            'FROM transactions WHERE source = ? ORDER BY sequence DESC LIMIT 1',
            ('paradex',)).fetchone()
    assert event == wall == native_ns
    assert receipt == available == 163 * NS


def test_fixed_grid_speed_summary_and_required_coverage(tmp_path):
    source = capture(tmp_path / 'capture', [
        quote(0.1, 100), quote(1.1, 101), quote(1.2, 101), quote(1.3, 101),
        quote(3.9, 101, BOOK), trade(4.1, 300, PERP),
    ])
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, 'trade_history': 'recovered.ndjson'}))
    (source / 'recovered.ndjson').write_text('sidecar is intentionally unreadable')
    output = tmp_path / 'report'
    report = json.loads(generate_report(source, output, CFG).read_text())
    assert report['status'] == 'measurement_only_uncalibrated'
    assert report['capture']['recovered_sidecar_excluded']
    assert {SPOT, BOOK, PERP, 'paradex'} == set(report['coverage'])
    assert report['coverage']['paradex']['status'] == 'missing'
    assert report['coverage'][BOOK]['accepted_quotes'] == 1
    assert report['coverage'][PERP]['accepted_trades'] == 1
    speed = report['speed'][SPOT]['quote']['horizons']['1']
    assert speed['samples'] == 3
    assert speed['mean_bps_s'] == pytest.approx(10000 * math.log(1.01) / 3)
    assert speed['max_abs_bps_s'] == pytest.approx(10000 * math.log(1.01))
    assert report['chronological_split']['status'] == 'guidance_only'
    assert {warning['code'] for warning in report['warnings']} >= {'short_sample', 'missing_source'}


def test_report_refuses_overwriting_and_records_failed_input(tmp_path):
    source = capture(tmp_path / 'capture', [quote(0, 100)])
    status_path = source / 'status.json'
    status = json.loads(status_path.read_text())
    status_path.write_text(json.dumps({**status, 'closed': False}))
    output = tmp_path / 'report'
    with pytest.raises(ValueError, match='healthy closed'):
        generate_report(source, output)
    report_path = output / 'report.json'
    original = report_path.read_text()
    assert json.loads(original)['status'] == 'failed'
    assert json.loads(original)['capture']['capture_path'] == str(source)
    with pytest.raises(FileExistsError):
        generate_report(source, output)
    assert report_path.read_text() == original


def test_cli_help_is_clean_and_cli_runs_without_overwrite(tmp_path):
    cli = Path(__file__).resolve().parents[1] / 'scripts' / 'calibrate_market_data.py'
    help_result = subprocess.run([sys.executable, cli, '--help'], cwd=tmp_path,
                                 capture_output=True, text=True, check=True)
    assert not help_result.stderr
    assert '--capture' in help_result.stdout and '--outcome-seconds' in help_result.stdout
    source = capture(tmp_path / 'capture', [quote(0, 100)])
    output = tmp_path / 'report'
    command = [sys.executable, cli, '--capture', source, '--output', output]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert Path(result.stdout.strip()) == output / 'report.json'
    second = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert second.returncode != 0 and 'exists' in second.stderr
    invalid = subprocess.run([*command, '--outcome-seconds', 'inf'], cwd=tmp_path,
                             capture_output=True, text=True)
    assert invalid.returncode != 0 and 'finite' in invalid.stderr
