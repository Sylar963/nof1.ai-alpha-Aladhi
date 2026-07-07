from datetime import datetime, timedelta

import pytest

from src.backtest.metrics import compute_metrics, format_report

T0 = datetime(2026, 7, 1, 12, 0, 0)


def make_result():
    curve = [
        {"timestamp": T0, "equity": 11_000.0},
        {"timestamp": T0 + timedelta(minutes=5), "equity": 9_900.0},
        {"timestamp": T0 + timedelta(minutes=10), "equity": 10_450.0},
    ]
    trades = [
        {
            "asset": "BTC", "side": "long", "size": 1.0,
            "entry_price": 100.0, "exit_price": 200.0,
            "pnl": 100.0, "fees": 2.0,
            "opened_at": T0, "closed_at": T0 + timedelta(hours=1),
            "hold_seconds": 3600.0, "reason": "tp",
        },
        {
            "asset": "BTC", "side": "long", "size": 1.0,
            "entry_price": 100.0, "exit_price": 50.0,
            "pnl": -50.0, "fees": 1.5,
            "opened_at": T0, "closed_at": T0 + timedelta(hours=2),
            "hold_seconds": 7200.0, "reason": "sl",
        },
        {
            "asset": "ETH", "side": "short", "size": 2.0,
            "entry_price": 100.0, "exit_price": 85.0,
            "pnl": 30.0, "fees": 1.0,
            "opened_at": T0, "closed_at": T0 + timedelta(hours=3),
            "hold_seconds": 10_800.0, "reason": "close",
        },
    ]
    return {
        "starting_equity": 10_000.0,
        "final_equity": 10_450.0,
        "equity_curve": curve,
        "closed_trades": trades,
        "open_positions": {},
        "fees_paid": 4.5,
        "skipped": [{"timestamp": T0, "asset": "SOL", "action": "buy", "reason": "missing sl_price"}],
        "clamp_notes": [],
    }


def test_metrics_computation():
    report = compute_metrics(make_result())

    assert report["total_return_pct"] == pytest.approx(4.5)
    assert report["trade_count"] == 3
    assert report["win_rate"] == pytest.approx(2 / 3)
    assert report["expectancy"] == pytest.approx(80.0 / 3)
    assert report["profit_factor"] == pytest.approx(130.0 / 50.0)
    assert report["max_drawdown_pct"] == pytest.approx((11_000 - 9_900) / 11_000 * 100)
    assert report["total_fees"] == pytest.approx(4.5)
    assert report["avg_hold_seconds"] == pytest.approx(7200.0)
    assert report["median_hold_seconds"] == pytest.approx(7200.0)
    assert report["skipped_count"] == 1

    btc = report["per_asset"]["BTC"]
    assert btc["trades"] == 2
    assert btc["win_rate"] == pytest.approx(0.5)
    assert btc["net_pnl"] == pytest.approx(50.0)
    eth = report["per_asset"]["ETH"]
    assert eth["trades"] == 1
    assert eth["win_rate"] == pytest.approx(1.0)


def test_metrics_empty_result():
    report = compute_metrics({
        "starting_equity": 10_000.0,
        "final_equity": 10_000.0,
        "equity_curve": [],
        "closed_trades": [],
        "open_positions": {},
        "fees_paid": 0.0,
        "skipped": [],
        "clamp_notes": [],
    })
    assert report["total_return_pct"] == 0.0
    assert report["win_rate"] == 0.0
    assert report["profit_factor"] == 0.0
    assert report["max_drawdown_pct"] == 0.0


def test_format_report_renders_key_figures():
    text = format_report(compute_metrics(make_result()))
    assert "Total return:  +4.50%" in text
    assert "Trades closed: 3" in text
    assert "Win rate:      66.7%" in text
    assert "Max drawdown:  10.00%" in text
    assert "BTC" in text and "ETH" in text
    assert "missing sl_price" in text
