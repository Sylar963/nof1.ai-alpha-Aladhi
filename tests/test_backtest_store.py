import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.store import group_rows, load_cycles_from_session, row_to_section
from src.database.models import MarketData, create_tables

T0 = datetime(2026, 7, 1, 12, 0, 0)


def make_row(asset, ts, price, interval="cycle", indicators=None):
    return MarketData(
        asset=asset,
        timestamp=ts,
        interval=interval,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000_000.0,
        open_interest=42.0,
        funding_rate=0.0001,
        indicators=indicators,
    )


def section_json(asset, price):
    return json.dumps({
        "asset": asset,
        "current_price": price,
        "intraday": {
            "sma99": price * 0.99,
            "avwap": price * 0.995,
            "keltner": {
                "middle": price, "upper": price * 1.01,
                "lower": price * 0.99, "position": "inside",
            },
            "opening_range": {},
        },
        "long_term": {"interval": "4h", "sma99": price * 0.98},
        "open_interest": 42.0,
        "prev_day_price": price * 0.97,
        "volume_24h": 1_000_000.0,
        "funding_rate": 0.0001,
        "funding_annualized_pct": 0.0001 * 24 * 365 * 100,
    })


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    create_tables(engine)
    sess = sessionmaker(bind=engine)()
    yield sess
    sess.close()


def seed(session):
    rows = [
        make_row("BTC", T0, 50_000.0, indicators=section_json("BTC", 50_000.0)),
        make_row("ETH", T0 + timedelta(seconds=5), 3_000.0, indicators=section_json("ETH", 3_000.0)),
        make_row("SOL", T0 + timedelta(seconds=20), 150.0),
        make_row("BTC", T0 + timedelta(minutes=5), 50_500.0, indicators=section_json("BTC", 50_500.0)),
        make_row("ETH", T0 + timedelta(minutes=5, seconds=3), 3_050.0, indicators=section_json("ETH", 3_050.0)),
        make_row("BTC", T0 - timedelta(hours=1), 49_000.0, interval="5m"),
    ]
    session.add_all(rows)
    session.commit()


def test_cycles_grouped_by_tolerance_window(session):
    seed(session)
    cycles = load_cycles_from_session(session)

    assert len(cycles) == 2
    assert cycles[0]["timestamp"] == T0
    assert set(cycles[0]["sections"]) == {"BTC", "ETH", "SOL"}
    assert set(cycles[1]["sections"]) == {"BTC", "ETH"}
    assert cycles[0]["sections"]["BTC"]["current_price"] == pytest.approx(50_000.0)
    assert cycles[1]["sections"]["BTC"]["current_price"] == pytest.approx(50_500.0)


def test_non_cycle_intervals_are_excluded(session):
    seed(session)
    cycles = load_cycles_from_session(session)
    all_ts = [c["timestamp"] for c in cycles]
    assert T0 - timedelta(hours=1) not in all_ts


def test_start_end_filtering(session):
    seed(session)
    cycles = load_cycles_from_session(session, start=T0 + timedelta(minutes=1))
    assert len(cycles) == 1
    assert cycles[0]["timestamp"] == T0 + timedelta(minutes=5)

    cycles = load_cycles_from_session(session, end=T0 + timedelta(minutes=1))
    assert len(cycles) == 1
    assert cycles[0]["timestamp"] == T0


def test_row_without_indicators_falls_back_to_columns(session):
    seed(session)
    cycles = load_cycles_from_session(session)
    sol = cycles[0]["sections"]["SOL"]
    assert sol["asset"] == "SOL"
    assert sol["current_price"] == pytest.approx(150.0)
    assert sol["volume_24h"] == pytest.approx(1_000_000.0)
    assert sol["funding_rate"] == pytest.approx(0.0001)


def test_group_rows_splits_beyond_tolerance():
    rows = [
        make_row("BTC", T0, 100.0),
        make_row("ETH", T0 + timedelta(seconds=59), 10.0),
        make_row("BTC", T0 + timedelta(seconds=61), 101.0),
    ]
    cycles = group_rows(rows, tolerance_seconds=60.0)
    assert len(cycles) == 2
    assert set(cycles[0]["sections"]) == {"BTC", "ETH"}
    assert set(cycles[1]["sections"]) == {"BTC"}


def test_row_to_section_handles_bad_json():
    row = make_row("BTC", T0, 100.0, indicators="{not json")
    section = row_to_section(row)
    assert section["asset"] == "BTC"
    assert section["current_price"] == pytest.approx(100.0)
