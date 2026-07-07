"""Regression tests for the code-enforced risk engine and PnL booking in bot_engine."""

import json
import logging
from datetime import datetime, timedelta, UTC
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.backend.bot_engine as bot_engine_module
from src.backend.bot_engine import BotState, TradingBotEngine
from src.backend.config_loader import CONFIG
from src.backend.models.trade_proposal import TradeProposal


def make_engine(tmp_path) -> TradingBotEngine:
    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.logger = logging.getLogger("test_risk_engine")
    engine.state = BotState()
    engine.active_trades = []
    engine.pending_proposals = []
    engine.is_paused = False
    engine.pause_reason = None
    engine.peak_account_value = 0.0
    engine._execution_failure_streak = 0
    engine.diary_path = tmp_path / "diary.jsonl"
    engine._risk_state_path = tmp_path / "risk_state.json"
    engine._risk_day = None
    engine._day_start_value = 0.0
    engine.hedge_manager = None
    engine.on_state_update = None
    return engine


def read_diary(engine):
    if not engine.diary_path.exists():
        return []
    return [json.loads(line) for line in engine.diary_path.read_text().splitlines() if line.strip()]


class TestValidateRiskLevels:
    def test_missing_stop_rejected(self):
        reason = TradingBotEngine._validate_risk_levels('buy', 100.0, 110.0, None)
        assert reason and 'sl_price' in reason

    def test_inverted_stop_rejected_for_buy(self):
        assert TradingBotEngine._validate_risk_levels('buy', 100.0, 110.0, 105.0)

    def test_inverted_tp_rejected_for_sell(self):
        assert TradingBotEngine._validate_risk_levels('sell', 100.0, 105.0, 110.0)

    def test_valid_buy_levels_pass(self):
        assert TradingBotEngine._validate_risk_levels('buy', 100.0, 110.0, 95.0) is None

    def test_valid_sell_levels_pass(self):
        assert TradingBotEngine._validate_risk_levels('sell', 100.0, 90.0, 105.0) is None


class TestClampAllocation:
    def test_risk_cap_clamps_oversized_allocation(self, tmp_path, monkeypatch):
        engine = make_engine(tmp_path)
        engine.state.total_value = 10_000.0
        engine.state.positions = []
        monkeypatch.setitem(CONFIG, "max_risk_per_trade_pct", 1.0)
        monkeypatch.setitem(CONFIG, "max_gross_leverage", 100.0)
        allocation, notes = engine._clamp_allocation(
            'BTC', 'buy', 50_000.0, 100_000.0, 99_000.0, 0.0,
        )
        assert allocation == pytest.approx(10_000.0)
        assert notes

    def test_gross_leverage_cap_counts_other_assets(self, tmp_path, monkeypatch):
        engine = make_engine(tmp_path)
        engine.state.total_value = 10_000.0
        engine.state.positions = [
            {"venue": "hyperliquid", "symbol": "ETH", "quantity": 5.0, "current_price": 5_000.0},
        ]
        monkeypatch.setitem(CONFIG, "max_risk_per_trade_pct", 0.0)
        monkeypatch.setitem(CONFIG, "max_gross_leverage", 3.0)
        allocation, notes = engine._clamp_allocation(
            'BTC', 'buy', 20_000.0, 100_000.0, 95_000.0, 0.0,
        )
        assert allocation == pytest.approx(5_000.0)
        assert notes

    def test_gross_cap_reached_zeroes_allocation(self, tmp_path, monkeypatch):
        engine = make_engine(tmp_path)
        engine.state.total_value = 10_000.0
        engine.state.positions = [
            {"venue": "hyperliquid", "symbol": "ETH", "quantity": 8.0, "current_price": 5_000.0},
        ]
        monkeypatch.setitem(CONFIG, "max_risk_per_trade_pct", 0.0)
        monkeypatch.setitem(CONFIG, "max_gross_leverage", 3.0)
        allocation, notes = engine._clamp_allocation(
            'BTC', 'buy', 20_000.0, 100_000.0, 95_000.0, 0.0,
        )
        assert allocation == 0.0
        assert notes

    def test_within_caps_untouched(self, tmp_path, monkeypatch):
        engine = make_engine(tmp_path)
        engine.state.total_value = 100_000.0
        engine.state.positions = []
        monkeypatch.setitem(CONFIG, "max_risk_per_trade_pct", 1.0)
        monkeypatch.setitem(CONFIG, "max_gross_leverage", 3.0)
        allocation, notes = engine._clamp_allocation(
            'BTC', 'buy', 10_000.0, 100_000.0, 95_000.0, 0.0,
        )
        assert allocation == pytest.approx(10_000.0)
        assert notes == []


class TestDrawdownPersistence:
    def test_peak_persists_across_restart(self, tmp_path):
        engine = make_engine(tmp_path)
        engine._update_peak_and_check_drawdown(10_000.0)
        engine._update_peak_and_check_drawdown(12_000.0)

        engine2 = make_engine(tmp_path)
        engine2._load_risk_state()
        assert engine2.peak_account_value == pytest.approx(12_000.0)

    def test_daily_loss_limit_trips_breaker(self, tmp_path, monkeypatch):
        engine = make_engine(tmp_path)
        monkeypatch.setitem(CONFIG, "max_daily_loss_pct", 3.0)
        engine._update_peak_and_check_drawdown(10_000.0)
        assert engine.is_paused is False
        engine._update_peak_and_check_drawdown(9_600.0)
        assert engine.is_paused is True
        assert "daily loss" in (engine.pause_reason or "")


class TestProposalExpiry:
    def test_stale_pending_proposal_expires(self, tmp_path, monkeypatch):
        engine = make_engine(tmp_path)
        monkeypatch.setitem(CONFIG, "proposal_ttl_seconds", 900)
        stale = TradeProposal(asset="BTC", action="buy", allocation=100.0)
        stale.timestamp = datetime.now(UTC) - timedelta(seconds=1800)
        fresh = TradeProposal(asset="ETH", action="sell", allocation=100.0)
        engine.pending_proposals = [stale, fresh]

        engine._prune_stale_failed_proposals()

        remaining = [p.asset for p in engine.pending_proposals]
        assert remaining == ["ETH"]

    def test_fresh_pending_proposal_survives(self, tmp_path, monkeypatch):
        engine = make_engine(tmp_path)
        monkeypatch.setitem(CONFIG, "proposal_ttl_seconds", 900)
        fresh = TradeProposal(asset="BTC", action="buy", allocation=100.0)
        engine.pending_proposals = [fresh]
        engine._prune_stale_failed_proposals()
        assert engine.pending_proposals == [fresh]


class TestTradeEventFeed:
    def test_holds_excluded_and_outcomes_survive(self, tmp_path):
        engine = make_engine(tmp_path)
        engine._write_diary_entry({"action": "trade_closed", "asset": "BTC", "realized_pnl_usd": -42.0})
        for _ in range(30):
            engine._write_diary_entry({"action": "hold", "asset": "ETH"})
        engine._write_diary_entry({"action": "buy", "asset": "SOL", "order_result": "raw-sdk-blob"})

        events = engine._load_recent_trade_events(limit=12)

        actions = [e["action"] for e in events]
        assert "hold" not in actions
        assert "trade_closed" in actions
        assert all("order_result" not in e for e in events)


class TestBookClosedTrade:
    @pytest.mark.asyncio
    async def test_realized_pnl_net_of_fees_booked(self, tmp_path, monkeypatch):
        engine = make_engine(tmp_path)
        opened_at = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        opened_ms = int(datetime.fromisoformat(opened_at).timestamp() * 1000)

        fills = [
            {"coin": "BTC", "time": opened_ms + 7_200_000, "px": 105_000.0,
             "sz": 0.1, "closedPnl": 500.0, "fee": 4.5, "dir": "Close Long"},
            {"coin": "BTC", "time": opened_ms + 10, "px": 100_000.0,
             "sz": 0.1, "closedPnl": 0.0, "fee": 4.5, "dir": "Open Long"},
            {"coin": "ETH", "time": opened_ms + 500, "px": 5_000.0,
             "sz": 1.0, "closedPnl": 99.0, "fee": 1.0, "dir": "Close Long"},
        ]
        engine.hyperliquid = SimpleNamespace(
            get_recent_fills=async_return(fills),
        )
        persisted = {}

        def fake_record(**kwargs):
            persisted.update(kwargs)
            return 1

        monkeypatch.setattr(
            bot_engine_module, "get_db_manager",
            lambda: SimpleNamespace(record_closed_trade=fake_record),
        )

        await engine._book_closed_trade({
            "venue": "hyperliquid",
            "asset": "BTC",
            "is_long": True,
            "amount": 0.1,
            "entry_price": 100_000.0,
            "tp_price": 105_000.0,
            "sl_price": 97_000.0,
            "opened_at": opened_at,
            "rationale": "test thesis",
        })

        entries = read_diary(engine)
        closed = [e for e in entries if e["action"] == "trade_closed"]
        assert len(closed) == 1
        assert closed[0]["realized_pnl_usd"] == pytest.approx(500.0 - 9.0)
        assert closed[0]["fees_usd"] == pytest.approx(9.0)
        assert closed[0]["exit_reason"] == "tp_hit"
        assert persisted["asset"] == "BTC"
        assert persisted["realized_pnl"] == pytest.approx(491.0)

    @pytest.mark.asyncio
    async def test_thalex_premium_flow_booked_from_fills(self, tmp_path, monkeypatch):
        engine = make_engine(tmp_path)
        engine.hyperliquid = SimpleNamespace()
        opened_at = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        opened_s = datetime.fromisoformat(opened_at).timestamp()
        engine.thalex = SimpleNamespace(
            get_recent_fills=async_return([
                {"instrument_name": "BTC-27JUN26-70000-C", "direction": "sell",
                 "amount": 0.1, "price": 1200.0, "time": opened_s + 60, "fee": 2.0},
                {"instrument_name": "BTC-27JUN26-70000-C", "direction": "buy",
                 "amount": 0.1, "price": 400.0, "time": opened_s + 1800, "fee": 2.0},
                {"instrument_name": "BTC-UNRELATED", "direction": "sell",
                 "amount": 1.0, "price": 999.0, "time": opened_s + 60, "fee": 1.0},
            ]),
        )
        persisted = {}
        monkeypatch.setattr(
            bot_engine_module, "get_db_manager",
            lambda: SimpleNamespace(record_closed_trade=lambda **k: persisted.update(k) or 1),
        )

        await engine._book_closed_trade({
            "venue": "thalex",
            "asset": "BTC",
            "instrument_names": ["BTC-27JUN26-70000-C"],
            "strategy": "credit_call_spread",
            "opened_at": opened_at,
        })

        entries = read_diary(engine)
        assert entries and entries[0]["action"] == "trade_closed"
        assert entries[0]["venue"] == "thalex"
        assert entries[0]["premium_flow_usd"] == pytest.approx(0.1 * 1200 - 0.1 * 400)
        assert entries[0]["realized_pnl_usd"] == pytest.approx(80.0 - 4.0)
        assert persisted["venue"] == "thalex"
        assert persisted["realized_pnl"] == pytest.approx(76.0)

    @pytest.mark.asyncio
    async def test_thalex_trade_without_fills_books_note_only(self, tmp_path, monkeypatch):
        engine = make_engine(tmp_path)
        engine.hyperliquid = SimpleNamespace()
        engine.thalex = SimpleNamespace(get_recent_fills=async_return([]))
        monkeypatch.setattr(
            bot_engine_module, "get_db_manager",
            lambda: SimpleNamespace(record_closed_trade=lambda **k: pytest.fail("should not persist")),
        )

        await engine._book_closed_trade({
            "venue": "thalex",
            "asset": "BTC",
            "instrument_names": ["BTC-27JUN26-70000-C"],
            "strategy": "credit_call_spread",
        })

        entries = read_diary(engine)
        assert entries and entries[0]["action"] == "trade_closed"
        assert entries[0]["fills_matched"] == 0
        assert entries[0]["realized_pnl_usd"] is None


class TestStopCoverage:
    @pytest.mark.asyncio
    async def test_missing_stop_replaced_from_trade_record(self, tmp_path):
        engine = make_engine(tmp_path)
        placed = []

        async def place_stop_loss(asset, is_long, size, price):
            placed.append((asset, is_long, size, price))
            return {"status": "ok"}

        engine.hyperliquid = SimpleNamespace(place_stop_loss=place_stop_loss)
        engine.active_trades = [{
            "venue": "hyperliquid", "asset": "BTC", "sl_price": 95_000.0,
        }]

        await engine._enforce_stop_coverage(
            [{"symbol": "BTC", "quantity": 0.5, "current_price": 100_000.0}],
            [],
        )

        assert placed == [("BTC", True, 0.5, 95_000.0)]

    @pytest.mark.asyncio
    async def test_covered_position_untouched(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.hyperliquid = SimpleNamespace(
            place_stop_loss=lambda *a, **k: pytest.fail("should not place"),
        )
        engine.active_trades = [{
            "venue": "hyperliquid", "asset": "BTC", "sl_price": 95_000.0,
        }]

        await engine._enforce_stop_coverage(
            [{"symbol": "BTC", "quantity": 0.5, "current_price": 100_000.0}],
            [{"coin": "BTC", "order_type": "trigger", "is_buy": False,
              "trigger_price": 95_000.0, "size": 0.5}],
        )

    @pytest.mark.asyncio
    async def test_tp_only_trigger_does_not_count_as_stop(self, tmp_path):
        engine = make_engine(tmp_path)
        placed = []

        async def place_stop_loss(asset, is_long, size, price):
            placed.append(asset)
            return {"status": "ok"}

        engine.hyperliquid = SimpleNamespace(place_stop_loss=place_stop_loss)
        engine.active_trades = [{
            "venue": "hyperliquid", "asset": "BTC", "sl_price": 95_000.0,
        }]

        await engine._enforce_stop_coverage(
            [{"symbol": "BTC", "quantity": 0.5, "current_price": 100_000.0}],
            [{"coin": "BTC", "order_type": "trigger", "is_buy": False,
              "trigger_price": 110_000.0, "size": 0.5}],
        )

        assert placed == ["BTC"]

    @pytest.mark.asyncio
    async def test_external_position_ignored(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.hyperliquid = SimpleNamespace(
            place_stop_loss=lambda *a, **k: pytest.fail("should not place"),
        )
        engine.active_trades = []

        await engine._enforce_stop_coverage(
            [{"symbol": "BTC", "quantity": 0.5, "current_price": 100_000.0}],
            [],
        )
        assert read_diary(engine) == []


class TestTransferBaselines:
    @pytest.mark.asyncio
    async def test_deposit_shifts_all_baselines(self, tmp_path):
        engine = make_engine(tmp_path)
        engine._session_start_ms = 1_000
        engine._net_transfers_usd = 0.0
        engine.initial_account_value = 10_000.0
        engine.peak_account_value = 11_000.0
        engine._day_start_value = 10_500.0
        engine.hyperliquid = SimpleNamespace(
            get_net_transfers_since=async_return(5_000.0),
        )

        await engine._adjust_baselines_for_transfers()

        assert engine.initial_account_value == pytest.approx(15_000.0)
        assert engine.peak_account_value == pytest.approx(16_000.0)
        assert engine._day_start_value == pytest.approx(15_500.0)
        assert engine._net_transfers_usd == pytest.approx(5_000.0)
        entries = read_diary(engine)
        assert entries and entries[0]["action"] == "transfer_detected"

    @pytest.mark.asyncio
    async def test_no_transfer_is_noop(self, tmp_path):
        engine = make_engine(tmp_path)
        engine._session_start_ms = 1_000
        engine._net_transfers_usd = 0.0
        engine.initial_account_value = 10_000.0
        engine.hyperliquid = SimpleNamespace(
            get_net_transfers_since=async_return(0.0),
        )
        await engine._adjust_baselines_for_transfers()
        assert engine.initial_account_value == pytest.approx(10_000.0)
        assert read_diary(engine) == []


class TestMarketSnapshotPersistence:
    def test_snapshot_roundtrip(self, tmp_path):
        from src.database.db_manager import DatabaseManager
        from src.database.models import MarketData

        dbm = DatabaseManager(db_url=f"sqlite:///{tmp_path}/bot.db")
        section = {
            "asset": "BTC",
            "current_price": 100_000.0,
            "intraday": {"sma99": 99_500.0},
        }
        count = dbm.save_market_snapshots([{
            "asset": "BTC",
            "timestamp": datetime(2026, 7, 7, 12, 0, 0),
            "price": 100_000.0,
            "volume_24h": 1_000_000.0,
            "open_interest": 50_000.0,
            "funding_rate": 0.0001,
            "indicators": section,
        }])
        assert count == 1

        with dbm.session_scope() as session:
            row = session.query(MarketData).one()
            assert row.asset == "BTC"
            assert row.interval == "cycle"
            assert row.close == pytest.approx(100_000.0)
            assert json.loads(row.indicators)["intraday"]["sma99"] == pytest.approx(99_500.0)


def async_return(value):
    async def _inner(*args, **kwargs):
        return value
    return _inner
