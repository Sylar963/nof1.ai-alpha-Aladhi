"""Manual-mode regression tests for Thalex proposal routing."""

from types import SimpleNamespace

import pytest

from src.backend.bot_engine import BotState, TradingBotEngine
from src.backend.trading.exchange_adapter import PositionSnapshot
from src.backend.models.trade_proposal import TradeProposal
from src.backend.trading.options_strategies import ExecutionResult
from src.gui.services.bot_service import BotService


@pytest.mark.asyncio
async def test_execute_proposal_routes_thalex_proposals_through_thalex_executor():
    proposal = TradeProposal(
        venue="thalex",
        asset="BTC",
        action="buy",
        size=0.05,
        rationale="cheap vol",
        market_conditions={
            "venue": "thalex",
            "strategy": "long_call_delta_hedged",
            "contracts": 0.05,
            "decision_payload": {
                "venue": "thalex",
                "asset": "BTC",
                "action": "buy",
                "strategy": "long_call_delta_hedged",
                "underlying": "BTC",
                "kind": "call",
                "tenor_days": 30,
                "target_strike": 65000,
                "contracts": 0.05,
                "rationale": "cheap vol",
            },
        },
    )
    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    engine.state = SimpleNamespace(pending_proposals=[])
    engine.pending_proposals = [proposal]
    engine.on_trade_executed = None
    engine.on_error = None
    engine._notify_state_update = lambda: None
    diary_entries = []
    engine._write_diary_entry = diary_entries.append

    calls = []

    async def _fake_execute(decision_payload):
        calls.append(decision_payload)
        engine._last_thalex_execution = {"execution_price": 1234.5}
        return True, ""

    engine._execute_thalex_decision = _fake_execute

    await engine._execute_proposal(proposal)

    assert calls == [proposal.market_conditions["decision_payload"]]
    assert proposal.status == "executed"
    assert proposal.execution_price == 1234.5
    assert diary_entries[0]["venue"] == "thalex"


@pytest.mark.asyncio
async def test_execute_proposal_marks_thalex_proposal_failed_on_rejection():
    proposal = TradeProposal(
        venue="thalex",
        asset="BTC",
        action="sell",
        market_conditions={
            "venue": "thalex",
            "decision_payload": {
                "venue": "thalex",
                "asset": "BTC",
                "action": "sell",
                "strategy": "credit_put_spread",
                "underlying": "BTC",
                "tenor_days": 14,
                "legs": [
                    {"kind": "put", "side": "sell", "target_strike": 55000, "contracts": 0.05},
                    {"kind": "put", "side": "buy", "target_strike": 50000, "contracts": 0.05},
                ],
                "rationale": "short vol",
            },
        },
    )
    errors = []
    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    engine.state = SimpleNamespace(pending_proposals=[])
    engine.pending_proposals = [proposal]
    engine.on_trade_executed = None
    engine.on_error = errors.append
    engine._notify_state_update = lambda: None
    engine._write_diary_entry = lambda *a, **k: None

    async def _fake_execute(decision_payload):
        return False, "risk cap"

    engine._execute_thalex_decision = _fake_execute

    await engine._execute_proposal(proposal)

    assert proposal.status == "failed"
    assert proposal.execution_error == "risk cap"
    assert errors == ["Failed to execute trade: risk cap"]


def test_bot_service_approve_proposal_calls_engine_synchronously():
    service = BotService()
    service._add_event = lambda *a, **k: None
    calls = []
    service.bot_engine = SimpleNamespace(
        is_running=True,
        approve_proposal=lambda proposal_id: calls.append(proposal_id) or True,
    )

    assert service.approve_proposal("abc123") is True
    assert calls == ["abc123"]


def test_bot_service_emits_hedge_degraded_and_recovered_events():
    service = BotService()

    degraded_state = BotState(
        hedge_status={
            "health": "degraded",
            "degraded_underlyings": {"BTC": "missing live delta"},
            "tracked_underlyings": 1,
            "active_underlyings": 1,
        }
    )
    recovered_state = BotState(
        hedge_status={
            "health": "healthy",
            "degraded_underlyings": {},
            "tracked_underlyings": 1,
            "active_underlyings": 1,
        }
    )

    service._on_state_update(degraded_state)
    service._on_state_update(recovered_state)

    messages = [event["message"] for event in service.get_recent_events(limit=10)]
    assert any("Delta hedge health degraded" in message for message in messages)
    assert any("Hedge degraded for BTC" in message for message in messages)
    assert any("Hedge recovered for BTC" in message for message in messages)


@pytest.mark.asyncio
async def test_bot_service_start_resets_session_trackers(monkeypatch):
    from src.gui.services import bot_service as bot_service_module

    class FakeEngine:
        def __init__(self, assets, interval, delta_hedge_enabled, on_state_update, on_trade_executed, on_error):
            self.assets = assets
            self.interval = interval
            self.delta_hedge_enabled = delta_hedge_enabled
            self.on_state_update = on_state_update
            self.on_trade_executed = on_trade_executed
            self.on_error = on_error
            self.is_running = False

        async def start(self):
            self.is_running = True

    service = BotService()
    service.equity_history = [{"time": "old", "value": 1.0}]
    service.recent_events = [{"time": "old", "message": "stale", "level": "info"}]
    service._last_hedge_health = "degraded"
    service._last_degraded_underlyings = {"BTC": "missing live delta"}

    monkeypatch.setitem(bot_service_module.CONFIG, "taapi_api_key", "taapi")
    monkeypatch.setitem(bot_service_module.CONFIG, "openrouter_api_key", "openrouter")
    monkeypatch.setitem(bot_service_module.CONFIG, "hyperliquid_private_key", "secret")
    monkeypatch.setitem(bot_service_module.CONFIG, "mnemonic", None)
    monkeypatch.setitem(bot_service_module.CONFIG, "thalex_key_id", None)
    monkeypatch.setitem(bot_service_module.CONFIG, "thalex_private_key_path", None)
    monkeypatch.setattr(bot_service_module, "TradingBotEngine", FakeEngine)

    await service.start(assets=["BTC"], interval="5m")

    assert service.equity_history == []
    assert service.recent_events == []
    assert service._last_hedge_health is None
    assert service._last_degraded_underlyings == {}


def test_handle_execution_failure_emits_error_callback():
    errors = []
    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.logger = SimpleNamespace(error=lambda *a, **k: None)
    engine.state = SimpleNamespace(error=None)
    engine.on_error = errors.append

    TradingBotEngine._handle_execution_failure(engine, "Thalex", "BTC", "risk cap")

    assert engine.state.error == "Thalex execution failed for BTC: risk cap"
    assert errors == ["Thalex execution failed for BTC: risk cap"]


@pytest.mark.asyncio
async def test_bot_service_update_config_updates_runtime_state(monkeypatch):
    from src.gui.services import bot_service as bot_service_module

    service = BotService()
    monkeypatch.setattr(service, "_save_config_file", lambda: None)

    ok = await service.update_config({
        "assets": ["BTC", "ETH"],
        "interval": "15m",
        "llm_model": "demo/model",
    })

    assert ok is True
    assert service.config["assets"] == ["BTC", "ETH"]
    assert service.config["interval"] == "15m"
    assert service.config["model"] == "demo/model"
    assert bot_service_module.CONFIG["assets"] == "BTC ETH"
    assert bot_service_module.CONFIG["interval"] == "15m"
    assert bot_service_module.CONFIG["llm_model"] == "demo/model"


@pytest.mark.asyncio
async def test_bot_service_refresh_market_data_normalizes_state(monkeypatch):
    import src.backend.trading.hyperliquid_api as hyperliquid_api_module
    import src.gui.services.bot_service as bot_service_module

    captured_states = []

    class FakeHyperliquidAPI:
        async def get_user_state(self):
            return {
                "balance": 1000.0,
                "total_value": 1050.0,
                "positions": [
                    {
                        "coin": "BTC",
                        "szi": 0.1,
                        "entryPx": 50000.0,
                        "pnl": 50.0,
                        "liquidationPx": 42000.0,
                        "leverage": {"value": 3},
                    }
                ],
            }

        async def get_current_price(self, asset):
            return {"BTC": 50500.0}[asset]

        async def get_funding_rate(self, asset):
            return {"BTC": 0.001}[asset]

        async def get_open_interest(self, asset):
            return {"BTC": 1234567.0}[asset]

        async def get_prev_day_price(self, asset):
            return {"BTC": 50000.0}[asset]

        async def get_daily_notional_volume(self, asset):
            return {"BTC": 432100000.0}[asset]

    service = BotService()
    service.config["assets"] = ["BTC"]
    service.state_manager = SimpleNamespace(update=lambda state: captured_states.append(state))
    monkeypatch.setattr(hyperliquid_api_module, "HyperliquidAPI", FakeHyperliquidAPI)
    monkeypatch.setitem(bot_service_module.CONFIG, "thalex_key_id", "")
    monkeypatch.setitem(bot_service_module.CONFIG, "thalex_private_key_path", "")

    ok = await service.refresh_market_data()

    assert ok is True
    state = captured_states[-1]
    assert state.balance == 1000.0
    assert state.total_value == 1050.0
    assert state.balance_breakdown == {"hyperliquid": 1000.0}
    assert state.total_value_breakdown == {"hyperliquid": 1050.0}
    assert state.positions == [
        {
            "row_id": "hyperliquid:BTC",
            "symbol": "BTC",
            "asset": "BTC",
            "instrument_name": "BTC",
            "venue": "hyperliquid",
            "quantity": 0.1,
            "entry_price": 50000.0,
            "current_price": 50500.0,
            "unrealized_pnl": 50.0,
            "liquidation_price": 42000.0,
            "leverage": 3,
            "opened_by": "External",
            "closable": True,
        }
    ]
    assert state.market_data == [
        {
            "asset": "BTC",
            "current_price": 50500.0,
            "price": 50500.0,
            "funding_rate": 0.001,
            "open_interest": 1234567.0,
            "prev_day_price": 50000.0,
            "volume_24h": 432100000.0,
            "timestamp": state.market_data[0]["timestamp"],
            "intraday": {
                "sma99": None,
                "avwap": None,
                "keltner": {
                    "middle": None,
                    "upper": None,
                    "lower": None,
                    "position": "unknown",
                },
                "opening_range": {},
                "series": {
                    "sma99": [],
                    "keltner_middle": [],
                    "keltner_upper": [],
                    "keltner_lower": [],
                    "timestamps": [],
                    "price_candles": {},
                },
            },
            "long_term": {
                "sma99": None,
                "avwap": None,
                "keltner": {
                    "middle": None,
                    "upper": None,
                    "lower": None,
                    "position": "unknown",
                },
                "opening_range": {},
                "series": {
                    "sma99": [],
                    "keltner_middle": [],
                    "keltner_upper": [],
                    "keltner_lower": [],
                    "timestamps": [],
                    "price_candles": {},
                },
                "interval": "15m",
            },
        }
    ]


@pytest.mark.asyncio
async def test_bot_service_refresh_market_data_can_include_indicators(monkeypatch):
    import src.backend.indicators.taapi_client as taapi_client_module
    import src.backend.trading.hyperliquid_api as hyperliquid_api_module
    import src.gui.services.bot_service as bot_service_module

    captured_states = []

    class FakeHyperliquidAPI:
        async def get_user_state(self):
            return {
                "balance": 1000.0,
                "total_value": 1050.0,
                "positions": [],
            }

        async def get_current_price(self, asset):
            return 50500.0

        async def get_funding_rate(self, asset):
            return 0.001

        async def get_open_interest(self, asset):
            return 1234567.0

        async def get_prev_day_price(self, asset):
            return 50000.0

        async def get_daily_notional_volume(self, asset):
            return 432100000.0

        async def get_candles(self, asset, interval, start_ms, end_ms):
            """Return minimal candle data for indicator computation."""
            # Generate 200 candles so SMA99 + Keltner(130) have enough data
            base = 50000.0
            candles = []
            for i in range(200):
                candles.append({
                    "t": start_ms + i * 5 * 60 * 1000,
                    "o": base + i * 0.5,
                    "h": base + i * 0.5 + 50,
                    "l": base + i * 0.5 - 50,
                    "c": base + i * 0.5 + 10,
                    "v": 100.0,
                })
            return candles

    service = BotService()
    service.config["assets"] = ["BTC"]
    service.state_manager = SimpleNamespace(update=lambda state: captured_states.append(state))
    monkeypatch.setattr(hyperliquid_api_module, "HyperliquidAPI", FakeHyperliquidAPI)
    monkeypatch.setitem(bot_service_module.CONFIG, "interval", "4h")
    monkeypatch.setitem(bot_service_module.CONFIG, "thalex_key_id", "")
    monkeypatch.setitem(bot_service_module.CONFIG, "thalex_private_key_path", "")

    ok = await service.refresh_market_data(include_indicators=True)

    assert ok is True
    state = captured_states[-1]
    assert state.balance_breakdown == {"hyperliquid": 1000.0}
    assert state.total_value_breakdown == {"hyperliquid": 1050.0}
    # Indicators now computed locally from Hyperliquid candles
    assert state.market_data[0]["intraday"]["avwap"] is not None
    assert state.market_data[0]["prev_day_price"] == 50000.0
    assert state.market_data[0]["volume_24h"] == 432100000.0
    assert state.market_data[0]["long_term"]["interval"] == "4h"
    # Keltner should have computed values from the candle series
    assert state.market_data[0]["long_term"]["keltner"]["upper"] is not None


def test_bot_service_parses_comma_separated_assets(monkeypatch):
    import src.gui.services.bot_service as bot_service_module

    monkeypatch.setitem(bot_service_module.CONFIG, "assets", "BTC,ETH,SOL")

    service = BotService()

    assert service.get_assets() == ["BTC", "ETH", "SOL"]


@pytest.mark.asyncio
async def test_bot_service_refresh_market_data_includes_thalex_portfolio_positions(monkeypatch):
    import src.backend.trading.hyperliquid_api as hyperliquid_api_module
    import src.backend.trading.thalex_api as thalex_api_module
    import src.gui.services.bot_service as bot_service_module

    captured_states = []

    class FakeHyperliquidAPI:
        async def get_user_state(self):
            return {
                "balance": 1000.0,
                "total_value": 1050.0,
                "positions": [],
            }

        async def get_current_price(self, asset):
            return {"BTC": 50500.0}[asset]

        async def get_funding_rate(self, asset):
            return {"BTC": 0.001}[asset]

        async def get_open_interest(self, asset):
            return {"BTC": 1234567.0}[asset]

        async def get_prev_day_price(self, asset):
            return {"BTC": 50000.0}[asset]

        async def get_daily_notional_volume(self, asset):
            return {"BTC": 432100000.0}[asset]

    class FakeThalexAPI:
        async def get_user_state(self):
            return SimpleNamespace(
                balance=250000.0,
                total_value=250500.0,
                positions=[
                    PositionSnapshot(
                        venue="thalex",
                        asset="BTC",
                        instrument_name="BTC-27JUN26-70000-C",
                        side="short",
                        size=2.0,
                        entry_price=1200.0,
                        current_price=900.0,
                        unrealized_pnl=600.0,
                    )
                ]
            )

        async def disconnect(self):
            return None

    service = BotService()
    service.config["assets"] = ["BTC"]
    service.state_manager = SimpleNamespace(update=lambda state: captured_states.append(state))
    monkeypatch.setattr(hyperliquid_api_module, "HyperliquidAPI", FakeHyperliquidAPI)
    monkeypatch.setattr(thalex_api_module, "ThalexAPI", FakeThalexAPI)
    monkeypatch.setitem(bot_service_module.CONFIG, "thalex_key_id", "demo-key")
    monkeypatch.setitem(bot_service_module.CONFIG, "thalex_private_key_path", "/tmp/demo.pem")

    ok = await service.refresh_market_data()

    assert ok is True
    state = captured_states[-1]
    assert state.balance == 251000.0
    assert state.total_value == 251550.0
    assert state.balance_breakdown == {"hyperliquid": 1000.0, "thalex": 250000.0}
    assert state.total_value_breakdown == {"hyperliquid": 1050.0, "thalex": 250500.0}
    assert state.positions == [
        {
            "row_id": "thalex:BTC-27JUN26-70000-C",
            "symbol": "BTC-27JUN26-70000-C",
            "asset": "BTC",
            "instrument_name": "BTC-27JUN26-70000-C",
            "venue": "thalex",
            "quantity": -2.0,
            "entry_price": 1200.0,
            "current_price": 900.0,
            "unrealized_pnl": 600.0,
            "liquidation_price": 0.0,
            "leverage": 1,
            "opened_by": "External",
            "closable": False,
        }
    ]


def test_build_positions_view_marks_ai_and_external_option_positions():
    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.active_trades = [
        {
            "venue": "thalex",
            "asset": "BTC",
            "instrument_names": ["BTC-27JUN26-70000-C"],
        },
        {
            "venue": "hyperliquid",
            "asset": "ETH",
        },
    ]

    view = TradingBotEngine._build_positions_view(
        engine,
        [
            {
                "symbol": "ETH",
                "quantity": 1.5,
                "entry_price": 3000.0,
                "current_price": 3100.0,
                "liquidation_price": 2500.0,
                "unrealized_pnl": 150.0,
                "leverage": 2,
            }
        ],
        [
            PositionSnapshot(
                venue="thalex",
                asset="BTC",
                instrument_name="BTC-27JUN26-70000-C",
                side="long",
                size=1.0,
                entry_price=1000.0,
                current_price=1200.0,
                unrealized_pnl=200.0,
            ),
            PositionSnapshot(
                venue="thalex",
                asset="BTC",
                instrument_name="BTC-27JUN26-65000-P",
                side="short",
                size=1.0,
                entry_price=800.0,
                current_price=700.0,
                unrealized_pnl=100.0,
            ),
        ],
    )
    rows = view["combined"]
    assert set(view.keys()) == {"hyperliquid", "thalex", "combined"}
    assert len(view["hyperliquid"]) == 1
    assert len(view["thalex"]) == 2
    assert all(r["venue"] == "hyperliquid" for r in view["hyperliquid"])
    assert all(r["venue"] == "thalex" for r in view["thalex"])

    assert rows == [
        {
            "row_id": "hyperliquid:ETH",
            "symbol": "ETH",
            "asset": "ETH",
            "instrument_name": "ETH",
            "venue": "hyperliquid",
            "quantity": 1.5,
            "entry_price": 3000.0,
            "current_price": 3100.0,
            "liquidation_price": 2500.0,
            "unrealized_pnl": 150.0,
            "leverage": 2,
            "opened_by": "AI",
            "closable": True,
        },
        {
            "row_id": "thalex:BTC-27JUN26-70000-C",
            "symbol": "BTC-27JUN26-70000-C",
            "asset": "BTC",
            "instrument_name": "BTC-27JUN26-70000-C",
            "venue": "thalex",
            "quantity": 1.0,
            "entry_price": 1000.0,
            "current_price": 1200.0,
            "liquidation_price": 0.0,
            "unrealized_pnl": 200.0,
            "leverage": 1,
            "opened_by": "AI",
            "closable": False,
        },
        {
            "row_id": "thalex:BTC-27JUN26-65000-P",
            "symbol": "BTC-27JUN26-65000-P",
            "asset": "BTC",
            "instrument_name": "BTC-27JUN26-65000-P",
            "venue": "thalex",
            "quantity": -1.0,
            "entry_price": 800.0,
            "current_price": 700.0,
            "liquidation_price": 0.0,
            "unrealized_pnl": 100.0,
            "leverage": 1,
            "opened_by": "External",
            "closable": False,
        },
    ]


def test_track_hedge_events_suppresses_initial_passive_boot_state():
    service = BotService()
    state = BotState(is_running=False, hedge_status={"health": "disabled"})

    service._track_hedge_events(state)

    assert service.recent_events == []


@pytest.mark.asyncio
async def test_execute_thalex_decision_counts_live_portfolio_positions_for_risk_caps():
    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
    engine.active_trades = []
    engine.hedge_manager = None
    engine._last_thalex_execution = {}
    engine.thalex = SimpleNamespace(
        connect=lambda: _async_noop(),
        get_user_state=lambda: _async_result(SimpleNamespace(positions=[object(), object()])),
        get_recent_fills=lambda limit=20: _async_result([]),
        get_current_price=lambda asset: _async_result(1111.0),
        place_take_profit=lambda *a, **k: _async_result(None),
    )

    seen = {}

    class _FakeExecutor:
        async def execute(self, decision, open_positions_count):
            seen["open_positions_count"] = open_positions_count
            return ExecutionResult(ok=False, reason="risk cap")

    engine.options_executor = _FakeExecutor()

    ok, reason = await TradingBotEngine._execute_thalex_decision(engine, {
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.05,
        "rationale": "x",
    })

    assert ok is False
    assert reason == "risk cap"
    assert seen["open_positions_count"] == 2


@pytest.mark.asyncio
async def test_execute_thalex_decision_places_native_stop_loss_on_single_leg():
    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
    engine.active_trades = []
    engine.hedge_manager = None
    engine._last_thalex_execution = {}
    calls = []

    class _FakeExecutor:
        async def execute(self, decision, open_positions_count):
            return ExecutionResult(
                ok=True,
                thalex_orders=[SimpleNamespace(order_id="o1", instrument_name="BTC-10MAY26-65000-C", amount=0.05)],
                hyperliquid_orders=[],
            )

    engine.options_executor = _FakeExecutor()
    engine.thalex = SimpleNamespace(
        connect=lambda: _async_noop(),
        get_user_state=lambda: _async_result(SimpleNamespace(positions=[])),
        get_recent_fills=lambda limit=20: _async_result([]),
        get_current_price=lambda asset: _async_result(1111.0),
        place_take_profit=lambda *a, **k: _async_result(None),
        place_stop_loss=lambda *a, **k: calls.append((a, k)) or _async_result(SimpleNamespace(order_id="sl1")),
        place_bracket_order=lambda *a, **k: _async_result(None),
    )

    ok, reason = await TradingBotEngine._execute_thalex_decision(engine, {
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.05,
        "sl_price": 800.0,
        "rationale": "x",
    })

    assert ok is True
    assert reason == ""
    assert calls and calls[0][0] == ("BTC-10MAY26-65000-C", True, 0.05, 800.0)


async def _async_noop():
    return None


async def _async_result(value):
    return value
