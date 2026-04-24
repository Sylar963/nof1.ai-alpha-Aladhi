"""_build_positions_view returns a per-venue split.

After B1, the perps LLM prompt must only carry Hyperliquid rows; the GUI
dashboard consumes the combined list. This test locks both halves of that
contract at the pure-function level (no full bot boot needed).
"""

from __future__ import annotations

from src.backend.bot_engine import TradingBotEngine
from src.backend.trading.exchange_adapter import PositionSnapshot


def _engine() -> TradingBotEngine:
    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.active_trades = []
    return engine


def test_positions_view_returns_three_keys():
    view = TradingBotEngine._build_positions_view(
        _engine(),
        [{"symbol": "ETH", "quantity": 1.0, "entry_price": 3000.0}],
        [
            PositionSnapshot(
                venue="thalex",
                asset="BTC",
                instrument_name="BTC-29MAY26-90000-C",
                side="long",
                size=3.0,
                entry_price=935.0,
                current_price=925.0,
                unrealized_pnl=-30.0,
            )
        ],
    )
    assert set(view.keys()) == {"hyperliquid", "thalex", "combined"}


def test_hyperliquid_key_excludes_thalex_positions():
    view = TradingBotEngine._build_positions_view(
        _engine(),
        [{"symbol": "BTC", "quantity": -0.1, "entry_price": 75000.0}],
        [
            PositionSnapshot(
                venue="thalex",
                asset="BTC",
                instrument_name="BTC-29MAY26-90000-C",
                side="long",
                size=3.0,
                entry_price=935.0,
                current_price=925.0,
                unrealized_pnl=-30.0,
            )
        ],
    )
    assert len(view["hyperliquid"]) == 1
    assert view["hyperliquid"][0]["venue"] == "hyperliquid"
    assert all(r["venue"] == "hyperliquid" for r in view["hyperliquid"])


def test_thalex_key_excludes_hyperliquid_positions():
    view = TradingBotEngine._build_positions_view(
        _engine(),
        [{"symbol": "BTC", "quantity": -0.1, "entry_price": 75000.0}],
        [
            PositionSnapshot(
                venue="thalex",
                asset="BTC",
                instrument_name="BTC-29MAY26-90000-C",
                side="long",
                size=3.0,
                entry_price=935.0,
                current_price=925.0,
                unrealized_pnl=-30.0,
            )
        ],
    )
    assert len(view["thalex"]) == 1
    assert all(r["venue"] == "thalex" for r in view["thalex"])


def test_combined_is_concatenation():
    view = TradingBotEngine._build_positions_view(
        _engine(),
        [{"symbol": "ETH", "quantity": 1.0, "entry_price": 3000.0}],
        [
            PositionSnapshot(
                venue="thalex",
                asset="BTC",
                instrument_name="BTC-27JUN26-70000-C",
                side="long",
                size=1.0,
                entry_price=1000.0,
                current_price=1100.0,
                unrealized_pnl=100.0,
            )
        ],
    )
    assert view["combined"] == view["hyperliquid"] + view["thalex"]


def test_empty_thalex_still_returns_three_keys():
    view = TradingBotEngine._build_positions_view(
        _engine(),
        [{"symbol": "BTC", "quantity": 0.05, "entry_price": 70000.0}],
        None,
    )
    assert view["thalex"] == []
    assert len(view["hyperliquid"]) == 1
    assert view["combined"] == view["hyperliquid"]


def test_delta_hedge_hl_leg_attributed_to_ai():
    """HL short opened as a delta hedge for a Thalex option must show as AI.

    The options executor records a single thalex active_trade with
    ``hyperliquid_orders=[...]`` for the hedge leg. The positions view must
    attribute the matching HL row to the AI, not to "External".
    """
    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.active_trades = [
        {
            "venue": "thalex",
            "asset": "BTC",
            "instrument_names": ["BTC-29MAY26-90000-C"],
            "hyperliquid_orders": ["hl-hedge-order-1"],
            "strategy": "long_call_delta_hedged",
        }
    ]
    view = TradingBotEngine._build_positions_view(
        engine,
        [{"symbol": "BTC", "quantity": -0.6527, "entry_price": 70000.0}],
        [
            PositionSnapshot(
                venue="thalex",
                asset="BTC",
                instrument_name="BTC-29MAY26-90000-C",
                side="long",
                size=3.0,
                entry_price=935.0,
                current_price=925.0,
                unrealized_pnl=-30.0,
            )
        ],
    )
    hl_row = view["hyperliquid"][0]
    thalex_row = view["thalex"][0]
    assert hl_row["opened_by"] == "AI"
    assert thalex_row["opened_by"] == "AI"


def test_hl_unrelated_position_still_external_when_only_thalex_hedge_trade():
    """A thalex trade with an HL hedge should only claim HL rows of the SAME asset."""
    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.active_trades = [
        {
            "venue": "thalex",
            "asset": "BTC",
            "instrument_names": ["BTC-29MAY26-90000-C"],
            "hyperliquid_orders": ["hl-hedge-order-1"],
        }
    ]
    view = TradingBotEngine._build_positions_view(
        engine,
        [{"symbol": "ETH", "quantity": 1.0, "entry_price": 3000.0}],
        None,
    )
    assert view["hyperliquid"][0]["opened_by"] == "External"


def test_thalex_trade_without_hl_orders_does_not_claim_hl_rows():
    """A naked Thalex trade (no hedge) must not attribute unrelated HL rows."""
    engine = TradingBotEngine.__new__(TradingBotEngine)
    engine.active_trades = [
        {
            "venue": "thalex",
            "asset": "BTC",
            "instrument_names": ["BTC-29MAY26-90000-C"],
            "hyperliquid_orders": [],
        }
    ]
    view = TradingBotEngine._build_positions_view(
        engine,
        [{"symbol": "BTC", "quantity": 0.1, "entry_price": 70000.0}],
        None,
    )
    assert view["hyperliquid"][0]["opened_by"] == "External"
