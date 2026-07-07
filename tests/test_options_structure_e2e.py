from datetime import date

import pytest

from src.backend.config_loader import CONFIG
from src.backend.options_intel.portfolio import aggregate_portfolio_greeks


class FakeGreeksSource:
    def __init__(self, table):
        self.table = table

    async def get_greeks(self, instrument_name):
        return self.table.get(instrument_name, {})


@pytest.mark.asyncio
async def test_aggregate_emits_structures_for_credit_put_spread():
    positions = [
        {"instrument_name": "BTC-27JUN26-100000-P", "size": 0.1, "side": "short"},
        {"instrument_name": "BTC-27JUN26-90000-P", "size": 0.1, "side": "long"},
    ]
    greeks = {
        "BTC-27JUN26-100000-P": {"delta": -0.30, "gamma": 0.001, "vega": 50, "theta": -5, "mark_iv": 0.60},
        "BTC-27JUN26-90000-P": {"delta": -0.10, "gamma": 0.0005, "vega": 20, "theta": -2, "mark_iv": 0.65},
    }
    result = await aggregate_portfolio_greeks(
        positions=positions,
        greeks_source=FakeGreeksSource(greeks),
        today=date(2026, 6, 13),
        spot=100000.0,
    )
    assert "structures" in result
    assert len(result["structures"]) == 1
    s = result["structures"][0]
    assert s["kind"] == "credit_put_spread"
    assert s["underlying"] == "BTC"
    assert s["is_credit"] is True


@pytest.mark.asyncio
async def test_aggregate_empty_positions_emits_no_structures():
    result = await aggregate_portfolio_greeks(
        positions=[],
        greeks_source=FakeGreeksSource({}),
        today=date(2026, 6, 13),
    )
    assert result["structures"] == []


@pytest.mark.asyncio
async def test_aggregate_emits_unknown_for_single_short_leg():
    positions = [
        {"instrument_name": "BTC-27JUN26-100000-P", "size": 0.1, "side": "short"},
    ]
    greeks = {
        "BTC-27JUN26-100000-P": {"delta": -0.30, "gamma": 0.001, "vega": 50, "theta": -5, "mark_iv": 0.60},
    }
    result = await aggregate_portfolio_greeks(
        positions=positions,
        greeks_source=FakeGreeksSource(greeks),
        today=date(2026, 6, 13),
        spot=100000.0,
    )
    assert len(result["structures"]) == 1
    assert result["structures"][0]["kind"] == "unknown"


def test_options_context_to_dict_does_not_include_structures():
    from src.backend.options_intel.snapshot import OptionsContext

    ctx = OptionsContext(
        timestamp_utc="2026-05-20T00:00:00Z",
        spot=100000.0,
        spot_24h_change_pct=0.0,
        opening_range={},
        keltner={},
        atm_iv_by_tenor={},
        skew_25d_by_tenor={},
        term_structure_slope=0.0,
        expected_move_pct_by_tenor={},
        vol_regime="fair",
        vol_regime_confidence="high",
        realized_iv_ratio_30d=1.0,
        straddle_test_30d={},
    )
    object.__setattr__(ctx, "structures", [{"kind": "credit_put_spread"}])
    payload = ctx.to_dict()
    assert "structures" not in payload, (
        "structures must remain a non-prompted field in Phase 1 — adding it to "
        "to_dict would change the LLM prompt"
    )


def test_persist_structures_upserts_open_marks_closed():
    """persist_options_structures writes new structures, updates existing ones, and
    closes structures that disappeared since the previous cycle."""
    from src.backend.bot_engine import persist_options_structures
    from src.database.db_manager import DatabaseManager

    db = DatabaseManager(db_url="sqlite:///:memory:")

    structures_cycle_1 = [{
        "structure_id": "abc123",
        "underlying": "BTC",
        "kind": "credit_put_spread",
        "tenor_days_min": 14,
        "tenor_days_max": 14,
        "net_premium": 20.0,
        "is_credit": True,
        "max_loss": 980.0,
        "max_profit": 20.0,
        "breakevens": [99800.0],
        "short_leg_delta": -0.30,
        "breach_state": "warning",
        "pnl_abs": 0.0,
        "pnl_pct": 0.0,
        "aggregate_greeks": {"delta": -0.20, "gamma": 0.0005, "vega": 30, "theta": -3},
        "confidence": 1.0,
        "legs": ["BTC-27JUN26-100000-P", "BTC-27JUN26-90000-P"],
    }]

    persist_options_structures(db, structures_cycle_1)
    open_ids = {row["structure_id"] for row in db.get_open_structures()}
    assert open_ids == {"abc123"}

    structures_cycle_2 = [{**structures_cycle_1[0], "pnl_abs": 5.0, "pnl_pct": 0.25, "breach_state": "warning"}]
    persist_options_structures(db, structures_cycle_2)
    rows = db.get_open_structures()
    assert len(rows) == 1
    assert rows[0]["last_pnl_abs"] == 5.0

    persist_options_structures(db, [])
    open_ids = {row["structure_id"] for row in db.get_open_structures()}
    assert open_ids == set()


def test_bot_service_exposes_structures_when_flag_on(monkeypatch):
    from src.gui.services.bot_service import build_positions_view
    from src.backend.config_loader import CONFIG

    monkeypatch.setitem(CONFIG, "options_structure_layer", True)
    state_payload = {
        "thalex_positions": [
            {"instrument_name": "BTC-27JUN26-100000-P", "size": 0.1, "side": "short", "kind": "put", "strike": 100000, "days_to_expiry": 14, "delta": -0.30},
            {"instrument_name": "BTC-27JUN26-90000-P", "size": 0.1, "side": "long", "kind": "put", "strike": 90000, "days_to_expiry": 14, "delta": -0.10},
        ],
        "structures": [
            {"structure_id": "abc123", "kind": "credit_put_spread", "underlying": "BTC",
             "tenor_days_min": 14, "tenor_days_max": 14, "net_premium": 20.0, "is_credit": True,
             "max_loss": 980.0, "max_profit": 20.0, "breakevens": [99800.0],
             "short_leg_delta": -0.30, "breach_state": "warning",
             "pnl_abs": 0.0, "pnl_pct": 0.0, "aggregate_greeks": {"delta": -0.20},
             "confidence": 1.0, "legs": ["BTC-27JUN26-100000-P", "BTC-27JUN26-90000-P"]},
        ],
    }
    view = build_positions_view(state_payload)
    assert "thalex_structures" in view
    assert len(view["thalex_structures"]) == 1
    assert view["thalex_structures"][0]["kind"] == "credit_put_spread"


def test_bot_service_omits_structures_when_flag_off(monkeypatch):
    from src.gui.services.bot_service import build_positions_view
    from src.backend.config_loader import CONFIG

    monkeypatch.setitem(CONFIG, "options_structure_layer", False)
    state_payload = {
        "thalex_positions": [],
        "structures": [{"structure_id": "x", "kind": "credit_put_spread"}],
    }
    view = build_positions_view(state_payload)
    assert view.get("thalex_structures", []) == []


_SPREAD_POSITIONS = [
    {"instrument_name": "BTC-27JUN26-100000-P", "size": 0.1, "side": "short"},
    {"instrument_name": "BTC-27JUN26-90000-P", "size": 0.1, "side": "long"},
]
_SPREAD_GREEKS = {
    "BTC-27JUN26-100000-P": {"delta": -0.30, "gamma": 0.001, "vega": 50, "theta": -5, "mark_price": 2000.0},
    "BTC-27JUN26-90000-P": {"delta": -0.10, "gamma": 0.0005, "vega": 20, "theta": -2, "mark_price": 800.0},
}


def _spread_structure_id():
    from types import SimpleNamespace

    from src.backend.options_intel.structure import compute_structure_id

    return compute_structure_id([
        SimpleNamespace(instrument_name="BTC-27JUN26-100000-P", side="short"),
        SimpleNamespace(instrument_name="BTC-27JUN26-90000-P", side="long"),
    ])


@pytest.mark.asyncio
async def test_aggregate_joins_entry_premium_into_nonzero_pnl():
    """A credit spread whose marks moved since open must report live pnl.

    Entry premium 200, current net premium 0.1*2000 - 0.1*800 = 120 →
    pnl_abs = 80, pnl_pct = 0.4 for a credit structure."""
    result = await aggregate_portfolio_greeks(
        positions=_SPREAD_POSITIONS,
        greeks_source=FakeGreeksSource(_SPREAD_GREEKS),
        today=date(2026, 6, 13),
        spot=100000.0,
        entry_premium_by_structure_id={_spread_structure_id(): 200.0},
    )
    s = result["structures"][0]
    assert s["kind"] == "credit_put_spread"
    assert s["net_premium"] == pytest.approx(120.0)
    assert s["pnl_abs"] == pytest.approx(80.0)
    assert s["pnl_pct"] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_aggregate_missing_or_none_entry_premium_yields_zero_pnl():
    for premium_map in (None, {}, {_spread_structure_id(): None}):
        result = await aggregate_portfolio_greeks(
            positions=_SPREAD_POSITIONS,
            greeks_source=FakeGreeksSource(_SPREAD_GREEKS),
            today=date(2026, 6, 13),
            spot=100000.0,
            entry_premium_by_structure_id=premium_map,
        )
        s = result["structures"][0]
        assert s["pnl_abs"] == pytest.approx(0.0)
        assert s["pnl_pct"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_builder_joins_persisted_entry_premium_into_structure_views(monkeypatch, tmp_path):
    """Production path: build_options_context must join DB entry premiums so
    the classified structures AND the LLM-facing structure views carry live pnl."""
    from datetime import datetime, timedelta, timezone

    from src.backend.options_intel.builder import build_options_context
    from src.backend.options_intel.iv_history_store import IVHistoryStore

    class FakeThalex:
        _instruments_cache = []

        async def get_user_state(self):
            return {"balance": 10000.0, "positions": _SPREAD_POSITIONS}

        async def get_greeks(self, instrument_name):
            return _SPREAD_GREEKS.get(instrument_name, {})

    class FakeDeribit:
        async def get_index_price(self, index_name="btc_usd"):
            return 100000.0

        async def get_book_summary_by_currency(self, currency, kind):
            return []

    sid = _spread_structure_id()
    rows = [{
        "structure_id": sid,
        "underlying": "BTC",
        "kind": "credit_put_spread",
        "opened_at": datetime.now(timezone.utc) - timedelta(days=3),
        "last_seen_at": datetime.now(timezone.utc),
        "entry_net_premium": 200.0,
        "last_pnl_abs": 0.0,
        "last_pnl_pct": 0.0,
        "last_breach_state": "nominal",
    }]

    class FakeDBManager:
        def get_open_structures(self):
            return rows

    import src.database.db_manager as db_manager_module
    monkeypatch.setattr(db_manager_module, "get_db_manager", lambda db_url=None: FakeDBManager())
    monkeypatch.setitem(CONFIG, "options_structure_prompt", True)

    ctx = await build_options_context(
        thalex=FakeThalex(),
        deribit=FakeDeribit(),
        iv_history=IVHistoryStore(db_path=str(tmp_path / "iv.db")),
        spot_history=[100000.0] * 16,
        today=date(2026, 6, 13),
    )

    assert len(ctx.structures) == 1
    assert ctx.structures[0]["pnl_abs"] == pytest.approx(80.0)
    assert ctx.structures[0]["pnl_pct"] == pytest.approx(0.4)

    assert len(ctx.structure_views) == 1
    view = ctx.structure_views[0]
    assert view.pnl_abs == pytest.approx(80.0)
    assert view.pnl_pct == pytest.approx(0.4)
    assert view.days_open == 3

    payload = ctx.to_dict()
    assert payload["structures"][0]["pnl_pct"] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_builder_new_structure_without_db_row_defaults_to_zero_pnl(monkeypatch, tmp_path):
    from src.backend.options_intel.builder import build_options_context
    from src.backend.options_intel.iv_history_store import IVHistoryStore

    class FakeThalex:
        _instruments_cache = []

        async def get_user_state(self):
            return {"balance": 10000.0, "positions": _SPREAD_POSITIONS}

        async def get_greeks(self, instrument_name):
            return _SPREAD_GREEKS.get(instrument_name, {})

    class FakeDeribit:
        async def get_index_price(self, index_name="btc_usd"):
            return 100000.0

        async def get_book_summary_by_currency(self, currency, kind):
            return []

    class FakeDBManager:
        def get_open_structures(self):
            return []

    import src.database.db_manager as db_manager_module
    monkeypatch.setattr(db_manager_module, "get_db_manager", lambda db_url=None: FakeDBManager())

    ctx = await build_options_context(
        thalex=FakeThalex(),
        deribit=FakeDeribit(),
        iv_history=IVHistoryStore(db_path=str(tmp_path / "iv.db")),
        spot_history=[100000.0] * 16,
        today=date(2026, 6, 13),
    )

    assert len(ctx.structures) == 1
    assert ctx.structures[0]["pnl_abs"] == pytest.approx(0.0)
    assert ctx.structures[0]["pnl_pct"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_structure_view_emitted_through_full_pipeline(monkeypatch):
    """Full pipeline: classifier output → structure_views in OptionsContext.

    With both flags on, the LLM-facing to_dict() emits a 'structures' key
    populated from the StructureView projection."""
    from src.backend.options_intel.snapshot import OptionsContext, StructureView

    monkeypatch.setenv("OPTIONS_STRUCTURE_LAYER", "1")
    monkeypatch.setitem(CONFIG, "options_structure_prompt", True)

    ctx = OptionsContext(
        timestamp_utc="2026-05-20T00:00:00Z", spot=100000.0, spot_24h_change_pct=0.0,
        opening_range={}, keltner={},
        atm_iv_by_tenor={}, skew_25d_by_tenor={}, term_structure_slope=0.0,
        expected_move_pct_by_tenor={},
        vol_regime="fair", vol_regime_confidence="high",
        realized_iv_ratio_30d=1.0, straddle_test_30d={},
    )

    raw_structures = [{
        "structure_id": "abc123", "kind": "credit_put_spread", "underlying": "BTC",
        "tenor_days_min": 14, "tenor_days_max": 14, "net_premium": 20.0, "is_credit": True,
        "max_loss": 980.0, "max_profit": 20.0, "breakevens": [99800.0],
        "short_leg_delta": -0.30, "breach_state": "warning",
        "pnl_abs": 0.0, "pnl_pct": 0.0, "aggregate_greeks": {"delta": -0.20},
        "confidence": 1.0,
        "legs": ["BTC-27JUN26-100000-P", "BTC-27JUN26-90000-P"],
    }]
    open_positions = [
        {"instrument_name": "BTC-27JUN26-100000-P", "kind": "put", "strike": 100000.0,
         "side": "short", "size": 0.1, "days_to_expiry": 14, "delta": -0.30},
        {"instrument_name": "BTC-27JUN26-90000-P", "kind": "put", "strike": 90000.0,
         "side": "long", "size": 0.1, "days_to_expiry": 14, "delta": -0.10},
    ]
    object.__setattr__(ctx, "structures", raw_structures)
    object.__setattr__(ctx, "structure_views", [
        StructureView.from_classifier_dict(raw_structures[0], open_positions, days_open=3)
    ])

    payload = ctx.to_dict()
    assert "structures" in payload
    assert len(payload["structures"]) == 1
    assert payload["structures"][0]["kind"] == "credit_put_spread"
    assert payload["structures"][0]["days_open"] == 3
    assert payload["structures"][0]["legs"][0]["strike"] == 100000.0
