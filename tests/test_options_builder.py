"""Tests for the options-intel builder orchestrator.

The builder is the entrypoint that pulls Thalex chain + Deribit, runs the
surface, regime, mispricing, and snapshot pipelines, and returns a single
``OptionsContext``. Tests use fake adapter / client objects so the full
pipeline runs without network IO."""

from datetime import date, datetime, timezone

import pytest

from src.backend.options_intel.builder import build_options_context
from src.backend.options_intel.iv_history_store import IVHistoryStore
from src.backend.options_intel.snapshot import OptionsContext


_TEST_TODAY = date(2026, 4, 10)


class FakeThalexAdapter:
    """Stand-in for ThalexAPI exposing only the bits the builder needs."""

    def __init__(self, instruments_cache, tickers):
        self._instruments_cache = instruments_cache
        self._tickers = tickers
        self.connected = True
        self.user_state_response = {"balance": 10000.0, "positions": []}

    async def connect(self):
        return None

    async def get_user_state(self):
        return self.user_state_response

    async def get_greeks(self, instrument_name):
        ticker = self._tickers.get(instrument_name) or {}
        return {
            "delta": ticker.get("delta", 0.5),
            "gamma": ticker.get("gamma", 0.0001),
            "vega": ticker.get("vega", 0.0),
            "theta": ticker.get("theta", 0.0),
            "mark_iv": ticker.get("mark_iv", 0.65),
        }


class FakeDeribitClient:
    def __init__(self, summaries, index_price=60000.0):
        self._summaries = summaries
        self._index_price = index_price

    async def get_book_summary_by_currency(self, currency, kind):
        return self._summaries

    async def get_index_price(self, index_name="btc_usd"):
        return self._index_price

    async def close(self):
        return None


def _utc_ts(year, month, day):
    return int(datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc).timestamp())


@pytest.fixture
def thalex_chain():
    expiry_15 = _utc_ts(2026, 4, 25)
    expiry_30 = _utc_ts(2026, 5, 10)
    return [
        # 15-day tenor
        {"instrument_name": "BTC-25APR26-60000-C", "type": "option", "underlying": "BTCUSD",
         "option_type": "call", "strike": 60000, "mark_iv": 0.65, "mark_price": 2500,
         "expiry_timestamp": expiry_15, "delta": 0.5},
        {"instrument_name": "BTC-25APR26-60000-P", "type": "option", "underlying": "BTCUSD",
         "option_type": "put", "strike": 60000, "mark_iv": 0.66, "mark_price": 2400,
         "expiry_timestamp": expiry_15, "delta": -0.5},
        # 30-day tenor
        {"instrument_name": "BTC-10MAY26-60000-C", "type": "option", "underlying": "BTCUSD",
         "option_type": "call", "strike": 60000, "mark_iv": 0.70, "mark_price": 3500,
         "expiry_timestamp": expiry_30, "delta": 0.52},
        {"instrument_name": "BTC-10MAY26-60000-P", "type": "option", "underlying": "BTCUSD",
         "option_type": "put", "strike": 60000, "mark_iv": 0.71, "mark_price": 3400,
         "expiry_timestamp": expiry_30, "delta": -0.48},
    ]


@pytest.fixture
def deribit_chain():
    expiry_15 = _utc_ts(2026, 4, 25)
    expiry_30 = _utc_ts(2026, 5, 10)
    return [
        {"instrument_name": "BTC-25APR26-60000-C", "kind": "option", "option_type": "call",
         "strike": 60000, "mark_iv": 0.62, "expiration_timestamp": expiry_15 * 1000},
        {"instrument_name": "BTC-25APR26-60000-P", "kind": "option", "option_type": "put",
         "strike": 60000, "mark_iv": 0.63, "expiration_timestamp": expiry_15 * 1000},
        {"instrument_name": "BTC-10MAY26-60000-C", "kind": "option", "option_type": "call",
         "strike": 60000, "mark_iv": 0.68, "expiration_timestamp": expiry_30 * 1000},
        {"instrument_name": "BTC-10MAY26-60000-P", "kind": "option", "option_type": "put",
         "strike": 60000, "mark_iv": 0.69, "expiration_timestamp": expiry_30 * 1000},
    ]


# ---------------------------------------------------------------------------
# build_options_context end-to-end with fakes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_options_context_returns_snapshot(thalex_chain, deribit_chain, tmp_path):
    thalex = FakeThalexAdapter(instruments_cache=thalex_chain, tickers={})
    deribit = FakeDeribitClient(summaries=deribit_chain)
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))

    ctx = await build_options_context(
        thalex=thalex,
        deribit=deribit,
        iv_history=store,
        spot_history=[60000.0] * 16,
        today=_TEST_TODAY,
    )

    assert isinstance(ctx, OptionsContext)
    assert ctx.spot == pytest.approx(60000.0)
    assert 15 in ctx.atm_iv_by_tenor
    assert ctx.atm_iv_by_tenor[15] == pytest.approx(0.655, abs=1e-3)


@pytest.mark.asyncio
async def test_build_persists_straddle_anchor_to_iv_history(thalex_chain, deribit_chain, tmp_path):
    thalex = FakeThalexAdapter(instruments_cache=thalex_chain, tickers={})
    deribit = FakeDeribitClient(summaries=deribit_chain)
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))

    await build_options_context(
        thalex=thalex,
        deribit=deribit,
        iv_history=store,
        spot_history=[60000.0] * 16,
        today=_TEST_TODAY,
    )

    rows = store.read_recent(tenor_days=15, limit=5)
    assert len(rows) == 1
    assert rows[0].tenor_days == 15
    assert rows[0].lower_strike < 60000 < rows[0].upper_strike


@pytest.mark.asyncio
async def test_build_includes_top_mispricings(thalex_chain, deribit_chain, tmp_path):
    thalex = FakeThalexAdapter(instruments_cache=thalex_chain, tickers={})
    deribit = FakeDeribitClient(summaries=deribit_chain)
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))

    ctx = await build_options_context(
        thalex=thalex,
        deribit=deribit,
        iv_history=store,
        spot_history=[60000.0] * 16,
        today=_TEST_TODAY,
    )

    assert isinstance(ctx.top_mispricings_vs_deribit, list)
    # We seeded a 200-300 bps Thalex-vs-Deribit gap on every aligned instrument.
    assert len(ctx.top_mispricings_vs_deribit) >= 1
    assert all("edge_bps" in entry for entry in ctx.top_mispricings_vs_deribit)


@pytest.mark.asyncio
async def test_build_passes_interpolation_flag_to_mispricing_scan(thalex_chain, tmp_path):
    """When use_interpolation=True, the builder must surface interpolated mispricings
    even when the Thalex tenor isn't an exact Deribit match."""
    from datetime import datetime as _dt, timezone as _tz

    def _ts(year, month, day):
        return int(_dt(year, month, day, 8, 0, 0, tzinfo=_tz.utc).timestamp())

    # Deribit has bracketing expiries (Apr 18 and May 30) but NOT Apr 25
    # which is what the Thalex chain uses for its 15d tenor.
    deribit_chain = [
        {"instrument_name": "BTC-18APR26-60000-C", "kind": "option", "option_type": "call",
         "strike": 60000, "mark_iv": 0.50, "expiration_timestamp": _ts(2026, 4, 18) * 1000},
        {"instrument_name": "BTC-18APR26-60000-P", "kind": "option", "option_type": "put",
         "strike": 60000, "mark_iv": 0.51, "expiration_timestamp": _ts(2026, 4, 18) * 1000},
        {"instrument_name": "BTC-30MAY26-60000-C", "kind": "option", "option_type": "call",
         "strike": 60000, "mark_iv": 0.70, "expiration_timestamp": _ts(2026, 5, 30) * 1000},
        {"instrument_name": "BTC-30MAY26-60000-P", "kind": "option", "option_type": "put",
         "strike": 60000, "mark_iv": 0.71, "expiration_timestamp": _ts(2026, 5, 30) * 1000},
    ]
    thalex = FakeThalexAdapter(instruments_cache=thalex_chain, tickers={})
    deribit = FakeDeribitClient(summaries=deribit_chain)
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))

    ctx = await build_options_context(
        thalex=thalex,
        deribit=deribit,
        iv_history=store,
        spot_history=[60000.0] * 16,
        today=_TEST_TODAY,
        use_interpolation=True,
        min_edge_bps=0.0,
    )

    # Without interpolation, the Apr 25 Thalex tenor would have no Deribit
    # match. With interpolation, the mispricing scanner finds at least one
    # synthetic match by interpolating between Apr 18 and May 30.
    assert len(ctx.top_mispricings_vs_deribit) >= 1


@pytest.mark.asyncio
async def test_build_populates_opening_range_when_intraday_supplied(thalex_chain, deribit_chain, tmp_path):
    """A non-empty intraday minute series must produce a real opening range."""
    from datetime import datetime as _dt, timezone as _tz

    today_dt = _dt(2026, 4, 10, tzinfo=_tz.utc)
    today_start = int(today_dt.timestamp())
    intraday = [
        (today_start + 5 * 60, 60050.0),
        (today_start + 30 * 60, 60500.0),  # high
        (today_start + 45 * 60, 59800.0),  # low
        (today_start + 59 * 60, 60100.0),
    ]

    thalex = FakeThalexAdapter(instruments_cache=thalex_chain, tickers={})
    deribit = FakeDeribitClient(summaries=deribit_chain)
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))

    ctx = await build_options_context(
        thalex=thalex,
        deribit=deribit,
        iv_history=store,
        spot_history=[60000.0] * 16,
        today=_TEST_TODAY,
        intraday_minute_prices=intraday,
    )

    assert ctx.opening_range["high"] == 60500.0
    assert ctx.opening_range["low"] == 59800.0
    assert ctx.opening_range["position"] == "inside"


@pytest.mark.asyncio
async def test_build_populates_keltner_when_long_daily_series_supplied(thalex_chain, deribit_chain, tmp_path):
    """When daily_closes_for_keltner has 25+ entries, the channel must populate."""
    thalex = FakeThalexAdapter(instruments_cache=thalex_chain, tickers={})
    deribit = FakeDeribitClient(summaries=deribit_chain)
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))

    daily = [60000.0 + i * 50 for i in range(30)]  # gentle uptrend, 30 closes
    ctx = await build_options_context(
        thalex=thalex,
        deribit=deribit,
        iv_history=store,
        spot_history=[60000.0] * 16,
        today=_TEST_TODAY,
        daily_closes_for_keltner=daily,
    )

    assert ctx.keltner["ema20"] is not None
    assert ctx.keltner["upper"] is not None
    assert ctx.keltner["lower"] is not None
    assert ctx.keltner["position"] in {"above", "inside", "below"}


@pytest.mark.asyncio
async def test_build_aggregates_portfolio_greeks_from_thalex_positions(thalex_chain, deribit_chain, tmp_path):
    """A Thalex user_state with open positions must produce real portfolio greeks."""
    thalex = FakeThalexAdapter(instruments_cache=thalex_chain, tickers={})
    thalex.user_state_response = {
        "balance": 12000.0,
        "positions": [
            {"instrument_name": "BTC-25APR26-65000-C", "size": 0.05, "side": "long"},
        ],
    }
    deribit = FakeDeribitClient(summaries=deribit_chain)
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))

    ctx = await build_options_context(
        thalex=thalex,
        deribit=deribit,
        iv_history=store,
        spot_history=[60000.0] * 16,
        today=_TEST_TODAY,
    )

    assert len(ctx.open_positions) == 1
    pos = ctx.open_positions[0]
    assert pos["instrument_name"] == "BTC-25APR26-65000-C"
    # FakeThalexAdapter.get_greeks returns delta=0.5 by default → portfolio delta = 0.025
    assert ctx.portfolio_greeks["delta"] == pytest.approx(0.025)
    assert ctx.open_position_count == 1


@pytest.mark.asyncio
async def test_build_tolerates_empty_thalex_chain(deribit_chain, tmp_path):
    """No instruments → builder must return a sensible empty-ish context, not raise."""
    thalex = FakeThalexAdapter(instruments_cache=[], tickers={})
    deribit = FakeDeribitClient(summaries=deribit_chain)
    store = IVHistoryStore(db_path=str(tmp_path / "iv.db"))

    ctx = await build_options_context(
        thalex=thalex,
        deribit=deribit,
        iv_history=store,
        spot_history=[60000.0] * 16,
        today=_TEST_TODAY,
    )

    assert ctx.atm_iv_by_tenor == {}
    assert ctx.top_mispricings_vs_deribit == []
    assert ctx.vol_regime == "unknown"
