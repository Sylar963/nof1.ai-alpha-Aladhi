"""Tests for the options-intel builder orchestrator.

The builder is the entrypoint that pulls Thalex chain + Deribit, runs the
surface, regime, mispricing, and snapshot pipelines, and returns a single
``OptionsContext``. Tests use fake adapter / client objects so the full
pipeline runs without network IO."""

from datetime import datetime, timezone

import pytest

from src.backend.options_intel.builder import build_options_context
from src.backend.options_intel.iv_history_store import IVHistoryStore
from src.backend.options_intel.snapshot import OptionsContext


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
    )

    assert isinstance(ctx.top_mispricings_vs_deribit, list)
    # We seeded a 200-300 bps Thalex-vs-Deribit gap on every aligned instrument.
    assert len(ctx.top_mispricings_vs_deribit) >= 1
    assert all("edge_bps" in entry for entry in ctx.top_mispricings_vs_deribit)


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
    )

    assert ctx.atm_iv_by_tenor == {}
    assert ctx.top_mispricings_vs_deribit == []
    assert ctx.vol_regime == "unknown"
