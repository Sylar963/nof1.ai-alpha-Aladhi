"""Chain enrichment — Thalex ``public/instruments`` carries metadata only.

These tests lock the contract that :func:`enrich_chain_with_tickers`:
  1. Targets only the DTE + strike-band subset (no wasted RPCs on far OTM).
  2. Merges ``iv``, ``mark_price``, ``delta`` into each targeted record.
  3. Reports accurate ``targeted / enriched / failed`` stats.
  4. Survives ticker RPC failures without mutating the input list.
"""

from __future__ import annotations

import time

import pytest

from src.backend.options_intel.chain_enricher import enrich_chain_with_tickers


class _FakeThalex:
    def __init__(self, tickers: dict, raise_for: set[str] | None = None) -> None:
        self.tickers = tickers
        self.raise_for = raise_for or set()
        self.calls: list[str] = []

    async def get_ticker_snapshot(self, name: str) -> dict:
        self.calls.append(name)
        if name in self.raise_for:
            raise RuntimeError("forced")
        return self.tickers.get(name, {})


def _instrument(name: str, *, strike: float, dte_days: int, kind: str = "call") -> dict:
    return {
        "instrument_name": name,
        "type": "option",
        "option_type": kind,
        "strike_price": strike,
        "expiration_timestamp": int(time.time()) + dte_days * 86400,
    }


@pytest.mark.asyncio
async def test_enrichment_merges_iv_mark_and_delta():
    chain = [_instrument("BTC-1-80000-C", strike=80000.0, dte_days=30)]
    fake = _FakeThalex({"BTC-1-80000-C": {"iv": 0.65, "mark_price": 0.04, "delta": 0.5}})
    enriched, stats = await enrich_chain_with_tickers(chain, thalex=fake, spot=80000.0)

    assert stats == {"targeted": 1, "enriched": 1, "failed": 0}
    assert enriched[0]["iv"] == 0.65
    assert enriched[0]["mark_price"] == 0.04
    assert enriched[0]["delta"] == 0.5


@pytest.mark.asyncio
async def test_enrichment_skips_records_outside_strike_band():
    """Strikes far from spot shouldn't waste a ticker RPC."""
    chain = [
        _instrument("BTC-1-80000-C", strike=80000.0, dte_days=30),
        _instrument("BTC-1-150000-C", strike=150000.0, dte_days=30),
        _instrument("BTC-1-40000-C", strike=40000.0, dte_days=30),
    ]
    fake = _FakeThalex({"BTC-1-80000-C": {"iv": 0.6}})
    _, stats = await enrich_chain_with_tickers(
        chain, thalex=fake, spot=80000.0, strike_band_pct=0.3,
    )
    assert stats["targeted"] == 1
    assert fake.calls == ["BTC-1-80000-C"]


@pytest.mark.asyncio
async def test_enrichment_skips_records_beyond_max_dte():
    chain = [
        _instrument("BTC-1-80000-C", strike=80000.0, dte_days=30),
        _instrument("BTC-1-80000-C-LONG", strike=80000.0, dte_days=180),
    ]
    fake = _FakeThalex({"BTC-1-80000-C": {"iv": 0.6}})
    _, stats = await enrich_chain_with_tickers(
        chain, thalex=fake, spot=80000.0, max_dte_days=90,
    )
    assert stats["targeted"] == 1


@pytest.mark.asyncio
async def test_enrichment_does_not_mutate_input_list():
    chain = [_instrument("BTC-1-80000-C", strike=80000.0, dte_days=30)]
    snapshot = [dict(r) for r in chain]
    fake = _FakeThalex({"BTC-1-80000-C": {"iv": 0.7}})
    enriched, _ = await enrich_chain_with_tickers(chain, thalex=fake, spot=80000.0)

    assert "iv" not in chain[0], "input must stay untouched"
    assert chain == snapshot
    assert enriched[0]["iv"] == 0.7


@pytest.mark.asyncio
async def test_enrichment_counts_rpc_failures():
    chain = [
        _instrument("BTC-1-80000-C", strike=80000.0, dte_days=30),
        _instrument("BTC-2-80000-C", strike=80000.0, dte_days=30),
    ]
    fake = _FakeThalex(
        {"BTC-1-80000-C": {"iv": 0.6}},
        raise_for={"BTC-2-80000-C"},
    )
    _, stats = await enrich_chain_with_tickers(chain, thalex=fake, spot=80000.0)
    assert stats["enriched"] == 1
    assert stats["failed"] == 1


@pytest.mark.asyncio
async def test_enrichment_returns_empty_stats_on_empty_chain():
    enriched, stats = await enrich_chain_with_tickers([], thalex=None, spot=80000.0)
    assert enriched == []
    assert stats == {"targeted": 0, "enriched": 0, "failed": 0}
