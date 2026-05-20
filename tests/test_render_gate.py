"""Tests for the RenderGate skip-if-unchanged helper.

Pages poll state every 1-5s but the bot only emits new market data on a
5-minute cycle. The gate is what prevents the heavy renderers (Plotly,
big tables, DB queries) from re-running 60x per real update. These tests
lock the contract: first call always renders, identical payloads skip,
any change re-renders, and reset() forces the next call to render.
"""

from unittest.mock import MagicMock

from src.gui.pages.market import _market_render_signature
from src.gui.services.ui_utils import RenderGate


def test_render_gate_first_call_renders():
    gate = RenderGate()
    assert gate.changed(("anything",)) is True


def test_render_gate_skips_identical_payload():
    gate = RenderGate()
    gate.changed(("a", 1, None))
    assert gate.changed(("a", 1, None)) is False
    # Repeated skips stay skipped.
    assert gate.changed(("a", 1, None)) is False


def test_render_gate_renders_on_any_change():
    gate = RenderGate()
    gate.changed(("a", 1))
    assert gate.changed(("a", 2)) is True
    assert gate.changed(("a", 2)) is False
    assert gate.changed(("b", 2)) is True


def test_render_gate_reset_forces_next_render():
    gate = RenderGate()
    gate.changed(("x",))
    assert gate.changed(("x",)) is False
    gate.reset()
    assert gate.changed(("x",)) is True


def test_market_signature_empty_state_is_stable():
    """No market data yet — the signature should be deterministic so the
    gate skips repeated empty ticks."""
    state = MagicMock(
        is_running=False, is_paused=False, error=None,
        hedge_status={}, hedge_metrics=[],
    )
    sig1 = _market_render_signature(state, None, 'BTC', '5m')
    sig2 = _market_render_signature(state, None, 'BTC', '5m')
    assert sig1 == sig2
    assert sig1[0] == 'empty'


def test_market_signature_differentiates_on_price_change():
    state = MagicMock(
        is_running=True, is_paused=False, error=None,
        hedge_status={'enabled': True, 'available': True},
        hedge_metrics=[],
    )
    md = {
        'price': 76000.0,
        'volume_24h': 1000.0,
        'intraday': {
            'series': {
                'price_candles': {
                    'time': [1, 2, 3], 'close': [76000.0, 76010.0, 76020.0],
                }
            },
            'opening_range': {'high': 77000.0, 'low': 76500.0, 'or_start_ms': 100},
            'avwap': 75500.0,
        },
    }
    sig_before = _market_render_signature(state, md, 'BTC', '5m')

    md_changed = {**md, 'price': 76050.0}
    sig_after = _market_render_signature(state, md_changed, 'BTC', '5m')
    assert sig_before != sig_after


def test_market_signature_stable_across_unrelated_fields():
    """Fields the market page doesn't display shouldn't flip the gate."""
    state = MagicMock(
        is_running=True, is_paused=False, error=None,
        hedge_status={}, hedge_metrics=[],
        # Fields the page does NOT mirror:
        sharpe_ratio=1.5,
        last_reasoning={'random': 'value'},
    )
    md = {'price': 76000.0, 'intraday': {}}
    sig_a = _market_render_signature(state, md, 'BTC', '5m')

    state.sharpe_ratio = 2.5
    state.last_reasoning = {'totally': 'different'}
    sig_b = _market_render_signature(state, md, 'BTC', '5m')

    assert sig_a == sig_b


def test_market_signature_differentiates_on_dropdown_change():
    state = MagicMock(
        is_running=True, is_paused=False, error=None,
        hedge_status={}, hedge_metrics=[],
    )
    md = {'price': 76000.0, 'intraday': {}}
    sig_btc = _market_render_signature(state, md, 'BTC', '5m')
    sig_eth = _market_render_signature(state, md, 'ETH', '5m')
    sig_btc_15m = _market_render_signature(state, md, 'BTC', '15m')
    assert sig_btc != sig_eth
    assert sig_btc != sig_btc_15m


def test_market_signature_includes_new_candle_arrival():
    """A new 5m bar should flip the signature so the chart refreshes."""
    state = MagicMock(
        is_running=True, is_paused=False, error=None,
        hedge_status={}, hedge_metrics=[],
    )
    md_t1 = {
        'price': 76000.0,
        'intraday': {
            'series': {'price_candles': {'time': [1, 2, 3], 'close': [76000.0] * 3}},
            'opening_range': {},
        },
    }
    md_t2 = {
        'price': 76000.0,
        'intraday': {
            'series': {'price_candles': {'time': [1, 2, 3, 4], 'close': [76000.0] * 4}},
            'opening_range': {},
        },
    }
    sig_t1 = _market_render_signature(state, md_t1, 'BTC', '5m')
    sig_t2 = _market_render_signature(state, md_t2, 'BTC', '5m')
    assert sig_t1 != sig_t2


def test_market_signature_flips_on_chart_intraday_15m_bar():
    """A new 15m bar in chart_intraday should also flip the gate so the
    Open Range & Keltner chart re-renders even when the 5m intraday frame
    is unchanged."""
    state = MagicMock(
        is_running=True, is_paused=False, error=None,
        hedge_status={}, hedge_metrics=[],
    )
    base = {
        'price': 76000.0,
        'intraday': {
            'series': {'price_candles': {'time': [1, 2], 'close': [76000.0] * 2}},
            'opening_range': {},
        },
        'chart_intraday': {
            'series': {'price_candles': {'time': [10, 20, 30], 'close': [76000.0] * 3}},
            'opening_range': {'high': 77000, 'low': 76000, 'or_start_ms': 5},
        },
    }
    bumped = {
        **base,
        'chart_intraday': {
            'series': {'price_candles': {'time': [10, 20, 30, 40], 'close': [76000.0] * 4}},
            'opening_range': {'high': 77000, 'low': 76000, 'or_start_ms': 5},
        },
    }
    sig_a = _market_render_signature(state, base, 'BTC', '15m')
    sig_b = _market_render_signature(state, bumped, 'BTC', '15m')
    assert sig_a != sig_b
