"""Options-intel orchestrator.

Single async entrypoint that pulls Thalex chain + Deribit, runs the surface
builder, regime classifier, and mispricing scanner, and returns a single
:class:`OptionsContext` snapshot ready for the LLM.

This module never owns long-lived state. Caller injects the Thalex adapter
(read-only — uses ``_instruments_cache``, ``get_user_state``, ``get_greeks``),
the Deribit client, and the IV history store. That keeps the builder easy to
test with fakes and lets the bot engine wire the real instances at runtime.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

from src.backend.options_intel.iv_history_store import IVHistoryRow, IVHistoryStore
from src.backend.options_intel.mispricing import scan_mispricings
from src.backend.options_intel.regime import classify_regime
from src.backend.options_intel.snapshot import OptionsContext
from src.backend.options_intel.vol_surface import build_vol_surface


logger = logging.getLogger(__name__)


async def build_options_context(
    thalex,
    deribit,
    iv_history: IVHistoryStore,
    spot_history: list[float],
    *,
    today: Optional[date] = None,
    persist_anchor: bool = True,
    top_mispricings: int = 5,
    min_edge_bps: float = 100.0,
    use_interpolation: bool = False,
) -> OptionsContext:
    """Run the full options-intel pipeline and return a snapshot.

    Args:
        thalex: ThalexAPI instance (or stand-in). Must expose
            ``_instruments_cache`` and ``get_user_state``. ``get_greeks`` is
            currently unused by the builder but kept in the interface for
            future expansion.
        deribit: DeribitPublicClient. Must expose
            ``get_book_summary_by_currency`` and ``get_index_price``.
        iv_history: persistent SQLite store for straddle anchors.
        spot_history: ~16 daily closes (most recent last) for realized vol.
        persist_anchor: when True, write today's 15-day straddle anchor to
            ``iv_history`` so the regime classifier has a fresh data point.
        top_mispricings: how many of the largest IV gaps to surface.
        min_edge_bps: drop mispricings below this absolute IV diff.

    Returns:
        :class:`OptionsContext` digest ready to feed the LLM.
    """
    spot = await deribit.get_index_price("btc_usd")
    if not spot or spot <= 0:
        spot = (spot_history[-1] if spot_history else 0.0) or 0.0

    thalex_chain: list[dict] = list(getattr(thalex, "_instruments_cache", []) or [])
    deribit_chain: list[dict] = []
    try:
        deribit_chain = await deribit.get_book_summary_by_currency("BTC", "option")
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Deribit book summary fetch failed: %s", exc)

    surface = build_vol_surface(thalex_chain, spot=spot, today=today)

    # Persist a 15-day anchor on each refresh so the regime classifier has
    # something to look back at after a few days of running.
    if persist_anchor and surface.atm_straddle_15d:
        try:
            anchor = surface.atm_straddle_15d
            iv_history.write(
                IVHistoryRow(
                    ts=datetime.now(timezone.utc),
                    tenor_days=int(anchor["tenor_days"]),
                    atm_iv=float(anchor["atm_iv"]),
                    atm_straddle_em=float(anchor["straddle_premium"]) / spot if spot else 0.0,
                    spot_at_init=float(anchor["spot_at_init"]),
                    lower_strike=float(anchor["lower_strike"]),
                    upper_strike=float(anchor["upper_strike"]),
                )
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("iv_history write failed: %s", exc)

    regime_reading = classify_regime(
        store=iv_history,
        current_spot=spot,
        current_atm_iv_15d=surface.atm_iv_by_tenor.get(15) or 0.0,
        spot_history=spot_history,
    )

    mispricing_report = scan_mispricings(
        thalex_chain=thalex_chain,
        deribit_chain=deribit_chain,
        top_n=top_mispricings,
        min_edge_bps=min_edge_bps,
        use_interpolation=use_interpolation,
    )

    user_state = {}
    try:
        user_state = await thalex.get_user_state() or {}
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("thalex get_user_state failed: %s", exc)
    capital_available = float(_user_state_field(user_state, "balance", default=0.0))

    return OptionsContext(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        spot=float(spot),
        spot_24h_change_pct=_spot_change_pct(spot_history),
        opening_range={"high": None, "low": None, "position": "unknown"},
        keltner={"ema20": None, "upper": None, "lower": None, "position": "unknown"},
        atm_iv_by_tenor=surface.atm_iv_by_tenor,
        skew_25d_by_tenor=surface.skew_25d_by_tenor,
        term_structure_slope=surface.term_structure_slope,
        expected_move_pct_by_tenor=surface.expected_move_pct_by_tenor,
        vol_regime=regime_reading.vol_regime,
        vol_regime_confidence=regime_reading.confidence,
        realized_iv_ratio_15d=regime_reading.rv_iv_ratio,
        straddle_test_15d={
            "lower": regime_reading.historical_anchor.lower_strike if regime_reading.historical_anchor else None,
            "upper": regime_reading.historical_anchor.upper_strike if regime_reading.historical_anchor else None,
            "current_spot": spot,
            "label": regime_reading.signal_1_label,
        },
        top_mispricings_vs_deribit=mispricing_report.top,
        open_positions=[],
        portfolio_greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
        capital_available=capital_available,
        max_contracts_per_trade=0.1,
        max_open_positions=3,
        open_position_count=0,
        recent_options_trades=[],
    )


def _user_state_field(user_state, key: str, default):
    if isinstance(user_state, dict):
        return user_state.get(key, default)
    return getattr(user_state, key, default)


def _spot_change_pct(spot_history: list[float]) -> float:
    if not spot_history or len(spot_history) < 2:
        return 0.0
    first = spot_history[-2]
    last = spot_history[-1]
    if first <= 0:
        return 0.0
    return (last - first) / first
