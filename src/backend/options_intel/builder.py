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
from typing import Any, Optional

from src.backend.options_intel.iv_history_store import IVHistoryRow, IVHistoryStore
from src.backend.options_intel.mispricing import scan_mispricings
from src.backend.options_intel.portfolio import aggregate_portfolio_greeks
from src.backend.options_intel.regime import classify_regime
from src.backend.options_intel.snapshot import OptionsContext
from src.backend.options_intel.technicals import (
    compute_keltner_channel,
    compute_opening_range,
)
from src.backend.options_intel.trade_history import fetch_recent_options_trades
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
    intraday_minute_prices: Optional[list[tuple[int, float]]] = None,
    daily_closes_for_keltner: Optional[list[float]] = None,
    db_session: Any = None,
    recent_trades_limit: int = 5,
    hyperliquid: Any = None,
    hedge_underlying: str = "BTC",
    recent_options_skips: Optional[list[dict]] = None,
) -> OptionsContext:
    """Run the full options-intel pipeline and return a snapshot.

    Args:
        thalex: ThalexAPI instance (or stand-in). Must expose
            ``_instruments_cache``, ``get_user_state``, and ``get_greeks``.
        deribit: DeribitPublicClient. Must expose
            ``get_book_summary_by_currency`` and ``get_index_price``.
        iv_history: persistent SQLite store for straddle anchors.
        spot_history: ~16 daily closes (most recent last) for realized vol.
        persist_anchor: when True, write today's 15-day straddle anchor to
            ``iv_history`` so the regime classifier has a fresh data point.
        top_mispricings: how many of the largest IV gaps to surface.
        min_edge_bps: drop mispricings below this absolute IV diff.
        use_interpolation: forward to ``scan_mispricings``; when True the
            scanner fills unmatched Thalex tenors with synthetic Deribit IVs.
        intraday_minute_prices: optional list of ``(unix_seconds, price)``
            tuples for the technical opening-range calc. When None or
            empty the snapshot's ``opening_range`` stays ``unknown``.
        daily_closes_for_keltner: optional longer daily-close series (>=20
            entries) used to compute the Keltner channel. Falls back to
            ``spot_history`` when not supplied.

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

    today_for_signals = today or datetime.now(timezone.utc).date()

    # Opening range — first hour of the current UTC day on BTC.
    opening_range = compute_opening_range(
        intraday_minute_prices or [],
        today=today_for_signals,
        current_spot=spot,
    )

    # Keltner channel — EMA20 ± 2×ATR(14) on BTC daily closes. Falls back
    # to the same spot_history we use for realized vol when no longer
    # series is supplied.
    keltner_closes = list(daily_closes_for_keltner or spot_history or [])
    keltner = compute_keltner_channel(
        closes=keltner_closes,
        period=20,
        atr_period=14,
        atr_multiplier=2.0,
        current_spot=spot,
    )

    # Portfolio greeks — walk Thalex positions and aggregate per-leg greeks.
    raw_positions = _extract_positions(user_state)
    portfolio = await aggregate_portfolio_greeks(
        positions=raw_positions,
        greeks_source=thalex,
        today=today_for_signals,
    )

    # Recent options trade history — last N closed Thalex trades pulled from
    # the bot's existing Trade table when a session is supplied. Skipped
    # silently when no session is wired (PR A/B/C tests don't need it).
    recent_options_trades: list[dict] = []
    if db_session is not None:
        recent_options_trades = fetch_recent_options_trades(db_session, limit=recent_trades_limit)

    # Hyperliquid hedge budget — the perp leg of delta-hedged options pulls
    # from HL collateral, so the options agent needs to see it to avoid
    # proposing strategies we'd have to unwind immediately.
    hl_free_margin = 0.0
    hl_max_leverage = 1
    max_hedge_notional = 0.0
    if hyperliquid is not None:
        try:
            info = await hyperliquid.get_free_margin_info()
            withdrawable = float(info.get("withdrawable") or 0.0)
            free_margin = float(info.get("free_margin") or 0.0)
            candidates = [v for v in (withdrawable, free_margin) if v and v > 0]
            hl_free_margin = min(candidates) if candidates else 0.0
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("hyperliquid get_free_margin_info failed: %s", exc)
        try:
            hl_max_leverage = max(int(await hyperliquid.get_max_leverage(hedge_underlying) or 1), 1)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("hyperliquid get_max_leverage(%s) failed: %s", hedge_underlying, exc)
        # Mirror the perps preflight math so the LLM's cap matches what the
        # guard will actually enforce at execution time (/1.05 buffer).
        max_hedge_notional = round((hl_free_margin * hl_max_leverage) / 1.05, 2) if hl_free_margin > 0 else 0.0

    return OptionsContext(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        spot=float(spot),
        spot_24h_change_pct=_spot_change_pct(spot_history),
        opening_range=opening_range,
        keltner=keltner,
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
        open_positions=portfolio["open_positions"],
        portfolio_greeks=portfolio["portfolio_greeks"],
        capital_available=capital_available,
        max_contracts_per_trade=0.1,
        max_open_positions=3,
        open_position_count=len(portfolio["open_positions"]),
        hyperliquid_free_margin=round(hl_free_margin, 2),
        hyperliquid_max_leverage=hl_max_leverage,
        max_hedge_notional=max_hedge_notional,
        recent_options_trades=recent_options_trades,
        recent_options_skips=list(recent_options_skips or []),
    )


def _extract_positions(user_state) -> list[dict]:
    """Pull the positions list from a user_state response in either shape.

    The Thalex adapter normalizes get_user_state into an :class:`AccountState`
    dataclass with a ``positions`` attribute. The pure-dict shape is also
    supported (handy for tests + future-proofing).
    """
    if user_state is None:
        return []
    if isinstance(user_state, dict):
        raw = user_state.get("positions") or []
    else:
        raw = getattr(user_state, "positions", []) or []

    out: list[dict] = []
    for entry in raw:
        if isinstance(entry, dict):
            normalized = dict(entry)
            instrument_name = (
                entry.get("instrument_name")
                or entry.get("instrument")
                or entry.get("asset")
            )
            if "size" in entry:
                size = entry.get("size")
            elif "position" in entry:
                size = entry.get("position")
            elif "amount" in entry:
                size = entry.get("amount")
            else:
                size = 0.0
            side = entry.get("side") or "long"
            normalized["instrument_name"] = instrument_name
            normalized["size"] = size
            normalized["side"] = side
            out.append(normalized)
            continue
        # PositionSnapshot dataclass — convert to the dict the aggregator expects
        out.append({
            "instrument_name": getattr(entry, "instrument_name", None) or getattr(entry, "asset", ""),
            "size": getattr(entry, "size", 0.0),
            "side": getattr(entry, "side", "long"),
        })
    return out


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
