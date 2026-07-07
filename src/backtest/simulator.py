"""Deterministic fill simulator mirroring the live engine's perps execution semantics."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Position:
    asset: str
    size: float
    entry_price: float
    opened_at: datetime
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    fees_open: float = 0.0


@dataclass
class ClosedTrade:
    asset: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    fees: float
    opened_at: datetime
    closed_at: datetime
    hold_seconds: float
    reason: str


class Simulator:
    """Single-net-position-per-asset fill simulator.

    Mirrors TradingBotEngine semantics: position-aware buy/sell handling
    (flips close then open), mandatory stop validation, per-trade risk cap
    and gross-leverage cap clamping, taker fees on every fill, and adverse
    slippage on market orders. TP/SL are evaluated against the next cycle's
    price with gap-through modelling on stops.
    """

    def __init__(
        self,
        starting_equity: float = 10_000.0,
        slippage_bps: float = 5.0,
        fee_bps: float = 4.5,
        max_risk_per_trade_pct: float = 1.0,
        max_gross_leverage: float = 3.0,
    ):
        self.starting_equity = float(starting_equity)
        self.cash = float(starting_equity)
        self.slippage_bps = float(slippage_bps)
        self.fee_bps = float(fee_bps)
        self.max_risk_per_trade_pct = float(max_risk_per_trade_pct)
        self.max_gross_leverage = float(max_gross_leverage)
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[ClosedTrade] = []
        self.skipped: List[Dict] = []
        self.clamp_notes: List[Dict] = []
        self.fees_paid = 0.0
        self.equity_curve: List[Dict] = []

    def equity(self, prices: Dict[str, float]) -> float:
        """Cash plus mark-to-market unrealized PnL at the given prices."""
        unrealized = 0.0
        for asset, pos in self.positions.items():
            px = prices.get(asset) or pos.entry_price
            unrealized += (px - pos.entry_price) * pos.size
        return self.cash + unrealized

    def _fill_price(self, price: float, is_buy: bool) -> float:
        slip = self.slippage_bps / 10_000.0
        return price * (1 + slip) if is_buy else price * (1 - slip)

    def _charge_fee(self, notional: float) -> float:
        fee = abs(notional) * self.fee_bps / 10_000.0
        self.fees_paid += fee
        self.cash -= fee
        return fee

    def _validate_risk_levels(
        self,
        action: str,
        current_price: float,
        tp_price: Optional[float],
        sl_price: Optional[float],
    ) -> Optional[str]:
        """Return a rejection reason when TP/SL levels are missing or inverted."""
        if sl_price is None:
            return "missing sl_price (a stop is mandatory for any new exposure)"
        sl = float(sl_price)
        tp = float(tp_price) if tp_price is not None else None
        if action == 'buy':
            if sl >= current_price:
                return f"sl_price {sl} must be below entry {current_price} for a buy"
            if tp is not None and tp <= current_price:
                return f"tp_price {tp} must be above entry {current_price} for a buy"
        else:
            if sl <= current_price:
                return f"sl_price {sl} must be above entry {current_price} for a sell"
            if tp is not None and tp >= current_price:
                return f"tp_price {tp} must be below entry {current_price} for a sell"
        return None

    def _clamp_allocation(
        self,
        asset: str,
        action: str,
        allocation: float,
        current_price: float,
        sl_price: float,
        existing_size: float,
        prices: Dict[str, float],
        timestamp: datetime,
    ) -> float:
        equity = self.equity(prices)
        if allocation <= 0 or equity <= 0 or current_price <= 0:
            return allocation

        def _note(text: str):
            self.clamp_notes.append({"timestamp": timestamp, "asset": asset, "note": text})

        stop_distance = abs(current_price - float(sl_price)) / current_price
        if self.max_risk_per_trade_pct > 0 and stop_distance > 0:
            risk_cap = equity * (self.max_risk_per_trade_pct / 100.0) / stop_distance
            if allocation > risk_cap:
                _note(
                    f"allocation clamped ${allocation:,.0f} -> ${risk_cap:,.0f} "
                    f"({self.max_risk_per_trade_pct}% risk cap at "
                    f"{stop_distance*100:.2f}% stop distance)"
                )
                allocation = risk_cap

        if self.max_gross_leverage > 0:
            gross_other = 0.0
            for pos_asset, pos in self.positions.items():
                px = prices.get(pos_asset) or pos.entry_price
                notional = abs(pos.size) * px
                if notional <= 0:
                    continue
                if pos_asset == asset:
                    is_add = (action == 'buy' and existing_size > 0) or (
                        action == 'sell' and existing_size < 0
                    )
                    if is_add:
                        gross_other += notional
                    continue
                gross_other += notional
            gross_cap = equity * self.max_gross_leverage - gross_other
            if gross_cap <= 0:
                _note(
                    f"gross exposure ${gross_other:,.0f} already at/above "
                    f"{self.max_gross_leverage}x equity cap; allocation -> 0"
                )
                return 0.0
            if allocation > gross_cap:
                _note(
                    f"allocation clamped ${allocation:,.0f} -> ${gross_cap:,.0f} "
                    f"({self.max_gross_leverage}x gross leverage cap)"
                )
                allocation = gross_cap

        return allocation

    def _open_position(
        self,
        asset: str,
        size: float,
        fill_price: float,
        timestamp: datetime,
        tp_price: Optional[float],
        sl_price: Optional[float],
    ):
        fee = self._charge_fee(abs(size) * fill_price)
        self.positions[asset] = Position(
            asset=asset,
            size=size,
            entry_price=fill_price,
            opened_at=timestamp,
            tp_price=tp_price,
            sl_price=sl_price,
            fees_open=fee,
        )

    def _add_to_position(
        self,
        pos: Position,
        add_size: float,
        fill_price: float,
        tp_price: Optional[float],
        sl_price: Optional[float],
    ):
        fee = self._charge_fee(add_size * fill_price)
        old_abs = abs(pos.size)
        new_abs = old_abs + add_size
        pos.entry_price = (old_abs * pos.entry_price + add_size * fill_price) / new_abs
        pos.size = new_abs if pos.size > 0 else -new_abs
        pos.fees_open += fee
        pos.tp_price = tp_price
        pos.sl_price = sl_price

    def _close_position(
        self,
        asset: str,
        exit_price: float,
        timestamp: datetime,
        reason: str,
    ) -> ClosedTrade:
        pos = self.positions.pop(asset)
        exit_fee = self._charge_fee(abs(pos.size) * exit_price)
        gross = (exit_price - pos.entry_price) * pos.size
        self.cash += gross
        fees = pos.fees_open + exit_fee
        hold = max((timestamp - pos.opened_at).total_seconds(), 0.0)
        trade = ClosedTrade(
            asset=asset,
            side='long' if pos.size > 0 else 'short',
            size=abs(pos.size),
            entry_price=pos.entry_price,
            exit_price=exit_price,
            pnl=gross - fees,
            fees=fees,
            opened_at=pos.opened_at,
            closed_at=timestamp,
            hold_seconds=hold,
            reason=reason,
        )
        self.closed_trades.append(trade)
        return trade

    def check_exits(self, prices: Dict[str, float], timestamp: datetime):
        """Evaluate TP/SL against this cycle's prices (the cycle after entry).

        A long's SL fills at ``min(sl, price)`` and a short's at
        ``max(sl, price)`` to model gap-through; TP fills exactly at the
        TP level, only when price crosses it. SL wins when both cross.
        """
        for asset in list(self.positions.keys()):
            pos = self.positions[asset]
            px = prices.get(asset)
            if not px or px <= 0:
                continue
            if pos.size > 0:
                if pos.sl_price is not None and px <= pos.sl_price:
                    self._close_position(asset, min(pos.sl_price, px), timestamp, 'sl')
                elif pos.tp_price is not None and px >= pos.tp_price:
                    self._close_position(asset, pos.tp_price, timestamp, 'tp')
            else:
                if pos.sl_price is not None and px >= pos.sl_price:
                    self._close_position(asset, max(pos.sl_price, px), timestamp, 'sl')
                elif pos.tp_price is not None and px <= pos.tp_price:
                    self._close_position(asset, pos.tp_price, timestamp, 'tp')

    def apply_decision(self, decision: Dict, prices: Dict[str, float], timestamp: datetime):
        """Apply one LLM trade decision at this cycle's prices."""
        asset = decision.get('asset')
        action = decision.get('action')
        if action not in ('buy', 'sell') or not asset:
            return
        price = prices.get(asset)
        if not price or price <= 0:
            self._skip(timestamp, asset, action, f"invalid price {price!r}")
            return

        allocation = float(decision.get('allocation_usd') or 0)
        tp_price = decision.get('tp_price')
        sl_price = decision.get('sl_price')

        existing = self.positions.get(asset)
        existing_size = existing.size if existing else 0.0

        if allocation > 0:
            reason = self._validate_risk_levels(action, price, tp_price, sl_price)
            if reason:
                self._skip(timestamp, asset, action, reason)
                return
            allocation = self._clamp_allocation(
                asset, action, allocation, price,
                float(sl_price), existing_size, prices, timestamp,
            )

        desired_size = allocation / price
        is_buy = action == 'buy'
        fill = self._fill_price(price, is_buy)

        if is_buy and existing_size < 0:
            self._close_position(asset, fill, timestamp, 'flip' if desired_size > 0 else 'close')
            if desired_size > 0:
                self._open_position(asset, desired_size, fill, timestamp, tp_price, sl_price)
        elif not is_buy and existing_size > 0:
            self._close_position(asset, fill, timestamp, 'flip' if desired_size > 0 else 'close')
            if desired_size > 0:
                self._open_position(asset, -desired_size, fill, timestamp, tp_price, sl_price)
        else:
            if desired_size <= 0:
                return
            if existing is None:
                signed = desired_size if is_buy else -desired_size
                self._open_position(asset, signed, fill, timestamp, tp_price, sl_price)
            else:
                self._add_to_position(existing, desired_size, fill, tp_price, sl_price)

    def _skip(self, timestamp: datetime, asset: str, action: str, reason: str):
        self.skipped.append({
            "timestamp": timestamp,
            "asset": asset,
            "action": action,
            "reason": reason,
        })

    def mark(self, prices: Dict[str, float], timestamp: datetime):
        """Record an equity-curve point at this cycle's prices."""
        self.equity_curve.append({"timestamp": timestamp, "equity": self.equity(prices)})

    def result(self) -> Dict:
        """Snapshot of simulator state for metrics computation."""
        final_equity = (
            self.equity_curve[-1]["equity"] if self.equity_curve else self.cash
        )
        return {
            "starting_equity": self.starting_equity,
            "final_equity": final_equity,
            "equity_curve": list(self.equity_curve),
            "closed_trades": list(self.closed_trades),
            "open_positions": dict(self.positions),
            "fees_paid": self.fees_paid,
            "skipped": list(self.skipped),
            "clamp_notes": list(self.clamp_notes),
        }
