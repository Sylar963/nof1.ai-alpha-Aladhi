"""Tests for options strategy execution and the delta-hedge auto-rebalancer.

These tests use in-memory fake adapters and a fake clock so the strategy logic
can be exercised end-to-end without touching the network. They cover:

- Credit put: validates the order is sized + routed correctly.
- Credit spread: produces two leg orders in the right direction.
- Long call delta-hedged: places a Thalex buy AND a Hyperliquid perp short.
- Long put delta-hedged: Thalex buy AND Hyperliquid perp long.
- Delta-hedge rebalance: only fires when |drift| > threshold.
- Risk caps: oversized contracts are rejected before any order goes out.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.backend.agent.decision_schema import parse_decision
from src.backend.trading.exchange_adapter import (
    AccountState,
    ExchangeAdapter,
    OrderResult,
    PositionSnapshot,
)
from src.backend.trading.options import RiskCaps
from src.backend.trading.options_strategies import (
    DeltaHedger,
    OptionsExecutor,
)


# ---------------------------------------------------------------------------
# Fake adapters
# ---------------------------------------------------------------------------


@dataclass
class _FakeOrder:
    venue: str
    method: str
    asset: str
    amount: float
    extra: dict = field(default_factory=dict)


class FakeThalex(ExchangeAdapter):
    venue = "thalex"

    def __init__(self):
        self.calls: list[_FakeOrder] = []
        self.instruments_by_intent: dict[tuple, str] = {}
        self.positions: list[PositionSnapshot] = []
        self.delta_per_position: dict[str, float] = {}
        self.greeks_by_instrument: dict[str, dict] = {}  # used by get_greeks fallthrough
        self.risk_caps = RiskCaps(
            max_contracts_per_trade=0.1,
            max_open_positions=3,
            allowed_underlyings=["BTC"],
        )

    async def get_greeks(self, instrument_name):
        return self.greeks_by_instrument.get(instrument_name, {})

    async def resolve_intent(self, intent):
        key = (intent.underlying, intent.kind, intent.tenor_days, intent.target_strike)
        return self.instruments_by_intent.get(key, "BTC-10MAY26-65000-C")

    def preflight(self, underlying, contracts, open_positions_count):
        from src.backend.trading.options import validate_options_order
        return validate_options_order(underlying, contracts, open_positions_count, self.risk_caps)

    async def place_buy_order(self, asset, amount, slippage=0.01):
        self.calls.append(_FakeOrder(self.venue, "buy", asset, amount))
        return OrderResult(venue=self.venue, order_id="t1", asset=asset, side="buy", amount=amount, status="ok")

    async def place_sell_order(self, asset, amount, slippage=0.01):
        self.calls.append(_FakeOrder(self.venue, "sell", asset, amount))
        return OrderResult(venue=self.venue, order_id="t2", asset=asset, side="sell", amount=amount, status="ok")

    async def place_take_profit(self, asset, is_buy, amount, tp_price): return OrderResult(venue=self.venue, order_id="", asset=asset, side="tp", amount=amount, status="not_supported")
    async def place_stop_loss(self, asset, is_buy, amount, sl_price): return OrderResult(venue=self.venue, order_id="", asset=asset, side="sl", amount=amount, status="not_supported")
    async def cancel_order(self, asset, order_id): return {"status": "ok"}
    async def cancel_all_orders(self, asset): return {"status": "ok"}
    async def get_open_orders(self): return []
    async def get_recent_fills(self, limit=50): return []

    async def get_user_state(self):
        return AccountState(venue=self.venue, balance=10000, total_value=10000, positions=list(self.positions))

    async def get_current_price(self, asset):
        return 1250.0


class FakeHyperliquid(ExchangeAdapter):
    venue = "hyperliquid"

    def __init__(self):
        self.calls: list[_FakeOrder] = []
        self.perp_position_size: float = 0.0  # signed: positive=long, negative=short
        self.btc_price = 60000.0

    async def place_buy_order(self, asset, amount, slippage=0.01):
        self.calls.append(_FakeOrder(self.venue, "buy", asset, amount))
        self.perp_position_size += amount
        return OrderResult(venue=self.venue, order_id="h1", asset=asset, side="buy", amount=amount, status="ok")

    async def place_sell_order(self, asset, amount, slippage=0.01):
        self.calls.append(_FakeOrder(self.venue, "sell", asset, amount))
        self.perp_position_size -= amount
        return OrderResult(venue=self.venue, order_id="h2", asset=asset, side="sell", amount=amount, status="ok")

    async def place_take_profit(self, asset, is_buy, amount, tp_price): return OrderResult(venue=self.venue, order_id="", asset=asset, side="tp", amount=amount, status="ok")
    async def place_stop_loss(self, asset, is_buy, amount, sl_price): return OrderResult(venue=self.venue, order_id="", asset=asset, side="sl", amount=amount, status="ok")
    async def cancel_order(self, asset, order_id): return {"status": "ok"}
    async def cancel_all_orders(self, asset): return {"status": "ok"}
    async def get_open_orders(self): return []
    async def get_recent_fills(self, limit=50): return []
    async def get_user_state(self):
        return AccountState(
            venue="hyperliquid",
            balance=10000,
            total_value=10000,
            positions=[
                PositionSnapshot(
                    venue="hyperliquid",
                    asset="BTC",
                    side="long" if self.perp_position_size >= 0 else "short",
                    size=abs(self.perp_position_size),
                    entry_price=self.btc_price,
                    current_price=self.btc_price,
                    unrealized_pnl=0.0,
                )
            ] if self.perp_position_size != 0 else [],
        )

    async def get_current_price(self, asset):
        return self.btc_price


# ---------------------------------------------------------------------------
# Strategy execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_long_call_delta_hedged_places_thalex_buy_and_hyperliquid_short():
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    # Pretend the option has delta 0.5 → contracts 0.05 → hedge size = 0.025 BTC short
    thalex.delta_per_position["BTC-10MAY26-65000-C"] = 0.5

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.05,
        "rationale": "vol cheap",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    assert len(thalex.calls) == 1 and thalex.calls[0].method == "buy"
    assert len(hl.calls) == 1 and hl.calls[0].method == "sell"
    # Hedge size = contracts × delta = 0.05 × 0.5 = 0.025
    assert hl.calls[0].amount == pytest.approx(0.025)


@pytest.mark.asyncio
async def test_long_put_delta_hedged_places_thalex_buy_and_hyperliquid_long():
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    thalex.delta_per_position["BTC-10MAY26-55000-P"] = -0.4

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_put_delta_hedged",
        "underlying": "BTC",
        "kind": "put",
        "tenor_days": 30,
        "target_strike": 55000,
        "contracts": 0.05,
        "rationale": "downside hedge",
    })
    thalex.instruments_by_intent[("BTC", "put", 30, 55000.0)] = "BTC-10MAY26-55000-P"
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    assert len(hl.calls) == 1
    assert hl.calls[0].method == "buy"
    # Hedge size = contracts × |delta| = 0.05 × 0.4 = 0.02
    assert hl.calls[0].amount == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_lookup_delta_falls_through_to_get_greeks_when_cache_empty():
    """When delta_per_position has no entry, the executor must call get_greeks."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    # No delta_per_position seed; get_greeks returns the live delta instead.
    thalex.greeks_by_instrument["BTC-10MAY26-65000-C"] = {"delta": 0.42}

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.05,
        "rationale": "use real greeks",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    # Hedge size = 0.05 × 0.42 = 0.021
    assert hl.calls[0].amount == pytest.approx(0.021)


@pytest.mark.asyncio
async def test_delta_hedged_execution_refuses_guessed_atm_delta():
    """Missing live greeks must fail closed instead of using +/-0.5 guesses."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    thalex.instruments_by_intent[("BTC", "call", 30, 65000.0)] = "BTC-10MAY26-65000-C"

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.05,
        "rationale": "must have live greeks",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is False
    assert "missing live delta" in result.reason
    assert thalex.calls == []
    assert hl.calls == []


@pytest.mark.asyncio
async def test_single_tenor_unwinds_thalex_leg_when_perp_hedge_fails():
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    thalex.delta_per_position["BTC-10MAY26-65000-C"] = 0.5

    async def _failing_sell(asset, amount, slippage=0.01):
        raise RuntimeError("hyperliquid unavailable")

    hl.place_sell_order = _failing_sell

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.05,
        "rationale": "vol cheap",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is False
    assert "hyperliquid hedge failed for BTC" in result.reason
    assert [call.method for call in thalex.calls] == ["buy", "sell"]


@pytest.mark.asyncio
async def test_long_call_with_target_gamma_btc_distributes_across_default_tenors():
    """When target_gamma_btc is set on a long_call_delta_hedged decision,
    the executor must expand it into multiple Thalex legs across the default
    tenor list (7/14/30/60 days) using the sizing helper."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)

    # Seed the fake adapter so the sizing helper has an ATM contract per tenor.
    # We use a single shared instrument name + delta for simplicity; the
    # executor will request a buy on each tenor's resolved instrument.
    for tenor in (7, 14, 30, 60):
        thalex.instruments_by_intent[("BTC", "call", tenor, 60000.0)] = f"BTC-{tenor}D-60000-C"
        thalex.delta_per_position[f"BTC-{tenor}D-60000-C"] = 0.5

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,  # used as initial intent for preflight
        "target_strike": 60000,
        "target_gamma_btc": 0.004,
        "rationale": "build long gamma across the curve",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    # 4 default tenors → 4 Thalex buys + 4 perp hedges
    assert len(thalex.calls) == 4
    assert all(c.method == "buy" for c in thalex.calls)
    instrument_names = {c.asset for c in thalex.calls}
    assert instrument_names == {"BTC-7D-60000-C", "BTC-14D-60000-C", "BTC-30D-60000-C", "BTC-60D-60000-C"}
    # One perp short per leg (long call → short BTC perp)
    assert len(hl.calls) == 4
    assert all(c.method == "sell" for c in hl.calls)


@pytest.mark.asyncio
async def test_target_gamma_btc_long_put_uses_long_perp_hedge():
    """Long puts hedge by going long the perp."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)

    for tenor in (7, 14, 30, 60):
        thalex.instruments_by_intent[("BTC", "put", tenor, 55000.0)] = f"BTC-{tenor}D-55000-P"
        thalex.delta_per_position[f"BTC-{tenor}D-55000-P"] = -0.4

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_put_delta_hedged",
        "underlying": "BTC",
        "kind": "put",
        "tenor_days": 30,
        "target_strike": 55000,
        "target_gamma_btc": 0.004,
        "rationale": "downside hedge",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    assert len(thalex.calls) == 4
    assert len(hl.calls) == 4
    assert all(c.method == "buy" for c in hl.calls)


@pytest.mark.asyncio
async def test_target_gamma_btc_falls_back_to_single_tenor_when_zero():
    """target_gamma_btc=0 must NOT trigger the multi-tenor path — fall through
    to the existing single-tenor delta-hedged execution."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    thalex.delta_per_position["BTC-10MAY26-65000-C"] = 0.5

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.05,
        "target_gamma_btc": 0,  # explicit zero
        "rationale": "single tenor",
    })
    result = await executor.execute(decision, open_positions_count=0)
    assert result.ok is True
    # Single Thalex buy + single perp hedge
    assert len(thalex.calls) == 1
    assert len(hl.calls) == 1


@pytest.mark.asyncio
async def test_multi_tenor_aborts_when_one_tenor_fails_to_resolve_no_orders_submitted():
    """If any tenor can't be resolved during the staging phase, the executor must
    return failure WITHOUT submitting any Thalex or Hyperliquid orders. The
    staged-resolve-first design means a half-built curve is impossible."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)

    # Wire 3 of 4 default tenors; tenor=14 deliberately missing.
    for tenor in (7, 30, 60):
        thalex.instruments_by_intent[("BTC", "call", tenor, 60000.0)] = f"BTC-{tenor}D-60000-C"
        thalex.delta_per_position[f"BTC-{tenor}D-60000-C"] = 0.5

    # Override resolve_intent so the missing tenor returns None instead of
    # the FakeThalex's default fallback (which always returns SOMETHING).
    real_resolve = thalex.resolve_intent

    async def _resolve(intent):
        key = (intent.underlying, intent.kind, intent.tenor_days, intent.target_strike)
        if key not in thalex.instruments_by_intent:
            return None
        return thalex.instruments_by_intent[key]

    thalex.resolve_intent = _resolve

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 60000,
        "target_gamma_btc": 0.004,
        "rationale": "build gamma curve",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is False
    assert "tenor" in result.reason.lower() or "instrument" in result.reason.lower()
    # The critical assertion: NOTHING was submitted on either venue.
    assert thalex.calls == []
    assert hl.calls == []


@pytest.mark.asyncio
async def test_multi_tenor_unwinds_thalex_legs_when_a_later_submit_fails():
    """If submission for a later tenor fails after retries, the executor must
    submit opposing orders for the already-placed legs (best-effort unwind)."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)

    for tenor in (7, 14, 30, 60):
        thalex.instruments_by_intent[("BTC", "call", tenor, 60000.0)] = f"BTC-{tenor}D-60000-C"
        thalex.delta_per_position[f"BTC-{tenor}D-60000-C"] = 0.5

    # Make the third Thalex submission fail on every retry.
    real_buy = thalex.place_buy_order

    async def _failing_buy(asset, amount, slippage=0.01):
        if asset == "BTC-30D-60000-C":
            raise RuntimeError("simulated thalex outage")
        return await real_buy(asset, amount, slippage)

    thalex.place_buy_order = _failing_buy

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 60000,
        "target_gamma_btc": 0.004,
        "rationale": "x",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is False
    # Two successful Thalex buys (7d, 14d) should each be unwound by an
    # opposing sell. We assert the unwind sells happened.
    sells = [c for c in thalex.calls if c.method == "sell"]
    assert len(sells) >= 2
    sell_assets = {c.asset for c in sells}
    assert "BTC-7D-60000-C" in sell_assets
    assert "BTC-14D-60000-C" in sell_assets

    # Both perp hedges from the successful legs should also be unwound.
    # Long call hedges = sells; unwinds = buys.
    hl_buys = [c for c in hl.calls if c.method == "buy"]
    assert len(hl_buys) >= 2


@pytest.mark.asyncio
async def test_multi_tenor_retries_transient_thalex_failure_and_succeeds():
    """A transient failure on the first attempt must be retried by the
    multi-tenor submit wrapper, and the leg should ultimately succeed."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)

    for tenor in (7, 14, 30, 60):
        thalex.instruments_by_intent[("BTC", "call", tenor, 60000.0)] = f"BTC-{tenor}D-60000-C"
        thalex.delta_per_position[f"BTC-{tenor}D-60000-C"] = 0.5

    real_buy = thalex.place_buy_order
    fail_count = {"n": 0}

    async def _flaky_buy(asset, amount, slippage=0.01):
        # Fail the FIRST attempt for the 14d tenor only, then succeed.
        if asset == "BTC-14D-60000-C" and fail_count["n"] == 0:
            fail_count["n"] += 1
            raise RuntimeError("transient")
        return await real_buy(asset, amount, slippage)

    thalex.place_buy_order = _flaky_buy

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 60000,
        "target_gamma_btc": 0.004,
        "rationale": "x",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    # All 4 buys succeed; no unwind.
    buys = [c for c in thalex.calls if c.method == "buy"]
    assert len(buys) == 4
    sells = [c for c in thalex.calls if c.method == "sell"]
    assert sells == []


@pytest.mark.asyncio
async def test_multi_tenor_logs_and_aborts_on_resolve_intent_exception():
    """If resolve_intent raises (adapter init failure, network error), the
    executor must catch the exception, log it, and abort cleanly without
    submitting any orders."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)

    async def _broken_resolve(intent):
        raise RuntimeError("adapter not initialized")

    thalex.resolve_intent = _broken_resolve

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 60000,
        "target_gamma_btc": 0.004,
        "rationale": "x",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is False
    assert thalex.calls == []
    assert hl.calls == []


@pytest.mark.asyncio
async def test_target_gamma_btc_respects_max_open_positions_cap():
    """The multi-tenor expansion still has to respect max_open_positions."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    for tenor in (7, 14, 30, 60):
        thalex.instruments_by_intent[("BTC", "call", tenor, 60000.0)] = f"BTC-{tenor}D-60000-C"

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 60000,
        "target_gamma_btc": 0.004,
        "rationale": "x",
    })
    result = await executor.execute(decision, open_positions_count=3)
    assert result.ok is False
    assert "max_open_positions" in result.reason


@pytest.mark.asyncio
async def test_credit_put_spread_executes_two_legs_in_correct_directions():
    """Sell near put + buy further OTM put = defined-risk credit put spread."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    thalex.instruments_by_intent[("BTC", "put", 14, 55000.0)] = "BTC-25APR26-55000-P"
    thalex.instruments_by_intent[("BTC", "put", 14, 50000.0)] = "BTC-25APR26-50000-P"

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "sell",
        "strategy": "credit_put_spread",
        "underlying": "BTC",
        "tenor_days": 14,
        "vol_view": "short_vol",
        "legs": [
            {"kind": "put", "side": "sell", "target_strike": 55000, "contracts": 0.05},
            {"kind": "put", "side": "buy", "target_strike": 50000, "contracts": 0.05},
        ],
        "rationale": "premium sale below support",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    methods = [c.method for c in thalex.calls]
    assert methods == ["sell", "buy"]
    assert hl.calls == []  # no perp hedge for credit spreads


@pytest.mark.asyncio
async def test_multi_leg_unwinds_submitted_legs_when_later_leg_fails():
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    thalex.instruments_by_intent[("BTC", "put", 14, 55000.0)] = "BTC-25APR26-55000-P"
    thalex.instruments_by_intent[("BTC", "put", 14, 50000.0)] = "BTC-25APR26-50000-P"

    real_buy = thalex.place_buy_order

    async def _failing_buy(asset, amount, slippage=0.01):
        if asset == "BTC-25APR26-50000-P":
            raise RuntimeError("second leg failed")
        return await real_buy(asset, amount, slippage)

    thalex.place_buy_order = _failing_buy

    decision = parse_decision({
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
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is False
    methods = [c.method for c in thalex.calls]
    assert methods == ["sell", "buy"]
    assert thalex.calls[1].asset == "BTC-25APR26-55000-P"
    assert hl.calls == []


@pytest.mark.asyncio
async def test_credit_call_spread_executes_two_call_legs():
    """Sell near call + buy further OTM call = defined-risk credit call spread."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    thalex.instruments_by_intent[("BTC", "call", 14, 65000.0)] = "BTC-25APR26-65000-C"
    thalex.instruments_by_intent[("BTC", "call", 14, 70000.0)] = "BTC-25APR26-70000-C"

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "sell",
        "strategy": "credit_call_spread",
        "underlying": "BTC",
        "tenor_days": 14,
        "legs": [
            {"kind": "call", "side": "sell", "target_strike": 65000, "contracts": 0.05},
            {"kind": "call", "side": "buy", "target_strike": 70000, "contracts": 0.05},
        ],
        "rationale": "selling upside premium",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    methods = [c.method for c in thalex.calls]
    assert methods == ["sell", "buy"]


@pytest.mark.asyncio
async def test_iron_condor_executes_four_legs_two_sides():
    """Iron condor = sell put spread + sell call spread, defined risk both sides."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    thalex.instruments_by_intent[("BTC", "put", 14, 55000.0)] = "BTC-25APR26-55000-P"
    thalex.instruments_by_intent[("BTC", "put", 14, 50000.0)] = "BTC-25APR26-50000-P"
    thalex.instruments_by_intent[("BTC", "call", 14, 65000.0)] = "BTC-25APR26-65000-C"
    thalex.instruments_by_intent[("BTC", "call", 14, 70000.0)] = "BTC-25APR26-70000-C"

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "sell",
        "strategy": "iron_condor",
        "underlying": "BTC",
        "tenor_days": 14,
        "vol_view": "short_vol",
        "legs": [
            {"kind": "put", "side": "sell", "target_strike": 55000, "contracts": 0.05},
            {"kind": "put", "side": "buy", "target_strike": 50000, "contracts": 0.05},
            {"kind": "call", "side": "sell", "target_strike": 65000, "contracts": 0.05},
            {"kind": "call", "side": "buy", "target_strike": 70000, "contracts": 0.05},
        ],
        "rationale": "vol crush, defined risk both sides",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    assert len(thalex.calls) == 4
    sells = [c for c in thalex.calls if c.method == "sell"]
    buys = [c for c in thalex.calls if c.method == "buy"]
    assert len(sells) == 2
    assert len(buys) == 2
    assert hl.calls == []  # iron condor is self-contained, no perp hedge


@pytest.mark.asyncio
async def test_iron_condor_requires_at_least_two_legs():
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "sell",
        "strategy": "iron_condor",
        "underlying": "BTC",
        "tenor_days": 14,
        "legs": [],
        "rationale": "x",
    })
    result = await executor.execute(decision, open_positions_count=0)
    assert result.ok is False
    assert "legs" in result.reason.lower()


@pytest.mark.asyncio
async def test_calendar_spread_uses_per_leg_tenor_days():
    """Calendar spread legs override decision-level tenor with their own tenor_days."""
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)
    # Different instruments per tenor
    thalex.instruments_by_intent[("BTC", "call", 7, 60000.0)] = "BTC-WEEKLY-60000-C"
    thalex.instruments_by_intent[("BTC", "call", 30, 60000.0)] = "BTC-MONTHLY-60000-C"

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "vol_arb",
        "underlying": "BTC",
        "entry_kind": "calendar",
        "legs": [
            {"kind": "call", "side": "sell", "target_strike": 60000, "contracts": 0.05, "tenor_days": 7},
            {"kind": "call", "side": "buy", "target_strike": 60000, "contracts": 0.05, "tenor_days": 30},
        ],
        "rationale": "term structure dislocation",
    })
    result = await executor.execute(decision, open_positions_count=0)

    assert result.ok is True
    assert len(thalex.calls) == 2
    # Each leg should resolve to a different instrument because tenors differ.
    instrument_names = {c.asset for c in thalex.calls}
    assert "BTC-WEEKLY-60000-C" in instrument_names
    assert "BTC-MONTHLY-60000-C" in instrument_names


@pytest.mark.asyncio
async def test_executor_rejects_oversized_order_via_risk_caps():
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)

    decision = parse_decision({
        "venue": "thalex",
        "asset": "BTC",
        "action": "buy",
        "strategy": "long_call_delta_hedged",
        "underlying": "BTC",
        "kind": "call",
        "tenor_days": 30,
        "target_strike": 65000,
        "contracts": 0.5,  # > 0.1 cap
        "rationale": "oversized",
    })
    result = await executor.execute(decision, open_positions_count=0)
    assert result.ok is False
    assert "max_contracts_per_trade" in result.reason
    assert thalex.calls == []
    assert hl.calls == []


@pytest.mark.asyncio
async def test_executor_rejects_when_max_open_positions_reached():
    thalex = FakeThalex()
    hl = FakeHyperliquid()
    executor = OptionsExecutor(thalex=thalex, hyperliquid=hl)

    decision = parse_decision({
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
    result = await executor.execute(decision, open_positions_count=3)
    assert result.ok is False
    assert "max_open_positions" in result.reason


# ---------------------------------------------------------------------------
# Delta hedger
# ---------------------------------------------------------------------------


def test_delta_hedger_no_action_when_drift_below_threshold():
    hedger = DeltaHedger(threshold=0.05)
    action = hedger.compute_rebalance(
        target_delta=-0.025,  # short 0.025 BTC equivalent
        current_perp_delta=-0.02,  # currently short 0.02
    )
    # |drift| = 0.005 < 0.05 → no action
    assert action.contracts_to_trade == pytest.approx(0.0)


def test_delta_hedger_acts_when_drift_above_threshold():
    hedger = DeltaHedger(threshold=0.05)
    action = hedger.compute_rebalance(
        target_delta=-0.10,
        current_perp_delta=-0.02,
    )
    # drift = -0.10 - (-0.02) = -0.08 → need to short 0.08 more BTC
    assert action.side == "sell"
    assert action.contracts_to_trade == pytest.approx(0.08)


def test_delta_hedger_handles_long_drift():
    hedger = DeltaHedger(threshold=0.05)
    action = hedger.compute_rebalance(
        target_delta=0.0,
        current_perp_delta=-0.10,
    )
    # Need to buy 0.10 BTC to flatten the short
    assert action.side == "buy"
    assert action.contracts_to_trade == pytest.approx(0.10)


def test_delta_hedger_default_threshold_is_002_btc():
    """Default threshold is 0.02 BTC drift — small enough to capture pullbacks
    but loose enough that quiet markets don't churn perp fees."""
    assert DeltaHedger().threshold == pytest.approx(0.02)


def test_delta_hedger_default_threshold_fires_at_0025_drift():
    """A 0.025 BTC drift breaches the new default and must trigger a rebalance."""
    hedger = DeltaHedger()  # uses default 0.02
    action = hedger.compute_rebalance(target_delta=-0.025, current_perp_delta=0.0)
    assert action.side == "sell"
    assert action.contracts_to_trade == pytest.approx(0.025)


def test_delta_hedger_default_threshold_holds_at_0015_drift():
    """A 0.015 BTC drift sits below the new default and must be a no-op."""
    hedger = DeltaHedger()  # uses default 0.02
    action = hedger.compute_rebalance(target_delta=-0.015, current_perp_delta=0.0)
    assert action.side == "noop"
    assert action.contracts_to_trade == pytest.approx(0.0)
