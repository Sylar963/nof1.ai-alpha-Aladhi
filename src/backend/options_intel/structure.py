from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Iterable, Literal, Optional, Sequence


class StructureKind(str, Enum):
    CREDIT_PUT_SPREAD = "credit_put_spread"
    CREDIT_CALL_SPREAD = "credit_call_spread"
    DEBIT_PUT_SPREAD = "debit_put_spread"
    DEBIT_CALL_SPREAD = "debit_call_spread"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    LONG_STRADDLE = "long_straddle"
    LONG_STRANGLE = "long_strangle"
    CALENDAR_PUT = "calendar_put"
    CALENDAR_CALL = "calendar_call"
    DIAGONAL_PUT = "diagonal_put"
    DIAGONAL_CALL = "diagonal_call"
    UNKNOWN = "unknown"


class BreachState(str, Enum):
    NOMINAL = "nominal"
    WARNING = "warning"
    BREACHED = "breached"


@dataclass(frozen=True)
class OptionLeg:
    instrument_name: str
    kind: Literal["call", "put"]
    strike: Decimal
    side: Literal["long", "short"]
    contracts: Decimal
    days_to_expiry: int
    mark_price: Decimal
    delta: Optional[Decimal]
    gamma: Optional[Decimal]
    vega: Optional[Decimal]
    theta: Optional[Decimal]


@dataclass(frozen=True)
class OptionStructure:
    structure_id: str
    kind: StructureKind
    underlying: str
    legs: tuple[OptionLeg, ...]
    tenor_days_min: int
    tenor_days_max: int
    net_premium: Decimal
    is_credit: bool
    max_loss: Optional[Decimal]
    max_profit: Optional[Decimal]
    breakevens: tuple[Decimal, ...]
    short_leg_delta: Optional[Decimal]
    breach_state: BreachState
    pnl_abs: Decimal
    pnl_pct: Decimal
    aggregate_greeks: dict[str, Decimal]
    confidence: float


def compute_structure_id(legs: Iterable[OptionLeg]) -> str:
    tuples = sorted(
        (leg.instrument_name.upper(), leg.side) for leg in legs
    )
    payload = "|".join(f"{name}:{side}" for name, side in tuples)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _net_premium(legs: Sequence[OptionLeg]) -> Decimal:
    total = Decimal("0")
    for leg in legs:
        side_sign = Decimal("1") if leg.side == "short" else Decimal("-1")
        total += side_sign * leg.contracts * leg.mark_price
    return total


def _tenor_minmax(legs: Sequence[OptionLeg]) -> tuple[int, int]:
    dtes = [leg.days_to_expiry for leg in legs]
    return min(dtes), max(dtes)


def _aggregate_greeks(legs: Sequence[OptionLeg]) -> dict[str, Decimal]:
    out = {"delta": Decimal("0"), "gamma": Decimal("0"), "vega": Decimal("0"), "theta": Decimal("0")}
    for leg in legs:
        side_sign = Decimal("1") if leg.side == "long" else Decimal("-1")
        for key in ("delta", "gamma", "vega", "theta"):
            value = getattr(leg, key)
            if value is None:
                continue
            out[key] += side_sign * leg.contracts * value
    return out


def _underlying_from_legs(legs: Sequence[OptionLeg]) -> str:
    if not legs:
        return ""
    name = legs[0].instrument_name.upper()
    return name.split("-", 1)[0] if "-" in name else name


def _match_template(legs: Sequence[OptionLeg]) -> tuple[StructureKind, float]:
    n = len(legs)
    if n == 0:
        return StructureKind.UNKNOWN, 0.0

    underlyings = {_underlying_from_legs([leg]) for leg in legs}
    if len(underlyings) > 1:
        return StructureKind.UNKNOWN, 0.0

    if n == 1:
        leg = legs[0]
        if leg.side != "long":
            return StructureKind.UNKNOWN, 0.0
        if leg.kind == "call":
            return StructureKind.LONG_CALL, 1.0
        if leg.kind == "put":
            return StructureKind.LONG_PUT, 1.0
        return StructureKind.UNKNOWN, 0.0

    if n == 2:
        kinds = {leg.kind for leg in legs}
        sides = {leg.side for leg in legs}
        tenors = {leg.days_to_expiry for leg in legs}

        if (
            len(kinds) == 1
            and len(sides) == 2
            and len(tenors) == 1
            and legs[0].contracts == legs[1].contracts
        ):
            kind = next(iter(kinds))
            net_prem = _net_premium(legs)
            if kind == "put":
                return (
                    (StructureKind.CREDIT_PUT_SPREAD, 1.0)
                    if net_prem > 0
                    else (StructureKind.DEBIT_PUT_SPREAD, 1.0)
                )
            if kind == "call":
                return (
                    (StructureKind.CREDIT_CALL_SPREAD, 1.0)
                    if net_prem > 0
                    else (StructureKind.DEBIT_CALL_SPREAD, 1.0)
                )

        if len(kinds) == 1 and len(tenors) == 2:
            kind = next(iter(kinds))
            strikes = {leg.strike for leg in legs}
            same_strike = len(strikes) == 1
            if kind == "put":
                return (
                    (StructureKind.CALENDAR_PUT, 1.0)
                    if same_strike
                    else (StructureKind.DIAGONAL_PUT, 1.0)
                )
            if kind == "call":
                return (
                    (StructureKind.CALENDAR_CALL, 1.0)
                    if same_strike
                    else (StructureKind.DIAGONAL_CALL, 1.0)
                )

        if len(kinds) == 2 and sides == {"long"} and len(tenors) == 1:
            strikes = {leg.strike for leg in legs}
            if len(strikes) == 1:
                return StructureKind.LONG_STRADDLE, 1.0
            return StructureKind.LONG_STRANGLE, 1.0

    if n == 4:
        calls = [leg for leg in legs if leg.kind == "call"]
        puts = [leg for leg in legs if leg.kind == "put"]
        tenors = {leg.days_to_expiry for leg in legs}
        contracts_set = {leg.contracts for leg in legs}
        if (
            len(calls) == 2
            and len(puts) == 2
            and len(tenors) == 1
            and len(contracts_set) == 1
        ):
            calls_by_side = {leg.side: leg for leg in calls}
            puts_by_side = {leg.side: leg for leg in puts}
            if (
                set(calls_by_side.keys()) == {"long", "short"}
                and set(puts_by_side.keys()) == {"long", "short"}
            ):
                if calls_by_side["short"].strike == puts_by_side["short"].strike:
                    return StructureKind.IRON_BUTTERFLY, 1.0
                return StructureKind.IRON_CONDOR, 1.0

    return StructureKind.UNKNOWN, 0.0


def _short_leg(legs: Sequence[OptionLeg]) -> Optional[OptionLeg]:
    shorts = [leg for leg in legs if leg.side == "short"]
    if not shorts:
        return None
    shorts_with_delta = [leg for leg in shorts if leg.delta is not None]
    if shorts_with_delta:
        return max(shorts_with_delta, key=lambda leg: abs(leg.delta))
    return shorts[0]


def _short_leg_delta(legs: Sequence[OptionLeg]) -> Optional[Decimal]:
    short = _short_leg(legs)
    if short is None or short.delta is None:
        return None
    return short.delta


def _max_loss_max_profit(
    kind: StructureKind, legs: Sequence[OptionLeg], net_premium: Decimal
) -> tuple[Optional[Decimal], Optional[Decimal]]:
    contracts = legs[0].contracts if legs else Decimal("0")

    if kind in (StructureKind.CREDIT_PUT_SPREAD, StructureKind.CREDIT_CALL_SPREAD):
        strikes = sorted({leg.strike for leg in legs})
        width = strikes[1] - strikes[0]
        return (width * contracts - net_premium, net_premium)

    if kind in (StructureKind.DEBIT_PUT_SPREAD, StructureKind.DEBIT_CALL_SPREAD):
        strikes = sorted({leg.strike for leg in legs})
        width = strikes[1] - strikes[0]
        debit_paid = -net_premium
        return (debit_paid, width * contracts - debit_paid)

    if kind in (StructureKind.IRON_CONDOR, StructureKind.IRON_BUTTERFLY):
        calls = sorted([leg.strike for leg in legs if leg.kind == "call"])
        puts = sorted([leg.strike for leg in legs if leg.kind == "put"])
        call_width = calls[1] - calls[0] if len(calls) == 2 else Decimal("0")
        put_width = puts[1] - puts[0] if len(puts) == 2 else Decimal("0")
        max_width = max(call_width, put_width)
        return (max_width * contracts - net_premium, net_premium)

    if kind in (
        StructureKind.LONG_CALL,
        StructureKind.LONG_PUT,
        StructureKind.LONG_STRADDLE,
        StructureKind.LONG_STRANGLE,
    ):
        return (-net_premium, None)

    return (None, None)


def _breakevens(
    kind: StructureKind, legs: Sequence[OptionLeg], net_premium: Decimal
) -> tuple[Decimal, ...]:
    contracts = legs[0].contracts if legs else Decimal("0")
    if contracts == 0:
        return ()
    prem_per_contract = net_premium / contracts

    if kind == StructureKind.CREDIT_PUT_SPREAD:
        short_strike = max(leg.strike for leg in legs if leg.side == "short")
        return (short_strike - prem_per_contract,)
    if kind == StructureKind.CREDIT_CALL_SPREAD:
        short_strike = min(leg.strike for leg in legs if leg.side == "short")
        return (short_strike + prem_per_contract,)
    if kind == StructureKind.DEBIT_PUT_SPREAD:
        long_strike = max(leg.strike for leg in legs if leg.side == "long")
        return (long_strike - abs(prem_per_contract),)
    if kind == StructureKind.DEBIT_CALL_SPREAD:
        long_strike = min(leg.strike for leg in legs if leg.side == "long")
        return (long_strike + abs(prem_per_contract),)
    if kind == StructureKind.LONG_CALL:
        return (legs[0].strike + abs(prem_per_contract),)
    if kind == StructureKind.LONG_PUT:
        return (legs[0].strike - abs(prem_per_contract),)
    if kind in (StructureKind.IRON_CONDOR, StructureKind.IRON_BUTTERFLY):
        short_put = max(
            (leg.strike for leg in legs if leg.kind == "put" and leg.side == "short"),
            default=None,
        )
        short_call = min(
            (leg.strike for leg in legs if leg.kind == "call" and leg.side == "short"),
            default=None,
        )
        if short_put is None or short_call is None:
            return ()
        return (short_put - prem_per_contract, short_call + prem_per_contract)
    return ()


def _delta_metric_for_breach(
    kind: StructureKind, legs: Sequence[OptionLeg]
) -> Optional[Decimal]:
    if kind in (
        StructureKind.LONG_CALL,
        StructureKind.LONG_PUT,
        StructureKind.LONG_STRADDLE,
        StructureKind.LONG_STRANGLE,
    ):
        aggregate = _aggregate_greeks(legs).get("delta", Decimal("0"))
        return abs(aggregate)
    if kind in (
        StructureKind.CALENDAR_PUT,
        StructureKind.CALENDAR_CALL,
        StructureKind.DIAGONAL_PUT,
        StructureKind.DIAGONAL_CALL,
    ):
        min_dte = min(leg.days_to_expiry for leg in legs)
        near_legs = [leg for leg in legs if leg.days_to_expiry == min_dte]
        if near_legs and near_legs[0].delta is not None:
            return abs(near_legs[0].delta)
        return None
    short_d = _short_leg_delta(legs)
    return abs(short_d) if short_d is not None else None


def _breach_state(kind: StructureKind, legs: Sequence[OptionLeg]) -> BreachState:
    if not legs:
        return BreachState.NOMINAL
    dte_min = min(leg.days_to_expiry for leg in legs)
    delta_metric = _delta_metric_for_breach(kind, legs)

    breached_delta = delta_metric is not None and delta_metric >= Decimal("0.40")
    breached_dte = dte_min < 2
    if breached_delta or breached_dte:
        return BreachState.BREACHED

    warning_delta = delta_metric is not None and delta_metric >= Decimal("0.25")
    warning_dte = dte_min < 5
    if warning_delta or warning_dte:
        return BreachState.WARNING

    return BreachState.NOMINAL


def _pnl(
    net_premium: Decimal,
    entry_net_premium: Optional[Decimal],
    is_credit: bool,
) -> tuple[Decimal, Decimal]:
    if entry_net_premium is None or entry_net_premium == 0:
        return Decimal("0"), Decimal("0")
    if is_credit:
        pnl_abs = entry_net_premium - net_premium
    else:
        pnl_abs = net_premium - entry_net_premium
    pnl_pct = pnl_abs / abs(entry_net_premium)
    return pnl_abs, pnl_pct


def classify(
    legs: Sequence[OptionLeg],
    *,
    entry_net_premium: Optional[Decimal] = None,
) -> OptionStructure:
    legs_tuple = tuple(legs)
    underlying = _underlying_from_legs(legs_tuple)
    structure_id = compute_structure_id(legs_tuple)
    tenor_min, tenor_max = _tenor_minmax(legs_tuple) if legs_tuple else (0, 0)
    net_premium = _net_premium(legs_tuple)
    aggregate_greeks = _aggregate_greeks(legs_tuple)

    kind, confidence = _match_template(legs_tuple)
    is_credit = net_premium > 0

    max_loss, max_profit = _max_loss_max_profit(kind, legs_tuple, net_premium)
    breakevens = _breakevens(kind, legs_tuple, net_premium)
    short_leg_delta = _short_leg_delta(legs_tuple)
    breach_state = _breach_state(kind, legs_tuple)
    pnl_abs, pnl_pct = _pnl(net_premium, entry_net_premium, is_credit)

    return OptionStructure(
        structure_id=structure_id,
        kind=kind,
        underlying=underlying,
        legs=legs_tuple,
        tenor_days_min=tenor_min,
        tenor_days_max=tenor_max,
        net_premium=net_premium,
        is_credit=is_credit,
        max_loss=max_loss,
        max_profit=max_profit,
        breakevens=breakevens,
        short_leg_delta=short_leg_delta,
        breach_state=breach_state,
        pnl_abs=pnl_abs,
        pnl_pct=pnl_pct,
        aggregate_greeks=aggregate_greeks,
        confidence=confidence,
    )


def classify_many(legs: Sequence[OptionLeg]) -> list[OptionStructure]:
    legs_tuple = tuple(legs)
    if not legs_tuple:
        return []

    primary = classify(legs_tuple)
    if primary.kind != StructureKind.UNKNOWN:
        return [primary]

    by_tenor: dict[int, list[OptionLeg]] = {}
    for leg in legs_tuple:
        by_tenor.setdefault(leg.days_to_expiry, []).append(leg)

    classified: list[OptionStructure] = []
    orphans: list[OptionLeg] = []
    for tenor in sorted(by_tenor.keys()):
        sub = classify(by_tenor[tenor])
        if sub.kind != StructureKind.UNKNOWN:
            classified.append(sub)
        else:
            orphans.extend(by_tenor[tenor])

    if classified:
        if orphans:
            classified.append(classify(orphans))
        return classified

    return [primary]
