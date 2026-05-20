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
        if len(calls) == 2 and len(puts) == 2 and len(tenors) == 1:
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


def classify(legs: Sequence[OptionLeg]) -> OptionStructure:
    legs_tuple = tuple(legs)
    underlying = _underlying_from_legs(legs_tuple)
    structure_id = compute_structure_id(legs_tuple)
    tenor_min, tenor_max = _tenor_minmax(legs_tuple) if legs_tuple else (0, 0)
    net_premium = _net_premium(legs_tuple)
    aggregate_greeks = _aggregate_greeks(legs_tuple)
    kind, confidence = _match_template(legs_tuple)

    return OptionStructure(
        structure_id=structure_id,
        kind=kind,
        underlying=underlying,
        legs=legs_tuple,
        tenor_days_min=tenor_min,
        tenor_days_max=tenor_max,
        net_premium=net_premium,
        is_credit=net_premium > 0,
        max_loss=None,
        max_profit=None,
        breakevens=(),
        short_leg_delta=None,
        breach_state=BreachState.NOMINAL,
        pnl_abs=Decimal("0"),
        pnl_pct=Decimal("0"),
        aggregate_greeks=aggregate_greeks,
        confidence=confidence,
    )
