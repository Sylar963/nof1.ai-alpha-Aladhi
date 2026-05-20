import dataclasses
import pytest
from decimal import Decimal

from src.backend.options_intel.structure import (
    BreachState,
    OptionLeg,
    OptionStructure,
    StructureKind,
)


def test_structure_kind_enum_has_all_phase1_kinds():
    expected = {
        "credit_put_spread", "credit_call_spread",
        "debit_put_spread", "debit_call_spread",
        "iron_condor", "iron_butterfly",
        "long_call", "long_put", "long_straddle", "long_strangle",
        "calendar_put", "calendar_call",
        "diagonal_put", "diagonal_call",
        "unknown",
    }
    actual = {k.value for k in StructureKind}
    assert actual == expected


def test_breach_state_enum_values():
    assert {b.value for b in BreachState} == {"nominal", "warning", "breached"}


def test_option_leg_is_frozen_dataclass():
    leg = OptionLeg(
        instrument_name="BTC-27JUN26-100000-P",
        kind="put",
        strike=Decimal("100000"),
        side="long",
        contracts=Decimal("0.1"),
        days_to_expiry=14,
        mark_price=Decimal("1500"),
        delta=Decimal("-0.30"),
        gamma=None,
        vega=None,
        theta=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        leg.kind = "call"


def test_option_structure_is_frozen_dataclass():
    leg = OptionLeg(
        instrument_name="BTC-27JUN26-100000-P",
        kind="put",
        strike=Decimal("100000"),
        side="long",
        contracts=Decimal("0.1"),
        days_to_expiry=14,
        mark_price=Decimal("1500"),
        delta=Decimal("-0.30"),
        gamma=None,
        vega=None,
        theta=None,
    )
    structure = OptionStructure(
        structure_id="abc123",
        kind=StructureKind.LONG_PUT,
        underlying="BTC",
        legs=(leg,),
        tenor_days_min=14,
        tenor_days_max=14,
        net_premium=Decimal("-150"),
        is_credit=False,
        max_loss=Decimal("150"),
        max_profit=None,
        breakevens=(Decimal("98500"),),
        short_leg_delta=None,
        breach_state=BreachState.NOMINAL,
        pnl_abs=Decimal("0"),
        pnl_pct=Decimal("0"),
        aggregate_greeks={"delta": Decimal("-0.30"), "gamma": Decimal("0"), "vega": Decimal("0"), "theta": Decimal("0")},
        confidence=1.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        structure.kind = StructureKind.LONG_CALL


from src.backend.options_intel.structure import compute_structure_id


def _put_leg(strike: int, side: str, contracts: str = "0.1") -> OptionLeg:
    return OptionLeg(
        instrument_name=f"BTC-27JUN26-{strike}-P",
        kind="put",
        strike=Decimal(strike),
        side=side,
        contracts=Decimal(contracts),
        days_to_expiry=14,
        mark_price=Decimal("100"),
        delta=Decimal("-0.20"),
        gamma=None,
        vega=None,
        theta=None,
    )


def test_structure_id_stable_across_leg_order():
    leg_a = _put_leg(100000, "long")
    leg_b = _put_leg(95000, "short")
    id1 = compute_structure_id([leg_a, leg_b])
    id2 = compute_structure_id([leg_b, leg_a])
    assert id1 == id2
    assert len(id1) == 40


def test_structure_id_differs_when_sides_swap():
    leg_a = _put_leg(100000, "long")
    leg_b = _put_leg(95000, "short")
    leg_a_short = _put_leg(100000, "short")
    leg_b_long = _put_leg(95000, "long")
    assert compute_structure_id([leg_a, leg_b]) != compute_structure_id([leg_a_short, leg_b_long])


def test_structure_id_case_insensitive_on_instrument_name():
    leg_lower = OptionLeg(
        instrument_name="btc-27jun26-100000-p",
        kind="put", strike=Decimal("100000"), side="long",
        contracts=Decimal("0.1"), days_to_expiry=14, mark_price=Decimal("100"),
        delta=None, gamma=None, vega=None, theta=None,
    )
    leg_upper = OptionLeg(
        instrument_name="BTC-27JUN26-100000-P",
        kind="put", strike=Decimal("100000"), side="long",
        contracts=Decimal("0.1"), days_to_expiry=14, mark_price=Decimal("100"),
        delta=None, gamma=None, vega=None, theta=None,
    )
    assert compute_structure_id([leg_lower]) == compute_structure_id([leg_upper])


from src.backend.options_intel.structure import classify


def _call_leg(strike: int, side: str, mark: str = "200", dte: int = 14, delta: str = "0.30") -> OptionLeg:
    return OptionLeg(
        instrument_name=f"BTC-27JUN26-{strike}-C",
        kind="call",
        strike=Decimal(strike),
        side=side,
        contracts=Decimal("0.1"),
        days_to_expiry=dte,
        mark_price=Decimal(mark),
        delta=Decimal(delta),
        gamma=None, vega=None, theta=None,
    )


def _put_leg_full(strike: int, side: str, mark: str = "200", dte: int = 14, delta: str = "-0.30") -> OptionLeg:
    return OptionLeg(
        instrument_name=f"BTC-27JUN26-{strike}-P",
        kind="put",
        strike=Decimal(strike),
        side=side,
        contracts=Decimal("0.1"),
        days_to_expiry=dte,
        mark_price=Decimal(mark),
        delta=Decimal(delta),
        gamma=None, vega=None, theta=None,
    )


def test_classify_long_call():
    s = classify([_call_leg(100000, "long")])
    assert s.kind == StructureKind.LONG_CALL
    assert s.is_credit is False
    assert s.confidence == 1.0


def test_classify_long_put():
    s = classify([_put_leg_full(100000, "long")])
    assert s.kind == StructureKind.LONG_PUT
    assert s.is_credit is False


def test_classify_credit_put_spread():
    legs = [
        _put_leg_full(100000, "short", mark="300", delta="-0.30"),
        _put_leg_full(90000, "long", mark="100", delta="-0.10"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.CREDIT_PUT_SPREAD
    assert s.is_credit is True
    assert s.confidence == 1.0


def test_classify_debit_put_spread():
    legs = [
        _put_leg_full(100000, "long", mark="300", delta="-0.30"),
        _put_leg_full(90000, "short", mark="100", delta="-0.10"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.DEBIT_PUT_SPREAD
    assert s.is_credit is False


def test_classify_credit_call_spread():
    legs = [
        _call_leg(100000, "short", mark="300", delta="0.30"),
        _call_leg(110000, "long", mark="100", delta="0.10"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.CREDIT_CALL_SPREAD
    assert s.is_credit is True


def test_classify_debit_call_spread():
    legs = [
        _call_leg(100000, "long", mark="300", delta="0.30"),
        _call_leg(110000, "short", mark="100", delta="0.10"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.DEBIT_CALL_SPREAD
    assert s.is_credit is False


def test_classify_iron_condor():
    legs = [
        _put_leg_full(95000, "short", mark="300"),
        _put_leg_full(90000, "long", mark="100"),
        _call_leg(105000, "short", mark="300"),
        _call_leg(110000, "long", mark="100"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.IRON_CONDOR
    assert s.is_credit is True
    assert s.confidence == 1.0


def test_classify_iron_butterfly():
    legs = [
        _put_leg_full(100000, "short", mark="600"),
        _call_leg(100000, "short", mark="600"),
        _put_leg_full(90000, "long", mark="100"),
        _call_leg(110000, "long", mark="100"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.IRON_BUTTERFLY
    assert s.is_credit is True


def test_classify_calendar_put():
    near = _put_leg_full(100000, "short", mark="200", dte=7)
    far = _put_leg_full(100000, "long", mark="500", dte=28)
    s = classify([near, far])
    assert s.kind == StructureKind.CALENDAR_PUT


def test_classify_calendar_call():
    near = _call_leg(100000, "short", mark="200", dte=7)
    far = _call_leg(100000, "long", mark="500", dte=28)
    s = classify([near, far])
    assert s.kind == StructureKind.CALENDAR_CALL


def test_classify_diagonal_put():
    near = _put_leg_full(95000, "short", mark="150", dte=7)
    far = _put_leg_full(100000, "long", mark="500", dte=28)
    s = classify([near, far])
    assert s.kind == StructureKind.DIAGONAL_PUT


def test_classify_diagonal_call():
    near = _call_leg(105000, "short", mark="150", dte=7)
    far = _call_leg(100000, "long", mark="500", dte=28)
    s = classify([near, far])
    assert s.kind == StructureKind.DIAGONAL_CALL


def test_classify_long_straddle():
    legs = [
        _call_leg(100000, "long", mark="500"),
        _put_leg_full(100000, "long", mark="500"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.LONG_STRADDLE


def test_classify_long_strangle():
    legs = [
        _call_leg(105000, "long", mark="200"),
        _put_leg_full(95000, "long", mark="200"),
    ]
    s = classify(legs)
    assert s.kind == StructureKind.LONG_STRANGLE


def test_classify_unknown_for_mismatched_contracts_vertical():
    leg_a = _put_leg_full(100000, "short")
    leg_b = OptionLeg(
        instrument_name="BTC-27JUN26-90000-P",
        kind="put", strike=Decimal("90000"), side="long",
        contracts=Decimal("0.2"),
        days_to_expiry=14, mark_price=Decimal("100"),
        delta=Decimal("-0.10"), gamma=None, vega=None, theta=None,
    )
    s = classify([leg_a, leg_b])
    assert s.kind == StructureKind.UNKNOWN
    assert s.confidence == 0.0


def test_classify_unknown_for_mixed_underlyings():
    btc_leg = _put_leg_full(100000, "long")
    eth_leg = OptionLeg(
        instrument_name="ETH-27JUN26-3000-P",
        kind="put", strike=Decimal("3000"), side="short",
        contracts=Decimal("0.1"), days_to_expiry=14, mark_price=Decimal("50"),
        delta=Decimal("-0.20"), gamma=None, vega=None, theta=None,
    )
    s = classify([btc_leg, eth_leg])
    assert s.kind == StructureKind.UNKNOWN
    assert s.confidence == 0.0


def test_classify_unknown_preserves_legs():
    legs = [_put_leg_full(100000, "short"), _call_leg(105000, "long")]
    s = classify(legs)
    assert s.kind == StructureKind.UNKNOWN
    assert len(s.legs) == 2


def test_classify_naked_short_is_unknown():
    legs = [_put_leg_full(100000, "short")]
    s = classify(legs)
    assert s.kind == StructureKind.UNKNOWN
    assert s.confidence == 0.0
