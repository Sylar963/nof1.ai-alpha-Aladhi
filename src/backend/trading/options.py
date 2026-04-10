"""Pure-logic helpers for options trading on Thalex.

Nothing in this module touches the network. It exists so that instrument
parsing, intent → instrument resolution, and risk-cap validation are easy to
test in isolation and reusable across the Thalex adapter, the strategy layer,
and any future GUI surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Iterable, Optional


# Thalex instrument-name format example:  BTC-27JUN25-100000-C
# Underlying-DDMMMYY-STRIKE-(C|P)

_MONTH_TO_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_NUM_TO_MONTH = {v: k for k, v in _MONTH_TO_NUM.items()}


@dataclass
class InstrumentSpec:
    """Structured representation of a Thalex option instrument name."""

    underlying: str
    expiry: date
    strike: float
    kind: str  # "call" | "put"

    def to_instrument_name(self) -> str:
        """Render this spec back into Thalex's canonical instrument name."""
        day = f"{self.expiry.day}"
        month = _NUM_TO_MONTH[self.expiry.month]
        year = f"{self.expiry.year % 100:02d}"
        strike = f"{int(self.strike)}" if float(self.strike).is_integer() else f"{self.strike}"
        kind_letter = "C" if self.kind == "call" else "P"
        return f"{self.underlying}-{day}{month}{year}-{strike}-{kind_letter}"


@dataclass
class OptionIntent:
    """High-level option order intent emitted by the LLM (Option B schema).

    The Thalex adapter resolves this intent against `public/instruments` and
    picks the instrument that best matches the requested tenor and strike (or
    delta target, when one is supplied)."""

    underlying: str
    kind: str  # "call" | "put"
    tenor_days: int
    target_strike: Optional[float] = None
    target_delta: Optional[float] = None  # 0..1, used when no strike is supplied


@dataclass
class RiskCaps:
    """Hard limits enforced before any options order is sent to the venue."""

    max_contracts_per_trade: float
    max_open_positions: int
    allowed_underlyings: list[str] = field(default_factory=list)


def parse_instrument_name(name: str) -> Optional[InstrumentSpec]:
    """Parse a Thalex option instrument name into a structured spec.

    Returns None for non-option instruments (perpetuals, futures, indices) or
    malformed strings rather than raising — the caller decides how to react.
    """
    if not name:
        return None
    parts = name.split("-")
    if len(parts) != 4:
        return None
    underlying, expiry_token, strike_token, kind_token = parts
    if kind_token not in {"C", "P"}:
        return None

    if len(expiry_token) < 7:
        return None
    day_str = expiry_token[:-5]
    month_str = expiry_token[-5:-2]
    year_str = expiry_token[-2:]
    if month_str not in _MONTH_TO_NUM:
        return None
    try:
        day = int(day_str)
        year = 2000 + int(year_str)
        expiry = date(year, _MONTH_TO_NUM[month_str], day)
    except (ValueError, KeyError):
        return None

    try:
        strike = float(strike_token)
    except ValueError:
        return None

    return InstrumentSpec(
        underlying=underlying,
        expiry=expiry,
        strike=strike,
        kind="call" if kind_token == "C" else "put",
    )


def _expiry_from_thalex(record: dict) -> Optional[date]:
    """Extract an expiry date from a Thalex instrument record using whatever
    field happens to be present. Falls back to the parsed instrument name."""
    ts = record.get("expiry_timestamp")
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
        except (OverflowError, OSError, ValueError):
            return None
    name = record.get("instrument_name") or ""
    spec = parse_instrument_name(name)
    return spec.expiry if spec else None


def find_best_instrument(
    instruments: Iterable[dict],
    intent: OptionIntent,
    today: Optional[date] = None,
) -> Optional[str]:
    """Pick the option instrument that best matches the intent.

    Selection rules:
    - Filter to options on the requested underlying with the requested kind.
    - Pick the expiry whose days-to-expiry is closest to ``intent.tenor_days``.
    - Within that expiry, pick the strike closest to ``intent.target_strike``.
    - Returns the instrument name, or None when nothing matches.
    """
    today = today or date.today()
    desired_kind = intent.kind.lower()
    candidates: list[tuple[int, float, str]] = []  # (tenor_distance, strike_distance, name)

    for record in instruments:
        if record.get("type") != "option":
            continue
        spec = parse_instrument_name(record.get("instrument_name") or "")
        if spec is None:
            continue
        if spec.underlying.upper() != intent.underlying.upper():
            continue
        if spec.kind != desired_kind:
            continue
        expiry = _expiry_from_thalex(record) or spec.expiry
        days_out = (expiry - today).days
        if days_out < 0:
            continue
        tenor_distance = abs(days_out - intent.tenor_days)
        if intent.target_strike is not None:
            strike_distance = abs(spec.strike - intent.target_strike)
        else:
            strike_distance = 0.0
        candidates.append((tenor_distance, strike_distance, record["instrument_name"]))

    if not candidates:
        return None

    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates[0][2]


def validate_options_order(
    underlying: str,
    contracts: float,
    open_positions_count: int,
    caps: RiskCaps,
) -> tuple[bool, str]:
    """Hard preflight check applied to every options order before submission.

    Returns ``(True, "ok")`` when the order is allowed, or ``(False, reason)``
    so the bot engine can log the rejection and skip the trade.
    """
    if contracts <= 0:
        return False, "contracts must be > 0"
    if caps.allowed_underlyings and underlying.upper() not in {u.upper() for u in caps.allowed_underlyings}:
        return False, f"underlying {underlying!r} not in allowed list {caps.allowed_underlyings}"
    if contracts > caps.max_contracts_per_trade:
        return False, (
            f"max_contracts_per_trade exceeded: {contracts} > {caps.max_contracts_per_trade}"
        )
    if open_positions_count >= caps.max_open_positions:
        return False, (
            f"max_open_positions reached: {open_positions_count}/{caps.max_open_positions}"
        )
    return True, "ok"
