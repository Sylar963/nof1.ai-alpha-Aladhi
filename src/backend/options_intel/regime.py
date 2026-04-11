"""Vol regime classifier — combines two signals into one label.

Signal 1 (your heuristic): the **15-day ATM straddle expected-move test**.
Look back at the straddle anchor we persisted ~15 days ago and ask: where is
spot now relative to that historical implied range?

    spot < lower_strike_15d_ago → 'cheap'
    spot > upper_strike_15d_ago → 'rich'
    inside the range            → 'fair'

(Per user direction. The standard "vol cheap if realized exceeded implied"
read is the *opposite* — we'll verify on first run with live data and flip
the labels if it turns out the standard read is correct.)

Signal 2 (standard): **realized vs implied vol ratio**.
Compute close-to-close annualized realized vol over the last 15 days and
compare to current ATM-15d implied vol:

    RV / IV > 1.2 → 'cheap'   (real moves outran implied)
    RV / IV < 0.8 → 'rich'    (implied was bigger than realized)
    in between    → 'fair'

Combined label rule:
    - both signals agree   → use that label, confidence='high'
    - signals disagree     → use Signal 2, confidence='low'
    - either input missing → 'unknown'

The combined reading exposes both raw inputs so the LLM (or you) can
sanity-check after the first few cycles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.backend.options_intel.iv_history_store import IVHistoryRow, IVHistoryStore


_RV_IV_RICH_CEILING = 0.8
_RV_IV_CHEAP_FLOOR = 1.2


@dataclass
class RegimeReading:
    """Combined regime classification with both raw signals attached."""

    vol_regime: str  # 'rich' | 'cheap' | 'fair' | 'unknown'
    confidence: str  # 'high' | 'low' | 'unknown'

    # Signal 1 raw
    signal_1_label: str  # 'rich' | 'cheap' | 'fair' | 'unknown'
    historical_anchor: Optional[IVHistoryRow]
    current_spot: float

    # Signal 2 raw
    signal_2_label: str  # 'rich' | 'cheap' | 'fair' | 'unknown'
    realized_vol_15d: float
    implied_vol_15d: float
    rv_iv_ratio: float

    def to_dict(self) -> dict:
        return {
            "vol_regime": self.vol_regime,
            "confidence": self.confidence,
            "signal_1": {
                "label": self.signal_1_label,
                "current_spot": self.current_spot,
                "lower_15d_ago": (
                    self.historical_anchor.lower_strike if self.historical_anchor else None
                ),
                "upper_15d_ago": (
                    self.historical_anchor.upper_strike if self.historical_anchor else None
                ),
            },
            "signal_2": {
                "label": self.signal_2_label,
                "rv_15d": self.realized_vol_15d,
                "iv_15d": self.implied_vol_15d,
                "rv_iv_ratio": self.rv_iv_ratio,
            },
        }


# ---------------------------------------------------------------------------
# Realized volatility helper
# ---------------------------------------------------------------------------


def realized_vol_close_to_close(
    spot_history: list[float],
    periods_per_year: int = 365,
) -> float:
    """Annualized close-to-close realized volatility from a price series.

    Standard formula:
        rv = std(log returns) * sqrt(periods_per_year)
    """
    if not spot_history or len(spot_history) < 2:
        return 0.0
    returns = []
    for i in range(1, len(spot_history)):
        prev = spot_history[i - 1]
        curr = spot_history[i]
        if prev <= 0 or curr <= 0:
            continue
        returns.append(math.log(curr / prev))
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _signal_1(anchor: Optional[IVHistoryRow], current_spot: float) -> str:
    """The user's straddle EM test."""
    if anchor is None:
        return "unknown"
    if current_spot < anchor.lower_strike:
        return "cheap"
    if current_spot > anchor.upper_strike:
        return "rich"
    return "fair"


def _signal_2(rv: float, iv: float) -> tuple[str, float]:
    """RV/IV ratio test. Returns (label, ratio)."""
    if iv <= 0 or math.isnan(rv) or math.isnan(iv):
        return "unknown", 0.0
    ratio = rv / iv
    if ratio > _RV_IV_CHEAP_FLOOR:
        return "cheap", ratio
    if ratio < _RV_IV_RICH_CEILING:
        return "rich", ratio
    return "fair", ratio


def classify_regime(
    store: IVHistoryStore,
    current_spot: float,
    current_atm_iv_15d: float,
    spot_history: list[float],
    now: Optional[datetime] = None,
    target_age_days: float = 15.0,
    tenor_days: int = 15,
) -> RegimeReading:
    """Combine the two signals into a single regime label.

    Args:
        store: persistent IV history store with anchors written by
            ``builder.refresh_vol_surface``.
        current_spot: spot price right now.
        current_atm_iv_15d: ATM 15-day implied vol from the latest surface.
        spot_history: ~16 daily closes (most recent last) for the RV calc.
        now: defaults to ``datetime.now(UTC)``.
        target_age_days: how far back to compare against (15 by default).
        tenor_days: which persisted tenor anchor to look up.
    """
    now = now or datetime.now(timezone.utc)

    anchor = store.lookback(tenor_days=tenor_days, target_age_days=target_age_days, now=now)
    s1_label = _signal_1(anchor, current_spot)

    rv = realized_vol_close_to_close(spot_history)
    s2_label, ratio = _signal_2(rv, current_atm_iv_15d)

    # Combined logic
    if s1_label == "unknown" or s2_label == "unknown":
        combined = "unknown"
        confidence = "unknown"
    elif s1_label == s2_label:
        combined = s1_label
        confidence = "high"
    else:
        combined = s2_label  # standard signal wins on disagreement
        confidence = "low"

    return RegimeReading(
        vol_regime=combined,
        confidence=confidence,
        signal_1_label=s1_label,
        historical_anchor=anchor,
        current_spot=current_spot,
        signal_2_label=s2_label,
        realized_vol_15d=rv,
        implied_vol_15d=current_atm_iv_15d,
        rv_iv_ratio=ratio,
    )
