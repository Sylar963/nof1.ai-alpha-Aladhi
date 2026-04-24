"""Normalize Deribit ``book_summary_by_currency`` records into the shape
consumed by :func:`build_vol_surface`.

Deribit's book-summary response carries ``mark_iv`` (percent), ``mark_price``
(in BTC, not USD), and ``underlying_price``, but NOT ``strike`` or
``option_type`` — those live on a separate ``get_instruments`` call. We
recover them by parsing the instrument name (e.g. ``BTC-25DEC26-65000-P``
→ strike=65000, put) via the existing :func:`parse_instrument_name`
helper.

The surface parser expects:
  - ``type=option``
  - ``option_type`` in {call, put}
  - ``strike`` OR ``strike_price`` > 0
  - ``iv`` OR ``mark_iv`` > 0 (decimal; normalizer divides percent by 100)
  - ``expiration_timestamp`` (either seconds or ms is fine — the surface
    parser auto-detects)
  - ``mark_price`` (BTC; we keep it as published)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.backend.trading.options import parse_instrument_name


def normalize_deribit_chain(chain: list[dict]) -> list[dict]:
    """Return a new list shaped like a Thalex chain so ``build_vol_surface``
    can consume Deribit book-summary records unchanged.

    Records that can't be parsed (missing instrument_name, unparseable
    strike/expiry) are dropped silently — no warnings, since the primary
    signal we care about here is the successful subset.
    """
    out: list[dict] = []
    if not chain:
        return out
    for record in chain:
        if not isinstance(record, dict):
            continue
        name = record.get("instrument_name")
        if not isinstance(name, str) or not name:
            continue
        spec = parse_instrument_name(name)
        if spec is None:
            continue

        mark_iv = record.get("mark_iv")
        if mark_iv is None or not isinstance(mark_iv, (int, float)):
            continue
        iv = float(mark_iv)
        if iv <= 0:
            continue

        expiry_dt = datetime(
            spec.expiry.year, spec.expiry.month, spec.expiry.day,
            tzinfo=timezone.utc,
        )
        expiry_seconds = int(expiry_dt.timestamp())

        normalized = {
            "instrument_name": name,
            "type": "option",
            "option_type": spec.kind,
            "strike": float(spec.strike),
            "strike_price": float(spec.strike),
            "iv": iv,
            "mark_iv": iv,
            "mark_price": _safe_float(record.get("mark_price")),
            "expiry_timestamp": expiry_seconds,
            "expiration_timestamp": expiry_seconds,
            "underlying_price": _safe_float(record.get("underlying_price")),
            "source": "deribit",
        }
        out.append(normalized)
    return out


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
