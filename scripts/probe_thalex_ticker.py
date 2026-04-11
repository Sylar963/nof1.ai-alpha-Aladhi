"""One-shot probe: print the raw Thalex ticker response shape for an option.

Run this once against testnet to confirm where greeks live in the ticker
response and what the subscription channel format / notification payload
looks like. The output guides the field paths in
``thalex_api._extract_greeks`` and ``delta_hedge_manager._extract_delta``.

Usage:
    ./venv/bin/python scripts/probe_thalex_ticker.py BTC-10MAY26-65000-C

Reads THALEX_NETWORK / THALEX_KEY_ID / THALEX_PRIVATE_KEY_PATH from .env via
the existing config loader. Does NOT place any orders. Subscribes to the
instrument's ticker channel for ~10 seconds and prints whatever notifications
arrive, then disconnects cleanly.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend.trading.thalex_api import ThalexAPI  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("probe_thalex_ticker")


def _print_block(title: str, payload) -> None:
    print(f"\n========== {title} ==========")
    try:
        print(json.dumps(payload, indent=2, default=str))
    except (TypeError, ValueError):
        print(repr(payload))
    print("=" * (22 + len(title)))


async def probe(instrument_name: str, watch_seconds: float = 10.0) -> None:
    adapter = ThalexAPI()
    print(f"Connecting to Thalex {adapter.network_name} as {adapter.key_id}…")
    await adapter.connect()
    print("Connected.")

    # 1. One-shot ticker RPC — shows the response shape get_greeks parses.
    raw = await adapter._request(adapter._client.ticker, instrument_name=instrument_name)
    _print_block(f"public/ticker {instrument_name}", raw)

    parsed_greeks = await adapter.get_greeks(instrument_name)
    _print_block("get_greeks parser output", parsed_greeks)

    if not parsed_greeks:
        print(
            "\n[WARNING] get_greeks returned an empty dict. The defensive parser "
            "did not find delta in any known field path. Add the correct path "
            "to thalex_api._GREEKS_FIELD_PATHS and re-run."
        )

    # 2. Subscribe to ticker channel for a short window — shows notification shape.
    received: list = []

    async def _on_ticker(payload):
        received.append(payload)
        _print_block(f"subscription notification #{len(received)}", payload)

    await adapter.subscribe_ticker(instrument_name, _on_ticker)
    print(f"\nSubscribed. Listening for {watch_seconds:.0f}s of ticker pushes…")
    await asyncio.sleep(watch_seconds)
    await adapter.unsubscribe_ticker(instrument_name)

    if not received:
        print(
            "\n[WARNING] No subscription notifications arrived. Either the "
            "instrument is illiquid right now, or the channel string format "
            "in subscribe_ticker is wrong. Compare against any docs you have."
        )
    else:
        print(f"\n[OK] Received {len(received)} ticker pushes during the window.")

    await adapter.disconnect()
    print("\nDisconnected. Probe complete.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/probe_thalex_ticker.py <instrument_name>")
        print("Example: python scripts/probe_thalex_ticker.py BTC-10MAY26-65000-C")
        sys.exit(2)
    instrument_name = sys.argv[1]
    watch_seconds = float(os.environ.get("PROBE_WATCH_SEC") or 10.0)
    asyncio.run(probe(instrument_name, watch_seconds))


if __name__ == "__main__":
    main()
