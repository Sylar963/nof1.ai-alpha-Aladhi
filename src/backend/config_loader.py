"""Centralized environment variable loading for the trading agent configuration."""

import json
import os
from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    """Fetch an environment variable with optional default and required validation."""
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for {name}: {raw}") from exc


def _get_json(name: str, default: dict | None = None) -> dict | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Environment variable {name} must be a JSON object")
        return parsed
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON for {name}: {raw}") from exc


def _get_list(name: str, default: list[str] | None = None) -> list[str] | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()
    # Support JSON-style lists
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise RuntimeError(f"Environment variable {name} must be a list if using JSON syntax")
            return [str(item).strip().strip('"\'') for item in parsed if str(item).strip()]
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON list for {name}: {raw}") from exc
    # Fallback: comma separated string
    values = []
    for item in raw.split(","):
        cleaned = item.strip().strip('"\'')
        if cleaned:
            values.append(cleaned)
    return values or default


CONFIG = {
    # API keys - not required during module import (checked when bot starts)
    "taapi_api_key": _get_env("TAAPI_API_KEY"),
    "hyperliquid_private_key": _get_env("HYPERLIQUID_PRIVATE_KEY") or _get_env("LIGHTER_PRIVATE_KEY"),
    "mnemonic": _get_env("MNEMONIC"),
    # Hyperliquid network/base URL overrides
    "hyperliquid_base_url": _get_env("HYPERLIQUID_BASE_URL"),
    "hyperliquid_network": _get_env("HYPERLIQUID_NETWORK", "mainnet"),
    # Thalex (options venue)
    "thalex_network": _get_env("THALEX_NETWORK", "test"),
    "thalex_key_id": _get_env("THALEX_KEY_ID"),
    "thalex_private_key_path": _get_env("THALEX_PRIVATE_KEY_PATH"),
    "thalex_account": _get_env("THALEX_ACCOUNT"),
    "thalex_max_contracts_per_trade": float(_get_env("THALEX_MAX_CONTRACTS_PER_TRADE", "0.1") or 0.1),
    "thalex_max_open_positions": _get_int("THALEX_MAX_OPEN_POSITIONS", 3),
    "thalex_underlyings": _get_list("THALEX_UNDERLYINGS", ["BTC"]),
    # Two-cadence options scheduler. Disabled by default — set
    # OPTIONS_SCHEDULER_ENABLED=1 to turn on the 15m surface refresh +
    # 3h decision loop alongside the existing 5m perps loop.
    "options_scheduler_enabled": _get_bool("OPTIONS_SCHEDULER_ENABLED", False),
    "options_vol_surface_interval_seconds": _get_int(
        "OPTIONS_VOL_SURFACE_INTERVAL_SECONDS", 900
    ),
    "options_decision_interval_seconds": _get_int(
        "OPTIONS_DECISION_INTERVAL_SECONDS", 10800
    ),
    # BTC delta drift (signed) before the perp hedge re-trades. Threshold-only —
    # the loop polls for free, only fires a perp order when |drift| > this.
    "thalex_delta_threshold": float(_get_env("THALEX_DELTA_THRESHOLD", "0.02") or 0.02),
    # LLM via OpenRouter
    "openrouter_api_key": _get_env("OPENROUTER_API_KEY"),
    "openrouter_base_url": _get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    "openrouter_referer": _get_env("OPENROUTER_REFERER"),
    "openrouter_app_title": _get_env("OPENROUTER_APP_TITLE", "trading-agent"),
    "llm_model": _get_env("LLM_MODEL", "x-ai/grok-4"),
    # Reasoning tokens
    "reasoning_enabled": _get_bool("REASONING_ENABLED", False),
    "reasoning_effort": _get_env("REASONING_EFFORT", "high"),
    # Provider routing
    "provider_config": _get_json("PROVIDER_CONFIG"),
    "provider_quantizations": _get_list("PROVIDER_QUANTIZATIONS"),
    # Runtime controls via env
    "assets": _get_env("ASSETS"),  # e.g., "BTC ETH SOL" or "BTC,ETH,SOL"
    "interval": _get_env("INTERVAL"),  # e.g., "5m", "1h"
    "trading_mode": _get_env("TRADING_MODE", "auto"),  # manual or auto
    # API server
    "api_host": _get_env("API_HOST", "0.0.0.0"),
    "api_port": _get_env("APP_PORT") or _get_env("API_PORT") or "3000",
}
