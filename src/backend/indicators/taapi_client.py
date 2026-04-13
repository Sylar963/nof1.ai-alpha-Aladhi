"""Client helper for interacting with the TAAPI technical analysis API."""

import requests
import os
import time
import logging
from datetime import datetime, timedelta, timezone
from src.backend.config_loader import CONFIG
from src.backend.indicators.taapi_cache import get_cache


AVWAP_ANCHOR_UTC = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


class TAAPIClient:
    """Fetches TA indicators with retry/backoff semantics for resilience."""

    def __init__(self, enable_cache: bool = True, cache_ttl: int = 60):
        """
        Initialize TAAPI credentials and base URL.
        
        Args:
            enable_cache: Enable caching to reduce API calls
            cache_ttl: Cache time-to-live in seconds (default: 60s)
        """
        self.api_key = CONFIG["taapi_api_key"]
        self.base_url = "https://api.taapi.io/"
        self.bulk_url = "https://api.taapi.io/bulk"
        self.enable_cache = enable_cache
        self.cache = get_cache(ttl=cache_ttl) if enable_cache else None

    def _get_with_retry(self, url, params, retries=3, backoff=0.5):
        """Perform a GET request with exponential backoff retry logic."""
        for attempt in range(retries):
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                # Retry on rate limit (429) or server errors (500+)
                if (e.response.status_code == 429 or e.response.status_code >= 500) and attempt < retries - 1:
                    wait = backoff * (2 ** attempt)
                    if e.response.status_code == 429:
                        logging.warning(f"TAAPI rate limit (429) hit, retrying in {wait}s (attempt {attempt + 1}/{retries})")
                    else:
                        logging.warning(f"TAAPI {e.response.status_code}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
            except requests.Timeout as e:
                if attempt < retries - 1:
                    wait = backoff * (2 ** attempt)
                    logging.warning(f"TAAPI timeout, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")

    def _post_with_retry(self, url, payload, retries=3, backoff=0.5):
        """Perform a POST request with exponential backoff retry logic."""
        for attempt in range(retries):
            try:
                resp = requests.post(url, json=payload, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                # Retry on rate limit (429) or server errors (500+)
                if (e.response.status_code == 429 or e.response.status_code >= 500) and attempt < retries - 1:
                    wait = backoff * (2 ** attempt)
                    if e.response.status_code == 429:
                        logging.warning(f"TAAPI Bulk rate limit (429), retrying in {wait}s (attempt {attempt + 1}/{retries})")
                    else:
                        logging.warning(f"TAAPI Bulk {e.response.status_code}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
            except requests.Timeout as e:
                if attempt < retries - 1:
                    wait = backoff * (2 ** attempt)
                    logging.warning(f"TAAPI Bulk timeout, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")

    def fetch_bulk_indicators(self, symbol, interval, indicators_config):
        """
        Fetch multiple indicators in one bulk request to TAAPI.

        Args:
            symbol: Market pair (e.g., "BTC/USDT")
            interval: Timeframe (e.g., "5m", "4h")
            indicators_config: List of indicator configurations
                Example: [
                    {"id": "ema20", "indicator": "ema", "period": 20, "results": 10},
                    {"id": "macd", "indicator": "macd", "results": 10},
                    {"id": "rsi14", "indicator": "rsi", "period": 14, "results": 10}
                ]

        Returns:
            Dict mapping indicator IDs to their results
            Example: {"ema20": [values...], "macd": {...}, "rsi14": [values...]}
        """
        try:
            # Build bulk request payload
            indicators = []
            for config in indicators_config:
                indicator_def = {
                    "id": config.get("id", config["indicator"]),
                    "indicator": config["indicator"]
                }

                # Forward endpoint-specific parameters (period, results,
                # anchorPeriod, atrLength, multiplier, etc.) transparently.
                for key, value in config.items():
                    if key not in {"id", "indicator"}:
                        indicator_def[key] = value

                indicators.append(indicator_def)

            payload = {
                "secret": self.api_key,
                "construct": {
                    "exchange": "binance",
                    "symbol": symbol,
                    "interval": interval,
                    "indicators": indicators
                }
            }

            # Make bulk POST request
            response = self._post_with_retry(self.bulk_url, payload)

            # Parse results by ID
            results = {}
            if isinstance(response, dict) and "data" in response:
                for item in response["data"]:
                    indicator_id = item.get("id")
                    if indicator_id:
                        results[indicator_id] = item.get("result")

            return results

        except Exception as e:
            logging.error(f"TAAPI bulk fetch exception for {symbol} {interval}: {e}")
            return {}

    def fetch_candles(self, symbol: str, interval: str, params: dict | None = None) -> list:
        """Fetch raw candle objects from TAAPI's ``candles`` endpoint."""
        base_params = {
            "secret": self.api_key,
            "exchange": "binance",
            "symbol": symbol,
            "interval": interval,
        }
        if params:
            base_params.update(params)
        try:
            data = self._get_with_retry(f"{self.base_url}candles", base_params)
            return data if isinstance(data, list) else []
        except Exception as e:
            logging.error(f"TAAPI candles fetch exception for {symbol} {interval}: {e}")
            return []

    def _extract_keltner_series(self, data):
        """Normalize TAAPI Keltner responses into parallel float arrays."""
        if not isinstance(data, dict):
            return {"lower": [], "middle": [], "upper": []}

        def _as_list(value):
            if isinstance(value, list):
                return [round(v, 4) if isinstance(v, (int, float)) else v for v in value]
            if isinstance(value, (int, float)):
                return [round(value, 4)]
            return []

        return {
            "lower": _as_list(data.get("lower")),
            "middle": _as_list(data.get("middle")),
            "upper": _as_list(data.get("upper")),
        }

    def _build_opening_range(self, candles: list, current_spot=None) -> dict:
        """Compute the first-hour opening range from 5m candle highs/lows."""
        highs = []
        lows = []
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            high = candle.get("high")
            low = candle.get("low")
            try:
                highs.append(float(high))
                lows.append(float(low))
            except (TypeError, ValueError):
                continue

        if not highs or not lows:
            return {"high": None, "low": None, "position": "unknown"}

        high = max(highs)
        low = min(lows)
        position = "unknown"
        if current_spot is not None:
            try:
                spot = float(current_spot)
                if spot > high:
                    position = "above"
                elif spot < low:
                    position = "below"
                else:
                    position = "inside"
            except (TypeError, ValueError):
                position = "unknown"
        return {"high": round(high, 4), "low": round(low, 4), "position": position}

    def _compute_avwap(self, candles: list) -> float | None:
        """Compute anchored VWAP from OHLCV candles."""
        notional = 0.0
        volume_total = 0.0
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            try:
                high = float(candle.get("high"))
                low = float(candle.get("low"))
                close = float(candle.get("close"))
                volume = float(candle.get("volume"))
            except (TypeError, ValueError):
                continue
            if volume <= 0:
                continue
            typical_price = (high + low + close) / 3.0
            notional += typical_price * volume
            volume_total += volume

        if volume_total <= 0:
            return None
        return round(notional / volume_total, 4)

    def _pause_for_rate_limit(self, request_pause=None) -> None:
        """Let the caller enforce TAAPI pacing without blocking the event loop."""
        if request_pause is None:
            return
        logging.info("Waiting 15s for TAAPI rate limit (Free plan: 1 req/15s)...")
        request_pause()

    def fetch_asset_indicators(self, asset, current_spot=None, request_pause=None):
        """
        Fetch the curated perps indicator set for an asset.

        Intraday (5m): SMA99, Keltner(130, atrLength=130, multiplier=4),
        anchored VWAP from 2026-01-01 00:00 UTC, and first-hour opening range
        from 5m candles.

        Higher timeframe (configured interval): SMA99, Keltner(130, 130, 4),
        and the same anchored VWAP.
        
        IMPORTANT: Free plan has 1 request per 15 seconds limit.
        The caller may inject a pause callback to pace each TAAPI HTTP call.
        
        Caching: Results are cached for 60 seconds by default to reduce API calls.

        Args:
            asset: Asset ticker (e.g., "BTC", "ETH")

        Returns:
            Dict with structure:
            {
                "5m": {
                    "sma99": [...],
                    "avwap": 12345.67,
                    "keltner": {"lower": [...], "middle": [...], "upper": [...]},
                    "opening_range": {"high": ..., "low": ..., "position": ...},
                },
                "4h": {
                    "sma99": [...],
                    "avwap": 12345.67,
                    "keltner": {"lower": [...], "middle": [...], "upper": [...]},
                },
            }
        """
        # Check cache first (for current interval from config)
        interval = CONFIG.get("interval", "4h")
        
        # Try to get cached data for both intervals
        if self.enable_cache and self.cache:
            cached_5m = self.cache.get(asset, "5m")
            cached_interval = self.cache.get(asset, interval)
            
            if cached_5m and cached_interval:
                logging.info(f"Using cached indicators for {asset} (5m + {interval})")
                return {"5m": cached_5m, interval: cached_interval}
        
        symbol = f"{asset}/USDT"
        result = {"5m": {}, interval: {}}

        # Bulk request for 5m indicators
        indicators_5m = [
            {"id": "sma99", "indicator": "sma", "period": 99, "results": 5},
            {
                "id": "keltner",
                "indicator": "keltnerchannels",
                "period": 130,
                "atrLength": 130,
                "multiplier": 4,
                "results": 5,
            },
        ]

        bulk_5m = self.fetch_bulk_indicators(symbol, "5m", indicators_5m)

        # Extract series data from bulk response
        result["5m"]["sma99"] = self._extract_series(bulk_5m.get("sma99"), "value")
        result["5m"]["keltner"] = self._extract_keltner_series(bulk_5m.get("keltner"))

        now = datetime.now(timezone.utc)
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        end = start + timedelta(hours=1)
        self._pause_for_rate_limit(request_pause)
        opening_range_candles = self.fetch_candles(
            symbol,
            "5m",
            params={
                "fromTimestamp": int(start.timestamp()),
                "toTimestamp": int(end.timestamp()),
            },
        )
        result["5m"]["opening_range"] = self._build_opening_range(opening_range_candles, current_spot=current_spot)

        self._pause_for_rate_limit(request_pause)
        avwap_candles = self.fetch_candles(
            symbol,
            "1d",
            params={
                "fromTimestamp": int(AVWAP_ANCHOR_UTC.timestamp()),
                "toTimestamp": int(now.timestamp()),
                "period": 300,
            },
        )
        avwap = self._compute_avwap(avwap_candles)
        result["5m"]["avwap"] = avwap

        indicators_long = [
            {"id": "sma99", "indicator": "sma", "period": 99, "results": 5},
            {
                "id": "keltner",
                "indicator": "keltnerchannels",
                "period": 130,
                "atrLength": 130,
                "multiplier": 4,
                "results": 5,
            },
        ]

        self._pause_for_rate_limit(request_pause)
        bulk_long = self.fetch_bulk_indicators(symbol, interval, indicators_long)

        # Extract values and series
        result[interval]["sma99"] = self._extract_series(bulk_long.get("sma99"), "value")
        result[interval]["keltner"] = self._extract_keltner_series(bulk_long.get("keltner"))
        result[interval]["avwap"] = avwap

        # Cache the results
        if self.enable_cache and self.cache:
            self.cache.set(asset, "5m", result["5m"])
            self.cache.set(asset, interval, result[interval])
            logging.info(f"Cached indicators for {asset} (5m + {interval})")

        return result

    def _extract_series(self, data, value_key="value"):
        """Extract and normalize series data from TAAPI response."""
        if not data:
            return []
        if isinstance(data, dict) and value_key in data:
            values = data[value_key]
            if isinstance(values, list):
                return [round(v, 4) if isinstance(v, (int, float)) else v for v in values]
        return []

    def _extract_value(self, data, value_key="value"):
        """Extract and normalize single value from TAAPI response."""
        if not data:
            return None
        if isinstance(data, dict) and value_key in data:
            val = data[value_key]
            return round(val, 4) if isinstance(val, (int, float)) else val
        return None

    def get_indicators(self, asset, interval):
        """Return a curated bundle of intraday indicators for ``asset``."""
        params = {
            "secret": self.api_key,
            "exchange": "binance",
            "symbol": f"{asset}/USDT",
            "interval": interval
        }
        rsi_response = self._get_with_retry(f"{self.base_url}rsi", params)
        macd_response = self._get_with_retry(f"{self.base_url}macd", params)
        sma_response = self._get_with_retry(f"{self.base_url}sma", params)
        ema_response = self._get_with_retry(f"{self.base_url}ema", params)
        bbands_response = self._get_with_retry(f"{self.base_url}bbands", params)
        return {
            "rsi": rsi_response.get("value"),
            "macd": macd_response,
            "sma": sma_response.get("value"),
            "ema": ema_response.get("value"),
            "bbands": bbands_response
        }

    def get_historical_indicator(self, indicator, symbol, interval, results=10, params=None):
        """Fetch historical indicator data with optional overrides."""
        base_params = {
            "secret": self.api_key,
            "exchange": "binance",
            "symbol": symbol,
            "interval": interval,
            "results": results
        }
        if params:
            base_params.update(params)
        response = self._get_with_retry(f"{self.base_url}{indicator}", base_params)
        return response

    def fetch_series(self, indicator: str, symbol: str, interval: str, results: int = 10, params: dict | None = None, value_key: str = "value") -> list:
        """Fetch and normalize a historical indicator series.

        Args:
            indicator: TAAPI indicator slug (e.g. ``"ema"``).
            symbol: Market pair identifier (e.g. ``"BTC/USDT"``).
            interval: Candle interval requested from TAAPI.
            results: Number of datapoints to request.
            params: Additional TAAPI query parameters.
            value_key: Key to extract from the TAAPI response payload.

        Returns:
            List of floats rounded to 4 decimals, or an empty list on error.
        """
        try:
            data = self.get_historical_indicator(indicator, symbol, interval, results=results, params=params)
            if isinstance(data, dict):
                # Simple indicators: {"value": [1,2,3]}
                if value_key in data and isinstance(data[value_key], list):
                    return [round(v, 4) if isinstance(v, (int, float)) else v for v in data[value_key]]
                # Error response
                if "error" in data:
                    import logging
                    logging.error(f"TAAPI error for {indicator} {symbol} {interval}: {data.get('error')}")
                    return []
            return []
        except Exception as e:
            import logging
            logging.error(f"TAAPI fetch_series exception for {indicator}: {e}")
            return []

    def fetch_value(self, indicator: str, symbol: str, interval: str, params: dict | None = None, key: str = "value"):
        """Fetch a single indicator value for the latest candle."""
        try:
            base_params = {
                "secret": self.api_key,
                "exchange": "binance",
                "symbol": symbol,
                "interval": interval
            }
            if params:
                base_params.update(params)
            data = self._get_with_retry(f"{self.base_url}{indicator}", base_params)
            if isinstance(data, dict):
                val = data.get(key)
                return round(val, 4) if isinstance(val, (int, float)) else val
            return None
        except Exception:
            return None
