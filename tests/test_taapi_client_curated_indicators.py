"""Tests for the curated TAAPI indicator bundle used by the perps loop."""

from src.backend.indicators.taapi_client import TAAPIClient


def test_fetch_bulk_indicators_forwards_endpoint_specific_params(monkeypatch):
    captured = {}

    def _fake_post(url, payload, retries=3, backoff=0.5):
        captured["payload"] = payload
        return {"data": []}

    client = TAAPIClient(enable_cache=False)
    monkeypatch.setattr(client, "_post_with_retry", _fake_post)

    client.fetch_bulk_indicators(
        "BTC/USDT",
        "5m",
        [
            {
                "id": "keltner",
                "indicator": "keltnerchannels",
                "period": 130,
                "atrLength": 130,
                "multiplier": 4,
                "results": 5,
            },
        ],
    )

    indicators = captured["payload"]["construct"]["indicators"]
    assert indicators[0]["atrLength"] == 130
    assert indicators[0]["multiplier"] == 4


def test_fetch_asset_indicators_returns_curated_bundle(monkeypatch):
    from src.backend import config_loader

    monkeypatch.setitem(config_loader.CONFIG, "interval", "4h")
    client = TAAPIClient(enable_cache=False)
    monkeypatch.setattr("src.backend.indicators.taapi_client.time.sleep", lambda _: None)
    pauses = []

    def _fake_bulk(symbol, interval, indicators_config):
        if interval == "5m":
            return {
                "sma99": {"value": [60000.0, 60010.0, 60020.0]},
                "keltner": {
                    "lower": [59500.0, 59510.0, 59520.0],
                    "middle": [60000.0, 60010.0, 60020.0],
                    "upper": [60500.0, 60510.0, 60520.0],
                },
            }
        return {
            "sma99": {"value": [59000.0, 59100.0, 59200.0]},
            "keltner": {
                "lower": [56000.0, 56100.0, 56200.0],
                "middle": [59000.0, 59100.0, 59200.0],
                "upper": [62000.0, 62100.0, 62200.0],
            },
        }

    monkeypatch.setattr(client, "fetch_bulk_indicators", _fake_bulk)
    monkeypatch.setattr(
        client,
        "fetch_candles",
        lambda symbol, interval, params=None: (
            [
                {"high": 60100.0, "low": 59800.0},
                {"high": 60200.0, "low": 59750.0},
            ]
            if interval == "5m"
            else [
                {"high": 59000.0, "low": 58000.0, "close": 58500.0, "volume": 10.0},
                {"high": 60000.0, "low": 59000.0, "close": 59500.0, "volume": 30.0},
            ]
        ),
    )

    result = client.fetch_asset_indicators(
        "BTC",
        current_spot=60050.0,
        request_pause=lambda: pauses.append("wait"),
    )

    assert result["5m"]["sma99"][-1] == 60020.0
    assert result["5m"]["avwap"] == 59250.0
    assert result["5m"]["keltner"]["upper"][-1] == 60520.0
    assert result["5m"]["opening_range"] == {
        "high": 60200.0,
        "low": 59750.0,
        "position": "inside",
    }
    assert result["4h"]["sma99"][-1] == 59200.0
    assert result["4h"]["avwap"] == 59250.0
    assert pauses == ["wait", "wait", "wait"]
