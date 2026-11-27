import json
import struct
from pathlib import Path
from xml.etree import ElementTree

import pytest

from src.backend.market_data.protocol import BinanceDepthSequence, ProtocolError, SequenceGap, decode_frame


MARKET = "BTC-USD-PERP"
NATIVE_US = 1_700_123_456_789_012
NATIVE_MS = 1_700_123_456_789
NULL_MANTISSA = -(2**63)
SPOT_BBO = {"u": 400900217, "s": "BTCUSDT", "b": "25.35190000", "B": "31.21000000",
            "a": "25.36520000", "A": "40.66000000"}
SPOT_DEPTH = {"e": "depthUpdate", "E": NATIVE_MS, "s": "BTCUSDT", "U": 157, "u": 160,
              "b": [["95000.00100000", "0.00000000"]], "a": []}
AGG_TRADE = {"e": "aggTrade", "E": NATIVE_MS, "s": "BTCUSDT", "a": 12345,
             "p": "95000.00010000", "q": "0.00001000", "f": 100, "l": 105,
             "T": NATIVE_MS - 3, "m": True}
MARK_PRICE = {"e": "markPriceUpdate", "E": NATIVE_MS, "s": "BTCUSDT", "p": "95010.12345678",
              "i": "95000.00000000", "P": "95000.00000000", "r": "-0.00001234",
              "T": NATIVE_MS + 3600000}
LIQUIDATION = {"e": "forceOrder", "E": NATIVE_MS,
               "o": {"s": "BTCUSDT", "S": "SELL", "o": "LIMIT", "f": "IOC", "q": "1.01",
                     "p": "95000.00", "ap": "95001.01", "X": "FILLED", "l": "1.01",
                     "z": "1.01", "T": NATIVE_MS - 6}}


def _combined(suffix: str, payload: dict, symbol: str = "btcusdt") -> str:
    return json.dumps({"stream": f"{symbol}@{suffix}", "data": payload})


def _bbo_frame(market: bytes = b"BTC-USD-PERP", *, padding: bytes = b"") -> bytes:
    fixed = struct.pack("<qqqqqq", NATIVE_US, 372, 9_223_372_036_854_775_807, 10_000_001,
                        NULL_MANTISSA, NULL_MANTISSA)
    return struct.pack("<HHHH", len(fixed) + len(padding), 2, 1, 1) + fixed + padding + bytes([len(market)]) + market


def _book_frame(*, package: int = 0, entry_length: int = 18, price: int = 6_789_012_345_001) -> bytes:
    fixed = struct.pack("<qqB9q", NATIVE_US, 9003, package, *(NULL_MANTISSA,) * 5,
                        6_789_012_345_001, 123_000_000, 6_789_012_345_002, 1)
    bids = struct.pack("<HHqq", entry_length, 1, price, 100) + b"\xff\xff"
    asks = struct.pack("<HHqq", 16, 1, 6_789_012_345_002, 0)
    return struct.pack("<HHHH", 92, 3, 1, 1) + fixed + b"\xab\xcd\xef" + bids + asks + b"\x0cBTC-USD-PERP"


def test_paradex_bbo_preserves_exact_price_nulls_and_native_timestamp():
    assert decode_frame("paradex", _bbo_frame(padding=b"\x00\x08")) == {
        "kind": "bbo", "channel": "bbo.BTC-USD-PERP", "market": MARKET, "encoding": "sbe",
        "source_event_time_ns": 1_700_123_456_789_012_000, "source_transaction_time_ns": None,
        "source_sequence": 372, "source_time_basis": "unix_us",
        "payload": {"market": MARKET, "ts_us": NATIVE_US, "seq_no": 372,
                    "bid_price": "92233720368.54775807", "bid_size": "0.10000001",
                    "ask_price": None, "ask_size": None},
    }


def test_paradex_depth_uses_wire_block_sizes_for_groups_and_preserves_book_fields():
    record = decode_frame("paradex", _book_frame())
    assert record["kind"] == "depth"
    assert record["channel"] == "order_book.BTC-USD-PERP"
    assert record["source_event_time_ns"] == NATIVE_US * 1_000
    assert record["source_sequence"] == 9003
    assert record["payload"] == {
        "market": MARKET, "ts_us": NATIVE_US, "seq_no": 9003, "pkg_type": "SNAPSHOT",
        "best_bid_price": None, "best_bid_size": None, "best_ask_price": None,
        "best_ask_size": None, "relative_spread": None,
        "best_bid_api_price": "67890.12345001", "best_bid_api_size": "1.23000000",
        "best_ask_api_price": "67890.12345002", "best_ask_api_size": "0.00000001",
        "bids": [["67890.12345001", "0.00000100"]], "asks": [["67890.12345002", "0.00000000"]],
    }
    assert decode_frame("paradex", _book_frame(package=1))["payload"]["pkg_type"] == "DELTA"


def test_paradex_heartbeat_retains_native_metadata():
    raw = struct.pack("<HHHHqq", 16, 40, 1, 1, NATIVE_US, 432)
    record = decode_frame("paradex", raw)
    assert (record["kind"], record["channel"], record["encoding"], record["market"]) == (
        "control", "heartbeat", "sbe", None,
    )
    assert record["payload"] == {"ts_us": NATIVE_US, "seq_no": 432}
    assert record["source_event_time_ns"] == NATIVE_US * 1_000
    assert record["source_sequence"] == 432


def test_paradex_trade_decodes_matched_price_and_matching_engine_time_without_fabricating_full_id():
    fixed = struct.pack('<qqqBqqqB', NATIVE_US, 123, -4455, 1, 6012340000000, 1250000, NATIVE_US - 100, 5)
    raw = struct.pack('<HHHH', len(fixed), 1, 1, 1) + fixed + bytes([len(MARKET)]) + MARKET.encode()
    record = decode_frame('paradex', raw)
    assert record['kind'] == 'trade'
    assert record['channel'] == 'trades.BTC-USD-PERP'
    assert record['payload']['price'] == '60123.40000000'
    assert record['payload']['size'] == '0.01250000'
    assert record['payload']['trade_type'] == 'RPI'
    assert record['payload']['trade_id_low64'] == '-4455'
    assert record['source_transaction_time_ns'] == (NATIVE_US - 100) * 1000


@pytest.mark.parametrize("raw", [
    _bbo_frame()[:6],
    struct.pack("<HHHH", 47, 2, 1, 1) + _bbo_frame()[8:],
    struct.pack("<HHHH", 500, 2, 1, 1) + _bbo_frame()[8:],
    struct.pack("<HHHH", 48, 2, 1, 3) + _bbo_frame()[8:],
    struct.pack("<HHHH", 48, 2, 2, 1) + _bbo_frame()[8:],
    struct.pack("<HHHH", 48, 1, 1, 1) + _bbo_frame()[8:],
    _bbo_frame(b""),
    _bbo_frame(b"\xff"),
    _bbo_frame()[:-1],
    _bbo_frame() + b"\x00",
    _book_frame()[:102],
    _book_frame()[:117],
    _book_frame(entry_length=4),
    _book_frame(entry_length=255),
    _book_frame(price=NULL_MANTISSA),
    _book_frame(package=2),
    struct.pack("<HHHH", 16, 40, 1, 1) + b"\x00",
])
def test_paradex_rejects_incompatible_and_malformed_frames(raw):
    with pytest.raises(ProtocolError):
        decode_frame("paradex", raw)


def test_paradex_fixture_layouts_match_archived_upstream_schema():
    schema_path = Path(__file__).resolve().parents[1] / "src/backend/market_data/schemas/paradex_1_0.xml"
    schema = ElementTree.parse(schema_path).getroot()
    assert (schema.attrib["id"], schema.attrib["version"], schema.attrib["byteOrder"]) == ("1", "3", "littleEndian")
    types = {value.attrib["name"]: value for value in schema.find("types")}
    assert [item.attrib["primitiveType"] for item in types["messageHeader"]] == ["uint16"] * 4
    assert [item.attrib["primitiveType"] for item in types["groupSizeEncoding"]] == ["uint16"] * 2
    assert types["Timestamp"].attrib["primitiveType"] == "int64"
    for type_name in ("Price8", "Price8NULL", "Qty8", "Qty8NULL", "Rate8NULL"):
        mantissa, exponent = types[type_name]
        assert mantissa.attrib["primitiveType"] == "int64"
        assert int(mantissa.attrib["nullValue"]) == NULL_MANTISSA
        assert exponent.attrib["presence"] == "constant" and exponent.text == "-8"
    assert types["varString8"][0].attrib["primitiveType"] == "uint8"
    assert types["varString8"][1].attrib["characterEncoding"] == "UTF-8"
    assert {item.text for item in types["PkgType"]} == {"0", "1"}
    messages = {node.attrib["id"]: node for node in schema if node.tag.endswith("}message")}
    bbo, book, heartbeat = (messages[template] for template in ("2", "3", "40"))
    assert [node.attrib["name"] for node in bbo] == [
        "ts", "seq", "bidPrice", "bidSize", "askPrice", "askSize", "market",
    ]
    assert [node.attrib["name"] for node in book] == [
        "ts", "seq", "pkgType", "bestBidPrice", "bestBidSize", "bestAskPrice", "bestAskSize",
        "relativeSpread", "bestBidApiPrice", "bestBidApiSize", "bestAskApiPrice", "bestAskApiSize",
        "bids", "asks", "market",
    ]
    assert [node.attrib["type"] for node in book.findall("field")] == [
        "Timestamp", "int64", "PkgType", "Price8NULL", "Qty8NULL", "Price8NULL", "Qty8NULL",
        "Rate8NULL", "Price8NULL", "Qty8NULL", "Price8NULL", "Qty8NULL",
    ]
    assert [node.attrib["name"] for node in heartbeat] == ["ts", "seq"]
    assert all(int(node.get("sinceVersion", 0)) <= 1 for message in (bbo, book, heartbeat) for node in message.iter())


def test_binance_spot_bbo_has_no_fabricated_timestamp_or_consecutive_sequence():
    raw = _combined("bookTicker", SPOT_BBO)
    record = decode_frame("binance_spot", raw)
    assert record == {
        "kind": "bbo", "channel": "btcusdt@bookTicker", "market": "BTCUSDT", "encoding": "json",
        "source_event_time_ns": None, "source_transaction_time_ns": None, "source_sequence": 400900217,
        "source_time_basis": None, "payload": SPOT_BBO,
    }
    payload = dict(SPOT_BBO, u=400900240, E=None)
    assert decode_frame("binance_spot", _combined("bookTicker", payload))["payload"] == payload


@pytest.mark.parametrize("source,suffix,payload,kind,seq,trade_time", [
    ("binance_spot", "aggTrade", dict(AGG_TRADE, M=True), "trade", 12345, NATIVE_MS - 3),
    ("binance_spot", "depth@100ms", SPOT_DEPTH, "depth", 160, None),
    ("binance_perp_book", "bookTicker", dict(SPOT_BBO, e="bookTicker", E=NATIVE_MS, T=NATIVE_MS - 1),
     "bbo", 400900217, NATIVE_MS - 1),
    ("binance_perp_book", "depth@100ms", dict(SPOT_DEPTH, T=NATIVE_MS - 2, pu=153, ps="BTCUSDT", st=1),
     "depth", 160, NATIVE_MS - 2),
    ("binance_perp_market", "aggTrade", dict(AGG_TRADE, nq="0.00000100"), "trade", 12345, NATIVE_MS - 3),
    ("binance_perp_market", "markPrice@1s", MARK_PRICE, "funding", None, None),
    ("binance_perp_market", "forceOrder", LIQUIDATION, "liquidation", None, NATIVE_MS - 6),
])
def test_binance_market_record_uses_documented_time_and_sequence_fields(source, suffix, payload, kind, seq, trade_time):
    record = decode_frame(source, _combined(suffix, payload))
    assert record["kind"] == kind
    assert record["channel"] == f"btcusdt@{suffix}"
    assert record["market"] == "BTCUSDT"
    assert record["payload"] == payload
    assert record["source_event_time_ns"] == NATIVE_MS * 1_000_000
    assert record["source_transaction_time_ns"] == (None if trade_time is None else trade_time * 1_000_000)
    assert record["source_sequence"] == seq
    assert record["source_time_basis"] == "unix_ms"


def test_binance_funding_preserves_optional_decimal_fields():
    payload = dict(MARK_PRICE, P=None, ap="95000.00010100")
    assert decode_frame("binance_perp_market", _combined("markPrice@1s", payload))["payload"] == payload
    del payload["P"]
    assert decode_frame("binance_perp_market", _combined("markPrice@1s", payload))["payload"] == payload


@pytest.mark.parametrize("source,payload,channel", [
    ("binance_spot", {"result": None, "id": 1}, None),
    ("binance_perp_book", {"code": 2, "msg": "Invalid request", "id": 1}, None),
    ("paradex", {"jsonrpc": "2.0", "result": {"channel": "bbo.BTC-USD-PERP"}, "id": 1}, "bbo.BTC-USD-PERP"),
    ("paradex", {"jsonrpc": "2.0", "error": {"code": -32602, "message": "Bad parameters"}, "id": None}, None),
])
def test_subscription_controls_retain_wire_payload(source, payload, channel):
    record = decode_frame(source, json.dumps(payload))
    assert record["kind"] == "control"
    assert record["encoding"] == "json"
    assert record["payload"] == payload
    assert record["channel"] == channel
    assert record["source_event_time_ns"] is None
    assert record["source_time_basis"] is None


def test_binance_spot_server_shutdown_preserves_schedule_without_transaction_time():
    payload = {"e": "serverShutdown", "E": NATIVE_MS, "T": NATIVE_MS + 500}
    record = decode_frame("binance_spot", json.dumps({"stream": "!serverShutdown", "data": payload}))
    assert record["kind"] == "control"
    assert record["channel"] == "!serverShutdown"
    assert record["payload"] == payload
    assert record["source_event_time_ns"] == NATIVE_MS * 1_000_000
    assert record["source_transaction_time_ns"] is None


@pytest.mark.parametrize("source,raw", [
    ("unknown", "{}"),
    ("binance_spot", "{"),
    ("binance_spot", "null"),
    ("binance_spot", "[]"),
    ("binance_spot", '{"result": NaN, "id": 1}'),
    ("binance_spot", '{"result": 1e999, "id": 1}'),
    ("binance_spot", '{"result": null, "id": 1, "id": 2}'),
    ("binance_spot", '{"stream": "btcusdt@bookTicker", "data": null, "result": null, "id": 1}'),
    ("binance_spot", json.dumps(SPOT_BBO)),
    ("binance_spot", _combined("bookTicker", SPOT_BBO, "ethusdt")),
    ("binance_spot", _combined("bookTicker", dict(SPOT_BBO, e="aggTrade"))),
    ("binance_spot", _combined("bookTicker", dict(SPOT_BBO, u=True))),
    ("binance_spot", _combined("bookTicker", dict(SPOT_BBO, B="NaN"))),
    ("binance_spot", _combined("bookTicker", dict(SPOT_BBO, b="Infinity"))),
    ("binance_spot", _combined("bookTicker", dict(SPOT_BBO, b=100.0))),
    ("binance_spot", _combined("bookTicker", dict(SPOT_BBO, b=None))),
    ("binance_spot", _combined("bookTicker", dict(SPOT_BBO, E="1234"))),
    ("binance_spot", _combined("depth@100ms", dict(SPOT_DEPTH, U=161))),
    ("binance_spot", _combined("depth@100ms", dict(SPOT_DEPTH, b=[["1.2", "0.3", "0"]]))),
    ("binance_spot", _combined("depth@100ms", dict(SPOT_DEPTH, a=None))),
    ("binance_spot", _combined("aggTrade", dict(AGG_TRADE, T=None))),
    ("binance_spot", _combined("aggTrade", dict(AGG_TRADE, f=123))),
    ("binance_perp_market", _combined("bookTicker", SPOT_BBO)),
    ("binance_perp_book", _combined("aggTrade", AGG_TRADE)),
    ("binance_perp_book", _combined("depth@100ms", dict(SPOT_DEPTH, T=NATIVE_MS))),
    ("binance_perp_book", _combined("depth@100ms", dict(SPOT_DEPTH, T=NATIVE_MS, pu=150, st=2))),
    ("binance_perp_market", _combined("markPrice@1s", dict(MARK_PRICE, r="NaN"))),
    ("binance_perp_market", _combined("markPrice@1s", dict(MARK_PRICE, ap="Infinity"))),
    ("binance_perp_market", _combined("forceOrder", dict(LIQUIDATION, o=None))),
    ("paradex", '{"jsonrpc":"2.0","result":{"channel":3},"id":1}'),
    ("paradex", '{"jsonrpc":"2.0","method":"subscription","params":{}}'),
])
def test_rejects_malformed_or_unexpected_json_frames(source, raw):
    with pytest.raises(ProtocolError):
        decode_frame(source, raw)


def test_binance_spot_depth_requires_bridge_and_range_continuity():
    sequence = BinanceDepthSequence(100)
    assert sequence.observe({"U": 95, "u": 100}) == "pre_snapshot"
    assert sequence.observe({"U": 101, "u": 105}) == "bridged"
    assert sequence.observe({"U": 104, "u": 106}) == "continuous"
    assert sequence.observe({"U": 104, "u": 106}) == "stale"
    with pytest.raises(SequenceGap):
        sequence.observe({"U": 108, "u": 109})
    with pytest.raises(SequenceGap):
        BinanceDepthSequence(100).observe({"U": 102, "u": 105})


def test_binance_perp_depth_bridges_snapshot_itself_and_uses_previous_final_update_id():
    sequence = BinanceDepthSequence(100, futures=True)
    assert sequence.observe({"U": 95, "u": 99, "pu": 92}) == "pre_snapshot"
    assert sequence.observe({"U": 99, "u": 100, "pu": 98}) == "bridged"
    assert sequence.observe({"U": 110, "u": 115, "pu": 100}) == "continuous"
    with pytest.raises(SequenceGap):
        sequence.observe({"U": 116, "u": 120, "pu": 114})
    with pytest.raises(SequenceGap):
        BinanceDepthSequence(100, futures=True).observe({"U": 101, "u": 106, "pu": 100})
    with pytest.raises(ProtocolError):
        BinanceDepthSequence(100, futures=True).observe({"U": 99, "u": 102})
