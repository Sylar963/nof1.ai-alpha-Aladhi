"""Decode public Binance combined streams and Paradex SBE frames."""

import json
import math
import re
import struct
from decimal import Decimal


class ProtocolError(ValueError):
    """A frame cannot be decoded under the selected public protocol."""


class SequenceGap(ValueError):
    """Binance depth updates no longer cover the snapshot or prior update."""


_BINANCE_STREAMS = {
    "binance_spot": {"bookTicker", "aggTrade", "depth@100ms"},
    "binance_perp_book": {"bookTicker", "depth@100ms"},
    "binance_perp_market": {"aggTrade", "markPrice@1s", "forceOrder"},
}
_DECIMAL_STRING = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_MILLISECONDS_TO_NS = 1_000_000
_MICROSECONDS_TO_NS = 1_000

_SBE_HEADER = struct.Struct("<HHHH")
_SBE_GROUP_HEADER = struct.Struct("<HH")
_SBE_LEVEL = struct.Struct("<qq")
_SBE_TEMPLATES = {
    2: struct.Struct("<6q"),
    3: struct.Struct("<qqB9q"),
    40: struct.Struct("<qq"),
}
_SBE_NULL = -(2**63)
_SBE_PACKAGES = {0: "SNAPSHOT", 1: "DELTA"}
_SBE_BBO_FIELDS = ("bid_price", "bid_size", "ask_price", "ask_size")
_SBE_BOOK_FIELDS = (
    "best_bid_price",
    "best_bid_size",
    "best_ask_price",
    "best_ask_size",
    "relative_spread",
    "best_bid_api_price",
    "best_bid_api_size",
    "best_ask_api_price",
    "best_ask_api_size",
)


def _object(value: object, field: str) -> dict:
    if not isinstance(value, dict):
        raise ProtocolError(f"{field} must be an object")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{field} must be a non-empty string")
    return value


def _unsigned(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ProtocolError(f"{field} must be a non-negative integer")
    return value


def _optional_unsigned(value: object, field: str) -> int | None:
    return None if value is None else _unsigned(value, field)


def _decimals(payload: dict, *fields: str) -> None:
    for field in fields:
        value = payload.get(field)
        if not isinstance(value, str) or _DECIMAL_STRING.fullmatch(value) is None:
            raise ProtocolError(f"{field} must be a finite decimal string")


def _levels(levels: object, field: str) -> None:
    if not isinstance(levels, list):
        raise ProtocolError(f"{field} must be a list of depth levels")
    for level in levels:
        if not isinstance(level, list) or len(level) != 2:
            raise ProtocolError(f"{field} entries must be price/quantity pairs")
        if any(not isinstance(value, str) or _DECIMAL_STRING.fullmatch(value) is None for value in level):
            raise ProtocolError(f"{field} entries must contain finite decimal strings")


def _record(
    kind: str,
    channel: str | None,
    market: str | None,
    payload: dict,
    *,
    encoding: str = "json",
    event_ns: int | None = None,
    transaction_ns: int | None = None,
    sequence: int | None = None,
    time_basis: str | None = None,
) -> dict:
    return {
        "kind": kind,
        "channel": channel,
        "market": market,
        "payload": payload,
        "encoding": encoding,
        "source_event_time_ns": event_ns,
        "source_transaction_time_ns": transaction_ns,
        "source_sequence": sequence,
        "source_time_basis": time_basis,
    }


def _json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ProtocolError("JSON frame has a non-finite number")
    return value


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise ProtocolError(f"JSON frame repeats field {key!r}")
        fields[key] = value
    return fields


def _decode_json(raw: str | bytes) -> dict:
    if not isinstance(raw, str):
        raise ProtocolError("expected a JSON text frame")
    try:
        return _object(json.loads(
            raw, parse_constant=_json_float, parse_float=_json_float, object_pairs_hook=_json_object,
        ), "JSON frame")
    except ProtocolError:
        raise
    except (ValueError, RecursionError) as exc:
        raise ProtocolError(f"invalid JSON frame: {exc}") from exc


def _depth_range(payload: dict, futures: bool) -> tuple[int, int, int | None]:
    first = _unsigned(payload.get("U"), "U")
    last = _unsigned(payload.get("u"), "u")
    if first > last:
        raise ProtocolError("depth update U exceeds u")
    previous = _unsigned(payload.get("pu"), "pu") if futures else None
    return first, last, previous


class BinanceDepthSequence:
    """Check diff coverage from one Binance REST snapshot and stream session."""

    def __init__(self, snapshot_id: int, futures: bool = False):
        self._snapshot_id = _unsigned(snapshot_id, "lastUpdateId")
        self._futures = futures
        self._last_update: int | None = None

    def observe(self, payload: dict) -> str:
        """Classify a diff relative to the snapshot and previously accepted diffs."""
        first, last, previous = _depth_range(_object(payload, "depth update"), self._futures)
        if self._last_update is None:
            target = self._snapshot_id if self._futures else self._snapshot_id + 1
            if last < target:
                return "pre_snapshot"
            if first > target:
                raise SequenceGap(f"snapshot requires update {target}, got range {first}-{last}")
            self._last_update = last
            return "bridged"

        if self._futures:
            if previous != self._last_update or last <= self._last_update:
                raise SequenceGap(f"expected pu={self._last_update} with a later u, got pu={previous}, u={last}")
        else:
            if last <= self._last_update:
                return "stale"
            if first > self._last_update + 1:
                raise SequenceGap(f"expected update {self._last_update + 1}, got range {first}-{last}")

        self._last_update = last
        return "continuous"


def _request_id(frame: dict, *, required: bool = True) -> None:
    if "id" not in frame:
        if required:
            raise ProtocolError("control reply is missing id")
        return
    value = frame["id"]
    if value is not None and type(value) is not int:
        _string(value, "id")


def _binance_control(frame: dict) -> dict:
    if "e" in frame:
        raise ProtocolError("expected a Binance combined-stream envelope")
    if "result" in frame and "code" not in frame and "msg" not in frame:
        _request_id(frame)
        result = frame["result"]
        if result is not None and type(result) is not bool:
            if not isinstance(result, list) or any(not isinstance(stream, str) for stream in result):
                raise ProtocolError("invalid Binance control result")
    elif "result" not in frame and type(frame.get("code")) is int:
        _string(frame.get("msg"), "msg")
        _request_id(frame, required=False)
    else:
        raise ProtocolError("invalid Binance control reply")
    return _record("control", None, None, frame)


def _binance_shutdown(channel: str, payload: dict) -> dict:
    if payload.get("e") != "serverShutdown":
        raise ProtocolError("unexpected Binance shutdown event type")
    event = _unsigned(payload.get("E"), "E")
    _unsigned(payload.get("T"), "T")
    return _record("control", channel, None, payload, event_ns=event * _MILLISECONDS_TO_NS, time_basis="unix_ms")


def _binance_trade(payload: dict) -> int:
    _decimals(payload, "p", "q")
    if "nq" in payload:
        _decimals(payload, "nq")
    first = _unsigned(payload.get("f"), "f")
    last = _unsigned(payload.get("l"), "l")
    if first > last:
        raise ProtocolError("aggregate trade f exceeds l")
    if type(payload.get("m")) is not bool or ("M" in payload and type(payload["M"]) is not bool):
        raise ProtocolError("invalid aggregate trade maker flag")
    return _unsigned(payload.get("a"), "a")


def _binance_liquidation(order: dict) -> None:
    _decimals(order, "q", "p", "ap", "l", "z")
    if order.get("S") not in ("BUY", "SELL"):
        raise ProtocolError("invalid liquidation side")
    for field in ("o", "f", "X"):
        _string(order.get(field), field)


def _decode_binance(source: str, frame: dict) -> dict:
    spot = source == "binance_spot"
    if "stream" not in frame and "data" not in frame:
        if spot and frame.get("e") == "serverShutdown":
            return _binance_shutdown("!serverShutdown", frame)
        return _binance_control(frame)

    channel = _string(frame.get("stream"), "stream")
    payload = _object(frame.get("data"), "data")
    if spot and channel == "!serverShutdown":
        return _binance_shutdown(channel, payload)

    symbol, _, suffix = channel.partition("@")
    if not symbol or suffix not in _BINANCE_STREAMS[source] or {"result", "code", "msg"} & frame.keys():
        raise ProtocolError(f"unexpected Binance channel {channel!r} for {source}")

    body = payload
    sequence = None
    transaction = None
    if suffix == "bookTicker":
        kind = "bbo"
        expected_event = "bookTicker"
        _decimals(payload, "b", "B", "a", "A")
        sequence = _unsigned(payload.get("u"), "u")
    elif suffix == "depth@100ms":
        kind = "depth"
        expected_event = "depthUpdate"
        _, sequence, _ = _depth_range(payload, futures=not spot)
        _levels(payload.get("b"), "b")
        _levels(payload.get("a"), "a")
    elif suffix == "aggTrade":
        kind = "trade"
        expected_event = "aggTrade"
        sequence = _binance_trade(payload)
    elif suffix == "markPrice@1s":
        kind = "funding"
        expected_event = "markPriceUpdate"
        _decimals(payload, "p", "i", "r")
        for field in ("P", "ap"):
            if payload.get(field) is not None:
                _decimals(payload, field)
        _unsigned(payload.get("T"), "T")
    else:
        kind = "liquidation"
        expected_event = "forceOrder"
        body = _object(payload.get("o"), "o")
        _binance_liquidation(body)

    market = _string(body.get("s"), "s")
    if market.lower() != symbol:
        raise ProtocolError(f"Binance channel {channel!r} disagrees with payload symbol {market!r}")
    if not spot:
        if "st" in payload and _unsigned(payload["st"], "st") != 1:
            raise ProtocolError("expected Binance USD-M symbol type")
        if "ps" in payload:
            _string(payload["ps"], "ps")

    if spot and kind == "bbo":
        if "e" in payload and payload["e"] != expected_event:
            raise ProtocolError("unexpected Binance bookTicker event type")
        event = _optional_unsigned(payload.get("E"), "E")
        transaction = _optional_unsigned(payload.get("T"), "T")
    else:
        if payload.get("e") != expected_event:
            raise ProtocolError(f"expected Binance {expected_event} event for {channel!r}")
        event = _unsigned(payload.get("E"), "E")
        if not (kind == "funding" or (spot and kind == "depth")):
            transaction = _unsigned(body.get("T"), "T")

    return _record(
        kind, channel, market, payload,
        event_ns=None if event is None else event * _MILLISECONDS_TO_NS,
        transaction_ns=None if transaction is None else transaction * _MILLISECONDS_TO_NS,
        sequence=sequence,
        time_basis="unix_ms" if event is not None or transaction is not None else None,
    )


def _paradex_control(frame: dict) -> dict:
    if frame.get("jsonrpc") != "2.0" or "method" in frame or ("error" in frame) == ("result" in frame):
        raise ProtocolError("expected a Paradex JSON-RPC control reply")
    _request_id(frame)
    channel = None
    if "error" in frame:
        error = _object(frame["error"], "error")
        if type(error.get("code")) is not int:
            raise ProtocolError("error.code must be an integer")
        _string(error.get("message"), "error.message")
    elif isinstance(frame["result"], dict) and "channel" in frame["result"]:
        channel = _string(frame["result"]["channel"], "result.channel")
    return _record("control", channel, None, frame)


def _sbe_decimal(value: int, *, nullable: bool = False) -> str | None:
    if value == _SBE_NULL:
        if nullable:
            return None
        raise ProtocolError("null mantissa in a required SBE depth level")
    return format(Decimal(f"{value}e-8"), "f")


def _sbe_levels(raw: bytes, offset: int) -> tuple[list[list[str]], int]:
    if offset + _SBE_GROUP_HEADER.size > len(raw):
        raise ProtocolError("truncated SBE group header")
    block_length, count = _SBE_GROUP_HEADER.unpack_from(raw, offset)
    offset += _SBE_GROUP_HEADER.size
    end = offset + count * block_length
    if block_length < _SBE_LEVEL.size or end > len(raw):
        raise ProtocolError("invalid or truncated SBE group block")
    levels = []
    for position in range(offset, end, block_length):
        price, size = _SBE_LEVEL.unpack_from(raw, position)
        levels.append([_sbe_decimal(price), _sbe_decimal(size)])
    return levels, end


def _sbe_market(raw: bytes, offset: int) -> tuple[str, int]:
    if offset >= len(raw):
        raise ProtocolError("truncated SBE market length")
    end = offset + 1 + raw[offset]
    if end > len(raw):
        raise ProtocolError("truncated SBE market")
    try:
        market = raw[offset + 1:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("invalid UTF-8 in SBE market") from exc
    return _string(market, "SBE market"), end


def _decode_paradex(raw: bytes) -> dict:
    if len(raw) < _SBE_HEADER.size:
        raise ProtocolError("truncated Paradex SBE header")
    block_length, template, schema, version = _SBE_HEADER.unpack_from(raw)
    if (schema, version) != (1, 1):
        raise ProtocolError(f"expected Paradex SBE schema 1:1, got {schema}:{version}")
    layout = _SBE_TEMPLATES.get(template)
    if layout is None:
        raise ProtocolError(f"unsupported Paradex SBE template {template}")
    offset = _SBE_HEADER.size + block_length
    if block_length < layout.size or offset > len(raw):
        raise ProtocolError(f"invalid or truncated SBE root block for template {template}")

    timestamp, sequence, *fields = layout.unpack_from(raw, _SBE_HEADER.size)
    _unsigned(timestamp, "SBE timestamp")
    _unsigned(sequence, "SBE sequence")
    payload = {"ts_us": timestamp, "seq_no": sequence}
    market = None
    if template == 40:
        kind, channel = "control", "heartbeat"
    else:
        if template == 2:
            kind, prefix = "bbo", "bbo"
            payload.update(zip(_SBE_BBO_FIELDS, (_sbe_decimal(value, nullable=True) for value in fields), strict=True))
        else:
            kind, prefix = "depth", "order_book"
            package = _SBE_PACKAGES.get(fields[0])
            if package is None:
                raise ProtocolError(f"invalid SBE book package type {fields[0]}")
            payload["pkg_type"] = package
            payload.update(zip(
                _SBE_BOOK_FIELDS, (_sbe_decimal(value, nullable=True) for value in fields[1:]), strict=True,
            ))
            payload["bids"], offset = _sbe_levels(raw, offset)
            payload["asks"], offset = _sbe_levels(raw, offset)
        market, offset = _sbe_market(raw, offset)
        payload["market"] = market
        channel = f"{prefix}.{market}"

    if offset != len(raw):
        raise ProtocolError(f"unexpected bytes after Paradex SBE template {template}")
    return _record(
        kind, channel, market, payload,
        encoding="sbe", event_ns=timestamp * _MICROSECONDS_TO_NS, sequence=sequence, time_basis="unix_us",
    )


def decode_frame(source: str, raw: str | bytes) -> dict:
    """Decode one public WebSocket frame into capture metadata.

    Timestamp fields express Unix nanoseconds. The time basis identifies the
    original clock unit, and the payload retains the source observations.
    """
    if source in _BINANCE_STREAMS:
        return _decode_binance(source, _decode_json(raw))
    if source == "paradex":
        if isinstance(raw, bytes):
            return _decode_paradex(raw)
        return _paradex_control(_decode_json(raw))
    raise ProtocolError(f"unsupported capture source {source!r}")
