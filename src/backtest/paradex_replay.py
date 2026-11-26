"""Offline Paradex replay, independent human/bot accounts, and teaching logs.

Only recorded replacement snapshots can supply fills. No exchange adapter,
credentials, wall-clock indicators, or network client is used here.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from collections import deque
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from src.backend.market_data.storage import CaptureStorageError, iter_capture_records
from src.backend.market_data.trades import trade_print


ZERO = Decimal("0")
NS = 1_000_000_000
REPLAY_VERSION = 2


def decimal(value, name: str, *, positive: bool = True) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not result.is_finite() or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return result


def utc(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / NS, timezone.utc)


@dataclass(frozen=True)
class ReplayConfig:
    market: str = "BTC-USD-PERP"
    session_timezone: str = "America/Tijuana"
    range_start: str = "20:00"
    range_minutes: int = 15
    order_latency_ms: int = 100
    stale_book_ms: int = 2000
    fee_bps: str = "5"
    starting_equity: str = "10000"
    max_position: str = "1"
    bot_quantity: str = "0.01"
    target_range_multiple: str = "1"
    price_source: str = 'auto'

    def __post_init__(self):
        ZoneInfo(self.session_timezone)
        time.fromisoformat(self.range_start)
        if len(self.range_start) != 5:
            raise ValueError("range_start must use HH:MM")
        for name, minimum, maximum in (("range_minutes", 1, 240),
                                       ("order_latency_ms", 0, 60000),
                                       ("stale_book_ms", 1, 60000)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
        for name in ("starting_equity", "max_position", "bot_quantity", "target_range_multiple"):
            decimal(getattr(self, name), name)
        decimal(self.fee_bps, "fee_bps", positive=False)
        if decimal(self.bot_quantity, "bot_quantity") > decimal(self.max_position, "max_position"):
            raise ValueError("bot_quantity exceeds max_position")
        if not self.market:
            raise ValueError("market is required")
        if self.price_source not in {'auto', 'trades', 'midpoint'}:
            raise ValueError('price_source must be auto, trades, or midpoint')

    @property
    def rule_version(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True).encode()
        return f"orb-{self.price_source}-v2-{hashlib.sha256(encoded).hexdigest()[:12]}"


def _levels(payload: dict, side: str) -> list[list[Decimal]]:
    rows = payload.get(side)
    if not isinstance(rows, list):
        raise ValueError(f"missing {side} in recorded book")
    levels = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("invalid recorded depth level")
        price = decimal(row[0], "price")
        quantity = decimal(row[1], "quantity", positive=False)
        if price in levels:
            raise ValueError("duplicate recorded depth price")
        levels[price] = quantity
    return [[p, q] for p, q in sorted(levels.items(), reverse=side == "bids") if q]


def book_levels(record: dict) -> tuple[list, list]:
    payload = record.get("payload") or {}
    if not isinstance(payload, dict) or payload.get("pkg_type") != "SNAPSHOT":
        raise ValueError("Paradex replay requires replacement snapshots")
    bids, asks = _levels(payload, "bids"), _levels(payload, "asks")
    if bids and asks and bids[0][0] >= asks[0][0]:
        raise ValueError("recorded depth is crossed or locked")
    return bids, asks


def replay_records(path: Path, market: str):
    """Preserve capture order; never sort by exchange timestamps."""
    for record in iter_capture_records(path):
        if record.get("source") != "paradex":
            continue
        if record["kind"] in {"depth", "bbo", "trade"}:
            if record.get("market") != market:
                continue
        elif record["kind"] not in {"gap", "connection_end", "connected", "quarantine"}:
            continue
        yield record


def inspect_capture(path: Path, market: str) -> dict:
    """Stream-validate a closed run before allowing a practice session."""
    path = Path(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        status = json.loads((path / "status.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read capture metadata: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("capture_schema_version") != 1:
        raise ValueError("unsupported capture schema")
    if not isinstance(status, dict) or status.get("closed") is not True or status.get("failed") is not False:
        raise ValueError("finish recording before replay; capture writer must be closed and healthy")
    count = gaps = trades = 0
    first = last = None
    previous = -1
    for record in replay_records(path, market):
        for name in ("received_at_ns", "received_monotonic_ns", "capture_sequence"):
            if type(record.get(name)) is not int or record[name] < 0:
                raise ValueError(f"missing or invalid {name}")
        if not isinstance(record.get("session_id"), str) or not record["session_id"]:
            raise ValueError("missing Paradex connection identity")
        if record["received_monotonic_ns"] < previous:
            raise ValueError("Paradex receive clock moved backwards")
        previous = record["received_monotonic_ns"]
        if record["kind"] == "quarantine":
            raise ValueError("capture contains quarantined Paradex frames")
        if record["kind"] == "depth":
            book_levels(record)
            count += 1
            first = record["received_at_ns"] if first is None else first
            last = record["received_at_ns"]
        elif record['kind'] == 'trade' and trade_print(record['payload']) is not None:
            trades += 1
        gaps += record["kind"] == "gap"
    if not count:
        raise ValueError(f"no Paradex depth snapshots for {market}")
    return {"run_id": manifest.get("run_id", path.name), "depth_snapshots": count,
            "gaps": gaps, "first_depth_ns": first, "last_depth_ns": last,
            "durable_records": status["durable_records"], "durable_bytes": status["durable_bytes"],
            "trade_count": trades}


@dataclass
class Account:
    position: Decimal = ZERO
    average_entry: Decimal = ZERO
    realized_pnl: Decimal = ZERO
    fees: Decimal = ZERO

    def fill(self, side: str, quantity: Decimal, price: Decimal, fee_bps: Decimal):
        signed = quantity if side == "buy" else -quantity
        if self.position * signed >= 0:
            self.average_entry = ((abs(self.position) * self.average_entry + quantity * price)
                                  / (abs(self.position) + quantity))
        else:
            closing = min(abs(self.position), quantity)
            self.realized_pnl += closing * (price - self.average_entry) * (1 if self.position > 0 else -1)
            if quantity > abs(self.position):
                self.average_entry = price
        self.position += signed
        if not self.position:
            self.average_entry = ZERO
        self.fees += quantity * price * fee_bps / Decimal("10000")

    def summary(self, midpoint: Decimal | None, starting_equity: str) -> dict:
        unrealized = ((midpoint - self.average_entry) * self.position if midpoint is not None else None)
        return {**asdict(self), "unrealized_pnl": unrealized,
                "equity": (Decimal(starting_equity) + self.realized_pnl - self.fees + unrealized
                           if unrealized is not None else None)}


@dataclass
class Order:
    id: int
    actor: str
    side: str
    quantity: Decimal
    remaining: Decimal
    order_type: str
    price: Decimal | None
    submitted_sequence: int
    eligible_ns: int
    reduce_only: bool = False
    status: str = "pending"


class ParadexReplay:
    """One forward-only practice attempt. Restart creates a separate attempt."""

    def __init__(self, path: Path, config: ReplayConfig | None = None, *, journal_dir: Path | None = None,
                 source=None):
        self.path = Path(path).resolve()
        self.config = config or ReplayConfig()
        self.capture = source.inspect(self.config.market) if source else inspect_capture(self.path, self.config.market)
        self.price_source = ('trades' if self.capture.get('trade_count') else 'midpoint') if self.config.price_source == 'auto' else self.config.price_source
        self.last_trade = None
        self.last_trade_time_ns = None
        self.last_trade_size = ZERO
        self.price_updated = False
        self._trade_ids = set()
        self._trade_id_order = deque()
        self.attempt_id = uuid4().hex
        self.accounts = {actor: Account() for actor in ("human", "bot")}
        self.orders: list[Order] = []
        self.events: list[dict] = []
        self.history: list[tuple[int, Decimal]] = []
        self.bids: list = []
        self.asks: list = []
        self._liquidity = {actor: ([], []) for actor in self.accounts}
        self.clock_ns = 0
        self.wall_ns = 0
        self.sequence = 0
        self.book_ns: int | None = None
        self.session_id = None
        self.finished = False
        self.failure: str | None = None
        self.baseline_enabled = True
        self.last_kind = None
        self.range: dict = {}
        self._first_depth_wall: int | None = None
        self._previous_depth_wall: int | None = None
        self._last_gap_wall: int | None = None
        self._bot_traded_session = None
        self._bot_stop: Decimal | None = None
        self._bot_target: Decimal | None = None
        self.journal_path = None
        if journal_dir is not None:
            directory = Path(journal_dir) / self.attempt_id
            directory.mkdir(parents=True, exist_ok=False)
            self.journal_path = directory / "journal.ndjson"
        self._iterator = source.records(self.config.market) if source else replay_records(self.path, self.config.market)
        self._next = next(self._iterator, None)
        if self._next is not None:
            self.clock_ns = self._next["received_monotonic_ns"]
            self.wall_ns = self._next["received_at_ns"]
        self._coverage_start_ns = self.capture.get('chart_history_start_ns') or self.wall_ns
        self._log("system", "attempt", config=asdict(self.config), capture=self.capture,
                  capture_path=str(self.path), rule_version=self.config.rule_version)

    @property
    def midpoint(self) -> Decimal | None:
        return (self.bids[0][0] + self.asks[0][0]) / 2 if self.bids and self.asks else None

    @property
    def signal_price(self):
        return self.last_trade if self.price_source == 'trades' else self.midpoint

    @property
    def mark_price(self):
        return self.last_trade if self.price_source == 'trades' else self.history[-1][1] if self.history else None

    @property
    def book_valid(self) -> bool:
        return (self.book_ns is not None and self.midpoint is not None
                and self.clock_ns - self.book_ns <= self.config.stale_book_ms * 1_000_000)

    def _log(self, actor: str, action: str, **fields):
        entry = {"actor": actor, "action": action, "sequence": self.sequence,
                 "replay_time_ns": self.clock_ns, "utc": utc(self.wall_ns).isoformat(), **fields}
        if self.journal_path:
            with self.journal_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, default=str, allow_nan=False) + "\n")
                file.flush()
                os.fsync(file.fileno())
        self.events.append(entry)

    def submit(self, actor: str, side: str, quantity, *, order_type="market", price=None,
               rationale="", reduce_only=False) -> Order:
        if actor not in self.accounts or side not in {"buy", "sell"}:
            raise ValueError("invalid actor or order side")
        if self.finished or self.failure:
            raise ValueError("replay is finished")
        if actor == "human" and not rationale.strip():
            raise ValueError("add your reason before recording an action")
        if not self.book_valid:
            raise ValueError("wait for a fresh two-sided Paradex snapshot")
        if order_type not in {"market", "limit", "stop"}:
            raise ValueError("order_type must be market, limit, or stop")
        qty = decimal(quantity, "quantity")
        limit = decimal(price, "price") if order_type != "market" else None
        pending = sum((o.remaining for o in self.orders if o.actor == actor and o.side == side
                       and o.status in {"pending", "resting", "partial"} and not o.reduce_only), ZERO)
        if not reduce_only and abs(self.accounts[actor].position + (qty + pending) * (1 if side == "buy" else -1)) > Decimal(self.config.max_position):
            raise ValueError("order exceeds simulated position cap")
        if reduce_only and (not self.accounts[actor].position or
                            (self.accounts[actor].position > 0) == (side == "buy")):
            raise ValueError("reduce-only order must reduce an existing position")
        order = Order(sum(o.actor == actor for o in self.orders) + 1, actor, side, qty, qty, order_type, limit,
                      self.sequence, self.clock_ns + self.config.order_latency_ms * 1_000_000,
                      reduce_only=reduce_only)
        self._log(actor, "order", order=asdict(order), rationale=rationale,
                  midpoint=self.midpoint, opening_range=dict(self.range))
        self.orders.append(order)
        return order

    def note(self, action: str, rationale: str):
        if self.finished or self.failure or action not in {"wait", "skip", "note"} or not rationale.strip():
            raise ValueError("record wait, skip, or note with a reason during replay")
        self._log("human", action, rationale=rationale, midpoint=self.midpoint, opening_range=dict(self.range))

    def cancel(self, actor: str, order_id: int, rationale: str):
        order = next((o for o in self.orders if o.id == order_id and o.actor == actor), None)
        if order is None or order.status not in {"pending", "resting", "partial"}:
            raise ValueError("no active order with that ID")
        if not rationale.strip() or self.finished:
            raise ValueError("add a cancellation reason during replay")
        self._log(actor, "cancel", order_id=order.id, rationale=rationale,
                  assumption="cancellation effective immediately at replay cursor")
        order.status = "cancelled"

    def flatten(self, actor: str, rationale: str):
        if actor not in self.accounts or not rationale.strip():
            raise ValueError("add a flatten reason")
        if not self.accounts[actor].position:
            raise ValueError("no position to close")
        # Validate before cancelling existing protective orders.
        if not self.book_valid or self.finished:
            raise ValueError("flatten requires a fresh book during replay")
        for order in self.orders:
            if order.actor == actor and order.status in {"pending", "resting", "partial"}:
                self.cancel(actor, order.id, rationale)
        position = self.accounts[actor].position
        return self.submit(actor, "sell" if position > 0 else "buy", abs(position),
                           rationale=rationale, reduce_only=True)

    def _invalidate_book(self, reason: str):
        self.bids, self.asks = [], []
        self.book_ns = None
        self._last_gap_wall = self.wall_ns
        for order in self.orders:
            if order.status in {"pending", "resting", "partial"}:
                order.status = "cancelled"
                self._log(order.actor, "cancel", order_id=order.id, rationale=reason)

    def _update_range(self, *, now_ns=None, observe=True):
        range_now = self.wall_ns if now_ns is None else now_ns
        now = utc(range_now)
        local = now.astimezone(ZoneInfo(self.config.session_timezone))
        start = datetime.combine(local.date(), time.fromisoformat(self.config.range_start), local.tzinfo)
        if local < start:
            start -= timedelta(days=1)
        start_ns = int(start.timestamp()) * NS
        end_ns = start_ns + self.config.range_minutes * 60 * NS
        if self.range.get("start_ns") != start_ns:
            self.range = {"start_ns": start_ns, "end_ns": end_ns, "high": None, "low": None,
                          "complete": False, "coverage_ok": True}
        state = self.range
        if self._last_gap_wall is not None and start_ns <= self._last_gap_wall < end_ns:
            state["coverage_ok"] = False
        if (self._last_gap_wall is not None and self._previous_depth_wall is not None
                and self._previous_depth_wall <= self._last_gap_wall < start_ns < self.wall_ns):
            state["coverage_ok"] = False
        if self._previous_depth_wall is not None and self.price_source == 'midpoint':
            stale_after = self._previous_depth_wall + self.config.stale_book_ms * 1_000_000
            if stale_after < self.wall_ns and stale_after < end_ns and self.wall_ns > start_ns:
                state["coverage_ok"] = False
        reference = self.signal_price
        if observe and start_ns <= range_now < end_ns and reference is not None:
            state["high"] = max(state["high"], reference) if state["high"] is not None else reference
            state["low"] = min(state["low"], reference) if state["low"] is not None else reference
        elif observe and start_ns <= range_now < end_ns:
            state["coverage_ok"] = False
        coverage_start = self._coverage_start_ns if self.price_source == 'trades' else self._first_depth_wall
        state["complete"] = bool(range_now >= end_ns and state["coverage_ok"]
                                 and coverage_start is not None and coverage_start <= start_ns
                                 and state["high"] is not None and state["high"] > state["low"])

    def _match(self):
        for order in self.orders:
            if order.status not in {"pending", "resting", "partial"}:
                continue
            if self.sequence <= order.submitted_sequence or self.clock_ns < order.eligible_ns:
                continue
            account = self.accounts[order.actor]
            if order.reduce_only and (not account.position or (account.position > 0) == (order.side == "buy")):
                order.status = "cancelled"
                self._log(order.actor, "cancel", order_id=order.id, rationale="reduce-only exposure no longer exists")
                continue
            bids, asks = self._liquidity[order.actor]
            levels = asks if order.side == "buy" else bids
            if order.order_type == "stop":
                trigger = levels and (levels[0][0] >= order.price if order.side == "buy" else levels[0][0] <= order.price)
                if not trigger:
                    order.status = "resting"
                    continue
                order.order_type = "market"
                self._log(order.actor, "stop_trigger", order_id=order.id)
            for level in levels:
                price, available = level
                if order.order_type == "limit" and (price > order.price if order.side == "buy" else price < order.price):
                    break
                size = min(order.remaining, available)
                if order.reduce_only:
                    size = min(size, abs(account.position))
                else:
                    room = Decimal(self.config.max_position) - account.position * (1 if order.side == "buy" else -1)
                    size = min(size, max(ZERO, room))
                if size <= 0:
                    continue
                self._log(order.actor, "fill", order_id=order.id, side=order.side, quantity=size, price=price)
                account.fill(order.side, size, price, Decimal(self.config.fee_bps))
                order.remaining -= size
                level[1] -= size
                if not order.remaining or (order.reduce_only and not account.position):
                    break
            if not order.remaining:
                order.status = "filled"
            elif order.order_type == "market":
                order.status = "partial_cancelled" if order.remaining < order.quantity else "unfilled"
                self._log(order.actor, "unfilled_remainder", order_id=order.id, quantity=order.remaining,
                          rationale="market IOC: insufficient recorded depth or position capacity")
            else:
                order.status = "partial" if order.remaining < order.quantity else "resting"

    def _baseline(self):
        """Transparent teaching baseline: one breakout per range; range stop/target."""
        if not self.baseline_enabled or not self.book_valid:
            return
        account = self.accounts["bot"]
        active = any(o.actor == "bot" and o.status in {"pending", "resting", "partial"} for o in self.orders)
        if active:
            return
        if account.position:
            if self._bot_stop is None or self._bot_target is None:
                return
            exit_price = self.bids[0][0] if account.position > 0 else self.asks[0][0]
            stop = exit_price <= self._bot_stop if account.position > 0 else exit_price >= self._bot_stop
            target = exit_price >= self._bot_target if account.position > 0 else exit_price <= self._bot_target
            if stop or target:
                self.flatten("bot", "range stop" if stop else "range target")
            return
        state = self.range
        if not state.get("complete") or self._bot_traded_session == state["start_ns"]:
            return
        mid = self.signal_price
        if mid is None:
            return
        side = "buy" if mid > state["high"] else "sell" if mid < state["low"] else None
        if side is None:
            return
        width = state["high"] - state["low"]
        self._bot_stop = state["low"] if side == "buy" else state["high"]
        self._bot_target = mid + width * Decimal(self.config.target_range_multiple) * (1 if side == "buy" else -1)
        self.submit("bot", side, self.config.bot_quantity, rationale=f"first {self.price_source} breakout of complete opening range")
        self._bot_traded_session = state["start_ns"]

    def step(self) -> bool:
        if self.finished or self.failure:
            return False
        if self._next is None:
            self.finished = True
            self._log("system", "finished")
            return False
        record = self._next
        self.price_updated = False
        try:
            self.clock_ns = max(self.clock_ns, record["received_monotonic_ns"])
            self.wall_ns = record["received_at_ns"]
            self.sequence = record["capture_sequence"]
            kind = record["kind"]
            self.last_kind = kind
            if self.session_id is not None and self.session_id != record["session_id"]:
                self._invalidate_book("Paradex connection changed")
            self.session_id = record["session_id"]
            if kind in {"gap", "connection_end", "quarantine"}:
                self._invalidate_book(f"recorded {kind}; book invalid until next snapshot")
                self._log("system", kind)
            elif kind == "depth":
                if self.book_ns is not None and self.clock_ns - self.book_ns > self.config.stale_book_ms * 1_000_000:
                    self._invalidate_book("stale interval; pending fills are unknown")
                self.bids, self.asks = book_levels(record)
                self.book_ns = self.clock_ns
                self._liquidity = {actor: ([row[:] for row in self.bids], [row[:] for row in self.asks])
                                   for actor in self.accounts}
                if self.midpoint is not None and self._first_depth_wall is None:
                    self._first_depth_wall = self.wall_ns
                self._update_range(now_ns=record.get('source_event_time_ns') if self.price_source == 'trades' else None,
                                   observe=self.price_source == 'midpoint')
                self._previous_depth_wall = self.wall_ns
                if self.midpoint is not None and self.price_source == 'midpoint':
                    self.history.append((self.wall_ns, self.midpoint))
                    self.price_updated = True
                    if len(self.history) > 5000:
                        del self.history[:1000]
                if self.book_valid:
                    self._match()
                    self._baseline()
            elif kind == 'trade' and self.price_source == 'trades':
                parsed = trade_print(record['payload'])
                if parsed is not None:
                    trade_ns, price, size, identity = parsed
                    if identity not in self._trade_ids:
                        self._trade_ids.add(identity)
                        self._trade_id_order.append(identity)
                        if len(self._trade_id_order) > 30000:
                            self._trade_ids.remove(self._trade_id_order.popleft())
                        if self.last_trade_time_ns is None or trade_ns >= self.last_trade_time_ns:
                            self.last_trade, self.last_trade_time_ns = price, trade_ns
                        self.last_trade_size = size
                        self.history.append((trade_ns, price))
                        if len(self.history) > 5000:
                            del self.history[:1000]
                        self.price_updated = True
                        if not self.range or trade_ns >= self.range['start_ns']:
                            self._update_range(now_ns=trade_ns)
                        self._baseline()
            self._next = next(self._iterator, None)
            if self._next is None:
                self.finished = True
                for order in self.orders:
                    if order.status in {"pending", "resting", "partial"}:
                        order.status = "expired"
                        self._log(order.actor, "expire", order_id=order.id, rationale="end of capture")
                self._log("system", "finished")
            return True
        except (OSError, ValueError, CaptureStorageError) as exc:
            self.failure = str(exc)
            self._invalidate_book("replay failed")
            raise

    def advance_to(self, target_ns: int, *, max_events: int = 1000):
        """Bound processing per UI tick without jumping over unread events."""
        count = 0
        while self._next is not None and self._next["received_monotonic_ns"] <= target_ns and count < max_events:
            if not self.step():
                break
            count += 1
        if not self.finished and (self._next is None or self._next["received_monotonic_ns"] > target_ns):
            delta = max(0, target_ns - self.clock_ns)
            self.clock_ns += delta
            self.wall_ns += delta
        return count

    @property
    def next_event_ns(self):
        return self._next["received_monotonic_ns"] if self._next else None

    def finish(self):
        if self.finished:
            return
        self._log('system', 'practice_end', rationale='user ended this attempt at the replay cursor')
        for order in self.orders:
            if order.status in {'pending', 'resting', 'partial'}:
                order.status = 'expired'
        self.finished = True
        self.close()

    def report(self) -> dict:
        if not self.finished or self.failure:
            raise ValueError("finish replay before revealing the bot comparison")
        return {"replay_version": REPLAY_VERSION, "attempt_id": self.attempt_id,
                "capture": self.capture, "capture_path": str(self.path), "config": asdict(self.config),
                "rule_version": self.config.rule_version, "opening_range": self.range,
                "price_source": self.price_source,
                "trade_history_provenance": self.capture.get('trade_history_provenance'),
                "mark": {"midpoint": self.mark_price, "price": self.mark_price,
                         "utc": utc(self.last_trade_time_ns if self.price_source == 'trades' else self.history[-1][0]).isoformat() if self.history else None,
                         "book_valid_at_end": self.book_valid},
                "accounts": {actor: account.summary(self.mark_price,
                                                     self.config.starting_equity)
                             for actor, account in self.accounts.items()},
                "orders": [asdict(order) for order in self.orders], "events": self.events,
                "assumptions": [f"chart and opening range price source: {self.price_source}",
                                "fills on first subsequent snapshot after configured arrival latency",
                                "market orders are IOC; limits fill only against crossing opposite depth",
                                "passive queue fills are not inferred from touches or disappearing size",
                                "snapshots refresh displayed liquidity; simulated orders do not change historical prices",
                                "human and bot have independent copies of available depth",
                                "cancels are immediate; gaps cancel pending orders; open positions remain",
                                "fees are user assumptions; funding, liquidation and margin are not modeled",
                                f"end positions marked at last observed {self.price_source} price, not forcibly liquidated"]}

    def save_report(self, directory: Path = Path("data/replay_sessions"), *, extra: dict | None = None) -> Path:
        report = self.report()
        if extra:
            report.update(extra)
        path = Path(directory) / self.attempt_id / "comparison.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(report, file, default=str, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
        return path

    def close(self):
        self._iterator.close()
