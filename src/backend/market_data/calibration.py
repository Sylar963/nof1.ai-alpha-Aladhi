"""Receipt-causal public market speed and transaction-level encounter measurements."""

from bisect import bisect_left, bisect_right, insort
from collections import Counter, OrderedDict, deque
from copy import copy
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta
from hashlib import sha256
import json
import math
from zoneinfo import ZoneInfo

from src.backend.market_data.feeds import SOURCE_NAMES
from src.backend.market_data.trades import trade_print


NS = 1_000_000_000
HORIZONS = (1, 5, 30, 120)
VERSION = 'speed-acceptance-v1'


@dataclass(frozen=True)
class CalibrationConfig:
    session_timezone: str = 'America/Tijuana'
    range_start: str = '20:00'
    range_minutes: int = 15
    max_age_s: float = 3
    history_seconds: int = 3600
    min_baseline_samples: int = 30
    or_max_gap_s: float = 30
    band_bps: float = 1
    min_dwell_s: float = 5
    min_outside_fraction: float = 0.7
    min_encounter_samples: int = 3
    encounter_timeout_s: float = 120
    recent_encounters: int = 128
    dedup_capacity: int = 8192

    def __post_init__(self):
        ZoneInfo(self.session_timezone)
        if len(self.range_start) != 5:
            raise ValueError('range_start must use HH:MM')
        time.fromisoformat(self.range_start)
        if type(self.range_minutes) is not int or not 1 <= self.range_minutes <= 240:
            raise ValueError('range_minutes must be an integer in 1..240')
        for name, value in asdict(self).items():
            if name in ('session_timezone', 'range_start'):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('history_seconds', 'min_baseline_samples', 'min_encounter_samples',
                     'recent_encounters', 'dedup_capacity'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if self.min_outside_fraction > 1 or self.min_baseline_samples > self.history_seconds:
            raise ValueError('invalid occupancy fraction or baseline warmup')


def _time_ns(value):
    return type(value) is int and 0 <= value < 2**63


def _positive(value):
    if isinstance(value, bool):
        raise ValueError('expected positive finite market value')
    result = float(value)
    # Keep derived level prices, basis ratios and accumulated notionals representable.
    if not 1e-100 <= result <= 1e100:
        raise ValueError('market value outside supported numeric range')
    return result


def _log_change(price, previous):
    return 10000 * math.log(price / previous)


def _empty_horizon(reason, count=0):
    return dict(speed_bps_s=None, acceleration_bps_s2=None, abs_percentile=None,
                baseline_samples=count, reason=reason)


class _Baseline:
    def __init__(self, config):
        self.values = deque(maxlen=config.history_seconds)
        self.sorted_values = []
        self.minimum = config.min_baseline_samples

    def measure(self, value):
        n = len(self.values)
        rank = None
        if n >= self.minimum:
            left = bisect_left(self.sorted_values, value)
            right = bisect_right(self.sorted_values, value)
            rank = 50 * (left + right) / n
        if n == self.values.maxlen:
            self.sorted_values.pop(bisect_left(self.sorted_values, self.values[0]))
        self.values.append(value)
        insort(self.sorted_values, value)
        return n, rank


class _Stream:
    def __init__(self, config):
        self.config = config
        self.price = self.ns = self.original_ns = self.channel = None
        self.samples = 0
        self.reset_history()

    def reset_history(self):
        self.grid = deque(maxlen=max(HORIZONS) + 1)
        self.grid_second = None
        self.baselines = {h: _Baseline(self.config) for h in HORIZONS}
        self.horizons = {str(h): _empty_horizon('insufficient_coverage') for h in HORIZONS}

    def clear(self):
        self.price = self.ns = self.original_ns = self.channel = None
        self.reset_history()

    def advance(self, target, on_sample=None):
        if self.ns is None:
            return
        first = (self.ns + NS - 1) // NS if self.grid_second is None else self.grid_second + 1
        last = min(target, (self.original_ns + int(self.config.max_age_s * NS)) // NS)
        for second in range(first, last + 1):
            self.grid.append(self.price)
            self.grid_second = second
            for h in HORIZONS:
                if len(self.grid) <= h:
                    continue
                speed = _log_change(self.price, self.grid[-1-h]) / h
                n, rank = self.baselines[h].measure(abs(speed))
                previous = self.horizons[str(h)]['speed_bps_s']
                self.horizons[str(h)] = dict(
                    speed_bps_s=speed, acceleration_bps_s2=None if previous is None else speed - previous,
                    abs_percentile=rank, baseline_samples=n, reason='baseline_warmup' if rank is None else None,
                )
            if on_sample is not None:
                on_sample(second * NS, {key: value.copy() for key, value in self.horizons.items()})

    def put(self, ns, original_ns, price, channel):
        if self.ns is not None and ns - self.original_ns > self.config.max_age_s * NS:
            self.reset_history()
        self.price, self.ns, self.original_ns, self.channel = price, ns, original_ns, channel
        self.samples += 1

    def view(self, now):
        stream = self
        if self.ns is not None and now is not None:
            first = (self.ns + NS - 1) // NS if self.grid_second is None else self.grid_second + 1
            last = min((now - 1) // NS, (self.original_ns + int(self.config.max_age_s * NS)) // NS)
            if last >= first:
                stream = copy(self)
                stream.grid = self.grid.copy()
                stream.horizons = {key: value.copy() for key, value in self.horizons.items()}
                stream.baselines = {}
                for horizon, baseline in self.baselines.items():
                    duplicate = copy(baseline)
                    duplicate.values = baseline.values.copy()
                    duplicate.sorted_values = baseline.sorted_values.copy()
                    stream.baselines[horizon] = duplicate
                stream.advance(last)
        age = None if self.ns is None else (now - self.original_ns) / NS
        reason = 'missing' if age is None else 'stale' if age > self.config.max_age_s else None
        return dict(price=self.price, channel=self.channel, received_ns=self.ns, original_receipt_ns=self.original_ns, age_s=age, reason=reason,
                    samples=self.samples, grid_ns=None if stream.grid_second is None else stream.grid_second * NS,
                    horizons={k: (_empty_horizon(reason, v['baseline_samples']) if reason else v.copy())
                              for k, v in stream.horizons.items()})


def _window(event, config):
    zone = ZoneInfo(config.session_timezone)
    local = datetime.fromtimestamp(event // NS, zone)
    start = time.fromisoformat(config.range_start)
    date = local.date() if local.time() >= start else local.date() - timedelta(days=1)
    open_at = datetime.combine(date, start, zone)
    return (date.isoformat(), int(open_at.timestamp()) * NS,
            int((open_at + timedelta(days=1)).timestamp()) * NS)


class _Range:
    def __init__(self, config):
        self.config = config
        self.max_gap = config.or_max_gap_s * NS
        self.date = self.start = self.end = self.next_start = None
        self.high = self.low = None
        self.count = 0
        self.failure = None
        self.closed = False
        self.levels = {}

    def fail(self, reason):
        self.failure = self.failure or reason
        self.levels.clear()

    def put(self, event, price, previous_event, receipt_gap):
        if self.start is None or event >= self.next_start:
            self.__init__(self.config)
            self.date, self.start, self.next_start = _window(event, self.config)
            self.end = self.start + self.config.range_minutes * 60 * NS
            if event > self.start and (previous_event is None or previous_event < self.start - self.max_gap):
                self.fail('missing_open')
        if self.closed:
            return
        if previous_event is not None and (event - previous_event > self.max_gap or receipt_gap > self.max_gap):
            self.fail('trade_gap')
        if event < self.end:
            self.count += 1
            self.high = price if self.high is None else max(self.high, price)
            self.low = price if self.low is None else min(self.low, price)
            return
        self.closed = True
        if self.count < 2:
            self.fail('missing_trades')
        if self.failure:
            return
        midpoint = (self.high + self.low) / 2
        step = 2 * (self.high - self.low)
        self.levels = dict(ORH=self.high, ORL=self.low, ORM=midpoint)
        for i in range(1, 5):
            self.levels[f'U{i}'] = midpoint * 1.0033 + (i-1) * step
            self.levels[f'D{i}'] = midpoint * 0.9967 - (i-1) * step

    def view(self, wall):
        expired = self.next_start is not None and wall is not None and wall >= self.next_start
        if expired or self.start is None:
            return dict(session_date=None, start_ns=None, end_ns=None, status='missing',
                        reason='no_session_trades', levels={}, trade_count=0)
        reason = self.failure
        if not self.closed and wall is not None and wall >= self.end:
            reason = reason or 'missing_cutoff_coverage'
        status = 'incomplete' if reason else 'complete' if self.closed else 'forming'
        return dict(session_date=self.date, start_ns=self.start, end_ns=self.end, status=status,
                    reason=reason or (None if self.closed else 'awaiting_cutoff_trade'),
                    levels=self.levels.copy(), trade_count=self.count)


def _copy_encounter(row):
    return {**row, 'arrival_horizons': {k: v.copy() for k, v in row['arrival_horizons'].items()}}


class _Encounter:
    def __init__(self, row, direction, price, band):
        self.row = row
        self.direction = direction
        self.first_price = self.last_price = price
        self.path = 0.0
        self.band = band
        self.ever_outside = False
        self.was_outside = False

    def update(self, ns, wall, event, price, size, side, config):
        r = self.row
        dt = (ns - r['last_ns']) / NS
        if self.was_outside:
            r['outside_s'] += dt
        r['dwell_s'] = (ns - r['start_ns']) / NS
        r['outside_fraction'] = r['outside_s'] / r['dwell_s'] if r['dwell_s'] else None
        r['signed_distance_bps'] = 10000 * (price - r['level_price']) / r['level_price']
        progress = self.direction * (price - r['level_price'])
        outside = progress > self.band
        reclaimed = self.ever_outside and progress < -self.band
        self.ever_outside |= outside
        self.was_outside = outside
        if reclaimed:
            r['reclaims'] += 1
        self.path += abs(price - self.last_price)
        self.last_price = price
        r['efficiency'] = abs(price - self.first_price) / self.path if self.path else None
        r['samples'] += 1
        r['volume'] += size
        if side:
            r['known_side_volume'] += size
            r['signed_volume'] = (r['signed_volume'] or 0) + side * size
        r.update(last_ns=ns, last_wall_ns=wall, last_event_ns=event, last_price=price)
        enough = r['dwell_s'] >= config.min_dwell_s and r['samples'] >= config.min_encounter_samples
        if progress < -self.band and (enough or reclaimed):
            r.update(state='rejection' if enough else 'insufficient_evidence', complete=True,
                     reason=None if enough else 'short_reclaim')
        elif enough and outside and r['outside_fraction'] >= config.min_outside_fraction:
            r.update(state='persistent_acceptance', reason=None)
        else:
            r.update(state='arrival' if not enough else 'insufficient_evidence', reason='limited_persistence')
        if r['dwell_s'] >= config.encounter_timeout_s:
            r.update(complete=True, reason=r['reason'] or 'measurement_window_end')


class _Source:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.quote, self.trade = _Stream(config), _Stream(config)
        self.range = _Range(config)
        self.market = self.session = self.event = self.ordinal = None
        self.counters = Counter()
        self.seen = OrderedDict()
        self.active = {}
        self.recent = deque(maxlen=config.recent_encounters)
        self.serial = 0
        self.last_break_reason = None

    def mark_break(self, reason):
        self.counters['trade_breaks'] += 1
        self.last_break_reason = reason

    def break_trade(self, event):
        self.mark_break('invalid_trade')
        if _time_ns(event):
            date, start, next_start = _window(event, self.config)
            if self.range.start is None or event >= self.range.next_start:
                self.range = _Range(self.config)
                self.range.date, self.range.start, self.range.next_start = date, start, next_start
                self.range.end = start + self.config.range_minutes * 60 * NS
            if self.range.start <= event < self.range.end:
                self.range.fail('invalid_range_trade')
        elif not self.range.closed:
            self.range.fail('invalid_range_trade')
        self.trade.clear()
        return self.close('invalid_trade')

    def close(self, reason):
        rows = []
        for encounter in self.active.values():
            encounter.row.update(state='insufficient_evidence', reason=reason, complete=True)
            self.recent.append(encounter.row)
            rows.append(_copy_encounter(encounter.row))
        self.active.clear()
        return rows

    def reset(self, reason):
        self.mark_break(reason)
        self.quote.clear()
        self.trade.clear()
        self.ordinal = None
        if not self.range.closed and self.range.start is not None:
            self.range.fail(reason)
        return self.close(reason)

    def put_trade(self, ns, wall, event, price, size, side, identity, channel, config_version, original, sequence):
        if identity in self.seen:
            self.counters['duplicate_trades'] += 1
            return []
        if len(self.seen) == self.config.dedup_capacity:
            self.seen.popitem(last=False)
        self.seen[identity] = None
        if self.event is not None and event < self.event:
            self.mark_break('late_trade')
            self.counters['late_trades'] += 1
            if self.range.start is not None and self.range.start <= event < self.range.end:
                self.range.fail('late_range_trade')
                return self.close('late_range_trade')
            return []
        if wall is not None and abs(event - wall) > self.config.or_max_gap_s * NS:
            self.mark_break('trade_clock_mismatch')
            self.counters['clock_mismatch_trades'] += 1
            if self.range.start is not None and self.range.start <= event < self.range.end:
                self.range.fail('trade_clock_mismatch')
                return self.close('trade_clock_mismatch')
            return []
        prev_ns, previous_price = self.trade.ns, self.trade.price
        delta = ns - prev_ns if prev_ns is not None else 0
        prior_receipt = self.trade.original_ns
        stale = (ns - original > self.config.max_age_s * NS
                 or (prior_receipt is not None and ns - prior_receipt > self.config.max_age_s * NS))
        prev_session_date = self.range.date
        self.range.put(event, price, self.event if prev_ns is not None else None,
                       max(delta, original - prior_receipt) if prior_receipt is not None else 0)
        self.event = event
        self.trade.put(ns, original, price, channel)
        if prev_session_date != self.range.date or stale:
            self.mark_break('session_boundary' if prev_session_date != self.range.date else 'stale')
            return self.close('session_boundary' if prev_session_date != self.range.date else 'stale')
        if not self.range.levels or prev_ns is None:
            return []
        changed = []
        for name, level in self.range.levels.items():
            if level <= 0:
                continue
            band = level * self.config.band_bps / 10000
            up = previous_price < level - band and price >= level - band
            down = previous_price > level + band and price <= level + band
            if up or down:
                key = (name, 1 if up else -1)
                prior = self.active.pop(key, None)
                if prior is not None:
                    prior.row.update(complete=True, reason='next_arrival')
                    self.recent.append(prior.row)
                    changed.append(_copy_encounter(prior.row))
                self.serial += 1
                row = dict(id=f'{self.name}/{self.range.date}/{self.session}/{ns}/{name}/{self.serial}',
                           source=self.name, session_id=self.session, session_date=self.range.date,
                           market=self.market, config_version=config_version, level=name, level_price=level,
                           direction=1 if up else -1, start_ns=ns, last_ns=ns, start_wall_ns=wall,
                           start_event_ns=event, start_receipt_ns=original, start_sequence=sequence,
                           arrival_horizons={k: v.copy() for k, v in self.trade.horizons.items()},
                           arrival_price=price, skipped_touch=abs(price-level) > band,
                           arrival_speed_bps_s=_log_change(price, previous_price) * NS / delta if delta else None,
                           samples=0, volume=0.0, signed_volume=None, known_side_volume=0.0,
                           outside_s=0.0, reclaims=0, complete=False)
                self.active[key] = _Encounter(row, row['direction'], price, band)
        for key, encounter in list(self.active.items()):
            encounter.update(ns, wall, event, price, size, side, self.config)
            encounter.row.update(last_receipt_ns=original, last_sequence=sequence)
            changed.append(_copy_encounter(encounter.row))
            if encounter.row['complete']:
                self.recent.append(encounter.row)
                del self.active[key]
        return changed


def _parse_trade(record, name, payload):
    if name == 'paradex':
        parsed = trade_print(payload)
        if parsed is None:
            return None
        event, price, size, identity = parsed
        side = {'BUY': 1, 'SELL': -1}.get(payload.get('side'), 0)
    else:
        price, size, identity = payload['p'], payload['q'], payload['a']
        if type(identity) is not int or identity < 0:
            raise ValueError('invalid aggregate trade ID')
        event = record.get('source_transaction_time_ns')
        if event is None:
            event = record.get('source_event_time_ns')
        maker = payload.get('m')
        side = (-1 if maker else 1) if type(maker) is bool else 0
    if not _time_ns(event):
        raise ValueError('invalid trade time')
    return event, _positive(price), _positive(size), side, identity


def _snapshot_best(levels, best):
    if not isinstance(levels, list):
        raise ValueError('missing snapshot levels')
    prices = []
    for level in levels:
        if not isinstance(level, list) or len(level) != 2:
            raise ValueError('invalid snapshot level')
        price = _positive(level[0])
        size = float(level[1])
        if isinstance(level[1], bool) or not 0 <= size <= 1e100:
            raise ValueError('invalid snapshot quantity')
        if size > 0:
            prices.append(price)
    return best(prices)


def _midpoint(name, kind, payload):
    if kind == 'depth':
        if name != 'paradex' or payload.get('pkg_type') != 'SNAPSHOT':
            return None
        bid, ask = payload.get('best_bid_api_price'), payload.get('best_ask_api_price')
        if bid is None:
            bid = _snapshot_best(payload.get('bids'), max)
        if ask is None:
            ask = _snapshot_best(payload.get('asks'), min)
    elif name == 'paradex':
        bid, ask = payload.get('bid_price'), payload.get('ask_price')
    else:
        bid, ask = payload.get('b'), payload.get('a')
    bid, ask = _positive(bid), _positive(ask)
    if bid > ask:
        raise ValueError('crossed quote')
    return (bid + ask) / 2


def _market_pair(market):
    if not isinstance(market, str):
        return None, None
    symbol = market.upper().removesuffix('-PERP').replace('-', '')
    for currency in ('USDT', 'USDC', 'USD', 'BTC', 'ETH'):
        if symbol.endswith(currency) and len(symbol) > len(currency):
            return symbol[:-len(currency)], currency
    return None, None


class SpeedAcceptanceEngine:
    """Use observe in capture sequence; snapshot clocks are monotonic availability nanoseconds."""

    def __init__(self, config=None, *, on_grid_sample=None):
        self.config = config or CalibrationConfig()
        digest = sha256(json.dumps(asdict(self.config), sort_keys=True).encode()).hexdigest()[:12]
        self.config_version = f'{VERSION}/{digest}'
        self._sources = {name: _Source(name, self.config) for name in SOURCE_NAMES}
        self._ns = self._wall = self._grid = None
        self._sequence = None
        self.on_grid_sample = on_grid_sample
        self.accepted_trade = None
        self._before_revisions = {}

    @property
    def trade_revisions(self):
        return {name: source.counters['trade_breaks'] for name, source in self._sources.items()}

    @property
    def trade_breaks(self):
        return {name: source.last_break_reason for name, source in self._sources.items()
                if source.counters['trade_breaks'] > self._before_revisions.get(name, 0)}

    def opening_range(self, source):
        return self._sources[source].range.view(self._wall)

    def observe(self, record):
        """Return independent copies of encounter rows changed or closed by this envelope."""
        self.accepted_trade = None
        self._before_revisions = self.trade_revisions
        name = record.get('source')
        source = self._sources.get(name)
        wall = record.get('received_at_ns')
        original = record.get('received_monotonic_ns')
        if original is None:
            original = wall
        available = record.get('available_monotonic_ns', original)
        if not _time_ns(original) or not _time_ns(available) or (wall is not None and not _time_ns(wall)):
            if source:
                source.counters['invalid_records'] += 1
            return []
        ns = max(self._ns or 0, available, original)
        if source and ns > original:
            source.counters['delayed_submissions'] += 1
        sequence = record.get('capture_sequence')
        self._sequence = sequence
        self._ns = ns
        if wall is not None:
            self._wall = wall + ns - original
        grid = (ns - 1) // NS
        if self._grid != grid:
            for source_name, item in self._sources.items():
                for stream_name in ('quote', 'trade'):
                    sink = None if self.on_grid_sample is None else (
                        lambda tick, horizons, name=source_name, kind=stream_name:
                        self.on_grid_sample(name, kind, tick, horizons))
                    getattr(item, stream_name).advance(grid, sink)
            self._grid = grid
        if source is None:
            return []
        source.counters['records'] += 1
        kind = record.get('kind')
        session = record.get('session_id')
        changed = []
        if session is not None and session != source.session:
            if source.session is not None:
                changed += source.reset('capture_session_changed')
                source.counters['session_changes'] += 1
            source.session = session
        if kind in ('gap', 'quarantine', 'connection_end') or record.get('depth_sequence_status') == 'gap' or (
            kind == 'depth_alignment' and record.get('state') == 'gap'
        ):
            source.counters['interruptions'] += 1
            return changed + source.reset('capture_interruption')
        ordinal = record.get('frame_ordinal')
        if type(ordinal) is int:
            if source.ordinal is not None and ordinal > source.ordinal + 1:
                source.counters['ordinal_gaps'] += 1
                changed += source.reset('frame_gap')
            source.ordinal = ordinal
        if kind not in ('bbo', 'depth', 'trade'):
            return changed
        payload = record.get('payload')
        if not isinstance(payload, dict):
            source.counters['invalid_records'] += 1
            if kind == 'trade':
                return changed + source.break_trade(record.get('source_transaction_time_ns'))
            return changed
        if payload.get('recovered') or record.get('recovered'):
            source.counters['recovered_excluded'] += 1
            return changed
        market = record.get('market') or payload.get('market')
        if source.market is not None and market is not None and source.market != market:
            source.counters['unexpected_market'] += 1
            return changed
        if market is not None:
            source.market = market
        channel = record.get('channel') or kind
        try:
            if kind == 'trade' and name != 'binance_perp_book':
                trade = _parse_trade(record, name, payload)
                if trade is not None:
                    previous_samples = source.trade.samples
                    changed += source.put_trade(ns, wall, *trade, channel, self.config_version,
                                                original, sequence)
                    if source.trade.samples > previous_samples:
                        self.accepted_trade = dict(source=name, market=source.market, session_id=source.session,
                            session_date=source.range.date, event_ns=trade[0], received_ns=ns,
                            original_receipt_ns=original, price=trade[1], capture_sequence=sequence,
                            continuity_revision=source.counters['trade_breaks'])
                else:
                    source.counters['non_price_forming'] += 1
            elif kind != 'trade' and name != 'binance_perp_market':
                if (name == 'paradex' and kind == 'bbo'
                        and (payload.get('bid_price') is None or payload.get('ask_price') is None)
                        and source.quote.channel != channel):
                    source.counters['unavailable_bbo'] += 1
                    return changed
                price = _midpoint(name, kind, payload)
                if price is not None:
                    if source.quote.original_ns is not None and original < source.quote.original_ns:
                        source.counters['late_quotes'] += 1
                    else:
                        source.quote.put(ns, original, price, channel)
        except (KeyError, TypeError, ValueError, ArithmeticError):
            source.counters['invalid_records'] += 1
            if kind in ('bbo', 'depth'):
                source.quote.clear()
            if kind == 'trade':
                event = record.get('source_transaction_time_ns')
                if event is None:
                    event = record.get('source_event_time_ns')
                if name == 'paradex':
                    native = payload.get('created_at_us')
                    if type(native) is int and native > 0:
                        event = native * 1000
                    elif type(payload.get('created_at')) is int:
                        event = payload['created_at'] * 1_000_000
                changed += source.break_trade(event)
        return changed

    def snapshot(self, now_ns=None):
        """Return latest observed prefix with optional elapsed-time aging; never advance a grid."""
        if now_ns is not None and (not _time_ns(now_ns) or (self._ns is not None and now_ns < self._ns)):
            raise ValueError('snapshot time must be at or after the last available record')
        now = self._ns if now_ns is None else now_ns
        wall = None if self._wall is None else self._wall + now - self._ns
        sources = {}
        for name, s in self._sources.items():
            trade_view = s.trade.view(now)
            encounters = [_copy_encounter(r) for r in s.recent]
            for encounter in s.active.values():
                row = _copy_encounter(encounter.row)
                if trade_view['reason'] == 'stale':
                    row.update(state='insufficient_evidence', reason='stale')
                encounters.append(row)
            sources[name] = dict(session_id=s.session, market=s.market, counters=dict(s.counters),
                                 quote=s.quote.view(now), trade=trade_view, encounters=encounters,
                                 opening_range=s.range.view(wall))
        basis = {}
        for label, name in (('perp_minus_spot', 'binance_perp_book'), ('paradex_minus_spot', 'paradex')):
            left = sources[name]['quote']
            right = sources['binance_spot']['quote']
            reason = left['reason'] or right['reason']
            base, currency = _market_pair(sources[name]['market'])
            reference_base, reference_currency = _market_pair(sources['binance_spot']['market'])
            compatible_currency = currency == reference_currency or {currency, reference_currency} <= {'USD', 'USDT', 'USDC'}
            if not base or base != reference_base or not compatible_currency:
                reason = reason or 'incompatible_markets'
            basis[label] = dict(source=name, reference='binance_spot', reason=reason,
                                quote_currencies=[currency, reference_currency], fx_adjusted=False,
                                bps=None if reason else 10000 * (left['price'] / right['price'] - 1),
                                age_s=[left['age_s'], right['age_s']])
        return dict(version=VERSION, config_version=self.config_version, config=asdict(self.config),
                    as_of_ns=now, capture_sequence=self._sequence, receipt_clock='monotonic_availability',
                    range_clock='exchange_transaction_or_event', range_timezone=self.config.session_timezone,
                    sources=sources, basis=basis)
