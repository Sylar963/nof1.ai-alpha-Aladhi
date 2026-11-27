"""Independent causal feature cursor for Arena Replay."""

from src.backend.market_data.calibration import SpeedAcceptanceEngine
from src.backtest.calibration_dataset import ResearchCapture


LABELS = {'fast_add', 'acceptance_no_add', 'hold_burst', 'release_hedge', 'wait', 'skip'}


class CalibrationReplay:
    def __init__(self, source, config=None, **index_options):
        self.dataset = ResearchCapture(source, **index_options)
        self.engine = SpeedAcceptanceEngine(config)
        self._records = self.dataset.records()
        self._next = next(self._records, None)

    def has_due(self, now_ns, sequence_bound=None):
        if self._next is None:
            return False
        available = self._next['available_monotonic_ns']
        return available < now_ns or (available == now_ns and
                                     (sequence_bound is None or self._next['capture_sequence'] <= sequence_bound))

    def advance(self, now_ns, sequence_bound=None, *, max_events=10000):
        if type(max_events) is not int or max_events < 0:
            raise ValueError('research event budget must be a nonnegative integer')
        count = 0
        while count < max_events and self.has_due(now_ns, sequence_bound):
            self.engine.observe(self._next)
            self._next = next(self._records, None)
            count += 1
        return {'events': count, 'more': self.has_due(now_ns, sequence_bound)}

    def snapshot(self, now_ns):
        return self.engine.snapshot(now_ns)

    def close(self):
        self._records.close()
