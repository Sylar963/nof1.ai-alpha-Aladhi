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

    def advance(self, now_ns, sequence_bound=None):
        while self._next is not None:
            record = self._next
            available = record['available_monotonic_ns']
            if available > now_ns or (available == now_ns and sequence_bound is not None
                                      and record['capture_sequence'] > sequence_bound):
                break
            self.engine.observe(record)
            self._next = next(self._records, None)

    def snapshot(self, now_ns):
        return self.engine.snapshot(now_ns)

    def close(self):
        self._records.close()
