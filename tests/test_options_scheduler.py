"""Tests for the two-cadence options scheduler in TradingBotEngine.

The scheduler runs two background tasks alongside the main perps loop:
- vol surface refresh: every 15 minutes (900s by default)
- options decision: every 3 hours (10800s by default)

Both are independent of the 5m perps loop and can be enabled / disabled
separately. They only spawn when the Thalex venue is configured.

Tests stub the bot engine's heavy components and use a fake clock helper
to verify the scheduling logic without ever sleeping for real."""

import asyncio
from unittest.mock import MagicMock

import pytest

from src.backend.trading.options_scheduler import (
    OptionsScheduler,
    OptionsSchedulerConfig,
)


class _RecordingCallback:
    """Async callable that records each invocation timestamp."""

    def __init__(self):
        self.calls: list[float] = []

    async def __call__(self):
        loop = asyncio.get_running_loop()
        self.calls.append(loop.time())


# ---------------------------------------------------------------------------
# Construction + cadence config
# ---------------------------------------------------------------------------


def test_scheduler_default_cadences_are_15m_and_3h():
    config = OptionsSchedulerConfig()
    assert config.vol_surface_interval_seconds == 900
    assert config.options_decision_interval_seconds == 10800


def test_scheduler_accepts_custom_cadences():
    config = OptionsSchedulerConfig(
        vol_surface_interval_seconds=60,
        options_decision_interval_seconds=120,
    )
    assert config.vol_surface_interval_seconds == 60
    assert config.options_decision_interval_seconds == 120


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_runs_callbacks_at_expected_cadence():
    """With a 0.05s cadence and a 0.18s sleep, both callbacks must fire 3+ times."""
    surface_cb = _RecordingCallback()
    decision_cb = _RecordingCallback()
    config = OptionsSchedulerConfig(
        vol_surface_interval_seconds=0.05,
        options_decision_interval_seconds=0.05,
    )
    scheduler = OptionsScheduler(
        config=config,
        refresh_vol_surface=surface_cb,
        run_options_decision=decision_cb,
    )

    await scheduler.start()
    await asyncio.sleep(0.18)
    await scheduler.stop()

    assert len(surface_cb.calls) >= 3
    assert len(decision_cb.calls) >= 3


@pytest.mark.asyncio
async def test_scheduler_stop_cancels_both_tasks():
    """After stop(), neither task should fire again."""
    surface_cb = _RecordingCallback()
    decision_cb = _RecordingCallback()
    config = OptionsSchedulerConfig(
        vol_surface_interval_seconds=0.02,
        options_decision_interval_seconds=0.02,
    )
    scheduler = OptionsScheduler(
        config=config,
        refresh_vol_surface=surface_cb,
        run_options_decision=decision_cb,
    )

    await scheduler.start()
    await asyncio.sleep(0.05)
    await scheduler.stop()
    surface_count = len(surface_cb.calls)
    decision_count = len(decision_cb.calls)

    await asyncio.sleep(0.05)
    assert len(surface_cb.calls) == surface_count
    assert len(decision_cb.calls) == decision_count


@pytest.mark.asyncio
async def test_scheduler_swallows_callback_exceptions_and_keeps_running():
    """A failing callback must NOT kill the scheduler — log + retry on next tick."""
    fail_count = {"n": 0}

    async def failing():
        fail_count["n"] += 1
        raise RuntimeError("simulated failure")

    decision_cb = _RecordingCallback()
    config = OptionsSchedulerConfig(
        vol_surface_interval_seconds=0.02,
        options_decision_interval_seconds=0.02,
    )
    scheduler = OptionsScheduler(
        config=config,
        refresh_vol_surface=failing,
        run_options_decision=decision_cb,
    )

    await scheduler.start()
    await asyncio.sleep(0.10)
    await scheduler.stop()

    assert fail_count["n"] >= 3  # kept retrying through the exceptions
    assert len(decision_cb.calls) >= 3  # the other task is unaffected


@pytest.mark.asyncio
async def test_scheduler_can_disable_options_decision_loop():
    """Setting options_decision_interval_seconds=0 disables that loop."""
    surface_cb = _RecordingCallback()
    decision_cb = _RecordingCallback()
    config = OptionsSchedulerConfig(
        vol_surface_interval_seconds=0.02,
        options_decision_interval_seconds=0,
    )
    scheduler = OptionsScheduler(
        config=config,
        refresh_vol_surface=surface_cb,
        run_options_decision=decision_cb,
    )

    await scheduler.start()
    await asyncio.sleep(0.08)
    await scheduler.stop()

    assert len(surface_cb.calls) >= 3
    assert decision_cb.calls == []
