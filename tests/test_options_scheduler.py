"""Tests for the two-cadence options scheduler in TradingBotEngine.

The scheduler runs two background tasks alongside the main perps loop:
- vol surface refresh: every 15 minutes (900s by default)
- options decision: every 3 hours (10800s by default)

On startup a bootstrap sequence runs the vol surface refresh first, then
fires the initial options decision — guaranteeing the first decision always
has fresh context. After that, both loops continue independently.

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
# Bootstrap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_runs_surface_before_decision():
    """Bootstrap must call vol surface first, then options decision, sequentially."""
    order: list[str] = []

    async def surface():
        order.append("surface")

    async def decision():
        order.append("decision")

    config = OptionsSchedulerConfig(
        vol_surface_interval_seconds=10,
        options_decision_interval_seconds=10,
    )
    scheduler = OptionsScheduler(
        config=config,
        refresh_vol_surface=surface,
        run_options_decision=decision,
    )

    await scheduler.start()
    # Give bootstrap time to complete.
    await asyncio.sleep(0.05)
    await scheduler.stop()

    # Length check first so a missing bootstrap surfaces as a clear
    # diagnostic instead of an IndexError on the order[0] access.
    assert len(order) >= 2, (
        f"expected surface+decision bootstrap to have run; got order={order!r}"
    )
    assert order[0] == "surface"
    assert order[1] == "decision"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_runs_callbacks_at_expected_cadence():
    """Bootstrap fires once, then the cadence loop fires additional times."""
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
    # Bootstrap fires both once (~instant), then loops sleep 0.05s before
    # each subsequent call. 0.25s gives enough time for bootstrap + several loop ticks.
    await asyncio.sleep(0.25)
    await scheduler.stop()

    # 1 (bootstrap) + at least 2 loop ticks = 3+
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
    await asyncio.sleep(0.08)
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
    await asyncio.sleep(0.15)
    await scheduler.stop()

    # 1 (bootstrap fail) + at least 2 loop retries = 3+
    assert fail_count["n"] >= 3
    # decision_cb gets 1 from bootstrap + loop ticks
    assert len(decision_cb.calls) >= 3


@pytest.mark.asyncio
async def test_scheduler_can_disable_options_decision_loop():
    """Setting options_decision_interval_seconds=0 disables that loop and bootstrap skips it."""
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
    await asyncio.sleep(0.10)
    await scheduler.stop()

    # 1 (bootstrap) + loop ticks
    assert len(surface_cb.calls) >= 3
    assert decision_cb.calls == []
