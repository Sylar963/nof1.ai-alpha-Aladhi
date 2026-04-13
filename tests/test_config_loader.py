"""Tests for config loader validation helpers."""

import pytest

from src.backend import config_loader


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("thalex_delta_threshold", 0),
        ("thalex_delta_threshold", -0.1),
        ("thalex_hedge_reconcile_interval_seconds", 0),
        ("thalex_hedge_reconcile_interval_seconds", -15),
    ],
)
def test_require_positive_rejects_non_positive_values(name, value):
    with pytest.raises(ValueError, match=name):
        config_loader._require_positive(name, value)


def test_require_positive_accepts_positive_values():
    assert config_loader._require_positive("thalex_delta_threshold", 0.02) == 0.02
    assert config_loader._require_positive("thalex_hedge_reconcile_interval_seconds", 15) == 15
