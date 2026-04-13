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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", ""),
        ("  # optional override", ""),
        ("https://api.hyperliquid-testnet.xyz", "https://api.hyperliquid-testnet.xyz"),
        (
            "https://api.hyperliquid-testnet.xyz  # optional override",
            "https://api.hyperliquid-testnet.xyz",
        ),
    ],
)
def test_clean_env_value_ignores_comment_only_placeholders(raw, expected):
    assert config_loader._clean_env_value(raw) == expected


def test_get_env_treats_comment_only_override_as_empty(monkeypatch):
    monkeypatch.setenv("HYPERLIQUID_BASE_URL", "  # optional override")

    assert config_loader._get_env("HYPERLIQUID_BASE_URL") == ""


def test_get_env_strips_inline_comment_from_url(monkeypatch):
    monkeypatch.setenv(
        "HYPERLIQUID_BASE_URL",
        "https://api.hyperliquid-testnet.xyz  # optional override",
    )

    assert (
        config_loader._get_env("HYPERLIQUID_BASE_URL")
        == "https://api.hyperliquid-testnet.xyz"
    )
