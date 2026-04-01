"""Tests for config.py defaults and overridable keys."""
from __future__ import annotations

import config


class TestGetDefaults:
    def test_returns_dict(self):
        defaults = config.get_defaults()
        assert isinstance(defaults, dict)

    def test_contains_all_overridable_keys(self):
        defaults = config.get_defaults()
        expected_keys = {"epochs", "learning_rate", "batch_size", "hidden_size", "use_bias", "seed", "dimensions"}
        assert expected_keys.issubset(defaults.keys())

    def test_values_match_module_constants(self):
        defaults = config.get_defaults()
        assert defaults["epochs"] == config.EPOCHS
        assert defaults["learning_rate"] == config.LEARNING_RATE
        assert defaults["batch_size"] == config.BATCH_SIZE
        assert defaults["hidden_size"] == config.HIDDEN_SIZE

    def test_apply_overrides(self):
        original = config.EPOCHS
        config.apply_config({"epochs": 999})
        assert config.EPOCHS == 999
        config.apply_config({"epochs": original})
        assert config.EPOCHS == original
