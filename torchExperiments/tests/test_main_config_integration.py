"""Integration tests for YAML config → main.py pipeline."""
from __future__ import annotations

import os
import tempfile

import yaml

import config
from experiment_config import expand_sweep, load_config, merge_with_defaults


class TestFullPipeline:
    def test_single_experiment_yaml(self):
        data = {"task": "ganea", "model": "hyperbolic", "optimizer": "Radam", "loss": "cross", "dataset": 10, "epochs": 100, "hidden_size": 128}
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.dump(data, f)
        cfg = load_config(path)
        experiments = expand_sweep(cfg)
        assert len(experiments) == 1
        merged = merge_with_defaults(experiments[0])
        assert merged["task"] == "ganea"
        assert merged["epochs"] == 100
        assert merged["hidden_size"] == 128
        assert "learning_rate" in merged
        os.unlink(path)

    def test_sweep_yaml(self):
        data = {"task": "ganea", "loss": "cross", "dataset": 10, "sweep": {"model": ["euclidean", "hyperbolic"], "optimizer": ["Adam", "Radam"]}}
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.dump(data, f)
        cfg = load_config(path)
        experiments = expand_sweep(cfg)
        assert len(experiments) == 4
        assert all(e["task"] == "ganea" for e in experiments)
        os.unlink(path)

    def test_apply_config_overrides_module(self):
        original_epochs = config.EPOCHS
        merged = merge_with_defaults({"epochs": 777})
        config.apply_config(merged)
        assert config.EPOCHS == 777
        config.apply_config({"epochs": original_epochs})
