"""Tests for YAML experiment config loading and sweep expansion."""
from __future__ import annotations
import os, tempfile
import pytest, yaml
from experiment_config import load_config, expand_sweep, apply_filters, merge_with_defaults, VALID_KEYS

def _write_yaml(data: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(data, f)
    return path

class TestLoadConfig:
    def test_loads_valid_yaml(self):
        path = _write_yaml({"task": "ganea", "model": "hyperbolic"})
        cfg = load_config(path)
        assert cfg["task"] == "ganea"
        os.unlink(path)

    def test_unknown_key_raises(self):
        path = _write_yaml({"task": "ganea", "bogus_key": 42})
        with pytest.raises(ValueError, match="Unknown config keys.*bogus_key"):
            load_config(path)
        os.unlink(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

class TestExpandSweep:
    def test_no_sweep_returns_single(self):
        experiments = expand_sweep({"task": "ganea", "model": "hyperbolic"})
        assert len(experiments) == 1

    def test_single_sweep_key(self):
        experiments = expand_sweep({"task": "ganea", "sweep": {"model": ["euclidean", "hyperbolic"]}})
        assert len(experiments) == 2
        assert {e["model"] for e in experiments} == {"euclidean", "hyperbolic"}
        assert all(e["task"] == "ganea" for e in experiments)

    def test_multi_key_sweep_is_product(self):
        experiments = expand_sweep({"sweep": {"model": ["euclidean", "hyperbolic"], "optimizer": ["Adam", "Radam"], "hidden_size": [32, 64]}})
        assert len(experiments) == 8

    def test_sweep_overrides_base(self):
        experiments = expand_sweep({"model": "euclidean", "sweep": {"model": ["hyperbolic"]}})
        assert experiments[0]["model"] == "hyperbolic"

    def test_scalar_in_sweep_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            expand_sweep({"sweep": {"model": "hyperbolic"}})

class TestApplyFilters:
    def test_filter_single_key(self):
        exps = [{"model": "euclidean"}, {"model": "hyperbolic"}, {"model": "hyperbolic"}]
        assert len(apply_filters(exps, "model=hyperbolic")) == 2

    def test_filter_multiple_keys(self):
        exps = [{"model": "euclidean", "optimizer": "Adam"}, {"model": "hyperbolic", "optimizer": "Adam"}, {"model": "hyperbolic", "optimizer": "Radam"}]
        assert len(apply_filters(exps, "model=hyperbolic,optimizer=Radam")) == 1

    def test_no_filter_returns_all(self):
        exps = [{"a": 1}, {"a": 2}]
        assert apply_filters(exps, None) == exps

    def test_filter_no_match_raises(self):
        with pytest.raises(ValueError, match="No experiments match"):
            apply_filters([{"model": "euclidean"}], "model=hyperbolic")

class TestMergeWithDefaults:
    def test_overrides_defaults(self):
        merged = merge_with_defaults({"epochs": 200})
        assert merged["epochs"] == 200

    def test_preserves_defaults(self):
        import config
        merged = merge_with_defaults({})
        assert merged["epochs"] == config.EPOCHS

    def test_passes_through_cli_keys(self):
        merged = merge_with_defaults({"task": "ganea", "model": "hyperbolic"})
        assert merged["task"] == "ganea"
