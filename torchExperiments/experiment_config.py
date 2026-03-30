"""YAML experiment configuration: loading, sweep expansion, and filtering."""
from __future__ import annotations

import itertools
from pathlib import Path

import yaml

import config

VALID_KEYS = {
    "task", "model", "optimizer", "loss", "dataset", "replace",
    "runs", "wandb_project", "debug",
    "epochs", "learning_rate", "batch_size", "hidden_size",
    "use_bias", "seed", "dimensions",
    "sweep",
}

def load_config(path: str) -> dict:
    """Load and validate a YAML experiment config file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    unknown = set(cfg.keys()) - VALID_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}. Valid keys: {sorted(VALID_KEYS - {'sweep'})}")
    if "sweep" in cfg:
        sweep_unknown = set(cfg["sweep"].keys()) - VALID_KEYS
        if sweep_unknown:
            raise ValueError(f"Unknown sweep keys: {sorted(sweep_unknown)}. Valid keys: {sorted(VALID_KEYS - {'sweep'})}")
    return cfg

def expand_sweep(cfg: dict) -> list[dict]:
    """Expand config with optional sweep into list of experiment configs via itertools.product."""
    sweep = cfg.get("sweep")
    base = {k: v for k, v in cfg.items() if k != "sweep"}
    if not sweep:
        return [dict(base)]
    for key, values in sweep.items():
        if not isinstance(values, list):
            raise ValueError(f"Sweep key '{key}' must be a list, got {type(values).__name__}: {values}")
    keys = list(sweep.keys())
    value_lists = [sweep[k] for k in keys]
    experiments = []
    for combo in itertools.product(*value_lists):
        experiment = dict(base)
        for key, value in zip(keys, combo):
            experiment[key] = value
        experiments.append(experiment)
    return experiments

def apply_filters(experiments: list[dict], filter_str: str | None) -> list[dict]:
    """Filter experiments by comma-separated key=value pairs."""
    if not filter_str:
        return experiments
    filters = {}
    for pair in filter_str.split(","):
        key, _, value = pair.partition("=")
        filters[key.strip()] = value.strip()
    filtered = [e for e in experiments if all(str(e.get(k)) == v for k, v in filters.items())]
    if not filtered:
        raise ValueError(f"No experiments match filter '{filter_str}'. Available values: { {k: sorted(set(str(e.get(k)) for e in experiments)) for k in filters} }")
    return filtered

def merge_with_defaults(experiment: dict) -> dict:
    """Merge experiment config with config.py defaults."""
    defaults = config.get_defaults()
    merged = dict(defaults)
    merged.update(experiment)
    return merged
