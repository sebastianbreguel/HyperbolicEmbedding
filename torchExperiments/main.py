"""Entry point for training and evaluating hyperbolic vs. Euclidean models.

Usage
-----
    # YAML config mode (recommended)
    python main.py --config experiments/sweep.yaml
    python main.py --config experiments/sweep.yaml --dry-run
    python main.py --config experiments/sweep.yaml --filter model=hyperbolic
    python main.py --config experiments/sweep.yaml --experiment 3

    # Legacy CLI mode
    python main.py --model hyperbolic --task ganea --loss cross \\
                   --dataset 10 --optimizer Radam --runs 10

Each run is tracked as a separate W&B run, making it easy to aggregate
statistics across multiple seeds in the W&B dashboard.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import wandb

import config


def resolve_device(requested: str = "auto") -> torch.device:
    """Return the best available torch device.

    Args:
        requested: ``"auto"`` (detect best), ``"cpu"``, ``"cuda"``, or ``"mps"``.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


from training import (
    get_data,
    get_model,
    getMNIST,
    obtain_loss,
    obtain_optimizer,
    run_MNIST,
    run_model,
)


def train_eval(
    option_model: str,
    optimizer_option: str,
    dataset: int,
    loss: str,
    replace: float,
    task: str,
    device: torch.device | None = None,
) -> None:
    """Build model, load data, and run one full training + evaluation cycle.

    Args:
        option_model: ``"euclidean"`` or ``"hyperbolic"``.
        optimizer_option: ``"Adam"``, ``"SGD"``, or ``"Radam"``.
        dataset: Prefix length (10, 30, 50) for ganea task; 0 for mircea.
        loss: ``"cross"`` (classification) or ``"mse"`` (regression).
        replace: Fraction of characters replaced in the ganea prefix task.
        task: ``"ganea"``, ``"mircea"``, or ``"MNIST"``.
        device: Torch device. Defaults to auto-detection.
    """
    if task == "MNIST":
        train_loader, test_loader = getMNIST()
    else:
        train_loader, val_loader, test_loader, y_test = get_data(dataset, replace, task)

    if device is None:
        device = resolve_device()
    model = get_model(option_model, dataset, task).to(device)
    criterion = obtain_loss(loss)
    optimizer = obtain_optimizer(optimizer_option, model)

    print(f"Running {option_model} model | {optimizer_option} optimizer | lr={config.LEARNING_RATE}")
    wandb.watch(model, log="gradients", log_freq=10)

    if task == "MNIST":
        run_MNIST(model, device, train_loader, test_loader, criterion, optimizer)
    else:
        run_model(model, device, loss, train_loader, val_loader, test_loader, criterion, optimizer, y_test)


def _print_sweep_table(experiments: list[dict], runs: int) -> None:
    """Print a table of sweep combinations for --dry-run."""
    sweep_keys = set()
    for exp in experiments:
        sweep_keys.update(exp.keys())
    varying = [k for k in sorted(sweep_keys) if len({str(e.get(k)) for e in experiments}) > 1]
    if not varying:
        varying = ["model", "optimizer", "task"]
    for i, exp in enumerate(experiments):
        parts = [f"{k}={exp.get(k)}" for k in varying]
        print(f"Experiment {i:3d}: {'  '.join(parts)}")
    print(f"\nTotal: {len(experiments)} experiments × {runs} runs = {len(experiments) * runs} runs")


def _run_experiment(exp_config: dict, run_index: int, device: torch.device | None = None) -> None:
    """Run a single experiment with the given config."""
    effective_seed = exp_config.get("seed", 0) + run_index
    torch.manual_seed(effective_seed)
    np.random.seed(effective_seed)
    config.apply_config(exp_config)
    task = exp_config["task"]
    model_name = exp_config["model"]
    optimizer_name = exp_config["optimizer"]
    loss = exp_config["loss"]
    dataset = exp_config.get("dataset", 10)
    replace = exp_config.get("replace", 0.5)
    wandb_project = exp_config.get("wandb_project", "hyperbolic-embedding")
    wandb.init(
        project=wandb_project,
        name=f"{model_name}_{task}_ds{dataset}_run{run_index}",
        config={**exp_config, "run_index": run_index, "effective_seed": effective_seed, "device": str(device)},
    )
    train_eval(model_name, optimizer_name, dataset, loss, replace, task, device)
    wandb.finish()


if "__main__" == __name__:
    parser = argparse.ArgumentParser(description="Train hyperbolic / Euclidean neural networks.")
    # YAML config mode
    parser.add_argument("--config", type=str, default=None, help="Path to YAML experiment config")
    parser.add_argument("--filter", type=str, default=None, help="Filter sweep: key=val,key=val")
    parser.add_argument("--experiment", type=int, default=None, help="Run single experiment by index")
    parser.add_argument("--dry-run", action="store_true", help="List experiments without running")
    # Legacy CLI mode
    parser.add_argument("--model", choices=["euclidean", "hyperbolic"], default=None)
    parser.add_argument("--optimizer", choices=["Adam", "SGD", "Radam"], default="Adam")
    parser.add_argument("--task", choices=["ganea", "mircea", "MNIST"], default=None)
    parser.add_argument("--loss", choices=["cross", "mse"], default=None)
    parser.add_argument("--dataset", type=int, default=10, help="Prefix length (ganea)")
    parser.add_argument("--replace", type=float, default=0.5, help="Noise fraction for ganea")
    parser.add_argument("--runs", type=int, default=None, help="Independent runs per experiment")
    parser.add_argument("--wandb_project", type=str, default="hyperbolic-embedding")
    parser.add_argument("--debug", action="store_true", help="Enable autograd anomaly detection")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps")
    args = parser.parse_args()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    if args.config:
        from experiment_config import apply_filters, expand_sweep, load_config, merge_with_defaults
        raw = load_config(args.config)
        experiments = expand_sweep(raw)
        if args.filter:
            experiments = apply_filters(experiments, args.filter)
        if args.experiment is not None:
            if args.experiment < 0 or args.experiment >= len(experiments):
                parser.error(f"--experiment {args.experiment} out of range (0-{len(experiments) - 1})")
            experiments = [experiments[args.experiment]]
        runs = args.runs if args.runs is not None else raw.get("runs", 10)
        experiments = [merge_with_defaults(exp) for exp in experiments]
        if args.dry_run:
            _print_sweep_table(experiments, runs)
        else:
            device_choice = args.device if args.device != "auto" else raw.get("device", "auto")
            device = resolve_device(device_choice)
            print(f"Device: {device}")
            total = len(experiments) * runs
            run_num = 0
            for exp_idx, exp in enumerate(experiments):
                for r in range(runs):
                    run_num += 1
                    print(f"=== [{run_num}/{total}] Experiment {exp_idx}, Run {r + 1}/{runs} ===")
                    _run_experiment(exp, r, device)
    else:
        if not args.model or not args.task or not args.loss:
            parser.error("--model, --task, and --loss are required (or use --config)")
        device = resolve_device(args.device)
        print(f"Device: {device}")
        num_runs = args.runs if args.runs is not None else 10
        for i in range(num_runs):
            effective_seed = config.SEED + i
            torch.manual_seed(effective_seed)
            np.random.seed(effective_seed)
            print(f"=== Run {i + 1}/{num_runs} (seed={effective_seed}) ===")
            wandb.init(
                project=args.wandb_project,
                name=f"{args.model}_{args.task}_dataset{args.dataset}_run{i}",
                config={
                    "model": args.model, "optimizer": args.optimizer, "task": args.task,
                    "loss": args.loss, "dataset": args.dataset, "replace": args.replace,
                    "epochs": config.EPOCHS, "batch_size": config.BATCH_SIZE,
                    "learning_rate": config.LEARNING_RATE, "run_index": i,
                },
            )
            train_eval(args.model, args.optimizer, args.dataset, args.loss, args.replace, args.task, device)
            wandb.finish()
