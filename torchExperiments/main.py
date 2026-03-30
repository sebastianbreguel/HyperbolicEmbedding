"""Entry point for training and evaluating hyperbolic vs. Euclidean models.

Usage
-----
    python main.py --model hyperbolic --task ganea --loss cross \\
                   --dataset 10 --optimizer Radam --runs 10

Each ``--runs`` invocation is tracked as a separate W&B run, making it easy
to aggregate statistics across multiple seeds in the W&B dashboard.
"""

from __future__ import annotations

import argparse

import torch
import wandb
from torch import Tensor

import config
from training import (
    get_data,
    get_metrics,
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
) -> None:
    """Build model, load data, and run one full training + evaluation cycle.

    Args:
        option_model: ``"euclidean"`` or ``"hyperbolic"``.
        optimizer_option: ``"Adam"``, ``"SGD"``, or ``"Radam"``.
        dataset: Prefix length (10, 30, 50) for ganea task; 0 for mircea.
        loss: ``"cross"`` (classification) or ``"mse"`` (regression).
        replace: Fraction of characters replaced in the ganea prefix task.
        task: ``"ganea"``, ``"mircea"``, or ``"MNIST"``.
    """
    if task == "MNIST":
        train_loader, test_loader = getMNIST()
    else:
        train_loader, val_loader, test_loader, y_test = get_data(dataset, replace, task)

    device = torch.device("cpu")
    model = get_model(option_model, dataset, task).to(device)
    criterion = obtain_loss(loss)
    optimizer = obtain_optimizer(optimizer_option, model)

    print(f"Running {option_model} model | {optimizer_option} optimizer | lr={config.LEARNING_RATE}")
    wandb.watch(model, log="gradients", log_freq=10)

    if task == "MNIST":
        run_MNIST(model, device, train_loader, test_loader, criterion, optimizer)
    else:
        run_model(model, device, loss, train_loader, val_loader, test_loader, criterion, optimizer, y_test)


if "__main__" == __name__:
    parser = argparse.ArgumentParser(description="Train hyperbolic / Euclidean neural networks.")
    parser.add_argument("--model", choices=["euclidean", "hyperbolic"], required=True)
    parser.add_argument("--optimizer", choices=["Adam", "SGD", "Radam"], default="Adam")
    parser.add_argument("--task", choices=["ganea", "mircea", "MNIST"], required=True)
    parser.add_argument("--loss", choices=["cross", "mse"], required=True)
    parser.add_argument("--dataset", type=int, default=10, help="Prefix length (ganea) or 0 (mircea)")
    parser.add_argument("--replace", type=float, default=0.5, help="Noise fraction for ganea prefix")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeated independent runs")
    parser.add_argument("--wandb_project", type=str, default="hyperbolic-embedding")
    parser.add_argument("--debug", action="store_true", help="Enable autograd anomaly detection (slow)")

    args = parser.parse_args()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    for i in range(args.runs):
        print(f"=== Run {i + 1}/{args.runs} ===")
        wandb.init(
            project=args.wandb_project,
            name=f"{args.model}_{args.task}_dataset{args.dataset}_run{i}",
            config={
                "model": args.model,
                "optimizer": args.optimizer,
                "task": args.task,
                "loss": args.loss,
                "dataset": args.dataset,
                "replace": args.replace,
                "epochs": config.EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "run_index": i,
            },
        )
        train_eval(args.model, args.optimizer, args.dataset, args.loss, args.replace, args.task)
        wandb.finish()
