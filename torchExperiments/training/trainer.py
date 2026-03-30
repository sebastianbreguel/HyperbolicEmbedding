"""Training loops, loss factories, and optimizer factories."""

from __future__ import annotations

import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.utils.data import DataLoader

import config
from manifolds import ManifoldParameter
from optimizer import RiemannianAdam

from .metrics import get_accuracy


def obtain_loss(option: str) -> nn.Module:
    """Return the loss criterion for the given option.

    Args:
        option: ``"cross"`` for CrossEntropyLoss, ``"mse"`` for MSELoss.
    """
    if option == "cross":
        return nn.CrossEntropyLoss()
    elif option == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss option: {option!r}")


def obtain_optimizer(optimizer_option: str, model: nn.Module) -> torch.optim.Optimizer:
    """Build the optimizer, separating ManifoldParameters (no weight decay).

    ManifoldParameters require Riemannian updates and must not receive L2
    regularisation, so they are placed in a separate parameter group with
    weight_decay=0.

    Args:
        optimizer_option: ``"Adam"``, ``"SGD"``, or ``"Radam"`` (Riemannian Adam).
        model: The network whose parameters will be optimised.
    """
    no_decay = ["bias", "scale"]
    weight_decay = 0.0001
    euclidean_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and not any(nd in n for nd in no_decay)
        and not isinstance(p, ManifoldParameter)
    ]
    manifold_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and (any(nd in n for nd in no_decay) or isinstance(p, ManifoldParameter))
    ]
    grouped = [
        {"params": euclidean_params, "weight_decay": weight_decay},
        {"params": manifold_params, "weight_decay": 0.0},
    ]

    if optimizer_option == "Adam":
        return torch.optim.Adam(grouped, lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    elif optimizer_option == "SGD":
        return torch.optim.SGD(grouped, lr=config.LEARNING_RATE, momentum=0.9)
    elif optimizer_option == "Radam":
        return RiemannianAdam(grouped, lr=config.LEARNING_RATE, stabilize=10, betas=(0.9, 0.999))
    raise ValueError(f"Unknown optimizer: {optimizer_option!r}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one epoch of training.

    Returns:
        Total training loss for the epoch (sum over batches).
    """
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def val_process(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute total validation loss (no gradient updates).

    Returns:
        Total validation loss for the epoch (sum over batches).
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            total_loss += criterion(model(X_val), y_val).item()
    model.train()
    return total_loss


def run_model(
    model: nn.Module,
    device: torch.device,
    loss: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    y_test: object,
) -> None:
    """Full training loop for Ganea/Mircea tasks.

    Logs per-epoch metrics to stdout and to W&B.
    """
    for e in range(1, config.EPOCHS + 1):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = val_process(model, val_loader, criterion, device)

        train_acc = get_accuracy(loss, y_test, model, train_loader, device)
        val_acc = get_accuracy(loss, y_test, model, val_loader, device)
        test_acc = get_accuracy(loss, y_test, model, test_loader, device)

        metrics = {
            "epoch": e,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
        }
        wandb.log(metrics)
        print(
            f"Epoch {e:03}: "
            f"Train Loss {metrics['train_loss']:.3f} | "
            f"Val Loss {metrics['val_loss']:.3f} | "
            f"Train Acc {train_acc:.3f} | "
            f"Val Acc {val_acc:.3f} | "
            f"Test Acc {test_acc:.3f}"
        )


def run_MNIST(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Training loop for the MNIST task with UMAP-reduced inputs."""
    for epoch in range(config.EPOCHS):
        model.train()
        partial = torch.tensor(0)
        total_partial = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(-1, config.DIMENSIONS)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            partial += (predicted == labels).sum()
            total_partial += labels.size(0)

        correct = torch.tensor(0)
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.view(-1, config.DIMENSIONS)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

        train_acc = (partial / total_partial * 100).item()
        test_acc = (100 * correct / total).item()
        epoch_loss = loss.item()

        wandb.log({"epoch": epoch, "train_acc": train_acc, "test_acc": test_acc, "loss": epoch_loss})
        print(f"Epoch {epoch}: Loss {loss.item():.4f} | Train Acc {train_acc:.2f}% | Test Acc {test_acc:.2f}%")
        model.train()
