"""Evaluation metrics for classification and regression tasks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import Tensor
from torch.utils.data import DataLoader

from config import NM


def get_accuracy(loss: str, y_test: np.ndarray, model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute accuracy (cross-entropy) or normalised L2 error (MSE) on a data loader.

    For cross-entropy: predictions come from argmax of model output; labels are
    decoded from one-hot encoding.
    For MSE: raw model outputs are compared against true continuous labels from
    the loader (not the hardcoded y_test, which is only used as a legacy argument).

    Args:
        loss: ``"cross"`` for classification accuracy, ``"mse"`` for regression error.
        y_test: Unused for cross-entropy; kept for API compatibility.
        model: The network to evaluate.
        loader: DataLoader for the split to evaluate (train / val / test).
        device: Torch device.

    Returns:
        Accuracy in [0, 1] for cross-entropy, or normalised L2 error for MSE.
    """
    was_training = model.training
    model.eval()
    y_pred: list = []
    y_true: list = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)

            if loss == "cross":
                y_pred.extend((torch.max(torch.exp(output), 1)[1]).data.cpu().numpy())
                y_true.extend(labels.data.cpu().numpy())
            else:  # mse — accumulate raw predictions and true labels from loader
                y_pred.extend(output.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

    if was_training:
        model.train()

    if loss == "cross":
        decoded = [int(np.where(i == 1)[0][0]) for i in y_true]
        return float(accuracy_score(decoded, y_pred))
    else:
        y_pred_arr = np.array(y_pred)
        y_true_arr = np.array(y_true)
        return round(float(np.linalg.norm(y_pred_arr - y_true_arr) / (0.2 * NM)), 4)


def get_metrics(
    loss: str,
    y_test: np.ndarray,
    y_pred_list: np.ndarray,
    model: nn.Module,
    test_loader: DataLoader,
) -> list[float] | float:
    """Compute full metrics on the test set (accuracy, F1, precision, recall).

    Args:
        loss: ``"cross"`` or ``"mse"``.
        y_test: Ground truth labels (used only for MSE normalisation).
        y_pred_list: Pre-computed predictions (used only for MSE).
        model: The network to evaluate.
        test_loader: DataLoader for the test set.

    Returns:
        For cross-entropy: [accuracy, *f1_per_class, *precision_per_class, *recall_per_class].
        For MSE: normalised L2 error (float).
    """
    was_training = model.training
    model.eval()

    result: list[float] | float

    if loss == "cross":
        y_pred: list = []
        y_true: list = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                output = model(inputs)
                y_pred.extend((torch.max(torch.exp(output), 1)[1]).data.cpu().numpy())
                y_true.extend(labels.data.cpu().numpy())

        decoded = [int(np.where(i == 1)[0][0]) for i in y_true]
        acc = accuracy_score(decoded, y_pred)
        print(f"Accuracy on {len(y_test)} test samples: {acc:.4f}", end=" | ")

        result = [round(acc, 3)]
        result += f1_score(decoded, y_pred, average=None).tolist()
        result += precision_score(decoded, y_pred, average=None, zero_division=1).tolist()
        result += recall_score(decoded, y_pred, average=None).tolist()

    elif loss == "mse":
        result = round(float(np.linalg.norm(y_pred_list - y_test) / (0.2 * NM)), 4)
        print(f"Loss on Test Data: {result}")

    if was_training:
        model.train()
    return result
