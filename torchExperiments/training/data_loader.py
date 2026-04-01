"""Model factory and DataLoader factories for all experimental tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

import config
from manifolds import Euclidean, PoincareBall
from models import HNN


def get_model(option: str, dataset: int, task: str) -> torch.nn.Module:
    """Build and return the appropriate HNN model for a given task and geometry.

    Args:
        option: ``"euclidean"`` or ``"hyperbolic"``.
        dataset: Prefix length (10, 30, 50) for the ganea task; ignored for others.
        task: ``"ganea"``, ``"mircea"``, or ``"MNIST"``.

    Returns:
        An HNN instance configured for the requested manifold and input size.
    """
    if task == "MNIST":
        inputs = config.DIMENSIONS
        outputs = 10
    elif task == "ganea":
        inputs = 20 * config.LARGE + int(dataset * 0.2)
        print(dataset, inputs)
        outputs = 2
    elif task == "mircea":
        inputs = 140
        outputs = 6
    else:
        raise ValueError(f"Unknown task: {task!r}")

    c = 0
    if option == "euclidean":
        manifold = Euclidean(c)
    elif option == "hyperbolic":
        c = 1
        manifold = PoincareBall(c)
    else:
        raise ValueError(f"Unknown model option: {option!r}")

    return HNN(manifold, inputs, outputs, c, config.HIDDEN_SIZE)


def get_data(
    dataset: int,
    replace: float,
    task: str,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Load a CSV dataset, split, scale, and return DataLoaders + test labels.

    Args:
        dataset: Prefix length (10, 30, 50) for ganea; ignored for mircea.
        replace: Noise fraction used when generating the ganea dataset (selects file).
        task: ``"ganea"`` or ``"mircea"``.

    Returns:
        A 4-tuple ``(train_loader, val_loader, test_loader, y_test)`` where
        ``y_test`` is the raw numpy array of test labels (used for metrics).
    """
    if task == "mircea":
        url = config.URL
    else:
        if dataset == 10:
            url = config.URL_PREFIX_10 + "_" + str(replace) + ".csv"
        elif dataset == 30:
            url = config.URL_PREFIX_30 + "_" + str(replace) + ".csv"
        elif dataset == 50:
            url = config.URL_PREFIX_50 + "_" + str(replace) + ".csv"
        else:
            raise ValueError(f"Unknown dataset prefix length: {dataset}")

    df = pd.read_csv(url, header=0)
    df = df.drop(df.columns[0], axis=1).sample(frac=1).reset_index(drop=True)

    if task == "mircea":
        X = df.iloc[:, :config.IN_FEATURES]
        y = df.iloc[:, config.IN_FEATURES:]
    else:
        X = df.iloc[:, 2:]
        y = df[["isPrefix", "isNotPrefix"]].iloc[:, :]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, random_state=21
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    if task == "ganea":
        # CrossEntropyLoss expects class indices (long), not one-hot
        y_train = np.argmax(y_train, axis=1).astype(np.int64)
        y_val = np.argmax(y_val, axis=1).astype(np.int64)
        y_test = np.argmax(y_test, axis=1).astype(np.int64)
    else:
        # mircea: regression with float targets
        y_train = y_train.astype(float)
        y_val = y_val.astype(float)
        y_test = y_test.astype(float)

    class RegressionDataset(Dataset):
        def __init__(self, X_data: np.ndarray, y_data: np.ndarray) -> None:
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.X_data[index], self.y_data[index]

        def __len__(self) -> int:
            return len(self.X_data)

    train_dataset = RegressionDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train) if task == "ganea" else torch.from_numpy(y_train).float()
    )
    val_dataset = RegressionDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val) if task == "ganea" else torch.from_numpy(y_val).float()
    )
    test_dataset = RegressionDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test) if task == "ganea" else torch.from_numpy(y_test).float()
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, y_test


MNIST_DATA_DIR = "data"


def getMNIST() -> tuple[DataLoader, DataLoader]:
    """Load pre-computed UMAP-reduced MNIST embeddings from CSV files.

    Expects ``{MNIST_DATA_DIR}/MNIST/train.csv`` and ``{MNIST_DATA_DIR}/MNIST/test.csv``.
    Generate them first with ``python data.py --task MNIST``.

    Returns:
        ``(train_loader, test_loader)`` with the UMAP-reduced embeddings.
    """
    from torch.utils.data import TensorDataset

    train_df = pd.read_csv(f"{MNIST_DATA_DIR}/MNIST/train.csv", header=0)
    test_df = pd.read_csv(f"{MNIST_DATA_DIR}/MNIST/test.csv", header=0)

    X_train = torch.from_numpy(train_df.iloc[:, 1:].to_numpy()).float()
    y_train = torch.from_numpy(train_df.iloc[:, 0].to_numpy()).long()
    X_test = torch.from_numpy(test_df.iloc[:, 1:].to_numpy()).float()
    y_test = torch.from_numpy(test_df.iloc[:, 0].to_numpy()).long()

    train_loader = DataLoader(
        dataset=TensorDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=TensorDataset(X_test, y_test),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, test_loader
