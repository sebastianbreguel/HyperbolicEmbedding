import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from manifolds import Euclidean, PoincareBall
from models import HNN
from utils.parameters import (
    URL,
    URL_PREFIX_10,
    URL_PREFIX_30,
    URL_PREFIX_50,
    IN_FEATURES,
    BATCH_SIZE,
    DIMENTIONS,
    LARGE,
    SEED,
)


def get_model(option: str, dataset: int, task: str) -> torch.nn.Module:

    if task == "MNIST":
        inputs = DIMENTIONS
        outputs = 10

    elif task == "ganea":
        inputs = 20 + int(dataset * 0.2)
        inputs *= LARGE
        outputs = 2

    elif task == "mircea":
        inputs = 140
        outputs = 6

    c = 0
    if option == "euclidean":
        manifold = Euclidean(c)
    elif option == "hyperbolic":
        c = 1
        manifold = PoincareBall(c)

    model = HNN(manifold, inputs, outputs, c, 64)

    return model


def get_data(dataset, replace, task) -> tuple:
    np.random.seed(SEED)

    if task == "mircea":
        url = URL

    else:
        if dataset == 10:
            url = URL_PREFIX_10 + "_" + str(replace) + ".csv"

        elif dataset == 30:
            url = URL_PREFIX_30 + "_" + str(replace) + ".csv"

        elif dataset == 50:
            url = URL_PREFIX_50 + "_" + str(replace) + ".csv"

    df = pd.read_csv(url, header=0)
    df = df.drop(df.columns[0], axis=1).sample(frac=1).reset_index(drop=True)

    if task == "mircea":
        X = df.iloc[:, :IN_FEATURES]
        y = df.iloc[:, IN_FEATURES:]

    else:
        X = df.iloc[:, 2:]
        y = df[["isPrefix", "isNotPrefix"]].iloc[:, :]

    ##########################
    #####Train — Validation — Test
    ##########################

    # Train - Test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69
    )
    # Split train into train-val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, random_state=21
    )

    ##########################
    #####Normalize Input
    ##########################

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    ##########################
    #####Convert Output Variable to Float
    ##########################

    y_train, y_test, y_val = (
        y_train.astype(float),
        y_test.astype(float),
        y_val.astype(float),
    )

    ##########################
    #####       Neural Network
    #####       Initialize Dataset
    ##########################

    class RegressionDataset(Dataset):
        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]

        def __len__(self):
            return len(self.X_data)

    train_dataset = RegressionDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )
    val_dataset = RegressionDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    )
    test_dataset = RegressionDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
    )

    ##########################
    #####   Initialize Dataloader
    ##########################

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader, test_loader, y_test


def getMNIST() -> tuple:

    train_dataset = dsets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = dsets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
    )

    X_train = pd.read_csv("data/MNIST/train.csv", header=0).to_numpy()
    train_dataset.data = torch.from_numpy(X_train)

    X_test = pd.read_csv("data/MNIST/test.csv", header=0).to_numpy()
    test_dataset.data = torch.from_numpy(X_test)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, test_loader
