import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Parameters
from parameters import *

# Custom NN and Manifolds
from Manifolds.euclidean import Euclidean
from Manifolds.poincare import PoincareBall
from NNs import HNNLayer


def get_model(option: str) -> torch.nn.Module:

    manifold = None

    if option == "euclidean":
        manifold = Euclidean()
        c = 0
    elif option == "hyperbolic":
        manifold = PoincareBall()
        c = 1

    model = HNNLayer(manifold, 35, OUT_FEATURES, c, 1)

    return model


def get_data():

    df = pd.read_csv(URL_GANEA_15, header=0)
    df = df.drop(df.columns[0], axis=1)

    ##########################
    #####Create Input and Output Data
    ##########################

    X = df.iloc[:, 2:]
    # columns isPrefix and isNotPrefix
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
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    return train_loader, val_loader, test_loader, y_test
