import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sn

# Parameters
from utils.parameters import (
    URL,
    URL_PREFIX_10,
    URL_PREFIX_30,
    URL_PREFIX_50,
    IN_FEATURES,
    BATCH_SIZE,
    NM,
    LARGE,
    SEED,
)

# Custom NN and Manifolds
from Manifolds.euclidean import Euclidean
from Manifolds.poincare import PoincareBall
from NNs import HNNLayer


def get_model(option: str, dataset: int, hidden: int) -> torch.nn.Module:
    inputs = 20 + int(dataset * 0.2)
    outputs = 2

    if dataset == 0:
        inputs = 140
        outputs = 6

    manifold = None
    inputs *= LARGE

    c = 0
    if option == "euclidean":
        manifold = Euclidean(c)
    elif option == "hyperbolic":
        c = 1
        manifold = PoincareBall(c)

    model = HNNLayer(manifold, inputs, outputs, c, 1, hidden)

    return model


def get_data(dataset, replace) -> tuple:
    np.random.seed(SEED)

    if dataset == 10:
        url = URL_PREFIX_10 + "_" + str(replace) + ".csv"

    elif dataset == 30:
        url = URL_PREFIX_30 + "_" + str(replace) + ".csv"

    elif dataset == 50:
        url = URL_PREFIX_50 + "_" + str(replace) + ".csv"

    elif dataset == 0:
        url = URL

    df = pd.read_csv(url, header=0)
    df = df.drop(df.columns[0], axis=1)

    if dataset == 0:
        X = df.iloc[:, :IN_FEATURES]
        y = df.iloc[:, IN_FEATURES:]

    else:
        X = df.iloc[:, 2:]
        # # columns isPrefix and isNotPrefix
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


def get_info(loss, y_test, y_pred_list, model, test_loader):
    if loss == "cross":

        y_pred = []
        y_true = []

        # iterate over test data
        for inputs, labels in test_loader:
            output = model(inputs)  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

        # constant for classes
        new = []
        # Build confusion matrix
        for i in y_true:
            # obtain the index ob the max
            index = np.where(i == 1)[0][0]
            new.append(index)

        y_true = new
        print(
            f"Accuracy on the {len(y_test)} test data: {accuracy_score(y_true, y_pred)} %",
            end=" | ",
        )

        list_info = [round(accuracy_score(y_true, y_pred), 3)]
        # add to list info each data
        list_info += f1_score(y_true, y_pred, average=None).tolist()
        list_info += precision_score(
            y_true, y_pred, average=None, zero_division=1
        ).tolist()
        list_info += recall_score(y_true, y_pred, average=None).tolist()

        return list_info

    elif loss == "mse":
        print(
            f"Loss on Test Data: {round(np.linalg.norm(y_pred_list-y_test)/(0.2 * NM), 4)}"
        )
        return round(np.linalg.norm(y_pred_list - y_test) / (0.2 * NM), 4)


def get_accuracy(loss, y_test, model, test_loader):

    y_pred = []
    y_true = []

    # iterate over test data

    with torch.no_grad():
        for inputs, labels in test_loader:
            output = model(inputs)  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

    if loss == "cross":
        new = []
        # Build confusion matrix
        for i in y_true:
            # obtain the index ob the max
            index = np.where(i == 1)[0][0]
            new.append(index)

        y_true = new

        return accuracy_score(y_true, y_pred)
    else:
        return round(np.linalg.norm(y_pred - y_test) / (0.2 * NM), 4)
