import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn
from utils.data_Params import NM

# Parameters
from parameters import (
    URL,
    URL_PREFIX_10,
    URL_PREFIX_30,
    URL_PREFIX_50,
    IN_FEATURES,
    OUT_FEATURES,
    BATCH_SIZE,
    LARGE,
)

# Custom NN and Manifolds
from Manifolds.euclidean import Euclidean
from Manifolds.poincare import PoincareBall
from NNs import HNNLayer


def get_model(option: str, dataset: int, hidden: int) -> torch.nn.Module:
    inputs = 20 
    outputs = 2

    if dataset == 10:
        inputs += 2

    elif dataset == 30:
        inputs += 6

    elif dataset == 50:
        inputs += 10
    elif dataset == 0:
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


def get_data(dataset) -> tuple:

    if dataset == 10:
        url = URL_PREFIX_10
    elif dataset == 30:
        url = URL_PREFIX_30
    elif dataset == 50:
        url = URL_PREFIX_50
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
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    return train_loader, val_loader, test_loader, y_test

def get_accuracy(loss, y_test, y_pred_list, model, test_loader):
    if loss == "cross":

        # correct = 0
        # for i in range(len(y_pred_list)):
        #     max_value = max(y_pred_list[i])
        #     index_max = y_pred_list[i].index(max_value)
        #     max_real = max(y_test[i])
        #     index_max_real = np.where(y_test[i] == max_real)[0][0]

        #     if index_max == index_max_real:
        #         correct += 1

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
        classes = ("Prefix", "Random")
        # obtain the position of the max
        new = []
        # Build confusion matrix
        for i in y_true:
            # obtain the index ob the max
            index = np.where(i == 1)[0][0]
            new.append(index)

        y_true = new
        # print("Accuracy", accuracy_score(y_true, y_pred))
        print(
            f"Accuracy of the network on the {len(y_test)} test data: {accuracy_score(y_true, y_pred)} %"
        )
        print(f1_score(y_true, y_pred, average=None))
        cf_matrix = confusion_matrix(y_true, y_pred)
        print(cf_matrix)
        df_cm = pd.DataFrame(
            cf_matrix / np.sum(cf_matrix) * 10,
            index=[i for i in classes],
            columns=[i for i in classes],
        )
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(f"output-{loss}.png")
        return accuracy_score(y_true, y_pred)#, f1_score(y_true, y_pred, average=None), cf_matrix
    elif loss == "mse":
        print(f"Loss on Test Data: {round(np.linalg.norm(y_pred_list-y_test)/(0.2 * NM), 4)}")
        return round(np.linalg.norm(y_pred_list-y_test)/(0.2 * NM), 4)
