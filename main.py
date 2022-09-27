from tkinter import E
from model_data import get_data, get_model
import torch
from parameters import EPOCHS, LEARNING_RATE, EPS
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# from Optimizer import RiemannianAdam
from Manifolds.base import ManifoldParameter

from geoopt.optim import RiemannianAdam, RiemannianSGD
from utils import generate_data
import argparse
from time import perf_counter
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from time import perf_counter, time
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def train_eval(option_model: str, optimizer_option: str, dataset: int) -> None:

    train_loader, val_loader, test_loader, y_test = get_data(dataset)

    ##########################
    #####    MODEL
    ##########################

    device = torch.device("cpu")
    model = get_model(option_model, dataset).to(device)
    # use all cpu cores for torch

    # Loss Function
    # loss function
    criterion = nn.MSELoss()

    no_decay = ["bias", "scale"]
    weight_decay = 0.0001
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and not any(nd in n for nd in no_decay)
                and not isinstance(p, ManifoldParameter)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and any(nd in n for nd in no_decay)
                or isinstance(p, ManifoldParameter)
            ],
            "weight_decay": 0.0,
        },
    ]

    if optimizer_option == "Adam":
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=LEARNING_RATE, betas=(0.9, 0.999)
        )

    elif optimizer_option == "SGD":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters, lr=LEARNING_RATE, momentum=0.9
        )

    # elif optimizer_option == "RSGD":
    #     optimizer = RiemannianSGD(
    #         optimizer_grouped_parameters, lr=LEARNING_RATE, momentum=0.9
    #     )

    elif optimizer_option == "Radam":
        optimizer = RiemannianAdam(
            optimizer_grouped_parameters,
            lr=LEARNING_RATE,
            stabilize=10,
            betas=(0.9, 0.999),
        )
    print(f"Running {option_model} Model - {optimizer_option} Optimizer", LEARNING_RATE)
    # print(model)

    ##########################
    #####  Train Model
    ##########################

    torch.autograd.set_detect_anomaly(True)
    initial = perf_counter()
    # start = initial
    print("Begin training.")
    e = 0
    # start = time()
    while e < EPOCHS:# and time()-start < 60:
        e += 1

        # TRAINING
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(
                device
            )
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)
            # print(y_train_pred)
            train_loss = criterion(y_train_pred, y_train_batch)

            if np.isnan(train_loss.item()):
                # print(train_loss.data)
                train_loss = torch.tensor(EPS, requires_grad=True)
            # print("TRAIN LOSS", y_train_batch, y_train_pred)

            train_loss.backward()

            optimizer.step()
            train_epoch_loss += train_loss.item()

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(
                    device
                )
                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
        if e % 10 == 0 or e ==1:
            print(
                f"Epoch {e+0:03}:\tTrain Loss: {train_epoch_loss/len(train_loader):.4f}\tVal Loss: {val_epoch_loss/len(val_loader):.4f}"
                + f"\tTime: {((perf_counter() - initial)/60):.2f} minutes"
                )

    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_list.append(y_test_pred.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    correct = 0

    for i in range(len(y_pred_list)):
        max_value = max(y_pred_list[i])
        index_max = y_pred_list[i].index(max_value)
        max_real = max(y_test[i])
        index_max_real = np.where(y_test[i] == max_real)[0][0]

        if index_max == index_max_real:
            correct += 1

    print(
        f"Accuracy of the network on the {len(y_test)} test data: {round(100 * correct /len(y_pred_list),3)} %"
    )

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
    # value = 0
    for i in y_true:
        # obtain the index ob the max
        index = np.where(i == 1)[0][0]
        new.append(index)

    y_true = new
    print("Accuracy", accuracy_score(y_true, y_pred))
    print(f1_score(y_true, y_pred, average=None))
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix) * 10,
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f"output-{option_model}.png")
    # print(y_true)
    # for i in y_true:
    #     # obtain the index ob the max
    #     print(i)
    #     index = np.where(i == 1)[0][0]
    #     new.append(index)
    #     # print(i, y_pred[value])
    #     # value+=1

    # y_true = new
    # # print(y_true, y_pred)
    # cf_matrix = confusion_matrix(y_true, y_pred)
    # print(cf_matrix)
    # df_cm = pd.DataFrame(
    #     cf_matrix / np.sum(cf_matrix) * 10,
    #     index=[i for i in classes],
    #     columns=[i for i in classes],
    # )
    # plt.figure(figsize=(12, 7))
    # sn.heatmap(df_cm, annot=True)
    # plt.savefig(f"output-{option_model}.png")


if "__main__" == __name__:

    parser = argparse.ArgumentParser()

    parser.add_argument("--gen_data", action="store_true", help="Generate data")
    parser.add_argument(
        "--make_train_eval", action="store_true", help="Train and evaluate model"
    )
    parser.add_argument(
        "--delete_folder", action="store_true", help="Delete data folder"
    )
    parser.add_argument(
        "--create_folder", action="store_true", help="Create data folder"
    )
    parser.add_argument("--model", action="store", help="Model to use")
    parser.add_argument("--optimizer", action="store", help="Optimizer to use")
    parser.add_argument("--replace", type=bool, help="", default=False)
    parser.add_argument("--dataset", type=int, help="", default=10)

    results = parser.parse_args()
    gen_data = results.gen_data
    make_train_eval = results.make_train_eval
    delete_folder = results.delete_folder
    create_folder = results.create_folder
    model = results.model
    optimizer = results.optimizer
    replace = results.replace
    dataset = results.dataset

    if gen_data:
        print("\n" + "#" * 21)
        print("## GENERATING DATA ##")
        print("#" * 21)
        generate_data(delete_folder, create_folder, replace)

    torch.manual_seed(18625541)


    for dataset in [10, 30, 50]:
        for model in ["euclidean", "hyperbolic"]:
            print("\n\ndataset:",dataset,"model:", model)
            if make_train_eval:
                print()
                train_eval(model, optimizer, dataset)
