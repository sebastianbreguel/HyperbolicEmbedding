from model_data import get_data, get_model
import torch
from parameters import *
import numpy as np
import torch.nn as nn
from Optimizer import RiemannianAdam
from geoopt import ManifoldParameter
from utils import generate_data
import argparse


def train_eval(option_model, optimizer_option):

    train_loader, val_loader, test_loader, y_test = get_data()

    ##########################
    #####    MODEL
    ##########################

    device = torch.device("cpu")
    model = get_model(option_model).to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

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
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=LEARNING_RATE)

    elif optimizer_option == "RiemannianAdam":
        optimizer = RiemannianAdam(
            optimizer_grouped_parameters, lr=LEARNING_RATE, stabilize=10
        )
    print("\n" + "#" * 25)
    print(f"Running {option_model} Model - {optimizer_option} Optimizer")
    print("#" * 25, "\n")
    # print(optimizer_grouped_parameters)
    # print(model)

    ##########################
    #####  Train Model
    ##########################

    torch.autograd.set_detect_anomaly(True)

    print("Begin training.")
    for e in range(1, EPOCHS + 1):

        # TRAINING
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(
                device
            )
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)

            train_loss.backward()

            optimizer.step()
            train_epoch_loss += train_loss.item()
        #     break
        # break
        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(
                    device
                )
                # print(X_val_batch)
                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()

        # loss_stats['train'].append(train_epoch_loss/len(train_loader))
        # loss_stats['val'].append(val_epoch_loss/len(val_loader))

        print(
            f"Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}"
        )

    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_list.append(y_test_pred.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    corrects = 0
    for i in range(len(y_pred_list)):

        max_value = max(y_pred_list[i])
        index_max = y_pred_list[i].index(max_value)
        max_real = max(y_test[i])

        index_max_real = np.where(y_test[i] == max_real)
        print(f"Valor:        Predicted: {index_max}, Real: {index_max_real[0][0]}")
        print(f"Probabilidad: Predicted: {y_pred_list[i]}")
        if index_max == index_max_real[0][0]:
            print("Correct")
            corrects += 1
        else:
            print("Incorrect")

    print(f"Accuracy: {corrects/len(y_pred_list)}")


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

    results = parser.parse_args()
    gen_data = results.gen_data
    make_train_eval = results.make_train_eval
    delete_folder = results.delete_folder
    create_folder = results.create_folder
    model = results.model
    optimizer = results.optimizer

    if gen_data:
        print("\n" + "#" * 21)
        print("## GENERATING DATA ##")
        print("#" * 21)
        generate_data(delete_folder, create_folder)

    torch.manual_seed(18625541)

    if make_train_eval:
        train_eval(model, optimizer)
