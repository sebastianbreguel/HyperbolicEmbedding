from model_data import get_data, get_model
import torch
from parameters import *
import numpy as np
import torch.nn as nn
from NNs.Optimizer import RiemannianAdam
from geoopt import ManifoldParameter
from utils import generate_data
import os

if "__main__" == __name__:
    generate_data()

    torch.manual_seed(18625541)

    train_loader, val_loader, test_loader, y_test = get_data()

    ##########################
    #####   Check for GPU
    ##########################

    option = int(input("Enter 1 for Euclidean, 2 for Hyperbolic: "))

    device = torch.device("cpu")
    model = get_model(option).to(device)

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

    print(optimizer_grouped_parameters[0], optimizer_grouped_parameters[1])
    print(optimizer_grouped_parameters)
    print(model)


    optimizer_option = int(input("Enter 1 for Adam, 2 for RiemannianAdam: "))

    if optimizer_option == 1:
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=LEARNING_RATE)

    elif optimizer_option == 2:
        optimizer = RiemannianAdam(
            optimizer_grouped_parameters, lr=LEARNING_RATE, stabilize=10
        )

    ##########################
    #####  Train Model
    ##########################

    # loss_stats = {"train": [], "val": []}
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
            train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))

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
                # print(y_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))

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

    y_pred_list = np.array(y_pred_list)

    ##########################
    #####   Evaluate Model
    ##########################
    wrong = [0] * 4
    error_7 = 0
    error_5 = 0
    error_3 = 0
    error_1 = 0

    for i in range(len(y_pred_list)):
        if round(y_pred_list[i], 0) != y_test[i]:
            # print("Predicted: ", round(y_pred_list[i],0),"-",y_pred_list[i], "Actual: ", y_test[i])
            wrong[0] += 1

        if round(y_pred_list[i], 1) != y_test[i]:
            # print("Predicted: ", round(y_pred_list[i],1),"-",y_pred_list[i], "Actual: ", y_test[i])
            wrong[1] += 1

        if round(y_pred_list[i], 2) != y_test[i]:
            # print("Predicted: ", round(y_pred_list[i],2),"-",y_pred_list[i], "Actual: ", y_test[i])
            wrong[2] += 1

        if round(y_pred_list[i], 3) != y_test[i]:
            # print("Predicted: ", round(y_pred_list[i],3),"-",y_pred_list[i], "Actual: ", y_test[i])
            wrong[3] += 1
        diff = round(y_pred_list[i], 0) - y_test[i]

        if diff > 1:
            # print(
            #     "Predicted: ",
            #     round(y_pred_list[i], 0),
            #     "-",
            #     y_pred_list[i],
            #     "Actual: ",
            #     y_test[i],
            # )
            error_1 += 1
            if diff > 3:
                error_3 += 1
                if diff > 5:
                    error_5 += 1
                    if diff > 7:
                        error_7 += 1

    print(
        wrong,
        len(y_pred_list),
        error_1,
        error_3,
        error_5,
        error_7,
    )
