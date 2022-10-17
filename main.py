from utils.model_data import get_data, get_model, get_accuracy, get_info
import torch
from utils.parameters import EPOCHS, LEARNING_RATE, EPS, SEED
import torch.nn as nn
import torch.nn.functional as F

from Optimizer import RiemannianAdam
from manifolds.base import ManifoldParameter

# from geoopt.optim import RiemannianAdam
from utils import generate_data
import argparse


def train_eval(
    option_model: str,
    optimizer_option: str,
    dataset: int,
    loss: str,
    replace,
) -> None:

    train_loader, val_loader, test_loader, y_test = get_data(dataset, replace)
    print(y_test)

    ##########################
    #####    MODEL
    ##########################

    device = torch.device("cpu")
    model = get_model(option_model, dataset).to(device)
    # use all cpu cores for torch

    # Loss Function
    if loss == "cross":
        criterion = nn.CrossEntropyLoss()
    # loss function
    elif loss == "mse":
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

    elif optimizer_option == "Radam":
        optimizer = RiemannianAdam(
            optimizer_grouped_parameters,
            lr=LEARNING_RATE,
            stabilize=10,
            betas=(0.9, 0.999),
        )

    # elif optimizer_option == "RSGD":
    #     optimizer = RiemannianSGD(
    #         optimizer_grouped_parameters,
    #         lr=LEARNING_RATE,
    #         stabilize=10,
    #         momentum=0.9,
    #     )

    print(f"Running {option_model} Model - {optimizer_option} Optimizer", LEARNING_RATE)
    print(model, device)

    ##########################
    #####  Train Model
    ##########################

    # torch.autograd.set_detect_anomaly(True)

    train_losses = []
    val_losses = []
    train_accuracy = []
    test_accuracy = []
    for e in range(1, EPOCHS + 1):
        # TRAINING
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = (
                X_train_batch.to(device),
                y_train_batch.to(device),
            )
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)
            # print("holaaaaa\n\na")
            train_loss = criterion(y_train_pred, y_train_batch)
            # print(y_train_pred,"wewewewe\n\na")
            train_loss.backward()
            # print("wwwwwwwww\n\na")

            optimizer.step()
            # print("xxxxx\n\na")
            train_epoch_loss += train_loss.item()
            # calculate the train accuracy

        acc1 = get_accuracy(loss, y_test, model, train_loader, device)
        train_accuracy.append(acc1)

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = (
                    X_val_batch.to(device),
                    y_val_batch.to(device),
                )
                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()

            # obtain the accuracy
            acc = get_accuracy(loss, y_test, model, test_loader, device)
            test_accuracy.append(acc)
            acc2 = get_accuracy(loss, y_test, model, val_loader, device)

        print(
            f"Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.3f} | Val Loss: {val_epoch_loss/len(val_loader):.3f} | Train accuracy:{acc1:.3f} | Val Accuracy: {acc2:.3f} | Test Accuracy: {acc:.3f}"
        )
        train_losses.append(train_epoch_loss / len(train_loader))
        val_losses.append(val_epoch_loss / len(val_loader))

    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_list.append(y_test_pred.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    info = get_info(loss, y_test, y_pred_list, model, test_loader)
    final = info + train_losses + val_losses + train_accuracy + test_accuracy
    return final


if "__main__" == __name__:
    # seed torch
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()

    parser.add_argument("--gen_data", action="store_true", help="Generate data")
    parser.add_argument(
        "--make_train_eval", action="store_true", help="Train and evaluate model"
    )
    parser.add_argument(
        "--create_folder", action="store_true", help="Create data folder"
    )
    parser.add_argument("--model", action="store", help="Model to use")
    parser.add_argument("--optimizer", action="store", help="Optimizer to use")
    parser.add_argument("--replace", type=float, help="", default=0.5)
    parser.add_argument("--dataset", type=int, help="", default=10)
    parser.add_argument("--loss", action="store", help="Loss to use")
    parser.add_argument("--task", action="store", help="task to use")

    results = parser.parse_args()
    gen_data = results.gen_data
    make_train_eval = results.make_train_eval
    create_folder = results.create_folder
    model = results.model
    optimizer = results.optimizer
    replace = results.replace
    dataset = results.dataset
    loss = results.loss
    task = results.task

    if gen_data:
        print("Generating data")
        generate_data(create_folder, replace, dataset, task)

    if make_train_eval:
        print("Training and evaluating model")
        train_eval(model, optimizer, dataset, loss, replace)
