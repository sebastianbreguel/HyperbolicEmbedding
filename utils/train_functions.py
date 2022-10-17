import torch.nn as nn
from .model_data import get_accuracy
import torch
from utils.parameters import LEARNING_RATE
import torch.nn as nn

from optimizer import RiemannianAdam
from manifolds.base import ManifoldParameter


def obtain_loss(option):

    if option == "cross":
        criterion = nn.CrossEntropyLoss()
    # loss function
    elif option == "mse":
        criterion = nn.MSELoss()

    return criterion


def obtain_optimizer(optimizer_option, model):
    n = nn.MSELoss()

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
    return optimizer


def val_process(model, val_loader, criterion, device):
    model.eval()
    val_epoch_loss = 0
    for X_val_batch, y_val_batch in val_loader:
        X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
        y_val_pred = model(X_val_batch)
        val_loss = criterion(y_val_pred, y_val_batch)
        val_epoch_loss += val_loss.item()

    return val_epoch_loss


def train_model(model, train_loader, criterion, optimizer, device):
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

    return train_epoch_loss
