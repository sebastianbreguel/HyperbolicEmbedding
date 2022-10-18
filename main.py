from utils.model_data import get_data, get_model, get_accuracy, get_info
import torch
from utils.parameters import EPOCHS, LEARNING_RATE, SEED
from utils import generate_data, obtain_loss, obtain_optimizer, train_model, val_process

import argparse


def train_eval(
    option_model: str,
    optimizer_option: str,
    dataset: int,
    loss: str,
    replace,
) -> None:

    train_loader, val_loader, test_loader, y_test = get_data(dataset, replace)

    ##########################
    #####    MODEL
    ##########################

    device = torch.device("cpu")
    model = get_model(option_model, dataset).to(device)
    # use all cpu cores for torch

    # Loss Function
    criterion = obtain_loss(loss)

    # Optimizer
    optimizer = obtain_optimizer(optimizer_option, model)

    print(f"Running {option_model} Model - {optimizer_option} Optimizer", LEARNING_RATE)

    ##########################
    #####  Train Model
    ##########################

    torch.autograd.set_detect_anomaly(True)
    for e in range(1, EPOCHS + 1):
        # TRAINING
        train_epoch_loss = train_model(
            model, train_loader, criterion, optimizer, device
        )

        # VALIDATION
        with torch.no_grad():
            val_epoch_loss = val_process(model, val_loader, criterion, device)

        if e % 10 == 0:
            test_acc = get_accuracy(loss, y_test, model, test_loader, device)
            val_acc = get_accuracy(loss, y_test, model, val_loader, device)
            train_acc = get_accuracy(loss, y_test, model, train_loader, device)

            print(
                f"Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.3f} | Val Loss: {val_epoch_loss/len(val_loader):.3f} | Train accuracy:{train_acc:.3f} | Val Accuracy: {val_acc:.3f} | Test Accuracy: {test_acc:.3f}"
            )


if "__main__" == __name__:
    # seed torch
    # torch.manual_seed(SEED)

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
        for i in range(10):
            print("Training and evaluating model {} time".format(i))
            train_eval(model, optimizer, dataset, loss, replace)
