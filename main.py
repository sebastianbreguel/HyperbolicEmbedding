from utils import get_data, get_model, get_metrics, getMNIST
import torch
from utils import EPOCHS, LEARNING_RATE, SEED
from utils import generate_data, obtain_loss, obtain_optimizer

import argparse
from utils import run_model, run_MNIST


def train_eval(
    option_model: str,
    optimizer_option: str,
    dataset: int,
    loss: str,
    replace,
) -> None:

    if task == "MNIST":
        train_loader, test_loader = getMNIST()

    else:
        train_loader, val_loader, test_loader, y_test = get_data(dataset, replace, task)

    device = torch.device("cpu")
    model = get_model(option_model, dataset, task).to(device)

    criterion = obtain_loss(loss)

    optimizer = obtain_optimizer(optimizer_option, model)

    print(f"Running {option_model} Model - {optimizer_option} Optimizer", LEARNING_RATE)

    torch.autograd.set_detect_anomaly(True)

    if task == "MNIST":
        run_MNIST(model, device, train_loader, test_loader, criterion, optimizer)

    else:
        run_model(
            model,
            device,
            loss,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            optimizer,
            y_test,
        )


if "__main__" == __name__:
    # seed torch
    torch.manual_seed(SEED)

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
        generate_data(create_folder, replace, task)

    if make_train_eval:
        for i in range(10):
            print("Training and evaluating model {} time".format(i))
            train_eval(model, optimizer, dataset, loss, replace)
