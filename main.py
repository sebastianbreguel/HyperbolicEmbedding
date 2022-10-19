from utils import get_data, get_model, get_metrics, getMNIST
import torch
from utils import EPOCHS, LEARNING_RATE, SEED
from utils import  obtain_loss, obtain_optimizer
from utils.parameters import LARGE

import argparse
from utils import run_model, run_MNIST

print(__name__)

def train_eval(
    option_model: str,
    optimizer_option: str,
    dataset: int,
    loss: str,
    replace,
    task,
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
print(__name__)

if "__main__" == __name__:
    print("AAAAA")
    # seed torch
    torch.manual_seed(SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", action="store", help="Model to use")
    parser.add_argument("--optimizer", action="store", help="Optimizer to use")
    parser.add_argument("--replace", type=float, help="", default=0.5)
    parser.add_argument("--dataset", type=int, help="", default=10)
    parser.add_argument("--loss", action="store", help="Loss to use")
    parser.add_argument("--task", action="store", help="task to use")

    results = parser.parse_args()
    model = results.model
    optimizer = results.optimizer
    replace = results.replace
    dataset = results.dataset
    loss = results.loss
    task = results.task
    print(LARGE)

    for i in range(10):
        print("Training and evaluating model {} time".format(i))
        train_eval(model, optimizer, dataset, loss, replace, task)
