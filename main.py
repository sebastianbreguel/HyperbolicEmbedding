from utils import get_data, get_model, get_accuracy, get_info, getMNIST
import torch
from utils import EPOCHS, LEARNING_RATE, SEED
from utils import generate_data, obtain_loss, obtain_optimizer, train_model, val_process

import argparse


def train_eval(
    option_model: str,
    optimizer_option: str,
    dataset: int,
    loss: str,
    replace,
) -> None:

    # train_loader, val_loader, test_loader, y_test = get_data(dataset, replace)
    train_loader, test_loader = getMNIST()

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
    print(model)
    ##########################
    #####  Train Model
    ##########################

    torch.autograd.set_detect_anomaly(True)
    iter = 0
    for epoch in range(EPOCHS):

        partial = 0
        total_partial = 0
        model.train()

        for i, (images, labels) in enumerate(train_loader):

            images, labels = images.to(device), labels.to(device)
            images = images.view(-1, 15).requires_grad_()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            partial += (predicted == labels).sum()
            total_partial += labels.size(0)
        iter += 1

        correct = 0
        total = 0

        for images, labels in test_loader:

            images = images.view(-1, 15).requires_grad_()
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        accuracy = 100 * correct / total

        print("training accuracy: {} ".format(partial / total_partial * 100))
        print(
            "Iteration: {}. Loss: {}. Accuracy: {}".format(iter, loss.item(), accuracy)
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
        generate_data(create_folder, replace, dataset, task)

    if make_train_eval:
        for i in range(10):
            print("Training and evaluating model {} time".format(i))
            train_eval(model, optimizer, dataset, loss, replace)
