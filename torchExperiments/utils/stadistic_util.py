import torch
import numpy as np

# Parameters
from utils.parameters import NM
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def get_accuracy(loss, y_test, model, test_loader, device):
    y_pred = []
    y_true = []

    # iterate over test data

    with torch.no_grad():
        for inputs, labels in test_loader:
            # pass to device
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

    if loss == "cross":
        new = []
        # Build confusion matrix
        for i in y_true:
            # obtain the index ob the max
            index = np.where(i == 1)[0][0]
            new.append(index)

        y_true = new

        return accuracy_score(y_true, y_pred)
    else:
        return round(np.linalg.norm(y_pred - y_test) / (0.2 * NM), 4)


def get_metrics(loss, y_test, y_pred_list, model, test_loader):
    if loss == "cross":
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
        new = []
        # Build confusion matrix
        for i in y_true:
            # obtain the index ob the max
            index = np.where(i == 1)[0][0]
            new.append(index)

        y_true = new
        print(
            f"Accuracy on the {len(y_test)} test data: {accuracy_score(y_true, y_pred)} %",
            end=" | ",
        )

        list_info = [round(accuracy_score(y_true, y_pred), 3)]
        # add to list info each data
        list_info += f1_score(y_true, y_pred, average=None).tolist()
        list_info += precision_score(
            y_true, y_pred, average=None, zero_division=1
        ).tolist()
        list_info += recall_score(y_true, y_pred, average=None).tolist()

        return list_info

    elif loss == "mse":
        print(
            f"Loss on Test Data: {round(np.linalg.norm(y_pred_list-y_test)/(0.2 * NM), 4)}"
        )
        return round(np.linalg.norm(y_pred_list - y_test) / (0.2 * NM), 4)
