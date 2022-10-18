import torch
from sklearn import accuracy_score
import numpy as np


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
