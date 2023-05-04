from utils import get_accuracy
import torch
from utils import EPOCHS, DIMENTIONS
from utils import train_model, val_process


def run_model(
    model,
    device,
    loss,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    y_test,
):
    for e in range(1, EPOCHS + 1):
        # TRAINING
        train_epoch_loss = train_model(
            model, train_loader, criterion, optimizer, device
        )

        # VALIDATION
        with torch.no_grad():
            val_epoch_loss = val_process(model, val_loader, criterion, device)

        test_acc = get_accuracy(loss, y_test, model, test_loader, device)
        val_acc = get_accuracy(loss, y_test, model, val_loader, device)
        train_acc = get_accuracy(loss, y_test, model, train_loader, device)

        print(
            f"Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.3f} | Val Loss: {val_epoch_loss/len(val_loader):.3f} | Train accuracy:{train_acc:.3f} | Val Accuracy: {val_acc:.3f} | Test Accuracy: {test_acc:.3f}"
        )
    return None


def run_MNIST(model, device, train_loader, test_loader, criterion, optimizer):
    for epoch in range(EPOCHS):
        partial = 0
        total_partial = 0
        model.train()

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            images = images.view(-1, DIMENTIONS).requires_grad_()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            partial += (predicted == labels).sum()
            total_partial += labels.size(0)

        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.view(-1, DIMENTIONS).requires_grad_()
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        accuracy = 100 * correct / total

        print("training accuracy: {} ".format(partial / total_partial * 100))
        print(
            "Iteration: {}. Loss: {}. Accuracy: {}".format(epoch, loss.item(), accuracy)
        )
