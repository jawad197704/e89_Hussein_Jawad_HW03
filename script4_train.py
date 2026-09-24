"""Train the Fashion-MNIST classifier and record loss/accuracy history."""

import torch
from torch import optim
from torchmetrics import Accuracy

from script2_dataloaders import train_loader, valid_loader
from script3_model import device, loss_fn, model

N_EPOCHS = 20
LEARNING_RATE = 0.1

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)


def train(model, train_loader, valid_loader, loss_fn, optimizer, n_epochs, device):
    history = {
        "loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    train_accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
    val_accuracy = Accuracy(task="multiclass", num_classes=10).to(device)

    for epoch in range(n_epochs):
        model.train()
        train_accuracy.reset()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_accuracy.update(outputs, labels)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = train_accuracy.compute().item()

        model.eval()
        val_accuracy.reset()
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_accuracy.update(outputs, labels)

        epoch_val_accuracy = val_accuracy.compute().item()

        history["loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_train_accuracy)
        history["val_accuracy"].append(epoch_val_accuracy)

        print(
            f"Epoch {epoch + 1}/{n_epochs} - "
            f"loss: {epoch_loss:.4f} - "
            f"train_accuracy: {epoch_train_accuracy:.4f} - "
            f"val_accuracy: {epoch_val_accuracy:.4f}"
        )

    return history


if __name__ == "__main__":
    history = train(
        model, train_loader, valid_loader, loss_fn, optimizer, N_EPOCHS, device
    )
