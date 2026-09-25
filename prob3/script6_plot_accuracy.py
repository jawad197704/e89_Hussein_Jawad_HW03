"""Plot the training accuracy recorded during training."""

import matplotlib.pyplot as plt
from torch import optim

from script2_dataloaders import train_loader, valid_loader
from script3_model import device, loss_fn, model
from script4_train import LEARNING_RATE, N_EPOCHS, train

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
history = train(model, train_loader, valid_loader, loss_fn, optimizer, N_EPOCHS, device)

# Plot training accuracy (from history["train_accuracy"]) against epoch number.
epochs = range(1, N_EPOCHS + 1)
plt.plot(epochs, history["train_accuracy"], marker="o", label="Training accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy over Epochs")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.savefig("training_accuracy.png")
plt.show()
