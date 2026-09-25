"""Define the Fashion-MNIST classifier: a simple fully connected network."""

import torch
from torch import nn

SEED = 42

torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FashionClassifier(nn.Module):
    """Flatten -> Dense(300, ReLU) -> Dense(100, ReLU) -> Dense(10)."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(28 * 28, 300)
        self.hidden2 = nn.Linear(300, 100)
        self.output = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        return self.output(x)  # raw logits; CrossEntropyLoss applies softmax internally


model = FashionClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    print(f"Device: {device}")
    print(model)
