"""Load Fashion-MNIST and split the training set into train/validation."""

import torch
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

SEED = 42
N_VALID = 5_000

torch.manual_seed(SEED)

# ToTensor() converts each PIL image to a float tensor scaled to [0, 1].
full_train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Split the 60,000-image training set into 55,000 train / 5,000 validation.
n_train = len(full_train_dataset) - N_VALID
generator = torch.Generator().manual_seed(SEED)
train_dataset, valid_dataset = random_split(
    full_train_dataset, [n_train, N_VALID], generator=generator
)

if __name__ == "__main__":
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Pixel value range: [{image.min():.4f}, {image.max():.4f}]")
    print(f"Label: {label}")
