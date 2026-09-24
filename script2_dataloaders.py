"""Create DataLoaders for the Fashion-MNIST train/validation/test sets."""

from torch.utils.data import DataLoader

from script1_load_dataset import test_dataset, train_dataset, valid_dataset

BATCH_SIZE = 32

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

if __name__ == "__main__":
    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Class name: {CLASS_NAMES[label]}")
