"""Use the trained model to predict the first 3 validation images."""

import torch
from torch import optim

from script1_load_dataset import valid_dataset
from script2_dataloaders import CLASS_NAMES, train_loader, valid_loader
from script3_model import device, loss_fn, model
from script4_train import LEARNING_RATE, N_EPOCHS, train

N_SAMPLES = 3
TOP_K = 4

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
train(model, train_loader, valid_loader, loss_fn, optimizer, N_EPOCHS, device)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal number of parameters: {n_params:,}")

model.eval()
images = torch.stack([valid_dataset[i][0] for i in range(N_SAMPLES)]).to(device)
labels = [valid_dataset[i][1] for i in range(N_SAMPLES)]

with torch.no_grad():
    logits = model(images)
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

for i in range(N_SAMPLES):
    predicted_label = predictions[i].item()
    actual_label = labels[i]

    print(f"\nSample {i + 1}")
    print(f"  Predicted: {CLASS_NAMES[predicted_label]}")
    print(f"  Actual:    {CLASS_NAMES[actual_label]}")

    print("  Probability for each class:")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        print(f"    {class_name:<15} {probabilities[i, class_idx].item():.4f}")

    top_probs, top_classes = torch.topk(probabilities[i], TOP_K)
    print(f"  Top {TOP_K} most likely classes:")
    for prob, class_idx in zip(top_probs, top_classes):
        print(f"    {CLASS_NAMES[class_idx.item()]:<15} {prob.item():.4f}")
