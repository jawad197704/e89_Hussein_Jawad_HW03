"""Optuna search with per-epoch reporting and median pruning of weak trials."""

from functools import partial

import optuna
import torch
from torch import nn, optim
from torchmetrics import Accuracy

from script2_dataloaders import train_loader, valid_loader
from script3_model import device

SEED = 42
N_TRIALS = 20
N_EPOCHS = 10


def build_model(n_hidden):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 10),
    ).to(device)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Run a single training epoch (no accuracy tracking needed here)."""
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate_accuracy(model, loader, device):
    model.eval()
    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            accuracy.update(outputs, labels)
    return accuracy.compute().item()


def objective(trial, train_loader, valid_loader, device):
    # Data loaders and device are passed in explicitly rather than read from
    # module-level globals, per the assignment requirement.
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)

    torch.manual_seed(SEED)
    model = build_model(n_hidden)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    best_val_accuracy = 0.0
    for epoch in range(N_EPOCHS):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_accuracy = evaluate_accuracy(model, valid_loader, device)
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

        # Report this epoch's accuracy so the median pruner can compare this
        # trial's progress against prior trials at the same epoch.
        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_accuracy


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner()  # stops trials falling below the median early
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(
        partial(objective, train_loader=train_loader, valid_loader=valid_loader, device=device),
        n_trials=N_TRIALS,
    )

    print("\nBest parameters:", study.best_params)
    print("Best validation accuracy:", study.best_value)
