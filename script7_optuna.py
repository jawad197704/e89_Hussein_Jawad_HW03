"""Use Optuna to tune the learning rate and hidden layer size."""

import optuna
import torch
from torch import nn, optim

from script2_dataloaders import train_loader, valid_loader
from script3_model import device
from script4_train import train

SEED = 42
N_TRIALS = 5
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


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)

    torch.manual_seed(SEED)
    model = build_model(n_hidden)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    history = train(model, train_loader, valid_loader, loss_fn, optimizer, N_EPOCHS, device)
    return max(history["val_accuracy"])


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest parameters:", study.best_params)
    print("Best validation accuracy:", study.best_value)
