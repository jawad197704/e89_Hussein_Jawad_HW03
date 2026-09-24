# Dialog Summary: Fashion-MNIST Classifier (Géron, Ch. 10)

This project was built incrementally, one script per step, following Chapter 10
of Géron's *Hands-On Machine Learning with Scikit-Learn and PyTorch* (2025).
Each script builds on the ones before it.

## script1_load_dataset.py

Loads the Fashion-MNIST dataset via TorchVision, converting images to float
tensors scaled to `[0, 1]` with `ToTensor()`. The 60,000-image training set is
split into 55,000 training images and 5,000 validation images using
`random_split` with a seeded generator (seed 42). The 10,000-image test set is
also loaded. Running the script prints the resulting split sizes and confirms
the image shape, dtype, and pixel value range.

## script2_dataloaders.py

Builds on script1 by wrapping `train_dataset`, `valid_dataset`, and
`test_dataset` in `DataLoader`s with a batch size of 32, shuffling only the
training loader. Also defines the `CLASS_NAMES` list for the 10 Fashion-MNIST
classes. Running the script prints the shape, dtype, and class name of the
first training sample.

## script3_model.py

Defines `FashionClassifier`, a fully connected network that flattens each
28x28 image and passes it through two hidden layers (300 and 100 neurons,
ReLU activations) to a 10-class output layer. The model is seeded with
`torch.manual_seed(42)` before initialization and moved to a GPU if one is
available (otherwise CPU). Also sets up `nn.CrossEntropyLoss()` as the loss
function.

## script4_train.py

Trains the model from script3 for 20 epochs using SGD with a learning rate of
0.1. Training and validation accuracy are tracked each epoch with
`torchmetrics.Accuracy`. The `train()` function prints the loss, training
accuracy, and validation accuracy after every epoch, and returns a `history`
dict (`loss`, `train_accuracy`, `val_accuracy`) for later plotting. A full run
took the loss from 0.61 to 0.19 and validation accuracy from ~84% to ~87-89%.

## script5_predictions.py

Retrains the model and uses it to predict the first 3 images of the
validation set. For each sample it prints the predicted vs. actual class
name, the probability of every class, and the top 4 most likely classes with
their probabilities. Also prints the total number of trainable parameters in
the model (266,610). All 3 sample predictions matched their actual labels.

## script6_plot_accuracy.py

Retrains the model to obtain the accuracy history, then plots training
accuracy vs. epoch using Matplotlib and saves the chart as
`training_accuracy.png`.

## script7_optuna.py

Introduces hyperparameter tuning with Optuna. Searches the learning rate
(1e-5 to 1e-1, log scale) and the number of neurons in the hidden layers
(20 to 300, same size shared by both layers). Each of 5 trials trains a
freshly built model for 10 epochs and is scored by its best validation
accuracy across those epochs. Uses a `TPESampler` seeded with 42 for
reproducible trials. Prints the best parameters and best score found.

## script8_optuna_pruning.py

Improves on script7 by pruning weak trials early instead of always running
the full epoch budget. Trains one epoch at a time, reports the validation
accuracy to Optuna after every epoch via `trial.report()`, and uses a
`MedianPruner` to stop trials that fall behind the median of prior trials at
the same epoch (`trial.should_prune()`). The data loaders and device are
passed explicitly into the `objective()` function (via `functools.partial`)
rather than read from module-level globals. Runs 20 trials with a seeded
`TPESampler` and prints the best parameters and best score. In a partial
verification run, trial 5 was pruned early as expected, confirming the
pruning logic works correctly.
