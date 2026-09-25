# Dialog Summary: CSCI E-89 Homework 03, Problem 3 (Fashion-MNIST Classifier)

This summarizes the conversation that produced `prob3/`: a Fashion-MNIST
image classifier built in PyTorch following Chapter 10 of Géron's *Hands-On
Machine Learning with Scikit-Learn and PyTorch* (2025), built as eight
standalone scripts and then combined into one notebook,
`e89_Hussein_Jawad_HW03_Prob3.ipynb`.

## Request and approach

Jawad asked for everything in a single prompt: all eight scripts, the
notebook that combines them, and this summary. The pipeline was not built
incrementally in back-and-forth exchanges — the eight separate scripts
were themselves part of what that one prompt specified, along with the
notebook's structure (a title cell; a labeled, descriptive markdown cell
before each script's code cell; short inline comments), the `prob3/`
folder location, the notebook's exact filename, and the two-stage
verification (run once with quick settings — 1 training epoch, 2 Optuna
trials — to catch errors cheaply, then restore the full settings — 20
epochs; 5 and 20 Optuna trials — for the final, committed run).

The generated code needed no corrections. The only follow-up was a second
request to fix the wording of this section (it had incorrectly described
the work as incremental) and to add the original prompt and this summary
into the notebook itself, as markdown cells.

## Scripts

### `script1_load_dataset.py`
Loads Fashion-MNIST via TorchVision, using `ToTensor()` so images become
float tensors scaled to `[0, 1]`. The 60,000-image training set is split
into 55,000 training / 5,000 validation images with a seeded (`42`)
`random_split`; the 10,000-image test set is loaded separately.

### `script2_dataloaders.py`
Wraps the three datasets in `DataLoader`s with batch size 32, shuffling only
the training loader. Defines the 10 `CLASS_NAMES`. Prints the shape, dtype,
and class name of the first training sample.

### `script3_model.py`
Defines `FashionClassifier`: flatten → Linear(784, 300) + ReLU →
Linear(300, 100) + ReLU → Linear(100, 10). Seeded with `torch.manual_seed(42)`
before initialization, moved to GPU if available (else CPU), paired with
`nn.CrossEntropyLoss()`.

### `script4_train.py`
Trains the model for 20 epochs with SGD (`lr=0.1`), tracking training and
validation accuracy each epoch via `torchmetrics.Accuracy`. `train()` prints
loss/train accuracy/validation accuracy every epoch and returns a `history`
dict used for plotting later. In the committed run, loss fell from about
0.61 to 0.19 and validation accuracy ended around 87-89%.

### `script5_predictions.py`
Uses the trained model to predict the first 3 validation images: predicted
vs. actual class name, the probability of every class, the top 4 most
likely classes, and the total parameter count of the model (266,610). All 3
sample predictions matched their actual labels in the committed run.

### `script6_plot_accuracy.py`
Plots training accuracy vs. epoch from `history["train_accuracy"]` and saves
`training_accuracy.png`.

### `script7_optuna.py`
Tunes the learning rate (1e-5 to 1e-1, log scale) and a shared hidden-layer
width (20 to 300) with Optuna. Each of 5 trials (seeded `TPESampler`, seed
42) trains a fresh model for 10 epochs and is scored by its best validation
accuracy; prints the best parameters and score. In the committed run, the
best trial found `lr≈0.0085`, `n_hidden=188`, with a validation accuracy of
about 85.1%.

### `script8_optuna_pruning.py`
Improves the search with early stopping: trains one epoch at a time,
reports validation accuracy to Optuna after every epoch, and uses a
`MedianPruner` to cut trials falling behind the median of prior trials at
the same epoch. The objective takes `train_loader`, `valid_loader`, and
`device` as explicit arguments (via `functools.partial`) instead of reading
module-level globals. Runs 20 trials and prints the best parameters/score.
In the committed run, trials 18 and 19 were pruned early as expected, and
the best trial found `lr≈0.089`, `n_hidden=193`, with a validation accuracy
of about 88.5%.

## Notebook assembly

All eight scripts were combined into `e89_Hussein_Jawad_HW03_Prob3.ipynb` in
order, each preceded by a markdown cell naming the script and describing
what it does, with short inline comments in the code. Because the notebook
shares one running kernel, the model trained once in the Script 4 cell is
reused directly by the Script 5 (predictions) and Script 6 (plotting) cells,
rather than retraining redundantly as the standalone `.py` scripts do when
run independently.

A "Setup: installs and imports" cell runs right after the title and prompt
cells, before Script 1. It starts with `%pip install torchmetrics optuna`
(the two packages the notebook needs that aren't preinstalled on Google
Colab or a fresh Anaconda environment) and then holds every import used
anywhere in the notebook, each listed once. The eight script cells below no
longer import anything themselves — they rely on the setup cell having
already run. The standalone `.py` files in `prob3/` were left untouched and
still import what they need, so each one still runs on its own.

## Follow-up requests

After the pipeline, notebook, and this summary were delivered, Jawad sent
two follow-ups.

The first asked to correct the wording of the "Request and approach"
section above (it had wrongly described the work as incremental, when it
was all specified in a single prompt), insert that original prompt into the
notebook as a markdown cell right after the title/name cell, and append the
corrected contents of this file to the notebook as a markdown cell at the
end. No code cells were changed to make these edits.

The second asked for the setup cell described above: gathering the
`%pip install` line and every import into one cell near the top of the
notebook, and stripping the now-redundant import lines out of the eight
script code cells (without changing anything else in them), while leaving
the standalone `.py` scripts as they were.
