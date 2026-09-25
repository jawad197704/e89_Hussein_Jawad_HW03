# Dialog Summary: CSCI E-89 Homework 03, Problem 3 (Fashion-MNIST Classifier)

This summarizes the conversation that produced `prob3/`: a Fashion-MNIST
image classifier built in PyTorch following Chapter 10 of Géron's *Hands-On
Machine Learning with Scikit-Learn and PyTorch* (2025), built as eight
standalone scripts and then combined into one notebook,
`e89_Hussein_Jawad_HW03_Prob3.ipynb`.

## Request and approach

Jawad asked for the pipeline to be built incrementally as separate scripts
(so each step could be checked on its own), then assembled into a single
notebook that runs top to bottom, with a markdown cell introducing each
script and short comments inside the code. The notebook needed a title cell
("CSCI E-89 Homework 03, Problem 3" / Jawad Hussein), and everything had to
live under a new `prob3/` folder in the repo. The instructions also asked
for a smoke test — run the notebook once with quick settings (1 training
epoch, 2 Optuna trials) to catch errors cheaply — before restoring the full
settings (20 epochs; 5 and 20 Optuna trials) for the final, committed run.

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
dict used for plotting later.

### `script5_predictions.py`
Uses the trained model to predict the first 3 validation images: predicted
vs. actual class name, the probability of every class, the top 4 most
likely classes, and the total parameter count of the model.

### `script6_plot_accuracy.py`
Plots training accuracy vs. epoch from `history["train_accuracy"]` and saves
`training_accuracy.png`.

### `script7_optuna.py`
Tunes the learning rate (1e-5 to 1e-1, log scale) and a shared hidden-layer
width (20 to 300) with Optuna. Each of 5 trials (seeded `TPESampler`, seed
42) trains a fresh model for 10 epochs and is scored by its best validation
accuracy; prints the best parameters and score.

### `script8_optuna_pruning.py`
Improves the search with early stopping: trains one epoch at a time,
reports validation accuracy to Optuna after every epoch, and uses a
`MedianPruner` to cut trials falling behind the median of prior trials at
the same epoch. The objective takes `train_loader`, `valid_loader`, and
`device` as explicit arguments (via `functools.partial`) instead of reading
module-level globals. Runs 20 trials and prints the best parameters/score.

## Notebook assembly

All eight scripts were combined into `e89_Hussein_Jawad_HW03_Prob3.ipynb` in
order, each preceded by a markdown cell naming the script and describing
what it does, with short inline comments in the code. Because the notebook
shares one running kernel, the model trained once in the Script 4 cell is
reused directly by the Script 5 (predictions) and Script 6 (plotting) cells,
rather than retraining redundantly as the standalone `.py` scripts do when
run independently.

## Corrections made along the way

- **Reorganized into `prob3/`.** The scripts, notebook, dialog summary, and
  training-accuracy plot were moved into a new `prob3/` folder as requested,
  and the notebook was renamed from an earlier working name to
  `e89_Hussein_Jawad_HW03_Prob3.ipynb`.
- **Title cell rewritten** to the exact required text ("CSCI E-89 Homework
  03, Problem 3") plus Jawad's name, replacing a more generic title.
- **Markdown cells expanded** so each one explicitly names the script file
  it corresponds to (e.g., "Script 4 — `script4_train.py`") and summarizes
  its purpose, rather than a bare numbered heading.
- **Inline comments added** to both the `.py` scripts and the notebook code
  cells to call out non-obvious steps (e.g., why `ToTensor()` gives a
  `[0, 1]`-scaled float tensor, why only the training loader is shuffled,
  why `trial.report()` is called every epoch for the pruner).
- **Verification run.** The notebook was first run with quick settings
  (`N_EPOCHS = 1` for training, `N_TRIALS = 2` with 1 epoch per trial for
  both Optuna searches) to confirm the full pipeline executes without
  errors. The settings were then restored to the assignment's full values
  (20 training epochs; 5 trials × 10 epochs for the basic Optuna search; 20
  trials × up to 10 epochs with pruning) and the notebook was re-executed
  end to end to produce the committed outputs.
