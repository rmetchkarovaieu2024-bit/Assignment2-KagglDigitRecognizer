# Digit Recognizer — Assignment 2

Handwritten digit classification (0–9) for the Kaggle competition
[Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer), with PyTorch.

**Author:** Raya Metchkarova 
**Date:** 28/10/2026 BDBA
**Kaggle username:** `rmetchkarovaieu2024-bit`
**Public leaderboard score:** **0.99539** (submitted file: `results/submission.csv`)

## What is in this project

Three models are trained and compared on a stratified 90/10 hold-out validation split:

| Model | Description | Validation accuracy |
|---|---|---|
| MLP (baseline) | Flatten → 256 → 128 → 10, ReLU | 0.9786 |
| CNN | 2 conv blocks (Conv → ReLU → MaxPool) + 2 linear layers | 0.9898 |
| Improved CNN | 2 conv blocks with BatchNorm + Dropout, data augmentation (RandomAffine), CosineAnnealingLR, label smoothing | **0.9962** |

The improved CNN is used for the Kaggle submission.

```
metchkarova_raya_assignment2/
├── README.md
├── pyproject.toml
├── requirements.txt / environment.yml
├── data/README.md                   how to download train.csv and test.csv
├── notebooks/digit_recognizer.ipynb main notebook (executed, outputs visible)
├── src/digit_recognizer/
│   ├── __init__.py
│   ├── data.py      CSV loading, Dataset, normalisation, augmentation, DataLoaders
│   ├── models.py    MLP, SimpleCNN, ImprovedCNN
│   ├── train.py     train_one_epoch, evaluate, fit (best-epoch checkpoint, schedulers)
│   ├── predict.py   predictions, submission file and format check
│   ├── plots.py     learning curves, confusion matrix, misclassified digits
│   └── utils.py     seeds, device, parameter count
├── results/         submission.csv, figures, model_comparison.csv, metrics.json
├── tests/           pytest tests (logits shape [B, 10], data shapes, submission format)
└── memory.pdf       report
```

## How to install and run

```bash
# 1. Create and activate an environment (conda or venv)
conda env create -f environment.yml
conda activate digit-recognizer
#    or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 2. Install the package in editable mode (from the project root, where pyproject.toml is)
python -m pip install -e .

# 3. Download the data into data/ (see data/README.md)

# 4. Run the tests
pytest -q

# 5. Run the notebook from top to bottom
jupyter notebook notebooks/digit_recognizer.ipynb
```

The full notebook runs on a laptop CPU in roughly 20–30 minutes (most of it is the improved CNN).
Seeds are fixed (`SEED = 42`), so results are reproducible up to small differences between machines.

## Data rules

Only Kaggle's `train.csv` and `test.csv` are used. The original MNIST test set, its labels and any other
external labelled digits are not used.

## Use of AI assistants

Claude (Anthropic) was used to help structure (and write parts of) the README.md and .DS_store and its structure. It was use and for the checking of the code. If the results are good or not. (his suggestents were: )
VS code native ai may be used. (im not sure)
