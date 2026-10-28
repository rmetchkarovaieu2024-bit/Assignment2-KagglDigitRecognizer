# Data

The CSV files are **not** included in this submission (assignment rule). To download them:

1. Go to https://www.kaggle.com/competitions/digit-recognizer, click **Join Competition** and accept the rules.
2. Open the **Data** tab and click **Download All** (or use the Kaggle API:
   `kaggle competitions download -c digit-recognizer`).
3. Unzip and place the two files in this folder:

```
data/
├── README.md
├── train.csv   # 42,000 labelled images: label + 784 pixel columns (28x28, 0-255)
└── test.csv    # 28,000 unlabelled images: 784 pixel columns
```

Only the Kaggle `train.csv` and `test.csv` are used. The original MNIST test set, its labels and any
other external labelled digits are **not** used.
