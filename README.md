# Adult Income Classification Using Artificial Neural Networks

## Project Overview

This project explores **binary income classification** using the Adult Income (Census Income) dataset. The objective is to predict whether an individual's annual income is:

- `0` — `<=50K`
- `1` — `>50K`

The notebook covers data cleaning, exploratory data analysis (EDA), preprocessing, a baseline Artificial Neural Network (ANN), and experiments comparing activation functions, weight initialisation techniques, and loss functions.

## Dataset

The notebook loads the dataset from a local CSV file named `data_adult.csv`.

- Original dataset size: **48,842 rows × 15 columns**
- Target column: `income`
- Missing values are represented by `?` in the raw data.
- After replacing `?` with missing values and dropping rows containing missing values, **45,222 rows** remain.
- Target distribution after cleaning:

| Income class | Meaning | Count |
|---|---|---:|
| 0 | `<=50K` | 34,014 |
| 1 | `>50K` | 11,208 |

The target is imbalanced, with approximately 75% of observations in class 0 and 25% in class 1.

> The CSV dataset is not included in this repository. Place `data_adult.csv` in the project directory and update the notebook's CSV path if necessary.

## Technologies and Libraries

The notebook uses:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras

## Project Workflow

### 1. Data Loading, Cleaning and EDA

The notebook:

1. Loads and inspects the dataset's shape, data types and first few records.
2. Removes leading and trailing whitespace from object columns.
3. Replaces `?` with `NaN` and drops rows containing missing values.
4. Drops `fnlwgt` and `education`, retaining `educational-num`.
5. Encodes the target column as `0` (`<=50K`) and `1` (`>50K`).
6. Visualises the income-class distribution.
7. Explores age, hours worked per week and educational level by income class.

The notebook's EDA observations indicate that the higher-income group is more concentrated around middle age, generally works more hours per week, and has greater representation at higher education levels.

### 2. Feature Preprocessing

- Separates the predictors (`X`) from the target (`y`).
- Applies one-hot encoding to categorical features using `pd.get_dummies(drop_first=True)`.
- Standardises selected numerical features using `StandardScaler`.
- Splits the data into training and testing sets using an 80:20 split, `random_state=42`, and stratification.

Resulting shapes recorded in the notebook:

| Dataset | Shape |
|---|---:|
| `X_train` | 36,177 × 80 |
| `X_test` | 9,045 × 80 |
| `y_train` | 36,177 |
| `y_test` | 9,045 |

### 3. Baseline ANN Model

A reusable `build_ann()` function creates and compiles a Keras Sequential model.

**Architecture**

| Layer | Configuration |
|---|---|
| Input | 80 features |
| Hidden layer 1 | Dense, 128 units, ReLU |
| Hidden layer 2 | Dense, 64 units, ReLU |
| Output | Dense, 1 unit, Sigmoid |

**Default training configuration**

- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Metric: Accuracy
- Epochs: 50
- Batch size: 64
- Validation split: 0.1

The notebook plots training and validation loss and accuracy, then evaluates the baseline model using a classification report and confusion matrix.

**Baseline test-set metrics recorded in the notebook**

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| `<=50K` (0) | 0.88 | 0.92 | 0.90 |
| `>50K` (1) | 0.72 | 0.61 | 0.66 |

Overall test accuracy: **0.84**.

### 4. Activation Function Experiment

Four ANN models are trained with different hidden-layer activation functions:

- ReLU
- Tanh
- Sigmoid
- ELU

Each model is trained for 50 epochs with a batch size of 64 and a validation split of 0.1. The notebook compares training and validation accuracy in a four-panel plot and includes the final validation accuracy and class-1 F1-score in the panel titles.

### 5. Weight Initialisation Experiment

Five models are trained using the following initialisers:

- Glorot Uniform
- Glorot Normal
- He Uniform
- He Normal
- Zeros

The notebook compares validation accuracy over epochs, includes an 84% reference line, and marks the first threshold-crossing epochs for the non-zero initialisers. It also plots the training loss and validation accuracy of the zeros-initialised model to demonstrate its failure to learn effectively.

### 6. Loss Function Experiment

The notebook compares the baseline Binary Cross-Entropy (BCE) model with a model trained using Mean Squared Error (MSE).

The MSE model uses:

- Hidden activation: ReLU
- Weight initialiser: He Normal
- Output activation: Sigmoid (retained by `build_ann()`)
- Epochs: 50
- Batch size: 64
- Validation split: 0.1

The notebook calculates class-1 F1-score on the test set and plots a bar chart comparing the two models.

| Model | Test F1-score for class 1 |
|---|---:|
| Baseline (BCE) | 0.6612 |
| MSE | 0.6649 |

**Important:** The notebook's Task 5.2 chart compares final test F1-scores. It does not plot validation F1-score across epochs.

## How to Run

1. Install Python and Jupyter Notebook (or use JupyterLab / VS Code).
2. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
   ```

3. Place `data_adult.csv` in the project folder.
4. Open `project2_deeplearning.ipynb`.
5. Update the `pd.read_csv(...)` path in the data-loading cell to match the location of your CSV file. The notebook currently contains a machine-specific Windows path.
6. Run the notebook cells in order.

## Notebook Scope and Notes

This README describes the uploaded notebook as it currently exists.

- The notebook contains Tasks 1 through 4 and Task 5.2.
- Task 5.3 (class-weighted Binary Cross-Entropy) is not yet present in the uploaded notebook.
- Some task numbers are absent from the notebook; this README does not assume those experiments were completed.
- The notebook standardises selected features before performing the train-test split. The README documents that implemented order rather than changing the workflow.
- The CSV dataset is an external input and must be supplied separately.

## Files

```text
project/
├── project2_deeplearning.ipynb
├── data_adult.csv
└── README.md
```

## Conclusion

This project demonstrates an end-to-end ANN classification workflow on the Adult Income dataset, including preprocessing, baseline evaluation, and controlled experiments with activation functions, weight initialisation methods, and loss functions. It also highlights the importance of evaluating the minority income class with precision, recall and F1-score rather than relying on accuracy alone.
