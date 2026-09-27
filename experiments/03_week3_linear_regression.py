# ============================================================
# DemandIQ
# Week 3 - Linear Regression Baseline
# ============================================================

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "results"
)

FIGURES_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "figures"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# 2. LOAD PROCESSED DATA
# ============================================================

print("\n" + "=" * 70)
print("DEMANDIQ - WEEK 3")
print("Linear Regression Baseline")
print("=" * 70)

print("\nLoading processed dataset...")

data = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "demandiq_features.csv"
    )
)

data["date"] = pd.to_datetime(data["date"])

print(
    f"Dataset loaded: "
    f"{data.shape[0]:,} rows × {data.shape[1]} columns"
)


# ============================================================
# 3. SORT BY DATE
# ============================================================

data = data.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# 4. DEFINE TARGET AND FEATURES
# ============================================================

TARGET = "sales"

feature_columns = [
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
    "onpromotion",
    "transactions",
    "dcoilwtico",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "is_holiday",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28"
]

X = data[feature_columns]
y = data[TARGET]


# ============================================================
# 5. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TIME-BASED TRAIN / TEST SPLIT")
print("=" * 70)

unique_dates = np.sort(
    data["date"].unique()
)

split_index = int(
    len(unique_dates) * 0.85
)

train_end_date = unique_dates[
    split_index - 1
]

test_start_date = unique_dates[
    split_index
]

train_mask = (
    data["date"] <= train_end_date
)

test_mask = (
    data["date"] >= test_start_date
)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\nTraining period:")

print(
    f"{data.loc[train_mask, 'date'].min().date()}"
    f" → "
    f"{data.loc[train_mask, 'date'].max().date()}"
)

print("\nTesting period:")

print(
    f"{data.loc[test_mask, 'date'].min().date()}"
    f" → "
    f"{data.loc[test_mask, 'date'].max().date()}"
)

print(
    f"\nTraining rows: {len(X_train):,}"
)

print(
    f"Testing rows : {len(X_test):,}"
)


# ============================================================
# 6. FEATURE TYPES
# ============================================================

categorical_features = [
    "family",
    "city",
    "state",
    "type"
]

numeric_features = [
    column
    for column in feature_columns
    if column not in categorical_features
]


# ============================================================
# 7. PREPROCESSING PIPELINE
# ============================================================

print("\nBuilding preprocessing pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# 8. LINEAR REGRESSION MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\nTraining Linear Regression...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 10. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test
)

predictions = np.maximum(
    predictions,
    0
)


# ============================================================
# 11. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LINEAR REGRESSION RESULTS")
print("=" * 70)

print(
    f"\nMAE  : {mae:,.4f}"
)

print(
    f"RMSE : {rmse:,.4f}"
)

print(
    f"R²   : {r2:,.4f}"
)


# ============================================================
# 13. ACTUAL VS PREDICTED GRAPH
# ============================================================

print("\nGenerating actual vs predicted graph...")

# Use a representative sample so the graph remains readable.
plot_sample_size = min(
    5000,
    len(y_test)
)

plot_indices = np.linspace(
    0,
    len(y_test) - 1,
    plot_sample_size,
    dtype=int
)

actual_sample = y_test.iloc[
    plot_indices
].values

predicted_sample = predictions[
    plot_indices
]

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    actual_sample,
    predicted_sample,
    alpha=0.25,
    s=10
)

max_value = max(
    actual_sample.max(),
    predicted_sample.max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual Sales"
)

plt.ylabel(
    "Predicted Sales"
)

plt.title(
    "Week 3 - Actual vs Predicted Sales"
)

plt.tight_layout()

actual_predicted_path = os.path.join(
    FIGURES_DIR,
    "week3_actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {actual_predicted_path}"
)


# ============================================================
# 14. RESIDUAL GRAPH
# ============================================================

print("Generating residual graph...")

residuals = (
    y_test.values -
    predictions
)

residual_sample = residuals[
    plot_indices
]

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    predicted_sample,
    residual_sample,
    alpha=0.25,
    s=10
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Sales"
)

plt.ylabel(
    "Residual (Actual - Predicted)"
)

plt.title(
    "Week 3 - Prediction Residuals"
)

plt.tight_layout()

residual_path = os.path.join(
    FIGURES_DIR,
    "week3_residuals.png"
)

plt.savefig(
    residual_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {residual_path}"
)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "model": [
            "Linear Regression"
        ],
        "MAE": [
            mae
        ],
        "RMSE": [
            rmse
        ],
        "R2": [
            r2
        ],
        "train_rows": [
            len(X_train)
        ],
        "test_rows": [
            len(X_test)
        ],
        "train_start": [
            data.loc[
                train_mask,
                "date"
            ].min()
        ],
        "train_end": [
            data.loc[
                train_mask,
                "date"
            ].max()
        ],
        "test_start": [
            data.loc[
                test_mask,
                "date"
            ].min()
        ],
        "test_end": [
            data.loc[
                test_mask,
                "date"
            ].max()
        ]
    }
)

result_path = os.path.join(
    RESULTS_DIR,
    "week3_linear_regression.csv"
)

results.to_csv(
    result_path,
    index=False
)

print(
    f"\nResults saved to:"
    f"\n{result_path}"
)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("WEEK 3 BASELINE COMPLETED")
print("=" * 70)

print(
    """
Model:
Linear Regression

Evaluation:
- MAE
- RMSE
- R²

Graphs:
- Actual vs Predicted Sales
- Prediction Residuals

Split:
Time-based 85% training / 15% testing

The Linear Regression result will serve as
our baseline for Week 4.
"""
)

print("=" * 70)