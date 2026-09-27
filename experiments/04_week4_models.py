# ============================================================
# DemandIQ
# Week 4 - Ridge, Lasso & Logistic Regression
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge, Lasso, LogisticRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    roc_curve
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "demandiq_features.csv"
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
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("DEMANDIQ - WEEK 4")
print("Ridge, Lasso & Logistic Regression")
print("=" * 70)

print("\nLoading processed dataset...")

data = pd.read_csv(DATA_PATH)

data["date"] = pd.to_datetime(data["date"])

data = data.sort_values(
    "date"
).reset_index(drop=True)

print(
    f"Dataset loaded: "
    f"{len(data):,} rows × {data.shape[1]} columns"
)


# ============================================================
# 3. FEATURES
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
# 4. TIME-BASED TRAIN / TEST SPLIT
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

train_data = data.loc[
    train_mask
].copy()

test_data = data.loc[
    test_mask
].copy()

print(
    f"\nTraining period:"
    f"\n{train_data['date'].min().date()}"
    f" → "
    f"{train_data['date'].max().date()}"
)

print(
    f"\nTesting period:"
    f"\n{test_data['date'].min().date()}"
    f" → "
    f"{test_data['date'].max().date()}"
)

print(
    f"\nTraining rows: {len(train_data):,}"
)

print(
    f"Testing rows : {len(test_data):,}"
)


# ============================================================
# 5. PREPROCESSOR
# ============================================================

def create_preprocessor():

    return ColumnTransformer(
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
                StandardScaler(),
                numeric_features
            )
        ]
    )


# ============================================================
# 6. REGRESSION DATA
# ============================================================

X_train = train_data[
    feature_columns
]

X_test = test_data[
    feature_columns
]

y_train = train_data[
    TARGET
]

y_test = test_data[
    TARGET
]


# ============================================================
# 7. REGRESSION EVALUATION
# ============================================================

regression_predictions = {}

def evaluate_regression(
    model,
    model_name,
    X_fit,
    y_fit
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"Training {model_name}..."
    )

    model.fit(
        X_fit,
        y_fit
    )

    print(
        f"{model_name} training completed."
    )

    print(
        "Generating predictions..."
    )

    predictions = model.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0
    )

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

    regression_predictions[
        model_name
    ] = predictions

    print(
        f"\n{model_name} Results"
    )

    print(
        f"MAE  : {mae:,.4f}"
    )

    print(
        f"RMSE : {rmse:,.4f}"
    )

    print(
        f"R²   : {r2:,.4f}"
    )

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 8. RIDGE REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1: RIDGE REGRESSION")
print("=" * 70)

ridge_model = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "regressor",
            Ridge(
                alpha=10.0
            )
        )
    ]
)

ridge_result = evaluate_regression(
    ridge_model,
    "Ridge Regression",
    X_train,
    y_train
)


# ============================================================
# 9. LASSO REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2: LASSO REGRESSION")
print("=" * 70)

print(
    "\nLasso is computationally expensive on "
    "the full multi-million-row dataset."
)

print(
    "Using a representative training sample "
    "for efficient optimization."
)

LASSO_SAMPLE_SIZE = 300_000

if len(train_data) > LASSO_SAMPLE_SIZE:

    lasso_train = train_data.sample(
        n=LASSO_SAMPLE_SIZE,
        random_state=42
    )

else:

    lasso_train = train_data.copy()

X_lasso = lasso_train[
    feature_columns
]

y_lasso = lasso_train[
    TARGET
]

print(
    f"Lasso training rows: "
    f"{len(X_lasso):,}"
)

lasso_model = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "regressor",
            Lasso(
                alpha=0.001,
                max_iter=2000,
                tol=0.001
            )
        )
    ]
)

lasso_result = evaluate_regression(
    lasso_model,
    "Lasso Regression",
    X_lasso,
    y_lasso
)


# ============================================================
# 10. SAVE REGRESSION RESULTS
# ============================================================

regression_results = pd.DataFrame(
    [
        ridge_result,
        lasso_result
    ]
)

regression_results_path = os.path.join(
    RESULTS_DIR,
    "week4_regression_results.csv"
)

regression_results.to_csv(
    regression_results_path,
    index=False
)

print(
    "\nRegression results saved to:"
)

print(
    regression_results_path
)


# ============================================================
# 11. REGRESSION MODEL COMPARISON GRAPH
# ============================================================

print(
    "\nGenerating regression comparison graph..."
)

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    regression_results["model"],
    regression_results["MAE"]
)

plt.ylabel(
    "MAE"
)

plt.title(
    "Week 4 - Regression Model Comparison"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

regression_graph_path = os.path.join(
    FIGURES_DIR,
    "week4_regression_model_comparison.png"
)

plt.savefig(
    regression_graph_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {regression_graph_path}"
)


# ============================================================
# 12. CLASSIFICATION TARGET
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION")
print("=" * 70)

print(
    "\nCreating high-demand target..."
)

high_demand_threshold = train_data[
    TARGET
].quantile(0.75)

print(
    f"High-demand threshold: "
    f"{high_demand_threshold:,.2f}"
)

train_data["high_demand"] = (
    train_data[TARGET]
    >= high_demand_threshold
).astype(int)

test_data["high_demand"] = (
    test_data[TARGET]
    >= high_demand_threshold
).astype(int)


# ============================================================
# 13. CLASS DISTRIBUTION
# ============================================================

print(
    "\nTraining class distribution:"
)

print(
    train_data[
        "high_demand"
    ].value_counts()
    .sort_index()
)

print(
    "\nTesting class distribution:"
)

print(
    test_data[
        "high_demand"
    ].value_counts()
    .sort_index()
)


# ============================================================
# 14. CLASSIFICATION DATA
# ============================================================

X_train_cls = train_data[
    feature_columns
]

X_test_cls = test_data[
    feature_columns
]

y_train_cls = train_data[
    "high_demand"
]

y_test_cls = test_data[
    "high_demand"
]


# ============================================================
# 15. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION")
print("=" * 70)

print(
    "\nTo keep computation practical, "
    "Logistic Regression uses a representative "
    "training sample."
)

LOGISTIC_SAMPLE_SIZE = 500_000

if len(train_data) > LOGISTIC_SAMPLE_SIZE:

    classification_train = train_data.sample(
        n=LOGISTIC_SAMPLE_SIZE,
        random_state=42
    )

else:

    classification_train = train_data.copy()

X_logistic = classification_train[
    feature_columns
]

y_logistic = classification_train[
    "high_demand"
]

print(
    f"Logistic training rows: "
    f"{len(X_logistic):,}"
)

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                solver="liblinear"
            )
        )
    ]
)

print(
    "\nTraining Logistic Regression..."
)

logistic_model.fit(
    X_logistic,
    y_logistic
)

print(
    "Logistic Regression training completed."
)


# ============================================================
# 16. CLASSIFICATION PREDICTIONS
# ============================================================

print(
    "\nGenerating classification predictions..."
)

class_predictions = (
    logistic_model
    .predict(X_test_cls)
)

class_probabilities = (
    logistic_model
    .predict_proba(
        X_test_cls
    )[:, 1]
)


# ============================================================
# 17. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test_cls,
    class_predictions
)

precision = precision_score(
    y_test_cls,
    class_predictions,
    zero_division=0
)

recall = recall_score(
    y_test_cls,
    class_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test_cls,
    class_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test_cls,
    class_probabilities
)


# ============================================================
# 18. DISPLAY RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "LOGISTIC REGRESSION RESULTS"
)

print(
    "=" * 70
)

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

print(
    "\nGenerating confusion matrix..."
)

ConfusionMatrixDisplay.from_predictions(
    y_test_cls,
    class_predictions,
    display_labels=[
        "Normal Demand",
        "High Demand"
    ],
    values_format="d"
)

plt.title(
    "Week 4 - Logistic Regression Confusion Matrix"
)

plt.tight_layout()

confusion_path = os.path.join(
    FIGURES_DIR,
    "week4_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {confusion_path}"
)


# ============================================================
# 20. ROC CURVE
# ============================================================

print(
    "Generating ROC curve..."
)

fpr, tpr, _ = roc_curve(
    y_test_cls,
    class_probabilities
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Week 4 - Logistic Regression ROC Curve"
)

plt.legend()

plt.tight_layout()

roc_path = os.path.join(
    FIGURES_DIR,
    "week4_logistic_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {roc_path}"
)


# ============================================================
# 21. SAVE CLASSIFICATION RESULTS
# ============================================================

classification_results = pd.DataFrame(
    [
        {
            "model": "Logistic Regression",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "high_demand_threshold":
                high_demand_threshold,
            "training_rows":
                len(X_logistic)
        }
    ]
)

classification_results_path = os.path.join(
    RESULTS_DIR,
    "week4_classification_results.csv"
)

classification_results.to_csv(
    classification_results_path,
    index=False
)

print(
    "\nClassification results saved to:"
)

print(
    classification_results_path
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WEEK 4 COMPLETED")
print("=" * 70)

print(
    """
Regression:
✓ Ridge Regression
✓ Lasso Regression

Classification:
✓ Logistic Regression

Graphs:
✓ Regression model comparison
✓ Classification confusion matrix
✓ Classification ROC curve

Regression metrics:
✓ MAE
✓ RMSE
✓ R²

Classification metrics:
✓ Accuracy
✓ Precision
✓ Recall
✓ F1
✓ ROC-AUC

Evaluation:
✓ Time-based split
✓ Same future test period
✓ No random train/test split

Computational optimization:
✓ Lasso uses 300,000 training rows
✓ Logistic Regression uses 500,000 training rows
"""
)

print("=" * 70)