# ============================================================
# DemandIQ
# Week 6 - Decision Trees
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
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
# 2. CONFIGURATION
# ============================================================

TARGET = "sales"

# Tree models are trained on a large representative subset
# to keep runtime practical.
TREE_SAMPLE_SIZE = 400_000

RANDOM_STATE = 42


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("DEMANDIQ - WEEK 6")
print("Interpretable Decision Trees")
print("=" * 70)

print("\nLoading processed dataset...")

data = pd.read_csv(DATA_PATH)

data["date"] = pd.to_datetime(
    data["date"]
)

data = data.sort_values(
    "date"
).reset_index(drop=True)

print(
    f"Dataset loaded: "
    f"{len(data):,} rows × {data.shape[1]} columns"
)


# ============================================================
# 4. FEATURES
# ============================================================

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

train_data = data[
    data["date"] <= train_end_date
].copy()

test_data = data[
    data["date"] >= test_start_date
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
    f"\nTraining rows: "
    f"{len(train_data):,}"
)

print(
    f"Testing rows : "
    f"{len(test_data):,}"
)


# ============================================================
# 6. TRAINING SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING SAMPLE")
print("=" * 70)

if len(train_data) > TREE_SAMPLE_SIZE:

    model_train = train_data.sample(
        n=TREE_SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

else:

    model_train = train_data.copy()

print(
    f"Rows used for tree training: "
    f"{len(model_train):,}"
)


# ============================================================
# 7. PREPROCESSING
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
                "passthrough",
                numeric_features
            )
        ]
    )


# ============================================================
# 8. REGRESSION TREE
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE REGRESSION")
print("=" * 70)

X_train_reg = model_train[
    feature_columns
]

y_train_reg = model_train[
    TARGET
]

X_test_reg = test_data[
    feature_columns
]

y_test_reg = test_data[
    TARGET
]

regression_tree = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "regressor",
            DecisionTreeRegressor(
                max_depth=12,
                min_samples_leaf=50,
                random_state=RANDOM_STATE
            )
        )
    ]
)

print("\nTraining Decision Tree Regressor...")

regression_tree.fit(
    X_train_reg,
    y_train_reg
)

print("Training completed.")

print("\nGenerating predictions...")

reg_predictions = regression_tree.predict(
    X_test_reg
)

reg_predictions = np.maximum(
    reg_predictions,
    0
)

reg_mae = mean_absolute_error(
    y_test_reg,
    reg_predictions
)

reg_rmse = np.sqrt(
    mean_squared_error(
        y_test_reg,
        reg_predictions
    )
)

reg_r2 = r2_score(
    y_test_reg,
    reg_predictions
)

print("\nRegression Tree Results")

print(
    f"MAE  : {reg_mae:.4f}"
)

print(
    f"RMSE : {reg_rmse:.4f}"
)

print(
    f"R²   : {reg_r2:.4f}"
)


# ============================================================
# 9. REGRESSION FEATURE IMPORTANCE
# ============================================================

reg_preprocessor = regression_tree.named_steps[
    "preprocessor"
]

reg_model = regression_tree.named_steps[
    "regressor"
]

reg_feature_names = (
    reg_preprocessor
    .get_feature_names_out()
)

reg_importance = pd.DataFrame(
    {
        "feature": reg_feature_names,
        "importance":
            reg_model.feature_importances_
    }
)

reg_importance = reg_importance.sort_values(
    "importance",
    ascending=False
)

reg_importance.to_csv(
    os.path.join(
        RESULTS_DIR,
        "week6_regression_tree_feature_importance.csv"
    ),
    index=False
)

top_reg_features = reg_importance.head(10)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    top_reg_features["feature"][::-1],
    top_reg_features["importance"][::-1]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Week 6 - Decision Tree Regression Feature Importance"
)

plt.tight_layout()

reg_importance_path = os.path.join(
    FIGURES_DIR,
    "week6_regression_tree_feature_importance.png"
)

plt.savefig(
    reg_importance_path,
    dpi=150
)

plt.close()

print(
    f"\nSaved: {reg_importance_path}"
)


# ============================================================
# 10. CLASSIFICATION TARGET
# ============================================================

print("\n" + "=" * 70)
print("HIGH-DEMAND CLASSIFICATION TARGET")
print("=" * 70)

# Same principle as Week 4:
# calculate threshold using TRAINING data only.

high_demand_threshold = train_data[
    TARGET
].quantile(0.75)

print(
    f"\nHigh-demand threshold: "
    f"{high_demand_threshold:.2f}"
)

model_train["high_demand"] = (
    model_train[TARGET]
    >= high_demand_threshold
).astype(int)

test_data["high_demand"] = (
    test_data[TARGET]
    >= high_demand_threshold
).astype(int)


# ============================================================
# 11. CLASSIFICATION TREE
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE CLASSIFICATION")
print("=" * 70)

X_train_cls = model_train[
    feature_columns
]

y_train_cls = model_train[
    "high_demand"
]

X_test_cls = test_data[
    feature_columns
]

y_test_cls = test_data[
    "high_demand"
]

classification_tree = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=50,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]
)

print(
    "\nTraining Decision Tree Classifier..."
)

classification_tree.fit(
    X_train_cls,
    y_train_cls
)

print(
    "Training completed."
)

print(
    "\nGenerating classification predictions..."
)

cls_predictions = classification_tree.predict(
    X_test_cls
)

cls_probabilities = (
    classification_tree
    .predict_proba(
        X_test_cls
    )[:, 1]
)

accuracy = accuracy_score(
    y_test_cls,
    cls_predictions
)

precision = precision_score(
    y_test_cls,
    cls_predictions,
    zero_division=0
)

recall = recall_score(
    y_test_cls,
    cls_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test_cls,
    cls_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test_cls,
    cls_probabilities
)

print("\nClassification Tree Results")

print(
    f"Accuracy  : {accuracy:.4f}"
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
# 12. CLASSIFICATION FEATURE IMPORTANCE
# ============================================================

cls_preprocessor = classification_tree.named_steps[
    "preprocessor"
]

cls_model = classification_tree.named_steps[
    "classifier"
]

cls_feature_names = (
    cls_preprocessor
    .get_feature_names_out()
)

cls_importance = pd.DataFrame(
    {
        "feature": cls_feature_names,
        "importance":
            cls_model.feature_importances_
    }
)

cls_importance = cls_importance.sort_values(
    "importance",
    ascending=False
)

cls_importance.to_csv(
    os.path.join(
        RESULTS_DIR,
        "week6_classification_tree_feature_importance.csv"
    ),
    index=False
)

top_cls_features = cls_importance.head(10)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    top_cls_features["feature"][::-1],
    top_cls_features["importance"][::-1]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Week 6 - Classification Tree Feature Importance"
)

plt.tight_layout()

cls_importance_path = os.path.join(
    FIGURES_DIR,
    "week6_classification_tree_feature_importance.png"
)

plt.savefig(
    cls_importance_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {cls_importance_path}"
)


# ============================================================
# 13. TREE STRUCTURE VISUALIZATION
# ============================================================

print(
    "\nGenerating interpretable tree visualization..."
)

plt.figure(
    figsize=(22, 10)
)

plot_tree(
    cls_model,
    feature_names=cls_feature_names,
    class_names=[
        "Normal Demand",
        "High Demand"
    ],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7
)

plt.title(
    "Week 6 - Decision Tree: Top Decision Rules"
)

plt.tight_layout()

tree_path = os.path.join(
    FIGURES_DIR,
    "week6_classification_tree_structure.png"
)

plt.savefig(
    tree_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {tree_path}"
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test_cls,
    cls_predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal Demand",
        "High Demand"
    ]
)

display.plot(
    values_format="d"
)

plt.title(
    "Week 6 - Decision Tree Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    FIGURES_DIR,
    "week6_decision_tree_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {cm_path}"
)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    [
        {
            "task": "Regression",
            "model":
                "Decision Tree Regressor",
            "MAE": reg_mae,
            "RMSE": reg_rmse,
            "R2": reg_r2,
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "ROC_AUC": np.nan
        },
        {
            "task": "Classification",
            "model":
                "Decision Tree Classifier",
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc
        }
    ]
)

results_path = os.path.join(
    RESULTS_DIR,
    "week6_decision_tree_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print(
    f"\nResults saved to:"
    f"\n{results_path}"
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WEEK 6 COMPLETED")
print("=" * 70)

print(
    """
Regression:
✓ Decision Tree Regressor
✓ MAE
✓ RMSE
✓ R²

Classification:
✓ Decision Tree Classifier
✓ Accuracy
✓ Precision
✓ Recall
✓ F1
✓ ROC-AUC

Interpretability:
✓ Regression feature importance
✓ Classification feature importance
✓ Decision-tree structure
✓ Confusion matrix

Purpose:
Understand the demand thresholds and
feature relationships used by tree models.
"""
)

print("=" * 70)