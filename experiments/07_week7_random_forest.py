# ============================================================
# DemandIQ
# Week 7 - Random Forest
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
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
    average_precision_score,
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

RF_SAMPLE_SIZE = 350_000

N_ESTIMATORS = 100

RANDOM_STATE = 42


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("DEMANDIQ - WEEK 7")
print("Random Forest Models")
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
    f"{len(data):,} rows × "
    f"{data.shape[1]} columns"
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
# 5. TIME SPLIT
# ============================================================

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

print("\n" + "=" * 70)
print("TIME-BASED TRAIN / TEST SPLIT")
print("=" * 70)

print(
    f"\nTraining rows: "
    f"{len(train_data):,}"
)

print(
    f"Testing rows : "
    f"{len(test_data):,}"
)


# ============================================================
# 6. REPRESENTATIVE SAMPLE
# ============================================================

if len(train_data) > RF_SAMPLE_SIZE:

    model_train = train_data.sample(
        n=RF_SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

else:

    model_train = train_data.copy()

print(
    f"\nRandom Forest training rows: "
    f"{len(model_train):,}"
)


# ============================================================
# 7. PREPROCESSOR
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
# 8. RANDOM FOREST REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST REGRESSION")
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

rf_regression = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                max_depth=18,
                min_samples_leaf=10,
                max_features="sqrt",
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        )
    ]
)

print(
    "\nTraining Random Forest Regressor..."
)

rf_regression.fit(
    X_train_reg,
    y_train_reg
)

print(
    "Training completed."
)

reg_predictions = rf_regression.predict(
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

reg_model = rf_regression.named_steps[
    "regressor"
]

print("\nRandom Forest Regression Results")

print(
    f"MAE    : {reg_mae:.4f}"
)

print(
    f"RMSE   : {reg_rmse:.4f}"
)

print(
    f"R²     : {reg_r2:.4f}"
)

print(
    f"OOB R² : {reg_model.oob_score_:.4f}"
)


# ============================================================
# 9. REGRESSION FEATURE IMPORTANCE
# ============================================================

reg_preprocessor = rf_regression.named_steps[
    "preprocessor"
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
        "week7_rf_regression_feature_importance.csv"
    ),
    index=False
)

top_reg = reg_importance.head(10)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    top_reg["feature"][::-1],
    top_reg["importance"][::-1]
)

plt.xlabel(
    "Feature Importance"
)

plt.title(
    "Week 7 - Random Forest Regression Feature Importance"
)

plt.tight_layout()

reg_importance_path = os.path.join(
    FIGURES_DIR,
    "week7_rf_regression_feature_importance.png"
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
print("RANDOM FOREST CLASSIFICATION")
print("=" * 70)

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
# 11. RANDOM FOREST CLASSIFIER
# ============================================================

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

rf_classifier = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=18,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced",
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        )
    ]
)

print(
    "\nTraining Random Forest Classifier..."
)

rf_classifier.fit(
    X_train_cls,
    y_train_cls
)

print(
    "Training completed."
)

cls_predictions = rf_classifier.predict(
    X_test_cls
)

cls_probabilities = (
    rf_classifier
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

pr_auc = average_precision_score(
    y_test_cls,
    cls_probabilities
)

cls_model = rf_classifier.named_steps[
    "classifier"
]

print("\nRandom Forest Classification Results")

print(
    f"Accuracy     : {accuracy:.4f}"
)

print(
    f"Precision    : {precision:.4f}"
)

print(
    f"Recall       : {recall:.4f}"
)

print(
    f"F1 Score     : {f1:.4f}"
)

print(
    f"ROC-AUC      : {roc_auc:.4f}"
)

print(
    f"PR-AUC       : {pr_auc:.4f}"
)

print(
    f"OOB Accuracy : "
    f"{cls_model.oob_score_:.4f}"
)


# ============================================================
# 12. CLASSIFICATION FEATURE IMPORTANCE
# ============================================================

cls_preprocessor = rf_classifier.named_steps[
    "preprocessor"
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
        "week7_rf_classification_feature_importance.csv"
    ),
    index=False
)

top_cls = cls_importance.head(10)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    top_cls["feature"][::-1],
    top_cls["importance"][::-1]
)

plt.xlabel(
    "Feature Importance"
)

plt.title(
    "Week 7 - Random Forest Classification Feature Importance"
)

plt.tight_layout()

cls_importance_path = os.path.join(
    FIGURES_DIR,
    "week7_rf_classification_feature_importance.png"
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
# 13. CONFUSION MATRIX
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
    "Week 7 - Random Forest Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    FIGURES_DIR,
    "week7_rf_confusion_matrix.png"
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
# 14. SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    [
        {
            "task": "Regression",
            "model":
                "Random Forest Regressor",
            "MAE": reg_mae,
            "RMSE": reg_rmse,
            "R2": reg_r2,
            "OOB":
                reg_model.oob_score_,
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "ROC_AUC": np.nan,
            "PR_AUC": np.nan
        },
        {
            "task": "Classification",
            "model":
                "Random Forest Classifier",
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "OOB":
                cls_model.oob_score_,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc
        }
    ]
)

results_path = os.path.join(
    RESULTS_DIR,
    "week7_random_forest_results.csv"
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
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WEEK 7 COMPLETED")
print("=" * 70)

print(
    """
Regression:
✓ Random Forest Regressor
✓ MAE
✓ RMSE
✓ R²
✓ OOB R²

Classification:
✓ Random Forest Classifier
✓ Accuracy
✓ Precision
✓ Recall
✓ F1
✓ ROC-AUC
✓ PR-AUC
✓ OOB Accuracy

Interpretability:
✓ Regression feature importance
✓ Classification feature importance
✓ Confusion matrix
"""
)

print("=" * 70)