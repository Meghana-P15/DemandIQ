# ============================================================
# DemandIQ
# Week 10 - Rigorous Temporal Evaluation
# ============================================================

import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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
    precision_recall_curve,
    roc_curve
)

from sklearn.calibration import (
    calibration_curve
)

from xgboost import (
    XGBRegressor,
    XGBClassifier
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

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# 2. CONFIG
# ============================================================

TARGET = "sales"

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

TRAIN_SAMPLE_SIZE = 400_000


# ============================================================
# 3. LOAD
# ============================================================

print("\n" + "=" * 70)
print("DEMANDIQ - WEEK 10")
print("Rigorous Temporal Evaluation")
print("=" * 70)

print("\nLoading processed dataset...")

data = pd.read_csv(
    DATA_PATH
)

data["date"] = pd.to_datetime(
    data["date"]
)

data = data.sort_values(
    "date"
).reset_index(
    drop=True
)

print(
    f"Dataset loaded: "
    f"{len(data):,} rows"
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
# 5. TEMPORAL SPLIT
# ============================================================

unique_dates = np.sort(
    data["date"].unique()
)

train_index = int(
    len(unique_dates)
    * TRAIN_RATIO
)

validation_index = int(
    len(unique_dates)
    * (
        TRAIN_RATIO
        + VALIDATION_RATIO
    )
)

validation_start = (
    unique_dates[
        train_index
    ]
)

test_start = (
    unique_dates[
        validation_index
    ]
)

train_data = data[
    data["date"]
    < validation_start
].copy()

validation_data = data[
    (
        data["date"]
        >= validation_start
    )
    &
    (
        data["date"]
        < test_start
    )
].copy()

test_data = data[
    data["date"]
    >= test_start
].copy()

print("\nTemporal split:")

print(
    f"Train      : "
    f"{len(train_data):,}"
)

print(
    f"Validation : "
    f"{len(validation_data):,}"
)

print(
    f"Test       : "
    f"{len(test_data):,}"
)


# ============================================================
# 6. TEMPORAL SPLIT GRAPH
# ============================================================

daily_sales = (
    data
    .groupby("date")["sales"]
    .sum()
)

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    daily_sales.index,
    daily_sales.values
)

plt.axvline(
    validation_start,
    linestyle="--",
    label="Train → Validation"
)

plt.axvline(
    test_start,
    linestyle="--",
    label="Validation → Test"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Total Sales"
)

plt.title(
    "Week 10 - Temporal Validation Strategy"
)

plt.legend()

plt.tight_layout()

temporal_path = os.path.join(
    FIGURES_DIR,
    "week10_temporal_split.png"
)

plt.savefig(
    temporal_path,
    dpi=150
)

plt.close()

print(
    f"\nSaved: {temporal_path}"
)


# ============================================================
# 7. TRAINING SAMPLE
# ============================================================

if (
    len(train_data)
    > TRAIN_SAMPLE_SIZE
):

    model_train = (
        train_data.sample(
            n=TRAIN_SAMPLE_SIZE,
            random_state=
                RANDOM_STATE
        )
    )

else:

    model_train = (
        train_data.copy()
    )


# ============================================================
# 8. PREPROCESS
# ============================================================

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

X_train = (
    preprocessor
    .fit_transform(
        model_train[
            feature_columns
        ]
    )
)

X_validation = (
    preprocessor
    .transform(
        validation_data[
            feature_columns
        ]
    )
)

X_test = (
    preprocessor
    .transform(
        test_data[
            feature_columns
        ]
    )
)


# ============================================================
# 9. REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("FINAL XGBOOST REGRESSION EVALUATION")
print("=" * 70)

regressor = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.80,
    colsample_bytree=0.80,
    reg_alpha=0.05,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    n_jobs=-1,
    random_state=
        RANDOM_STATE,
    early_stopping_rounds=50
)

regressor.fit(
    X_train,
    model_train[TARGET],
    eval_set=[
        (
            X_validation,
            validation_data[TARGET]
        )
    ],
    verbose=False
)

test_predictions = (
    regressor.predict(
        X_test
    )
)

test_predictions = np.maximum(
    test_predictions,
    0
)

mae = mean_absolute_error(
    test_data[TARGET],
    test_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        test_data[TARGET],
        test_predictions
    )
)

r2 = r2_score(
    test_data[TARGET],
    test_predictions
)

print(
    f"\nMAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 10. ACTUAL VS PREDICTED
# ============================================================

plot_size = min(
    5000,
    len(test_data)
)

rng = np.random.default_rng(
    RANDOM_STATE
)

plot_indices = rng.choice(
    len(test_data),
    size=plot_size,
    replace=False
)

actual_plot = (
    test_data[TARGET]
    .iloc[
        plot_indices
    ]
)

prediction_plot = (
    test_predictions[
        plot_indices
    ]
)

plt.figure(
    figsize=(9, 7)
)

plt.scatter(
    actual_plot,
    prediction_plot,
    alpha=0.4
)

maximum = max(
    actual_plot.max(),
    prediction_plot.max()
)

plt.plot(
    [0, maximum],
    [0, maximum],
    linestyle="--"
)

plt.xlabel(
    "Actual Sales"
)

plt.ylabel(
    "Predicted Sales"
)

plt.title(
    "Week 10 - Actual vs Predicted Demand"
)

plt.tight_layout()

actual_path = os.path.join(
    FIGURES_DIR,
    "week10_actual_vs_predicted.png"
)

plt.savefig(
    actual_path,
    dpi=150
)

plt.close()


# ============================================================
# 11. RESIDUALS
# ============================================================

residuals = (
    test_data[TARGET].to_numpy()
    - test_predictions
)

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    prediction_plot,
    residuals[
        plot_indices
    ],
    alpha=0.4
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Sales"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Week 10 - XGBoost Residual Analysis"
)

plt.tight_layout()

residual_path = os.path.join(
    FIGURES_DIR,
    "week10_residual_analysis.png"
)

plt.savefig(
    residual_path,
    dpi=150
)

plt.close()


# ============================================================
# 12. CLASSIFICATION TARGET
# ============================================================

high_demand_threshold = (
    model_train[
        TARGET
    ].quantile(
        0.75
    )
)

y_train_cls = (
    model_train[TARGET]
    >= high_demand_threshold
).astype(int)

y_validation_cls = (
    validation_data[TARGET]
    >= high_demand_threshold
).astype(int)

y_test_cls = (
    test_data[TARGET]
    >= high_demand_threshold
).astype(int)

positive = y_train_cls.sum()
negative = (
    len(y_train_cls)
    - positive
)

scale_pos_weight = (
    negative
    / max(
        positive,
        1
    )
)


# ============================================================
# 13. CLASSIFIER
# ============================================================

print("\n" + "=" * 70)
print("FINAL XGBOOST CLASSIFICATION EVALUATION")
print("=" * 70)

classifier = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.80,
    colsample_bytree=0.80,
    reg_alpha=0.05,
    reg_lambda=1.0,
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    scale_pos_weight=
        scale_pos_weight,
    random_state=
        RANDOM_STATE,
    early_stopping_rounds=50
)

classifier.fit(
    X_train,
    y_train_cls,
    eval_set=[
        (
            X_validation,
            y_validation_cls
        )
    ],
    verbose=False
)

validation_probabilities = (
    classifier.predict_proba(
        X_validation
    )[:, 1]
)


# ============================================================
# 14. THRESHOLD SELECTION
# ============================================================

best_threshold = 0.50
best_f1 = -1

for threshold in np.arange(
    0.20,
    0.81,
    0.02
):

    predictions = (
        validation_probabilities
        >= threshold
    ).astype(int)

    score = f1_score(
        y_validation_cls,
        predictions
    )

    if score > best_f1:

        best_f1 = score
        best_threshold = threshold


print(
    f"\nSelected probability threshold: "
    f"{best_threshold:.2f}"
)

test_probabilities = (
    classifier.predict_proba(
        X_test
    )[:, 1]
)

test_class_predictions = (
    test_probabilities
    >= best_threshold
).astype(int)


accuracy = accuracy_score(
    y_test_cls,
    test_class_predictions
)

precision = precision_score(
    y_test_cls,
    test_class_predictions,
    zero_division=0
)

recall = recall_score(
    y_test_cls,
    test_class_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test_cls,
    test_class_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test_cls,
    test_probabilities
)

pr_auc = average_precision_score(
    y_test_cls,
    test_probabilities
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
    f"F1        : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)


# ============================================================
# 15. PRECISION-RECALL GRAPH
# ============================================================

precision_curve, recall_curve, _ = (
    precision_recall_curve(
        y_test_cls,
        test_probabilities
    )
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_curve,
    precision_curve
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Week 10 - Precision Recall Curve"
)

plt.tight_layout()

pr_path = os.path.join(
    FIGURES_DIR,
    "week10_precision_recall_curve.png"
)

plt.savefig(
    pr_path,
    dpi=150
)

plt.close()


# ============================================================
# 16. ROC GRAPH
# ============================================================

fpr, tpr, _ = roc_curve(
    y_test_cls,
    test_probabilities
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr
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
    "Week 10 - ROC Curve"
)

plt.tight_layout()

roc_path = os.path.join(
    FIGURES_DIR,
    "week10_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=150
)

plt.close()


# ============================================================
# 17. CALIBRATION
# ============================================================

prob_true, prob_pred = (
    calibration_curve(
        y_test_cls,
        test_probabilities,
        n_bins=10
    )
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    prob_pred,
    prob_true,
    marker="o"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "Predicted Probability"
)

plt.ylabel(
    "Observed High-Demand Frequency"
)

plt.title(
    "Week 10 - High-Demand Probability Calibration"
)

plt.tight_layout()

calibration_path = os.path.join(
    FIGURES_DIR,
    "week10_calibration_curve.png"
)

plt.savefig(
    calibration_path,
    dpi=150
)

plt.close()


# ============================================================
# 18. SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    [
        {
            "regression_MAE": mae,
            "regression_RMSE":
                rmse,
            "regression_R2": r2,
            "classification_accuracy":
                accuracy,
            "classification_precision":
                precision,
            "classification_recall":
                recall,
            "classification_F1":
                f1,
            "classification_ROC_AUC":
                roc_auc,
            "classification_PR_AUC":
                pr_auc,
            "high_demand_threshold":
                high_demand_threshold,
            "probability_threshold":
                best_threshold
        }
    ]
)

results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "week10_final_evaluation.csv"
    ),
    index=False
)


# ============================================================
# 19. SAVE MODEL BUNDLE
# ============================================================

model_bundle = {
    "feature_columns":
        feature_columns,

    "preprocessor":
        preprocessor,

    "regressor":
        regressor,

    "classifier":
        classifier,

    "high_demand_threshold":
        high_demand_threshold,

    "classification_threshold":
        best_threshold
}

model_path = os.path.join(
    MODELS_DIR,
    "week10_model_bundle.joblib"
)

joblib.dump(
    model_bundle,
    model_path
)


# ============================================================
# 20. FINAL
# ============================================================

print("\n" + "=" * 70)
print("WEEK 10 COMPLETED")
print("=" * 70)

print(
    """
✓ Strict chronological train/validation/test split
✓ Final XGBoost regression evaluation
✓ Final high-demand classification evaluation
✓ Probability threshold selection
✓ Precision-Recall analysis
✓ ROC analysis
✓ Calibration analysis
✓ Final reusable model bundle

Graphs saved in:
reports/figures/

Model saved in:
models/week10_model_bundle.joblib
"""
)

print("=" * 70)