# ============================================================
# DemandIQ
# Week 8 - XGBoost Champion
# ============================================================

import os
import warnings

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
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. IMPORT XGBOOST
# ============================================================

try:

    from xgboost import (
        XGBRegressor,
        XGBClassifier
    )

except ImportError:

    raise ImportError(
        "\nXGBoost is not installed.\n"
        "Run:\n"
        "pip install xgboost\n"
    )


# ============================================================
# 2. PATHS
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

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

TARGET = "sales"

XGB_SAMPLE_SIZE = 400_000

RANDOM_STATE = 42


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("DEMANDIQ - WEEK 8")
print("XGBoost Champion Model")
print("=" * 70)

print(
    "\nLoading processed dataset..."
)

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
    f"{len(data):,} rows × "
    f"{data.shape[1]} columns"
)


# ============================================================
# 5. FEATURES
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
# 6. TEMPORAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

unique_dates = np.sort(
    data["date"].unique()
)

validation_index = int(
    len(unique_dates) * 0.75
)

test_index = int(
    len(unique_dates) * 0.85
)

validation_start = unique_dates[
    validation_index
]

test_start = unique_dates[
    test_index
]

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

print(
    f"\nTraining period:"
    f"\n{train_data['date'].min().date()}"
    f" → "
    f"{train_data['date'].max().date()}"
)

print(
    f"\nValidation period:"
    f"\n{validation_data['date'].min().date()}"
    f" → "
    f"{validation_data['date'].max().date()}"
)

print(
    f"\nTest period:"
    f"\n{test_data['date'].min().date()}"
    f" → "
    f"{test_data['date'].max().date()}"
)


# ============================================================
# 7. REPRESENTATIVE TRAINING SAMPLE
# ============================================================

if len(train_data) > XGB_SAMPLE_SIZE:

    model_train = train_data.sample(
        n=XGB_SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

else:

    model_train = train_data.copy()

print(
    f"\nXGBoost training rows: "
    f"{len(model_train):,}"
)

print(
    f"Validation rows       : "
    f"{len(validation_data):,}"
)

print(
    f"Test rows             : "
    f"{len(test_data):,}"
)


# ============================================================
# 8. PREPROCESSING
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

print(
    "\nFitting preprocessing pipeline..."
)

X_train_raw = model_train[
    feature_columns
]

X_valid_raw = validation_data[
    feature_columns
]

X_test_raw = test_data[
    feature_columns
]

X_train = preprocessor.fit_transform(
    X_train_raw
)

X_valid = preprocessor.transform(
    X_valid_raw
)

X_test = preprocessor.transform(
    X_test_raw
)

feature_names = (
    preprocessor
    .get_feature_names_out()
)

print(
    "Preprocessing completed."
)


# ============================================================
# 9. XGBOOST REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST REGRESSION")
print("=" * 70)

y_train_reg = model_train[
    TARGET
]

y_valid_reg = validation_data[
    TARGET
]

y_test_reg = test_data[
    TARGET
]

xgb_regressor = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.80,
    colsample_bytree=0.80,
    reg_alpha=0.05,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    early_stopping_rounds=50
)

print(
    "\nTraining XGBoost Regressor..."
)

xgb_regressor.fit(
    X_train,
    y_train_reg,
    eval_set=[
        (
            X_valid,
            y_valid_reg
        )
    ],
    verbose=False
)

print(
    "Training completed."
)

reg_predictions = (
    xgb_regressor
    .predict(
        X_test
    )
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

print("\nXGBoost Regression Results")

print(
    f"MAE  : {reg_mae:.4f}"
)

print(
    f"RMSE : {reg_rmse:.4f}"
)

print(
    f"R²   : {reg_r2:.4f}"
)

print(
    f"Best iteration: "
    f"{xgb_regressor.best_iteration}"
)


# ============================================================
# 10. REGRESSION FEATURE IMPORTANCE
# ============================================================

reg_importance = pd.DataFrame(
    {
        "feature":
            feature_names,
        "importance":
            xgb_regressor
            .feature_importances_
    }
)

reg_importance = (
    reg_importance
    .sort_values(
        "importance",
        ascending=False
    )
)

reg_importance.to_csv(
    os.path.join(
        RESULTS_DIR,
        "week8_xgb_regression_feature_importance.csv"
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
    "Week 8 - XGBoost Regression Feature Importance"
)

plt.tight_layout()

reg_importance_path = os.path.join(
    FIGURES_DIR,
    "week8_xgb_regression_feature_importance.png"
)

plt.savefig(
    reg_importance_path,
    dpi=150
)

plt.close()

print(
    f"\nSaved: "
    f"{reg_importance_path}"
)


# ============================================================
# 11. CLASSIFICATION TARGET
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST HIGH-DEMAND CLASSIFICATION")
print("=" * 70)

# Threshold uses pre-test development data only.
development_data = data[
    data["date"] < test_start
]

high_demand_threshold = (
    development_data[
        TARGET
    ].quantile(0.75)
)

print(
    f"\nHigh-demand threshold: "
    f"{high_demand_threshold:.2f}"
)

y_train_cls = (
    model_train[TARGET]
    >= high_demand_threshold
).astype(int)

y_valid_cls = (
    validation_data[TARGET]
    >= high_demand_threshold
).astype(int)

y_test_cls = (
    test_data[TARGET]
    >= high_demand_threshold
).astype(int)


# ============================================================
# 12. CLASS IMBALANCE WEIGHT
# ============================================================

positive_count = y_train_cls.sum()

negative_count = (
    len(y_train_cls)
    - positive_count
)

if positive_count > 0:

    scale_pos_weight = (
        negative_count /
        positive_count
    )

else:

    scale_pos_weight = 1.0

print(
    f"scale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# 13. XGBOOST CLASSIFIER
# ============================================================

xgb_classifier = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.80,
    colsample_bytree=0.80,
    reg_alpha=0.05,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    scale_pos_weight=
        scale_pos_weight,
    early_stopping_rounds=50
)

print(
    "\nTraining XGBoost Classifier..."
)

xgb_classifier.fit(
    X_train,
    y_train_cls,
    eval_set=[
        (
            X_valid,
            y_valid_cls
        )
    ],
    verbose=False
)

print(
    "Training completed."
)


# ============================================================
# 14. CLASSIFICATION PREDICTIONS
# ============================================================

cls_probabilities = (
    xgb_classifier
    .predict_proba(
        X_test
    )[:, 1]
)

cls_predictions = (
    cls_probabilities >= 0.50
).astype(int)


# ============================================================
# 15. CLASSIFICATION METRICS
# ============================================================

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

print("\nXGBoost Classification Results")

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

print(
    f"PR-AUC    : {pr_auc:.4f}"
)

print(
    f"Best iteration: "
    f"{xgb_classifier.best_iteration}"
)


# ============================================================
# 16. CLASSIFICATION FEATURE IMPORTANCE
# ============================================================

cls_importance = pd.DataFrame(
    {
        "feature":
            feature_names,
        "importance":
            xgb_classifier
            .feature_importances_
    }
)

cls_importance = (
    cls_importance
    .sort_values(
        "importance",
        ascending=False
    )
)

cls_importance.to_csv(
    os.path.join(
        RESULTS_DIR,
        "week8_xgb_classification_feature_importance.csv"
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
    "Week 8 - XGBoost Classification Feature Importance"
)

plt.tight_layout()

cls_importance_path = os.path.join(
    FIGURES_DIR,
    "week8_xgb_classification_feature_importance.png"
)

plt.savefig(
    cls_importance_path,
    dpi=150
)

plt.close()

print(
    f"Saved: "
    f"{cls_importance_path}"
)


# ============================================================
# 17. CONFUSION MATRIX
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
    "Week 8 - XGBoost Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    FIGURES_DIR,
    "week8_xgb_confusion_matrix.png"
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
# 18. SHAP EXPLANATIONS
# ============================================================

print("\n" + "=" * 70)
print("SHAP EXPLANATIONS")
print("=" * 70)

shap_status = "Skipped"

try:

    import shap

    print(
        "\nGenerating SHAP explanations..."
    )

    SHAP_SAMPLE_SIZE = min(
        1500,
        X_test.shape[0]
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    shap_indices = rng.choice(
        X_test.shape[0],
        size=SHAP_SAMPLE_SIZE,
        replace=False
    )

    X_shap = X_test[
        shap_indices
    ]

    # Regression SHAP
    reg_explainer = shap.TreeExplainer(
        xgb_regressor
    )

    reg_shap_values = (
        reg_explainer
        .shap_values(
            X_shap
        )
    )

    shap.summary_plot(
        reg_shap_values,
        X_shap,
        feature_names=
            feature_names,
        show=False,
        max_display=12
    )

    plt.title(
        "Week 8 - XGBoost Regression SHAP Summary"
    )

    plt.tight_layout()

    reg_shap_path = os.path.join(
        FIGURES_DIR,
        "week8_xgb_regression_shap.png"
    )

    plt.savefig(
        reg_shap_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {reg_shap_path}"
    )

    # Classification SHAP
    cls_explainer = shap.TreeExplainer(
        xgb_classifier
    )

    cls_shap_values = (
        cls_explainer
        .shap_values(
            X_shap
        )
    )

    shap.summary_plot(
        cls_shap_values,
        X_shap,
        feature_names=
            feature_names,
        show=False,
        max_display=12
    )

    plt.title(
        "Week 8 - XGBoost Classification SHAP Summary"
    )

    plt.tight_layout()

    cls_shap_path = os.path.join(
        FIGURES_DIR,
        "week8_xgb_classification_shap.png"
    )

    plt.savefig(
        cls_shap_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {cls_shap_path}"
    )

    shap_status = "Completed"

except Exception as error:

    print(
        "\nSHAP could not be generated."
    )

    print(
        f"Reason: {error}"
    )

    print(
        "\nInstall SHAP with:"
    )

    print(
        "pip install shap"
    )


# ============================================================
# 19. SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    [
        {
            "task": "Regression",
            "model":
                "XGBoost Regressor",
            "MAE": reg_mae,
            "RMSE": reg_rmse,
            "R2": reg_r2,
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "ROC_AUC": np.nan,
            "PR_AUC": np.nan,
            "best_iteration":
                xgb_regressor.best_iteration
        },
        {
            "task": "Classification",
            "model":
                "XGBoost Classifier",
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "best_iteration":
                xgb_classifier.best_iteration
        }
    ]
)

results_path = os.path.join(
    RESULTS_DIR,
    "week8_xgboost_results.csv"
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
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WEEK 8 COMPLETED")
print("=" * 70)

print(
    f"""
Regression:
✓ XGBoost Regressor
✓ Early stopping
✓ MAE
✓ RMSE
✓ R²

Classification:
✓ XGBoost Classifier
✓ Early stopping
✓ Accuracy
✓ Precision
✓ Recall
✓ F1
✓ ROC-AUC
✓ PR-AUC

Interpretability:
✓ Regression feature importance
✓ Classification feature importance
✓ Classification confusion matrix
✓ SHAP status: {shap_status}

Week 8 establishes the XGBoost
champion-model candidate.
"""
)

print("=" * 70)