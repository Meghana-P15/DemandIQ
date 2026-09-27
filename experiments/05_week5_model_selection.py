# ============================================================
# DemandIQ
# Week 5 - Model Selection & Cross-Validation
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge, Lasso

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
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

CV_SAMPLE_SIZE = 400_000

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]

LASSO_ALPHAS = [0.0001, 0.001, 0.01]


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("DEMANDIQ - WEEK 5")
print("Model Selection & Time-Aware Cross-Validation")
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
# 5. CREATE TIME PERIODS
# ============================================================

print("\n" + "=" * 70)
print("TIME-AWARE VALIDATION SETUP")
print("=" * 70)

unique_dates = np.sort(
    data["date"].unique()
)

main_split_index = int(
    len(unique_dates) * 0.85
)

final_train_end = unique_dates[
    main_split_index - 1
]

final_test_start = unique_dates[
    main_split_index
]

development_data = data[
    data["date"] < final_test_start
].copy()

final_test_data = data[
    data["date"] >= final_test_start
].copy()

print(
    f"\nDevelopment period:"
    f"\n{development_data['date'].min().date()}"
    f" → "
    f"{development_data['date'].max().date()}"
)

print(
    f"\nFinal test period:"
    f"\n{final_test_data['date'].min().date()}"
    f" → "
    f"{final_test_data['date'].max().date()}"
)

print(
    f"\nDevelopment rows: "
    f"{len(development_data):,}"
)

print(
    f"Final test rows   : "
    f"{len(final_test_data):,}"
)


# ============================================================
# 6. SAMPLE DEVELOPMENT DATA
# ============================================================

print("\n" + "=" * 70)
print("MODEL SELECTION SAMPLE")
print("=" * 70)

if len(development_data) > CV_SAMPLE_SIZE:

    model_data = development_data.sample(
        n=CV_SAMPLE_SIZE,
        random_state=42
    ).sort_values(
        "date"
    ).reset_index(drop=True)

else:

    model_data = development_data.copy()

print(
    f"Rows used for model selection: "
    f"{len(model_data):,}"
)


# ============================================================
# 7. TIME-AWARE CROSS-VALIDATION
# ============================================================

dates = np.sort(
    model_data["date"].unique()
)

n_dates = len(dates)

fold1_train_end = int(n_dates * 0.60)
fold1_valid_end = int(n_dates * 0.70)

fold2_train_end = int(n_dates * 0.70)
fold2_valid_end = int(n_dates * 0.80)

fold3_train_end = int(n_dates * 0.80)
fold3_valid_end = int(n_dates * 0.90)

fold_boundaries = [
    (
        fold1_train_end,
        fold1_valid_end
    ),
    (
        fold2_train_end,
        fold2_valid_end
    ),
    (
        fold3_train_end,
        fold3_valid_end
    )
]


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


def evaluate_model(
    model_type,
    alpha,
    train_df,
    valid_df
):

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        TARGET
    ]

    X_valid = valid_df[
        feature_columns
    ]

    y_valid = valid_df[
        TARGET
    ]

    if model_type == "Ridge":

        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor()
                ),
                (
                    "regressor",
                    Ridge(
                        alpha=alpha
                    )
                )
            ]
        )

    elif model_type == "Lasso":

        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor()
                ),
                (
                    "regressor",
                    Lasso(
                        alpha=alpha,
                        max_iter=2000,
                        tol=0.001
                    )
                )
            ]
        )

    else:

        raise ValueError(
            "Unknown model type."
        )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_valid
    )

    predictions = np.maximum(
        predictions,
        0
    )

    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            predictions
        )
    )

    r2 = r2_score(
        y_valid,
        predictions
    )

    return mae, rmse, r2


# ============================================================
# 8. RIDGE CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RIDGE CROSS-VALIDATION")
print("=" * 70)

ridge_cv_results = []

for alpha in RIDGE_ALPHAS:

    print(
        f"\nTesting Ridge alpha = {alpha}"
    )

    fold_results = []

    for fold_number, (
        train_end,
        valid_end
    ) in enumerate(
        fold_boundaries,
        start=1
    ):

        train_end_date = dates[
            train_end - 1
        ]

        valid_start_date = dates[
            train_end
        ]

        valid_end_date = dates[
            valid_end - 1
        ]

        fold_train = model_data[
            model_data["date"]
            <= train_end_date
        ]

        fold_valid = model_data[
            (
                model_data["date"]
                >= valid_start_date
            )
            &
            (
                model_data["date"]
                <= valid_end_date
            )
        ]

        print(
            f"  Fold {fold_number}: "
            f"train={len(fold_train):,}, "
            f"validation={len(fold_valid):,}"
        )

        mae, rmse, r2 = evaluate_model(
            "Ridge",
            alpha,
            fold_train,
            fold_valid
        )

        print(
            f"    MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"R²={r2:.4f}"
        )

        fold_results.append(
            (
                mae,
                rmse,
                r2
            )
        )

    mean_mae = np.mean(
        [x[0] for x in fold_results]
    )

    mean_rmse = np.mean(
        [x[1] for x in fold_results]
    )

    mean_r2 = np.mean(
        [x[2] for x in fold_results]
    )

    std_mae = np.std(
        [x[0] for x in fold_results],
        ddof=1
    )

    ridge_cv_results.append(
        {
            "model": "Ridge",
            "alpha": alpha,
            "mean_MAE": mean_mae,
            "std_MAE": std_mae,
            "mean_RMSE": mean_rmse,
            "mean_R2": mean_r2
        }
    )


# ============================================================
# 9. LASSO CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("LASSO CROSS-VALIDATION")
print("=" * 70)

print(
    "\nLasso uses fewer candidate values because "
    "its optimization is more computationally expensive."
)

lasso_cv_results = []

for alpha in LASSO_ALPHAS:

    print(
        f"\nTesting Lasso alpha = {alpha}"
    )

    fold_results = []

    for fold_number, (
        train_end,
        valid_end
    ) in enumerate(
        fold_boundaries,
        start=1
    ):

        train_end_date = dates[
            train_end - 1
        ]

        valid_start_date = dates[
            train_end
        ]

        valid_end_date = dates[
            valid_end - 1
        ]

        fold_train = model_data[
            model_data["date"]
            <= train_end_date
        ]

        fold_valid = model_data[
            (
                model_data["date"]
                >= valid_start_date
            )
            &
            (
                model_data["date"]
                <= valid_end_date
            )
        ]

        print(
            f"  Fold {fold_number}: "
            f"train={len(fold_train):,}, "
            f"validation={len(fold_valid):,}"
        )

        mae, rmse, r2 = evaluate_model(
            "Lasso",
            alpha,
            fold_train,
            fold_valid
        )

        print(
            f"    MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"R²={r2:.4f}"
        )

        fold_results.append(
            (
                mae,
                rmse,
                r2
            )
        )

    mean_mae = np.mean(
        [x[0] for x in fold_results]
    )

    mean_rmse = np.mean(
        [x[1] for x in fold_results]
    )

    mean_r2 = np.mean(
        [x[2] for x in fold_results]
    )

    std_mae = np.std(
        [x[0] for x in fold_results],
        ddof=1
    )

    lasso_cv_results.append(
        {
            "model": "Lasso",
            "alpha": alpha,
            "mean_MAE": mean_mae,
            "std_MAE": std_mae,
            "mean_RMSE": mean_rmse,
            "mean_R2": mean_r2
        }
    )


# ============================================================
# 10. COMBINE CV RESULTS
# ============================================================

cv_results = pd.DataFrame(
    ridge_cv_results +
    lasso_cv_results
)

print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(
    cv_results.to_string(
        index=False
    )
)


# ============================================================
# 11. GRAPH 1 - CV MAE COMPARISON
# ============================================================

print(
    "\nGenerating cross-validation comparison graph..."
)

plt.figure(figsize=(10, 6))

for model_name in ["Ridge", "Lasso"]:

    model_results = cv_results[
        cv_results["model"] == model_name
    ]

    plt.plot(
        model_results["alpha"],
        model_results["mean_MAE"],
        marker="o",
        label=model_name
    )

plt.xlabel("Alpha")
plt.ylabel("Mean Cross-Validation MAE")
plt.title(
    "DemandIQ - Model Selection: CV MAE vs Alpha"
)

plt.xscale("log")

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

cv_graph_path = os.path.join(
    FIGURES_DIR,
    "week5_cv_mae_comparison.png"
)

plt.savefig(
    cv_graph_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {cv_graph_path}"
)


# ============================================================
# 12. SELECT BEST CONFIGURATION
# ============================================================

best_row = cv_results.loc[
    cv_results["mean_MAE"].idxmin()
]

best_model = best_row[
    "model"
]

best_alpha = best_row[
    "alpha"
]

best_mean_mae = best_row[
    "mean_MAE"
]

best_std_mae = best_row[
    "std_MAE"
]


print("\n" + "=" * 70)
print("BEST CROSS-VALIDATED CONFIGURATION")
print("=" * 70)

print(
    f"\nModel : {best_model}"
)

print(
    f"Alpha : {best_alpha}"
)

print(
    f"Mean CV MAE : "
    f"{best_mean_mae:.4f}"
)

print(
    f"CV MAE Std  : "
    f"{best_std_mae:.4f}"
)


# ============================================================
# 13. ONE-STANDARD-ERROR RULE
# ============================================================

print("\n" + "=" * 70)
print("ONE-STANDARD-ERROR RULE")
print("=" * 70)

one_se_limit = (
    best_mean_mae +
    best_std_mae
)

eligible_models = cv_results[
    cv_results["mean_MAE"]
    <= one_se_limit
].copy()

print(
    f"\nBest mean MAE: "
    f"{best_mean_mae:.4f}"
)

print(
    f"One-standard-error limit: "
    f"{one_se_limit:.4f}"
)

print(
    "\nConfigurations within one standard error:"
)

print(
    eligible_models[
        [
            "model",
            "alpha",
            "mean_MAE",
            "std_MAE"
        ]
    ].to_string(
        index=False
    )
)

if "Ridge" in eligible_models[
    "model"
].values:

    ridge_candidates = eligible_models[
        eligible_models["model"]
        == "Ridge"
    ]

    selected_row = ridge_candidates.loc[
        ridge_candidates["mean_MAE"].idxmin()
    ]

else:

    selected_row = eligible_models.loc[
        eligible_models["mean_MAE"].idxmin()
    ]

selected_model = selected_row[
    "model"
]

selected_alpha = selected_row[
    "alpha"
]


print(
    f"\nSelected baseline:"
)

print(
    f"Model = {selected_model}"
)

print(
    f"Alpha = {selected_alpha}"
)


# ============================================================
# 14. TRAIN SELECTED MODEL
# ============================================================

print("\n" + "=" * 70)
print("FINAL BASELINE TRAINING")
print("=" * 70)

X_development = development_data[
    feature_columns
]

y_development = development_data[
    TARGET
]

X_final_test = final_test_data[
    feature_columns
]

y_final_test = final_test_data[
    TARGET
]

if selected_model == "Ridge":

    final_model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "regressor",
                Ridge(
                    alpha=selected_alpha
                )
            )
        ]
    )

else:

    final_model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "regressor",
                Lasso(
                    alpha=selected_alpha,
                    max_iter=2000,
                    tol=0.001
                )
            )
        ]
    )

print(
    f"\nTraining final {selected_model} "
    f"model on {len(development_data):,} rows..."
)

final_model.fit(
    X_development,
    y_development
)

print(
    "Final baseline training completed."
)


# ============================================================
# 15. FINAL TEST PREDICTIONS
# ============================================================

print(
    "\nEvaluating on untouched final test period..."
)

final_predictions = final_model.predict(
    X_final_test
)

final_predictions = np.maximum(
    final_predictions,
    0
)


# ============================================================
# 16. FINAL METRICS
# ============================================================

final_mae = mean_absolute_error(
    y_final_test,
    final_predictions
)

final_rmse = np.sqrt(
    mean_squared_error(
        y_final_test,
        final_predictions
    )
)

final_r2 = r2_score(
    y_final_test,
    final_predictions
)


print("\n" + "=" * 70)
print("FINAL BASELINE RESULTS")
print("=" * 70)

print(
    f"\nModel: "
    f"{selected_model}"
)

print(
    f"Alpha: "
    f"{selected_alpha}"
)

print(
    f"\nMAE  : "
    f"{final_mae:.4f}"
)

print(
    f"RMSE : "
    f"{final_rmse:.4f}"
)

print(
    f"R²   : "
    f"{final_r2:.4f}"
)


# ============================================================
# 17. GRAPH 2 - ACTUAL VS PREDICTED
# ============================================================

print(
    "\nGenerating actual vs predicted graph..."
)

# Use a small sample for visualization so the graph
# remains readable instead of plotting 443,718 points.

plot_sample_size = min(
    5000,
    len(y_final_test)
)

plot_indices = np.linspace(
    0,
    len(y_final_test) - 1,
    plot_sample_size
).astype(int)

actual_sample = y_final_test.iloc[
    plot_indices
].values

predicted_sample = final_predictions[
    plot_indices
]

plt.figure(figsize=(10, 6))

plt.scatter(
    actual_sample,
    predicted_sample,
    alpha=0.25,
    s=12
)

max_value = max(
    actual_sample.max(),
    predicted_sample.max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.title(
    f"DemandIQ - Actual vs Predicted Sales ({selected_model})"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

actual_predicted_path = os.path.join(
    FIGURES_DIR,
    "week5_actual_vs_predicted.png"
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
# 18. SAVE CROSS-VALIDATION RESULTS
# ============================================================

cv_results_path = os.path.join(
    RESULTS_DIR,
    "week5_cross_validation_results.csv"
)

cv_results.to_csv(
    cv_results_path,
    index=False
)

print(
    "\nCross-validation results saved to:"
)

print(
    cv_results_path
)


# ============================================================
# 19. SAVE FINAL RESULTS
# ============================================================

final_results = pd.DataFrame(
    [
        {
            "selected_model":
                selected_model,
            "selected_alpha":
                selected_alpha,
            "final_MAE":
                final_mae,
            "final_RMSE":
                final_rmse,
            "final_R2":
                final_r2,
            "cv_mean_MAE":
                best_mean_mae,
            "cv_std_MAE":
                best_std_mae,
            "development_rows":
                len(development_data),
            "final_test_rows":
                len(final_test_data)
        }
    ]
)

final_results_path = os.path.join(
    RESULTS_DIR,
    "week5_final_baseline.csv"
)

final_results.to_csv(
    final_results_path,
    index=False
)

print(
    "\nFinal baseline saved to:"
)

print(
    final_results_path
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WEEK 5 COMPLETED")
print("=" * 70)

print(
    """
✓ Time-aware cross-validation
✓ Ridge hyperparameter comparison
✓ Lasso hyperparameter comparison
✓ One-standard-error analysis
✓ Baseline model selection
✓ Final evaluation on untouched future period
✓ CV comparison graph
✓ Actual vs predicted graph
✓ Cross-validation results saved
✓ Final baseline results saved

The final test period was NOT used during
hyperparameter selection.
"""
)

print("=" * 70)