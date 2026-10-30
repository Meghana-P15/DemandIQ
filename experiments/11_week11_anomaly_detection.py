# ============================================================
# DemandIQ
# Week 11 - Demand Anomaly Detection
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    IsolationForest
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
# 2. CONFIG
# ============================================================

RANDOM_STATE = 42

TRAIN_SAMPLE_SIZE = 250_000

CONTAMINATION = 0.02

RESIDUAL_Z_THRESHOLD = 3.0


# ============================================================
# 3. LOAD
# ============================================================

print("\n" + "=" * 70)
print("DEMANDIQ - WEEK 11")
print("Demand Anomaly Detection")
print("=" * 70)

print("\nLoading processed dataset...")

data = pd.read_csv(
    DATA_PATH
)

data["date"] = pd.to_datetime(
    data["date"]
)

data = data.sort_values(
    [
        "store_nbr",
        "family",
        "date"
    ]
).reset_index(
    drop=True
)

print(
    f"Dataset loaded: "
    f"{len(data):,} rows"
)


# ============================================================
# 4. RESIDUAL FEATURES
# ============================================================

data["expected_demand"] = (
    data["rolling_mean_7"]
)

data["demand_residual"] = (
    data["sales"]
    - data["expected_demand"]
)

group_stats = (
    data
    .groupby(
        [
            "store_nbr",
            "family"
        ]
    )["demand_residual"]
    .agg(
        residual_mean="mean",
        residual_std="std"
    )
    .reset_index()
)

data = data.merge(
    group_stats,
    on=[
        "store_nbr",
        "family"
    ],
    how="left"
)

data[
    "residual_std"
] = (
    data["residual_std"]
    .replace(0, np.nan)
    .fillna(
        data[
            "demand_residual"
        ].std()
    )
)

data["residual_z"] = (
    (
        data["demand_residual"]
        - data["residual_mean"]
    )
    / data["residual_std"]
)


# ============================================================
# 5. ISOLATION FOREST FEATURES
# ============================================================

anomaly_features = [
    "sales",
    "onpromotion",
    "transactions",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "demand_residual",
    "residual_z"
]

X = (
    data[
        anomaly_features
    ]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ============================================================
# 6. TRAIN SAMPLE
# ============================================================

if len(X) > TRAIN_SAMPLE_SIZE:

    sample_indices = (
        data.sample(
            n=TRAIN_SAMPLE_SIZE,
            random_state=
                RANDOM_STATE
        ).index
    )

    X_train = X.loc[
        sample_indices
    ]

else:

    X_train = X


# ============================================================
# 7. ISOLATION FOREST
# ============================================================

print(
    "\nTraining Isolation Forest..."
)

model = IsolationForest(
    n_estimators=120,
    contamination=
        CONTAMINATION,
    random_state=
        RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train
)

print(
    "Training completed."
)

print(
    "\nScoring full dataset..."
)

isolation_prediction = (
    model.predict(
        X
    )
)

isolation_score = (
    -model.decision_function(
        X
    )
)

data["isolation_anomaly"] = (
    isolation_prediction == -1
).astype(int)

data["anomaly_score"] = (
    isolation_score
)


# ============================================================
# 8. COMBINE RULES
# ============================================================

data[
    "residual_anomaly"
] = (
    data[
        "residual_z"
    ].abs()
    >= RESIDUAL_Z_THRESHOLD
).astype(int)

data["is_anomaly"] = (
    (
        data[
            "isolation_anomaly"
        ]
        == 1
    )
    |
    (
        data[
            "residual_anomaly"
        ]
        == 1
    )
).astype(int)


data["anomaly_type"] = np.where(
    (
        data["is_anomaly"] == 1
    )
    &
    (
        data["demand_residual"] > 0
    ),
    "Demand Spike",
    np.where(
        data["is_anomaly"] == 1,
        "Demand Drop",
        "Normal"
    )
)


anomaly_count = int(
    data["is_anomaly"].sum()
)

print(
    f"\nDetected anomalies: "
    f"{anomaly_count:,}"
)

print(
    f"Anomaly percentage: "
    f"{100 * anomaly_count / len(data):.2f}%"
)


# ============================================================
# 9. GRAPH - TIMELINE
# ============================================================

daily = (
    data
    .groupby("date")
    .agg(
        total_sales=("sales", "sum"),
        anomaly_count=(
            "is_anomaly",
            "sum"
        )
    )
)

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    daily.index,
    daily["total_sales"],
    label="Daily Sales"
)

anomaly_days = daily[
    daily[
        "anomaly_count"
    ] > 0
]

plt.scatter(
    anomaly_days.index,
    anomaly_days[
        "total_sales"
    ],
    s=12,
    label="Anomaly Days"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Total Sales"
)

plt.title(
    "Week 11 - Demand Anomalies Over Time"
)

plt.legend()

plt.tight_layout()

timeline_path = os.path.join(
    FIGURES_DIR,
    "week11_anomaly_timeline.png"
)

plt.savefig(
    timeline_path,
    dpi=150
)

plt.close()


# ============================================================
# 10. ANOMALY SCORE DISTRIBUTION
# ============================================================

plot_sample = (
    data["anomaly_score"]
    .sample(
        n=min(
            100_000,
            len(data)
        ),
        random_state=
            RANDOM_STATE
    )
)

plt.figure(
    figsize=(9, 5)
)

plt.hist(
    plot_sample,
    bins=60
)

plt.xlabel(
    "Anomaly Score"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Week 11 - Anomaly Score Distribution"
)

plt.tight_layout()

score_path = os.path.join(
    FIGURES_DIR,
    "week11_anomaly_score_distribution.png"
)

plt.savefig(
    score_path,
    dpi=150
)

plt.close()


# ============================================================
# 11. NORMAL VS ANOMALOUS
# ============================================================

comparison = (
    data
    .groupby(
        "is_anomaly"
    )["sales"]
    .mean()
)

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    [
        "Normal",
        "Anomalous"
    ],
    [
        comparison.get(
            0,
            0
        ),
        comparison.get(
            1,
            0
        )
    ]
)

plt.ylabel(
    "Average Sales"
)

plt.title(
    "Week 11 - Normal vs Anomalous Demand"
)

plt.tight_layout()

comparison_path = os.path.join(
    FIGURES_DIR,
    "week11_normal_vs_anomalous_demand.png"
)

plt.savefig(
    comparison_path,
    dpi=150
)

plt.close()


# ============================================================
# 12. ANOMALIES BY FAMILY
# ============================================================

family_anomalies = (
    data[
        data["is_anomaly"] == 1
    ]
    .groupby("family")
    .size()
    .sort_values(
        ascending=False
    )
    .head(10)
)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    family_anomalies.index[::-1],
    family_anomalies.values[::-1]
)

plt.xlabel(
    "Number of Anomalies"
)

plt.title(
    "Week 11 - Product Families with Most Anomalies"
)

plt.tight_layout()

family_path = os.path.join(
    FIGURES_DIR,
    "week11_anomalies_by_family.png"
)

plt.savefig(
    family_path,
    dpi=150
)

plt.close()


# ============================================================
# 13. ANOMALIES BY STORE
# ============================================================

store_anomalies = (
    data[
        data["is_anomaly"] == 1
    ]
    .groupby("store_nbr")
    .size()
    .sort_values(
        ascending=False
    )
    .head(15)
)

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    store_anomalies.index.astype(
        str
    ),
    store_anomalies.values
)

plt.xlabel(
    "Store"
)

plt.ylabel(
    "Anomalies"
)

plt.title(
    "Week 11 - Stores with Most Demand Anomalies"
)

plt.tight_layout()

store_path = os.path.join(
    FIGURES_DIR,
    "week11_anomalies_by_store.png"
)

plt.savefig(
    store_path,
    dpi=150
)

plt.close()


# ============================================================
# 14. SAVE ANOMALIES
# ============================================================

anomalies = data[
    data["is_anomaly"] == 1
].copy()

columns_to_save = [
    "id",
    "date",
    "store_nbr",
    "family",
    "sales",
    "anomaly_score",
    "residual_z",
    "anomaly_type"
]

columns_to_save = [
    column
    for column in columns_to_save
    if column in anomalies.columns
]

anomalies[
    columns_to_save
].to_csv(
    os.path.join(
        RESULTS_DIR,
        "week11_detected_anomalies.csv"
    ),
    index=False
)


# ============================================================
# 15. SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "total_rows":
                len(data),

            "anomaly_count":
                anomaly_count,

            "anomaly_percentage":
                (
                    100
                    * anomaly_count
                    / len(data)
                ),

            "isolation_contamination":
                CONTAMINATION,

            "residual_z_threshold":
                RESIDUAL_Z_THRESHOLD
        }
    ]
)

summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "week11_anomaly_summary.csv"
    ),
    index=False
)


# ============================================================
# 16. FINAL
# ============================================================

print("\n" + "=" * 70)
print("WEEK 11 COMPLETED")
print("=" * 70)

print(
    """
✓ Isolation Forest anomaly detection
✓ Residual-based anomaly detection
✓ Demand spike/drop identification
✓ Timeline visualization
✓ Score distribution
✓ Family anomaly analysis
✓ Store anomaly analysis

Graphs saved in:
reports/figures/

Results saved in:
reports/results/
"""
)

print("=" * 70)