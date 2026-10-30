# ============================================================
# DemandIQ
# Week 12 - Replenishment Decision Support
# ============================================================

import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "week10_model_bundle.joblib"
)

ANOMALY_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "results",
    "week11_detected_anomalies.csv"
)

FAMILY_CLUSTER_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "results",
    "week9_family_clusters.csv"
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
# 2. CONFIGURATION
# ============================================================

DEMAND_WEIGHT = 0.35
HIGH_DEMAND_WEIGHT = 0.30
VOLATILITY_WEIGHT = 0.15
ANOMALY_WEIGHT = 0.10
PROMOTION_WEIGHT = 0.10


# ============================================================
# 3. START
# ============================================================

print("\n" + "=" * 70)
print("DEMANDIQ - WEEK 12")
print("Replenishment Decision Support")
print("=" * 70)


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\nLoading processed dataset...")

data = pd.read_csv(
    DATA_PATH
)

data["date"] = pd.to_datetime(
    data["date"]
)

print(
    f"Dataset loaded: "
    f"{len(data):,} rows"
)


# ============================================================
# 5. LOAD WEEK 10 MODEL
# ============================================================

print(
    "\nLoading Week 10 model..."
)

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        "\nWeek 10 model bundle was not found.\n"
        f"Expected file:\n{MODEL_PATH}\n\n"
        "Run Week 10 successfully first."
    )

model_bundle = joblib.load(
    MODEL_PATH
)

feature_columns = (
    model_bundle[
        "feature_columns"
    ]
)

preprocessor = (
    model_bundle[
        "preprocessor"
    ]
)

regressor = (
    model_bundle[
        "regressor"
    ]
)

classifier = (
    model_bundle[
        "classifier"
    ]
)

print(
    "✓ Week 10 model bundle loaded."
)


# ============================================================
# 6. LATEST DATE
# ============================================================

latest_date = (
    data["date"].max()
)

latest = data[
    data["date"] == latest_date
].copy()

print(
    f"\nDecision-support date: "
    f"{latest_date.date()}"
)

print(
    f"Rows scored: "
    f"{len(latest):,}"
)


# ============================================================
# 7. CHECK REQUIRED FEATURES
# ============================================================

missing_features = [
    column
    for column
    in feature_columns
    if column
    not in latest.columns
]

if missing_features:

    raise ValueError(
        "\nWeek 12 is missing features required "
        "by the Week 10 model:\n"
        f"{missing_features}"
    )


# ============================================================
# 8. MODEL PREDICTIONS
# ============================================================

print(
    "\nGenerating demand forecasts "
    "and high-demand probabilities..."
)

X_latest = (
    preprocessor.transform(
        latest[
            feature_columns
        ]
    )
)

latest[
    "predicted_demand"
] = np.maximum(
    regressor.predict(
        X_latest
    ),
    0
)

latest[
    "high_demand_probability"
] = (
    classifier.predict_proba(
        X_latest
    )[:, 1]
)

print(
    "✓ Predictions generated."
)


# ============================================================
# 9. HISTORICAL DEMAND VOLATILITY
# ============================================================

print(
    "\nCalculating historical demand volatility..."
)

volatility = (
    data
    .groupby(
        [
            "store_nbr",
            "family"
        ]
    )["sales"]
    .std()
    .fillna(0)
    .reset_index(
        name="demand_volatility"
    )
)

latest = latest.merge(
    volatility,
    on=[
        "store_nbr",
        "family"
    ],
    how="left"
)

latest[
    "demand_volatility"
] = (
    latest[
        "demand_volatility"
    ]
    .fillna(0)
)

print(
    "✓ Demand volatility calculated."
)


# ============================================================
# 10. WEEK 11 ANOMALY SIGNAL
# ============================================================

latest[
    "anomaly_signal"
] = 0

if os.path.exists(
    ANOMALY_PATH
):

    print(
        "\nLoading Week 11 anomaly results..."
    )

    anomaly_data = pd.read_csv(
        ANOMALY_PATH
    )

    # Best matching method:
    # use ID if available.
    if (
        "id" in latest.columns
        and
        "id" in anomaly_data.columns
    ):

        anomaly_ids = set(
            anomaly_data[
                "id"
            ]
        )

        latest[
            "anomaly_signal"
        ] = (
            latest[
                "id"
            ]
            .isin(
                anomaly_ids
            )
            .astype(int)
        )

    # Fallback:
    # store + family + date
    elif all(
        column in anomaly_data.columns
        for column in [
            "date",
            "store_nbr",
            "family"
        ]
    ):

        anomaly_data[
            "date"
        ] = pd.to_datetime(
            anomaly_data[
                "date"
            ]
        )

        anomaly_keys = set(
            zip(
                anomaly_data[
                    "date"
                ],
                anomaly_data[
                    "store_nbr"
                ],
                anomaly_data[
                    "family"
                ]
            )
        )

        latest[
            "anomaly_signal"
        ] = [
            1
            if (
                row["date"],
                row["store_nbr"],
                row["family"]
            )
            in anomaly_keys
            else 0
            for _, row
            in latest.iterrows()
        ]

    print(
        "✓ Week 11 anomaly signal loaded."
    )

else:

    print(
        "\nWarning: Week 11 anomaly file "
        "was not found."
    )

    print(
        "Anomaly signal will remain 0."
    )


# ============================================================
# 11. WEEK 9 BEHAVIOURAL SEGMENT
# ============================================================

latest[
    "behaviour_cluster"
] = -1

if os.path.exists(
    FAMILY_CLUSTER_PATH
):

    print(
        "\nLoading Week 9 family "
        "behavioural segments..."
    )

    cluster_data = pd.read_csv(
        FAMILY_CLUSTER_PATH
    )

    if (
        "family"
        in cluster_data.columns
        and
        "cluster"
        in cluster_data.columns
    ):

        # IMPORTANT:
        # Original Favorita data already has a
        # "cluster" column for store metadata.
        #
        # Therefore rename the Week 9 K-Means
        # cluster BEFORE merging.

        cluster_map = (
            cluster_data[
                [
                    "family",
                    "cluster"
                ]
            ]
            .drop_duplicates(
                subset="family"
            )
            .rename(
                columns={
                    "cluster":
                        "behaviour_cluster_week9"
                }
            )
        )

        latest = latest.merge(
            cluster_map,
            on="family",
            how="left"
        )

        latest[
            "behaviour_cluster"
        ] = (
            latest[
                "behaviour_cluster_week9"
            ]
            .fillna(-1)
            .astype(int)
        )

        latest = latest.drop(
            columns=[
                "behaviour_cluster_week9"
            ]
        )

        print(
            "✓ Week 9 behavioural "
            "segments loaded."
        )

    else:

        print(
            "Warning: Week 9 family cluster "
            "file does not contain both "
            "'family' and 'cluster'."
        )

else:

    print(
        "\nWarning: Week 9 family cluster "
        "file was not found."
    )

    print(
        "Behaviour cluster will remain -1."
    )


# ============================================================
# 12. NORMALIZATION HELPER
# ============================================================

def percentile_score(
    series
):

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    return (
        numeric
        .rank(
            method="average",
            pct=True
        )
        .fillna(0)
    )


# ============================================================
# 13. COMPONENT SCORES
# ============================================================

latest[
    "demand_score"
] = percentile_score(
    latest[
        "predicted_demand"
    ]
)

latest[
    "risk_score"
] = (
    latest[
        "high_demand_probability"
    ]
    .clip(
        0,
        1
    )
)

latest[
    "volatility_score"
] = percentile_score(
    latest[
        "demand_volatility"
    ]
)

latest[
    "promotion_score"
] = (
    latest[
        "onpromotion"
    ] > 0
).astype(int)


# ============================================================
# 14. REPLENISHMENT PRIORITY SCORE
# ============================================================

latest[
    "replenishment_priority_score"
] = (
    DEMAND_WEIGHT
    * latest[
        "demand_score"
    ]
    +
    HIGH_DEMAND_WEIGHT
    * latest[
        "risk_score"
    ]
    +
    VOLATILITY_WEIGHT
    * latest[
        "volatility_score"
    ]
    +
    ANOMALY_WEIGHT
    * latest[
        "anomaly_signal"
    ]
    +
    PROMOTION_WEIGHT
    * latest[
        "promotion_score"
    ]
)


# ============================================================
# 15. PRIORITY LABEL
# ============================================================

def assign_priority(
    score
):

    if score >= 0.75:

        return "CRITICAL"

    elif score >= 0.55:

        return "HIGH"

    elif score >= 0.35:

        return "MEDIUM"

    else:

        return "LOW"


latest[
    "priority"
] = (
    latest[
        "replenishment_priority_score"
    ]
    .apply(
        assign_priority
    )
)


# ============================================================
# 16. RECOMMENDED ACTION
# ============================================================

action_map = {

    "CRITICAL":
        "Review / replenish first",

    "HIGH":
        "Prioritize replenishment",

    "MEDIUM":
        "Monitor closely",

    "LOW":
        "Normal monitoring"
}

latest[
    "recommended_action"
] = (
    latest[
        "priority"
    ]
    .map(
        action_map
    )
)


# ============================================================
# 17. SORT PRIORITY TABLE
# ============================================================

priority_table = (
    latest
    .sort_values(
        "replenishment_priority_score",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 18. PRINT PRIORITY DISTRIBUTION
# ============================================================

print(
    "\nPriority distribution:"
)

print(
    priority_table[
        "priority"
    ]
    .value_counts()
)


# ============================================================
# 19. GRAPH 1
# TOP REPLENISHMENT PRIORITIES
# ============================================================

top = priority_table.head(
    20
).copy()

top[
    "label"
] = (
    "Store "
    + top[
        "store_nbr"
    ].astype(str)
    + " - "
    + top[
        "family"
    ].astype(str)
)

plt.figure(
    figsize=(12, 8)
)

plt.barh(
    top[
        "label"
    ][::-1],
    top[
        "replenishment_priority_score"
    ][::-1]
)

plt.xlabel(
    "Replenishment Priority Score"
)

plt.ylabel(
    "Store - Product Family"
)

plt.title(
    "Week 12 - Top Replenishment Priorities"
)

plt.tight_layout()

top_path = os.path.join(
    FIGURES_DIR,
    "week12_top_replenishment_priorities.png"
)

plt.savefig(
    top_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {top_path}"
)


# ============================================================
# 20. GRAPH 2
# PRIORITY DISTRIBUTION
# ============================================================

priority_order = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL"
]

priority_counts = (
    priority_table[
        "priority"
    ]
    .value_counts()
    .reindex(
        priority_order,
        fill_value=0
    )
)

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    priority_counts.index,
    priority_counts.values
)

plt.ylabel(
    "Number of Store-Family Records"
)

plt.xlabel(
    "Priority Level"
)

plt.title(
    "Week 12 - Replenishment Priority Distribution"
)

plt.tight_layout()

distribution_path = os.path.join(
    FIGURES_DIR,
    "week12_priority_distribution.png"
)

plt.savefig(
    distribution_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {distribution_path}"
)


# ============================================================
# 21. GRAPH 3
# PRIORITY BY STORE
# ============================================================

store_priority = (
    priority_table
    .groupby(
        "store_nbr"
    )[
        "replenishment_priority_score"
    ]
    .mean()
    .sort_values(
        ascending=False
    )
    .head(15)
)

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    store_priority.index.astype(
        str
    ),
    store_priority.values
)

plt.xlabel(
    "Store Number"
)

plt.ylabel(
    "Average Priority Score"
)

plt.title(
    "Week 12 - Stores Requiring Most Attention"
)

plt.tight_layout()

store_path = os.path.join(
    FIGURES_DIR,
    "week12_priority_by_store.png"
)

plt.savefig(
    store_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {store_path}"
)


# ============================================================
# 22. GRAPH 4
# PRIORITY BY PRODUCT FAMILY
# ============================================================

family_priority = (
    priority_table
    .groupby(
        "family"
    )[
        "replenishment_priority_score"
    ]
    .mean()
    .sort_values(
        ascending=False
    )
    .head(12)
)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    family_priority.index[::-1],
    family_priority.values[::-1]
)

plt.xlabel(
    "Average Priority Score"
)

plt.ylabel(
    "Product Family"
)

plt.title(
    "Week 12 - Product Families Requiring Most Attention"
)

plt.tight_layout()

family_path = os.path.join(
    FIGURES_DIR,
    "week12_priority_by_family.png"
)

plt.savefig(
    family_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {family_path}"
)


# ============================================================
# 23. GRAPH 5
# STORE × FAMILY PRIORITY HEATMAP
# ============================================================

top_families = (
    family_priority
    .head(10)
    .index
)

top_stores = (
    store_priority
    .head(10)
    .index
)

heatmap_subset = (
    priority_table[
        priority_table[
            "family"
        ].isin(
            top_families
        )
        &
        priority_table[
            "store_nbr"
        ].isin(
            top_stores
        )
    ]
)

heatmap_data = (
    heatmap_subset
    .pivot_table(
        index="family",
        columns="store_nbr",
        values=
            "replenishment_priority_score",
        aggfunc="mean"
    )
    .fillna(0)
)

if (
    heatmap_data.shape[0] > 0
    and
    heatmap_data.shape[1] > 0
):

    plt.figure(
        figsize=(12, 7)
    )

    image = plt.imshow(
        heatmap_data.values,
        aspect="auto"
    )

    plt.colorbar(
        image,
        label="Priority Score"
    )

    plt.xticks(
        range(
            len(
                heatmap_data.columns
            )
        ),
        heatmap_data.columns,
        rotation=45
    )

    plt.yticks(
        range(
            len(
                heatmap_data.index
            )
        ),
        heatmap_data.index
    )

    plt.xlabel(
        "Store"
    )

    plt.ylabel(
        "Product Family"
    )

    plt.title(
        "Week 12 - Replenishment Priority Heatmap"
    )

    plt.tight_layout()

    heatmap_path = os.path.join(
        FIGURES_DIR,
        "week12_replenishment_heatmap.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {heatmap_path}"
    )


# ============================================================
# 24. SAVE FINAL PRIORITY TABLE
# ============================================================

output_columns = [
    "date",
    "store_nbr",
    "family",
    "predicted_demand",
    "high_demand_probability",
    "demand_volatility",
    "onpromotion",
    "anomaly_signal",
    "behaviour_cluster",
    "demand_score",
    "risk_score",
    "volatility_score",
    "promotion_score",
    "replenishment_priority_score",
    "priority",
    "recommended_action"
]

# Only keep columns that really exist.
output_columns = [
    column
    for column
    in output_columns
    if column
    in priority_table.columns
]

priority_output_path = os.path.join(
    RESULTS_DIR,
    "week12_replenishment_priorities.csv"
)

priority_table[
    output_columns
].to_csv(
    priority_output_path,
    index=False
)

print(
    f"\nSaved: {priority_output_path}"
)


# ============================================================
# 25. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "decision_support_date":
                latest_date,

            "rows_scored":
                len(
                    priority_table
                ),

            "critical_count":
                int(
                    (
                        priority_table[
                            "priority"
                        ]
                        == "CRITICAL"
                    ).sum()
                ),

            "high_count":
                int(
                    (
                        priority_table[
                            "priority"
                        ]
                        == "HIGH"
                    ).sum()
                ),

            "medium_count":
                int(
                    (
                        priority_table[
                            "priority"
                        ]
                        == "MEDIUM"
                    ).sum()
                ),

            "low_count":
                int(
                    (
                        priority_table[
                            "priority"
                        ]
                        == "LOW"
                    ).sum()
                ),

            "average_priority_score":
                float(
                    priority_table[
                        "replenishment_priority_score"
                    ].mean()
                ),

            "maximum_priority_score":
                float(
                    priority_table[
                        "replenishment_priority_score"
                    ].max()
                )
        }
    ]
)

summary_path = os.path.join(
    RESULTS_DIR,
    "week12_replenishment_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(
    f"Saved: {summary_path}"
)


# ============================================================
# 26. DISPLAY TOP 20
# ============================================================

print("\n" + "=" * 70)
print("TOP REPLENISHMENT PRIORITIES")
print("=" * 70)

display_columns = [
    "store_nbr",
    "family",
    "predicted_demand",
    "high_demand_probability",
    "replenishment_priority_score",
    "priority",
    "recommended_action"
]

print(
    priority_table[
        display_columns
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 27. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("WEEK 12 COMPLETED")
print("=" * 70)