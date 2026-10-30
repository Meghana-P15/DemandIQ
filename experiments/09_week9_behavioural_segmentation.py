# ============================================================
# DemandIQ
# Week 9 - Behavioural Segmentation
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score
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
MIN_K = 2
MAX_K = 6


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("DEMANDIQ - WEEK 9")
print("Behavioural Segmentation")
print("=" * 70)

print("\nLoading processed dataset...")

data = pd.read_csv(DATA_PATH)

data["date"] = pd.to_datetime(
    data["date"]
)

print(
    f"Dataset loaded: "
    f"{len(data):,} rows × {data.shape[1]} columns"
)


# ============================================================
# 4. FEATURE BUILDING
# ============================================================

def build_entity_features(
    df,
    entity_column,
    entity_name
):

    print(
        f"\nBuilding {entity_name} behavioural features..."
    )

    grouped = df.groupby(
        entity_column
    )

    features = pd.DataFrame(
        index=grouped.size().index
    )

    features["average_demand"] = (
        grouped["sales"].mean()
    )

    features["demand_volatility"] = (
        grouped["sales"]
        .std()
        .fillna(0)
    )

    features["total_demand"] = (
        grouped["sales"].sum()
    )

    features["maximum_demand"] = (
        grouped["sales"].max()
    )

    features["zero_sales_rate"] = (
        grouped["sales"]
        .apply(
            lambda values:
            (values == 0).mean()
        )
    )

    features["promotion_rate"] = (
        grouped["onpromotion"]
        .apply(
            lambda values:
            (values > 0).mean()
        )
    )

    # Promotion sensitivity
    promo_sales = (
        df[df["onpromotion"] > 0]
        .groupby(entity_column)["sales"]
        .mean()
    )

    normal_sales = (
        df[df["onpromotion"] == 0]
        .groupby(entity_column)["sales"]
        .mean()
    )

    features["promotion_sales"] = promo_sales
    features["normal_sales"] = normal_sales

    features["promotion_sensitivity"] = (
        features["promotion_sales"]
        - features["normal_sales"]
    ) / (
        features["normal_sales"].abs()
        + 1
    )

    # Seasonality strength
    monthly = (
        df
        .assign(
            month=data["date"].dt.month
        )
        .groupby(
            [entity_column, "month"]
        )["sales"]
        .mean()
        .reset_index()
    )

    seasonal_std = (
        monthly
        .groupby(entity_column)["sales"]
        .std()
        .fillna(0)
    )

    features["seasonality_strength"] = (
        seasonal_std
        / (
            features["average_demand"].abs()
            + 1
        )
    )

    # Store-specific transaction behaviour
    if entity_column == "store_nbr":

        features["average_transactions"] = (
            grouped["transactions"]
            .mean()
        )

        features["product_diversity"] = (
            grouped["family"]
            .nunique()
        )

    features = (
        features
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        .reset_index()
    )

    return features


# ============================================================
# 5. CLUSTERING FUNCTION
# ============================================================

def cluster_entities(
    features,
    entity_column,
    prefix
):

    print("\n" + "=" * 70)
    print(
        f"{prefix.upper()} SEGMENTATION"
    )
    print("=" * 70)

    numeric_columns = [
        column
        for column in features.columns
        if column != entity_column
        and pd.api.types.is_numeric_dtype(
            features[column]
        )
    ]

    X = features[
        numeric_columns
    ].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    max_k = min(
        MAX_K,
        len(features) - 1
    )

    k_values = list(
        range(
            MIN_K,
            max_k + 1
        )
    )

    silhouette_scores = []

    print("\nTesting cluster counts...")

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20
        )

        labels = model.fit_predict(
            X_scaled
        )

        score = silhouette_score(
            X_scaled,
            labels
        )

        silhouette_scores.append(
            score
        )

        print(
            f"K={k} | "
            f"Silhouette={score:.4f}"
        )

    best_index = int(
        np.argmax(
            silhouette_scores
        )
    )

    best_k = k_values[
        best_index
    ]

    print(
        f"\nBest K = {best_k}"
    )


    # ========================================================
    # GRAPH 1 - K SELECTION
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        k_values,
        silhouette_scores,
        marker="o"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.title(
        f"Week 9 - {prefix.title()} Cluster Selection"
    )

    plt.tight_layout()

    k_path = os.path.join(
        FIGURES_DIR,
        f"week9_{prefix}_k_selection.png"
    )

    plt.savefig(
        k_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {k_path}"
    )


    # ========================================================
    # FINAL K-MEANS
    # ========================================================

    final_model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=30
    )

    labels = final_model.fit_predict(
        X_scaled
    )

    features["cluster"] = labels

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    davies_bouldin = davies_bouldin_score(
        X_scaled,
        labels
    )


    # ========================================================
    # STABILITY
    # ========================================================

    stability_scores = []

    for seed in range(5):

        temp_model = KMeans(
            n_clusters=best_k,
            random_state=seed,
            n_init=20
        )

        temp_labels = temp_model.fit_predict(
            X_scaled
        )

        stability_scores.append(
            adjusted_rand_score(
                labels,
                temp_labels
            )
        )

    stability = np.mean(
        stability_scores
    )

    print(
        f"\nSilhouette Score     : "
        f"{silhouette:.4f}"
    )

    print(
        f"Davies-Bouldin Index : "
        f"{davies_bouldin:.4f}"
    )

    print(
        f"Cluster Stability     : "
        f"{stability:.4f}"
    )


    # ========================================================
    # PCA
    # ========================================================

    pca = PCA(
        n_components=2
    )

    pca_values = pca.fit_transform(
        X_scaled
    )

    features["PCA_1"] = (
        pca_values[:, 0]
    )

    features["PCA_2"] = (
        pca_values[:, 1]
    )


    # ========================================================
    # GRAPH 2 - PCA
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    scatter = plt.scatter(
        features["PCA_1"],
        features["PCA_2"],
        c=features["cluster"],
        alpha=0.8
    )

    plt.colorbar(
        scatter,
        label="Cluster"
    )

    plt.xlabel(
        "Principal Component 1"
    )

    plt.ylabel(
        "Principal Component 2"
    )

    plt.title(
        f"Week 9 - {prefix.title()} Behavioural Segments"
    )

    plt.tight_layout()

    pca_path = os.path.join(
        FIGURES_DIR,
        f"week9_{prefix}_pca_clusters.png"
    )

    plt.savefig(
        pca_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {pca_path}"
    )


    # ========================================================
    # GRAPH 3 - CLUSTER PROFILES
    # ========================================================

    cluster_profile = (
        features
        .groupby("cluster")[
            numeric_columns
        ]
        .mean()
    )

    profile_scaled = (
        cluster_profile
        - cluster_profile.mean()
    ) / (
        cluster_profile.std(
            ddof=0
        ).replace(0, 1)
    )

    plt.figure(
        figsize=(
            max(
                10,
                len(numeric_columns)
            ),
            5
        )
    )

    image = plt.imshow(
        profile_scaled.values,
        aspect="auto"
    )

    plt.colorbar(
        image,
        label="Standardized Behaviour"
    )

    plt.xticks(
        range(
            len(numeric_columns)
        ),
        numeric_columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(
            len(cluster_profile)
        ),
        [
            f"Cluster {index}"
            for index
            in cluster_profile.index
        ]
    )

    plt.title(
        f"Week 9 - {prefix.title()} Cluster Profiles"
    )

    plt.tight_layout()

    profile_path = os.path.join(
        FIGURES_DIR,
        f"week9_{prefix}_cluster_profiles.png"
    )

    plt.savefig(
        profile_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {profile_path}"
    )


    # ========================================================
    # OPTIONAL DBSCAN
    # ========================================================

    dbscan = DBSCAN(
        eps=1.5,
        min_samples=3
    )

    dbscan_labels = (
        dbscan.fit_predict(
            X_scaled
        )
    )

    features[
        "dbscan_cluster"
    ] = dbscan_labels

    plt.figure(
        figsize=(9, 6)
    )

    scatter = plt.scatter(
        features["PCA_1"],
        features["PCA_2"],
        c=dbscan_labels,
        alpha=0.8
    )

    plt.colorbar(
        scatter,
        label="DBSCAN Cluster"
    )

    plt.title(
        f"Week 9 - {prefix.title()} DBSCAN Exploration"
    )

    plt.xlabel(
        "Principal Component 1"
    )

    plt.ylabel(
        "Principal Component 2"
    )

    plt.tight_layout()

    dbscan_path = os.path.join(
        FIGURES_DIR,
        f"week9_{prefix}_dbscan.png"
    )

    plt.savefig(
        dbscan_path,
        dpi=150
    )

    plt.close()


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    features.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"week9_{prefix}_clusters.csv"
        ),
        index=False
    )

    return {
        "entity": prefix,
        "best_k": best_k,
        "silhouette": silhouette,
        "davies_bouldin":
            davies_bouldin,
        "stability": stability
    }


# ============================================================
# 6. RUN FAMILY SEGMENTATION
# ============================================================

family_features = (
    build_entity_features(
        data,
        "family",
        "family"
    )
)

family_metrics = cluster_entities(
    family_features,
    "family",
    "family"
)


# ============================================================
# 7. RUN STORE SEGMENTATION
# ============================================================

store_features = (
    build_entity_features(
        data,
        "store_nbr",
        "store"
    )
)

store_metrics = cluster_entities(
    store_features,
    "store_nbr",
    "store"
)


# ============================================================
# 8. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        family_metrics,
        store_metrics
    ]
)

summary_path = os.path.join(
    RESULTS_DIR,
    "week9_clustering_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 9. FINAL
# ============================================================

print("\n" + "=" * 70)
print("WEEK 9 COMPLETED")
print("=" * 70)

print(
    """
✓ Family behavioural features
✓ Store behavioural features
✓ K-Means clustering
✓ Silhouette-based K selection
✓ Davies-Bouldin evaluation
✓ Cluster stability
✓ PCA visualization
✓ Cluster profile visualization
✓ DBSCAN exploration

All graphs saved in:
reports/figures/

All results saved in:
reports/results/
"""
)

print("=" * 70)