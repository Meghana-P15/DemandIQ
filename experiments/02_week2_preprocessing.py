# ============================================================
# DemandIQ
# Week 2 - Data Cleaning, Merging & Feature Engineering
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("DEMANDIQ - WEEK 2")
print("Data Cleaning & Feature Engineering")
print("=" * 70)

print("\nLoading datasets...")

train = pd.read_csv(
    os.path.join(RAW_DIR, "train.csv")
)

stores = pd.read_csv(
    os.path.join(RAW_DIR, "stores.csv")
)

oil = pd.read_csv(
    os.path.join(RAW_DIR, "oil.csv")
)

holidays = pd.read_csv(
    os.path.join(RAW_DIR, "holidays_events.csv")
)

transactions = pd.read_csv(
    os.path.join(RAW_DIR, "transactions.csv")
)

print("Datasets loaded.")


# ============================================================
# 3. DATE CONVERSION
# ============================================================

print("\nConverting date columns...")

train["date"] = pd.to_datetime(train["date"])
oil["date"] = pd.to_datetime(oil["date"])
holidays["date"] = pd.to_datetime(holidays["date"])
transactions["date"] = pd.to_datetime(transactions["date"])


# ============================================================
# 4. SORT TRAINING DATA
# ============================================================

train = train.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)


# ============================================================
# 5. MERGE STORE INFORMATION
# ============================================================

print("Merging store information...")

store_columns = [
    "store_nbr",
    "city",
    "state",
    "type",
    "cluster"
]

train = train.merge(
    stores[store_columns],
    on="store_nbr",
    how="left"
)


# ============================================================
# 6. MERGE TRANSACTIONS
# ============================================================

print("Merging transaction information...")

transactions_subset = transactions[
    ["date", "store_nbr", "transactions"]
]

train = train.merge(
    transactions_subset,
    on=["date", "store_nbr"],
    how="left"
)


# ============================================================
# 7. MERGE OIL PRICE
# ============================================================

print("Merging oil price...")

oil_subset = oil[
    ["date", "dcoilwtico"]
]

train = train.merge(
    oil_subset,
    on="date",
    how="left"
)


# ============================================================
# 8. OIL PRICE MISSING VALUES
# ============================================================

# Oil prices are not available for every calendar day.
# Forward-fill uses the most recently known oil price.

train["dcoilwtico"] = (
    train["dcoilwtico"]
    .ffill()
)

train["dcoilwtico"] = (
    train["dcoilwtico"]
    .bfill()
)


# ============================================================
# 9. CALENDAR FEATURES
# ============================================================

print("Creating calendar features...")

train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["day_of_week"] = train["date"].dt.dayofweek
train["week_of_year"] = (
    train["date"].dt.isocalendar().week.astype(int)
)

train["is_weekend"] = (
    train["day_of_week"] >= 5
).astype(int)


# ============================================================
# 10. HOLIDAY FEATURES
# ============================================================

print("Creating holiday features...")

# Only actual holidays/events are considered here.
# Transferred holidays are excluded from the direct holiday flag.

holiday_dates = (
    holidays.loc[
        holidays["transferred"] == False,
        "date"
    ]
    .drop_duplicates()
)

train["is_holiday"] = (
    train["date"]
    .isin(holiday_dates)
    .astype(int)
)


# ============================================================
# 11. PROMOTION FEATURE
# ============================================================

train["onpromotion"] = train["onpromotion"].fillna(0)


# ============================================================
# 12. HISTORICAL DEMAND FEATURES
# ============================================================

print("Creating lag features...")

group_columns = [
    "store_nbr",
    "family"
]

# IMPORTANT:
# shift() ensures only historical sales are used.

train["lag_1"] = (
    train
    .groupby(group_columns)["sales"]
    .shift(1)
)

train["lag_7"] = (
    train
    .groupby(group_columns)["sales"]
    .shift(7)
)

train["lag_14"] = (
    train
    .groupby(group_columns)["sales"]
    .shift(14)
)

train["lag_28"] = (
    train
    .groupby(group_columns)["sales"]
    .shift(28)
)


# ============================================================
# 13. ROLLING DEMAND FEATURES
# ============================================================

print("Creating rolling demand features...")

# Shift first so today's sales are NOT included.

previous_sales = (
    train
    .groupby(group_columns)["sales"]
    .shift(1)
)

train["rolling_mean_7"] = (
    previous_sales
    .groupby(
        [
            train["store_nbr"],
            train["family"]
        ]
    )
    .transform(
        lambda x: x.rolling(7).mean()
    )
)

train["rolling_mean_14"] = (
    previous_sales
    .groupby(
        [
            train["store_nbr"],
            train["family"]
        ]
    )
    .transform(
        lambda x: x.rolling(14).mean()
    )
)

train["rolling_mean_28"] = (
    previous_sales
    .groupby(
        [
            train["store_nbr"],
            train["family"]
        ]
    )
    .transform(
        lambda x: x.rolling(28).mean()
    )
)


# ============================================================
# 14. CHECK FEATURE CREATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)

feature_columns = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28"
]

print(
    train[feature_columns].describe()
)


# ============================================================
# 15. REMOVE RECORDS WITHOUT HISTORY
# ============================================================

print("\nRemoving records without sufficient historical data...")

before = len(train)

train = train.dropna(
    subset=[
        "lag_28",
        "rolling_mean_28"
    ]
)

after = len(train)

print(
    f"Rows before: {before:,}"
)

print(
    f"Rows after : {after:,}"
)

print(
    f"Rows removed: {before - after:,}"
)


# ============================================================
# 16. HANDLE REMAINING MISSING VALUES
# ============================================================

print("\nHandling remaining missing values...")

numeric_columns = train.select_dtypes(
    include=[np.number]
).columns

for column in numeric_columns:

    if train[column].isnull().any():

        train[column] = (
            train[column]
            .fillna(train[column].median())
        )


# ============================================================
# 17. CHECK DATA QUALITY AFTER PROCESSING
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATA QUALITY CHECK")
print("=" * 70)

print(
    "Remaining missing values:",
    train.isnull().sum().sum()
)

print(
    "Remaining duplicate rows:",
    train.duplicated().sum()
)


# ============================================================
# 18. FINAL COLUMN LIST
# ============================================================

print("\n" + "=" * 70)
print("FINAL FEATURES")
print("=" * 70)

for i, column in enumerate(train.columns, start=1):
    print(f"{i:>2}. {column}")


# ============================================================
# 19. SAVE PROCESSED DATA
# ============================================================

output_path = os.path.join(
    PROCESSED_DIR,
    "demandiq_features.csv"
)

print("\nSaving processed dataset...")

train.to_csv(
    output_path,
    index=False
)

print(
    f"Saved to: {output_path}"
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WEEK 2 PREPROCESSING COMPLETED")
print("=" * 70)

print(
    f"""
Final dataset shape:
{train.shape[0]:,} rows × {train.shape[1]} columns

Important feature groups:

Store:
- store_nbr
- city
- state
- type
- cluster

Calendar:
- year
- month
- day
- day_of_week
- week_of_year
- is_weekend
- is_holiday

Business:
- onpromotion
- transactions
- dcoilwtico

Historical demand:
- lag_1
- lag_7
- lag_14
- lag_28
- rolling_mean_7
- rolling_mean_14
- rolling_mean_28

Target:
- sales

Output:
data/processed/demandiq_features.csv
"""
)

print("=" * 70)