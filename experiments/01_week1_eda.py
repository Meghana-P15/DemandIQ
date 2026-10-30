# ============================================================
# DemandIQ
# Week 1 - Dataset Understanding & Essential EDA
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
FIGURE_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("DEMANDIQ - WEEK 1")
print("Dataset Understanding & Exploratory Data Analysis")
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

print("All datasets loaded successfully.")


# ============================================================
# 3. DATASET OVERVIEW
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET OVERVIEW")
print("=" * 70)

datasets = {
    "train": train,
    "stores": stores,
    "oil": oil,
    "holidays": holidays,
    "transactions": transactions
}

for name, df in datasets.items():
    print(
        f"{name:<15} "
        f"{df.shape[0]:>10,} rows × "
        f"{df.shape[1]:>3} columns"
    )


# ============================================================
# 4. TRAINING DATA STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("2. TRAINING DATA STRUCTURE")
print("=" * 70)

print("\nColumns:")
print(train.columns.tolist())

print("\nData types:")
print(train.dtypes)

print("\nFirst 5 rows:")
print(train.head())


# ============================================================
# 5. DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("3. DATA QUALITY")
print("=" * 70)

print("\nMissing values in train.csv:")

missing = train.isnull().sum()

missing = missing[missing > 0]

if missing.empty:
    print("No missing values.")
else:
    print(missing)

print(
    f"\nDuplicate rows: "
    f"{train.duplicated().sum():,}"
)


# ============================================================
# 6. DATE ANALYSIS
# ============================================================

train["date"] = pd.to_datetime(train["date"])

print("\n" + "=" * 70)
print("4. DATE RANGE")
print("=" * 70)

print("Start date :", train["date"].min())
print("End date   :", train["date"].max())
print(
    "Unique dates:",
    train["date"].nunique()
)


# ============================================================
# 7. STORE & PRODUCT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("5. STORE & PRODUCT INFORMATION")
print("=" * 70)

print(
    "Number of stores:",
    train["store_nbr"].nunique()
)

print(
    "Number of product families:",
    train["family"].nunique()
)

print("\nProduct families:")
print(
    sorted(train["family"].unique())
)


# ============================================================
# 8. BASIC SALES STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("6. SALES STATISTICS")
print("=" * 70)

print(
    f"Total sales  : {train['sales'].sum():,.2f}"
)

print(
    f"Mean sales   : {train['sales'].mean():,.2f}"
)

print(
    f"Median sales : {train['sales'].median():,.2f}"
)

print(
    f"Maximum sales: {train['sales'].max():,.2f}"
)

print(
    f"Zero-sales records: "
    f"{(train['sales'] == 0).sum():,}"
)


# ============================================================
# 9. DAILY SALES TREND
# ============================================================

daily_sales = (
    train
    .groupby("date")["sales"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(12, 5))

plt.plot(
    daily_sales["date"],
    daily_sales["sales"]
)

plt.title("Daily Total Sales")
plt.xlabel("Date")
plt.ylabel("Total Sales")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "week1_daily_sales.png"
    ),
    dpi=150
)

plt.close()

print(
    "\nSaved: "
    "reports/figures/week1_daily_sales.png"
)


# ============================================================
# 10. TOP PRODUCT FAMILIES
# ============================================================

family_sales = (
    train
    .groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 70)
print("7. TOP PRODUCT FAMILIES")
print("=" * 70)

print(
    family_sales.head(10)
)


top_families = (
    family_sales
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_families.index,
    top_families.values
)

plt.title(
    "Top 10 Product Families by Total Sales"
)

plt.xlabel("Total Sales")
plt.ylabel("Product Family")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "week1_top_product_families.png"
    ),
    dpi=150
)

plt.close()

print(
    "Saved: "
    "reports/figures/week1_top_product_families.png"
)


# ============================================================
# 11. PROMOTION VS SALES
# ============================================================

train["promotion_status"] = np.where(
    train["onpromotion"] > 0,
    "Promotion",
    "No Promotion"
)

promotion_sales = (
    train
    .groupby("promotion_status")["sales"]
    .mean()
)

print("\n" + "=" * 70)
print("8. PROMOTION VS SALES")
print("=" * 70)

print(
    "\nAverage sales:"
)

print(promotion_sales)


plt.figure(figsize=(7, 5))

plt.bar(
    promotion_sales.index,
    promotion_sales.values
)

plt.title(
    "Average Sales: Promotion vs No Promotion"
)

plt.xlabel("Promotion Status")
plt.ylabel("Average Sales")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "week1_promotion_vs_sales.png"
    ),
    dpi=150
)

plt.close()

print(
    "Saved: "
    "reports/figures/week1_promotion_vs_sales.png"
)


# ============================================================
# 12. SUPPORTING DATASETS - BASIC CHECK ONLY
# ============================================================

print("\n" + "=" * 70)
print("9. SUPPORTING DATASETS")
print("=" * 70)

print("\nStores:")
print(stores.head())

print("\nOil:")
print(oil.head())

print("\nHolidays:")
print(holidays.head())

print("\nTransactions:")
print(transactions.head())


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WEEK 1 COMPLETED")
print("=" * 70)

print(
    """
We have completed:

✓ Dataset loading
✓ Dataset size inspection
✓ Column/type inspection
✓ Missing-value check
✓ Duplicate check
✓ Date-range analysis
✓ Store analysis
✓ Product-family analysis
✓ Basic sales analysis
✓ Daily demand trend
✓ Product-family comparison
✓ Promotion vs sales analysis

Generated only 3 essential visualizations.
"""
)

print("=" * 70)