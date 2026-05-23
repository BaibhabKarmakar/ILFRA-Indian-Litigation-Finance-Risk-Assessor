"""
scripts/eda.py
--------------
Exploratory Data Analysis for ILFRA IBBI dataset.
Run from project root:
    python scripts/eda.py

Outputs printed to terminal + saved to scripts/eda_output/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/raw/ibbi_real.csv")
OUTPUT_DIR = Path("scripts/eda_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["cirp_start_date"] = pd.to_datetime(df["cirp_start_date"], errors="coerce")
df["resolution_date"] = pd.to_datetime(df["resolution_date"], errors="coerce")

NUMERIC_FEATURES = [
    "admitted_claim_cr",
    "liquidation_value",
    "fair_value",
    "realisable_amount",
    "realisation_pct",
    "duration_days",
    "log_admitted_claim",
    "admission_year",
    "no_of_financial_creditors",
    "resolution_applicants_received",
]
# Only keep columns that actually exist
NUMERIC_FEATURES = [f for f in NUMERIC_FEATURES if f in df.columns]

TARGETS = ["favourable_outcome", "realisation_pct", "duration_days"]
TARGETS = [t for t in TARGETS if t in df.columns]

print("=" * 60)
print("ILFRA — Exploratory Data Analysis")
print("=" * 60)

# ── 1. Basic shape ────────────────────────────────────────────────────────────
print(f"\n── 1. Dataset Shape ──────────────────────────────────")
print(f"  Rows    : {len(df)}")
print(f"  Columns : {len(df.columns)}")
print(f"  Date range: {df['cirp_start_date'].min().date()} → {df['cirp_start_date'].max().date()}")

# ── 2. Target distributions ───────────────────────────────────────────────────
print(f"\n── 2. Target Distributions ───────────────────────────")
print("\n  Resolution Status:")
print(df["resolution_status"].value_counts().to_string())
print(f"\n  Class balance (favourable_outcome):")
print(df["favourable_outcome"].value_counts(normalize=True).round(3).to_string())

print(f"\n  Duration (days):")
print(df["duration_days"].describe().round(1).to_string())

print(f"\n  Realisation %:")
print(df["realisation_pct"].describe().round(2).to_string())

print(f"\n  Admitted Claim (₹ Cr):")
print(df["admitted_claim_cr"].describe().round(2).to_string())

# ── 3. Outlier detection ──────────────────────────────────────────────────────
print(f"\n── 3. Outlier Detection (IQR method) ─────────────────")
for col in ["duration_days", "admitted_claim_cr", "realisation_pct"]:
    if col not in df.columns:
        continue
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    outliers = df[(df[col] > upper) | (df[col] < lower)]
    print(f"  {col:30s}: {len(outliers):4d} outliers "
          f"(lower={lower:.1f}, upper={upper:.1f})")
    if col == "duration_days" and len(outliers) > 0:
        print(f"    Longest cases:")
        print(df.nlargest(5, col)[["company_name", col, "resolution_status"]]
              .to_string(index=False))

# ── 4. Missing values ─────────────────────────────────────────────────────────
print(f"\n── 4. Missing Values ─────────────────────────────────")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("  No missing values.")
else:
    for col, count in missing.items():
        print(f"  {col:35s}: {count:4d} ({count/len(df)*100:.1f}%)")

# ── 5. Feature correlation with targets ───────────────────────────────────────
print(f"\n── 5. Feature Correlation with Targets ───────────────")
for target in TARGETS:
    print(f"\n  vs {target}:")
    corrs = {}
    for f in NUMERIC_FEATURES:
        if f == target or f not in df.columns:
            continue
        corr = df[f].corr(df[target])
        if not np.isnan(corr):
            corrs[f] = corr
    corrs = dict(sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True))
    for f, c in corrs.items():
        bar = "█" * int(abs(c) * 20)
        sign = "+" if c > 0 else "-"
        print(f"    {f:35s}: {sign}{abs(c):.3f} {bar}")

# ── 6. Multicollinearity check ────────────────────────────────────────────────
print(f"\n── 6. High Multicollinearity (|r| > 0.7) ─────────────")
available = [f for f in NUMERIC_FEATURES if f in df.columns]
corr_matrix = df[available].corr()
found = False
for i in range(len(available)):
    for j in range(i + 1, len(available)):
        c = corr_matrix.iloc[i, j]
        if abs(c) > 0.7:
            print(f"  {available[i]:30s} ↔ {available[j]:30s}: {c:.3f}")
            found = True
if not found:
    print("  No high multicollinearity found.")

# ── 7. Categorical feature analysis ──────────────────────────────────────────
print(f"\n── 7. Categorical Feature Analysis ──────────────────")
if "cirp_initiated_by" in df.columns:
    print("\n  Resolution rate by cirp_initiated_by:")
    print(df.groupby("cirp_initiated_by")["favourable_outcome"]
          .agg(["mean", "count"]).round(3).to_string())

if "resolution_status" in df.columns:
    print("\n  Avg realisation by resolution_status:")
    print(df.groupby("resolution_status")["realisation_pct"]
          .agg(["mean", "median", "count"]).round(2).to_string())

print("\n  Admission year distribution:")
print(df["admission_year"].value_counts().sort_index().to_string())

# ── 8. Plots ──────────────────────────────────────────────────────────────────
print(f"\n── 8. Saving Plots → {OUTPUT_DIR} ───────────────────")

# Plot 1: Target distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Target Variable Distributions", fontsize=13, fontweight="bold")

axes[0].bar(["Unfavourable (0)", "Favourable (1)"],
            df["favourable_outcome"].value_counts().sort_index().values,
            color=["#E24B4A", "#1A7A4A"])
axes[0].set_title("Outcome (class balance)")
axes[0].set_ylabel("Count")

axes[1].hist(df["duration_days"].dropna(), bins=40, color="#3B82F6", edgecolor="white")
axes[1].set_title("Duration Distribution (days)")
axes[1].set_xlabel("Days")
axes[1].axvline(df["duration_days"].median(), color="red", linestyle="--",
                label=f"Median: {df['duration_days'].median():.0f}")
axes[1].legend()

axes[2].hist(df["realisation_pct"].dropna(), bins=40, color="#6366F1", edgecolor="white")
axes[2].set_title("Realisation % Distribution")
axes[2].set_xlabel("Realisation %")
axes[2].axvline(df["realisation_pct"].median(), color="red", linestyle="--",
                label=f"Median: {df['realisation_pct'].median():.1f}%")
axes[2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_target_distributions.png", bbox_inches="tight")
plt.close()
print("  Saved: 01_target_distributions.png")

# Plot 2: Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[available].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, square=True,
            annot_kws={"size": 8})
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("  Saved: 02_correlation_heatmap.png")

# Plot 3: Feature vs outcome
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Feature Distribution by Outcome", fontsize=13, fontweight="bold")
axes = axes.flatten()

plot_features = [f for f in [
    "admitted_claim_cr", "log_admitted_claim", "duration_days",
    "realisation_pct", "admission_year",
    "no_of_financial_creditors"
] if f in df.columns][:6]

for i, feat in enumerate(plot_features):
    fav   = df[df["favourable_outcome"] == 1][feat].dropna()
    unfav = df[df["favourable_outcome"] == 0][feat].dropna()
    axes[i].hist(unfav, bins=30, alpha=0.6, color="#E24B4A", label="Unfavourable", density=True)
    axes[i].hist(fav,   bins=30, alpha=0.6, color="#1A7A4A", label="Favourable",   density=True)
    axes[i].set_title(feat)
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_feature_by_outcome.png", bbox_inches="tight")
plt.close()
print("  Saved: 03_feature_by_outcome.png")

# Plot 4: Admission year trend
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Trends Over Time", fontsize=13, fontweight="bold")

yearly = df.groupby("admission_year").agg(
    cases=("favourable_outcome", "count"),
    resolution_rate=("favourable_outcome", "mean"),
    avg_realisation=("realisation_pct", "mean"),
).reset_index()

axes[0].bar(yearly["admission_year"], yearly["cases"], color="#3B82F6")
axes[0].set_title("Cases by Admission Year")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Count")

axes[1].plot(yearly["admission_year"], yearly["resolution_rate"],
             marker="o", color="#1A7A4A", label="Resolution Rate")
axes[1].set_title("Resolution Rate by Year")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Rate")
axes[1].set_ylim(0, 1)
axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_yearly_trends.png", bbox_inches="tight")
plt.close()
print("  Saved: 04_yearly_trends.png")

# Plot 5: Outlier boxplots
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Outlier Detection — Boxplots", fontsize=13, fontweight="bold")

for i, col in enumerate(["duration_days", "admitted_claim_cr", "realisation_pct"]):
    if col in df.columns:
        axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#DBEAFE"),
                        medianprops=dict(color="#1D4ED8", linewidth=2))
        axes[i].set_title(col)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_outlier_boxplots.png", bbox_inches="tight")
plt.close()
print("  Saved: 05_outlier_boxplots.png")

print(f"\n✅ EDA complete. All plots saved to {OUTPUT_DIR.resolve()}")
print("=" * 60)