"""
EDA — OhioT1DM Dataset
NeuroMetabolic Dashboard / Phase 3

Run: python ml/notebooks/eda_ohiot1dm.py
Outputs: ml/notebooks/eda_output/  (PNG charts)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent / "eda_output"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use("dark_background")
PALETTE = ["#b49dff", "#7dd3fc", "#86efac", "#ffb085", "#fca5a5", "#f0abfc"]

train = pd.read_parquet(PROCESSED_DIR / "training.parquet")
test  = pd.read_parquet(PROCESSED_DIR / "testing.parquet")
df    = pd.concat([train, test], ignore_index=True)

subjects = sorted(df["subject_id"].unique())
print(f"Subjects: {subjects}")
print(f"Total rows: {len(df):,}  |  Train: {len(train):,}  |  Test: {len(test):,}")
print(f"Features: {df.columns.tolist()}\n")

# ---------------------------------------------------------------------------
# 1. Glucose distribution per subject
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("Glucose Distribution per Subject (mg/dL)", fontsize=14, color="#f0eeff")

for ax, sid, color in zip(axes.flat, subjects, PALETTE):
    data = df[df["subject_id"] == sid]["glucose_mg_dl"].dropna()
    ax.hist(data, bins=60, color=color, alpha=0.8, edgecolor="none")
    ax.axvline(70,  color="#fca5a5", linestyle="--", linewidth=1, label="Hypo <70")
    ax.axvline(180, color="#ffb085", linestyle="--", linewidth=1, label="Hyper >180")
    ax.set_title(f"Subject {sid}", color="#f0eeff")
    ax.set_xlabel("mg/dL", color="#9b93c4", fontsize=9)
    ax.tick_params(colors="#5a5480")
    mean = data.mean()
    ax.axvline(mean, color=color, linestyle="-", linewidth=1.5, alpha=0.6)
    ax.text(mean + 5, ax.get_ylim()[1] * 0.85, f"μ={mean:.0f}", color=color, fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_glucose_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ 01_glucose_distribution.png")

# ---------------------------------------------------------------------------
# 2. Time-in-Range per subject (ADA zones)
# ---------------------------------------------------------------------------

zones = []
for sid in subjects:
    g = df[df["subject_id"] == sid]["glucose_mg_dl"].dropna()
    total = len(g)
    zones.append({
        "subject": sid,
        "Severe Hypo (<54)":  (g < 54).sum()  / total * 100,
        "Hypo (54–70)":       ((g >= 54) & (g < 70)).sum()  / total * 100,
        "TIR (70–180)":       ((g >= 70) & (g <= 180)).sum() / total * 100,
        "Hyper (180–250)":    ((g > 180) & (g <= 250)).sum() / total * 100,
        "Severe Hyper (>250)":(g > 250).sum()  / total * 100,
    })

zones_df = pd.DataFrame(zones).set_index("subject")
print("\nTime-in-Range breakdown (%):")
print(zones_df.round(1).to_string())

zone_colors = ["#fca5a5", "#ffb085", "#86efac", "#7dd3fc", "#b49dff"]
ax = zones_df.plot(kind="bar", stacked=True, figsize=(12, 5),
                   color=zone_colors, edgecolor="none", width=0.6)
ax.set_facecolor("#0d0d1a")
ax.figure.set_facecolor("#0d0d1a")
ax.axhline(70, color="#86efac", linestyle="--", linewidth=1, alpha=0.5, label="ADA TIR target 70%")
ax.set_title("Time-in-Range by ADA Zone per Subject", color="#f0eeff", pad=12)
ax.set_xlabel("Subject", color="#9b93c4")
ax.set_ylabel("% of readings", color="#9b93c4")
ax.tick_params(colors="#9b93c4")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_time_in_range.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ 02_time_in_range.png")

# ---------------------------------------------------------------------------
# 3. Feature correlations
# ---------------------------------------------------------------------------

feature_cols = [
    "glucose_mg_dl", "glucose_delta_1", "glucose_delta_3",
    "bolus_last_1h", "basal_rate", "carbs_last_1h",
    "hour_sin", "hour_cos", "is_hypo", "is_hyper",
]
corr = train[feature_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
fig.set_facecolor("#0d0d1a")
ax.set_facecolor("#0d0d1a")
sns.heatmap(
    corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, linewidths=0.5, linecolor="#1a1a2e",
    annot_kws={"size": 8}, cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Matrix (training set)", color="#f0eeff", pad=12)
ax.tick_params(colors="#9b93c4", labelsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ 03_correlation_matrix.png")

# ---------------------------------------------------------------------------
# 4. Missing data analysis
# ---------------------------------------------------------------------------

missing = train[feature_cols].isnull().sum() / len(train) * 100
print(f"\nMissing data (% of training rows):")
print(missing[missing > 0].round(3).to_string() or "  None — all features complete ✅")

fig, ax = plt.subplots(figsize=(10, 4))
fig.set_facecolor("#0d0d1a")
ax.set_facecolor("#0d0d1a")
bars = ax.bar(missing.index, missing.values, color="#b49dff", alpha=0.8, edgecolor="none")
ax.set_title("Missing Data per Feature (%)", color="#f0eeff", pad=12)
ax.set_ylabel("% missing", color="#9b93c4")
ax.tick_params(colors="#9b93c4", axis="x", rotation=30, labelsize=8)
ax.tick_params(colors="#9b93c4", axis="y")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_missing_data.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ 04_missing_data.png")

# ---------------------------------------------------------------------------
# 5. Hypo event frequency per subject
# ---------------------------------------------------------------------------

print("\nHypo event summary per subject:")
for sid in subjects:
    g = df[df["subject_id"] == sid]["glucose_mg_dl"].dropna()
    hypo_pct   = (g < 70).sum() / len(g) * 100
    severe_pct = (g < 54).sum() / len(g) * 100
    print(f"  Subject {sid}: hypo={hypo_pct:.1f}%  severe={severe_pct:.1f}%  n={len(g):,}")

# ---------------------------------------------------------------------------
# 6. Summary stats table
# ---------------------------------------------------------------------------

print("\n=== Summary Statistics (training set) ===")
summary = train.groupby("subject_id")["glucose_mg_dl"].agg(
    ["count", "mean", "std", "min", "max"]
).round(1)
summary.columns = ["n", "mean", "std", "min", "max"]
print(summary.to_string())

print(f"\n✅ All charts saved to: {OUTPUT_DIR}")
print("   Next step: ml/scripts/train_tft.py")