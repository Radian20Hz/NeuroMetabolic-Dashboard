"""
add_subject.py
==============
Converts Medtronic 780G CareLink CSV export to OhioT1DM-compatible
training.parquet format and appends as new subject "wiktor".

Usage:
    python3 ml/add_subject.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH = Path("ml/data/Wiktor Waryszak 22-03-2026(2).csv")
PARQUET_PATH = Path("ml/data/processed/training.parquet")
SUBJECT_ID = "wiktor"

# ── 1. Load raw CSV ──────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, skiprows=6, encoding='utf-8-sig', low_memory=False)
df = df[df['Date'].str.match(r'^\d{4}/\d{2}/\d{2}$', na=False)].copy()
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values('datetime').reset_index(drop=True)

# ── 2. Build 5-min timeline ──────────────────────────────────────────────────
t_start = df['datetime'].min().floor('5min')
t_end   = df['datetime'].max().ceil('5min')
timeline = pd.date_range(t_start, t_end, freq='5min')
base = pd.DataFrame({'datetime': timeline})

# ── 3. Sensor glucose — forward-fill max 3 steps (15 min) ───────────────────
sg = df[['datetime', 'Sensor Glucose (mg/dL)']].dropna()
sg = sg.rename(columns={'Sensor Glucose (mg/dL)': 'glucose_mg_dl'})
sg['glucose_mg_dl'] = pd.to_numeric(sg['glucose_mg_dl'], errors='coerce')
sg = sg.set_index('datetime').resample('5min').mean()
base = base.merge(sg, on='datetime', how='left')
base['glucose_mg_dl'] = base['glucose_mg_dl'].ffill(limit=3)

# ── 4. Basal rate — forward-fill (changes infrequently) ─────────────────────
basal = df[['datetime', 'Basal Rate (U/h)']].dropna()
basal['Basal Rate (U/h)'] = pd.to_numeric(basal['Basal Rate (U/h)'], errors='coerce')
basal = basal.set_index('datetime').resample('5min').mean()
base = base.merge(basal.rename(columns={'Basal Rate (U/h)': 'basal_rate'}),
                  on='datetime', how='left')
base['basal_rate'] = base['basal_rate'].ffill(limit=288)

# ── 5. Bolus — sum within each 5-min bucket ──────────────────────────────────
bolus = df[['datetime', 'Bolus Volume Delivered (U)']].dropna()
bolus['Bolus Volume Delivered (U)'] = pd.to_numeric(
    bolus['Bolus Volume Delivered (U)'], errors='coerce')
bolus = bolus.set_index('datetime').resample('5min').sum()
base = base.merge(bolus.rename(columns={'Bolus Volume Delivered (U)': 'bolus_volume_u'}),
                  on='datetime', how='left')
base['bolus_volume_u'] = base['bolus_volume_u'].fillna(0.0)

# ── 6. Carbs (BWZ exchanges → grams, 1 exchange = 10g) ───────────────────────
carbs = df[['datetime', 'BWZ Carb Input (exchanges)']].dropna()
carbs['BWZ Carb Input (exchanges)'] = pd.to_numeric(
    carbs['BWZ Carb Input (exchanges)'], errors='coerce')
carbs = carbs.set_index('datetime').resample('5min').sum()
base = base.merge(carbs.rename(columns={'BWZ Carb Input (exchanges)': 'carbs_g'}),
                  on='datetime', how='left')
base['carbs_g'] = base['carbs_g'].fillna(0.0) * 10.0

# ── 7. Drop rows with no glucose ─────────────────────────────────────────────
base = base.dropna(subset=['glucose_mg_dl'])
base['subject_id'] = SUBJECT_ID

# ── 8. Stats ─────────────────────────────────────────────────────────────────
g = base['glucose_mg_dl']
tir = ((g >= 70) & (g <= 180)).mean() * 100
print(f"Wiktor rows:  {len(base)}")
print(f"Avg glucose:  {g.mean():.1f} mg/dL")
print(f"Std dev:      {g.std():.1f} mg/dL")
print(f"TIR 70-180:   {tir:.1f}%")
print(f"Basal nulls:  {base['basal_rate'].isna().sum()}")
print(f"Bolus sum:    {base['bolus_volume_u'].sum():.1f} U")
print(f"Carbs sum:    {base['carbs_g'].sum():.0f} g")

# ── 9. Append to training.parquet ────────────────────────────────────────────
existing = pd.read_parquet(PARQUET_PATH)
existing['subject_id'] = existing['subject_id'].astype(str)

# Remove old wiktor rows if re-running
existing = existing[existing['subject_id'] != SUBJECT_ID]

# Align columns
for col in existing.columns:
    if col not in base.columns:
        base[col] = np.nan
base = base[existing.columns]

combined = pd.concat([existing, base], ignore_index=True)
combined.to_parquet(PARQUET_PATH, index=False)
print(f"\ntraining.parquet updated: {len(existing)} → {len(combined)} rows")
print(f"Subjects: {sorted(combined['subject_id'].unique())}")
