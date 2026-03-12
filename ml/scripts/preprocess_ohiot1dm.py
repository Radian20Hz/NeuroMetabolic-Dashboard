"""
OhioT1DM Dataset Preprocessor — Phase 3 / NMD ML Pipeline
Converts raw OhioT1DM XML files into normalized feature DataFrames
ready for TFT training.

Dataset: OhioT1DM (Marling & Bunescu, 2018/2020)
Subjects: 12 T1D patients, ~8 weeks each
Download: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html
  (requires academic registration)

Expected raw directory structure:
  ml/data/raw/
    OhioT1DM-training/
      559-ws-training.xml
      563-ws-training.xml
      ...
    OhioT1DM-testing/
      559-ws-testing.xml
      ...

Output:
  ml/data/processed/
    train.parquet
    test.parquet
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# OhioT1DM CGM interval is 5 minutes
CGM_INTERVAL_MIN = 5

# ADA clinical thresholds (mg/dL)
HYPO_THRESHOLD = 70.0
HYPER_THRESHOLD = 180.0

# TFT input sequence length (60 min lookback = 12 CGM readings)
LOOKBACK_STEPS = 12

# Prediction horizon (60 min = 12 steps)
HORIZON_STEPS = 12


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def parse_ohiot1dm_xml(filepath: Path) -> dict[str, pd.DataFrame]:
    """
    Parse a single OhioT1DM XML file into a dict of DataFrames.
    Keys: 'glucose_level', 'bolus', 'basal', 'meal', 'exercise', 'sleep'

    Returns empty DataFrames for missing event types.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    result: dict[str, pd.DataFrame] = {}

    # --- Glucose ---
    glucose_events = []
    for event in root.iter("glucose_level"):
        for e in event:
            ts = e.get("ts")
            value = e.get("value")
            if ts and value:
                try:
                    glucose_events.append({
                        "timestamp": datetime.strptime(ts, "%d-%m-%Y %H:%M:%S"),
                        "glucose_mg_dl": float(value),
                    })
                except ValueError:
                    continue
    result["glucose_level"] = pd.DataFrame(glucose_events)

    # --- Bolus insulin ---
    bolus_events = []
    for event in root.iter("bolus"):
        for e in event:
            ts = e.get("ts_begin") or e.get("ts")
            dose = e.get("dose")
            btype = e.get("type", "normal")
            if ts and dose:
                try:
                    bolus_events.append({
                        "timestamp": datetime.strptime(ts, "%d-%m-%Y %H:%M:%S"),
                        "bolus_dose_u": float(dose),
                        "bolus_type": btype,
                    })
                except ValueError:
                    continue
    result["bolus"] = pd.DataFrame(bolus_events)

    # --- Basal insulin ---
    basal_events = []
    for event in root.iter("basal"):
        for e in event:
            ts = e.get("ts")
            rate = e.get("value")
            if ts and rate:
                try:
                    basal_events.append({
                        "timestamp": datetime.strptime(ts, "%d-%m-%Y %H:%M:%S"),
                        "basal_rate_u_hr": float(rate),
                    })
                except ValueError:
                    continue
    result["basal"] = pd.DataFrame(basal_events)

    # --- Meal ---
    meal_events = []
    for event in root.iter("meal"):
        for e in event:
            ts = e.get("ts")
            carbs = e.get("carbs")
            if ts and carbs:
                try:
                    meal_events.append({
                        "timestamp": datetime.strptime(ts, "%d-%m-%Y %H:%M:%S"),
                        "carbs_g": float(carbs),
                    })
                except ValueError:
                    continue
    result["meal"] = pd.DataFrame(meal_events)

    return result


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_frame(data: dict[str, pd.DataFrame], subject_id: str) -> pd.DataFrame:
    """
    Merge all event streams onto CGM timeline, compute derived features,
    and return a feature DataFrame indexed by timestamp.

    Features:
      - glucose_mg_dl         : raw CGM reading
      - glucose_delta_1       : 1-step CGM delta (5 min)
      - glucose_delta_3       : 3-step CGM delta (15 min)
      - bolus_last_1h         : total bolus insulin in last 60 min
      - basal_rate            : current basal rate (forward-filled)
      - carbs_last_1h         : total carbs ingested in last 60 min
      - hour_of_day           : cyclical hour sin/cos
      - day_of_week           : cyclical day sin/cos
      - is_hypo               : binary flag glucose < 70
      - is_hyper              : binary flag glucose > 180
    """
    glucose = data["glucose_level"].copy()
    if glucose.empty:
        return pd.DataFrame()

    glucose = glucose.sort_values("timestamp").drop_duplicates("timestamp")
    glucose = glucose.set_index("timestamp")

    # Resample to strict 5-min grid and forward-fill short gaps (≤3 steps)
    glucose = glucose.resample(f"{CGM_INTERVAL_MIN}min").mean()
    glucose["glucose_mg_dl"] = glucose["glucose_mg_dl"].interpolate(
        method="linear", limit=3
    )

    # Drop rows still missing after interpolation
    glucose = glucose.dropna(subset=["glucose_mg_dl"])

    df = glucose.copy()

    # Deltas
    df["glucose_delta_1"] = df["glucose_mg_dl"].diff(1)
    df["glucose_delta_3"] = df["glucose_mg_dl"].diff(3)

    # Bolus — rolling 60 min sum
    if not data["bolus"].empty:
        bolus = (
            data["bolus"]
            .set_index("timestamp")["bolus_dose_u"]
            .resample(f"{CGM_INTERVAL_MIN}min")
            .sum()
        )
        df["bolus_last_1h"] = bolus.reindex(df.index, fill_value=0).rolling(
            12, min_periods=1
        ).sum()
    else:
        df["bolus_last_1h"] = 0.0

    # Basal — forward-fill rate
    if not data["basal"].empty:
        basal = (
            data["basal"]
            .set_index("timestamp")["basal_rate_u_hr"]
            .resample(f"{CGM_INTERVAL_MIN}min")
            .last()
        )
        df["basal_rate"] = basal.reindex(df.index).ffill().fillna(0.0)
    else:
        df["basal_rate"] = 0.0

    # Carbs — rolling 60 min sum
    if not data["meal"].empty:
        carbs = (
            data["meal"]
            .set_index("timestamp")["carbs_g"]
            .resample(f"{CGM_INTERVAL_MIN}min")
            .sum()
        )
        df["carbs_last_1h"] = carbs.reindex(df.index, fill_value=0).rolling(
            12, min_periods=1
        ).sum()
    else:
        df["carbs_last_1h"] = 0.0

    # Cyclical time features
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Clinical flags
    df["is_hypo"] = (df["glucose_mg_dl"] < HYPO_THRESHOLD).astype(int)
    df["is_hyper"] = (df["glucose_mg_dl"] > HYPER_THRESHOLD).astype(int)

    # Subject identifier for multi-subject training
    df["subject_id"] = subject_id

    return df.reset_index()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_split(split: str) -> pd.DataFrame:
    """
    Process all subjects for a given split ('training' or 'testing').
    Expects flat structure: ml/data/raw/559-ws-training.xml, ...
    """
    pattern = f"*-ws-{split}.xml"
    xml_files = sorted(RAW_DIR.glob(pattern))

    if not xml_files:
        print(f"[WARN] No files matching '{pattern}' in {RAW_DIR}")
        return pd.DataFrame()

    frames = []
    for xml_path in xml_files:
        subject_id = xml_path.stem.split("-")[0]
        print(f"  Processing subject {subject_id}...")
        data = parse_ohiot1dm_xml(xml_path)
        frame = build_feature_frame(data, subject_id)
        if not frame.empty:
            frames.append(frame)
            print(f"    → {len(frame)} rows")
        else:
            print(f"    → skipped (empty)")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print("=== NMD — OhioT1DM Preprocessing Pipeline ===\n")

    for split in ("training", "testing"):
        print(f"[{split.upper()}]")
        df = process_split(split)

        if df.empty:
            print(f"  No data for {split} split — skipping.\n")
            continue

        out_path = PROCESSED_DIR / f"{split}.parquet"
        df.to_parquet(out_path, index=False)

        print(f"\n  ✅ Saved {len(df):,} rows → {out_path}")
        print(f"  Features: {df.columns.tolist()}")
        print(f"  Subjects: {df['subject_id'].unique().tolist()}")
        print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}\n")

    print("Done. Next step: ml/scripts/train_tft.py")


if __name__ == "__main__":
    main()
