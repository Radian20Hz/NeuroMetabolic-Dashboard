"""
Porównuje liczbę sampli i szybkość dataloadera dla obu podejść do time_idx.
Odpal z katalogu projektu: python check_dataset_size.py
"""
import time
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer

DATA_DIR = "ml/data/processed"
TARGET = "glucose_mg_dl"
TIME_VARYING_KNOWN_REALS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TIME_VARYING_UNKNOWN_REALS = [
    TARGET, "glucose_delta_1", "glucose_delta_3",
    "bolus_last_1h", "basal_rate", "carbs_last_1h",
]
HORIZON = 12
CONTEXT = 48

def make_dataset(df):
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["group"],
        min_encoder_length=CONTEXT // 2,
        max_encoder_length=CONTEXT,
        min_prediction_length=1,
        max_prediction_length=HORIZON,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=EncoderNormalizer(transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

def benchmark_loader(ds, name):
    loader = ds.to_dataloader(train=True, batch_size=64, num_workers=0)
    # Zmierz czas pierwszych 50 batchy
    t0 = time.time()
    for i, _ in enumerate(loader):
        if i >= 49:
            break
    elapsed = time.time() - t0
    it_s = 50 / elapsed
    print(f"  {name}: {len(ds):,} sampli | {it_s:.1f} it/s (pierwsze 50 batchy)")

df = pd.read_parquet(f"{DATA_DIR}/training.parquet")
df["subject_id"] = df["subject_id"].astype(str)
for col in ["glucose_delta_1", "glucose_delta_3"]:
    df[col] = df[col].fillna(0.0)
split_idx = int(len(df) * 0.85)

print("=" * 60)

# Podejście A: oryginalny cumcount per-subject (PRZED dropem)
df_a = df.copy()
df_a = df_a.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)
df_a["time_idx"] = df_a.groupby("subject_id").cumcount()
df_a = df_a.drop(columns=["subject_id"])
df_a["group"] = "population"
train_a = df_a.iloc[:split_idx].copy()
ds_a = make_dataset(train_a)
benchmark_loader(ds_a, "cumcount per-subject (mój fix)")

print()

# Podejście B: globalny index (oryginalny błąd)
df_b = df.copy()
df_b = df_b.sort_values(["timestamp"]).reset_index(drop=True)
df_b["time_idx"] = df_b.index
df_b = df_b.drop(columns=["subject_id"])
df_b["group"] = "population"
train_b = df_b.iloc[:split_idx].copy()
ds_b = make_dataset(train_b)
benchmark_loader(ds_b, "globalny index (oryginał)")

print("=" * 60)
print("\nWniosek: różnica w liczbie sampli = źródło różnicy w it/s")