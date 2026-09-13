import pandas as pd

df = pd.read_parquet("ml/data/processed/training.parquet")
df = df.sort_values(["timestamp"]).reset_index(drop=True)

switches = (df["subject_id"] != df["subject_id"].shift()).sum()
total = len(df)
print(f"Zmiany pacjenta w kolejnych wierszach: {switches:,} / {total:,} ({switches/total*100:.1f}%)")

print(f"\nPierwsze 20 wierszy subject_id + timestamp:")
print(df[["subject_id", "timestamp"]].head(20).to_string())

print(f"\nUnikalni pacjenci: {df['subject_id'].nunique()}")
print(f"Łączna liczba wierszy: {total:,}")
print(f"\nLiczba wierszy per pacjent:")
print(df["subject_id"].value_counts().sort_index())