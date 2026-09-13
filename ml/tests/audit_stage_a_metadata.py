"""Read-only split/timestamp audit; outputs aggregate metadata, never health values.

Run from repo root:
  .venv/bin/python ml/tests/audit_stage_a_metadata.py
Only three parquet columns are read. No training, predictions or metrics.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(frame):
    results = []
    for _, group in frame.groupby("subject_id"):
        train = group[group.source_split == "train"].sort_values("timestamp")
        test = group[group.source_split == "test"].sort_values("timestamp")
        cut = int(len(train) * .85)
        before, after = train.iloc[:cut], train.iloc[cut:]
        results.append({
            "has_train": len(train) > 0,
            "has_test": len(test) > 0,
            "train_val_ordered": bool(len(before) and len(after) and before.timestamp.max() < after.timestamp.min()),
            "train_test_ordered": bool(len(train) and len(test) and train.timestamp.max() < test.timestamp.min()),
            "overlap_rows": len(set(train.timestamp) & set(test.timestamp)),
            "duplicate_rows": int(group.duplicated(["source_split", "timestamp"]).sum()),
            "gaps_gt_1h": sum(int((part.sort_values("timestamp").timestamp.diff() > pd.Timedelta("1h")).sum())
                              for _, part in group.groupby("source_split")),
            "offgrid_rows": int(((group.timestamp.dt.minute % 5 != 0) |
                                 (group.timestamp.dt.second != 0) |
                                 (group.timestamp.dt.microsecond != 0) |
                                 (group.timestamp.dt.nanosecond != 0)).sum()),
        })
    return {
        "subjects": len(results),
        "split_labels": sorted(frame.source_split.dropna().unique().tolist()),
        "missing_split_labels": int(frame.source_split.isna().sum()),
        "missing_subjects": int(frame.subject_id.isna().sum()),
        "missing_timestamps": int(frame.timestamp.isna().sum()),
        **{key: sum(row[key] for row in results) for key in (
            "has_train", "has_test", "train_val_ordered", "train_test_ordered",
            "overlap_rows", "duplicate_rows", "gaps_gt_1h", "offgrid_rows")},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/processed/training.parquet")
    args = parser.parse_args()
    data = pd.read_parquet(args.parquet, columns=["subject_id", "source_split", "timestamp"])
    print(json.dumps(summarize(data), indent=2))
