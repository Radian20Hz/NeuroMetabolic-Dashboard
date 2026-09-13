"""Regenerate and audit data only. Never constructs/trains/evaluates a model.

Use from repository root: python -m ml.scripts.diagnostics.regenerate_stage_a
Writes a new local directory; historical data and experiments are untouched.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd

from ml.scripts import train_tft_population_v2 as pipeline
from ml.tests.audit_stage_a_metadata import summarize

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_stage_a.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["resample_freq"] != "5min" or config["encoder_ffill_limit_steps"] != 6:
        raise ValueError("Configuration differs from implemented Stage A time policy")
    raw_dir = ROOT / config["raw_dir"]
    raw_files = sorted(raw_dir.rglob("*.xml"))
    if not raw_files:
        raise FileNotFoundError("No raw OhioT1DM XML files")
    code = sorted(list((ROOT / "ml/scripts").rglob("*.py")) + list((ROOT / "ml/tests").rglob("*.py")))
    code_hashes = {str(path.relative_to(ROOT)): digest(path) for path in code}
    raw_hash = hashlib.sha256("".join(digest(p) for p in raw_files).encode()).hexdigest()
    run_id = config["protocol"] + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / config["output_parent"] / run_id
    output.mkdir(parents=True, exist_ok=False)
    # Keep any existing patient-specific production logging local and ignored.
    with (output / "preprocessing.log").open("w") as log:
        subprocess.run([
            sys.executable, "-m", "ml.scripts.preprocess_ohiot1dm",
            "--data-dir", str(raw_dir), "--out-dir", str(output),
            "--resample-freq", config["resample_freq"],
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    logging.getLogger().setLevel(logging.ERROR)
    train, val = pipeline.load_and_preprocess_data(output)
    test = pipeline.load_test_data(output, max(train.time_idx.max(), val.time_idx.max()))
    train_ds, val_ds = pipeline.build_datasets(train, val, SimpleNamespace(
        context=config["context"], horizon=config["horizon"]))
    test_ds = pipeline.create_time_series_dataset(test, reference_dataset=train_ds, horizon=config["horizon"])
    metadata = summarize(pd.read_parquet(output / "training.parquet",
                                        columns=["subject_id", "source_split", "timestamp"]))
    if any(metadata[key] for key in ("missing_split_labels", "missing_subjects", "missing_timestamps",
                                     "overlap_rows", "duplicate_rows", "offgrid_rows")):
        raise RuntimeError("Regenerated metadata integrity failed")
    if metadata["train_test_ordered"] != metadata["subjects"]:
        raise RuntimeError("Regenerated train/test chronology failed")
    for subject in train.subject_id.unique():
        if train.loc[train.subject_id == subject, "timestamp"].max() >= val.loc[val.subject_id == subject, "timestamp"].min():
            raise RuntimeError("Actual train/val boundary overlaps")
    eligibility = {}
    for name, frame, dataset in (("train", train, train_ds), ("validation", val, val_ds), ("test", test, test_ds)):
        pipeline.assert_observed_evaluation(dataset, frame)
        eligibility[name] = {
            **dataset.stage_a_window_stats,
            "rows": len(frame),
            "observed_cgm_rows": int(frame.target_observed.sum()),
            "unresolved_encoder_rows": int(frame.glucose_mg_dl.isna().sum()),
        }
    if code_hashes != {str(path.relative_to(ROOT)): digest(path) for path in code}:
        raise RuntimeError("Source code changed during regeneration; rerun in a fresh directory")
    if raw_hash != hashlib.sha256("".join(digest(p) for p in raw_files).encode()).hexdigest():
        raise RuntimeError("Raw data changed during regeneration")
    manifest = {
        "run_id": run_id, "configuration": config,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_code_sha256": code_hashes,
        "raw_xml_count": len(raw_files),
        "raw_contents_digest": raw_hash,
        "parquet_sha256": digest(output / "training.parquet"),
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "scipy", "pyarrow", "torch", "lightning", "pytorch-forecasting")},
        "metadata": metadata, "window_eligibility": eligibility,
        "model_training_performed": False, "model_performance_inspected": False,
    }
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "metadata": metadata,
                      "window_eligibility": eligibility}, indent=2))


if __name__ == "__main__":
    main()
