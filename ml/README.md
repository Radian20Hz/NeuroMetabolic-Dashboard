# NMD — ML Pipeline (Phase 3)

This directory contains the TFT training pipeline for glucose trend prediction.

## Setup

```bash
cd ml
pip install -r requirements.txt
```

## Data

Download the **OhioT1DM dataset** (academic registration required):
- http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html

Place extracted folders here:
```
ml/data/raw/
  OhioT1DM-training/   # 559-ws-training.xml, 563-ws-training.xml, ...
  OhioT1DM-testing/    # 559-ws-testing.xml,  ...
```

Raw data is gitignored — never commit patient data.

## Pipeline

### Step 1 — Preprocess

```bash
python scripts/preprocess_ohiot1dm.py
```

Outputs `ml/data/processed/training.parquet` and `testing.parquet`.

Features per row:
- `glucose_mg_dl` — raw CGM reading
- `glucose_delta_1` — 5 min delta
- `glucose_delta_3` — 15 min delta
- `bolus_last_1h` — total bolus insulin, last 60 min
- `basal_rate` — current basal rate (U/hr)
- `carbs_last_1h` — carbs ingested, last 60 min
- `hour_sin/cos`, `dow_sin/cos` — cyclical time features
- `is_hypo`, `is_hyper` — ADA clinical flags

### Step 2 — Train TFT *(coming next)*

```bash
python scripts/train_tft.py
```

### Step 3 — Export to ONNX *(coming next)*

```bash
python scripts/export_onnx.py
```

## Validation Targets

| Metric | Target |
|---|---|
| MARD | < 10% |
| Clarke Error Grid A+B | > 95% |
| TIR prediction accuracy | > 90% |
