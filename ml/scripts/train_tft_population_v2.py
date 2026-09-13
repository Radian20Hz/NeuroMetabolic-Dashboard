"""
ml/scripts/train_tft_population_v2.py
=======================================
Phase 4 v19 — Zero-Trust Audit Fixes (round 9)

Fixes applied on top of v18
──────────────────────────────────────────
  [FIX-CRIT-1]  ic_ratio_deviation was written as a real (contaminated) rolling
            value to the parquet and was absent from _SENTINEL_ZERO_COLS, so the
            sentinel guard never fired.  Now sentineled to NaN in the preprocessor
            (matching dynamic_isf_estimate) and added to PRE_SPLIT_FILLNA_SKIP,
            POST_SPLIT_COMPUTED, POST_SPLIT_FINAL_FILLNA_SKIP, and the test-data
            drop set.  The existing _compute_long_window_features() recomputation
            is unchanged.

  [FIX-CRIT-1]  basal_bolus_ratio used rolling(24*12, min_periods=12).sum()
            on the full unsplit patient series (same class as bolus_count_3h,
            FIX-ROLL-BOUNDARY).  Now sentineled to 0.0 in the preprocessor
            and recomputed post-split in _compute_long_window_features().
            Added to PRE_SPLIT_FILLNA_SKIP, POST_SPLIT_COMPUTED, and all
            related skip/cleanup sets.

  [FIX-CRIT-1]  lbgi_30m, hbgi_30m, bgri used rolling(6).mean() on the full
            unsplit series.  Now sentineled to 0.0 in the preprocessor and
            recomputed post-split in _compute_long_window_features().

  [FIX-HIGH-2]  steps_since_last_bolus had broken index arithmetic in the
            preprocessor (label index vs positional index mismatch after
            dropna + reset_index).  Fixed in the preprocessor using positional
            np.arange arithmetic; no training-script change needed.

  [FIX-CRIT-2]  GroupNormalizer fitted-on-train guard was logically vacuous:
            hasattr(_norm, 'center_') is True even on an unfitted normalizer
            because __init__ sets center_ = None.  Replaced with
            getattr(_norm, 'center_', None) is not None, which correctly
            verifies fit() has been called.

  [FIX-MED-1]  Replaced the CR-8 variance heuristic (train_std vs val_std × 3)
            with a sentinel-recomputation check.  The old heuristic could not
            detect the failure mode it was designed to catch: a sentinel column
            still holding 0.0 after a failed recomputation produces std=0 in
            both splits, trivially passing the 3× threshold.  New check: after
            _compute_long_window_features() runs, 0.0-sentinel columns must
            have at least some non-zero values in the train split, and NaN-
            sentinel columns must have no remaining NaNs.

All previous fixes from v18 (FIX-NADIR-ASYM, FIX-TDD-SENTINEL,
FIX-NORM-GROUPNORM, and all prior v17 and earlier fixes) retained without
modification.

Usage:
    python ml/scripts/train_tft_population_v2.py [--epochs 60] [--batch-size 64]
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer
from pytorch_forecasting.metrics import QuantileLoss

import torch.serialization
from pytorch_forecasting.data.encoders import EncoderNormalizer, GroupNormalizer
from pytorch_forecasting.data import NaNLabelEncoder

import numpy._core.multiarray

torch.serialization.add_safe_globals([
    EncoderNormalizer,
    GroupNormalizer,
    NaNLabelEncoder,
    numpy._core.multiarray.scalar,
])

# ── Version guard ─────────────────────────────────────────────────────────────
import pytorch_forecasting as _ptf

_PTF_VERSION = tuple(
    int(x) for x in re.findall(r"\d+", _ptf.__version__)[:2]
)
if _PTF_VERSION < (1, 0):
    warnings.warn(
        f"pytorch-forecasting {_ptf.__version__} < 1.0 detected. "
        "Target tuple format in loss functions may differ from expected. "
        "Validate ClinicalQuantileLoss behaviour before production use.",
        DeprecationWarning,
        stacklevel=1,
    )

os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)
warnings.filterwarnings(
    "ignore",
    message=".*does not have many workers.*",
    category=UserWarning,
)

# ── Optional statsmodels for ARIMA baseline [AUDIT-MED-NEW-1] ─────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False
    warnings.warn(
        "statsmodels not installed — ARIMA baseline unavailable. "
        "Install with: pip install statsmodels",
        ImportWarning,
        stacklevel=1,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "ml" / "data" / "processed"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

TARGET_COL  = "glucose_mg_dl"
GROUP_COL   = "subject_id"
SUBJECT_COL = "subject_id"

MAX_ENCODER_LENGTH    = 48
MAX_PREDICTION_LENGTH = 12

# [LEAK-B] Train/val gap: 2 * MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH + 1 = 109
TRAIN_VAL_GAP: int = 2 * MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH + 1  # 109
TIME_IDX_PATIENT_GAP: int = MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH + 1  # 61

STATIC_CATEGORICALS: list[str] = ["subject_id"]
STATIC_REALS:        list[str] = []

TIME_VARYING_KNOWN_REALS: list[str] = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # OhioT1DM de-identification shifts calendar months; month-of-year is not
    # a valid seasonal signal for this dataset.
    "is_dawn_window",
    "is_breakfast_window",
    "is_lunch_window",
    "is_dinner_window",
    "is_weekend",
    "minutes_since_midnight",
]

TIME_VARYING_UNKNOWN_REALS: list[str] = [
    TARGET_COL,
    # Actual delivered basal can be modified by temp-basal/suspend events and
    # is therefore not guaranteed to be known over the decoder horizon.
    "basal_rate",
    "bolus_last_1h",
    "carbs_last_1h",
    "weekend_meal_flag",
    "weekend_meal_prior",
    "insulin_on_board",
    "carb_on_board",
    "net_insulin_effect",
    "carb_to_bolus_ratio_1h",
    "basal_rate_deviation",
    "ic_ratio_deviation",
    "bolus_count_3h",
    "iob_delta",
    "iob_acceleration",
    "cob_iob_ratio",
    "meal_occurred_last_30m",
    "bolus_fraction_of_tdd",
    "is_temp_basal",
    "basal_bolus_ratio",
    "dynamic_isf_estimate",
    "postprandial_phase",
    "steps_since_last_bolus",
    "correction_bolus_prior",
    "hr_variability_15m",
    "hr_above_resting",
    "hr_trend_30m",
    "gsr_stress_deviation",
    "skin_temp_deviation",
    "steps_last_30m",
    "composite_stress_index",
    "autonomic_stress_index",
    "is_aerobic_exercise",
    "is_stress_response",
    "exercise_acute_effect",
    "exercise_epoc_effect",
    "exercise_iob_danger",
    "exercise_cob_coverage",
    "step_insulin_offset",
    "skin_temp_drop_rate",
    "hr_recovery_slope",
    "cgm_gap_flag",
    # [FIX-ISSUE-19] Explicit glucose lag features — strictly causal (shift +
    # first-observed-value fill per patient in preprocessing).  Previously
    # computed and stored in the parquet but never registered here, so the TFT
    # could not see them.
    "glucose_lag_1",
    "glucose_lag_2",
    "glucose_lag_3",
    "glucose_lag_6",
    "glucose_lag_12",
    "glucose_lag_24",
]

# Columns skipped by the pre-split fillna(0.0) loop.
PRE_SPLIT_FILLNA_SKIP: frozenset[str] = frozenset({
    "dynamic_isf_estimate",
    "correction_bolus_prior",
    "weekend_meal_prior",
    "ic_ratio_deviation",
    "tdd_rolling_7d",
    "bolus_fraction_of_tdd",
    "hr_recovery_slope",
    # [ZT-CRIT-NEW-1/2] new sentinels — must not be zeroed before post-split computation
    "hr_resting_estimate",
    "hr_reserve_pct",
    "is_aerobic_exercise",
    "is_stress_response",
    "hr_above_resting",
    "exercise_iob_danger",
    "exercise_cob_coverage",
    "composite_stress_index",
    # [FIX-ISSUE-11] sentinel 0.0 in parquet, recomputed post-split
    "autonomic_stress_index",
    # [FIX-C2] sentinel 0.0 in parquet, recomputed post-split
    "gsr_stress_deviation",
    "skin_temp_deviation",
    # [FIX-H2] sentinel 0.0 in parquet, recomputed post-split
    "glucose_lag_1",
    "glucose_lag_2",
    "glucose_lag_3",
    "glucose_lag_6",
    "glucose_lag_12",
    "glucose_lag_24",
    # [FIX-BASAL-DEV] sentinel 0.0 in parquet, recomputed post-split
    "basal_rate_deviation",
    # [FIX-ROLL-BOUNDARY] sentinel 0.0/0.5 in parquet, recomputed post-split
    "bolus_count_3h",
    "glucose_sample_entropy_60m",
    "glucose_tir_2h",
    "glucose_hyper_ratio_2h",
    "glucose_hypo_ratio_2h",
    "glucose_dfa_alpha_2h",
    # [FIX-NADIR-ASYM] sentinel 0.0 in parquet, recomputed post-split
    "glucose_nadir_proximity",
    "glucose_asymmetry_index",
    # [FIX-TDD-SENTINEL] sentinel 0.0 in parquet, recomputed post-split
    "tdd_rolling_7d",
    "bolus_fraction_of_tdd",
    # [FIX-CRIT-1] sentinel 0.0 in parquet, recomputed post-split
    "basal_bolus_ratio",
    "lbgi_30m",
    "hbgi_30m",
    "bgri",
    # [FIX-CRIT-1] sentinel NaN in parquet, recomputed post-split
    "ic_ratio_deviation",
})


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL CORRECTION BOLUS PRIOR  [AUDIT-CRIT-2]
# ─────────────────────────────────────────────────────────────────────────────
def _compute_correction_bolus_prior(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values("timestamp").reset_index(drop=True)
    bolus_occurred = (
        (df["bolus_event"].rolling(6, min_periods=1).max().fillna(0.0)) > 0
    ).astype(float)
    return (
        bolus_occurred
        .rolling(24 * 12, min_periods=12)
        .mean()
        .fillna(0.1)
    )


def _compute_correction_bolus_prior_with_history(
    current: pd.DataFrame,
    history: "pd.DataFrame | None" = None,
) -> pd.Series:
    """Compute the causal prior with exact preceding history when contiguous."""
    cur = current.sort_values("timestamp").reset_index(drop=True)
    if history is None or history.empty or cur.empty:
        return _compute_correction_bolus_prior(cur)

    hist = history.sort_values("timestamp").reset_index(drop=True)
    gap = pd.Timestamp(cur["timestamp"].iloc[0]) - pd.Timestamp(hist["timestamp"].iloc[-1])
    if gap <= pd.Timedelta(0):
        raise RuntimeError(
            f"[AUDIT-CBP] History overlaps/follows current split (gap={gap})."
        )
    if gap > pd.Timedelta("10min"):
        log.warning(
            f"  [AUDIT-CBP] Boundary gap {gap} is >10 min; "
            "using current-split-only correction_bolus_prior."
        )
        return _compute_correction_bolus_prior(cur)

    # rolling(6) followed by rolling(288): retain enough true prior rows for both.
    hist_tail = hist.tail(24 * 12 + 6)
    combined = pd.concat([hist_tail, cur], ignore_index=True, sort=False)
    values = _compute_correction_bolus_prior(combined)
    return values.tail(len(cur)).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# [LEAK-A] CAUSAL WEEKEND MEAL PRIOR
# ─────────────────────────────────────────────────────────────────────────────
def _compute_weekend_meal_prior(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values("timestamp").reset_index(drop=True)
    meal_occurred = (
        df.get("meal_event", pd.Series(0.0, index=df.index)) > 0
    ).astype(float)
    return (
        meal_occurred
        .rolling(7 * 24 * 12, min_periods=12)
        .mean()
        .fillna(0.0)
    )


# ─────────────────────────────────────────────────────────────────────────────
# [LEAK-C] CAUSAL RESTING HR ESTIMATE
# ─────────────────────────────────────────────────────────────────────────────
def _compute_hr_resting_causal(
    hr: pd.Series,
    warmstart_value: "float | None" = None,
) -> pd.Series:
    """
    Causal resting HR via rolling 10th-percentile (window = 96 steps = 8 h,
    min_periods = 24 steps = 2 h).

    Warm-up fill — strictly causal:
      1. ffill(limit=96): carry the most recent observed HR reading forward
         into the warm-up window.  This is causal because ffill only ever
         looks backward.
      2. fillna(warmstart_value or first_valid): any remaining NaNs before the
         very first HR reading are filled with the train-split tail estimate
         when available (via FIX-ISSUE-15), or with the first observed HR value.

    [FIX-ISSUE-8] Previous version used bfill(limit=24), which filled
    warm-up NaNs BACKWARD from the rolling-quantile value at t+2h — i.e.
    it stamped a statistic computed on future HR data onto t=0..t+2h.
    bfill of any kind is non-causal and has been removed.

    [FIX-ISSUE-15] `warmstart_value`: when provided (val split only), the
    leading NaN warm-up window is filled with the train split's last resting-HR
    estimate rather than the first observed HR of the val split.  This is causal
    because it uses only data that precedes the val period.
    """
    resting = (
        hr.rolling(96, min_periods=24)
        .quantile(0.10)
    )
    # Step 1: causal forward-fill.
    resting = resting.ffill(limit=96)
    # Step 2: fill any remaining leading NaNs.
    if warmstart_value is not None:
        fill_val = float(warmstart_value)
    else:
        fill_val = float(hr.dropna().iloc[0]) if hr.notna().any() else 70.0
    resting = resting.fillna(fill_val)
    return resting


# ─────────────────────────────────────────────────────────────────────────────
# [ZT-CRIT-NEW-1/2] CAUSAL EXERCISE + WEARABLE FEATURE RECOMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def _compute_exercise_features_causal(
    df: pd.DataFrame,
    hr_resting_warmstart: "float | None" = None,
) -> pd.DataFrame:
    """
    Post-split causal recomputation of all features that depended on the
    leaked hr_resting_estimate in the v9 preprocessor.

    Replaces sentinel 0.0 values for:
      hr_resting_estimate    [ZT-CRIT-NEW-1]
      hr_reserve_pct         [ZT-CRIT-NEW-1]
      is_aerobic_exercise    [ZT-CRIT-NEW-1]
      is_stress_response     [ZT-CRIT-NEW-1]
      hr_above_resting       [ZT-CRIT-NEW-2]
      gsr_stress_deviation   [FIX-C2]
      skin_temp_deviation    [FIX-C2]
      autonomic_stress_index [FIX-ISSUE-11]
      exercise_iob_danger    (derived from is_aerobic_exercise)
      exercise_cob_coverage  (derived from is_aerobic_exercise)
      composite_stress_index (depends on hr_above_resting + gsr_stress_deviation)

    Uses _compute_hr_resting_causal() — causal ffill + first observed value.
    Never uses whole-group median or any backward fill (bfill removed in FIX-ISSUE-8).

    [FIX-ISSUE-15] Accepts `hr_resting_warmstart`: the last resting-HR estimate
    from the train split, used to fill the leading NaNs in the val split's causal
    rolling quantile instead of falling back to the first observed HR value.
    This eliminates the cold-start bias in hr_resting_estimate at val t=0.
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    has_hr   = "heart_rate" in df.columns and df["heart_rate"].notna().any()
    has_gsr  = "gsr"              in df.columns and df["gsr"].notna().any()
    has_temp = "skin_temperature" in df.columns and df["skin_temperature"].notna().any()
    has_step = "steps_last_30m" in df.columns

    # [FIX-C2] Recompute gsr_stress_deviation and skin_temp_deviation on the
    # isolated split data.  The preprocessor writes 0.0 as a sentinel because
    # the rolling(WIN_BASELINE=12).mean() baseline was computed on the full
    # unsplit series, letting the 12 rows nearest the split boundary incorporate
    # future sensor readings.  Recomputing here ensures each split sees only
    # its own causal rolling baseline.
    _WIN_BASELINE_C2 = 12
    if has_gsr:
        gsr_ma = df["gsr"].rolling(_WIN_BASELINE_C2, min_periods=3).mean()
        df["gsr_stress_deviation"] = (df["gsr"] - gsr_ma).fillna(0.0)
    else:
        df["gsr_stress_deviation"] = 0.0

    if has_temp:
        temp_ma = df["skin_temperature"].rolling(_WIN_BASELINE_C2, min_periods=3).mean()
        df["skin_temp_deviation"] = (df["skin_temperature"] - temp_ma).fillna(0.0)
    else:
        df["skin_temp_deviation"] = 0.0

    if has_hr:
        # [ZT-CRIT-NEW-1] Causal resting HR
        # [FIX-ISSUE-15] If a warm-start value is provided (from train split tail),
        # override the first-observed-HR fallback so val t=0 starts correctly.
        hr_resting = _compute_hr_resting_causal(
            df["heart_rate"],
            warmstart_value=hr_resting_warmstart,
        )
        df["hr_resting_estimate"] = hr_resting

        max_hr = 180.0
        df["hr_reserve_pct"] = (
            (df["heart_rate"] - hr_resting)
            / (max_hr - hr_resting).clip(lower=20.0)
        ).clip(0.0, 1.0).fillna(0.0)

        # [ZT-CRIT-NEW-2] Causal hr_above_resting
        df["hr_above_resting"] = (
            df["heart_rate"] - hr_resting
        ).clip(lower=0.0).fillna(0.0)

        steps_col = df["steps_last_30m"] if has_step else pd.Series(0.0, index=df.index)

        # [FIX-ISSUE-12] Use causal rolling quantiles instead of global split quantiles.
        # Global quantile(0.75/0.25) over the full split leaks future step distribution
        # into early timesteps: the threshold at t=0 is set by data from t=0..T.
        # Rolling window = 96 steps (8 h), min_periods = 12 (1 h warm-up); ffill fills
        # any remaining leading NaNs with the first computed quantile (still causal).
        _STEP_ROLL_WIN = 96   # 8 h at 5-min resolution
        _STEP_ROLL_MIN = 12   # 1 h minimum before emitting a value
        if has_step:
            steps_75pct = (
                steps_col
                .rolling(_STEP_ROLL_WIN, min_periods=_STEP_ROLL_MIN)
                .quantile(0.75)
                .ffill()
                .fillna(float(steps_col.iloc[0]))
            )
            steps_25pct = (
                steps_col
                .rolling(_STEP_ROLL_WIN, min_periods=_STEP_ROLL_MIN)
                .quantile(0.25)
                .ffill()
                .fillna(float(steps_col.iloc[0]))
            )
        else:
            steps_75pct = pd.Series(0.0, index=df.index)
            steps_25pct = pd.Series(0.0, index=df.index)

        df["is_aerobic_exercise"] = (
            (steps_col > steps_75pct) & (df["hr_reserve_pct"] > 0.40)
        ).astype(float)

        gsr_col = df.get("gsr_stress_deviation", pd.Series(0.0, index=df.index))
        df["is_stress_response"] = (
            (steps_col <= steps_25pct)
            & (df["hr_reserve_pct"] > 0.50)
            & (gsr_col > 0)
        ).astype(float)

    else:
        df["hr_resting_estimate"] = 0.0
        df["hr_reserve_pct"]      = 0.0
        df["hr_above_resting"]    = 0.0
        df["is_aerobic_exercise"] = 0.0
        df["is_stress_response"]  = 0.0

    # Recompute derived features
    df["exercise_iob_danger"]   = df["is_aerobic_exercise"] * df["insulin_on_board"]
    df["exercise_cob_coverage"] = df["is_aerobic_exercise"] * df["carb_on_board"]

    # Recompute composite_stress_index with the corrected hr_above_resting
    WIN_BASELINE = 12
    hr_roll_std  = (
        df["hr_above_resting"]
        .rolling(WIN_BASELINE, min_periods=3)
        .std()
        .fillna(1.0)
        .clip(lower=1e-6)
    )
    gsr_col_stress = df.get("gsr_stress_deviation", pd.Series(0.0, index=df.index))
    gsr_roll_std   = (
        gsr_col_stress
        .rolling(WIN_BASELINE, min_periods=3)
        .std()
        .fillna(1.0)
        .clip(lower=1e-6)
    )
    # [FIX-ISSUE-16] Clip z-scores to [-10, 10] before averaging.
    # When hr_above_resting is near-constant (patient at rest), rolling std → 0
    # which is clamped to 1e-6, causing hr_z to explode to ~5e6.  This value is
    # finite so the inf-only cleanup loop does NOT catch it; it would propagate
    # directly into TIME_VARYING_UNKNOWN_REALS and corrupt the TFT encoder.
    hr_z  = (df["hr_above_resting"] / hr_roll_std).clip(-10.0, 10.0)
    gsr_z = (gsr_col_stress          / gsr_roll_std).clip(-10.0, 10.0)
    df["composite_stress_index"] = ((hr_z + gsr_z) / 2.0).fillna(0.0)

    # [FIX-ISSUE-11] Recompute autonomic_stress_index post-split.
    # The preprocessor writes 0.0 as a sentinel because the z-score normalisation
    # (rolling mean/std over WIN_BASELINE=12 steps) was previously computed on the
    # full unsplit patient series, letting early-split rows inherit normalisation
    # statistics that included data from the other split.  Here we recompute it on
    # the isolated split only, using the same formula as the original preprocessor.
    if has_hr and has_gsr and has_temp:
        hr_ma    = df["heart_rate"].rolling(WIN_BASELINE, min_periods=3).mean()
        hr_std   = df["heart_rate"].rolling(WIN_BASELINE, min_periods=3).std().fillna(1.0)
        gsr_raw  = df.get("gsr",              pd.Series(np.nan, index=df.index))
        temp_raw = df.get("skin_temperature", pd.Series(np.nan, index=df.index))
        gsr_std  = gsr_raw.rolling(WIN_BASELINE, min_periods=3).std().fillna(1.0)
        temp_std = temp_raw.rolling(WIN_BASELINE, min_periods=3).std().fillna(1.0)
        # [FIX-ISSUE-16] Clip each z-score to [-10, 10] — same guard applied to
        # composite_stress_index above.  Prevents constant-signal patients from
        # producing z-scores of ~5e6 that bypass the inf-only cleanup.
        hr_zscore   = ((df["heart_rate"] - hr_ma) / hr_std.clip(lower=0.1)).clip(-10.0, 10.0).fillna(0.0)
        gsr_zscore  = (gsr_col_stress / gsr_std.clip(lower=0.1)).clip(-10.0, 10.0).fillna(0.0)
        skin_dev    = df.get("skin_temp_deviation", pd.Series(0.0, index=df.index))
        temp_zscore = (skin_dev / temp_std.clip(lower=0.1)).clip(-10.0, 10.0).fillna(0.0)
        df["autonomic_stress_index"] = (
            (hr_zscore + gsr_zscore - temp_zscore) / 3.0
        ).fillna(0.0)
    else:
        # Fall back to composite_stress_index (same as original preprocessor fallback)
        df["autonomic_stress_index"] = df["composite_stress_index"]

    return df


def _compute_hr_recovery_slope_causal(
    df: pd.DataFrame,
    hr_resting_warmstart: "float | None" = None,
) -> pd.Series:
    """[ZT-HIGH-3] + [LEAK-C] Causal HR recovery slope.

    [FIX-ISSUE-18] Accepts hr_resting_warmstart so the resting-HR baseline
    used here is consistent with the one used by _compute_exercise_features_causal().
    Previously this called _compute_hr_resting_causal() without a warmstart, so
    the two resting-HR estimates diverged at the start of every val/test split,
    producing slightly incorrect recovery slopes for the first ~8 h.
    """
    if "heart_rate" not in df.columns or df["heart_rate"].isna().all():
        return pd.Series(0.0, index=df.index)

    exercise_active     = df["exercise_duration_min"] > 0
    prev_active         = exercise_active.shift(1).fillna(0).astype(bool)
    exercise_just_ended = (~exercise_active) & prev_active

    hr_at_end  = df["heart_rate"].where(exercise_just_ended)
    # [FIX-ISSUE-18] Forward warmstart so both recovery-slope and hr_resting_estimate
    # use the same causal baseline throughout the split.
    hr_resting = _compute_hr_resting_causal(
        df["heart_rate"],
        warmstart_value=hr_resting_warmstart,
    )

    hr_drop_target = (hr_at_end  - hr_resting).clip(lower=0.0)
    current_drop   = (df["heart_rate"] - hr_resting).clip(lower=0.0)
    raw_slope      = (hr_drop_target - current_drop) / (30.0 + 1e-6)

    return (
        raw_slope
        .where(exercise_just_ended, other=np.nan)
        .ffill(limit=6)
        .fillna(0.0)
    )


def _extract_train_warmstart(train_grp: pd.DataFrame) -> dict:
    """
    [FIX-ISSUE-15] Extract the terminal rolling state from the train split so
    that the val split can warm-start its long-window features instead of
    restarting from scratch.

    The carry-over is strictly CAUSAL: all values are derived from the tail of
    the train split only — no val data is used.
    """
    train_grp = train_grp.sort_values("timestamp").reset_index(drop=True)
    ws: dict = {}

    WIN_TDD_LOCAL = 7 * 24 * 12
    if "bolus_event" in train_grp.columns and "basal_rate" in train_grp.columns:
        bolus_sum  = train_grp["bolus_event"].rolling(WIN_TDD_LOCAL, min_periods=1).sum()
        basal_mean = train_grp["basal_rate"].rolling(WIN_TDD_LOCAL, min_periods=1).mean()
        # Mean total daily dose (U/day) over the trailing 7-day window:
        # (7-day bolus total + estimated 7-day basal total) / 7.
        tdd        = (bolus_sum + basal_mean * 24.0 * 7.0) / 7.0
        ws["tdd_last"] = float(tdd.iloc[-1]) if len(tdd) else 20.0

    if "bolus_event" in train_grp.columns:
        bolus_occurred = (
            (train_grp["bolus_event"].rolling(6, min_periods=1).max().fillna(0.0)) > 0
        ).astype(float)
        cbp = bolus_occurred.rolling(24 * 12, min_periods=12).mean().fillna(0.1)
        ws["cbp_last"] = float(cbp.iloc[-1]) if len(cbp) else 0.1

    if "carb_to_bolus_ratio_1h" in train_grp.columns:
        ratio_ma = (
            train_grp["carb_to_bolus_ratio_1h"]
            .rolling(24 * 12, min_periods=24 * 12)
            .median()
        )
        ws["ic_ratio_ma_last"] = float(ratio_ma.dropna().iloc[-1]) if ratio_ma.notna().any() else None

    if "heart_rate" in train_grp.columns and train_grp["heart_rate"].notna().any():
        hr_resting = _compute_hr_resting_causal(train_grp["heart_rate"])
        ws["hr_resting_last"] = float(hr_resting.iloc[-1])

    # Preserve true trailing history for exact causal warm-start of rolling/lag
    # features.  A scalar summary cannot reproduce a 7-day rolling window.
    # Keep slightly more than the longest row-based window (7 days).
    ws["_history_df"] = train_grp.tail(WIN_TDD_LOCAL + 24).copy()

    return ws


def _compute_long_window_features(
    df:        pd.DataFrame,
    warmstart: "dict | None" = None,
) -> pd.DataFrame:
    """
    Features requiring causal post-split recomputation.
    [ZT-CRIT-NEW-1/2] also calls _compute_exercise_features_causal().

    [FIX-ISSUE-15] Accepts an optional `warmstart` dict produced by
    _extract_train_warmstart() so the val split begins its rolling windows
    from the correct prior state rather than cold-starting from scratch.
    Without this, the first ~24h of val features (tdd_rolling_7d,
    correction_bolus_prior, ic_ratio_deviation) are systematically biased low.
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    ws = warmstart or {}

    # Exact causal warm-start: prepend the real train tail when it is truly
    # contiguous with the current split, recompute all trailing features once,
    # then return only the current rows.  This is not target leakage: every
    # prepended observation is strictly earlier than the first current row.
    # If there is a material timestamp gap, do NOT pretend the row-count rolling
    # windows are contiguous; fall back to the scalar/cold-start logic below.
    history = ws.get("_history_df")
    if isinstance(history, pd.DataFrame) and not history.empty and not df.empty:
        hist = history.copy().sort_values("timestamp").reset_index(drop=True)
        hist_end = pd.Timestamp(hist["timestamp"].iloc[-1])
        cur_start = pd.Timestamp(df["timestamp"].iloc[0])
        boundary_gap = cur_start - hist_end
        if pd.Timedelta(0) < boundary_gap <= pd.Timedelta("10min"):
            current_n = len(df)
            combined = pd.concat([hist, df], ignore_index=True, sort=False)
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            recomputed = _compute_long_window_features(combined, warmstart=None)
            return recomputed.tail(current_n).reset_index(drop=True)
        if boundary_gap <= pd.Timedelta(0):
            raise RuntimeError(
                "[AUDIT-WARMSTART] Train-tail history overlaps or follows the "
                f"current split (gap={boundary_gap}). Refusing warm-start."
            )
        log.warning(
            f"  [AUDIT-WARMSTART] Boundary gap {boundary_gap} is >10 min; "
            "not prepending row-based train history."
        )

    # [LEAK-A] weekend_meal_prior
    df["weekend_meal_prior"] = _compute_weekend_meal_prior(df)

    # [LEAK-D] ic_ratio_deviation
    if "carb_to_bolus_ratio_1h" in df.columns:
        ratio_ma = (
            df["carb_to_bolus_ratio_1h"]
            .rolling(24 * 12, min_periods=24 * 12)
            .median()
        )
        raw_dev = df["carb_to_bolus_ratio_1h"] - ratio_ma
        # [FIX-ISSUE-15] If a warm-start median is available from the train tail,
        # fill leading warm-up NaNs with the deviation from that carried-over baseline
        # instead of 0.0.  Strictly causal: prior comes from train split only.
        ic_warmstart_fill = ws.get("ic_ratio_ma_last", None)
        if ic_warmstart_fill is not None:
            ic_warmstart_dev = df["carb_to_bolus_ratio_1h"] - float(ic_warmstart_fill)
            raw_dev = raw_dev.where(ratio_ma.notna(), other=ic_warmstart_dev)
        # [FIX-ISSUE-7] Any remaining NaNs (no warmstart or warmstart also None) → 0.0
        df["ic_ratio_deviation"] = raw_dev.fillna(0.0)
    else:
        df["ic_ratio_deviation"] = 0.0

    # [AUDIT-HIGH-3] dynamic_isf_estimate
    if "glucose_mg_dl" in df.columns and "insulin_on_board" in df.columns:
        g_change    = df["glucose_mg_dl"].diff(12)
        iob_change  = df["insulin_on_board"].diff(12)
        iob_falling = iob_change < -0.05
        raw_isf     = g_change / (iob_change.abs() + 1e-6)
        df["dynamic_isf_estimate"] = (
            raw_isf
            .where(iob_falling, other=np.nan)
            .rolling(24 * 12, min_periods=24 * 12)
            .median()
            .fillna(50.0)
            .clip(-200.0, 200.0)
        )
    else:
        df["dynamic_isf_estimate"] = 50.0

    # tdd_rolling_7d / bolus_fraction_of_tdd
    # [FIX-ISSUE-15] Blend val's own growing rolling sum with the train-tail prior
    # using a linear ramp over the first WIN_TDD_LOCAL/2 steps.  This eliminates
    # the cold-start underestimation of TDD at the start of the val period.
    if "bolus_event" in df.columns and "basal_rate" in df.columns:
        WIN_TDD_LOCAL = 7 * 24 * 12
        bolus_rolling = df["bolus_event"].rolling(WIN_TDD_LOCAL, min_periods=1).sum()
        basal_daily   = (
            df["basal_rate"].rolling(WIN_TDD_LOCAL, min_periods=1).mean() * 24.0
        )
        # U/day, not a 7-day cumulative total.
        tdd_raw = (bolus_rolling + basal_daily * 7.0) / 7.0
        if "tdd_last" in ws:
            tdd_prior    = float(ws["tdd_last"])
            n_steps      = pd.Series(np.arange(1, len(df) + 1), index=df.index)
            weight_own   = (n_steps / (WIN_TDD_LOCAL / 2)).clip(upper=1.0)
            weight_prior = 1.0 - weight_own
            df["tdd_rolling_7d"] = weight_own * tdd_raw + weight_prior * tdd_prior
        else:
            df["tdd_rolling_7d"] = tdd_raw
        if "bolus_last_1h" in df.columns:
            # Dimensionless share of the daily dose delivered as bolus in the
            # preceding hour.  Both numerator and denominator are insulin units.
            df["bolus_fraction_of_tdd"] = (
                df["bolus_last_1h"] / (df["tdd_rolling_7d"] + 1e-6)
            ).clip(0.0, 1.0)

    # [ZT-HIGH-3] + [LEAK-C] hr_recovery_slope
    # [FIX-ISSUE-18] Forward hr_resting_last warmstart so recovery-slope uses the
    # same resting-HR baseline as hr_resting_estimate and hr_above_resting.
    df["hr_recovery_slope"] = _compute_hr_recovery_slope_causal(
        df, hr_resting_warmstart=ws.get("hr_resting_last")
    )

    # [ZT-CRIT-NEW-1/2] Exercise and wearable features using causal resting HR
    # [FIX-ISSUE-15] Pass hr_resting warm-start to avoid cold-starting the val
    # split's rolling 10th-pct HR estimate from the first observed value.
    df = _compute_exercise_features_causal(
        df, hr_resting_warmstart=ws.get("hr_resting_last")
    )

    # [ZT-INFO-2] composite_stress_index is already recomputed inside
    # _compute_exercise_features_causal(), so no additional call needed here.

    # [FIX-H2] Recompute glucose_lag_* post-split on the isolated split series.
    # Previously shift() ran on the full patient train series before the 85/15
    # split, making glucose_lag_1 at val[0] = last train glucose (cross-boundary
    # leak).  Here the shift runs on each split in isolation.
    # Warm-up fill: use the first non-NaN glucose value in this split only.
    _first_glucose = (
        float(df["glucose_mg_dl"].dropna().iloc[0])
        if df["glucose_mg_dl"].notna().any() else 100.0
    )
    for _lag in [1, 2, 3, 6, 12, 24]:
        df[f"glucose_lag_{_lag}"] = (
            df["glucose_mg_dl"].shift(_lag).fillna(_first_glucose)
        )

    # [FIX-BASAL-DEV] Recompute basal_rate_deviation post-split.
    # The rolling(WIN_BASELINE=12).mean() baseline is now computed on the
    # isolated split only, so the window never crosses the train/val boundary.
    _WIN_BASELINE_BD = 12
    if "basal_rate" in df.columns:
        basal_ma = df["basal_rate"].rolling(_WIN_BASELINE_BD, min_periods=3).mean()
        df["basal_rate_deviation"] = (df["basal_rate"] - basal_ma).fillna(0.0)
    else:
        df["basal_rate_deviation"] = 0.0

    # [FIX-ROLL-BOUNDARY] Recompute rolling glucose regularity and DFA features
    # post-split.  When computed on the full unsplit series, windows of 12–24
    # steps at the split boundary incorporate future (val-side) glucose values.
    # Recomputing on the isolated split guarantees strict causality.
    g = df["glucose_mg_dl"]
    _WIN_ENTROPY_LOCAL = 12
    _WIN_TIR_LOCAL     = 24

    def _sample_entropy_local(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
        from scipy.spatial.distance import pdist as _pdist
        n   = len(x)
        std = np.std(x)
        if n < m + 2 or std < 1e-6:
            return 0.0
        r = r_factor * std
        tm  = np.array([x[i: i + m]     for i in range(n - m)],     dtype=float)
        tm1 = np.array([x[i: i + m + 1] for i in range(n - m - 1)], dtype=float)
        B = int(np.sum(_pdist(tm[:-1],  metric="chebyshev") < r)) * 2
        A = int(np.sum(_pdist(tm1,      metric="chebyshev") < r)) * 2
        if B <= 0 or A <= 0:
            return 0.0
        return float(-np.log(A / B))

    def _dfa_alpha_local(x: np.ndarray, scales: tuple = (4, 6, 8, 12)) -> float:
        if len(x) < max(scales) * 2:
            return 0.5
        x_cum = np.cumsum(x - np.mean(x))
        flucts, vscales = [], []
        for s in scales:
            n_seg = len(x_cum) // s
            if n_seg < 2:
                continue
            segs = x_cum[: n_seg * s].reshape(n_seg, s)
            t = np.arange(s, dtype=float)
            A = np.column_stack([t, np.ones(s)])
            coeffs, _, _, _ = np.linalg.lstsq(A, segs.T, rcond=None)
            flucts.append(np.sqrt(np.mean((segs - (A @ coeffs).T) ** 2)))
            vscales.append(s)
        if len(flucts) < 2:
            return 0.5
        return float(np.polyfit(np.log(vscales), np.log(np.array(flucts) + 1e-12), 1)[0])

    df["glucose_sample_entropy_60m"] = (
        g.rolling(_WIN_ENTROPY_LOCAL, min_periods=_WIN_ENTROPY_LOCAL // 2)
         .apply(_sample_entropy_local, raw=True).fillna(0.0)
    )
    df["glucose_tir_2h"] = (
        g.rolling(_WIN_TIR_LOCAL, min_periods=_WIN_TIR_LOCAL // 2)
         .apply(lambda x: float(((x >= 70) & (x <= 180)).mean()) if len(x) > 0 else 0.5, raw=True)
         .fillna(0.5)
    )
    df["glucose_hyper_ratio_2h"] = (
        g.rolling(_WIN_TIR_LOCAL, min_periods=_WIN_TIR_LOCAL // 2)
         .apply(lambda x: float((x > 180).mean()) if len(x) > 0 else 0.0, raw=True)
         .fillna(0.0)
    )
    df["glucose_hypo_ratio_2h"] = (
        g.rolling(_WIN_TIR_LOCAL, min_periods=_WIN_TIR_LOCAL // 2)
         .apply(lambda x: float((x < 70).mean()) if len(x) > 0 else 0.0, raw=True)
         .fillna(0.0)
    )
    df["glucose_dfa_alpha_2h"] = (
        g.rolling(_WIN_TIR_LOCAL, min_periods=_WIN_TIR_LOCAL // 2)
         .apply(_dfa_alpha_local, raw=True).fillna(0.5)
    )

    # [FIX-ROLL-BOUNDARY] Recompute bolus_count_3h post-split.
    # The 36-step rolling window on the full series incorporated future bolus
    # events for the 36 rows nearest the split boundary.
    if "bolus_event" in df.columns:
        df["bolus_count_3h"] = (
            (df["bolus_event"] > 0).astype(float)
            .rolling(36, min_periods=1).sum()
        )
    else:
        df["bolus_count_3h"] = 0.0

    # [FIX-NADIR-ASYM] Recompute glucose_nadir_proximity post-split.
    # rolling(WIN_TIR=24).min() on the full train-XML series caused the first
    # 24 val rows to incorporate training-side glucose values.  Recomputing
    # here on the isolated split guarantees strict causality.
    _WIN_NADIR = 24  # WIN_TIR
    g_nadir = df["glucose_mg_dl"]
    rolling_min_2h = g_nadir.rolling(_WIN_NADIR, min_periods=6).min()
    df["glucose_nadir_proximity"] = (g_nadir - rolling_min_2h).fillna(0.0)

    # [FIX-NADIR-ASYM] Recompute glucose_asymmetry_index post-split.
    # rolling(WIN_TIR=24).sum() had the same cross-boundary contamination.
    hypo_time  = (g_nadir < 70).astype(float).rolling(_WIN_NADIR, min_periods=6).sum()
    hyper_time = (g_nadir > 180).astype(float).rolling(_WIN_NADIR, min_periods=6).sum()
    df["glucose_asymmetry_index"] = (
        hypo_time / (hyper_time + 1.0)
    ).clip(0.0, 10.0).fillna(0.0)

    # [FIX-CRIT-1] Recompute basal_bolus_ratio post-split.
    # rolling(24*12, min_periods=12).sum() on the full series contaminated the
    # 288 rows nearest the split boundary with future bolus events.
    if "bolus_event" in df.columns and "basal_rate" in df.columns:
        basal_24h = (
            df["basal_rate"].rolling(24 * 12, min_periods=12).mean() * 24.0
        )
        bolus_24h = df["bolus_event"].rolling(24 * 12, min_periods=12).sum()
        df["basal_bolus_ratio"] = (
            basal_24h / (bolus_24h + 1e-6)
        ).clip(0.0, 10.0).fillna(1.0)
    else:
        df["basal_bolus_ratio"] = 1.0

    # [FIX-CRIT-1] Recompute lbgi_30m, hbgi_30m, bgri post-split.
    # rolling(6).mean() on the full series contaminated the 6 rows nearest the
    # split boundary with future glucose values.
    if "glucose_mg_dl" in df.columns:
        g_risk = df["glucose_mg_dl"].clip(lower=1.0)
        f_risk = 1.509 * (np.log(g_risk) ** 1.084 - 5.381)
        rl     = 10 * (f_risk.clip(upper=0.0)) ** 2
        rh     = 10 * (f_risk.clip(lower=0.0)) ** 2
        df["lbgi_30m"] = rl.rolling(6, min_periods=3).mean().fillna(0.0)
        df["hbgi_30m"] = rh.rolling(6, min_periods=3).mean().fillna(0.0)
        df["bgri"]     = df["lbgi_30m"] + df["hbgi_30m"]
    else:
        df["lbgi_30m"] = 0.0
        df["hbgi_30m"] = 0.0
        df["bgri"]     = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL QUANTILE LOSS
# ─────────────────────────────────────────────────────────────────────────────
class ClinicalQuantileLoss(QuantileLoss):
    HYPO_THRESHOLD:      float = 70.0
    HYPO_PENALTY_WEIGHT: float = 2.5

    _LOSS_SANITY_MIN: float = 0.001
    _LOSS_SANITY_MAX: float = 200.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._validated: bool = False

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(target, (tuple, list)):
            # MultiHorizonMetric normally unwraps (target, weight) before loss(),
            # but keep this defensive path correct for direct/unit-test calls.
            target_mgdl = target[0]
        else:
            target_mgdl = target

        base_loss = super().loss(y_pred, target_mgdl)

        if base_loss.ndim == 3:
            base_loss = base_loss.mean(dim=-1)

        if base_loss.ndim != 2:
            raise ValueError(
                f"ClinicalQuantileLoss: base_loss shape {tuple(base_loss.shape)}, "
                f"expected (batch, horizon)."
            )

        if not self._validated:
            loss_mean = base_loss.mean().item()
            t_min = float(target_mgdl.min().item())
            t_max = float(target_mgdl.max().item())
            if not (self._LOSS_SANITY_MIN <= loss_mean <= self._LOSS_SANITY_MAX):
                warnings.warn(
                    f"ClinicalQuantileLoss: base_loss mean={loss_mean:.4f} "
                    f"outside expected range [{self._LOSS_SANITY_MIN}, "
                    f"{self._LOSS_SANITY_MAX}]. Check for scale mismatch.",
                    RuntimeWarning, stacklevel=2,
                )
            if not (20.0 <= t_min and t_max <= 600.0):
                warnings.warn(
                    f"ClinicalQuantileLoss: target range "
                    f"[{t_min:.1f}, {t_max:.1f}] outside physiological range.",
                    RuntimeWarning, stacklevel=2,
                )
            self._validated = True

        hypo_mask = (target_mgdl < self.HYPO_THRESHOLD).float()
        weights   = 1.0 + (self.HYPO_PENALTY_WEIGHT - 1.0) * hypo_mask

        if weights.shape != base_loss.shape:
            raise ValueError(
                f"Weight shape {tuple(weights.shape)} != "
                f"base_loss shape {tuple(base_loss.shape)}."
            )

        return base_loss * weights


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL TFT
# ─────────────────────────────────────────────────────────────────────────────
class ClinicalTFT(TemporalFusionTransformer):
    COSINE_EPOCHS: int = 15

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-2,
            eps=1e-7,
        )

        total_steps     = self.trainer.estimated_stepping_batches
        max_epochs      = self.trainer.max_epochs
        steps_per_epoch = max(1, total_steps // max_epochs)
        t0_steps        = self.COSINE_EPOCHS * steps_per_epoch
        warmup_steps    = max(50, total_steps // 20)

        log.info(
            f"ClinicalTFT LR schedule: warmup={warmup_steps} steps, "
            f"cosine T_0={t0_steps} steps"
        )

        def _warmup_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        warmup_sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=_warmup_lambda
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0_steps, T_mult=2, eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
                "frequency": 1,
                "monitor":   "val_loss",
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Population TFT for glucose forecasting"
    )
    p.add_argument("--epochs",               type=int,   default=60)
    p.add_argument("--batch-size",           type=int,   default=64)
    p.add_argument("--horizon",              type=int,   default=MAX_PREDICTION_LENGTH)
    p.add_argument("--context",              type=int,   default=MAX_ENCODER_LENGTH)
    p.add_argument("--lr",                   type=float, default=3e-4)
    p.add_argument("--hidden-size",          type=int,   default=64)
    p.add_argument("--attention-heads",      type=int,   default=4)
    p.add_argument("--dropout",              type=float, default=0.3)
    p.add_argument("--hidden-continuous-size", type=int, default=16)
    p.add_argument("--lstm-layers",          type=int,   choices=(1, 2), default=1)
    p.add_argument("--gradient-clip",        type=float, default=1.0)
    p.add_argument("--no-gpu",    action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-swa",    action="store_true")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPER
# ─────────────────────────────────────────────────────────────────────────────
def find_best_checkpoint(prefix: str = "tft-pop-audited") -> Optional[str]:
    all_ckpts = [
        c for c in MODEL_DIR.glob(f"{prefix}-*.ckpt")
        if "last" not in c.name
    ]

    if not all_ckpts:
        log.info("  No existing checkpoints found — starting fresh.")
        return None

    pattern = rf"^{re.escape(prefix)}-epoch=(\d+)-val_loss=(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$"
    parseable:   list[tuple[float, Path]] = []
    unparseable: list[Path]               = []

    for path in all_ckpts:
        match = re.match(pattern, path.stem)
        if match:
            try:
                loss_val = float(match.group(2))
                if 0.0 < loss_val < 1_000.0:
                    parseable.append((loss_val, path))
                    continue
            except ValueError:
                pass
        unparseable.append(path)

    if unparseable:
        log.warning(
            f"  {len(unparseable)} checkpoint(s) could not be parsed: "
            f"{[p.name for p in unparseable]}"
        )

    if parseable:
        parseable.sort(key=lambda t: t[0])
        best_loss, best_path = parseable[0]
        log.info(f"  Best checkpoint: {best_path.name}  (val_loss={best_loss:.6g})")
        return str(best_path)

    log.warning("  All checkpoints unparseable. Falling back to newest by mtime.")
    newest = max(unparseable, key=lambda p: p.stat().st_mtime)
    return str(newest)


# ─────────────────────────────────────────────────────────────────────────────
# TIME-INDEX ASSIGNMENT  [AUDIT-CRIT-1] [ZT-CRIT-3] [LEAK-B]
# ─────────────────────────────────────────────────────────────────────────────
def _timestamp_steps_5min(timestamps: pd.Series) -> np.ndarray:
    """Convert real timestamps to integer 5-minute offsets without collapsing gaps."""
    ts = pd.to_datetime(timestamps, errors="raise")
    if len(ts) == 0:
        return np.array([], dtype=np.int64)
    delta_min = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 60.0
    steps_f   = delta_min / 5.0
    steps     = np.rint(steps_f).astype(np.int64)
    if not np.allclose(steps_f, steps, atol=1e-6, rtol=0.0):
        bad = float(np.max(np.abs(steps_f - steps)))
        raise RuntimeError(
            f"Timestamps are not aligned to the expected 5-minute grid "
            f"(max fractional-step error={bad:.6g})."
        )
    if len(steps) > 1 and np.any(np.diff(steps) <= 0):
        raise RuntimeError("Timestamps must be strictly increasing within each patient.")
    return steps


def _assign_gapped_time_idx(
    df:  pd.DataFrame,
    gap: int = TIME_IDX_PATIENT_GAP,
) -> pd.DataFrame:
    return _assign_gapped_time_idx_from_offset(df, start_offset=0, gap=gap)


def _assign_gapped_time_idx_from_offset(
    df:           pd.DataFrame,
    start_offset: int,
    gap:          int = TIME_IDX_PATIENT_GAP,
) -> pd.DataFrame:
    # [AUDIT-TIME-1] Preserve real missing intervals. The previous np.arange(n)
    # collapsed every dropped CGM outage, so a multi-hour gap could look like a
    # single 5-minute step. allow_missing_timesteps=True only works when time_idx
    # itself retains those missing integer steps.
    df = df.copy()
    df["time_idx"] = 0
    current_offset = int(start_offset)

    for patient_id in df[SUBJECT_COL].unique():
        idx = df.index[df[SUBJECT_COL] == patient_id]
        if len(idx) == 0:
            continue
        ordered_idx = df.loc[idx].sort_values("timestamp").index
        rel_steps   = _timestamp_steps_5min(df.loc[ordered_idx, "timestamp"])
        values      = rel_steps + current_offset
        df.loc[ordered_idx, "time_idx"] = values
        current_offset = int(values[-1]) + 1 + gap

    return df


def _insert_split_gaps(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve real time gaps and add an embargo so val encoders cannot enter train."""
    train_df = train_df.copy()
    val_df   = val_df.copy()
    train_df["time_idx"] = 0
    val_df["time_idx"]   = 0

    current_offset = 0
    patients = sorted(set(train_df[SUBJECT_COL].unique()) |
                      set(val_df[SUBJECT_COL].unique()))

    patient_last_train: dict[str, int] = {}
    patient_first_val:  dict[str, int] = {}

    for patient_id in patients:
        tr_idx = train_df.index[train_df[SUBJECT_COL] == patient_id]
        if len(tr_idx) > 0:
            tr_idx = train_df.loc[tr_idx].sort_values("timestamp").index
            tr_rel = _timestamp_steps_5min(train_df.loc[tr_idx, "timestamp"])
            tr_values = tr_rel + current_offset
            train_df.loc[tr_idx, "time_idx"] = tr_values
            patient_last_train[patient_id] = int(tr_values[-1])
            current_offset = int(tr_values[-1]) + 1

        va_idx = val_df.index[val_df[SUBJECT_COL] == patient_id]
        if len(va_idx) > 0:
            if patient_id in patient_last_train:
                current_offset = patient_last_train[patient_id] + 1 + TRAIN_VAL_GAP
            va_idx = val_df.loc[va_idx].sort_values("timestamp").index
            va_rel = _timestamp_steps_5min(val_df.loc[va_idx, "timestamp"])
            va_values = va_rel + current_offset
            val_df.loc[va_idx, "time_idx"] = va_values
            patient_first_val[patient_id] = int(va_values[0])
            current_offset = int(va_values[-1]) + 1 + TIME_IDX_PATIENT_GAP
        elif patient_id in patient_last_train:
            current_offset = patient_last_train[patient_id] + 1 + TIME_IDX_PATIENT_GAP

    for patient_id in patients:
        if patient_id not in patient_last_train or patient_id not in patient_first_val:
            continue
        last_train        = patient_last_train[patient_id]
        first_val         = patient_first_val[patient_id]
        earliest_lookback = first_val - MAX_ENCODER_LENGTH
        if earliest_lookback <= last_train:
            raise RuntimeError(
                f"[LEAK-B] Patient {patient_id}: val encoder lookback "
                f"({earliest_lookback}) reaches into training space "
                f"(last_train={last_train}). Increase TRAIN_VAL_GAP "
                f"(currently {TRAIN_VAL_GAP})."
            )

    log.info(
        f"  [LEAK-B/AUDIT-TIME-1] Train/val embargo={TRAIN_VAL_GAP} steps; "
        f"real timestamp gaps preserved for all {len(patients)} patients."
    )

    return train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess_data(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading data...")
    parquet_path = data_dir / "training.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {parquet_path}\n"
            "Run: python ml/scripts/preprocess_ohiot1dm.py"
        )

    df = pd.read_parquet(parquet_path)
    n_before = len(df)
    df = df[df["source_split"] == "train"].copy()
    log.info(f"  Filtered to source_split='train': {len(df):,} / {n_before:,} rows")
    df[SUBJECT_COL] = df[SUBJECT_COL].astype(str)

    # [LEAK-1] Guard
    if "correction_bolus_prior" in df.columns:
        raise RuntimeError(
            "[LEAK-1] 'correction_bolus_prior' found in training.parquet. "
            "Delete training.parquet and rerun preprocess_ohiot1dm.py."
        )
    log.info("  [LEAK-1 guard] correction_bolus_prior absent from parquet ✓")

    # Pre-split fillna — skip sentinel columns
    for col in TIME_VARYING_UNKNOWN_REALS:
        if col in PRE_SPLIT_FILLNA_SKIP:
            continue
        if col in df.columns and df[col].dtype in (float, "float32", "float64"):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    POST_SPLIT_COMPUTED = {
        "correction_bolus_prior",
        "weekend_meal_prior",
        "ic_ratio_deviation",
        "dynamic_isf_estimate",
        "tdd_rolling_7d",
        "bolus_fraction_of_tdd",
        "hr_recovery_slope",
        # [ZT-CRIT-NEW-1/2]
        "hr_resting_estimate",
        "hr_reserve_pct",
        "is_aerobic_exercise",
        "is_stress_response",
        "hr_above_resting",
        "exercise_iob_danger",
        "exercise_cob_coverage",
        "composite_stress_index",
        # [FIX-ISSUE-11]
        "autonomic_stress_index",
        # [FIX-C2]
        "gsr_stress_deviation",
        "skin_temp_deviation",
        # [FIX-BASAL-DEV]
        "basal_rate_deviation",
        # [FIX-ROLL-BOUNDARY]
        "bolus_count_3h",
        "glucose_sample_entropy_60m",
        "glucose_tir_2h",
        "glucose_hyper_ratio_2h",
        "glucose_hypo_ratio_2h",
        "glucose_dfa_alpha_2h",
        # [FIX-NADIR-ASYM] sentinel 0.0 in parquet, recomputed post-split
        "glucose_nadir_proximity",
        "glucose_asymmetry_index",
        # [FIX-TDD-SENTINEL] sentinel 0.0 in parquet, recomputed post-split
        "tdd_rolling_7d",
        "bolus_fraction_of_tdd",
        # [FIX-CRIT-1] sentinel 0.0 in parquet, recomputed post-split
        "basal_bolus_ratio",
        "lbgi_30m",
        "hbgi_30m",
        "bgri",
        # [FIX-CRIT-1] sentinel NaN in parquet, recomputed post-split
        "ic_ratio_deviation",
    }
    unknown_reals_check = [
        c for c in TIME_VARYING_UNKNOWN_REALS
        if c not in POST_SPLIT_COMPUTED and c != TARGET_COL
    ]
    required_cols = (
        [SUBJECT_COL, "timestamp", TARGET_COL]
        + TIME_VARYING_KNOWN_REALS
        + unknown_reals_check
        + STATIC_CATEGORICALS
        + ["bolus_event"]
    )
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in training data: {missing}\n"
            "Re-run preprocess_ohiot1dm.py to regenerate all features."
        )

    df = df.sort_values([SUBJECT_COL, "timestamp"]).reset_index(drop=True)

    for col in ["glucose_delta_1", "glucose_delta_3", "iob_delta", "iob_acceleration"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # ── Split 85/15 ────────────────────────────────────────────────────────
    log.info("  Splitting patients 85/15 (time_idx assigned post-split)...")

    train_parts: list[pd.DataFrame] = []
    val_parts:   list[pd.DataFrame] = []
    patients = sorted(df[SUBJECT_COL].unique())
    log.info(f"  Found {len(patients)} patients: {patients}")

    for patient in patients:
        pdf      = df[df[SUBJECT_COL] == patient].copy()
        n        = len(pdf)
        split_at = int(n * 0.85)

        min_required = MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH
        if split_at < min_required:
            log.warning(f"  Patient {patient}: train split ({split_at}) < minimum.")
        if (n - split_at) < min_required:
            log.warning(f"  Patient {patient}: val split ({n - split_at}) < minimum.")

        train_parts.append(pdf.iloc[:split_at])
        val_parts.append(pdf.iloc[split_at:])
        log.info(
            f"  Patient {patient}: total={n:,}  "
            f"train={split_at:,}  val={n - split_at:,}"
        )

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df   = pd.concat(val_parts,   ignore_index=True)

    # Post-split feature computation — each split in isolation
    log.info("  Computing correction_bolus_prior post-split... [AUDIT-CRIT-2]")
# [FIX-FINAL] Brutalne wyczyszczenie indeksów, żeby zabić błąd "duplicate labels"
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    log.info("  Computing correction_bolus_prior post-split... [AUDIT-CRIT-2]")

    # Używamy transform - to jest najbezpieczniejsza i najszybsza metoda w Pandas
    train_df["correction_bolus_prior"] = (
        train_df.groupby(SUBJECT_COL)["bolus_event"]
        .transform(lambda x: (x > 0).astype(float).rolling(24 * 12, min_periods=1).mean())
        .fillna(0.0)
    )
    val_df["correction_bolus_prior"] = (
        val_df.groupby(SUBJECT_COL)["bolus_event"]
        .transform(lambda x: (x > 0).astype(float).rolling(24 * 12, min_periods=1).mean())
        .fillna(0.0)
    )
    # Validation starts immediately after each patient's train segment, so use
    # the actual preceding bolus history rather than an approximate scalar blend.
    val_df_parts: list[pd.DataFrame] = []
    for patient in patients:
        tr_grp = train_df[train_df[SUBJECT_COL] == patient].copy()
        va_grp = val_df[val_df[SUBJECT_COL] == patient].copy()
        if va_grp.empty:
            continue
        va_grp = va_grp.sort_values("timestamp").reset_index(drop=True)
        va_grp["correction_bolus_prior"] = _compute_correction_bolus_prior_with_history(
            va_grp, tr_grp if not tr_grp.empty else None
        ).to_numpy()
        val_df_parts.append(va_grp)
    val_df = pd.concat(val_df_parts, ignore_index=True) if val_df_parts else val_df

    log.info("  Computing long-window features post-split... [FIX-ISSUE-15: warmstart active]")
    # Extract train-tail rolling state per patient BEFORE overwriting train features.
    train_warmstarts: dict[str, dict] = {
        patient: _extract_train_warmstart(
            train_df[train_df[SUBJECT_COL] == patient]
        )
        for patient in patients
        if not train_df[train_df[SUBJECT_COL] == patient].empty
    }
    train_df = pd.concat([
        _compute_long_window_features(grp)
        for _, grp in train_df.groupby(SUBJECT_COL)
    ], ignore_index=True)
    val_df = pd.concat([
        _compute_long_window_features(
            grp,
            warmstart=train_warmstarts.get(str(patient_id)),
        )
        for patient_id, grp in val_df.groupby(SUBJECT_COL)
    ], ignore_index=True)

    # Final cleanup
    POST_SPLIT_FINAL_FILLNA_SKIP = frozenset({
        "ic_ratio_deviation",
        # [ZT-CRIT-NEW-1/2] These are now computed by _compute_exercise_features_causal()
        # inside _compute_long_window_features(), so inf-only cleanup is sufficient.
        "hr_resting_estimate",
        "hr_reserve_pct",
        "is_aerobic_exercise",
        "is_stress_response",
        "hr_above_resting",
        "exercise_iob_danger",
        "exercise_cob_coverage",
        "composite_stress_index",
        # [FIX-ISSUE-11] recomputed post-split; inf-only cleanup only
        "autonomic_stress_index",
        # [FIX-C2] recomputed post-split; inf-only cleanup only
        "gsr_stress_deviation",
        "skin_temp_deviation",
        # [FIX-BASAL-DEV] recomputed post-split; inf-only cleanup only
        "basal_rate_deviation",
        # [FIX-ROLL-BOUNDARY] recomputed post-split; inf-only cleanup only
        "bolus_count_3h",
        "glucose_sample_entropy_60m",
        "glucose_tir_2h",
        "glucose_hyper_ratio_2h",
        "glucose_hypo_ratio_2h",
        "glucose_dfa_alpha_2h",
        # [FIX-NADIR-ASYM] recomputed post-split; inf-only cleanup only
        "glucose_nadir_proximity",
        "glucose_asymmetry_index",
        # [FIX-TDD-SENTINEL] recomputed post-split; inf-only cleanup only
        "tdd_rolling_7d",
        "bolus_fraction_of_tdd",
        # [FIX-CRIT-1] recomputed post-split; inf-only cleanup only
        "basal_bolus_ratio",
        "lbgi_30m",
        "hbgi_30m",
        "bgri",
        # [FIX-CRIT-1] ic_ratio_deviation: NaN sentinel, recomputed post-split;
        # already in POST_SPLIT_FINAL_FILLNA_SKIP via ic_ratio_deviation above.
        "ic_ratio_deviation",
    })
    for col in PRE_SPLIT_FILLNA_SKIP:
        if col in POST_SPLIT_FINAL_FILLNA_SKIP:
            continue
        if col in train_df.columns:
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if col in val_df.columns:
            val_df[col] = val_df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Inf-only cleanup for ic_ratio_deviation [LEAK-D]
    for df_ in (train_df, val_df):
        if "ic_ratio_deviation" in df_.columns:
            mask = ~np.isfinite(df_["ic_ratio_deviation"])
            df_.loc[mask, "ic_ratio_deviation"] = 0.0

    # Inf-only cleanup for ZT-CRIT-NEW-1/2 + FIX-ISSUE-11 + FIX-C2 columns
    _new_sentinel_cols = [
        "hr_resting_estimate", "hr_reserve_pct", "is_aerobic_exercise",
        "is_stress_response", "hr_above_resting", "exercise_iob_danger",
        "exercise_cob_coverage", "composite_stress_index",
        "autonomic_stress_index",        # [FIX-ISSUE-11]
        "gsr_stress_deviation",          # [FIX-C2]
        "skin_temp_deviation",           # [FIX-C2]
        "basal_rate_deviation",          # [FIX-BASAL-DEV]
        "bolus_count_3h",                # [FIX-ROLL-BOUNDARY]
        "glucose_sample_entropy_60m",    # [FIX-ROLL-BOUNDARY]
        "glucose_tir_2h",                # [FIX-ROLL-BOUNDARY]
        "glucose_hyper_ratio_2h",        # [FIX-ROLL-BOUNDARY]
        "glucose_hypo_ratio_2h",         # [FIX-ROLL-BOUNDARY]
        "glucose_dfa_alpha_2h",          # [FIX-ROLL-BOUNDARY]
        "glucose_nadir_proximity",       # [FIX-NADIR-ASYM]
        "glucose_asymmetry_index",       # [FIX-NADIR-ASYM]
        "tdd_rolling_7d",                # [FIX-TDD-SENTINEL]
        "bolus_fraction_of_tdd",         # [FIX-TDD-SENTINEL]
        "basal_bolus_ratio",             # [FIX-CRIT-1]
        "lbgi_30m",                      # [FIX-CRIT-1]
        "hbgi_30m",                      # [FIX-CRIT-1]
        "bgri",                          # [FIX-CRIT-1]
    ]
    for df_ in (train_df, val_df):
        for col in _new_sentinel_cols:
            if col in df_.columns:
                mask = ~np.isfinite(df_[col])
                df_.loc[mask, col] = 0.0

    # [ZT-CRIT-3] + [LEAK-B] Assign time_idx AFTER split with wider gap
    log.info(f"  Assigning time_idx gap={TRAIN_VAL_GAP} steps [LEAK-B / ZT-CRIT-3]...")
    train_df, val_df = _insert_split_gaps(train_df, val_df)

    # Verify no overlap
    tr_idx_set = set(train_df["time_idx"].tolist())
    va_idx_set = set(val_df["time_idx"].tolist())
    overlap    = tr_idx_set & va_idx_set
    if overlap:
        raise RuntimeError(
            f"[ZT-CRIT-3] time_idx overlap between train and val: "
            f"{len(overlap)} shared indices."
        )
    log.info("  [ZT-CRIT-3] time_idx: no train/val overlap ✓")

    # [FIX-MED-1] CR-8 sentinel-recomputation verification.
    # The old variance heuristic (train_std vs val_std * 3) cannot detect the
    # most likely failure mode: a sentinel column that was never recomputed
    # post-split still holds 0.0 in both splits, so both stds are 0 and the
    # 3× threshold is never breached — the check reports "passed" for exactly
    # the scenario it is supposed to catch.
    #
    # New check: after _compute_long_window_features() has run, every 0.0-sentinel
    # column (except the ones that are legitimately zero for some patients, e.g.
    # exercise features when HR data is absent) must have at least some non-zero
    # values in the training split — if all values are still 0.0, the recomputation
    # silently did nothing and we have a bug.  NaN-sentinel columns (ic_ratio_deviation,
    # dynamic_isf_estimate) must have NO remaining NaNs after recomputation.
    _MUST_HAVE_NONZERO_AFTER_RECOMPUTE = [
        "basal_bolus_ratio", "lbgi_30m", "hbgi_30m", "bgri",
        "glucose_tir_2h", "glucose_nadir_proximity",
        "basal_rate_deviation", "bolus_count_3h",
        "tdd_rolling_7d", "glucose_lag_1",
    ]
    _MUST_HAVE_NO_NAN_AFTER_RECOMPUTE = [
        "ic_ratio_deviation",
        "dynamic_isf_estimate",
    ]
    sentinel_recompute_ok = True
    for col in _MUST_HAVE_NONZERO_AFTER_RECOMPUTE:
        if col not in train_df.columns:
            continue
        if (train_df[col] == 0.0).all():
            log.warning(
                f"  ⚠ [FIX-MED-1] {col} is all-zero in train split after "
                f"post-split recomputation — recomputation may have silently failed. [CR-8]"
            )
            sentinel_recompute_ok = False
    for col in _MUST_HAVE_NO_NAN_AFTER_RECOMPUTE:
        if col not in train_df.columns:
            continue
        n_nan = int(train_df[col].isna().sum())
        if n_nan > 0:
            log.warning(
                f"  ⚠ [FIX-MED-1] {col} has {n_nan} NaN values in train split "
                f"after post-split recomputation — sentinel was not overwritten. [CR-8]"
            )
            sentinel_recompute_ok = False
    if sentinel_recompute_ok:
        log.info("  [FIX-MED-1 / CR-8] Sentinel recomputation check passed ✓")
    else:
        log.warning(
            "  ⚠ [FIX-MED-1 / CR-8] One or more sentinel columns appear "
            "unrecomputed. Review _compute_long_window_features()."
        )

    log.info(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")

    # [FIX] Zmuszenie time_idx do bycia czystym integerem dla PyTorch Forecasting
    train_df["time_idx"] = train_df["time_idx"].astype(int)
    val_df["time_idx"] = val_df["time_idx"].astype(int)

    return train_df, val_df


def load_test_data(
    data_dir:           Path = DATA_DIR,
    train_max_time_idx: int  = 0,
) -> pd.DataFrame:
    parquet_path = data_dir / "training.parquet"
    full_parquet = pd.read_parquet(parquet_path)

    df = full_parquet[full_parquet["source_split"] == "test"].copy()
    log.info(f"  Test set: {len(df):,} rows, {df['subject_id'].nunique()} patients")
    df[SUBJECT_COL] = df[SUBJECT_COL].astype(str)

    if "correction_bolus_prior" in df.columns:
        raise RuntimeError(
            "[LEAK-1] 'correction_bolus_prior' found in test parquet."
        )

    POST_SPLIT_COMPUTED = {
        "correction_bolus_prior", "weekend_meal_prior",
        "ic_ratio_deviation", "dynamic_isf_estimate",
        "tdd_rolling_7d", "bolus_fraction_of_tdd",
        "hr_recovery_slope",
        "hr_resting_estimate", "hr_reserve_pct",
        "is_aerobic_exercise", "is_stress_response",
        "hr_above_resting", "exercise_iob_danger",
        "exercise_cob_coverage", "composite_stress_index",
        "autonomic_stress_index",  # [FIX-ISSUE-11]
        "gsr_stress_deviation",    # [FIX-C2]
        "skin_temp_deviation",     # [FIX-C2]
        "basal_rate_deviation",    # [FIX-BASAL-DEV]
        "bolus_count_3h",          # [FIX-ROLL-BOUNDARY]
        "glucose_sample_entropy_60m",  # [FIX-ROLL-BOUNDARY]
        "glucose_tir_2h",          # [FIX-ROLL-BOUNDARY]
        "glucose_hyper_ratio_2h",  # [FIX-ROLL-BOUNDARY]
        "glucose_hypo_ratio_2h",   # [FIX-ROLL-BOUNDARY]
        "glucose_dfa_alpha_2h",    # [FIX-ROLL-BOUNDARY]
        "glucose_nadir_proximity", # [FIX-NADIR-ASYM]
        "glucose_asymmetry_index", # [FIX-NADIR-ASYM]
        "tdd_rolling_7d",          # [FIX-TDD-SENTINEL]
        "bolus_fraction_of_tdd",   # [FIX-TDD-SENTINEL]
        "basal_bolus_ratio",       # [FIX-CRIT-1]
        "lbgi_30m",                # [FIX-CRIT-1]
        "hbgi_30m",                # [FIX-CRIT-1]
        "bgri",                    # [FIX-CRIT-1]
        "ic_ratio_deviation",      # [FIX-CRIT-1]
    }
    for col in POST_SPLIT_COMPUTED:
        if col in df.columns:
            df = df.drop(columns=[col])

    # [FIX-ISSUE-17] Build per-patient warmstarts from the training-XML data so
    # that the test split's long-window features start from the correct prior
    # state, matching the warm-start treatment already applied to the val split.
    # All warmstart values are derived strictly from source_split='train' rows —
    # no test information is used.
    train_rows = full_parquet[full_parquet["source_split"] == "train"].copy()
    train_rows[SUBJECT_COL] = train_rows[SUBJECT_COL].astype(str)

    test_warmstarts: dict[str, dict] = {}
    for pid in df[SUBJECT_COL].unique():
        tr_grp = train_rows[train_rows[SUBJECT_COL] == pid]
        if not tr_grp.empty:
            test_warmstarts[str(pid)] = _extract_train_warmstart(tr_grp)
            log.info(
                f"  [FIX-ISSUE-17] Test warmstart for patient {pid}: "
                f"tdd_last={test_warmstarts[str(pid)].get('tdd_last', 'N/A'):.1f}, "
                f"hr_resting_last={test_warmstarts[str(pid)].get('hr_resting_last', 'N/A')}"
            )
        else:
            log.warning(
                f"  [FIX-ISSUE-17] No training rows found for test patient {pid} — "
                "test split will cold-start (no warmstart available)."
            )

    df = pd.concat([
        _compute_long_window_features(
            grp,
            warmstart=test_warmstarts.get(str(patient_id)),
        )
        for patient_id, grp in df.groupby(SUBJECT_COL)
    ], ignore_index=True)

    # correction_bolus_prior: use exact train-tail history only when the source
    # XMLs are chronologically contiguous; the helper cold-starts across a gap.
    cbp_parts: list[pd.DataFrame] = []
    for pid in sorted(df[SUBJECT_COL].unique()):
        va_grp = df[df[SUBJECT_COL] == pid].copy()
        if va_grp.empty:
            continue
        tr_grp = train_rows[train_rows[SUBJECT_COL] == str(pid)].copy()
        va_grp = va_grp.sort_values("timestamp").reset_index(drop=True)
        va_grp["correction_bolus_prior"] = _compute_correction_bolus_prior_with_history(
            va_grp, tr_grp if not tr_grp.empty else None
        ).to_numpy()
        cbp_parts.append(va_grp)
    df = pd.concat(cbp_parts, ignore_index=True) if cbp_parts else df

    for col in TIME_VARYING_UNKNOWN_REALS:
        if col in df.columns and df[col].dtype in (float, "float32", "float64"):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df = df.sort_values([SUBJECT_COL, "timestamp"]).reset_index(drop=True)

    test_start_offset = train_max_time_idx + TIME_IDX_PATIENT_GAP
    df = _assign_gapped_time_idx_from_offset(df, start_offset=test_start_offset)

    # [FIX] Czysty integer dla datasetu testowego
    df["time_idx"] = df["time_idx"].astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def create_time_series_dataset(
    df:                pd.DataFrame,
    reference_dataset: Optional[TimeSeriesDataSet] = None,
    predict_mode:      bool = False,
    context_length:    int  = MAX_ENCODER_LENGTH,
    horizon:           int  = MAX_PREDICTION_LENGTH,
) -> TimeSeriesDataSet:
    if reference_dataset is not None:
        return TimeSeriesDataSet.from_dataset(
            reference_dataset,
            df,
            predict=predict_mode,
            stop_randomization=True,
            min_prediction_length=horizon,
            max_prediction_length=horizon,
        )

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=TARGET_COL,
        group_ids=[GROUP_COL],
        min_encoder_length=context_length // 2,
        max_encoder_length=context_length,
        min_prediction_length=1,
        max_prediction_length=horizon,
        static_categoricals=STATIC_CATEGORICALS,
        static_reals=STATIC_REALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        # [FIX-NORM-GROUPNORM] GroupNormalizer replaces EncoderNormalizer.
        # EncoderNormalizer fit per-sample on the encoder window: for val/test
        # samples that window contains validation/test history, so its scaling
        # statistics vary sample-by-sample. GroupNormalizer is instead fitted
        # on the training TimeSeriesDataSet and its fitted state is carried into
        # validation/test via from_dataset(), avoiding target-period refitting.
        target_normalizer=GroupNormalizer(
            groups=[GROUP_COL],
            transformation="log",
            center=True,
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def build_datasets(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    args:     argparse.Namespace,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    log.info("Building TimeSeriesDataSets...")
    training = create_time_series_dataset(
        train_df,
        context_length=args.context,
        horizon=args.horizon,
    )
    # [AUDIT-VAL-1] Full rolling validation, not predict-mode validation.
    # In PyTorch Forecasting, predict=True keeps only the LAST prediction window
    # per group.  With only a handful of patients that makes val_loss and Optuna
    # tuning depend on just a handful of sequences.  For validation we want all
    # eligible rolling windows and a fixed full forecast horizon.
    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        predict=False,
        stop_randomization=True,
        min_prediction_length=args.horizon,
        max_prediction_length=args.horizon,
    )

    # [FIX-NORM-GROUPNORM] The target normalizer must be fitted ONLY on training
    # data and reused for validation.  GroupNormalizer stores its fitted group
    # statistics in norm_.  Also reject unseen validation subjects: this model
    # uses subject_id both as a static categorical and as the normalization group,
    # so it is intentionally a within-subject temporal model, not a cold-start
    # model for unseen patients.
    _train_norm = training.target_normalizer
    _val_norm   = validation.target_normalizer
    for _name, _norm in (("training", _train_norm), ("validation", _val_norm)):
        if not isinstance(_norm, GroupNormalizer):
            raise RuntimeError(
                f"[FIX-NORM-GROUPNORM] {_name} target_normalizer is "
                f"{type(_norm).__name__}, expected GroupNormalizer."
            )
        if getattr(_norm, "norm_", None) is None:
            raise RuntimeError(
                f"[FIX-NORM-GROUPNORM] {_name} GroupNormalizer is not fitted."
            )

    _train_subjects = set(train_df[GROUP_COL].astype(str).unique())
    _val_subjects   = set(val_df[GROUP_COL].astype(str).unique())
    _unseen_subjects = _val_subjects - _train_subjects
    if _unseen_subjects:
        raise RuntimeError(
            "[FIX-NORM-GROUPNORM] Validation contains unseen subject_id values: "
            f"{sorted(_unseen_subjects)}. This configuration is only valid for "
            "within-subject temporal validation."
        )

    try:
        _norm_values = np.asarray(
            _val_norm.get_norm(val_df[[GROUP_COL]].copy()), dtype=float
        )
        if not np.isfinite(_norm_values).all():
            raise RuntimeError(
                "[FIX-NORM-GROUPNORM] Validation normalization contains NaN/Inf."
            )
    except RuntimeError:
        raise
    except Exception as _exc:
        raise RuntimeError(
            f"[FIX-NORM-GROUPNORM] Could not verify validation normalization: {_exc}"
        ) from _exc

    log.info(
        f"  Validation windows: {len(validation):,} "
        f"(rolling, full horizon={args.horizon})"
    )
    return training, validation


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT NORM LOGGER
# ─────────────────────────────────────────────────────────────────────────────
class GradientNormLogger(pl.Callback):
    EXPLOSION_PATIENCE: int = 30

    def __init__(self, clip_val: float = 1.0, warmup_steps: int = 200) -> None:
        super().__init__()
        self.clip_val     = clip_val
        self.warmup_steps = warmup_steps
        self._high_grad_consecutive: int = 0

    def on_after_backward(
        self,
        trainer:   pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        grad_norms = [
            p.grad.detach().norm(2)
            for p in pl_module.parameters()
            if p.grad is not None
        ]
        if not grad_norms:
            return

        grad_norm = float(torch.stack(grad_norms).norm(2).item())
        pl_module.log(
            "train/grad_norm_pre_clip",
            grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        alert_threshold = self.clip_val * 100
        post_warmup     = trainer.global_step > self.warmup_steps

        # [FIX-ISSUE-14] Reset was previously INSIDE the alert block, meaning it fired
        # on every single high-gradient step — the counter could never accumulate past 1
        # and the explosion warning (>= EXPLOSION_PATIENCE) was permanently unreachable.
        # Correct logic: increment while above threshold, reset only when below it.
        if post_warmup and grad_norm > alert_threshold:
            self._high_grad_consecutive += 1
            log.warning(
                f"  ⚠ High gradient: norm={grad_norm:.2f} at step "
                f"{trainer.global_step} "
                f"(consecutive={self._high_grad_consecutive}/{self.EXPLOSION_PATIENCE})"
            )
            if self._high_grad_consecutive >= self.EXPLOSION_PATIENCE:
                log.warning(
                    f"  ⚠ {self.EXPLOSION_PATIENCE} consecutive high-gradient steps. "
                    f"Consider reducing LR or gradient_clip_val."
                )
                # Reset after firing so the warning fires once per burst, not every step.
                self._high_grad_consecutive = 0
        else:
            self._high_grad_consecutive = 0


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def build_model(
    training:  TimeSeriesDataSet,
    args:      argparse.Namespace,
    ckpt_path: Optional[str],
) -> ClinicalTFT:
    loss_fn = ClinicalQuantileLoss(quantiles=QUANTILES)

    if ckpt_path:
        log.info(f"  Resuming from checkpoint: {ckpt_path}")
        log.warning(
            "  [CR-6] Checkpoint resume: CLI hyperparameters are IGNORED. "
            "Use --no-resume to start fresh with new hyperparameters."
        )
        model = ClinicalTFT.load_from_checkpoint(
            ckpt_path,
            map_location="cpu",
            loss=loss_fn,
        )
    else:
        log.info("  Building model from scratch...")
        model = ClinicalTFT.from_dataset(
            training,
            learning_rate=args.lr,
            hidden_size=args.hidden_size,
            attention_head_size=args.attention_heads,
            dropout=args.dropout,
            hidden_continuous_size=args.hidden_continuous_size,
            lstm_layers=args.lstm_layers,
            loss=loss_fn,
            log_interval=10,
            log_val_interval=1,
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Trainable parameters: {n_params:,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(
    model:      ClinicalTFT,
    training:   TimeSeriesDataSet,
    validation: TimeSeriesDataSet,
    args:       argparse.Namespace,
    ckpt_path:  Optional[str],
) -> pl.Trainer:
    if args.num_workers is not None:
        num_workers = args.num_workers
    elif os.name == "nt":
        num_workers = 0
    elif torch.cuda.is_available() and not args.no_gpu:
        num_workers = min(4, os.cpu_count() or 1)
    else:
        num_workers = 0

    use_gpu        = torch.cuda.is_available() and not args.no_gpu
    use_pin_memory = use_gpu
    prefetch       = 2 if num_workers > 0 else None
    mp_context     = None
    if num_workers > 0:
        if use_gpu and os.name != "nt":
            mp_context = "spawn"
        elif os.name != "nt":
            mp_context = "fork"

    train_loader = training.to_dataloader(
        train=True,
        batch_size=args.batch_size,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=use_pin_memory,
        prefetch_factor=prefetch,
        drop_last=True,
        multiprocessing_context=mp_context,
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=args.batch_size * 2,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=use_pin_memory,
        prefetch_factor=prefetch,
        multiprocessing_context=mp_context,
    )

    accelerator = "cpu" if (args.no_gpu or not torch.cuda.is_available()) else "gpu"

    swa_annealing_epochs = 5
    base_patience        = 20

    if not args.no_swa:
        swa_start   = int(args.epochs * 0.75)
        swa_lr      = max(args.lr * 0.05, 1e-5)
        es_patience = base_patience + swa_annealing_epochs
        swa_callback = StochasticWeightAveraging(
            swa_lrs=swa_lr,
            swa_epoch_start=swa_start,
            annealing_epochs=swa_annealing_epochs,
            annealing_strategy="cos",
        )
        log.info(
            f"  SWA enabled: starts epoch {swa_start}, swa_lr={swa_lr:.2e}, "
            f"EarlyStopping patience extended to {es_patience} [ZT-MED-3]"
        )
    else:
        swa_callback = None
        es_patience  = base_patience
        log.info("  SWA disabled (--no-swa flag)")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=es_patience,
        mode="min",
        min_delta=1e-4,
        check_on_train_epoch_end=False,
        verbose=True,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename="tft-pop-audited-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
        verbose=True,
    )
    lr_monitor  = LearningRateMonitor(logging_interval="epoch")
    grad_logger = GradientNormLogger(clip_val=args.gradient_clip, warmup_steps=200)

    callbacks = []
    if swa_callback is not None:
        callbacks.append(swa_callback)
    callbacks += [early_stopping, checkpoint_callback, lr_monitor, grad_logger]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=args.gradient_clip,
        gradient_clip_algorithm="norm",
        precision=32,
        enable_progress_bar=True,
        log_every_n_steps=10,
        num_sanity_val_steps=2,
        callbacks=callbacks,
        deterministic=False,
    )

    log.info("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path if not args.no_resume else None,
    )

    log.info(f"  Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# CLARKE ERROR GRID
# ─────────────────────────────────────────────────────────────────────────────
def clarke_error_grid(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> dict[str, float]:
    """Classify points using the standard Clarke Error Grid decision rules."""
    n = len(y_true)
    if n == 0:
        return {z: 0.0 for z in "ABCDE"}

    yt = y_true.float()
    yp = y_pred.float()

    rel_err = (yp - yt).abs() / yt.abs().clamp(min=1.0)

    # Decision order follows the canonical Clarke implementation: A -> E -> C -> D -> B.
    zone_a = ((yt <= 70.0) & (yp <= 70.0)) | (rel_err <= 0.20)

    remaining = ~zone_a
    zone_e = remaining & (
        ((yt >= 180.0) & (yp <= 70.0))
        | ((yt <= 70.0) & (yp >= 180.0))
    )

    remaining = remaining & ~zone_e
    zone_c = remaining & (
        ((yt >= 70.0) & (yt <= 290.0) & (yp >= yt + 110.0))
        | ((yt >= 130.0) & (yt <= 180.0) & (yp <= (7.0 / 5.0) * yt - 182.0))
    )

    remaining = remaining & ~zone_c
    zone_d = remaining & (
        ((yt >= 240.0) & (yp >= 70.0) & (yp <= 180.0))
        | ((yt <= (175.0 / 3.0)) & (yp >= 70.0) & (yp <= 180.0))
        | (
            (yt >= (175.0 / 3.0))
            & (yt <= 70.0)
            & (yp >= (6.0 / 5.0) * yt)
        )
    )

    zone_b = ~(zone_a | zone_c | zone_d | zone_e)

    counts = {
        "A": int(zone_a.sum().item()),
        "B": int(zone_b.sum().item()),
        "C": int(zone_c.sum().item()),
        "D": int(zone_d.sum().item()),
        "E": int(zone_e.sum().item()),
    }
    assert sum(counts.values()) == n
    return {z: round(counts[z] / n * 100.0, 2) for z in "ABCDE"}


# ─────────────────────────────────────────────────────────────────────────────
# TIME-IDX ALIGNMENT CHECK  [AUDIT-MED-2]
# ─────────────────────────────────────────────────────────────────────────────
def _verify_time_idx_alignment(
    val_df:          pd.DataFrame,
    dec_time_idx:    np.ndarray,
    subject_ids:     np.ndarray,
    coverage_thresh: float = 0.90,
) -> None:
    val_lookup_keys = set(
        zip(val_df[SUBJECT_COL].astype(str), val_df["time_idx"].astype(int))
    )
    last_enc_ti  = dec_time_idx[:, 0] - 1
    lookup_keys  = list(zip(subject_ids.astype(str), last_enc_ti.astype(int)))
    n_found      = sum(1 for k in lookup_keys if k in val_lookup_keys)
    coverage     = n_found / max(len(lookup_keys), 1)

    log.info(
        f"  [AUDIT-MED-2] Persistence lookup alignment: "
        f"{n_found}/{len(lookup_keys)} keys found ({coverage:.1%})"
    )
    if coverage < coverage_thresh:
        raise RuntimeError(
            f"[AUDIT-MED-2] Persistence baseline alignment failure: "
            f"only {coverage:.1%} of decoder_time_idx - 1 keys found in val_df."
        )


# ─────────────────────────────────────────────────────────────────────────────
# [AUDIT-MED-NEW-1] ARIMA BASELINE
# ─────────────────────────────────────────────────────────────────────────────
def _compute_arima_baseline(
    encoder_glucose: np.ndarray,
    horizon:         int,
) -> Optional[np.ndarray]:
    """
    Fit ARIMA(1,1,0) on the encoder window and predict `horizon` steps.
    Returns predicted values in mg/dL, or None on failure.

    [AUDIT-MED-NEW-1] Provides a clinically meaningful comparison baseline
    (differenced autoregressive baseline) alongside flat persistence.
    The ARIMA is fit strictly on the encoder window — no future data.
    """
    if not _STATSMODELS_AVAILABLE:
        return None
    try:
        model  = _ARIMA(encoder_glucose, order=(1, 1, 0))
        result = model.fit(method_kwargs={"warn_convergence": False})
        forecast = result.forecast(steps=horizon)
        return np.array(forecast, dtype=float)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    model:      "ClinicalTFT",
    validation: TimeSeriesDataSet,
    args,
    val_df:     pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate model on a validation/test TimeSeriesDataSet.

    [LEAK-E] Persistence baseline — miss → exclude, not substitute.
    [AUDIT-MED-NEW-1] ARIMA(1,1,0) baseline added alongside flat persistence.
    """
    log.info("Evaluating on validation set...")
    log.info(
        "  [ZT-MED-4] NOTE: metrics are WITHIN-SUBJECT TEMPORAL validation. "
        "They do NOT measure subject-agnostic generalisation."
    )

    val_loader = validation.to_dataloader(
        train=False,
        batch_size=args.batch_size * 2,
        num_workers=0,
        shuffle=False,
    )

    eval_accelerator = (
        "cpu" if (args.no_gpu or not torch.cuda.is_available()) else "gpu"
    )

    predictions = model.predict(
        val_loader,
        return_y=True,
        return_x=True,  # required by persistence/ARIMA reconstruction below
        mode="quantiles",
        trainer_kwargs={"accelerator": eval_accelerator, "logger": False},
    )

    # PyTorch Forecasting returns y as (unscaled_target, sample_weight).  The
    # second element is NOT target_scale.  Quantile predictions returned by
    # model.predict(mode="quantiles") are already transformed back to target
    # space by BaseModel.transform_output().
    y_tuple = predictions.y
    if isinstance(y_tuple, (tuple, list)):
        y_true        = y_tuple[0]
        sample_weight = y_tuple[1] if len(y_tuple) >= 2 else None
    else:
        y_true        = y_tuple
        sample_weight = None

    if isinstance(y_true, (tuple, list)):
        y_true = y_true[0]

    log.info(
        f"  predictions.y: y_true.shape={tuple(y_true.shape)}, "
        f"sample_weight={'None' if sample_weight is None else tuple(sample_weight.shape)}"
    )

    raw_output = predictions.output
    if isinstance(raw_output, torch.Tensor):
        output = raw_output
    elif hasattr(raw_output, "prediction"):
        output = raw_output.prediction
    else:
        for field in raw_output._fields:
            candidate = getattr(raw_output, field)
            if isinstance(candidate, torch.Tensor) and candidate.dim() == 3:
                output = candidate
                break
        else:
            raise ValueError(
                f"Cannot find 3-D output tensor in predictions.output. "
                f"Fields: {raw_output._fields}"
            )

    output = output.cpu()
    y_true = y_true.cpu()

    assert output.shape[0] == y_true.shape[0]

    # [AUDIT-EVAL-1] No manual inverse transform here.  `mode="quantiles"`
    # returns de-normalized predictions, and TimeSeriesDataSet returns the
    # unscaled continuous target in y.  The previous heuristic could double-
    # transform very poor predictions simply because output_max < 20 mg/dL.
    output_mgdl = output
    y_true_mgdl = y_true
    it_strategy = "native_real_space"

    if not torch.isfinite(output_mgdl).all():
        n_bad = int((~torch.isfinite(output_mgdl)).sum().item())
        log.warning(f"  Replacing {n_bad} non-finite values in output_mgdl")
        output_mgdl = torch.nan_to_num(output_mgdl, nan=0.0, posinf=400.0, neginf=40.0)

    if not torch.isfinite(y_true_mgdl).all():
        log.warning(
            "  y_true contains non-finite values; affected entries will be "
            "excluded by the metric masks."
        )

    log.info("  Evaluation tensors are already in native mg/dL target space ✓")

    try:
        median_idx = QUANTILES.index(0.5)
    except ValueError as exc:
        raise ValueError(f"0.5 not found in QUANTILES={QUANTILES}.") from exc

    y_pred_mgdl = (
        output_mgdl[:, :, median_idx] if output_mgdl.dim() == 3 else output_mgdl
    )
    y_pred_mgdl = torch.nan_to_num(y_pred_mgdl, nan=0.0, posinf=400.0, neginf=40.0)

    # ── Multi-horizon metrics ─────────────────────────────────────────────
    horizon_steps = {
        5:  0,
        15: 2,
        30: 5,
        45: 8,
        60: MAX_PREDICTION_LENGTH - 1,
    }

    metrics: dict[str, float] = {
        "eval_scope":           0.0,
        "inverse_transform_ok": 1.0,
    }

    log.info("  ── Validation Metrics by Horizon (mg/dL) ─────────────────")
    for horizon_min, step_idx in horizon_steps.items():
        y_pred_h = y_pred_mgdl[:, step_idx]
        y_true_h = y_true_mgdl[:, step_idx]

        valid_mask = (
            ~torch.isnan(y_true_h)
            & ~torch.isnan(y_pred_h)
            & torch.isfinite(y_pred_h)
            & torch.isfinite(y_true_h)
            & (y_true_h > 20.0)
            & (y_true_h < 400.0)
            & (y_pred_h > 0.0)
        )

        y_pred_v = y_pred_h[valid_mask]
        y_true_v = y_true_h[valid_mask]

        if len(y_pred_v) == 0:
            log.warning(f"  t+{horizon_min}: no valid samples after filtering.")
            continue

        mae  = (y_pred_v - y_true_v).abs().mean().item()
        rmse = torch.sqrt(((y_pred_v - y_true_v) ** 2).mean()).item()
        mard = (
            (y_pred_v - y_true_v).abs()
            / y_true_v.abs().clamp(min=1.0)
        ).mean().item() * 100.0

        metrics[f"val_mae_{horizon_min}m_mg_dl"]  = round(mae,  4)
        metrics[f"val_rmse_{horizon_min}m_mg_dl"] = round(rmse, 4)
        metrics[f"val_mard_{horizon_min}m_pct"]   = round(mard, 4)

        flag = "✓" if mard < 10.0 else ("~" if mard < 15.0 else "✗")
        log.info(
            f"    t+{horizon_min:2d} min | MAE={mae:6.2f} | "
            f"RMSE={rmse:6.2f} | MARD={mard:5.2f}%  {flag}"
        )

    step_60  = horizon_steps[60]
    valid_60 = (
        ~torch.isnan(y_true_mgdl[:, step_60])
        & ~torch.isnan(y_pred_mgdl[:, step_60])
        & torch.isfinite(y_pred_mgdl[:, step_60])
        & torch.isfinite(y_true_mgdl[:, step_60])
        & (y_true_mgdl[:, step_60] > 20.0)
        & (y_true_mgdl[:, step_60] < 400.0)
        & (y_pred_mgdl[:, step_60] > 0.0)
    )
    y_pred_v60 = y_pred_mgdl[:, step_60][valid_60]
    y_true_v60 = y_true_mgdl[:, step_60][valid_60]
    n_valid    = int(valid_60.sum().item())

    if len(y_pred_v60) > 0:
        mard_60 = metrics.get("val_mard_60m_pct", 0.0)
        if mard_60 < 10.0:
            log.info("    ✓ t+60 MARD < 10% — clinical accuracy target MET")
        elif mard_60 < 15.0:
            log.warning(f"    ~ t+60 MARD {mard_60:.1f}% — approaching 15% target")
        else:
            log.warning(f"    ✗ t+60 MARD {mard_60:.1f}% — above 10% target")

        metrics["n_valid_samples"] = float(n_valid)

        quantile_coverage: dict[str, float] = {}
        if output_mgdl.dim() == 3:
            for i, q_val in enumerate(QUANTILES):
                q_pred   = output_mgdl[:, step_60, i][valid_60]
                coverage = (y_true_v60 <= q_pred).float().mean().item()
                quantile_coverage[f"coverage_q{int(q_val * 100):02d}"] = round(coverage, 4)
        metrics.update(quantile_coverage)

        ceg = clarke_error_grid(y_true_v60, y_pred_v60)
        log.info("  ── Clarke Error Grid (t+60 min, geometric) ────────────────")
        for zone, pct in ceg.items():
            flag = "✓" if zone in ("A", "B") else ("⚠" if zone == "C" else "✗")
            log.info(f"    Zone {zone}: {pct:5.1f}%  {flag}")
        ab_pct = ceg["A"] + ceg["B"]
        if ab_pct >= 99.0:
            log.info(f"  ✓ Zone A+B = {ab_pct:.1f}% — clinically SAFE (≥99%)")
        elif ab_pct >= 95.0:
            log.warning(f"  ~ Zone A+B = {ab_pct:.1f}% — acceptable but below 99%")
        else:
            log.warning(f"  ✗ Zone A+B = {ab_pct:.1f}% — {100.0 - ab_pct:.1f}% in C/D/E")
        metrics.update({f"clarke_zone_{z}_pct": v for z, v in ceg.items()})

        if quantile_coverage:
            log.info("  ── Quantile Calibration ──────────────────────────────────")
            for q_name, coverage in quantile_coverage.items():
                q_val = float(q_name.split("q")[1]) / 100.0
                delta = abs(coverage - q_val)
                flag  = "✓" if delta < 0.05 else "⚠"
                log.info(
                    f"    Q{q_val:.2f}: expected={q_val:.3f}  "
                    f"observed={coverage:.3f}  (Δ={delta:.3f}) {flag}"
                )

    # ── [LEAK-E] Persistence + [AUDIT-MED-NEW-1] ARIMA baselines ─────────
    log.info("  ── Persistence & ARIMA Baselines (t+60) ───────────────────")
    persistence_approximate = False
    last_encoder_glucose     = None
    persistence_sample_mask  = None
    encoder_windows: dict[int, np.ndarray] = {}  # sample_idx → encoder glucose array

    try:
        dec_time_idx = predictions.x.get("decoder_time_idx", None)

        if dec_time_idx is not None:
            # Use the public TimeSeriesDataSet API to decode the sample index.
            # Manual decoding of x["groups"] through categorical_encoders is brittle:
            # PyTorch Forecasting internally stores group IDs under synthetic
            # __group_id__* encoders and those internals have changed across releases.
            # x_to_index() returns the original group labels plus the first decoder
            # time_idx in exactly the same sample order as predictions.x.
            sample_index = validation.x_to_index(predictions.x)
            if SUBJECT_COL not in sample_index.columns or "time_idx" not in sample_index.columns:
                raise RuntimeError(
                    "[AUDIT] TimeSeriesDataSet.x_to_index() did not return the expected "
                    f"columns ({SUBJECT_COL!r}, 'time_idx'). Got: {list(sample_index.columns)}"
                )
            if len(sample_index) != len(dec_time_idx):
                raise RuntimeError(
                    "[AUDIT] Prediction/index length mismatch: "
                    f"x_to_index={len(sample_index)} vs decoder_time_idx={len(dec_time_idx)}."
                )

            subject_ids = sample_index[SUBJECT_COL].astype(str).to_numpy()
            first_dec   = sample_index["time_idx"].astype(int).to_numpy()
            last_enc_ti = first_dec - 1

            # Cross-check the public decoded index against the tensor returned by
            # the dataloader. A mismatch would make persistence/ARIMA comparisons
            # invalid, so fail loudly rather than silently score the wrong rows.
            first_dec_tensor = dec_time_idx[:, 0].detach().cpu().numpy().astype(int)
            if not np.array_equal(first_dec, first_dec_tensor):
                raise RuntimeError(
                    "[AUDIT] x_to_index time_idx does not match decoder_time_idx[:, 0]. "
                    "Refusing to compute persistence/ARIMA baselines."
                )

            _verify_time_idx_alignment(
                val_df,
                dec_time_idx.cpu().numpy(),
                subject_ids,
            )

            val_lookup = (
                val_df.set_index([SUBJECT_COL, "time_idx"])["glucose_mg_dl"]
            )

            # [FIX-ISSUE-13] Build ARIMA encoder windows in raw mg/dL using val_lookup.
            # The previous approach extracted encoder_cont[:, :, target_feat_idx], which
            # is the log-robust NORMALISED representation of glucose. ARIMA forecast
            # values therefore lived in normalised space (~[-2, 2]) while y_true_mgdl
            # was in mg/dL (~70-400), making arima_mard meaningless and
            # improvement_over_arima_pp inflated by orders of magnitude.
            #
            # Fix: for each sample, recover the actual encoder time indices from
            # dec_time_idx and encoder_lengths, then look up glucose_mg_dl from
            # val_lookup (subject_id, time_idx) → mg/dL.  encoder_lengths tells us
            # the true (non-padded) window length, so we never feed leading-zero
            # padding artefacts into the ARIMA model.
            #
            # Samples with > 30% NaN coverage in their mg/dL window are excluded
            # (ARIMA is unreliable on heavily gapped series); forward-fill handles
            # short gaps up to 15 min before handing the array to ARIMA.
            encoder_lengths_raw = predictions.x.get("encoder_lengths", None)
            if encoder_lengths_raw is not None and dec_time_idx is not None:
                enc_lens_np = encoder_lengths_raw.cpu().numpy().astype(int)
                for i in range(len(subject_ids)):
                    sid      = str(subject_ids[i])
                    last_ti  = int(last_enc_ti[i])
                    enc_len  = int(enc_lens_np[i])
                    start_ti = last_ti - enc_len + 1
                    window   = np.array([
                        float(val_lookup[(sid, int(ti))])
                        if (sid, int(ti)) in val_lookup.index
                        else np.nan
                        for ti in range(start_ti, last_ti + 1)
                    ], dtype=float)
                    nan_frac = float(np.isnan(window).mean())
                    if nan_frac > 0.30:
                        continue  # too many gaps; skip this sample for ARIMA
                    # Short-gap forward-fill (≤3 steps = 15 min) — causal
                    window_series = pd.Series(window).ffill(limit=3)
                    if window_series.isna().any():
                        # Remaining NaNs at start: fill with first valid value
                        first_valid = float(window_series.dropna().iloc[0]) if window_series.notna().any() else np.nan
                        if np.isnan(first_valid):
                            continue
                        window_series = window_series.fillna(first_valid)
                    encoder_windows[i] = window_series.to_numpy(dtype=float)
            else:
                log.warning(
                    "  [FIX-ISSUE-13] encoder_lengths not available in predictions.x; "
                    "ARIMA baseline will be skipped (cannot reconstruct mg/dL windows)."
                )

            last_glucose_list  = []
            lookup_found_flags = []
            n_miss = 0
            for sid, ti in zip(subject_ids, last_enc_ti):
                key = (str(sid), int(ti))
                if key in val_lookup.index:
                    last_glucose_list.append(val_lookup[key])
                    lookup_found_flags.append(True)
                else:
                    last_glucose_list.append(float("nan"))
                    lookup_found_flags.append(False)
                    n_miss += 1

            if n_miss > 0:
                log.warning(
                    f"  [LEAK-E] {n_miss}/{len(last_glucose_list)} samples "
                    f"excluded from persistence comparison (no lookup)."
                )
                persistence_approximate = True

            last_encoder_glucose  = torch.tensor(last_glucose_list,  dtype=torch.float32)
            lookup_found_flags_t  = torch.tensor(lookup_found_flags, dtype=torch.bool)
            persistence_sample_mask = valid_60 & lookup_found_flags_t

        else:
            log.warning(
                "  [LEAK-E / ZT-CRIT-1] decoder_time_idx absent. "
                "Persistence/ARIMA metrics omitted."
            )
            persistence_approximate = True

    except RuntimeError:
        raise
    except Exception as e:
        log.warning(f"  Persistence/ARIMA setup failed ({e}). Baselines omitted.")
        persistence_approximate = True

    if (
        last_encoder_glucose is not None
        and persistence_sample_mask is not None
        and len(y_pred_v60) > 0
    ):
        n_clean   = int(persistence_sample_mask.sum().item())
        n_valid60 = int(valid_60.sum().item())
        clean_pct = n_clean / max(n_valid60, 1)

        log.info(
            f"  [LEAK-E] Clean persistence lookups: "
            f"{n_clean}/{n_valid60} ({clean_pct:.1%})"
        )

        if clean_pct < 0.50:
            log.warning(
                f"  [LEAK-E] < 50% clean lookups ({clean_pct:.1%}). "
                "Baselines omitted."
            )
            metrics["persistence_is_approximate"] = 1.0
        else:
            pers_clean  = last_encoder_glucose[persistence_sample_mask]
            ytrue_clean = y_true_mgdl[:, step_60][persistence_sample_mask]
            ypred_clean = y_pred_mgdl[:, step_60][persistence_sample_mask]

            persistence_mae  = (pers_clean - ytrue_clean).abs().mean().item()
            persistence_mard = (
                (pers_clean - ytrue_clean).abs()
                / ytrue_clean.abs().clamp(min=1.0)
            ).mean().item() * 100.0

            approx_flag = " (some excluded)" if persistence_approximate else ""
            log.info(f"    Persistence MAE  : {persistence_mae:.2f} mg/dL{approx_flag}")
            log.info(f"    Persistence MARD : {persistence_mard:.2f}%{approx_flag}")

            model_mard_clean = (
                (ypred_clean - ytrue_clean).abs()
                / ytrue_clean.abs().clamp(min=1.0)
            ).mean().item() * 100.0
            improvement_over_persistence = persistence_mard - model_mard_clean
            log.info(f"    Model MARD (same subset): {model_mard_clean:.2f}%")
            log.info(
                f"    Improvement over persistence: "
                f"{improvement_over_persistence:+.2f}pp"
            )

            if improvement_over_persistence <= 0:
                log.warning(
                    "    ⚠ Model does NOT beat persistence at t+60. [ZT-MED-4]"
                )

            metrics["persistence_mard_60m_pct"]       = round(persistence_mard,  4)
            metrics["persistence_mae_60m_mg_dl"]      = round(persistence_mae,   4)
            metrics["persistence_is_approximate"]     = float(persistence_approximate)
            metrics["n_persistence_lookups_used"]     = float(n_clean)
            metrics["model_mard_60m_persistence_subset"] = round(model_mard_clean, 4)

            # ── [AUDIT-MED-NEW-1] ARIMA(1,1,0) baseline ──────────────────
            if _STATSMODELS_AVAILABLE and encoder_windows:
                clean_indices   = persistence_sample_mask.nonzero(as_tuple=True)[0].tolist()
                arima_preds     = []
                arima_trues     = []
                # [FIX-ISSUE-5] Track the exact batch indices that contributed
                # to arima_preds so that the model-vs-ARIMA comparison is
                # computed on the SAME set of samples as the ARIMA metrics.
                # The old code used `i < len(arima_preds)` (a positional count)
                # which included indices where ARIMA had failed, producing a
                # model MARD over a different (larger) population than ARIMA.
                arima_batch_indices = []
                n_arima_ok      = 0
                n_arima_fail    = 0

                for idx in clean_indices:
                    if idx not in encoder_windows:
                        n_arima_fail += 1
                        continue
                    enc_glucose = encoder_windows[idx]
                    forecast    = _compute_arima_baseline(enc_glucose, MAX_PREDICTION_LENGTH)
                    if forecast is None or not np.all(np.isfinite(forecast)):
                        n_arima_fail += 1
                        continue
                    pred_60 = forecast[step_60]
                    true_60 = float(y_true_mgdl[idx, step_60].item())
                    if not (20.0 < true_60 < 400.0):
                        # Sample excluded from ARIMA metrics; do NOT add to
                        # arima_batch_indices either (keeps sets identical).
                        continue
                    arima_preds.append(pred_60)
                    arima_trues.append(true_60)
                    arima_batch_indices.append(idx)   # parallel to arima_preds
                    n_arima_ok += 1

                if n_arima_ok >= 10:
                    ap = np.array(arima_preds, dtype=float)
                    at = np.array(arima_trues, dtype=float)
                    arima_mard = float(
                        np.mean(np.abs(ap - at) / np.clip(np.abs(at), 1.0, None)) * 100.0
                    )
                    arima_mae  = float(np.mean(np.abs(ap - at)))
                    log.info(
                        f"    ARIMA(1,1,0) MAE  : {arima_mae:.2f} mg/dL "
                        f"(n={n_arima_ok}, {n_arima_fail} failed)"
                    )
                    log.info(f"    ARIMA(1,1,0) MARD : {arima_mard:.2f}%")

                    # Model MARD on the exact same samples as ARIMA.
                    # [FIX-ISSUE-5] arima_batch_indices is in 1-to-1
                    # correspondence with arima_preds — no positional mismatch.
                    arima_clean_idx_t = torch.tensor(
                        arima_batch_indices, dtype=torch.long,
                    )
                    if len(arima_clean_idx_t) > 0:
                        model_preds_arima_subset = y_pred_mgdl[:, step_60][arima_clean_idx_t]
                        true_arima_subset        = y_true_mgdl[:, step_60][arima_clean_idx_t]
                        model_mard_arima = (
                            (model_preds_arima_subset - true_arima_subset).abs()
                            / true_arima_subset.abs().clamp(min=1.0)
                        ).mean().item() * 100.0
                        improvement_over_arima = arima_mard - model_mard_arima
                        log.info(
                            f"    Model MARD (ARIMA subset): {model_mard_arima:.2f}%"
                        )
                        log.info(
                            f"    Improvement over ARIMA: {improvement_over_arima:+.2f}pp"
                        )
                        if improvement_over_arima <= 0:
                            log.warning(
                                "    ⚠ Model does NOT beat ARIMA(1,1,0) at t+60. "
                                "[AUDIT-MED-NEW-1]"
                            )
                        metrics["arima_mard_60m_pct"]        = round(arima_mard, 4)
                        metrics["arima_mae_60m_mg_dl"]       = round(arima_mae,  4)
                        metrics["n_arima_samples"]           = float(n_arima_ok)
                        metrics["model_mard_arima_subset"]   = round(model_mard_arima, 4)
                        metrics["improvement_over_arima_pp"] = round(
                            improvement_over_arima, 4
                        )
                else:
                    log.warning(
                        f"  [AUDIT-MED-NEW-1] Only {n_arima_ok} successful ARIMA fits "
                        f"({n_arima_fail} failed). ARIMA metrics omitted."
                    )
            elif not _STATSMODELS_AVAILABLE:
                log.warning(
                    "  [AUDIT-MED-NEW-1] statsmodels not available — "
                    "ARIMA baseline skipped. Install with: pip install statsmodels"
                )

    metrics["eval_scope"] = 0.0
    log.info(
        "  [ZT-MED-4] eval_scope=within_subject_temporal. "
        "For population generalisation, re-evaluate on held-out patients."
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if args.horizon < MAX_PREDICTION_LENGTH:
        raise ValueError(
            f"--horizon must be at least {MAX_PREDICTION_LENGTH} steps "
            f"({MAX_PREDICTION_LENGTH * 5} min), because evaluation computes "
            "t+60 metrics and baselines."
        )
    if args.context < 2:
        raise ValueError("--context must be at least 2 steps.")
    if args.hidden_size % args.attention_heads != 0:
        raise ValueError(
            "--hidden-size must be divisible by --attention-heads "
            f"(got {args.hidden_size} and {args.attention_heads})."
        )

    pl.seed_everything(args.seed, workers=True)

    log.info("=" * 65)
    log.info("NMD — TFT Population Model Training (Phase 4 v19 zero-trust fixed)")
    log.info(f"  pytorch-forecasting : {_ptf.__version__}")
    log.info(f"  Seed                : {args.seed}")
    log.info(f"  Horizon             : {args.horizon} steps ({args.horizon * 5} min)")
    log.info(f"  Context             : {args.context} steps ({args.context * 5} min)")
    log.info(f"  LR (peak)           : {args.lr}")
    log.info(f"  Hidden              : {args.hidden_size} (heads: {args.attention_heads})")
    log.info(f"  LSTM layers         : {args.lstm_layers}")
    log.info(f"  Dropout             : {args.dropout}")
    log.info(f"  Gradient clip       : {args.gradient_clip}")
    log.info(f"  Train/val gap       : {TRAIN_VAL_GAP} steps [LEAK-B]")
    log.info(
        f"  ARIMA baseline      : "
        f"{'enabled' if _STATSMODELS_AVAILABLE else 'disabled (pip install statsmodels)'}"
    )
    log.info(
        f"  Fixes applied       : "
        f"FIX-NADIR-ASYM (glucose_nadir_proximity + glucose_asymmetry_index sentineled → post-split recompute), "
        f"FIX-TDD-SENTINEL (tdd_rolling_7d + bolus_fraction_of_tdd parquet sentineled), "
        f"FIX-NORM-GROUPNORM (EncoderNormalizer → GroupNormalizer, train-fitted and frozen), "
        f"FIX-ISSUE-19 (glucose_lag_* registered in TIME_VARYING_UNKNOWN_REALS — were orphaned), "
        f"FIX-ISSUE-18 (hr_recovery_slope warmstart forwarded — resting-HR consistency), "
        f"FIX-ISSUE-17 (test warm-start from training XML — asymmetric cold-start eliminated), "
        f"FIX-ISSUE-16 (z-score clip [-10,10] in composite/autonomic_stress_index — numerical bomb fixed), "
        f"FIX-ISSUE-14 (GradientNormLogger consecutive counter reset — explosion detection restored), "
        f"FIX-ISSUE-15 (val warm-start: tdd/cbp/ic_ratio/hr_resting cold-start bias eliminated), "
        f"FIX-ISSUE-11 (autonomic_stress_index sentineled → post-split recompute), "
        f"FIX-ISSUE-12 (step quantile thresholds → causal rolling quantile), "
        f"FIX-ISSUE-13 (ARIMA encoder windows: normalised → mg/dL via val_lookup), "
        f"FIX-ISSUE-4 (bfill removed from hr_resting_causal → causal ffill), "
        f"FIX-ISSUE-5 (ARIMA index uses parallel arima_batch_indices), "
        f"FIX-ISSUE-7 (ic_ratio warm-up fill = 0.0 not first_real), "
        f"+ all v10 fixes: ZT-CRIT-NEW-1/2, AUDIT-HIGH-NEW-1, AUDIT-MED-NEW-1, "
        f"+ all v9 fixes: LEAK-A/B/C/D/E, AUDIT-CRIT-1/2, AUDIT-HIGH-2/3, "
        f"AUDIT-MED-2, ZT-CRIT-1/3, ZT-HIGH-3/5, ZT-MED-2/3/4, ZT-INFO-2"
    )
    log.info("=" * 65)

    ckpt_path = None if args.no_resume else find_best_checkpoint()

    train_df, val_df    = load_and_preprocess_data()
    training_ds, val_ds = build_datasets(train_df, val_df, args)
    model               = build_model(training_ds, args, ckpt_path)
    trainer             = train(model, training_ds, val_ds, args, ckpt_path)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    if best_ckpt:
        log.info(f"Loading best model for evaluation: {best_ckpt}")
        model = ClinicalTFT.load_from_checkpoint(
            best_ckpt,
            map_location="cpu",
            loss=ClinicalQuantileLoss(quantiles=QUANTILES),
        )
    else:
        log.warning("No best checkpoint found — evaluating last model state")

    metrics = evaluate(model, val_ds, args, val_df=val_df)

    # ── Out-of-sample test evaluation ─────────────────────────────────────
    log.info("=" * 65)
    log.info("Evaluating OUT-OF-SAMPLE on source_split='test'...")

    combined_max_time_idx = max(
        int(train_df["time_idx"].max()),
        int(val_df["time_idx"].max()),
    )

    try:
        test_df = load_test_data(train_max_time_idx=combined_max_time_idx)
        test_ds = create_time_series_dataset(
            test_df,
            reference_dataset=training_ds,
            predict_mode=False,
        )
        log.info(f"  Test windows: {len(test_ds):,}")
        test_metrics = evaluate(model, test_ds, args, val_df=test_df)
        log.info("── Test Metrics (out-of-sample, same patients) ──────────")
        log.info(f"  MARD t+60 : {test_metrics.get('val_mard_60m_pct', 'N/A')}%")
        log.info(f"  MARD t+30 : {test_metrics.get('val_mard_30m_pct', 'N/A')}%")
        log.info(f"  MAE  t+60 : {test_metrics.get('val_mae_60m_mg_dl', 'N/A')} mg/dL")
        log.info(
            f"  Zone A+B  : "
            f"{test_metrics.get('clarke_zone_A_pct', 0) + test_metrics.get('clarke_zone_B_pct', 0):.1f}%"
        )
        if not test_metrics.get("persistence_is_approximate", 1.0):
            log.info(
                f"  Persistence MARD : {test_metrics.get('persistence_mard_60m_pct', 'N/A')}%"
            )
        if "arima_mard_60m_pct" in test_metrics:
            log.info(
                f"  ARIMA MARD t+60  : {test_metrics.get('arima_mard_60m_pct', 'N/A')}%"
            )
            log.info(
                f"  vs ARIMA (pp)    : {test_metrics.get('improvement_over_arima_pp', 'N/A')}"
            )
    except Exception as e:
        log.error(f"  Test evaluation failed: {e}")
    log.info("=" * 65)

    zone_a = metrics.get("clarke_zone_A_pct", 0.0)
    zone_b = metrics.get("clarke_zone_B_pct", 0.0)

    log.info("=" * 65)
    log.info("Training complete.")
    log.info(f"  Best checkpoint  : {best_ckpt or 'N/A'}")
    log.info(f"  MARD t+60 (val)  : {metrics.get('val_mard_60m_pct', 'N/A')}%")
    log.info(f"  MARD t+30 (val)  : {metrics.get('val_mard_30m_pct', 'N/A')}%")
    log.info(f"  MARD t+15 (val)  : {metrics.get('val_mard_15m_pct', 'N/A')}%")
    log.info(f"  MAE  t+60 (val)  : {metrics.get('val_mae_60m_mg_dl', 'N/A')} mg/dL")
    log.info(f"  Zone A+B         : {zone_a + zone_b:.1f}%")
    log.info(f"  Eval scope       : within-subject temporal [ZT-MED-4]")
    if not metrics.get("persistence_is_approximate", 1.0):
        log.info(f"  Persistence MARD : {metrics.get('persistence_mard_60m_pct', 'N/A')}%")
        log.info(f"  n lookups used   : {metrics.get('n_persistence_lookups_used', 'N/A')}")
    if "arima_mard_60m_pct" in metrics:
        log.info(f"  ARIMA MARD t+60  : {metrics.get('arima_mard_60m_pct', 'N/A')}%")
        log.info(
            f"  vs ARIMA (pp)    : {metrics.get('improvement_over_arima_pp', 'N/A')}"
        )
    log.info("=" * 65)


if __name__ == "__main__":
    main()