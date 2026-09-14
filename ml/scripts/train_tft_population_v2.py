"""NMD Baseline v1.0 Stage A remediation.

See docs/BASELINE_AUDIT_STAGE_A_REMEDIATION.md for the approved protocol.
Historical audit findings are preserved in docs/BASELINE_AUDIT_STAGE_A.md.
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

if __package__ in (None, ""):
    from checkpoint_registry import contract as checkpoint_contract, sha256 as artifact_sha256
    from baseline_training import registry_for, load_model as load_registered_model, train_baseline
    from finite_training import FiniteMetric, FiniteModel, require_finite, gradient_norm, CheckedAdamW
    from temporal_protocol import regular_timeline, TARGET_OBSERVED, validate_observation_indicator
    from observed_windows import validate_dense_frame, filter_observed_windows, assert_observed_evaluation
else:
    from .checkpoint_registry import contract as checkpoint_contract, sha256 as artifact_sha256
    from .baseline_training import registry_for, load_model as load_registered_model, train_baseline
    from .finite_training import FiniteMetric, FiniteModel, require_finite, gradient_norm, CheckedAdamW
    from .temporal_protocol import regular_timeline, TARGET_OBSERVED, validate_observation_indicator
    from .observed_windows import validate_dense_frame, filter_observed_windows, assert_observed_evaluation

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
MODEL_DIR = ROOT / "ml" / "models" / "baseline_v1_stage_a"
BASELINE_REGISTRY = ROOT / "ml" / "models" / "baseline_v1_stage_b" / "runs"
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
    "bolus_event_prior",
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
    "bolus_event_prior",
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
def _compute_bolus_event_prior(df: pd.DataFrame) -> pd.Series:
    df = regular_timeline(df)
    return (df["bolus_event"] > 0).astype(float).rolling(288, min_periods=1).mean()


def _compute_bolus_event_prior_with_history(
    current: pd.DataFrame,
    history: "pd.DataFrame | None" = None,
) -> pd.Series:
    """Compute the causal prior with exact preceding history when contiguous."""
    cur = current.sort_values("timestamp").reset_index(drop=True)
    if history is None or history.empty or cur.empty:
        return _compute_bolus_event_prior(cur)

    hist = history.sort_values("timestamp").reset_index(drop=True)
    gap = pd.Timestamp(cur["timestamp"].iloc[0]) - pd.Timestamp(hist["timestamp"].iloc[-1])
    if gap <= pd.Timedelta(0):
        raise RuntimeError(
            f"[AUDIT-CBP] History overlaps/follows current split (gap={gap})."
        )
    if gap > pd.Timedelta("10min"):
        log.warning(
            f"  [AUDIT-CBP] Boundary gap {gap} is >10 min; "
            "using current-split-only bolus_event_prior."
        )
        return _compute_bolus_event_prior(cur)

    hist_tail = hist.tail(288)
    hist_tail = hist_tail.copy()
    if "source_split" in cur:
        hist_tail["source_split"] = cur["source_split"].iloc[0]
    combined = pd.concat([hist_tail, cur], ignore_index=True, sort=False)
    values = _compute_bolus_event_prior(combined)
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
    """Trailing HR statistic; before first observation, missing or past state.

    A02: first observation fallback is available only at/after its timestamp.
    A valid supplied historical state may initialize an otherwise missing prefix.
    """
    resting = hr.rolling(96, min_periods=24).quantile(0.10).ffill(limit=96)
    available_hr = hr.ffill(limit=96)
    if warmstart_value is not None and np.isfinite(warmstart_value):
        available_hr = available_hr.fillna(float(warmstart_value))
    return resting.fillna(available_hr)


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

    has_hr   = "heart_rate" in df.columns
    has_gsr  = "gsr" in df.columns
    has_temp = "skin_temperature" in df.columns
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
        df["hr_resting_estimate"] = np.nan
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
        available = df[["heart_rate", "gsr", "skin_temperature"]].notna().all(axis=1)
        df["autonomic_stress_index"] = (
            (hr_zscore + gsr_zscore - temp_zscore) / 3.0
        ).where(available, df["composite_stress_index"]).fillna(0.0)
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
    if "bolus_dose" in train_grp.columns and "basal_rate" in train_grp.columns:
        bolus_sum  = train_grp["bolus_dose"].rolling(WIN_TDD_LOCAL, min_periods=1).sum()
        basal_mean = train_grp["basal_rate"].rolling(WIN_TDD_LOCAL, min_periods=1).mean()
        # Mean total daily dose (U/day) over the trailing 7-day window:
        # (7-day bolus total + estimated 7-day basal total) / 7.
        tdd        = (bolus_sum + basal_mean * 24.0 * 7.0) / 7.0
        ws["tdd_last"] = float(tdd.iloc[-1]) if len(tdd) else 20.0

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
    bolus_event_prior, ic_ratio_deviation) are systematically biased low.
    """
    df = regular_timeline(df)
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
            if "source_split" in df:
                hist["source_split"] = df["source_split"].iloc[0]
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
    if "bolus_dose" in df.columns and "basal_rate" in df.columns:
        WIN_TDD_LOCAL = 7 * 24 * 12
        bolus_rolling = df["bolus_dose"].rolling(WIN_TDD_LOCAL, min_periods=1).sum()
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
    for lag in [1, 2, 3, 6, 12, 24]:
        lagged = df["glucose_mg_dl"].shift(lag)
        df[f"glucose_lag_{lag}_available"] = lagged.notna().astype(float)
        df[f"glucose_lag_{lag}"] = lagged.fillna(0.)

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
    if "bolus_dose" in df.columns and "basal_rate" in df.columns:
        basal_24h = (
            df["basal_rate"].rolling(24 * 12, min_periods=12).mean() * 24.0
        )
        bolus_24h = df["bolus_dose"].rolling(24 * 12, min_periods=12).sum()
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
class ClinicalQuantileLoss(FiniteMetric, QuantileLoss):
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

        require_finite(target_mgdl, "target")
        require_finite(y_pred, "prediction")
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

        result = base_loss * weights
        require_finite(result, "clinical per-position loss")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL TFT
# ─────────────────────────────────────────────────────────────────────────────
class ClinicalTFT(FiniteModel, TemporalFusionTransformer):
    COSINE_EPOCHS: int = 15

    def configure_optimizers(self):
        if self.trainer.max_epochs == 0:
            raise ValueError("Training epoch budget must be positive")
        optimizer = CheckedAdamW(
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
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Population TFT for glucose forecasting"
    )
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
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
    p.add_argument("--no-swa", action="store_true", default=True)
    p.add_argument("--swa", dest="no_swa", action="store_false", help="Outside certified baseline; rejected")
    p.add_argument("--mode", choices=["fresh", "resume-last", "inference-best", "weights-only"], default="fresh")
    p.add_argument("--run-id")
    p.add_argument("--parent-run-id")
    p.add_argument("--registry-dir", type=Path, default=BASELINE_REGISTRY)
    p.add_argument("--accumulate-grad-batches", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline_v1_stage_c.json")
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline_v1_stage_c.json")
    selected, _ = selector.parse_known_args(argv)
    import json
    configured = json.loads(selected.config.read_text())
    required = {
        "protocol": "nmd-baseline-v1.0-stage-c-1",
        "normalizer_revision": "observed-train-encoded-subject-v2",
        "evaluation_revision": "nmd-evaluation-stage-c-1",
        "data_protocol": "baseline_v1_stage_a",
        "canonical_stage_a_sha256": "d14fe31b1b81971639ca8d5ba17b4882fbd6711569e30785f0da7b91a0c031cf",
        "precision": "32-true", "swa": False, "mid_epoch_resume": False,
        "resume_boundary": "completed_training_batches_and_validation",
        "automatic_test_evaluation": False,
        "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 20, "min_delta": 0.0001},
        "clinical_loss": {"factor": 2, "hypo_threshold_mg_dl": 70, "hypo_weight": 2.5},
    }
    if __package__ in (None, ""):
        from numerical_profile import profile
    else:
        from .numerical_profile import profile
    required["numerical_profiles"] = {device: profile(device) for device in ("cpu", "cuda")}
    for key, value in required.items():
        if configured.get(key) != value:
            raise ValueError(f"Unsupported certified baseline configuration: {key}")
    defaults = {key: configured[key] for key in (
        "epochs", "batch_size", "context", "horizon", "lr", "hidden_size", "attention_heads",
        "dropout", "hidden_continuous_size", "lstm_layers", "gradient_clip", "num_workers", "seed",
        "accumulate_grad_batches", "mode")}
    defaults["data_dir"] = ROOT / configured["data_dir"]
    defaults["no_swa"] = not configured["swa"]
    p.set_defaults(**defaults)
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPER
# ─────────────────────────────────────────────────────────────────────────────
def find_best_checkpoint(prefix="tft-stage-a-v1", *, registry=None, run_id=None, expected=None):
    """BEST is an explicit owned artifact; discovery by filename/mtime is disabled."""
    if registry is None or run_id is None or expected is None:
        raise ValueError("Automatic filename checkpoint discovery is disabled; specify a registered BEST run")
    return str(registry.verified(run_id, "best", expected)[0])


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
    validate_observation_indicator(df)
    n_before = len(df)
    df = df[df["source_split"] == "train"].copy()
    log.info(f"  Filtered to source_split='train': {len(df):,} / {n_before:,} rows")
    df[SUBJECT_COL] = df[SUBJECT_COL].astype(str)

    # [LEAK-1] Guard
    if "bolus_event_prior" in df.columns:
        raise RuntimeError(
            "[LEAK-1] 'bolus_event_prior' found in training.parquet. "
            "Delete training.parquet and rerun preprocess_ohiot1dm.py."
        )
    log.info("  [LEAK-1 guard] bolus_event_prior absent from parquet ✓")

    # Pre-split fillna — skip sentinel columns
    for col in TIME_VARYING_UNKNOWN_REALS:
        if col in PRE_SPLIT_FILLNA_SKIP or col == TARGET_COL:
            continue
        if col in df.columns and df[col].dtype in (float, "float32", "float64"):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    POST_SPLIT_COMPUTED = {
        "bolus_event_prior",
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
        + ["bolus_event", "bolus_dose", TARGET_OBSERVED]
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
        n = len(pdf)
        eligible = pdf.loc[pdf[TARGET_COL].notna(), "timestamp"]
        cut = int(len(eligible) * 0.85)
        if cut == 0 or cut == len(eligible):
            raise ValueError("Insufficient eligible CGM for temporal split")
        boundary = eligible.iloc[cut]
        split_at = int((pdf.timestamp < boundary).sum())

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

    # A07: identical event-frequency helper for every split; no bolus type inference.
    train_df = pd.concat([
        grp.assign(bolus_event_prior=_compute_bolus_event_prior(grp).to_numpy())
        for _, grp in train_df.groupby(SUBJECT_COL)
    ], ignore_index=True)
    # Validation starts immediately after each patient's train segment, so use
    # the actual preceding bolus history rather than an approximate scalar blend.
    val_df_parts: list[pd.DataFrame] = []
    for patient in patients:
        tr_grp = train_df[train_df[SUBJECT_COL] == patient].copy()
        va_grp = val_df[val_df[SUBJECT_COL] == patient].copy()
        if va_grp.empty:
            continue
        va_grp = va_grp.sort_values("timestamp").reset_index(drop=True)
        va_grp["bolus_event_prior"] = _compute_bolus_event_prior_with_history(
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
        if col in POST_SPLIT_FINAL_FILLNA_SKIP or col == "hr_resting_estimate":
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
            if col in df_.columns and col != "hr_resting_estimate":
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
    validate_observation_indicator(full_parquet)

    df = full_parquet[full_parquet["source_split"] == "test"].copy()
    log.info(f"  Test set: {len(df):,} rows, {df['subject_id'].nunique()} patients")
    df[SUBJECT_COL] = df[SUBJECT_COL].astype(str)

    if "bolus_event_prior" in df.columns:
        raise RuntimeError(
            "[LEAK-1] 'bolus_event_prior' found in test parquet."
        )

    POST_SPLIT_COMPUTED = {
        "bolus_event_prior", "weekend_meal_prior",
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

    # bolus_event_prior: use exact train-tail history only when the source
    # XMLs are chronologically contiguous; the helper cold-starts across a gap.
    cbp_parts: list[pd.DataFrame] = []
    for pid in sorted(df[SUBJECT_COL].unique()):
        va_grp = df[df[SUBJECT_COL] == pid].copy()
        if va_grp.empty:
            continue
        tr_grp = train_rows[train_rows[SUBJECT_COL] == str(pid)].copy()
        va_grp = va_grp.sort_values("timestamp").reset_index(drop=True)
        va_grp["bolus_event_prior"] = _compute_bolus_event_prior_with_history(
            va_grp, tr_grp if not tr_grp.empty else None
        ).to_numpy()
        cbp_parts.append(va_grp)
    df = pd.concat(cbp_parts, ignore_index=True) if cbp_parts else df

    for col in TIME_VARYING_UNKNOWN_REALS:
        if col == TARGET_COL:
            continue
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
    # A04/A05: PF 1.7 fills skipped time_idx values by repeating targets.
    # Require dense input and disable this path. NaN placeholders are storage
    # only; the supported index filter removes every sequence exposing one.
    df = df.sort_values([GROUP_COL, "time_idx"]).reset_index(drop=True)
    validate_dense_frame(df)
    prepared = df.copy()
    prepared[TARGET_COL] = prepared[TARGET_COL].fillna(1.0)
    unknown = TIME_VARYING_UNKNOWN_REALS + [
        f"glucose_lag_{lag}_available" for lag in [1, 2, 3, 6, 12, 24]]
    for col in unknown:
        if col != TARGET_COL:
            prepared[col] = prepared[col].replace([np.inf, -np.inf], np.nan).fillna(0.)
    if reference_dataset is not None:
        norm = reference_dataset.target_normalizer
        if getattr(norm, "nmd_revision", None) != "observed-train-encoded-subject-v2":
            raise ValueError("Normalizer compatibility: historical semantics cannot be migrated")
        if not set(df[GROUP_COL]).issubset(norm.nmd_subject_mapping):
            raise ValueError("Normalizer compatibility: unseen subject")
        dataset = TimeSeriesDataSet.from_dataset(
            reference_dataset, prepared, predict=predict_mode,
            stop_randomization=True, min_prediction_length=horizon,
            max_prediction_length=horizon, allow_missing_timesteps=False)
    else:
        # Storage placeholders must never enter fitted target statistics.
        observed = df[TARGET_OBSERVED].astype(bool)
        if set(df.loc[observed, GROUP_COL]) != set(df[GROUP_COL]):
            raise ValueError("Normalizer requires observed training targets for every subject")
        normalizer = GroupNormalizer(groups=[GROUP_COL], transformation="log", center=True)
        # C02: PF encodes group categories before target normalization. Use
        # exactly the same fitted mapping for statistics and both PF encoders.
        from pytorch_forecasting.data.encoders import NaNLabelEncoder
        subject_encoder = NaNLabelEncoder().fit(df[GROUP_COL])
        encoded_fit = df.loc[observed].copy()
        encoded_fit[GROUP_COL] = subject_encoder.transform(encoded_fit[GROUP_COL])
        normalizer.fit(df.loc[observed, TARGET_COL], encoded_fit)
        normalizer.nmd_revision = "observed-train-encoded-subject-v2"
        normalizer.nmd_subject_mapping = dict(subject_encoder.classes_)
        dataset = TimeSeriesDataSet(
            prepared, time_idx="time_idx", target=TARGET_COL, group_ids=[GROUP_COL],
            min_encoder_length=context_length // 2, max_encoder_length=context_length,
            min_prediction_length=1, max_prediction_length=horizon,
            static_categoricals=STATIC_CATEGORICALS, static_reals=STATIC_REALS,
            time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
            time_varying_unknown_reals=unknown, target_normalizer=normalizer,
            categorical_encoders={GROUP_COL: subject_encoder,
                                  "__group_id__" + GROUP_COL: subject_encoder},
            add_relative_time_idx=True, add_target_scales=True,
            add_encoder_length=True, allow_missing_timesteps=False)
    dataset = filter_observed_windows(dataset, df)
    log.info("Stage A window eligibility: %s", dataset.stage_a_window_stats)
    return dataset


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
    validation = create_time_series_dataset(
        val_df, reference_dataset=training, horizon=args.horizon)


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
            _val_norm.get_norm(val_df[[GROUP_COL]].assign(**{GROUP_COL: val_df[GROUP_COL].map(_val_norm.nmd_subject_mapping)})), dtype=float
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

    def state_dict(self):
        return {"high_grad_consecutive": self._high_grad_consecutive,
                "clip_val": self.clip_val, "warmup_steps": self.warmup_steps}

    def load_state_dict(self, state):
        if state.get("clip_val") != self.clip_val or state.get("warmup_steps") != self.warmup_steps:
            raise ValueError("GradientNormLogger configuration mismatch")
        self._high_grad_consecutive = int(state["high_grad_consecutive"])

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

        norm_tensor = gradient_norm(pl_module.parameters())
        grad_norm = float(norm_tensor)
        pl_module.log(
            "train/grad_norm_pre_clip",
            norm_tensor,
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

    mode = getattr(args, "mode", "fresh")
    if ckpt_path and mode == "fresh":
        raise ValueError("Fresh mode cannot load a checkpoint; select an explicit compatible role")
    if mode in ("resume-last", "inference-best"):
        expected = checkpoint_contract(training, args, QUANTILES)
        registry = registry_for(args, BASELINE_REGISTRY)
        model, _, _ = load_registered_model(ClinicalTFT, training, args, ckpt_path,
                                             registry, expected, mode, args.run_id)
    else:
        model = ClinicalTFT.from_dataset(
            training, learning_rate=args.lr, hidden_size=args.hidden_size,
            attention_head_size=args.attention_heads, dropout=args.dropout,
            hidden_continuous_size=args.hidden_continuous_size, lstm_layers=args.lstm_layers,
            loss=loss_fn, log_interval=-1, log_val_interval=-1)
        if mode == "weights-only":
            expected = checkpoint_contract(training, args, QUANTILES)
            registry = registry_for(args, BASELINE_REGISTRY)
            parent_id = getattr(args, "parent_run_id", None)
            weights, _, record = load_registered_model(ClinicalTFT, training, args, ckpt_path,
                                                       registry, expected, mode, parent_id)
            model.load_state_dict(weights, strict=True)
            model._weights_only_parent = {"run_id": parent_id, "checkpoint_sha256": record["sha256"]}

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Trainable parameters: {n_params:,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(model, training, validation, args, ckpt_path, *, extra_callbacks=()):
    """Certified SWA-OFF path; no automatic evaluation or checkpoint discovery."""
    return train_baseline(model, training, validation, args, ckpt_path,
                          BASELINE_REGISTRY, GradientNormLogger, extra_callbacks)


def train_experimental_legacy(
    model:      ClinicalTFT,
    training:   TimeSeriesDataSet,
    validation: TimeSeriesDataSet,
    args:       argparse.Namespace,
    ckpt_path:  Optional[str],
) -> pl.Trainer:
    """Uncertified historical experimental path; SWA ON remains deferred."""
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
        filename="tft-stage-a-v1-{epoch:02d}-{val_loss:.4f}",
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
    coverage_thresh: float = 1.0,
) -> None:
    if coverage_thresh != 1.0:
        raise ValueError("Partial persistence coverage is not supported")
    lookup = set(zip(val_df[SUBJECT_COL].astype(str), val_df["time_idx"].astype(int)))
    keys = [(str(s), int(t)-1) for s, t in zip(subject_ids, dec_time_idx[:, 0])]
    if len(keys) != len(dec_time_idx) or any(key not in lookup for key in keys):
        raise ValueError("Persistence alignment requires every source lookup")


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
def evaluate(model, validation, args, val_df, *, context=None, output_dir=None):
    """C-core reporting: explicit provenance, full evaluation W, raw mg/dL.

    Training/early-stopping datasets retain their Stage A indices. This entry
    qualifies a separate evaluation copy and never fits a model or calibrator.
    """
    if __package__ in (None, ""):
        from evaluation_stage_c import evaluate as evaluate_stage_c
    else:
        from .evaluation_stage_c import evaluate as evaluate_stage_c
    return evaluate_stage_c(model, validation, args, val_df,
                            context=context, output_dir=output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Explicit training entry; evaluation remains a separate, later-stage action."""
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    parquet = args.data_dir / "training.parquet"
    args.dataset_sha256 = artifact_sha256(parquet)
    if args.dataset_sha256 != "d14fe31b1b81971639ca8d5ba17b4882fbd6711569e30785f0da7b91a0c031cf":
        raise ValueError("Certified baseline requires the canonical Stage A dataset SHA-256")
    train_df, val_df = load_and_preprocess_data(args.data_dir)
    training, validation = build_datasets(train_df, val_df, args)
    expected = checkpoint_contract(training, args, QUANTILES)
    registry = registry_for(args, BASELINE_REGISTRY)
    checkpoint = None
    if args.mode in ("resume-last", "inference-best"):
        role = "last" if args.mode == "resume-last" else "best"
        checkpoint = registry.verified(args.run_id, role, expected)[0]
    model = build_model(training, args, checkpoint)
    if args.mode == "inference-best":
        log.info("Verified BEST loaded; no automatic evaluation performed")
        return
    trainer = train(model, training, validation, args, checkpoint)
    log.info("Training segment complete; owned run_id=%s", trainer.nmd_run_id)


if __name__ == "__main__":
    main()
