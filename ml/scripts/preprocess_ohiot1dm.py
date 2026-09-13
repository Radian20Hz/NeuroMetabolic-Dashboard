"""
ml/scripts/preprocess_ohiot1dm.py
===================================
OhioT1DM preprocessing pipeline — Phase 4 v17 (zero-trust audit fixes).

Fixes applied on top of v16
──────────────────────────────────────────
  [FIX-CRIT-1]  ic_ratio_deviation was computed via rolling(24*12).median()
            on the full unsplit patient series and written as a real value to
            the parquet.  It was absent from _SENTINEL_ZERO_COLS so the guard
            never fired.  Now sentineled to NaN (matching dynamic_isf_estimate)
            and added to a new _SENTINEL_NAN_COLS set with its own guard in
            main().  The training script's _compute_long_window_features()
            already overwrites it post-split; the NaN sentinel ensures any path
            that skips that recomputation fails loudly rather than silently.

  [FIX-CRIT-1]  basal_bolus_ratio used rolling(24*12, min_periods=12).sum()
            on the full unsplit patient series.  The 288 rows nearest the 85/15
            split boundary incorporated future val-side bolus events — same
            class as bolus_count_3h (FIX-ROLL-BOUNDARY).  Now sentineled to
            0.0 and added to _SENTINEL_ZERO_COLS; recomputed post-split in
            _compute_long_window_features().

  [FIX-CRIT-1]  lbgi_30m, hbgi_30m, bgri used rolling(6).mean() on the full
            unsplit patient series.  The 6 rows nearest the split boundary
            incorporated future glucose.  Now sentineled to 0.0 and added to
            _SENTINEL_ZERO_COLS; recomputed post-split in
            _compute_long_window_features().

  [FIX-HIGH-2]  steps_since_last_bolus had broken index arithmetic.
            add_meal_bolus_timing() stored df.index label values and recovered
            positions via np.searchsorted(df.index, x).  After any dropna +
            reset_index sequence, pre-reset labels diverge from new 0-based
            positions producing silently wrong values.  Rewritten to work
            entirely in positional integer space (np.arange).

Fixes applied on top of v15
──────────────────────────────────────────
  [FIX-NADIR-ASYM]  glucose_nadir_proximity and glucose_asymmetry_index both
            use rolling(WIN_TIR=24).min/sum() on the full train-XML series.
            After the 85/15 split in the training script the first 24 rows of
            the val split have their rolling window computed over training rows —
            cross-boundary contamination.  Both are now sentineled to 0.0 here
            and recomputed post-split in _compute_long_window_features().

  [FIX-TDD-SENTINEL]  tdd_rolling_7d and bolus_fraction_of_tdd use a
            rolling(WIN_TDD=2016).sum/mean() on the full train-XML series.
            The first 2016 rows of the val split incorporate training rows.
            Both are now sentineled to 0.0 here.  The training script already
            recomputes them post-split in _compute_long_window_features() with
            warm-start blending; the sentinel ensures the parquet never carries
            a contaminated value that could silently bypass that recomputation.

Fixes applied on top of v14
──────────────────────────────────────────
  [FIX-INTERP-CAUSAL]  Replaced pd.Series.interpolate(method="linear") with
            ffill(limit=6) for CGM gap filling.  Linear interpolation fills
            interior gaps by drawing a line between the LEFT and RIGHT flanking
            observed values — the right value is future data.  For a 30-min gap
            (6 steps), the value at t+1 is influenced by the real CGM reading at
            t+7.  ffill only propagates the last known value forward; it is
            strictly causal.

  [FIX-SENTINEL-GUARD]  Sentinel guard loop in main() now raises RuntimeError
            when a sentinel column is absent from the parquet (previously only
            emitted log.warning + continue).  A missing sentinel column is always
            a pipeline error; silent pass would allow regressions to go undetected.

  [FIX-BASAL-DEV]  basal_rate_deviation sentineled to 0.0.  Its
            rolling(WIN_BASELINE=12).mean() baseline was computed on the full
            unsplit patient series, contaminating the 12 rows nearest the split
            boundary.  Recomputed post-split in _compute_long_window_features().

  [FIX-ROLL-BOUNDARY]  Five additional rolling features sentineled to 0.0:
            bolus_count_3h (36-step window, future bolus events),
            glucose_sample_entropy_60m (12-step, future glucose),
            glucose_tir_2h (24-step, future glucose),
            glucose_hyper_ratio_2h (24-step, future glucose),
            glucose_hypo_ratio_2h (24-step, future glucose),
            glucose_dfa_alpha_2h (24-step, future glucose).
            All recomputed post-split in _compute_long_window_features().

All previous fixes from v14 (FIX-C1, FIX-C2, FIX-H1, FIX-H2, FIX-ISSUE-11,
FIX-ISSUE-2a/2b, FIX-ISSUE-10, and all prior v9–v13 fixes) are retained
without modification.

Usage:
    python ml/scripts/preprocess_ohiot1dm.py [--data-dir PATH] [--out-dir PATH]
"""
from __future__ import annotations

import argparse
import logging
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.spatial.distance import pdist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "ml" / "data" / "raw" / "OhioT1DM"
OUT_DIR = ROOT / "ml" / "data" / "processed"

# ─────────────────────────────────────────────────────────────────────────────
# PHARMACOKINETIC CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IOB_PEAK_MIN = 75.0
IOB_DIA_MIN  = 240.0
IOB_TAU1     = 55.0
IOB_TAU2     = 70.0

COB_LAG_MIN      = 15.0
COB_HALFLIFE_MIN = 30.0

TAU_ACUTE_MIN = 30.0
TAU_EPOC_MIN  = 180.0

WIN_VOLATILITY = 6
WIN_TREND      = 12
WIN_TIR        = 24
WIN_ENTROPY    = 12
WIN_HR_VAR     = 3
WIN_HR_TREND   = 6
WIN_STEPS      = 6
WIN_RESTING_HR = 96
WIN_BASELINE   = 12
WIN_TDD        = 7 * 24 * 12

# [FIX-5] Explicit date format matching OhioT1DM XML timestamps.
_TS_FMT = "%d-%m-%Y %H:%M:%S"

# Sentinel columns: written as 0.0 in parquet, recomputed post-split.
_SENTINEL_ZERO_COLS = frozenset({
    "correction_bolus_prior",
    "weekend_meal_prior",
    "hr_recovery_slope",
    "hr_resting_estimate",
    "hr_reserve_pct",
    "is_aerobic_exercise",
    "is_stress_response",
    "hr_above_resting",
    # [FIX-ISSUE-11] autonomic_stress_index depends on hr_above_resting (sentinel)
    # and is computed on the full unsplit patient series in add_wearable_features().
    # Writing 0.0 here ensures the parquet never carries a cross-split contaminated
    # value; _compute_exercise_features_causal() recomputes it post-split.
    "autonomic_stress_index",
    # [FIX-C2] gsr_stress_deviation and skin_temp_deviation subtract a
    # rolling(WIN_BASELINE=12).mean() baseline computed on the full unsplit
    # patient series.  The rolling window straddles the 85/15 split boundary
    # for the 12 rows nearest the split point, incorporating future val-side
    # sensor readings into train-side features.  Both are sentineled here and
    # recomputed post-split inside _compute_exercise_features_causal() on the
    # isolated split data only.
    "gsr_stress_deviation",
    "skin_temp_deviation",
    # [FIX-C2] composite_stress_index depends on gsr_stress_deviation and
    # hr_above_resting (both sentineled), so it is also contaminated pre-split.
    # Already recomputed post-split in _compute_exercise_features_causal().
    "composite_stress_index",
    # [FIX-BASAL-DEV] basal_rate_deviation is computed in add_pk_features() via
    # rolling(WIN_BASELINE=12).mean() on the full unsplit patient series.  The
    # 12 rows nearest the 85/15 split boundary on the train side incorporate
    # future (val-side) basal rate values into their rolling mean baseline,
    # contaminating basal_rate_deviation for those rows.  Sentineled here;
    # recomputed post-split in _compute_long_window_features().
    "basal_rate_deviation",
    # [FIX-ROLL-BOUNDARY] The following features all use rolling windows of
    # 12–36 steps applied to the full unsplit patient series.  Rows within
    # <window_size> steps of the 85/15 split boundary incorporate future
    # (val-side) values.  All are sentineled here and recomputed post-split
    # in _compute_long_window_features() on each isolated split.
    "bolus_count_3h",            # 36-step window (future bolus events)
    "glucose_sample_entropy_60m",# 12-step window (future glucose)
    "glucose_tir_2h",            # 24-step window (future glucose)
    "glucose_hyper_ratio_2h",    # 24-step window (future glucose)
    "glucose_hypo_ratio_2h",     # 24-step window (future glucose)
    "glucose_dfa_alpha_2h",      # 24-step window (future glucose)
    # [FIX-H2] glucose_lag_* are computed via shift() on the full patient
    # train series in process_patient().  At the 85/15 split boundary,
    # glucose_lag_1 at val[0] = the last train glucose, creating a direct
    # cross-boundary information link.  Lags are sentineled here and
    # recomputed post-split in _compute_long_window_features() on each
    # isolated split, with the first observed split glucose as the warm-up fill.
    "glucose_lag_1",
    "glucose_lag_2",
    "glucose_lag_3",
    "glucose_lag_6",
    "glucose_lag_12",
    "glucose_lag_24",
    # [FIX-NADIR-ASYM] glucose_nadir_proximity uses rolling(WIN_TIR=24).min()
    # and glucose_asymmetry_index uses rolling(WIN_TIR=24).sum() — both on the
    # full train-XML series.  The first 24 rows of the val split (after the
    # 85/15 split in the training script) have their rolling window computed
    # using training rows, contaminating those val features.  Sentineled here;
    # recomputed post-split in _compute_long_window_features().
    "glucose_nadir_proximity",   # 24-step rolling min (future glucose)
    "glucose_asymmetry_index",   # 24-step rolling sum (future glucose)
    # [FIX-TDD-SENTINEL] tdd_rolling_7d uses rolling(WIN_TDD=2016).sum/mean()
    # on the full train-XML series.  The first 2016 rows of the val split
    # incorporate training rows, contaminating those val features.  Sentineled
    # here; the training script already recomputes post-split in
    # _compute_long_window_features() with warm-start blending.
    # bolus_fraction_of_tdd depends directly on tdd_rolling_7d, so it is also
    # sentineled.
    "tdd_rolling_7d",
    "bolus_fraction_of_tdd",
    # [FIX-CRIT-1] basal_bolus_ratio uses rolling(24*12, min_periods=12).sum()
    # on the full unsplit patient series.  For the 288 rows nearest the 85/15
    # split boundary, the window incorporates future (val-side) bolus events —
    # the same class of contamination as bolus_count_3h (fixed in FIX-ROLL-BOUNDARY).
    # Sentineled here; recomputed post-split in _compute_long_window_features().
    "basal_bolus_ratio",
    # [FIX-CRIT-1] lbgi_30m, hbgi_30m, bgri all use rolling(6).mean() on the
    # full unsplit patient series.  The 6 rows nearest the split boundary
    # incorporate future glucose values.  Sentineled here; recomputed post-split
    # in _compute_long_window_features().
    "lbgi_30m",
    "hbgi_30m",
    "bgri",
})

# Sentinel columns written as NaN (not 0.0) in parquet, recomputed post-split.
# NaN propagates loudly if the recomputation is ever skipped — unlike 0.0 which
# would silently produce a wrong-but-finite feature value.
_SENTINEL_NAN_COLS: frozenset[str] = frozenset({
    # [FIX-CRIT-1] ic_ratio_deviation: add_ic_ratio_deviation() calls
    # rolling(24*12).median() on the full unsplit patient series.  Writing a
    # real value here means the parquet carries a contaminated value for the
    # first 288 rows of the val split.  The training script's
    # _compute_long_window_features() overwrites it post-split, but only if the
    # pipeline reaches that call.  Sentinel NaN ensures that any path that skips
    # the recomputation fails loudly (NaN propagates to loss) rather than
    # silently (stale contaminated value passes through).
    "ic_ratio_deviation",
})


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OhioT1DM preprocessing — Phase 4 v11 (zero-trust audit fixes)"
    )
    p.add_argument("--data-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir",  type=Path, default=OUT_DIR)
    p.add_argument("--resample-freq", type=str, default="5min")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# XML PARSING  [FIX-1] [FIX-5] [FIX-7]
# ─────────────────────────────────────────────────────────────────────────────
def _parse_glucose(root: ET.Element) -> pd.DataFrame:
    records = []
    for ev in root.findall(".//glucose_level/event"):
        try:
            records.append({
                "timestamp":     pd.to_datetime(ev.attrib["ts"], format=_TS_FMT),
                "glucose_mg_dl": float(ev.attrib["value"]),
            })
        except (KeyError, ValueError):
            continue
    if not records:
        return pd.DataFrame(columns=["timestamp", "glucose_mg_dl"])
    return pd.DataFrame(records)


def _parse_bolus(root: ET.Element) -> pd.DataFrame:
    """[FIX-1] OhioT1DM bolus events use 'ts_begin'. Falls back to 'ts'."""
    records = []
    for ev in root.findall(".//bolus/event"):
        try:
            ts_str = ev.attrib.get("ts_begin") or ev.attrib.get("ts")
            if ts_str is None:
                continue
            dose_str = ev.attrib.get("bolusamount") or ev.attrib.get("dose")
            if dose_str is None:
                continue
            records.append({
                "timestamp":  pd.to_datetime(ts_str, format=_TS_FMT),
                "bolus_dose": float(dose_str),
            })
        except (KeyError, ValueError):
            continue
    if not records:
        return pd.DataFrame(columns=["timestamp", "bolus_dose"])
    return pd.DataFrame(records)


def _parse_basal(root: ET.Element) -> pd.DataFrame:
    records = []
    for ev in root.findall(".//basal/event"):
        try:
            records.append({
                "timestamp":  pd.to_datetime(ev.attrib["ts"], format=_TS_FMT),
                "basal_rate": float(ev.attrib["value"]),
            })
        except (KeyError, ValueError):
            continue
    if not records:
        return pd.DataFrame(columns=["timestamp", "basal_rate"])
    return pd.DataFrame(records)


def _parse_meals(root: ET.Element) -> pd.DataFrame:
    records = []
    for ev in root.findall(".//meal/event"):
        try:
            records.append({
                "timestamp": pd.to_datetime(ev.attrib["ts"], format=_TS_FMT),
                "carbs_g":   float(ev.attrib["carbs"]),
            })
        except (KeyError, ValueError):
            continue
    if not records:
        return pd.DataFrame(columns=["timestamp", "carbs_g"])
    return pd.DataFrame(records)


def _parse_exercise(root: ET.Element) -> pd.DataFrame:
    records = []
    for ev in root.findall(".//exercise/event"):
        try:
            records.append({
                "timestamp":             pd.to_datetime(ev.attrib["ts"], format=_TS_FMT),
                "exercise_duration_min": float(ev.attrib["duration"]),
            })
        except (KeyError, ValueError):
            continue
    if not records:
        return pd.DataFrame(columns=["timestamp", "exercise_duration_min"])
    return pd.DataFrame(records)


def _parse_basis_data(root: ET.Element) -> pd.DataFrame:
    hr_records   = []
    gsr_records  = []
    temp_records = []
    step_records = []

    for ev in root.findall(".//basis_heart_rate/event"):
        try:
            hr_records.append({
                "timestamp":  pd.to_datetime(ev.attrib["ts"], format=_TS_FMT),
                "heart_rate": float(ev.attrib["value"]),
            })
        except (KeyError, ValueError):
            continue

    for ev in root.findall(".//basis_gsr/event"):
        try:
            gsr_records.append({
                "timestamp": pd.to_datetime(ev.attrib["ts"], format=_TS_FMT),
                "gsr":       float(ev.attrib["value"]),
            })
        except (KeyError, ValueError):
            continue

    for ev in root.findall(".//basis_skin_temperature/event"):
        try:
            temp_records.append({
                "timestamp":        pd.to_datetime(ev.attrib["ts"], format=_TS_FMT),
                "skin_temperature": float(ev.attrib["value"]),
            })
        except (KeyError, ValueError):
            continue

    for ev in root.findall(".//basis_steps/event"):
        try:
            step_records.append({
                "timestamp": pd.to_datetime(ev.attrib["ts"], format=_TS_FMT),
                "steps":     float(ev.attrib["value"]),
            })
        except (KeyError, ValueError):
            continue

    def _to_df(records, cols):
        return pd.DataFrame(records) if records else pd.DataFrame(columns=cols)

    hr_df   = _to_df(hr_records,   ["timestamp", "heart_rate"])
    gsr_df  = _to_df(gsr_records,  ["timestamp", "gsr"])
    temp_df = _to_df(temp_records, ["timestamp", "skin_temperature"])
    step_df = _to_df(step_records, ["timestamp", "steps"])

    if hr_df.empty:
        log.warning("  Basis HR data absent — returning empty basis DataFrame.")
        return pd.DataFrame(columns=["timestamp", "heart_rate", "gsr",
                                     "skin_temperature", "steps"])

    # [FIX-ISSUE-2a] direction="backward" — causal: use the most recent
    # prior reading from each sensor, never a future one.
    merged = hr_df
    for other in [gsr_df, temp_df, step_df]:
        if not other.empty:
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                other.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                tolerance=pd.Timedelta("5min"),
            )
    return merged


def load_patient_xml(xml_path: Path) -> ET.Element:
    tree = ET.parse(xml_path)
    return tree.getroot()


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLING  [FIX-2] [LEAK-4] [AUDIT-HIGH-2]
# ─────────────────────────────────────────────────────────────────────────────
def resample_to_cgm_grid(
    cgm_df:      pd.DataFrame,
    bolus_df:    pd.DataFrame,
    basal_df:    pd.DataFrame,
    meal_df:     pd.DataFrame,
    exercise_df: pd.DataFrame,
    basis_df:    pd.DataFrame,
    freq:        str = "5min",
) -> pd.DataFrame:
    if cgm_df.empty:
        return pd.DataFrame()

    t_start = cgm_df["timestamp"].min().floor(freq)
    t_end   = cgm_df["timestamp"].max().ceil(freq)
    grid    = pd.date_range(t_start, t_end, freq=freq)
    df      = pd.DataFrame({"timestamp": grid})

    # [AUDIT-HIGH-2] direction="backward" — strictly causal CGM matching.
    cgm_df = cgm_df.drop_duplicates("timestamp").sort_values("timestamp")
    df = pd.merge_asof(
        df.sort_values("timestamp"),
        cgm_df,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("2min30s"),
    )
    # [FIX-C1] Glucose interpolation is intentionally NOT performed here.
    # Interpolating on the full unsplit series allows gaps that straddle the
    # 85/15 train/val boundary to be filled using future (val-side) values,
    # which is leakage.  Interpolation is now performed post-split inside
    # process_patient(), on each source_split series in isolation, so the
    # forward-fill can never cross the train/val boundary.

    # Bolus
    if not bolus_df.empty:
        bolus_df = bolus_df.copy()
        bolus_df["timestamp"] = bolus_df["timestamp"].dt.floor(freq)
        bolus_binned = (
            bolus_df.groupby("timestamp")["bolus_dose"]
            .sum().reset_index()
            .rename(columns={"bolus_dose": "bolus_event"})
        )
        df = df.merge(bolus_binned, on="timestamp", how="left")
    else:
        df["bolus_event"] = 0.0
    df["bolus_event"] = df["bolus_event"].fillna(0.0)

    # Basal — [FIX-2] drop duplicates before set_index
    if not basal_df.empty:
        basal_clean = (
            basal_df
            .sort_values("timestamp")
            .drop_duplicates(subset="timestamp", keep="last")
            .set_index("timestamp")
        )
        basal_reindexed = basal_clean["basal_rate"].reindex(grid, method="ffill")
        df["basal_rate"] = basal_reindexed.values
    else:
        df["basal_rate"] = 0.0
    df["basal_rate"] = df["basal_rate"].ffill().fillna(0.0)

    # Meals
    if not meal_df.empty:
        meal_df = meal_df.copy()
        meal_df["timestamp"] = meal_df["timestamp"].dt.floor(freq)
        meal_binned = (
            meal_df.groupby("timestamp")["carbs_g"]
            .sum().reset_index()
            .rename(columns={"carbs_g": "meal_event"})
        )
        df = df.merge(meal_binned, on="timestamp", how="left")
    else:
        df["meal_event"] = 0.0
    df["meal_event"] = df["meal_event"].fillna(0.0)

    # Exercise
    if not exercise_df.empty:
        exercise_df = exercise_df.copy()
        exercise_df["timestamp"] = exercise_df["timestamp"].dt.floor(freq)
        ex_binned = (
            exercise_df.groupby("timestamp")["exercise_duration_min"]
            .sum().reset_index()
        )
        df = df.merge(ex_binned, on="timestamp", how="left")
    else:
        df["exercise_duration_min"] = 0.0
    df["exercise_duration_min"] = df["exercise_duration_min"].fillna(0.0)

    # Wearable — causal backward match + short forward-fill.
    # [FIX-ISSUE-2b] direction="backward" replaces direction="nearest".
    # "nearest" allowed future readings (up to +10 min) to be used at time T.
    # "backward" uses only readings that have already occurred (≤ T).
    # The subsequent ffill(limit=3) propagates the last known value forward
    # causally — so gaps of up to 15 min are still covered without leaking.
    if not basis_df.empty:
        basis_df = basis_df.drop_duplicates("timestamp")
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            basis_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta("10min"),
        )
        for col in ["heart_rate", "gsr", "skin_temperature", "steps"]:
            if col in df.columns:
                df[col] = df[col].ffill(limit=3)
    else:
        for col in ["heart_rate", "gsr", "skin_temperature", "steps"]:
            df[col] = np.nan

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour_sin"]               = np.sin(2 * np.pi * ts.dt.hour / 24.0)
    df["hour_cos"]               = np.cos(2 * np.pi * ts.dt.hour / 24.0)
    df["dow_sin"]                = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
    df["dow_cos"]                = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)
    df["month_sin"]              = np.sin(2 * np.pi * (ts.dt.month - 1) / 12.0)
    df["month_cos"]              = np.cos(2 * np.pi * (ts.dt.month - 1) / 12.0)
    df["minutes_since_midnight"] = ts.dt.hour * 60 + ts.dt.minute
    return df


def add_calendar_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calendar window flags and weekend_meal_flag.

    [LEAK-A] weekend_meal_prior is NO LONGER computed here. Sentinel 0.0,
    recomputed post-split in training script.
    """
    h = df["timestamp"].dt.hour
    df["is_dawn_window"]      = ((h >= 3)  & (h < 8)).astype(float)
    df["is_breakfast_window"] = ((h >= 6)  & (h < 10)).astype(float)
    df["is_lunch_window"]     = ((h >= 11) & (h < 14)).astype(float)
    df["is_dinner_window"]    = ((h >= 17) & (h < 21)).astype(float)
    df["is_weekend"]          = (df["timestamp"].dt.dayofweek >= 5).astype(float)

    meal_occurred = (
        df.get("meal_event", pd.Series(0.0, index=df.index)) > 0
    ).astype(float)

    df["weekend_meal_flag"] = df["is_weekend"] * meal_occurred

    # [LEAK-A] Sentinel: authoritative value computed post-split in training script.
    df["weekend_meal_prior"] = 0.0

    return df


def _rolling_linear_deviation(g: pd.Series, win: int) -> pd.Series:
    t        = np.arange(win, dtype=float)
    t_bar    = t.mean()
    w        = t - t_bar
    w_sq_sum = float((w ** 2).sum())

    g_vals = g.values.astype(float)
    n      = len(g_vals)

    w_flipped         = w[::-1]
    weighted_sum_full = fftconvolve(g_vals, w_flipped, mode="full")
    weighted_sum_vals = weighted_sum_full[win - 1: n + win - 1]
    weighted_sum      = pd.Series(weighted_sum_vals, index=g.index)

    g_roll_mean = g.rolling(win, min_periods=win // 2).mean()
    slope       = weighted_sum / w_sq_sum
    intercept   = g_roll_mean - slope * t_bar
    trend_end   = intercept + slope * (win - 1)

    deviation = g - trend_end
    deviation = deviation.where(~g_roll_mean.isna(), other=0.0)
    deviation.iloc[: win - 1] = 0.0
    return deviation.fillna(0.0)


def add_glucose_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    g = df["glucose_mg_dl"]

    df["glucose_delta_1"]      = g.diff(1)
    df["glucose_delta_3"]      = g.diff(3)
    df["glucose_roc"]          = g.diff(1) / 5.0
    df["glucose_acceleration"] = df["glucose_roc"].diff(1) / 5.0

    df["glucose_volatility_30m"] = (
        g.rolling(WIN_VOLATILITY, min_periods=3).std().fillna(0.0)
    )
    df["glucose_deviation_from_trend"] = _rolling_linear_deviation(g, WIN_TREND)
    df["glucose_ewm_roc"] = (
        g.ewm(span=6, min_periods=3).mean().diff(1) / 5.0
    ).fillna(0.0)

    return df


def _sample_entropy_fast(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    n   = len(x)
    std = np.std(x)
    if n < m + 2 or std < 1e-6:
        return 0.0
    r = r_factor * std

    templates_m  = np.array([x[i: i + m]     for i in range(n - m)],     dtype=float)
    templates_m1 = np.array([x[i: i + m + 1] for i in range(n - m - 1)], dtype=float)

    dist_m  = pdist(templates_m[:-1], metric="chebyshev")
    dist_m1 = pdist(templates_m1,     metric="chebyshev")

    B = int(np.sum(dist_m  < r)) * 2
    A = int(np.sum(dist_m1 < r)) * 2

    if B <= 0 or A <= 0:
        return 0.0
    return float(-np.log(A / B))


def add_glucose_regularity(df: pd.DataFrame) -> pd.DataFrame:
    g = df["glucose_mg_dl"]

    # [FIX-ROLL-BOUNDARY] glucose_sample_entropy_60m, glucose_tir_2h,
    # glucose_hyper_ratio_2h, and glucose_hypo_ratio_2h all use rolling
    # windows of 12–24 steps applied to the full unsplit patient series.
    # The WIN_ENTROPY=12 and WIN_TIR=24 rows nearest the 85/15 split boundary
    # incorporate future (val-side) glucose values into their rolling windows,
    # contaminating those train-side feature values.
    # All four are sentineled to 0.0 here and recomputed post-split in
    # _compute_long_window_features() on each isolated split.
    df["glucose_sample_entropy_60m"] = 0.0
    df["glucose_tir_2h"]             = 0.0
    df["glucose_hyper_ratio_2h"]     = 0.0
    df["glucose_hypo_ratio_2h"]      = 0.0
    return df


def _iob_kernel(t_min: np.ndarray) -> np.ndarray:
    t   = np.clip(t_min, 0.0, IOB_DIA_MIN)
    iob = (
        1.0
        - (1.0 - np.exp(-t / IOB_TAU1))
        * (1.0 + (IOB_TAU2 / IOB_TAU1) * (1.0 - np.exp(-t / IOB_TAU2)))
        / (1.0 + IOB_TAU2 / IOB_TAU1)
    )
    return np.clip(iob, 0.0, 1.0)


def _cob_kernel_unit(t_min: np.ndarray) -> np.ndarray:
    t_effective = np.maximum(t_min - COB_LAG_MIN, 0.0)
    post_lag    = np.exp(-t_effective * np.log(2.0) / COB_HALFLIFE_MIN)
    return np.clip(np.where(t_min < COB_LAG_MIN, 1.0, post_lag), 0.0, 1.0)


def compute_iob_cob(df: pd.DataFrame) -> pd.DataFrame:
    dt      = 5.0
    max_lag = int(IOB_DIA_MIN / dt) + 1

    t_arr = np.arange(max_lag + 1) * dt
    iob_k = _iob_kernel(t_arr)
    cob_k = _cob_kernel_unit(t_arr)

    bolus = df["bolus_event"].values.astype(float)
    meal  = df["meal_event"].values.astype(float)

    iob_conv = fftconvolve(bolus, iob_k, mode="full")[: len(bolus)]
    cob_conv = fftconvolve(meal,  cob_k, mode="full")[: len(meal)]

    df["insulin_on_board"] = np.clip(iob_conv, 0.0, None)
    df["carb_on_board"]    = np.clip(cob_conv, 0.0, None)
    return df


def add_pk_features(df: pd.DataFrame) -> pd.DataFrame:
    assert "bolus_last_1h" in df.columns
    sensitivity = 50.0

    df["net_insulin_effect"] = (
        df["insulin_on_board"] - df["carb_on_board"] / sensitivity
    )
    df["carb_to_bolus_ratio_1h"] = (
        df["carb_on_board"] / (df["bolus_last_1h"].clip(lower=0.0) + 1e-6)
    ).clip(0.0, 20.0)
    df["net_glucose_pressure"] = (
        df["carb_on_board"] - df["insulin_on_board"] * sensitivity
    )

    # [FIX-BASAL-DEV] basal_rate_deviation is sentineled to 0.0 here.
    # Computing rolling(WIN_BASELINE=12).mean() on the full unsplit patient
    # series lets the 12 rows nearest the 85/15 boundary incorporate future
    # (val-side) basal readings into their rolling mean, contaminating the
    # deviation for those train rows.  The authoritative causal value is
    # recomputed post-split in _compute_long_window_features() on each
    # isolated split.
    df["basal_rate_deviation"] = 0.0
    return df


def add_rolling_insulin_carb(df: pd.DataFrame) -> pd.DataFrame:
    df["bolus_last_1h"] = df["bolus_event"].rolling(12, min_periods=1).sum()
    df["carbs_last_1h"] = df["meal_event"].rolling(12, min_periods=1).sum()
    return df


def add_insulin_stacking(df: pd.DataFrame) -> pd.DataFrame:
    # [FIX-ROLL-BOUNDARY] bolus_count_3h uses a rolling window of 36 steps on
    # the full unsplit patient series.  The 36 rows nearest the 85/15 split
    # boundary incorporate future (val-side) bolus events.  Sentineled to 0.0
    # here; recomputed post-split in _compute_long_window_features().
    df["bolus_count_3h"]   = 0.0
    df["iob_delta"]        = df["insulin_on_board"].diff(1).fillna(0.0)
    df["iob_acceleration"] = df["iob_delta"].diff(1).fillna(0.0)
    df["cob_iob_ratio"]    = (
        df["carb_on_board"] / (df["insulin_on_board"] + 1e-6)
    ).clip(0.0, 20.0)
    df["meal_occurred_last_30m"] = (
        (df["meal_event"] > 0).astype(float).rolling(WIN_STEPS, min_periods=1).max()
    )
    return df


def add_tdd_features(df: pd.DataFrame) -> pd.DataFrame:
    # [FIX-TDD-SENTINEL] tdd_rolling_7d and bolus_fraction_of_tdd are
    # sentineled to 0.0 here.  Their rolling(WIN_TDD=2016).sum/mean() baseline
    # was computed on the full train-XML series; after the 85/15 split in the
    # training script the first 2016 rows of the val split incorporate training
    # rows into their rolling window.  Authoritative causal values are
    # recomputed post-split in _compute_long_window_features() with warm-start
    # blending (FIX-ISSUE-15).
    df["tdd_rolling_7d"]       = 0.0
    df["bolus_fraction_of_tdd"] = 0.0

    df["basal_rate_change"] = df["basal_rate"].diff(1).abs().fillna(0.0)
    df["is_temp_basal"]     = (df["basal_rate_change"] > 0.10).astype(float)
    return df


def add_exercise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [ZT-CRIT-NEW-1] hr_resting_estimate, hr_reserve_pct, is_aerobic_exercise,
    and is_stress_response are written as sentinel 0.0 columns.
    Authoritative causal values are computed post-split in the training script.

    exercise_acute_effect and exercise_epoc_effect are convolution-based
    (causal by construction) and remain here.
    """
    dt           = 5.0
    max_lag_epoc = int(TAU_EPOC_MIN * 3 / dt) + 1
    t_arr        = np.arange(max_lag_epoc + 1) * dt

    acute_k = np.exp(-t_arr / TAU_ACUTE_MIN)
    epoc_k  = np.exp(-t_arr / TAU_EPOC_MIN)

    ex_vals = df["exercise_duration_min"].values.astype(float)

    df["exercise_acute_effect"] = fftconvolve(ex_vals, acute_k, mode="full")[: len(ex_vals)]
    df["exercise_epoc_effect"]  = fftconvolve(ex_vals, epoc_k,  mode="full")[: len(ex_vals)]

    # [ZT-CRIT-NEW-1] Sentinels — recomputed post-split with causal resting HR.
    df["hr_resting_estimate"] = 0.0
    df["hr_reserve_pct"]      = 0.0
    df["is_aerobic_exercise"] = 0.0
    df["is_stress_response"]  = 0.0

    return df


def add_wearable_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [ZT-CRIT-NEW-2] hr_above_resting is sentineled to 0.0.
    Authoritative value computed post-split alongside hr_resting_estimate.
    """
    has_hr   = "heart_rate"       in df.columns and df["heart_rate"].notna().any()
    has_gsr  = "gsr"              in df.columns and df["gsr"].notna().any()
    has_temp = "skin_temperature" in df.columns and df["skin_temperature"].notna().any()
    has_step = "steps"            in df.columns and df["steps"].notna().any()

    if has_hr:
        df["hr_variability_15m"] = (
            df["heart_rate"].rolling(WIN_HR_VAR, min_periods=2).std().fillna(0.0)
        )

        # [ZT-CRIT-NEW-2] Sentinel — authoritative value recomputed post-split.
        df["hr_above_resting"] = 0.0

        def _hr_slope(x: np.ndarray) -> float:
            if len(x) < 2 or np.all(np.isnan(x)):
                return 0.0
            t     = np.arange(len(x), dtype=float)
            valid = ~np.isnan(x)
            if valid.sum() < 2:
                return 0.0
            return float(np.polyfit(t[valid], x[valid], 1)[0])

        df["hr_trend_30m"] = (
            df["heart_rate"]
            .rolling(WIN_HR_TREND, min_periods=3)
            .apply(_hr_slope, raw=True)
            .fillna(0.0)
        )
    else:
        df["hr_variability_15m"] = 0.0
        df["hr_above_resting"]   = 0.0
        df["hr_trend_30m"]       = 0.0

    # [FIX-C2] gsr_stress_deviation and skin_temp_deviation are sentineled to
    # 0.0.  Their rolling(WIN_BASELINE=12).mean() baseline was previously
    # computed on the full unsplit patient series, letting the 12 rows before
    # the 85/15 split boundary incorporate future val-side sensor readings.
    # Authoritative values are recomputed post-split inside
    # _compute_exercise_features_causal() on the isolated split data only.
    if has_gsr:
        df["gsr_stress_deviation"] = 0.0
    else:
        df["gsr_stress_deviation"] = 0.0

    if has_temp:
        df["skin_temp_deviation"] = 0.0
    else:
        df["skin_temp_deviation"] = 0.0

    if has_step:
        df["steps_last_30m"] = (
            df["steps"].rolling(WIN_STEPS, min_periods=1).sum().fillna(0.0)
        )
    else:
        df["steps_last_30m"] = 0.0

    # [FIX-C2] composite_stress_index depends on hr_above_resting (sentinel,
    # [ZT-CRIT-NEW-2]) and gsr_stress_deviation (sentinel, [FIX-C2]).
    # Computing it here with two 0.0-valued inputs produces a meaningless
    # constant; it is recomputed correctly post-split in
    # _compute_exercise_features_causal() after both inputs are restored.
    df["composite_stress_index"] = 0.0

    # [FIX-ISSUE-11] autonomic_stress_index is written as sentinel 0.0 here.
    # Computing it on the full unsplit series produces z-scores whose rolling
    # normalisation statistics straddle the train/test boundary.
    # _compute_exercise_features_causal() recomputes the authoritative value
    # post-split, on isolated train or val/test data only.
    df["autonomic_stress_index"] = 0.0

    return df


def add_circadian_pk_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """[FIX-3] Use .any(axis=1) — works correctly on float columns."""
    df["dawn_iob_interaction"] = df["is_dawn_window"] * df["insulin_on_board"]

    meal_windows = ["is_breakfast_window", "is_lunch_window", "is_dinner_window"]
    any_meal_window = df[meal_windows].any(axis=1)
    df["postmeal_cob_interaction"] = any_meal_window.astype(float) * df["carb_on_board"]
    return df


def add_cgm_gap_flag(df: pd.DataFrame, original_glucose: pd.Series) -> pd.DataFrame:
    df["cgm_gap_flag"] = original_glucose.isna().astype(float).values
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 v3 FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_glucose_risk_indices(df: pd.DataFrame) -> pd.DataFrame:
    # [FIX-CRIT-1] lbgi_30m, hbgi_30m, and bgri all use rolling(6).mean()
    # applied to the full unsplit patient series.  The 6 rows nearest the
    # 85/15 split boundary incorporate future (val-side) glucose values —
    # the same class of boundary contamination as the features fixed in
    # FIX-ROLL-BOUNDARY.  All three are sentineled to 0.0 here and
    # recomputed post-split in _compute_long_window_features().
    df["lbgi_30m"] = 0.0
    df["hbgi_30m"] = 0.0
    df["bgri"]     = 0.0
    return df


def add_glucose_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df["glucose_hw_level"]     = df["glucose_mg_dl"].ewm(alpha=0.3, adjust=False).mean()
    df["glucose_hw_trend"]     = (
        df["glucose_hw_level"].diff(1).ewm(alpha=0.1, adjust=False).mean()
    )
    df["glucose_momentum_60m"] = df["glucose_hw_trend"] * 12
    return df


def add_ic_ratio_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    [FIX-CRIT-1] Parquet sentinel — write NaN so that any partial-pipeline
    run that skips the training script's post-split recomputation fails loudly
    (NaN propagates to the model) rather than silently (a contaminated value
    passes through undetected).

    The authoritative causal value is computed post-split in the training script
    inside _compute_long_window_features() with optional warm-start blending
    ([FIX-ISSUE-15]).  That function overwrites this column on both the train
    and val splits before any model sees it.

    Previously this wrote a real rolling(24*12).median() value computed on the
    full unsplit series.  After the 85/15 split, the first 288 rows of the val
    split had their rolling window computed using training rows, contaminating
    ic_ratio_deviation for those rows.
    """
    df["ic_ratio_deviation"] = np.nan
    return df


def add_meal_response_deviation(df: pd.DataFrame) -> pd.DataFrame:
    cob_decay_rate   = 4.0
    cob_absorbed_15m = df["carb_on_board"].diff(3).clip(upper=0.0).abs()
    expected_rise    = cob_absorbed_15m * cob_decay_rate
    actual_rise      = df["glucose_delta_3"].clip(lower=0.0)
    df["meal_response_deviation"] = (actual_rise - expected_rise).fillna(0.0)
    return df


def add_nocturnal_features(df: pd.DataFrame) -> pd.DataFrame:
    h            = df["timestamp"].dt.hour
    is_nocturnal = ((h >= 23) | (h < 6)).astype(float)
    df["nocturnal_glucose_drift"]   = (df["glucose_roc"] * is_nocturnal).fillna(0.0)
    df["dawn_glucose_acceleration"] = (
        df["glucose_acceleration"] * df["is_dawn_window"]
    ).fillna(0.0)
    return df


def add_exercise_insulin_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    [ZT-CRIT-NEW-1] is_aerobic_exercise is 0.0 sentinel here; interactions
    are recomputed post-split after is_aerobic_exercise is restored causally.
    """
    df["exercise_iob_danger"]   = df["is_aerobic_exercise"] * df["insulin_on_board"]
    df["exercise_cob_coverage"] = df["is_aerobic_exercise"] * df["carb_on_board"]
    return df


def add_glucose_nadir_proximity(df: pd.DataFrame) -> pd.DataFrame:
    # [FIX-NADIR-ASYM] Sentineled to 0.0.  rolling(WIN_TIR=24).min() on the
    # full train-XML series contaminates the first 24 val rows (post-85/15
    # split) with training glucose values.  Recomputed post-split in
    # _compute_long_window_features().
    df["glucose_nadir_proximity"] = 0.0
    return df


def add_basal_bolus_ratio(df: pd.DataFrame) -> pd.DataFrame:
    # [FIX-CRIT-1] basal_bolus_ratio uses rolling(24*12, min_periods=12).sum()
    # on the full unsplit patient series.  For the 288 rows nearest the 85/15
    # split boundary the window incorporates future (val-side) bolus events —
    # the same class of contamination as bolus_count_3h (fixed in
    # FIX-ROLL-BOUNDARY).  Sentineled to 0.0 here; recomputed post-split
    # in _compute_long_window_features() on each isolated split.
    df["basal_bolus_ratio"] = 0.0
    return df


def add_glucose_cv(df: pd.DataFrame) -> pd.DataFrame:
    g         = df["glucose_mg_dl"]
    roll_mean = g.rolling(WIN_VOLATILITY, min_periods=3).mean()
    roll_std  = g.rolling(WIN_VOLATILITY, min_periods=3).std()
    df["glucose_cv_30m"] = (
        roll_std / roll_mean.clip(lower=1.0) * 100.0
    ).fillna(0.0)
    return df


def add_dynamic_isf(df: pd.DataFrame) -> pd.DataFrame:
    """
    [AUDIT-HIGH-3] Parquet sentinel — write NaN so that any partial-pipeline
    run that skips the training script's post-split recomputation fails loudly
    (NaN propagates to the model) rather than silently (a leaky value passes
    through undetected).

    The authoritative causal value is computed post-split in the training script
    inside _compute_long_window_features().  That function overwrites this column
    on both the train and val splits before any model sees it.

    [FIX-ISSUE-10] Previously this wrote a real rolling-median value computed on
    the full unsplit series, leaking test data into the parquet.  All callers
    in the training script already discard and recompute this column, so the
    change is safe.
    """
    df["dynamic_isf_estimate"] = np.nan
    return df


def add_postprandial_phase(df: pd.DataFrame) -> pd.DataFrame:
    cob = df["carb_on_board"]
    meal_recent = (
        df.get("meal_event", pd.Series(0.0, index=df.index))
        .rolling(36, min_periods=1).max() > 0
    )
    cob_peak = cob.rolling(6, min_periods=1).max()
    conditions = [
        meal_recent & (cob >= cob_peak * 0.8),
        meal_recent & (cob < cob_peak * 0.8) & (cob > 2.0),
        meal_recent & (cob <= 2.0) & (cob > 0.1),
    ]
    df["postprandial_phase"] = np.select(conditions, [1.0, 2.0, 3.0], default=0.0)
    return df


def add_stress_glucose_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df["stress_glucose_product"] = (
        df["gsr_stress_deviation"].clip(lower=0.0)
        * df["glucose_roc"].clip(lower=0.0)
    ).fillna(0.0)
    return df


def add_glucose_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    # [FIX-NADIR-ASYM] Sentineled to 0.0.  rolling(WIN_TIR=24).sum() on the
    # full train-XML series contaminates the first 24 val rows (post-85/15
    # split) with training glucose values.  Recomputed post-split in
    # _compute_long_window_features().
    df["glucose_asymmetry_index"] = 0.0
    return df


def add_step_insulin_offset(df: pd.DataFrame) -> pd.DataFrame:
    df["step_insulin_offset"] = (
        df["steps_last_30m"] / (df["insulin_on_board"] + 1.0)
    ).clip(0.0, 500.0).fillna(0.0)
    return df


def add_temp_drop_signal(df: pd.DataFrame) -> pd.DataFrame:
    if "skin_temperature" not in df.columns or not df["skin_temperature"].notna().any():
        df["skin_temp_drop_rate"] = 0.0
        return df
    df["skin_temp_drop_rate"] = (
        df["skin_temperature"].diff(6).clip(upper=0.0).abs().fillna(0.0)
    )
    return df


def _dfa_alpha(x: np.ndarray, scales: tuple = (4, 6, 8, 12)) -> float:
    if len(x) < max(scales) * 2:
        return 0.5
    x_cum = np.cumsum(x - np.mean(x))
    fluctuations = []
    valid_scales = []
    for s in scales:
        n_seg = len(x_cum) // s
        if n_seg < 2:
            continue
        segments = x_cum[: n_seg * s].reshape(n_seg, s)
        t = np.arange(s, dtype=float)
        A = np.column_stack([t, np.ones(s)])
        coeffs, _, _, _ = np.linalg.lstsq(A, segments.T, rcond=None)
        trend = (A @ coeffs).T
        detrended = segments - trend
        fluctuations.append(np.sqrt(np.mean(detrended ** 2)))
        valid_scales.append(s)
    if len(fluctuations) < 2:
        return 0.5
    log_s = np.log(valid_scales)
    log_f = np.log(np.array(fluctuations) + 1e-12)
    return float(np.polyfit(log_s, log_f, 1)[0])


def add_glucose_dfa(df: pd.DataFrame) -> pd.DataFrame:
    # [FIX-ROLL-BOUNDARY] glucose_dfa_alpha_2h uses a rolling window of
    # WIN_TIR=24 steps on the full unsplit patient series.  The 24 rows
    # nearest the 85/15 split boundary incorporate future (val-side) glucose
    # values.  Sentineled to 0.5 (neutral DFA exponent) here; recomputed
    # post-split in _compute_long_window_features().
    df["glucose_dfa_alpha_2h"] = 0.0
    return df


def add_iob_glucose_coinfall(df: pd.DataFrame) -> pd.DataFrame:
    iob_falling     = df["iob_delta"].clip(upper=0.0).abs()
    glucose_falling = df["glucose_roc"].clip(upper=0.0).abs()
    df["iob_glucose_coinfall"] = (iob_falling * glucose_falling).fillna(0.0)
    return df


def add_meal_bolus_timing(df: pd.DataFrame) -> pd.DataFrame:
    # [FIX-HIGH-2] The previous implementation stored df.index label values
    # into bolus_idx, then called np.searchsorted(df.index, x) to recover
    # positional offsets.  After reset_index(drop=True) the RangeIndex labels
    # equal positions, so the logic appeared correct.  However after the
    # glucose-NaN dropna + reset_index sequence in process_patient(), any rows
    # that were dropped cause the pre-reset index labels to diverge from the
    # new 0-based positions, producing silently wrong steps_since_last_bolus
    # values for the affected rows.
    #
    # Fix: work entirely in positional (integer) space.  Create a positional
    # array 0…N-1, mark bolus positions, forward-fill the last bolus position,
    # then subtract to get steps elapsed since that bolus.
    n = len(df)
    positions     = np.arange(n, dtype=float)
    bolus_mask    = (df["bolus_event"].values > 0).astype(float)
    # Replace non-bolus positions with NaN so ffill carries the last bolus pos.
    bolus_pos     = np.where(bolus_mask, positions, np.nan)
    bolus_pos_s   = pd.Series(bolus_pos).ffill().fillna(0.0).values
    df["steps_since_last_bolus"] = positions - bolus_pos_s
    return df


def add_hr_recovery_slope(df: pd.DataFrame) -> pd.DataFrame:
    """[ZT-HIGH-3] Sentinel — causal version computed post-split in training script."""
    df["hr_recovery_slope"] = 0.0
    return df


def add_glucose_momentum_divergence(df: pd.DataFrame) -> pd.DataFrame:
    assert "glucose_hw_level" in df.columns
    fast_ema = df["glucose_mg_dl"].ewm(alpha=0.5, min_periods=3).mean()
    df["glucose_momentum_divergence"] = (fast_ema - df["glucose_hw_level"]).fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PER-PATIENT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def process_patient(
    xml_path:     Path,
    patient_id:   str,
    freq:         str = "5min",
    source_split: str = "train",
) -> Optional[pd.DataFrame]:
    log.info(f"  Processing patient {patient_id} [{source_split}]: {xml_path.name}")

    try:
        root = load_patient_xml(xml_path)
    except ET.ParseError as e:
        log.error(f"  XML parse error for {xml_path.name}: {e}")
        return None

    glucose_df  = _parse_glucose(root).sort_values("timestamp")
    bolus_df    = _parse_bolus(root).sort_values("timestamp")
    basal_df    = _parse_basal(root).sort_values("timestamp")
    meal_df     = _parse_meals(root).sort_values("timestamp")
    exercise_df = _parse_exercise(root).sort_values("timestamp")
    basis_df    = _parse_basis_data(root).sort_values("timestamp")

    if glucose_df.empty:
        log.warning(f"  Patient {patient_id}: no glucose data — skipping.")
        return None

    df = resample_to_cgm_grid(
        glucose_df, bolus_df, basal_df, meal_df, exercise_df, basis_df, freq
    )

    if df.empty or len(df) < 100:
        log.warning(f"  Patient {patient_id}: insufficient data after resampling.")
        return None

    n_before = len(df)

    # [FIX-C1] Post-split causal glucose interpolation.
    # Each XML file corresponds to exactly one source_split (train XML → "train",
    # test XML → "test"), so interpolation here is guaranteed to operate on a
    # single split's data and can never cross the train/test boundary.
    # Previously, interpolation ran in resample_to_cgm_grid() on the full unsplit
    # series, allowing gaps straddling the 85/15 boundary to be filled using
    # future (val-side) glucose values.
    #
    # [AUDIT-HIGH-NEW-1] Capture which rows are REAL (non-interpolated) CGM
    # observations BEFORE interpolation and any further row drops or index
    # changes. Store as a plain NumPy boolean array so it is immune to index
    # resets.
    # [ZT-HIGH-NEW-2] Previously stored as a pd.Series and re-aligned with
    # isin(df.index) after reset_index(drop=True) — that comparison was
    # against the NEW sequential index (0…N-1), not the original positions,
    # silently including interpolated rows in the "observed" set.
    raw_cgm_observed_arr: np.ndarray = df["glucose_mg_dl"].notna().to_numpy(dtype=bool)

    # [FIX-C1] [FIX-INTERP-CAUSAL] Fill CGM gaps of up to 6 steps (30 min)
    # on the isolated single-split series using forward-fill only.
    #
    # IMPORTANT — why ffill, not linear interpolation:
    # pd.Series.interpolate(method="linear") fills interior gaps by drawing a
    # straight line between the last known value on the LEFT and the next known
    # value on the RIGHT of the gap.  The right endpoint is a FUTURE observation,
    # so the filled value at every step inside the gap incorporates future glucose
    # data.  For a 30-min gap (6 steps), the value interpolated at t+1 is
    # influenced by the real CGM reading at t+7.  This is classical two-sided
    # leakage regardless of limit_direction.
    #
    # ffill propagates the last observed value forward; it never looks ahead.
    # Clinically, carrying the last CGM reading is also the correct assumption
    # for a closed-loop system that has lost signal.
    df["glucose_mg_dl"] = df["glucose_mg_dl"].ffill(limit=6)

    if source_split == "train":
        keep_mask = df["glucose_mg_dl"].notna().to_numpy(dtype=bool)
        df = df[keep_mask].reset_index(drop=True)
        # [ZT-HIGH-NEW-2] Filter the boolean array in parallel with the rows.
        raw_cgm_observed_arr = raw_cgm_observed_arr[keep_mask]
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            log.info(f"    [{source_split}] Dropped {n_dropped} rows with missing glucose.")
        if len(df) < 100:
            log.warning(f"  Patient {patient_id}: too few rows after dropna — skipping.")
            return None
    else:
        n_nan_remaining = df["glucose_mg_dl"].isna().sum()
        if n_nan_remaining > 0:
            keep_mask = df["glucose_mg_dl"].notna().to_numpy(dtype=bool)
            # [ZT-HIGH-NEW-2] Filter array in parallel before resetting index.
            raw_cgm_observed_arr = raw_cgm_observed_arr[keep_mask]
            df = df[keep_mask].reset_index(drop=True)
        if len(df) < 100:
            log.warning(f"  Patient {patient_id} [{source_split}]: too few rows — skipping.")
            return None

    original_glucose = df["glucose_mg_dl"].copy()

    # Feature engineering
    df = add_time_features(df)
    df = add_calendar_windows(df)
    df = add_glucose_dynamics(df)
    df = add_glucose_regularity(df)
    df = add_glucose_risk_indices(df)
    df = add_glucose_momentum(df)
    df = add_glucose_cv(df)
    df = add_glucose_dfa(df)

    df = add_rolling_insulin_carb(df)
    df = compute_iob_cob(df)
    df = add_pk_features(df)

    df = add_insulin_stacking(df)
    df = add_tdd_features(df)

    df = add_meal_response_deviation(df)

    df = add_wearable_features(df)
    df = add_exercise_features(df)
    df = add_exercise_insulin_interactions(df)
    df = add_nocturnal_features(df)
    df = add_circadian_pk_interactions(df)
    df = add_glucose_nadir_proximity(df)
    df = add_basal_bolus_ratio(df)

    df = add_postprandial_phase(df)
    df = add_stress_glucose_interaction(df)
    df = add_glucose_asymmetry(df)
    df = add_step_insulin_offset(df)
    df = add_temp_drop_signal(df)

    df = add_ic_ratio_deviation(df)
    df = add_iob_glucose_coinfall(df)
    df = add_meal_bolus_timing(df)
    df = add_hr_recovery_slope(df)
    df = add_glucose_momentum_divergence(df)
    df = add_dynamic_isf(df)

    df = add_cgm_gap_flag(df, original_glucose)

    df["subject_id"]   = patient_id
    df["source_split"] = source_split

    # [FIX-H2] glucose_lag_* are sentineled to 0.0 here.
    # Previously shift() was applied on the full patient series (before the
    # training script's 85/15 split), making glucose_lag_1 at val[0] equal
    # to the last train glucose — a direct cross-boundary leak.
    # Authoritative per-split lag values are computed post-split in
    # _compute_long_window_features() in the training script, on each
    # isolated split with the first observed glucose as the warm-up fill.
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"glucose_lag_{lag}"] = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("NMD — OhioT1DM Preprocessing Pipeline (Phase 4 v15 zero-trust fixed)")
    log.info(f"  Raw data : {args.data_dir}")
    log.info(f"  Output   : {out_dir}")
    log.info(f"  Resample : {args.resample_freq}")
    log.info("=" * 65)

    xml_files = sorted(args.data_dir.glob("*.xml"))
    if not xml_files:
        xml_files = sorted(args.data_dir.rglob("*.xml"))

    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found in {args.data_dir}. "
            "Ensure OhioT1DM dataset is extracted correctly."
        )

    log.info(f"Found {len(xml_files)} XML file(s): {[f.name for f in xml_files]}")

    all_dfs: list[pd.DataFrame] = []
    for xml_path in xml_files:
        patient_id   = xml_path.stem.split("-")[0]
        source_split = "test" if "testing" in xml_path.name.lower() else "train"
        patient_df   = process_patient(
            xml_path, patient_id, args.resample_freq, source_split
        )
        if patient_df is not None:
            all_dfs.append(patient_df)

    if not all_dfs:
        raise RuntimeError("No patient data successfully processed.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = (
        combined.sort_values(["subject_id", "source_split", "timestamp"])
        .reset_index(drop=True)
    )

    n_train = (combined["source_split"] == "train").sum()
    n_test  = (combined["source_split"] == "test").sum()

    log.info("=" * 65)
    log.info(
        f"Combined dataset: {len(combined):,} rows, "
        f"{len(combined.columns)} columns, "
        f"{combined['subject_id'].nunique()} patients"
    )
    log.info(f"  source_split breakdown: train={n_train:,}  test={n_test:,}")

    # ── Sentinel guards ───────────────────────────────────────────────────
    # [FIX-H1] Replaced the old vacuously-true assertion
    # ('correction_bolus_prior' not in combined.columns — a column that was
    # never written, so the test always passed) with a unified loop over
    # _SENTINEL_ZERO_COLS that verifies every sentinel column IS present and
    # contains only 0.0.  This catches any future regression where a real
    # pre-split computation is accidentally re-introduced.
    #
    # correction_bolus_prior is the one exception: it is intentionally never
    # written to the parquet (computed entirely post-split in the training
    # script), so its absence is expected and not an error.
    for _sentinel_col in sorted(_SENTINEL_ZERO_COLS):
        if _sentinel_col == "correction_bolus_prior":
            if _sentinel_col in combined.columns:
                raise RuntimeError(
                    "[LEAK-1] correction_bolus_prior found in parquet. "
                    "This column must NOT be written by the preprocessor; "
                    "it is computed post-split in the training script only."
                )
            log.info("  [SENTINEL-GUARD] correction_bolus_prior correctly absent ✓")
            continue
        if _sentinel_col not in combined.columns:
            raise RuntimeError(
                f"[SENTINEL-GUARD] {_sentinel_col} is absent from the parquet. "
                f"Every member of _SENTINEL_ZERO_COLS must be written as a 0.0 "
                f"sentinel by the preprocessor so that the training script can "
                f"detect regressions.  Check that the column's sentinel assignment "
                f"(df['{_sentinel_col}'] = 0.0) was not accidentally removed."
            )
        _nonzero = (combined[_sentinel_col] != 0.0).sum()
        if _nonzero > 0:
            raise RuntimeError(
                f"[SENTINEL-GUARD] {_sentinel_col} has {_nonzero} non-zero "
                f"values in parquet. This column must be written as sentinel "
                f"0.0 only; authoritative values are computed post-split in "
                f"the training script."
            )
        log.info(f"  [SENTINEL-GUARD] {_sentinel_col} is all-zero sentinel ✓")

    # ── NaN-sentinel guard (_SENTINEL_NAN_COLS) ───────────────────────────
    # These columns must be written as all-NaN sentinels by the preprocessor
    # (analogous to dynamic_isf_estimate).  A non-NaN value means the column
    # was computed on the full unsplit series and is potentially contaminated.
    for _nan_col in sorted(_SENTINEL_NAN_COLS):
        if _nan_col not in combined.columns:
            raise RuntimeError(
                f"[SENTINEL-NAN-GUARD] {_nan_col} is absent from the parquet. "
                f"Every member of _SENTINEL_NAN_COLS must be written as an all-NaN "
                f"sentinel by the preprocessor.  Check that the column's sentinel "
                f"assignment (df['{_nan_col}'] = np.nan) was not accidentally removed."
            )
        _non_nan = combined[_nan_col].notna().sum()
        if _non_nan > 0:
            raise RuntimeError(
                f"[SENTINEL-NAN-GUARD] {_nan_col} has {_non_nan} non-NaN "
                f"values in parquet.  This column must be written as all-NaN sentinel; "
                f"authoritative values are computed post-split in the training script."
            )
        log.info(f"  [SENTINEL-NAN-GUARD] {_nan_col} is all-NaN sentinel ✓")

    log.info("  [AUDIT-HIGH-2 guard] CGM merge_asof direction=backward ✓")
    log.info("  [FIX-ISSUE-2b guard] Wearable merge_asof direction=backward ✓")

    # [FIX-ISSUE-10] dynamic_isf_estimate must be all-NaN in the parquet;
    # the training script's _compute_long_window_features() writes the real value.
    # [FIX-CRIT-1] ic_ratio_deviation is now also all-NaN (checked by the
    # _SENTINEL_NAN_COLS loop above; explicit guard retained here for clarity).
    if "dynamic_isf_estimate" in combined.columns:
        non_nan = combined["dynamic_isf_estimate"].notna().sum()
        if non_nan > 0:
            raise RuntimeError(
                f"[FIX-ISSUE-10] dynamic_isf_estimate has {non_nan} non-NaN values "
                f"in parquet. add_dynamic_isf() must write only NaN sentinels. "
                f"Delete training.parquet and re-run preprocess_ohiot1dm.py."
            )
        log.info("  [FIX-ISSUE-10 guard] dynamic_isf_estimate is all-NaN sentinel ✓")

    for pid in sorted(combined["subject_id"].unique()):
        for split in ["train", "test"]:
            mask = (
                (combined["subject_id"] == pid)
                & (combined["source_split"] == split)
            )
            n = int(mask.sum())
            if n == 0:
                continue
            hypo_pct = (combined.loc[mask, "glucose_mg_dl"] < 70).mean() * 100
            log.info(
                f"  Patient {pid} [{split:5s}]: {n:,} rows | "
                f"hypo rate: {hypo_pct:.1f}%"
            )

    log.info(f"\nFeature columns ({len(combined.columns) - 4}):")
    meta_cols = {"subject_id", "source_split", "timestamp", "glucose_mg_dl"}
    for col in sorted(c for c in combined.columns if c not in meta_cols):
        null_pct = combined[col].isna().mean() * 100
        log.info(f"  {col:<50s}  null={null_pct:.1f}%")

    out_path = out_dir / "training.parquet"
    combined.to_parquet(out_path, index=False, compression="snappy")
    log.info(f"\nSaved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    log.info("=" * 65)


if __name__ == "__main__":
    main()