"""Shared Stage A time semantics. No patient data or fitted state at import."""
import numpy as np
import pandas as pd

TARGET_OBSERVED = "target_observed"


def regular_timeline(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore five-minute elapsed time for a single patient/split.

    Missing event bins mean no recorded event; CGM/sensors remain missing.
    The production preprocessor supplies every bin, including events during CGM
    outages. This helper cannot recover events already lost by a legacy export.
    """
    if "timestamp" not in frame:
        return frame.copy()  # array-only helpers already use a regular grid
    for key in ("subject_id", "source_split"):
        if key in frame and frame[key].nunique() > 1:
            raise ValueError(f"regular_timeline requires one {key}")
    out = frame.sort_values("timestamp").copy()
    ts = pd.DatetimeIndex(out.timestamp)
    if ts.has_duplicates or ts.hasnans:
        raise ValueError("Duplicate/missing timestamp")
    if not (ts == ts.floor("5min")).all():
        raise ValueError("Expected five-minute grid")
    if len(out) < 2:
        return out.reset_index(drop=True)
    grid = pd.date_range(ts[0], ts[-1], freq="5min")
    if len(grid) == len(out):
        return out.reset_index(drop=True)
    out = out.set_index("timestamp").reindex(grid).rename_axis("timestamp").reset_index()
    for col in ("bolus_dose", "bolus_event", "meal_event", "exercise_duration_min"):
        if col in out:
            out[col] = out[col].fillna(0.)
    for col in ("subject_id", "source_split", "basal_rate"):
        if col in out:
            out[col] = out[col].ffill()
    if TARGET_OBSERVED in out:
        out[TARGET_OBSERVED] = out[TARGET_OBSERVED].fillna(False).astype(bool)
    if "cgm_gap_flag" in out:
        out["cgm_gap_flag"] = out["cgm_gap_flag"].fillna(1.)
    return out


def validate_observation_indicator(frame):
    if TARGET_OBSERVED not in frame:
        raise ValueError("Missing target_observed; regenerate with Stage A preprocessing")
    observed = frame[TARGET_OBSERVED]
    if observed.isna().any() or not observed.isin([0, 1, False, True]).all():
        raise ValueError("target_observed must be binary and non-missing")
    if not np.isfinite(frame.loc[observed.astype(bool), "glucose_mg_dl"]).all():
        raise ValueError("Observed CGM must be finite")
