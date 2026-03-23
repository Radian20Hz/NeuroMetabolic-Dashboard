"""
subject_matcher.py
==================
Zero-shot subject matching for multi-patient TFT inference.

When a new user's glucose data is provided, this module selects the
closest OhioT1DM training subject based on Euclidean distance across
three normalized glycemic features: avg glucose, std dev, and TIR.

This allows inference on unseen patients without retraining.
"""
from __future__ import annotations

import math

# OhioT1DM subject profiles computed from training.parquet
# Format: subject_id -> (avg_glucose, std_dev, tir_percent)
SUBJECT_PROFILES: dict[str, tuple[float, float, float]] = {
    "559": (166.5, 70.5, 56.4),
    "563": (146.0, 49.7, 74.2),
    "570": (187.6, 62.3, 43.0),
    "575": (141.9, 60.4, 69.2),
    "588": (164.9, 50.5, 63.7),
    "591": (155.9, 58.1, 64.2),
    "wiktor": (136.2, 47.2, 82.6),
}

# Normalization ranges for feature scaling
_AVG_RANGE = (136.2, 187.6)
_STD_RANGE = (47.2, 70.5)
_TIR_RANGE = (43.0, 82.6)


def _normalize(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return (value - low) / (high - low)


def find_closest_subject(
    avg_glucose: float,
    std_dev: float,
    tir_percent: float,
) -> tuple[str, float]:
    """
    Find the OhioT1DM subject whose glycemic profile is closest
    to the given statistics using normalized Euclidean distance.

    Parameters
    ----------
    avg_glucose : float
        Mean glucose value (mg/dL)
    std_dev : float
        Standard deviation of glucose (mg/dL)
    tir_percent : float
        Time in range 70-180 mg/dL (%)

    Returns
    -------
    (subject_id, distance) : tuple[str, float]
        Best matching subject ID and its normalized distance score.
    """
    user = (
        _normalize(avg_glucose, *_AVG_RANGE),
        _normalize(std_dev, *_STD_RANGE),
        _normalize(tir_percent, *_TIR_RANGE),
    )

    best_id = ""
    best_dist = float("inf")

    for sid, (avg, std, tir) in SUBJECT_PROFILES.items():
        profile = (
            _normalize(avg, *_AVG_RANGE),
            _normalize(std, *_STD_RANGE),
            _normalize(tir, *_TIR_RANGE),
        )
        dist = math.sqrt(sum((u - p) ** 2 for u, p in zip(user, profile)))
        if dist < best_dist:
            best_dist = dist
            best_id = sid

    return best_id, round(best_dist, 4)


def match_subject_from_readings(glucose_values: list[float]) -> tuple[str, float]:
    """
    Convenience wrapper: compute stats from raw glucose values
    and return the closest matching subject.

    Parameters
    ----------
    glucose_values : list[float]
        Recent CGM readings (mg/dL)

    Returns
    -------
    (subject_id, distance) : tuple[str, float]
    """
    if not glucose_values:
        raise ValueError("glucose_values must not be empty")

    n = len(glucose_values)
    avg = sum(glucose_values) / n
    variance = sum((x - avg) ** 2 for x in glucose_values) / n
    std = math.sqrt(variance)
    tir = sum(1 for x in glucose_values if 70 <= x <= 180) / n * 100

    return find_closest_subject(avg, std, tir)
