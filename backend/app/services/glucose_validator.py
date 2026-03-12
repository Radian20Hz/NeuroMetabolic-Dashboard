"""
Glucose Validator – classifies glucose readings according to
ADA (American Diabetes Association) clinical standards.
"""

from enum import Enum
from dataclasses import dataclass


class GlucoseZone(Enum):
    SEVERE_HYPO = "severe_hypoglycemia"
    HYPO = "hypoglycemia"
    NORMAL = "normal"
    HYPER = "hyperglycemia"
    SEVERE_HYPER = "severe_hyperglycemia"


@dataclass
class ValidationResult:
    glucose_mg_dl: float
    zone: GlucoseZone
    is_critical: bool
    message: str


def classify_glucose(glucose_mg_dl: float) -> ValidationResult:
    """
    Classify a glucose reading into a clinical zone.
    Based on ADA 2024 standards for Time-in-Range targets.
    """
    if glucose_mg_dl < 54:
        return ValidationResult(
            glucose_mg_dl=glucose_mg_dl,
            zone=GlucoseZone.SEVERE_HYPO,
            is_critical=True,
            message="CRITICAL: Severe hypoglycemia. Immediate action required.",
        )
    elif glucose_mg_dl < 70:
        return ValidationResult(
            glucose_mg_dl=glucose_mg_dl,
            zone=GlucoseZone.HYPO,
            is_critical=True,
            message="WARNING: Hypoglycemia detected. Carbohydrate intake recommended.",
        )
    elif glucose_mg_dl <= 180:
        return ValidationResult(
            glucose_mg_dl=glucose_mg_dl,
            zone=GlucoseZone.NORMAL,
            is_critical=False,
            message="OK: Glucose within target range.",
        )
    elif glucose_mg_dl <= 250:
        return ValidationResult(
            glucose_mg_dl=glucose_mg_dl,
            zone=GlucoseZone.HYPER,
            is_critical=False,
            message="WARNING: Hyperglycemia detected. Monitor closely.",
        )
    else:
        return ValidationResult(
            glucose_mg_dl=glucose_mg_dl,
            zone=GlucoseZone.SEVERE_HYPER,
            is_critical=True,
            message="CRITICAL: Severe hyperglycemia. Medical attention may be required.",
        )


def calculate_time_in_range(readings: list[float]) -> float:
    """
    Calculate Time-in-Range (TIR) percentage.
    TIR = percentage of readings in 70–180 mg/dL range.
    ADA target: > 70% TIR for T1D patients.
    """
    if not readings:
        return 0.0
    in_range = sum(1 for r in readings if 70 <= r <= 180)
    return round((in_range / len(readings)) * 100, 2)


def calculate_gmi(avg_glucose: float) -> float:
    """
    Calculate Glucose Management Indicator (GMI).
    Estimates HbA1c from average CGM glucose.

    Formula: GMI (%) = 3.31 + 0.02392 × avg_glucose (mg/dL)
    Source: Bergenstal et al., Diabetes Care 2018.
    """
    return round(3.31 + 0.02392 * avg_glucose, 2)


def calculate_cv(std_dev: float, avg_glucose: float) -> float:
    """
    Calculate Coefficient of Variation (CV) as a percentage.
    CV = (std_dev / avg_glucose) × 100

    ADA target for glycemic stability: CV < 36%.
    """
    if avg_glucose == 0:
        return 0.0
    return round((std_dev / avg_glucose) * 100, 2)


def calculate_statistics(readings: list[float]) -> dict:
    """
    Calculate full glycemic statistics for a set of readings.
    Returns ADA-compliant metrics: TIR, GMI, CV, and descriptive stats.
    """
    if not readings:
        return {
            "count": 0,
            "min_glucose": None,
            "max_glucose": None,
            "avg_glucose": None,
            "std_dev": None,
            "time_in_range_percent": 0.0,
            "gmi": None,
            "cv_percent": None,
            "cv_is_stable": None,
        }

    count = len(readings)
    min_glucose = round(min(readings), 2)
    max_glucose = round(max(readings), 2)
    avg_glucose = round(sum(readings) / count, 2)

    variance = sum((r - avg_glucose) ** 2 for r in readings) / count
    std_dev = round(variance ** 0.5, 2)

    tir = calculate_time_in_range(readings)
    gmi = calculate_gmi(avg_glucose)
    cv = calculate_cv(std_dev, avg_glucose)

    return {
        "count": count,
        "min_glucose": min_glucose,
        "max_glucose": max_glucose,
        "avg_glucose": avg_glucose,
        "std_dev": std_dev,
        "time_in_range_percent": tir,
        "gmi": gmi,
        "cv_percent": cv,
        "cv_is_stable": cv < 36.0,
    }
