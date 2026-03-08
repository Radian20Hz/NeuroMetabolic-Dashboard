"""
Glucose Validator – classifies glucose readings according to
ADA (American Diabetes Association) clinical standards.
"""

from enum import Enum
from dataclasses import dataclass


class GlucoseZone(Enum):
    SEVERE_HYPO="Severe_Hypoglycemia"
    HYPO="Hypoglycemia"
    NORMAL="Normal"
    HYPER="Hyperglycemia"
    SEVERE_HYPER="Severe_Hyperglycemia"


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
    TIR = percentage of readings in 70-180 mg/dL range.
    ADA target: > 70% TIR for T1D patients.
    """
    if not readings:
        return 0.0

    in_range = sum(1 for r in readings if 70 <= r <= 180)
    return round((in_range / len(readings)) * 100, 2)