"""
Unit tests – Glucose Validator
Tests clinical classification based on ADA 2024 standards.
"""

from app.services.glucose_validator import (
    GlucoseZone,
    calculate_cv,
    calculate_gmi,
    calculate_statistics,
    calculate_time_in_range,
    classify_glucose,
)


# ---------------------------------------------------------------------------
# classify_glucose
# ---------------------------------------------------------------------------

def test_severe_hypoglycemia():
    result = classify_glucose(40.0)
    assert result.zone == GlucoseZone.SEVERE_HYPO
    assert result.is_critical is True


def test_hypoglycemia():
    result = classify_glucose(60.0)
    assert result.zone == GlucoseZone.HYPO
    assert result.is_critical is True


def test_normal_range():
    result = classify_glucose(120.0)
    assert result.zone == GlucoseZone.NORMAL
    assert result.is_critical is False


def test_hyperglycemia():
    result = classify_glucose(200.0)
    assert result.zone == GlucoseZone.HYPER
    assert result.is_critical is False


def test_severe_hyperglycemia():
    result = classify_glucose(300.0)
    assert result.zone == GlucoseZone.SEVERE_HYPER
    assert result.is_critical is True


def test_boundary_hypo_normal():
    assert classify_glucose(69.9).zone == GlucoseZone.HYPO
    assert classify_glucose(70.0).zone == GlucoseZone.NORMAL


def test_boundary_normal_hyper():
    assert classify_glucose(180.0).zone == GlucoseZone.NORMAL
    assert classify_glucose(180.1).zone == GlucoseZone.HYPER


# ---------------------------------------------------------------------------
# calculate_time_in_range
# ---------------------------------------------------------------------------

def test_time_in_range_all_normal():
    readings = [80.0, 100.0, 120.0, 150.0, 170.0]
    assert calculate_time_in_range(readings) == 100.0


def test_time_in_range_mixed():
    readings = [50.0, 100.0, 120.0, 300.0]
    assert calculate_time_in_range(readings) == 50.0


def test_time_in_range_empty():
    assert calculate_time_in_range([]) == 0.0


# ---------------------------------------------------------------------------
# calculate_gmi
# ---------------------------------------------------------------------------

def test_gmi_known_value():
    # avg 154 mg/dL → GMI = 3.31 + 0.02392 * 154 = 7.993...
    result = calculate_gmi(154.0)
    assert result == round(3.31 + 0.02392 * 154.0, 2)


def test_gmi_low_glucose():
    result = calculate_gmi(70.0)
    assert result == round(3.31 + 0.02392 * 70.0, 2)


# ---------------------------------------------------------------------------
# calculate_cv
# ---------------------------------------------------------------------------

def test_cv_known_value():
    # std_dev=30, avg=150 → CV = 20.0%
    assert calculate_cv(30.0, 150.0) == 20.0


def test_cv_stable_below_36():
    assert calculate_cv(30.0, 150.0) < 36.0


def test_cv_unstable_above_36():
    assert calculate_cv(60.0, 120.0) > 36.0


def test_cv_zero_avg_returns_zero():
    assert calculate_cv(10.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# calculate_statistics
# ---------------------------------------------------------------------------

def test_statistics_empty():
    result = calculate_statistics([])
    assert result["count"] == 0
    assert result["gmi"] is None
    assert result["cv_percent"] is None
    assert result["cv_is_stable"] is None


def test_statistics_full():
    readings = [80.0, 120.0, 160.0, 200.0]
    result = calculate_statistics(readings)
    assert result["count"] == 4
    assert result["min_glucose"] == 80.0
    assert result["max_glucose"] == 200.0
    assert result["gmi"] is not None
    assert result["cv_percent"] is not None
    assert isinstance(result["cv_is_stable"], bool)


def test_statistics_cv_is_stable_flag():
    # tight readings → should be stable
    readings = [100.0, 105.0, 102.0, 98.0]
    result = calculate_statistics(readings)
    assert result["cv_is_stable"] is True
