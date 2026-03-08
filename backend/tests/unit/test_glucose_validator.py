"""
Unit tests – Glucose Validator
Tests clinical classification based on ADA 2024 standards.
"""

from app.services.glucose_validator import (
    classify_glucose,
    calculate_time_in_range,
    GlucoseZone,
)


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


def test_time_in_range_all_normal():
    readings = [80.0, 100.0, 120.0, 150.0, 170.0]
    assert calculate_time_in_range(readings) == 100.0


def test_time_in_range_mixed():
    readings = [50.0, 100.0, 120.0, 300.0]
    assert calculate_time_in_range(readings) == 50.0


def test_time_in_range_empty():
    assert calculate_time_in_range([]) == 0.0
