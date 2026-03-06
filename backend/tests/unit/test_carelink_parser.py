"""
Unit tests – CareLink CSV Parser
"""

import pytest
from datetime import datetime
from app.services.carelink_parser import parse_glucose_readings, GlucoseReading


def test_parse_valid_csv(tmp_path):
    csv_content = """Timestamp (YYYY-MM-DDThh:mm:ss),Sensor Glucose (mg/dL)
2026-03-06T08:00:00,120.0
2026-03-06T08:05:00,118.5
2026-03-06T08:10:00,115.0
"""
    csv_file = tmp_path / "test_carelink.csv"
    csv_file.write_text(csv_content)

    readings = parse_glucose_readings(str(csv_file))

    assert len(readings) == 3
    assert readings[0].glucose_mg_dl == 120.0
    assert readings[0].timestamp == datetime(2026, 3, 6, 8, 0, 0)


def test_skip_invalid_rows(tmp_path):
    csv_content = """Timestamp (YYYY-MM-DDThh:mm:ss),Sensor Glucose (mg/dL)
2026-03-06T08:00:00,120.0
INVALID_ROW,not_a_number
2026-03-06T08:10:00,115.0
"""
    csv_file = tmp_path / "test_carelink_invalid.csv"
    csv_file.write_text(csv_content)

    readings = parse_glucose_readings(str(csv_file))

    assert len(readings) == 2


def test_empty_csv(tmp_path):
    csv_content = """Timestamp (YYYY-MM-DDThh:mm:ss),Sensor Glucose (mg/dL)
"""
    csv_file = tmp_path / "test_empty.csv"
    csv_file.write_text(csv_content)

    readings = parse_glucose_readings(str(csv_file))

    assert len(readings) == 0