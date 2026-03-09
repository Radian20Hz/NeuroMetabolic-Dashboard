"""
Unit tests – CareLink CSV Parser
Uses real Medtronic 780G CareLink export format.
"""

import pytest
from datetime import datetime
from app.services.carelink_parser import parse_glucose_readings, GlucoseReading


def test_parse_valid_csv(tmp_path):
    csv_content = """Device: Medtronic 780G
Serial: ABC123
Index,Date,Time,BG Source,Sensor Glucose (mg/dL)
0,2026/03/06,08:00:00,,120
1,2026/03/06,08:05:00,,118
2,2026/03/06,08:10:00,,115
"""
    csv_file = tmp_path / "test_carelink.csv"
    csv_file.write_text(csv_content)

    readings = parse_glucose_readings(str(csv_file))

    assert len(readings) == 3
    assert readings[0].glucose_mg_dl == 120.0
    assert readings[0].timestamp == datetime(2026, 3, 6, 8, 0, 0)


def test_skip_invalid_rows(tmp_path):
    csv_content = """Index,Date,Time,BG Source,Sensor Glucose (mg/dL)
0,2026/03/06,08:00:00,,120
1,INVALID,INVALID,,not_a_number
2,2026/03/06,08:10:00,,115
"""
    csv_file = tmp_path / "test_carelink_invalid.csv"
    csv_file.write_text(csv_content)

    readings = parse_glucose_readings(str(csv_file))

    assert len(readings) == 2


def test_empty_csv(tmp_path):
    csv_content = """Index,Date,Time,BG Source,Sensor Glucose (mg/dL)
"""
    csv_file = tmp_path / "test_empty.csv"
    csv_file.write_text(csv_content)

    readings = parse_glucose_readings(str(csv_file))

    assert len(readings) == 0


def test_handle_high_low_values(tmp_path):
    csv_content = """Device Info
Patient: Test
Index,Date,Time,BG Source,Sensor Glucose (mg/dL)
0,2026/03/06,08:00:00,,Low
1,2026/03/06,08:05:00,,High
2,2026/03/06,08:10:00,,120
"""
    csv_file = tmp_path / "test_high_low.csv"
    csv_file.write_text(csv_content)

    readings = parse_glucose_readings(str(csv_file))

    assert len(readings) == 3
    assert readings[0].glucose_mg_dl == 39.0
    assert readings[1].glucose_mg_dl == 401.0
    assert readings[2].glucose_mg_dl == 120.0


def test_skip_metadata_headers(tmp_path):
    csv_content = """Device: Medtronic 780G
Serial: ABC123
Export Date: 2026-03-06
Index,Date,Time,BG Source,Sensor Glucose (mg/dL)
0,2026/03/06,08:00:00,,120
"""
    csv_file = tmp_path / "test_metadata.csv"
    csv_file.write_text(csv_content)

    readings = parse_glucose_readings(str(csv_file))

    assert len(readings) == 1
    assert readings[0].glucose_mg_dl == 120.0
