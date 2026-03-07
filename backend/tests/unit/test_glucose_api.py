"""
Unit tests – Glucose API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)


def test_upload_rejects_non_csv():
    response = client.post(
        "/api/v1/glucose/upload",
        files={"file": ("data.txt",b"some content", "text/plain")},
    )
    assert response.status_code == 400
    assert "CSV" in response.json()["detail"]



def test_upload_rejects_empty_csv():
    csv_content = b"Timestamp (YYYY-MM-DDThh:mm:ss),Sensor Glucose (mg/dL)\n"

    response = client.post(
        "/api/v1/glucose/upload",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 422


@patch("app.api.glucose.InfluxDBService")
def test_upload_valid_csv_returns_count(mock_influx):
    mock_service = MagicMock()
    mock_service.write_glucose_readings.return_value = 2
    mock_influx.return_value = mock_service

    csv_content = (
        b"Timestamp (YYYY-MM-DDThh:mm:ss),Sensor Glucose (mg/dL)\n"
        b"2026-03-06T08:00:00,120.0\n"
        b"2026-03-06T08:05:00,118.5\n"
    )

    response = client.post(
        "/api/v1/glucose/upload",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["readings_saved"] == 2
    assert response.json()["status"] == "success"


@patch("app.api.glucose.InfluxDBService")
def test_get_latest_readings_default(mock_influx):
    mock_service = MagicMock()
    mock_service.get_latest_readings.return_value = [
        {"timestamp": "2026-03-06T08:00:00", "glucose_mg_dl": 120.0}
    ]
    mock_influx.return_value = mock_service

    response = client.get("/api/v1/glucose/latest")

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["hours_requested"] == 24


def test_get_latest_readings_invalid_hours():
    response = client.get("/api/v1/glucose/latest?hours=999")
    assert response.status_code == 400
