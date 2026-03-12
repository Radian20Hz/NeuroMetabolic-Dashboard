"""
Unit tests – Glucose API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.services.influxdb_service import get_influxdb

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CSV_TWO_ROWS = (
    b"Index,Date,Time,BG Source,Sensor Glucose (mg/dL)\n"
    b"0,2026/03/06,08:00:00,,120\n"
    b"1,2026/03/06,08:05:00,,118\n"
)

VALID_CSV_FOUR_ROWS = (
    b"Index,Date,Time,BG Source,Sensor Glucose (mg/dL)\n"
    b"0,2026/03/06,08:00:00,,120\n"
    b"1,2026/03/06,08:05:00,,80\n"
    b"2,2026/03/06,08:10:00,,50\n"
    b"3,2026/03/06,08:15:00,,200\n"
)

EMPTY_CSV = b"Index,Date,Time,BG Source,Sensor Glucose (mg/dL)\n"


def _mock_db(write_return: int = 0) -> MagicMock:
    """Return a MagicMock InfluxDBService with configurable write return."""
    mock = MagicMock()
    mock.write_glucose_readings.return_value = write_return
    return mock


# ---------------------------------------------------------------------------
# /upload
# ---------------------------------------------------------------------------

def test_upload_rejects_non_csv():
    response = client.post(
        "/api/v1/glucose/upload",
        files={"file": ("data.txt", b"some content", "text/plain")},
    )
    assert response.status_code == 400
    assert "CSV" in response.json()["detail"]


def test_upload_rejects_empty_csv():
    response = client.post(
        "/api/v1/glucose/upload",
        files={"file": ("data.csv", EMPTY_CSV, "text/csv")},
    )
    assert response.status_code == 422


def test_upload_valid_csv_returns_count():
    mock_db = _mock_db(write_return=2)
    app.dependency_overrides[get_influxdb] = lambda: mock_db

    response = client.post(
        "/api/v1/glucose/upload",
        files={"file": ("data.csv", VALID_CSV_TWO_ROWS, "text/csv")},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert data["readings_saved"] == 2
    assert data["status"] == "success"
    assert data["gmi"] is not None
    assert data["cv_percent"] is not None
    assert isinstance(data["cv_is_stable"], bool)


# ---------------------------------------------------------------------------
# /latest
# ---------------------------------------------------------------------------

def test_get_latest_readings_default():
    mock_db = _mock_db()
    mock_db.get_latest_readings.return_value = [
        {"timestamp": "2026-03-06T08:00:00+00:00", "glucose_mg_dl": 120.0}
    ]
    app.dependency_overrides[get_influxdb] = lambda: mock_db

    response = client.get("/api/v1/glucose/latest")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["hours_requested"] == 24


def test_get_latest_readings_invalid_hours():
    response = client.get("/api/v1/glucose/latest?hours=999")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# /classify
# ---------------------------------------------------------------------------

def test_classify_normal_glucose():
    response = client.post(
        "/api/v1/glucose/classify",
        json={"glucose_mg_dl": 120.0},
    )
    assert response.status_code == 200
    assert response.json()["zone"] == "normal"
    assert response.json()["is_critical"] is False


def test_classify_hypoglycemia():
    response = client.post(
        "/api/v1/glucose/classify",
        json={"glucose_mg_dl": 60.0},
    )
    assert response.status_code == 200
    assert response.json()["zone"] == "hypoglycemia"
    assert response.json()["is_critical"] is True


def test_classify_invalid_glucose():
    response = client.post(
        "/api/v1/glucose/classify",
        json={"glucose_mg_dl": 9999.0},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /statistics
# ---------------------------------------------------------------------------

def test_statistics_valid_csv():
    response = client.post(
        "/api/v1/glucose/statistics",
        files={"file": ("data.csv", VALID_CSV_FOUR_ROWS, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 4
    assert data["min_glucose"] == 50.0
    assert data["max_glucose"] == 200.0
    assert data["time_in_range_percent"] == 50.0
    assert data["gmi"] is not None
    assert data["cv_percent"] is not None
    assert isinstance(data["cv_is_stable"], bool)


def test_statistics_rejects_non_csv():
    response = client.post(
        "/api/v1/glucose/statistics",
        files={"file": ("data.txt", b"some content", "text/plain")},
    )
    assert response.status_code == 400


def test_statistics_rejects_empty_csv():
    response = client.post(
        "/api/v1/glucose/statistics",
        files={"file": ("data.csv", EMPTY_CSV, "text/csv")},
    )
    assert response.status_code == 422
