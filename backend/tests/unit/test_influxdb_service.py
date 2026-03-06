"""
Unit tests – InfluxDB Service
Uses mocking to avoid requiring a real InfluxDB instance.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

from app.services.carelink_parser import GlucoseReading
from app.services.influxdb_service import InfluxDBService


@patch("app.services.influxdb_service.InfluxDBClient")
def test_write_glucose_readings_returns_correct_count(mock_client):
    mock_write_api = MagicMock()
    mock_client.return_value.write_api.return_value = mock_write_api
    mock_client.return_value.query_api.return_value = MagicMock()

    service = InfluxDBService()

    readings = [
        GlucoseReading(
            timestamp=datetime(2026, 3, 6, 8, 0, 0),
            glucose_mg_dl=120.0,
        ),
        GlucoseReading(
            timestamp=datetime(2026, 3, 6, 8, 5, 0),
            glucose_mg_dl=118.5,
        ),
    ]
    
    result = service.write_glucose_readings(readings)

    assert result == 2
    assert mock_write_api.write.called


@patch("app.services.influxdb_service.InfluxDBClient")
def test_write_empty_list_returns_zero(mock_client):
    mock_write_api = MagicMock()
    mock_client.return_value.write_api.return_value = mock_write_api
    mock_client.return_value.query_api.return_value = MagicMock()

    service = InfluxDBService()
    result = service.write_glucose_readings([])

    assert result == 0
    