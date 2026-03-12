"""
InfluxDB Service – persists CGM glucose readings to time-series database.
"""
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from app.core.config import settings
from app.services.carelink_parser import GlucoseReading


class InfluxDBService:
    """Handles all InfluxDB read/write operations for CGM data."""

    def __init__(self) -> None:
        self.client = InfluxDBClient(
            url=settings.INFLUXDB_URL,
            token=settings.INFLUXDB_TOKEN,
            org=settings.INFLUXDB_ORG,
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()

    def write_glucose_readings(self, readings: list[GlucoseReading]) -> int:
        """
        Write a list of GlucoseReading objects to InfluxDB.
        Returns the number of successfully written points.
        """
        points = [
            Point("glucose_reading")
            .tag("units", reading.sensor_units)
            .field("glucose_mg_dl", reading.glucose_mg_dl)
            .time(reading.timestamp, WritePrecision.S)
            for reading in readings
        ]
        self.write_api.write(
            bucket=settings.INFLUXDB_BUCKET,
            org=settings.INFLUXDB_ORG,
            record=points,
        )
        return len(points)

    def get_latest_readings(self, hours: int = 24) -> list[dict]:
        """
        Query the last N hours of glucose readings from InfluxDB.
        Returns a list of dicts with timestamp and glucose value.
        """
        query = f"""
        from(bucket: "{settings.INFLUXDB_BUCKET}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r._measurement == "glucose_reading")
          |> filter(fn: (r) => r._field == "glucose_mg_dl")
          |> sort(columns: ["_time"])
        """
        tables = self.query_api.query(query, org=settings.INFLUXDB_ORG)
        return [
            {
                "timestamp": record.get_time().isoformat(),
                "glucose_mg_dl": record.get_value(),
            }
            for table in tables
            for record in table.records
        ]

    def close(self) -> None:
        """Close the InfluxDB client connection."""
        self.client.close()


# Module-level singleton — one client for the entire app lifetime.
# Injected into route handlers via FastAPI Depends(get_influxdb).
_influxdb_instance: InfluxDBService | None = None


def get_influxdb() -> InfluxDBService:
    """
    Dependency provider for InfluxDBService.
    Returns a shared singleton instance — never creates a new client per request.
    """
    global _influxdb_instance
    if _influxdb_instance is None:
        _influxdb_instance = InfluxDBService()
    return _influxdb_instance
