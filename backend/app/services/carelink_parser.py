"""
CareLink CSV Parser – ETL fallback layer
Parses exported Medtronic CareLink CSV files into structured data.
"""

import csv
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GlucoseReading:
    timestamp: datetime
    glucose_mg_dl: float
    sensor_units: str = "mg/dL"


def parse_glucose_readings(filepath: str) -> list[GlucoseReading]:
    """
    Parse CGM glucose readings from a CareLink CSV export file.
    Returns a list of GlucoseReading objects.
    """
    readings = []

    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            try:
                timestamp = datetime.strptime(
                    row["Timestamp (YYYY-MM-DDThh:mm:ss)"],
                    "%Y-%m-%dT%H:%M:%S"
                )
                glucose = float(row["Sensor Glucose (mg/dL)"])

                readings.append(
                    GlucoseReading(
                        timestamp=timestamp,
                        glucose_mg_dl=glucose,
                    )
                )
            except (ValueError, KeyError):
                continue

    return readings
