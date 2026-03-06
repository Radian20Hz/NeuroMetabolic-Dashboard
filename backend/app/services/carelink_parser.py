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
    Handles metadata headers and sensor threshold values (High/Low).
    Returns a list of GlucoseReading objects.
    """
    readings = []

    with open(filepath, newline="", encoding="utf-8") as csvfile:

        for line in csvfile:
            if "Timestamp" in line:
                break

        reader = csv.DictReader(csvfile, fieldnames=[
            "Timestamp (YYYY-MM-DDThh:mm:ss)",
            "Sensor Glucose (mg/dL)",
        ])

        for row in reader:
            try:
                timestamp = datetime.strptime(
                    row["Timestamp (YYYY-MM-DDThh:mm:ss)"],
                    "%Y-%m-%dT%H:%M:%S"
                )

                raw_glucose = row["Sensor Glucose (mg/dL)"]
                if raw_glucose == "Low":
                    glucose = 39.0
                elif raw_glucose == "High":
                    glucose = 401.0
                else:
                    glucose = float(raw_glucose)

                readings.append(
                    GlucoseReading(
                        timestamp=timestamp,
                        glucose_mg_dl=glucose,
                    )
                )
            except (ValueError, KeyError):
                continue

    return readings
