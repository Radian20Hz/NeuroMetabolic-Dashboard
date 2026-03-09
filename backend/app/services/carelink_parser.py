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
    Handles Medtronic 780G real export format:
    - Metadata header lines before data
    - Separate Date and Time columns
    - Multiple header repetitions in file
    - Date format: YYYY/MM/DD
    """
    readings = []

    with open(filepath, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile)
        headers = None
        glucose_col = None
        date_col = None
        time_col = None

        for row in reader:
            if "Sensor Glucose (mg/dL)" in row:
                headers = row
                glucose_col = row.index("Sensor Glucose (mg/dL)")
                date_col = row.index("Date")
                time_col = row.index("Time")
                continue

            if headers is None:
                continue

            if glucose_col is None or date_col is None or time_col is None:
                continue

            try:
                if glucose_col >= len(row):
                    continue

                raw_glucose = row[glucose_col].strip()
                if not raw_glucose:
                    continue

                date_str = row[date_col].strip()
                time_str = row[time_col].strip()
                timestamp = datetime.strptime(
                    f"{date_str} {time_str}",
                    "%Y/%m/%d %H:%M:%S"
                )

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
            except (ValueError, KeyError, IndexError):
                continue

    return readings
