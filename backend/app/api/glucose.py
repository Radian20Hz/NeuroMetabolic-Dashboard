"""
Glucose API endpoints – handles CGM data ingestion and retrieval.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.carelink_parser import parse_glucose_readings
from app.services.influxdb_service import InfluxDBService
import tempfile
import os

router = APIRouter(prefix="/glucose", tags=["glucose"])


@router.post("/upload")
async def upload_carelink_data(file: UploadFile = File(...)):
    """
    Upload a CareLink CSV export file.
    Parses glucose readings and stores them in InfluxDB.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted.",
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv",
        mode="wb"
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        readings = parse_glucose_readings(tmp_path)

        if not readings:
            raise HTTPException(
                status_code=422,
                detail="No valid glucose readings found in the file.",
            )

        service = InfluxDBService()
        count = service.write_glucose_readings(readings)
        service.close()

        return {
            "status": "success",
            "readings_saved": count,
        }

    finally:
        os.unlink(tmp_path)


@router.get("/latest")
async def get_latest_readings(hours: int = 24):
    """
    Get the latest glucose readings from InfluxDB.
    Default: last 24 hours.
    """
    if hours < 1 or hours > 168:
        raise HTTPException(
            status_code=400,
            detail="Hours must be between 1 and 168 (7 days)."
        )

    service = InfluxDBService()
    readings = service.get_latest_readings(hours=hours)
    service.close()

    return {
        "status": "success",
        "hours_requested": hours,
        "count": len(readings),
        "readings": readings,
    }
