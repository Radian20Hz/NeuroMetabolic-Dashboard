"""
Glucose API endpoints – handles CGM data ingestion and retrieval.
"""

from app.services.carelink_scraper import CareLinkScraper
from app.services.glucose_validator import classify_glucose, calculate_statistics
from app.models.glucose import (
    ClassifyRequest,
    ClassifyResponse,
    GlucoseStatisticsResponse,
    UploadResponse,
    LatestReadingsResponse,
)
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.carelink_parser import parse_glucose_readings
from app.services.influxdb_service import InfluxDBService
import tempfile
import os

router = APIRouter(prefix="/glucose", tags=["glucose"])


@router.post("/upload", response_model=UploadResponse)
async def upload_carelink_csv(file: UploadFile = File(...)):
    """
    Upload a CareLink CSV export file.
    Parses glucose readings, stores them in InfluxDB,
    and returns full glycemic statistics.
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

        glucose_values = [r.glucose_mg_dl for r in readings]
        stats = calculate_statistics(glucose_values)

        return UploadResponse(
            status="success",
            readings_saved=count,
            min_glucose=stats["min_glucose"],
            max_glucose=stats["max_glucose"],
            avg_glucose=stats["avg_glucose"],
            std_dev=stats["std_dev"],
            time_in_range_percent=stats["time_in_range_percent"],
        )

    finally:
        os.unlink(tmp_path)


@router.get("/latest", response_model=LatestReadingsResponse)
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


@router.post("/classify", response_model=ClassifyResponse)
async def classify_glucose_reading(request: ClassifyRequest):
    """
    Classify a glucose reading into a clinical zone.
    Based on ADA 2024 standards.
    """
    result = classify_glucose(request.glucose_mg_dl)

    return ClassifyResponse(
        glucose_mg_dl=result.glucose_mg_dl,
        zone=result.zone.value,
        is_critical=result.is_critical,
        message=result.message,
    )


@router.post("/statistics", response_model=GlucoseStatisticsResponse)
async def get_glucose_statistics(file: UploadFile = File(...)):
    """
    Upload a CareLink CSV and get full glycemic statistics.
    Returns min, max, avg, std_dev and Time-in-Range.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted."
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
                detail="No valid glucose readings found in file."
            )

        glucose_values = [r.glucose_mg_dl for r in readings]
        stats = calculate_statistics(glucose_values)

        return GlucoseStatisticsResponse(**stats)

    finally:
        os.unlink(tmp_path)


@router.post("/scrape", response_model=UploadResponse)
async def scrape_carelink():
    """
    Fetch latest glucose readings directly from CareLink EU API.
    Requires CARELINK_USERNAME and CARELINK_PASSWORD in .env
    """
    scraper = CareLinkScraper()

    authenticated = scraper.authenticate()
    if not authenticated:
        raise HTTPException(
            status_code=401,
            detail="CareLink authentication failed. Check credentials in .env"
        )

    try:
        readings = scraper.fetch_recent_readings()
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if not readings:
        raise HTTPException(
            status_code=422,
            detail="No glucose readings returned from CareLink."
        )

    service = InfluxDBService()
    count = service.write_glucose_readings(readings)
    service.close()

    glucose_values = [r.glucose_mg_dl for r in readings]
    stats = calculate_statistics(glucose_values)

    return UploadResponse(
        status="success",
        readings_saved=count,
        min_glucose=stats["min_glucose"],
        max_glucose=stats["max_glucose"],
        avg_glucose=stats["avg_glucose"],
        std_dev=stats["std_dev"],
        time_in_range_percent=stats["time_in_range_percent"],
    )
