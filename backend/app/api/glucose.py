"""
Glucose API endpoints – handles CGM data ingestion and retrieval.
"""
import os
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.models.glucose import (
    ClassifyRequest,
    ClassifyResponse,
    GlucoseStatisticsResponse,
    LatestReadingsResponse,
    UploadResponse,
)
from app.services.carelink_parser import parse_glucose_readings
from app.services.carelink_scraper import CareLinkScraper
from app.services.glucose_validator import calculate_statistics, classify_glucose
from app.services.influxdb_service import InfluxDBService, get_influxdb

router = APIRouter(prefix="/glucose", tags=["glucose"])


def _require_csv(file: UploadFile) -> None:
    """Raise 400 if uploaded file is not a CSV."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")


async def _write_to_tmp(file: UploadFile) -> str:
    """Persist upload to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
        tmp.write(await file.read())
        return tmp.name


def _build_upload_response(count: int, stats: dict) -> UploadResponse:
    """Construct UploadResponse from write count and stats dict."""
    return UploadResponse(
        status="success",
        readings_saved=count,
        min_glucose=stats["min_glucose"],
        max_glucose=stats["max_glucose"],
        avg_glucose=stats["avg_glucose"],
        std_dev=stats["std_dev"],
        time_in_range_percent=stats["time_in_range_percent"],
        gmi=stats["gmi"],
        cv_percent=stats["cv_percent"],
        cv_is_stable=stats["cv_is_stable"],
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_carelink_csv(
    file: UploadFile = File(...),
    db: InfluxDBService = Depends(get_influxdb),
):
    """
    Upload a CareLink CSV export file.
    Parses glucose readings, stores them in InfluxDB,
    and returns full glycemic statistics including GMI and CV.
    """
    _require_csv(file)
    tmp_path = await _write_to_tmp(file)

    try:
        readings = parse_glucose_readings(tmp_path)
        if not readings:
            raise HTTPException(
                status_code=422,
                detail="No valid glucose readings found in the file.",
            )
        count = db.write_glucose_readings(readings)
        stats = calculate_statistics([r.glucose_mg_dl for r in readings])
        return _build_upload_response(count, stats)
    finally:
        os.unlink(tmp_path)


@router.get("/latest", response_model=LatestReadingsResponse)
async def get_latest_readings(
    hours: int = 24,
    db: InfluxDBService = Depends(get_influxdb),
):
    """
    Get the latest glucose readings from InfluxDB.
    Default: last 24 hours. Max: 168 hours (7 days).
    """
    if hours < 1 or hours > 168:
        raise HTTPException(
            status_code=400,
            detail="Hours must be between 1 and 168 (7 days).",
        )
    readings = db.get_latest_readings(hours=hours)
    return {
        "status": "success",
        "hours_requested": hours,
        "count": len(readings),
        "readings": readings,
    }


@router.post("/classify", response_model=ClassifyResponse)
async def classify_glucose_reading(request: ClassifyRequest):
    """
    Classify a single glucose reading into an ADA 2024 clinical zone.
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
    Returns TIR, GMI, CV, std_dev and descriptive stats.
    """
    _require_csv(file)
    tmp_path = await _write_to_tmp(file)

    try:
        readings = parse_glucose_readings(tmp_path)
        if not readings:
            raise HTTPException(
                status_code=422,
                detail="No valid glucose readings found in file.",
            )
        stats = calculate_statistics([r.glucose_mg_dl for r in readings])
        return GlucoseStatisticsResponse(**stats)
    finally:
        os.unlink(tmp_path)


@router.post("/scrape", response_model=UploadResponse)
async def scrape_carelink(db: InfluxDBService = Depends(get_influxdb)):
    """
    Fetch latest glucose readings directly from CareLink EU API.
    Requires CARELINK_USERNAME and CARELINK_PASSWORD in .env
    """
    scraper = CareLinkScraper()
    if not scraper.authenticate():
        raise HTTPException(
            status_code=401,
            detail="CareLink authentication failed. Check credentials in .env",
        )

    try:
        readings = scraper.fetch_recent_readings()
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if not readings:
        raise HTTPException(
            status_code=422,
            detail="No glucose readings returned from CareLink.",
        )

    count = db.write_glucose_readings(readings)
    stats = calculate_statistics([r.glucose_mg_dl for r in readings])
    return _build_upload_response(count, stats)
