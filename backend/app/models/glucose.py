"""
Pydantic models – data schemas for Glucose API requests and responses.
"""

from pydantic import BaseModel, Field
from datetime import datetime


class GlucoseReadingResponse(BaseModel):
    timestamp: datetime
    glucose_mg_dl: float = Field(..., ge=0.0, le=600.0)
    units: str = "mg/dl"


class UploadResponse(BaseModel):
    status: str
    readings_saved: int = Field(..., ge=0)


class LatestReadingsResponse(BaseModel):
    status: str
    hours_requested: int = Field(..., ge=1, le=168)
    count: int = Field(..., ge=0)
    readings: list[GlucoseReadingResponse]


class GlucoseStats(BaseModel):
    min_glucose: float = Field(..., ge=0.0)
    max_glucose: float = Field(..., ge=0.0)
    avg_glucose: float = Field(..., ge=0.0)
    readings_count: int = Field(..., ge=0)
    time_in_range_percent: float = Field(..., ge=0.0, le=100.0)
