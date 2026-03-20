"""
app/api/clarke.py
=================
POST /api/v1/clarke  – Clarke Error Grid Analysis endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.services.clarke_egz import run_clarke_ega

router = APIRouter(prefix="/clarke", tags=["clarke"])


class ClarkeRequest(BaseModel):
    reference_values: list[float] = Field(
        ..., min_length=2, description="Reference (actual) glucose values mg/dL"
    )
    predicted_values: list[float] = Field(
        ..., min_length=2, description="Predicted glucose values mg/dL"
    )

    @model_validator(mode="after")
    def check_equal_length(self) -> "ClarkeRequest":
        if len(self.reference_values) != len(self.predicted_values):
            raise ValueError("reference_values and predicted_values must have equal length")
        return self


class ClarkePoint(BaseModel):
    reference: float
    predicted: float
    zone: str


class ClarkeResponse(BaseModel):
    total: int
    zone_counts: dict[str, int]
    zone_percents: dict[str, float]
    clinically_acceptable_percent: float
    points: list[ClarkePoint]


@router.post("", response_model=ClarkeResponse)
async def clarke_ega(request: ClarkeRequest) -> ClarkeResponse:
    """Run Clarke Error Grid Analysis on paired reference/predicted glucose values."""
    try:
        result = run_clarke_ega(request.reference_values, request.predicted_values)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clarke EGA error: {e}")

    return ClarkeResponse(
        total=result["total"],
        zone_counts=result["zone_counts"],
        zone_percents=result["zone_percents"],
        clinically_acceptable_percent=result["clinically_acceptable_percent"],
        points=[ClarkePoint(**p) for p in result["points"]],
    )
