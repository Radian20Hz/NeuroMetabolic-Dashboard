"""
app/api/predict.py
==================
POST /api/v1/predict  – 60-minute glucose forecast via TFT model.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

router = APIRouter(prefix="/predict", tags=["prediction"])

VALID_SUBJECT_IDS = {"559", "563", "570", "575", "588", "591"}


class PredictRequest(BaseModel):
    glucose_mg_dl: list[float] = Field(
        ...,
        min_length=24,
        max_length=48,
        description="Recent glucose readings (mg/dL), 5-min intervals, most-recent last.",
        examples=[[120.0, 122.0, 125.0, 118.0]],
    )
    bolus_last_1h: Optional[list[float]] = Field(None)
    basal_rate: Optional[list[float]] = Field(None)
    carbs_last_1h: Optional[list[float]] = Field(None)
    subject_id: str = Field(
        "559",
        description=f"Patient identifier. Valid: {', '.join(sorted(VALID_SUBJECT_IDS))}.",
    )

    @field_validator("glucose_mg_dl")
    @classmethod
    def validate_glucose_range(cls, v: list[float]) -> list[float]:
        for val in v:
            if not (20.0 <= val <= 600.0):
                raise ValueError(f"Glucose value {val} mg/dL out of range (20–600).")
        return v

    @field_validator("subject_id")
    @classmethod
    def validate_subject_id(cls, v: str) -> str:
        if v not in VALID_SUBJECT_IDS:
            valid = ', '.join(sorted(VALID_SUBJECT_IDS))
            raise ValueError(f"Unknown subject_id '{v}'. Valid: {valid}")
        return v


class PredictionPoint(BaseModel):
    minutes_ahead: int
    glucose_mg_dl: float
    lower_mg_dl: float
    upper_mg_dl: float


class PredictResponse(BaseModel):
    status: str = "success"
    subject_id: str
    horizon_steps: int
    predictions: list[PredictionPoint]
    last_known_glucose: float
    model_version: str = "TFT-Phase3-epoch47"


@router.post("", response_model=PredictResponse)
async def predict_glucose(request: PredictRequest) -> PredictResponse:
    """
    Predict the next 60 minutes of glucose (12 × 5-min steps).
    Returns median + 80% confidence interval from TFT model.
    """
    try:
        from app.services.tft_inference import predict_from_history
        result = predict_from_history(
            glucose_values=request.glucose_mg_dl,
            bolus_values=request.bolus_last_1h,
            basal_values=request.basal_rate,
            carbs_values=request.carbs_last_1h,
            subject_id=request.subject_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    predictions = [
        PredictionPoint(
            minutes_ahead=result["horizon_minutes"][i],
            glucose_mg_dl=result["predictions_mg_dl"][i],
            lower_mg_dl=result["lower_mg_dl"][i],
            upper_mg_dl=result["upper_mg_dl"][i],
        )
        for i in range(len(result["horizon_minutes"]))
    ]

    return PredictResponse(
        subject_id=request.subject_id,
        horizon_steps=len(predictions),
        predictions=predictions,
        last_known_glucose=request.glucose_mg_dl[-1],
    )
