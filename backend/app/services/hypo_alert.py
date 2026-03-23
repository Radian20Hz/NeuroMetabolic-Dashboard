"""
hypo_alert.py
=============
Proactive hypoglycemia alert engine with 20-minute lead time.

Analyses TFT predictions and current glucose to produce a structured
alert that the /predict endpoint embeds in every response.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

HYPO_THRESHOLD = 70.0   # mg/dL — ADA clinical hypoglycemia
ALERT_WINDOW_MIN = 20   # look-ahead window for imminent alert


AlertLevel = Literal["NONE", "HYPO_NOW", "HYPO_IMMINENT"]


@dataclass(frozen=True)
class HypoAlert:
    level: AlertLevel
    message: str
    triggered_at_min: int | None  # minutes_ahead that triggered alert
    min_predicted_glucose: float | None

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "message": self.message,
            "triggered_at_min": self.triggered_at_min,
            "min_predicted_glucose": self.min_predicted_glucose,
        }


def evaluate_hypo_alert(
    last_known_glucose: float,
    predictions: list[dict],
) -> HypoAlert:
    """
    Evaluate hypoglycemia risk from current glucose and TFT predictions.

    Parameters
    ----------
    last_known_glucose : float
        Most recent CGM reading (mg/dL).
    predictions : list[dict]
        List of prediction dicts with keys:
        minutes_ahead, glucose_mg_dl, lower_mg_dl, upper_mg_dl.

    Returns
    -------
    HypoAlert
        Structured alert with level, message and trigger details.
    """
    # Check current glucose first
    if last_known_glucose < HYPO_THRESHOLD:
        return HypoAlert(
            level="HYPO_NOW",
            message=f"Current glucose {last_known_glucose:.0f} mg/dL is below "
                    f"{HYPO_THRESHOLD:.0f} mg/dL. Take action immediately.",
            triggered_at_min=0,
            min_predicted_glucose=last_known_glucose,
        )

    # Check predictions within alert window
    window = [
        p for p in predictions
        if p["minutes_ahead"] <= ALERT_WINDOW_MIN
    ]

    if not window:
        return HypoAlert(
            level="NONE",
            message="No hypoglycemia risk detected.",
            triggered_at_min=None,
            min_predicted_glucose=None,
        )

    # Use lower confidence bound for conservative alerting
    min_pred = min(window, key=lambda p: p["lower_mg_dl"])
    min_glucose = min_pred["lower_mg_dl"]
    trigger_min = min_pred["minutes_ahead"]

    if min_glucose < HYPO_THRESHOLD:
        return HypoAlert(
            level="HYPO_IMMINENT",
            message=f"Hypoglycemia predicted in ~{trigger_min} min "
                    f"(lower bound: {min_glucose:.0f} mg/dL). Consider preventive action.",
            triggered_at_min=trigger_min,
            min_predicted_glucose=round(min_glucose, 1),
        )

    return HypoAlert(
        level="NONE",
        message="No hypoglycemia risk detected.",
        triggered_at_min=None,
        min_predicted_glucose=None,
    )
