"""
clarke_egz.py
=============
Clarke Error Grid Analysis (EGA) for continuous glucose monitoring.
Classifies (reference, predicted) pairs into zones A–E per the original
Clarke et al. 1987 specification.

Zone A – clinically accurate (within 20% or both ≤70 mg/dL)
Zone B – benign errors (outside A but no treatment change required)
Zone C – overcorrection errors
Zone D – dangerous failure to detect hypo/hyperglycaemia
Zone E – erroneous treatment errors
"""
from __future__ import annotations


def classify_point(ref: float, pred: float) -> str:
    """Classify a single (reference, predicted) pair into Clarke zone A-E."""
    # Zone A: within 20% of reference, or both in hypoglycaemic range
    if ref <= 70 and pred <= 70:
        return "A"
    if ref >= 70:
        upper = ref * 1.20
        lower = ref * 0.80
        if lower <= pred <= upper:
            return "A"

    # Zone E (must check before D)
    if ref <= 70 and pred >= 180:
        return "E"
    if ref >= 180 and pred <= 70:
        return "E"

    # Zone D
    if ref >= 240 and 70 <= pred <= 180:
        return "D"
    if pred <= 70 and ref >= 180:
        return "D"

    # Zone C
    if ref >= 70 and pred >= ref * 1.20 and pred >= 180:
        return "C"
    if ref <= 180 and pred <= ref * 0.80 and pred <= 70:
        return "C"

    # Zone B (everything else outside A)
    return "B"


def run_clarke_ega(
    reference_values: list[float],
    predicted_values: list[float],
) -> dict:
    """
    Run Clarke EGA on paired lists of reference and predicted glucose values.

    Returns a dict with:
      - points: list of {reference, predicted, zone}
      - zone_counts: {A: n, B: n, C: n, D: n, E: n}
      - zone_percents: {A: %, B: %, ...}
      - total: int
      - clinically_acceptable_percent: float  (zones A+B)
    """
    if len(reference_values) != len(predicted_values):
        raise ValueError(
            "reference and predicted lists must have equal length")
    if not reference_values:
        raise ValueError("empty input")

    points = []
    zone_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    for ref, pred in zip(reference_values, predicted_values):
        zone = classify_point(float(ref), float(pred))
        points.append({"reference": ref, "predicted": pred, "zone": zone})
        zone_counts[zone] += 1

    total = len(points)
    zone_percents = {z: round(n / total * 100, 1)
                     for z, n in zone_counts.items()}
    acceptable = zone_counts["A"] + zone_counts["B"]

    return {
        "points": points,
        "zone_counts": zone_counts,
        "zone_percents": zone_percents,
        "total": total,
        "clinically_acceptable_percent": round(acceptable / total * 100, 1),
    }
