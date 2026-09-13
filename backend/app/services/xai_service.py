"""
xai_service.py
==============
Explainability layer for TFT inference.

Extracts attention weights and variable importance scores from a
trained Temporal Fusion Transformer and returns them as plain dicts
suitable for JSON serialisation and frontend visualisation.
"""
from __future__ import annotations

import numpy as np


ENCODER_FEATURE_LABELS = {
    "glucose_mg_dl": "Glucose",
    "glucose_delta_1": "Δ Glucose (5 min)",
    "glucose_delta_3": "Δ Glucose (15 min)",
    "bolus_last_1h": "Bolus (last 1h)",
    "basal_rate": "Basal Rate",
    "carbs_last_1h": "Carbs (last 1h)",
    "hour_sin": "Time of Day (sin)",
    "hour_cos": "Time of Day (cos)",
    "dow_sin": "Day of Week (sin)",
    "dow_cos": "Day of Week (cos)",
    "relative_time_idx": "Relative Time",
}


def extract_xai(
    raw_output: dict,
    model,
    context_steps: int = 48,
) -> dict:
    """
    Extract XAI signals from TFT raw prediction output.

    Parameters
    ----------
    raw_output : dict
        Output from model.predict(..., mode='raw', return_x=True)
        Must contain 'output' and 'x' keys.
    model : TemporalFusionTransformer
        Loaded TFT model instance.
    context_steps : int
        Number of historical steps in encoder (default 48 = 4h).

    Returns
    -------
    dict with keys:
        variable_importance : list[{name, label, score}]
            Encoder variable importance scores, sorted descending.
        attention_weights : list[{step, minutes_ago, weight}]
            Attention weight per historical timestep.
    """
    # raw_output is (predictions, x) tuple when return_x=True
    out = raw_output[0] if isinstance(raw_output, tuple) else raw_output
    # Extract prediction tensor from namedtuple if needed
    if hasattr(out, 'prediction'):
        out = out.prediction
    elif hasattr(out, 'output'):
        out = out.output
    interpretation = model.interpret_output(out, reduction="mean")

    # ── Variable importance ───────────────────────────────────────────
    enc_imp = interpretation["encoder_variables"].cpu().numpy()
    var_names = model.encoder_variables
    importance = [
        {
            "name": name,
            "label": ENCODER_FEATURE_LABELS.get(name, name),
            "score": round(float(enc_imp[i]), 4),
        }
        for i, name in enumerate(var_names)
    ]
    importance.sort(key=lambda x: x["score"], reverse=True)

    # ── Attention weights ─────────────────────────────────────────────
    attn = interpretation["attention"].cpu().numpy()
    # attn shape: (horizon, context) — mean over horizon
    attn_mean = attn.mean(axis=0) if attn.ndim == 2 else attn
    attn_norm = attn_mean / (attn_mean.sum() + 1e-8)

    n = len(attn_norm)
    attention = [
        {
            "step": i,
            "minutes_ago": (n - i) * 5,
            "weight": round(float(attn_norm[i]), 4),
        }
        for i in range(n)
    ]

    return {
        "variable_importance": importance,
        "attention_weights": attention,
    }
