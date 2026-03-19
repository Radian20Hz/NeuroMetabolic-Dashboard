"""
app/services/tft_inference.py
=============================
TFT inference service – loads the trained checkpoint once and exposes
a predict() function that returns 60-minute glucose forecasts with
uncertainty intervals (10th / 50th / 90th quantile).
"""
from __future__ import annotations

import inspect
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "ml" / "models"
DATA_DIR = ROOT / "ml" / "data" / "processed"

TARGET = "glucose_mg_dl"
GROUP_ID = "subject_id"
TIME_VARYING_KNOWN_REALS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TIME_VARYING_UNKNOWN_REALS = [TARGET, "glucose_delta_1", "glucose_delta_3",
                               "bolus_last_1h", "basal_rate", "carbs_last_1h"]  # noqa: E127
STATIC_CATEGORICALS = [GROUP_ID]
HORIZON = 12
CONTEXT = 48

_model = None
_dataset = None


def _find_best_checkpoint() -> Path:
    ckpts = list(MODEL_DIR.glob("tft-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No TFT checkpoints found in {MODEL_DIR}")

    def _loss(p: Path) -> float:
        try:
            return float(p.stem.split("val_loss=")[-1])
        except ValueError:
            return float("inf")

    return min(ckpts, key=_loss)


def _load_model():
    global _model, _dataset

    _orig = torch.load
    def _patched(f, map_location=None, pickle_module=None, *,
                 weights_only=False, mmap=None, **kw):
        return _orig(f, map_location=map_location,
                     pickle_module=pickle_module,
                     weights_only=False, mmap=mmap, **kw)
    torch.load = _patched

    try:
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        ckpt = _find_best_checkpoint()
        log.info(f"[TFT] Loading checkpoint: {ckpt.name}")

        raw_ckpt = torch.load(str(ckpt), weights_only=False)
        hp = raw_ckpt.get("hyper_parameters", {})
        valid_keys = set(
            inspect.signature(
                TemporalFusionTransformer.__init__).parameters.keys())
        unknown = [k for k in list(hp.keys()) if k not in valid_keys]
        if unknown:
            for k in unknown:
                hp.pop(k)
            log.info(
                f"[TFT] Patched checkpoint (removed unknown hp: {unknown})")

        tmp = tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False)
        torch.save(raw_ckpt, tmp.name)
        tmp.close()
        try:
            _model = TemporalFusionTransformer.load_from_checkpoint(tmp.name)
        finally:
            os.unlink(tmp.name)
        _model.eval()

        df = pd.read_parquet(DATA_DIR / "training.parquet")
        df[GROUP_ID] = df[GROUP_ID].astype(str)
        df = df.sort_values([GROUP_ID, "timestamp"]).reset_index(drop=True)
        df["time_idx"] = df.groupby(GROUP_ID).cumcount()
        for col in ["glucose_delta_1", "glucose_delta_3"]:
            df[col] = df[col].fillna(0.0)

        _dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=TARGET,
            group_ids=[GROUP_ID],
            min_encoder_length=CONTEXT // 2,
            max_encoder_length=CONTEXT,
            min_prediction_length=1,
            max_prediction_length=HORIZON,
            static_categoricals=STATIC_CATEGORICALS,
            time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
            time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
            target_normalizer=GroupNormalizer(
                groups=[GROUP_ID], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        _model.output_transformer = _dataset.target_normalizer
        log.info("[TFT] Model ready.")
    finally:
        torch.load = _orig


def get_model():
    if _model is None:
        _load_model()
    return _model, _dataset


def predict_from_history(
    glucose_values: list[float],
    bolus_values: Optional[list[float]] = None,
    basal_values: Optional[list[float]] = None,
    carbs_values: Optional[list[float]] = None,
    subject_id: str = "559",
) -> dict:
    model, dataset = get_model()

    n = min(len(glucose_values), CONTEXT)
    glucose_values = glucose_values[-n:]

    def _pad(vals: Optional[list[float]]) -> list[float]:
        if vals is None:
            return [0.0] * n
        v = vals[-n:]
        return ([0.0] * (n - len(v))) + list(v)

    glucose = glucose_values
    bolus = _pad(bolus_values)
    basal = _pad(basal_values)
    carbs = _pad(carbs_values)

    delta1 = [0.0] + [glucose[i] - glucose[i - 1] for i in range(1, n)]
    delta3 = [0.0] * min(3, n) + [
        glucose[i] - glucose[i - 3] for i in range(3, n)
    ]

    import math
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="5min")
    hour_sin = [math.sin(2 * math.pi * t.hour / 24) for t in timestamps]
    hour_cos = [math.cos(2 * math.pi * t.hour / 24) for t in timestamps]
    dow_sin = [math.sin(2 * math.pi * t.dayofweek / 7) for t in timestamps]
    dow_cos = [math.cos(2 * math.pi * t.dayofweek / 7) for t in timestamps]

    future_timestamps = pd.date_range(
        start=timestamps[-1] + pd.Timedelta("5min"),
        periods=HORIZON, freq="5min"
    )
    future_hour_sin = [math.sin(2 * math.pi * t.hour / 24)
                       for t in future_timestamps]
    future_hour_cos = [math.cos(2 * math.pi * t.hour / 24)
                       for t in future_timestamps]
    future_dow_sin = [math.sin(2 * math.pi * t.dayofweek / 7)
                      for t in future_timestamps]
    future_dow_cos = [math.cos(2 * math.pi * t.dayofweek / 7)
                      for t in future_timestamps]

    enc_rows = {
        "timestamp": list(timestamps),
        "subject_id": [subject_id] * n,
        "time_idx": list(range(n)),
        "glucose_mg_dl": [float(g) for g in glucose],
        "glucose_delta_1": delta1,
        "glucose_delta_3": delta3,
        "bolus_last_1h": bolus,
        "basal_rate": basal,
        "carbs_last_1h": carbs,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
    }
    dec_rows = {
        "timestamp": list(future_timestamps),
        "subject_id": [subject_id] * HORIZON,
        "time_idx": list(range(n, n + HORIZON)),
        "glucose_mg_dl": [float(glucose[-1])] * HORIZON,
        "glucose_delta_1": [0.0] * HORIZON,
        "glucose_delta_3": [0.0] * HORIZON,
        "bolus_last_1h": [0.0] * HORIZON,
        "basal_rate": [basal[-1]] * HORIZON,
        "carbs_last_1h": [0.0] * HORIZON,
        "hour_sin": future_hour_sin,
        "hour_cos": future_hour_cos,
        "dow_sin": future_dow_sin,
        "dow_cos": future_dow_cos,
    }

    df_pred = pd.DataFrame(
        {**{k: enc_rows[k] + dec_rows[k] for k in enc_rows}})
    df_pred["subject_id"] = df_pred["subject_id"].astype(str)

    from pytorch_forecasting import TimeSeriesDataSet

    pred_dataset = TimeSeriesDataSet.from_dataset(
        dataset, df_pred, predict=True, stop_randomization=True
    )
    loader = pred_dataset.to_dataloader(
        train=False, batch_size=1, num_workers=0)

    with torch.no_grad():
        raw_preds = model.predict(loader, mode="raw", return_x=False)

    # PF 1.1.x: raw_preds may be Output namedtuple or direct tensor
    if hasattr(raw_preds, 'prediction'):
        output = raw_preds.prediction
    elif hasattr(raw_preds, 'output'):
        output = raw_preds.output
    else:
        output = raw_preds

    if isinstance(output, torch.Tensor) and output.dim() == 3:
        lower = output[0, :, 0].tolist()
        median = output[0, :, 1].tolist()
        upper = output[0, :, 2].tolist()
    elif isinstance(output, torch.Tensor) and output.dim() == 2:
        median = output[0].tolist()
        lower = median
        upper = median
    else:
        median = output[0].tolist()
        lower = median
        upper = median

    return {
        "predictions_mg_dl": [round(v, 1) for v in median],
        "lower_mg_dl": [round(v, 1) for v in lower],
        "upper_mg_dl": [round(v, 1) for v in upper],
        "horizon_minutes": [i * 5 for i in range(1, HORIZON + 1)],
    }
