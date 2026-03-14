"""
ml/scripts/export_onnx.py
=========================
Phase 3 - Export trained TFT to ONNX for edge inference.

Usage:
    python ml/scripts/export_onnx.py
    python ml/scripts/export_onnx.py --ckpt ml/models/tft-epoch=47-val_loss=1.7104.ckpt

Output:
    ml/models/nmd_tft.onnx       - ONNX model
    ml/models/nmd_tft_meta.json  - input/output metadata for inference
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

# PyTorch 2.6 compat patch
_orig_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=None, *,
                        weights_only=False, mmap=None, **kwargs):
    return _orig_torch_load(f, map_location=map_location,
                            pickle_module=pickle_module,
                            weights_only=False, mmap=mmap, **kwargs)

torch.load = _patched_torch_load

import numpy as np
import pandas as pd
import onnxruntime as ort
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data" / "processed"
MODEL_DIR = ROOT / "ml" / "models"
ONNX_PATH = MODEL_DIR / "nmd_tft.onnx"
META_PATH = MODEL_DIR / "nmd_tft_meta.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET = "glucose_mg_dl"
GROUP_ID = "subject_id"
TIME_VARYING_KNOWN_REALS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TIME_VARYING_UNKNOWN_REALS = [TARGET, "glucose_delta_1", "glucose_delta_3", "bolus_last_1h", "basal_rate", "carbs_last_1h"]
STATIC_CATEGORICALS = [GROUP_ID]
HORIZON = 12   # 60 min
CONTEXT = 48   # 240 min


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (auto-detected if omitted)")
    return p.parse_args()


def find_best_checkpoint() -> str:
    ckpts = list(MODEL_DIR.glob("tft-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {MODEL_DIR}")

    def _loss(p: Path) -> float:
        try:
            return float(p.stem.split("val_loss=")[-1])
        except ValueError:
            return float("inf")

    return str(min(ckpts, key=_loss))


def load_data() -> pd.DataFrame:
    log.info("Loading training data for dataset schema ...")
    df = pd.read_parquet(DATA_DIR / "training.parquet")
    df[GROUP_ID] = df[GROUP_ID].astype(str)
    df = df.sort_values([GROUP_ID, "timestamp"]).reset_index(drop=True)
    df["time_idx"] = df.groupby(GROUP_ID).cumcount()
    for col in ["glucose_delta_1", "glucose_delta_3"]:
        df[col] = df[col].fillna(0.0)
    return df


def build_dataset(df: pd.DataFrame) -> TimeSeriesDataSet:
    return TimeSeriesDataSet(
        df, time_idx="time_idx", target=TARGET, group_ids=[GROUP_ID],
        min_encoder_length=CONTEXT // 2, max_encoder_length=CONTEXT,
        min_prediction_length=1, max_prediction_length=HORIZON,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(groups=[GROUP_ID], transformation="softplus"),
        add_relative_time_idx=True, add_target_scales=True,
        add_encoder_length=True, allow_missing_timesteps=True,
    )


def get_sample_batch(dataset: TimeSeriesDataSet) -> dict:
    """Extract one sample batch for tracing."""
    loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    x, _ = next(iter(loader))
    return x


def export_onnx(model: TemporalFusionTransformer, sample_x: dict, ckpt_path: str) -> None:
    log.info(f"Exporting to ONNX: {ONNX_PATH}")

    model.eval()

    # TFT.forward() takes a dict — wrap in a tuple-based callable for ONNX tracing
    # We use torch.onnx.export with a wrapper that accepts tensors positionally
    encoder_cont = sample_x["encoder_cont"]
    decoder_cont = sample_x["decoder_cont"]
    encoder_cat = sample_x["encoder_cat"]
    decoder_cat = sample_x["decoder_cat"] if "decoder_cat" in sample_x else torch.zeros(1, HORIZON, 0, dtype=torch.long)
    encoder_lengths = sample_x["encoder_lengths"]
    decoder_lengths = sample_x["decoder_lengths"]
    encoder_target = sample_x["encoder_target"] if isinstance(sample_x.get("encoder_target"), torch.Tensor) else encoder_cont[:, :, 0:1]
    target_scale = sample_x["target_scale"]

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, tft):
            super().__init__()
            self.tft = tft

        def forward(self, encoder_cont, decoder_cont, encoder_cat,
                    decoder_cat, encoder_lengths, decoder_lengths,
                    encoder_target, target_scale):
            x = {
                "encoder_cont": encoder_cont,
                "decoder_cont": decoder_cont,
                "encoder_cat": encoder_cat,
                "decoder_cat": decoder_cat,
                "encoder_lengths": encoder_lengths,
                "decoder_lengths": decoder_lengths,
                "encoder_target": [encoder_target],
                "target_scale": target_scale,
            }
            out = self.tft(x)
            # Return median prediction (quantile index 1)
            pred = out.prediction
            if pred.dim() == 3:
                return pred[:, :, 1]
            return pred

    wrapper = ONNXWrapper(model)
    wrapper.eval()

    input_names = [
        "encoder_cont", "decoder_cont", "encoder_cat", "decoder_cat",
        "encoder_lengths", "decoder_lengths", "encoder_target", "target_scale"
    ]
    output_names = ["glucose_prediction"]

    dynamic_axes = {
        "encoder_cont": {0: "batch", 1: "encoder_len"},
        "decoder_cont": {0: "batch", 1: "horizon"},
        "encoder_cat":  {0: "batch", 1: "encoder_len"},
        "decoder_cat":  {0: "batch", 1: "horizon"},
        "encoder_target": {0: "batch", 1: "encoder_len"},
        "glucose_prediction": {0: "batch", 1: "horizon"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (encoder_cont, decoder_cont, encoder_cat, decoder_cat,
             encoder_lengths, decoder_lengths, encoder_target, target_scale),
            str(ONNX_PATH),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11,
            do_constant_folding=True,
            dynamo=False,  # legacy jit.trace exporter; torch.export fails on TFT
        )

    log.info(f"  Saved: {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1024:.1f} KB)")


def verify_onnx(model: TemporalFusionTransformer, sample_x: dict) -> None:
    log.info("Verifying ONNX output vs PyTorch ...")

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    encoder_cont = sample_x["encoder_cont"]
    decoder_cont = sample_x["decoder_cont"]
    encoder_cat = sample_x["encoder_cat"]
    decoder_cat = sample_x.get("decoder_cat", torch.zeros(1, HORIZON, 0, dtype=torch.long))
    encoder_lengths = sample_x["encoder_lengths"]
    decoder_lengths = sample_x["decoder_lengths"]
    encoder_target = sample_x.get("encoder_target", encoder_cont[:, :, 0:1])
    if isinstance(encoder_target, list):
        encoder_target = encoder_target[0]
    target_scale = sample_x["target_scale"]

    # PyTorch prediction
    model.eval()
    with torch.no_grad():
        x = {
            "encoder_cont": encoder_cont,
            "decoder_cont": decoder_cont,
            "encoder_cat": encoder_cat,
            "decoder_cat": decoder_cat,
            "encoder_lengths": encoder_lengths,
            "decoder_lengths": decoder_lengths,
            "encoder_target": [encoder_target],
            "target_scale": target_scale,
        }
        pt_out = model(x).prediction
        if pt_out.dim() == 3:
            pt_out = pt_out[:, :, 1]
        pt_np = pt_out.numpy()

    # ONNX prediction
    onnx_out = sess.run(["glucose_prediction"], {
        "encoder_cont": encoder_cont.numpy(),
        "decoder_cont": decoder_cont.numpy(),
        "encoder_cat": encoder_cat.numpy(),
        "decoder_cat": decoder_cat.numpy(),
        "encoder_lengths": encoder_lengths.numpy(),
        "decoder_lengths": decoder_lengths.numpy(),
        "encoder_target": encoder_target.numpy(),
        "target_scale": target_scale.numpy(),
    })[0]

    max_diff = np.abs(pt_np - onnx_out).max()
    log.info(f"  Max absolute diff PyTorch vs ONNX: {max_diff:.6f}")

    if max_diff < 1e-3:
        log.info("  ONNX verification PASSED")
    else:
        log.warning(f"  ONNX verification WARNING: diff {max_diff:.6f} > 1e-3")


def save_metadata(model: TemporalFusionTransformer) -> None:
    meta = {
        "model": "TemporalFusionTransformer",
        "horizon_steps": HORIZON,
        "horizon_minutes": HORIZON * 5,
        "context_steps": CONTEXT,
        "context_minutes": CONTEXT * 5,
        "target": TARGET,
        "quantiles": [0.1, 0.5, 0.9],
        "output_index": 1,
        "output_description": "median glucose prediction (mg/dL)",
        "time_varying_known_reals": TIME_VARYING_KNOWN_REALS,
        "time_varying_unknown_reals": TIME_VARYING_UNKNOWN_REALS,
        "static_categoricals": STATIC_CATEGORICALS,
        "onnx_path": str(ONNX_PATH),
        "onnx_inputs": [
            "encoder_cont", "decoder_cont", "encoder_cat", "decoder_cat",
            "encoder_lengths", "decoder_lengths", "encoder_target", "target_scale"
        ],
        "onnx_outputs": ["glucose_prediction"],
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"  Metadata saved: {META_PATH}")


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("NeuroMetabolic Dashboard -- ONNX Export (Phase 3)")
    log.info("=" * 60)

    ckpt_path = args.ckpt or find_best_checkpoint()
    log.info(f"Checkpoint: {ckpt_path}")

    log.info("Loading model ...")
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    model.eval()

    df = load_data()
    dataset = build_dataset(df)
    sample_x = get_sample_batch(dataset)

    try:
        export_onnx(model, sample_x, ckpt_path)
        verify_onnx(model, sample_x)
    except Exception as e:
        log.error(f"ONNX export failed: {e}")
        log.info("Saving metadata only (ONNX export requires compatible opset)")
        raise

    save_metadata(model)

    log.info("=" * 60)
    log.info("ONNX export complete")
    log.info(f"  Model : {ONNX_PATH}")
    log.info(f"  Meta  : {META_PATH}")
    log.info("=" * 60)
    log.info("Next: git add ml/ && git commit -m 'feat(ml): Phase 3 TFT training complete'")


if __name__ == "__main__":
    main()