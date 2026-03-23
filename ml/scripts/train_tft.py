"""
ml/scripts/train_tft.py
=======================
Phase 3 - Temporal Fusion Transformer training on OhioT1DM dataset.

Usage:
    python ml/scripts/train_tft.py [--epochs 50] [--batch-size 64] [--horizon 12]

Auto-resumes from the best checkpoint in ml/models/ if one exists.
Force a specific checkpoint: --ckpt ml/models/tft-epoch=47-val_loss=1.7100.ckpt
Skip resume:                 --no-resume
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

# PyTorch 2.6 compat: lightning passes weights_only=True which breaks
# pytorch-forecasting checkpoints. Patch before importing lightning.
_orig_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=None, *,
                        weights_only=False, mmap=None, **kwargs):
    return _orig_torch_load(f, map_location=map_location,
                            pickle_module=pickle_module,
                            weights_only=False, mmap=mmap, **kwargs)

torch.load = _patched_torch_load

import mlflow
import mlflow.pytorch
import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data" / "processed"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET = "glucose_mg_dl"
GROUP_ID = "subject_id"
TIME_VARYING_KNOWN_REALS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TIME_VARYING_UNKNOWN_REALS = [TARGET, "glucose_delta_1", "glucose_delta_3", "bolus_last_1h", "basal_rate", "carbs_last_1h"]
STATIC_CATEGORICALS = [GROUP_ID]


exec(open('/tmp/patch_tft.py').read())
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--context", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--experiment", type=str, default="nmd-tft-ohiot1dm")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def find_best_checkpoint():
    ckpts = list(MODEL_DIR.glob("tft-*.ckpt"))
    if not ckpts:
        return None
    def _loss(p):
        try:
            return float(p.stem.split("val_loss=")[-1])
        except ValueError:
            return float("inf")
    return str(min(ckpts, key=_loss))


def resolve_checkpoint(args):
    if args.no_resume:
        log.info("  --no-resume: starting from scratch")
        return None
    if args.ckpt:
        log.info(f"  Explicit checkpoint: {args.ckpt}")
        return args.ckpt
    found = find_best_checkpoint()
    if found:
        log.info(f"  Auto-detected checkpoint: {found}")
        return found
    log.info("  No checkpoint found, starting from scratch")
    return None


def load_data(args):
    log.info("Loading data ...")
    train_df = pd.read_parquet(DATA_DIR / "training.parquet")
    test_df = pd.read_parquet(DATA_DIR / "testing.parquet")
    log.info(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
    for df in [train_df, test_df]:
        df[GROUP_ID] = df[GROUP_ID].astype(str)
    train_df = train_df.sort_values([GROUP_ID, "timestamp"]).reset_index(drop=True)
    test_df = test_df.sort_values([GROUP_ID, "timestamp"]).reset_index(drop=True)
    train_df["time_idx"] = train_df.groupby(GROUP_ID).cumcount()
    test_df["time_idx"] = test_df.groupby(GROUP_ID).cumcount()
    for col in ["glucose_delta_1", "glucose_delta_3"]:
        train_df[col] = train_df[col].fillna(0.0)
        test_df[col] = test_df[col].fillna(0.0)
    return train_df, test_df


def build_datasets(train_df, test_df, args):
    log.info("Building datasets ...")
    training = TimeSeriesDataSet(
        train_df, time_idx="time_idx", target=TARGET, group_ids=[GROUP_ID],
        min_encoder_length=args.context // 2, max_encoder_length=args.context,
        min_prediction_length=1, max_prediction_length=args.horizon,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(groups=[GROUP_ID], transformation="softplus"),
        add_relative_time_idx=True, add_target_scales=True,
        add_encoder_length=True, allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, train_df, predict=True, stop_randomization=True)
    log.info(f"  Train: {len(training):,} | Val: {len(validation):,}")
    return training, validation


def build_model(training, args, ckpt_path):
    log.info("Building model ...")
    if ckpt_path:
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
        log.info("  Loaded from checkpoint")
    else:
        model = TemporalFusionTransformer.from_dataset(
            training, learning_rate=args.lr, hidden_size=args.hidden_size,
            attention_head_size=args.attention_heads, dropout=args.dropout,
            hidden_continuous_size=32, loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            log_interval=10, optimizer="adam", reduce_on_plateau_patience=4,
        )
    log.info(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def train(model, training, validation, args, ckpt_path):
    train_loader = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0, persistent_workers=False)
    val_loader = validation.to_dataloader(train=False, batch_size=args.batch_size * 2, num_workers=0, persistent_workers=False)

    accelerator = "cpu" if args.no_gpu or not torch.cuda.is_available() else "gpu"
    log.info(f"  Accelerator: {accelerator.upper()}")

    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator=accelerator, devices=1,
        gradient_clip_val=0.1, enable_progress_bar=True, log_every_n_steps=10,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=8, mode="min", verbose=True),
            ModelCheckpoint(dirpath=MODEL_DIR, filename="tft-{epoch:02d}-val_loss={val_loss:.4f}",
                            monitor="val_loss", save_top_k=2, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        num_sanity_val_steps=0,
    )

    log.info("Starting training ...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    log.info(f"  Best: {trainer.checkpoint_callback.best_model_path}")
    return trainer


def evaluate(model, validation, args):
    log.info("Evaluating ...")
    val_loader = validation.to_dataloader(train=False, batch_size=args.batch_size * 2, num_workers=2)

    predictions = model.predict(val_loader, return_y=True, trainer_kwargs={"accelerator": "cpu"})

    # pytorch-forecasting 1.6 returns output shape: (samples, horizon, quantiles) or (samples, horizon)
    output = predictions.output
    if output.dim() == 3:
        y_pred = output[:, :, 1]  # median quantile
    else:
        y_pred = output  # already median

    y_true = predictions.y[0]

    mae = MAE()(y_pred, y_true).item()
    rmse = RMSE()(y_pred, y_true).item()
    mape = MAPE()(y_pred, y_true).item()
    mard = ((y_pred - y_true).abs() / (y_true.abs() + 1e-8)).mean().item() * 100

    metrics = {
        "val_mae_mg_dl": round(mae, 4),
        "val_rmse_mg_dl": round(rmse, 4),
        "val_mape_pct": round(mape * 100, 4),
        "val_mard_pct": round(mard, 4),
    }

    log.info("  -- Validation metrics --")
    for k, v in metrics.items():
        log.info(f"    {k}: {v}")

    if mard < 10.0:
        log.info("  MARD < 10% -- meets clinical accuracy target")
    else:
        log.warning(f"  MARD {mard:.1f}% -- above 10% clinical target")

    return metrics


def log_run(args, metrics, best_model_path):
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="tft-ohiot1dm"):
        mlflow.log_params({"epochs": args.epochs, "batch_size": args.batch_size,
                           "horizon_steps": args.horizon, "context_steps": args.context,
                           "lr": args.lr, "hidden_size": args.hidden_size,
                           "attention_heads": args.attention_heads, "dropout": args.dropout})
        mlflow.log_metrics(metrics)
        if best_model_path:
            mlflow.log_artifact(best_model_path, artifact_path="checkpoints")
    log.info(f"  MLflow logged -> {args.experiment}")


def main():
    args = parse_args()
    log.info("=" * 60)
    log.info("NeuroMetabolic Dashboard -- TFT Training (Phase 3)")
    log.info(f"  Horizon: {args.horizon} steps ({args.horizon * 5} min)")
    log.info(f"  Context: {args.context} steps ({args.context * 5} min)")
    log.info("=" * 60)

    ckpt_path = resolve_checkpoint(args)
    train_df, test_df = load_data(args)
    training_ds, validation_ds = build_datasets(train_df, test_df, args)
    model = build_model(training_ds, args, ckpt_path)
    trainer = train(model, training_ds, validation_ds, args, ckpt_path)

    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        log.info(f"Loading best checkpoint: {best_path}")
        model = TemporalFusionTransformer.load_from_checkpoint(best_path)

    metrics = evaluate(model, validation_ds, args)
    log_run(args, metrics, best_path)
    log.info("Done. Next: python ml/scripts/export_onnx.py")


if __name__ == "__main__":
    main()