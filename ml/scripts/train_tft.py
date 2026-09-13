"""
ml/scripts/train_tft_population.py
===================================
Phase 4 — Population TFT: subject-agnostic glucose forecasting.

Key differences from Phase 3 (train_tft.py):
- subject_id removed from STATIC_CATEGORICALS
- EncoderNormalizer replaces GroupNormalizer (normalizes per-window, not per-subject)
- group_ids uses a synthetic constant "population" group
- New users can be served without OhioT1DM subject IDs

Usage:
    python ml/scripts/train_tft_population.py [--epochs 50] [--batch-size 64]
"""
from __future__ import annotations
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, QuantileLoss
from pytorch_forecasting.data import EncoderNormalizer
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import lightning.pytorch as pl
import pandas as pd

import argparse
import logging
from pathlib import Path

import torch

_orig_torch_load = torch.load


def _patched_torch_load(f, map_location=None, pickle_module=None, *,
                        weights_only=False, mmap=None, **kwargs):
    return _orig_torch_load(f, map_location=map_location,
                            pickle_module=pickle_module,
                            weights_only=False, mmap=mmap, **kwargs)


torch.load = _patched_torch_load


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data" / "processed"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

TARGET = "glucose_mg_dl"
GROUP_ID = "group"  # synthetic constant — no subject identity
TIME_VARYING_KNOWN_REALS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TIME_VARYING_UNKNOWN_REALS = [
    TARGET,
    "glucose_delta_1",
    "glucose_delta_3",
    "bolus_last_1h",
    "basal_rate",
    "carbs_last_1h",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--context", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def find_best_checkpoint(prefix="tft-pop"):
    ckpts = list(MODEL_DIR.glob(f"{prefix}-*.ckpt"))
    if not ckpts:
        return None

    def _loss(p):
        try:
            return float(p.stem.split("val_loss=")[-1])
        except ValueError:
            return float("inf")
    return str(min(ckpts, key=_loss))


def load_data(args):
    log.info("Loading data ...")
    df = pd.read_parquet(DATA_DIR / "training.parquet")

    # Drop subject identity — population model is subject-agnostic
    df = df.drop(columns=["subject_id"], errors="ignore")
    df[GROUP_ID] = "population"

    df = df.sort_values(["timestamp"]).reset_index(drop=True)
    df["time_idx"] = df.index  # global time index across all subjects

    for col in ["glucose_delta_1", "glucose_delta_3"]:
        df[col] = df[col].fillna(0.0)

    # 85/15 train/val split by time
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    log.info(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")
    return train_df, val_df


def build_datasets(train_df, val_df, args):
    log.info("Building datasets ...")
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=[GROUP_ID],
        min_encoder_length=args.context // 2,
        max_encoder_length=args.context,
        min_prediction_length=1,
        max_prediction_length=args.horizon,
        static_categoricals=[],           # no subject ID
        static_reals=[],
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=EncoderNormalizer(transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, predict=False, stop_randomization=True
    )
    log.info(f"  Train samples: {len(training):,} | Val: {len(validation):,}")
    return training, validation


def build_model(training, args, ckpt_path):
    log.info("Building model ...")
    if ckpt_path:
        log.info(f"  Resuming from: {ckpt_path}")
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    else:
        model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=args.lr,
            hidden_size=args.hidden_size,
            attention_head_size=args.attention_heads,
            dropout=args.dropout,
            hidden_continuous_size=32,
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            log_interval=10,
            log_val_interval=0,
            optimizer="adam",
            reduce_on_plateau_patience=4,
        )
    log.info(
        f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def train(model, training, validation, args, ckpt_path):
    train_loader = training.to_dataloader(
        train=True,
        batch_size=args.batch_size,
        num_workers=2,
        persistent_workers=False)
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=args.batch_size * 2,
        num_workers=2,
        persistent_workers=False)

    accelerator = "cpu" if args.no_gpu or not torch.cuda.is_available() else "gpu"
    log.info(f"  Accelerator: {accelerator.upper()}")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=8,
                mode="min",
                verbose=True),
            ModelCheckpoint(
                dirpath=MODEL_DIR,
                filename="tft-pop-{epoch:02d}-val_loss={val_loss:.4f}",
                monitor="val_loss",
                save_top_k=2,
                mode="min",
            ),
            LearningRateMonitor(
                logging_interval="epoch"),
        ],
    )

    log.info("Starting training ...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path if not args.no_resume else None,
    )
    log.info(
        f"  Best checkpoint: {
            trainer.checkpoint_callback.best_model_path}")
    return trainer


def evaluate(model, validation, args):
    log.info("Evaluating ...")
    val_loader = validation.to_dataloader(
        train=False, batch_size=args.batch_size * 2, num_workers=2
    )
    predictions = model.predict(
        val_loader, return_y=True, trainer_kwargs={"accelerator": "cpu"}
    )

    output = predictions.output
    y_pred = output[:, :, 1] if output.dim() == 3 else output
    y_true = predictions.y[0]

    # Ekstrakcja tylko OSTATNIEGO kroku horyzontu (t+60 min)
# Ekstrakcja tylko OSTATNIEGO kroku horyzontu (t+60 min)
    # Ekstrakcja tylko OSTATNIEGO kroku horyzontu (t+60 min)
    y_pred_60 = y_pred[:, -1]
    y_true_60 = y_true[:, -1]

    # --- TYTANOWA MASKA ---
    # Odrzucamy NaN z OBU stron
    valid_mask = ~torch.isnan(y_true_60) & ~torch.isnan(y_pred_60)

    y_pred_valid = y_pred_60[valid_mask]
    y_true_valid = y_true_60[valid_mask]

    # Obliczanie metryk wyłącznie na istniejących, prawdziwych danych
    mae_60 = (y_pred_valid - y_true_valid).abs().mean().item()
    rmse_60 = torch.sqrt(((y_pred_valid - y_true_valid) ** 2).mean()).item()
    mard_60 = ((y_pred_valid - y_true_valid).abs() / (y_true_valid.abs() + 1e-8)).mean().item() * 100

    metrics = {
        "val_mae_60m_mg_dl": round(mae_60, 4),
        "val_rmse_60m_mg_dl": round(rmse_60, 4),
        "val_mard_60m_pct": round(mard_60, 4),
    }

    log.info("  -- Validation metrics --")
    for k, v in metrics.items():
        log.info(f"    {k}: {v}")

    if mard_60 < 10.0:
        log.info("  ✓ MARD < 10% — meets clinical accuracy target")
    else:
        log.warning(f"  ✗ MARD {mard_60:.1f}% — above 10% clinical target")

    return metrics


def main():
    args = parse_args()
    log.info("=" * 60)
    log.info("NMD — TFT Population Model Training (Phase 4)")
    log.info(f"  Horizon : {args.horizon} steps ({args.horizon * 5} min)")
    log.info(f"  Context : {args.context} steps ({args.context * 5} min)")
    log.info("  Subject ID: REMOVED — population model")
    log.info("=" * 60)

    ckpt_path = None if args.no_resume else find_best_checkpoint()
    train_df, val_df = load_data(args)
    training_ds, validation_ds = build_datasets(train_df, val_df, args)
    model = build_model(training_ds, args, ckpt_path)
    trainer = train(model, training_ds, validation_ds, args, ckpt_path)

    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        model = TemporalFusionTransformer.load_from_checkpoint(best_path)

    evaluate(model, validation_ds, args)
    log.info("Done. Model saved to ml/models/tft-pop-*.ckpt")


if __name__ == "__main__":
    main()