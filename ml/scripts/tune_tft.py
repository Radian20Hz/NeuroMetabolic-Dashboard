"""
ml/scripts/tune_tft.py
========================
Optuna hyperparameter search dla TFT Population Model.

Użycie:
    # Sesja 1 — odpal, idź spać/do szkoły
    python ml/scripts/tune_tft.py --n-trials 20

    # Sesja 2 — kontynuuje automatycznie z tego samego study
    python ml/scripts/tune_tft.py --n-trials 20

    # Podejrzyj wyniki bez trenowania
    python ml/scripts/tune_tft.py --show-results

Wyniki zapisywane do: ml/models/optuna.db (SQLite)

Poprawki względem v1:
  [FIX-T1] precision: bf16-mixed → twarde 32 (identycznie jak w train_tft_population_v2.py)
            bf16 powoduje NaN-y w outputcie przy ekstremalnych wartościach glukozy,
            co zaburza val_loss w trialach i daje fałszywe rankingi.
  [FIX-T2] build_datasets() przeniesione do main() — wywoływane RAZ dla wszystkich triali.
            Poprzednio TimeSeriesDataSet był budowany wewnątrz objective(), co kosztowało
            kilka sekund per trial i przy 20+ trialach sumowało się do ~5 min straty.

Poprawki względem v2 (code review):
  [FIX-T3] Niedozwolone kombinacje hidden_size/attention_heads nie podnoszą już
            TrialPruned (co zaburzało statystyki MedianPrunera). Przestrzeń
            sugestii heads jest teraz dynamicznie ograniczana do wartości
            kompatybilnych z wylosowanym hidden_size.
  [FIX-T4] Zakres lr poszerzony z [5e-5, 5e-4] do [1e-5, 1e-3], żeby objąć
            rzeczywisty sweet-spot TFT na danych klinicznych.
  [FIX-T5] Dodano lstm_layers {1, 2} do przestrzeni — wysokie ROI per trial,
            brak ograniczeń kompatybilności.
  [FIX-T6] Dodano gradient_clip_val {0.1, 0.5, 1.0, 5.0} jako parametr trialu.
            ClinicalQuantileLoss jest niegładka; optymalny clip może się mocno
            różnić między konfiguracjami.
  [FIX-T7] batch_size rozszerzony o 32 — mniejsze batche działają jak
            implicit regularizer dla rzadkich fenotypów pacjentów.
  [FIX-T8] MedianPruner → HyperbandPruner: lepiej dostosowany do budżetu
            20 triali × 25 epok. MedianPruner wymagał co najmniej kilkunastu
            ukończonych triali, zanim zaczął efektywnie przycinać.
  [FIX-T9] TPESampler z multivariate=True — modeluje korelacje między
            hidden_size, lr i heads zamiast traktować je niezależnie.
  [FIX-T10] Jawne czyszczenie GPU po każdym trialu: del model/trainer/loadery
             + torch.cuda.empty_cache() + gc.collect(). Lightning nie zwalnia
             puli CUDA allokatora samodzielnie przy wyjściu z objective().
  [FIX-T11] enable_progress_bar=False w Trainerze — 500 renderów tqdm przy
             20 trialach × 25 epokach generowało zbędny narzut I/O.
  [FIX-T12] num_workers=0 podczas triali — workery są respawnowane per
             DataLoader, nie per trial. Przy persistent_workers=True mogą nie
             terminować się czysto między trialami (szczególnie przy spawn
             context). Dataset jest już zbudowany raz, więc narzut I/O jest
             pomijalny.
  [FIX-T13] SWA jawnie nieobecne w Trainerze triali. SWA w Lightning:
             (a) dezaktywuje PyTorchLightningPruningCallback,
             (b) produkuje val_loss nieporównywalny z finalnym treningiem,
             (c) przy 25 epokach i swa_epoch_start=20 daje tylko 5 epok
             uśredniania — za mało do stabilizacji.
  [FIX-T14] show_results() rozszerzone o lstm_layers i gradient_clip_val
             w tabeli wyników.
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import lightning.pytorch as pl
import optuna
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet

import torch.serialization
from pytorch_forecasting.data.encoders import EncoderNormalizer
from pytorch_forecasting.data import NaNLabelEncoder

import numpy._core.multiarray

torch.serialization.add_safe_globals([
    EncoderNormalizer,
    NaNLabelEncoder,
    numpy._core.multiarray.scalar,
])

from ml.scripts.train_tft_population_v2 import (
    load_and_preprocess_data,
    build_datasets,
    ClinicalQuantileLoss,
    ClinicalTFT,
    QUANTILES,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    MODEL_DIR,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STAŁE
# ─────────────────────────────────────────────────────────────────────────────
STUDY_NAME = "tft_optimization_v2_clean"
STORAGE_PATH = MODEL_DIR / "optuna.db"
STORAGE_URL  = f"sqlite:///{STORAGE_PATH}"

TRIAL_EPOCHS      = 25
TRIAL_PATIENCE    = 8

# [FIX-T12] Ustawiamy num_workers=0 podczas triali — workery DataLoadera są
# respawnowane per obiekt DataLoader, nie per trial. Przy persistent_workers=True
# mogą nie terminować się czysto (szczególnie przy spawn context na Linuksie).
# Dataset jest już zbudowany raz w main(), więc narzut CPU jest pomijalny.
TRIAL_NUM_WORKERS = 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search dla TFT Population Model"
    )
    p.add_argument(
        "--n-trials", type=int, default=20,
        help="Liczba triali do wykonania w tej sesji (default: 20)"
    )
    p.add_argument(
        "--show-results", action="store_true",
        help="Pokaż wyniki istniejącego study i zakończ"
    )
    p.add_argument(
        "--no-gpu", action="store_true",
        help="Wymuś CPU (debug)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE
# [FIX-T2] Przyjmuje gotowe datasety zamiast budować je wewnątrz
# ─────────────────────────────────────────────────────────────────────────────
def objective(
    trial:       optuna.Trial,
    training_ds: TimeSeriesDataSet,
    val_ds:      TimeSeriesDataSet,
    use_gpu:     bool,
) -> float:
    """
    Jeden trial Optuny.
    Zwraca best val_loss z tego trialu (niższy = lepszy).
    """
    # ── 1. Sugestie hiperparametrów ───────────────────────────────────────

    # [FIX-T4] Zakres poszerzony z [5e-5, 5e-4] → [1e-5, 1e-3].
    # Poprzedni lower bound (5e-5) mógł być powyżej optimum dla sekwencji
    # z długozakresowymi zależnościami (np. efekty posiłkowe na glikemię).
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])

    # [FIX-T3] Dynamiczne ograniczenie przestrzeni heads do wartości
    # kompatybilnych z wylosowanym hidden_size. Poprzednio niekompatybilne
    # kombinacje podnosiły TrialPruned, co zaburzało statystyki HyperbandPrunera
    # (pruned-by-design != pruned-by-performance).
    valid_heads = [h for h in [2, 4, 8] if hidden_size % h == 0]
    attention_heads = trial.suggest_categorical("attention_heads", valid_heads)

    dropout = trial.suggest_float("dropout", 0.05, 0.30, step=0.05)

    hidden_continuous_size = trial.suggest_categorical(
        "hidden_continuous_size", [16, 32, 64]
    )

    # [FIX-T7] Dodano batch_size=32 — mniejsze batche działają jak implicit
    # regularizer dla rzadkich fenotypów pacjentów w population model.
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # [FIX-T5] Dodano lstm_layers — wysoki ROI na trial, brak ograniczeń
    # kompatybilności. TFT ma stackowane LSTM w encoderze/decoderze;
    # lstm_layers=2 często daje największy zysk na unit parametrów.
    lstm_layers = trial.suggest_categorical("lstm_layers", [1, 2])

    # [FIX-T6] gradient_clip_val jako parametr trialu. ClinicalQuantileLoss
    # jest niegładka; optymalny clip może się istotnie różnić między
    # konfiguracjami hidden_size i lr.
    gradient_clip_val = trial.suggest_categorical(
        "gradient_clip_val", [0.1, 0.5, 1.0, 5.0]
    )

    log.info(
        f"\n{'='*60}\n"
        f"Trial {trial.number}\n"
        f"  lr={lr:.2e}  hidden={hidden_size}  heads={attention_heads}"
        f"  lstm_layers={lstm_layers}\n"
        f"  dropout={dropout}  hcs={hidden_continuous_size}  "
        f"batch={batch_size}  grad_clip={gradient_clip_val}\n"
        f"{'='*60}"
    )

    # ── 2. Model ──────────────────────────────────────────────────────────
    loss_fn = ClinicalQuantileLoss(quantiles=QUANTILES)

    # [FIX-T13] SWA jawnie nieobecne — nie przekazujemy żadnego
    # StochasticWeightAveraging callbacka. Patrz docstring modułu.
    model = ClinicalTFT.from_dataset(
        training_ds,
        learning_rate=lr,
        hidden_size=hidden_size,
        attention_head_size=attention_heads,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        lstm_layers=lstm_layers,           # [FIX-T5]
        loss=loss_fn,
        log_interval=-1,
        log_val_interval=1,
    )

    # ── 3. DataLoadery ────────────────────────────────────────────────────
    # [FIX-T12] num_workers=0 — brak ryzyka wycieku workerów między trialami.
    # Przy spawn context na Linuksie persistent_workers mogą nie terminować
    # się czysto gdy DataLoader wychodzi ze scope wewnątrz objective().
    train_loader = training_ds.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=TRIAL_NUM_WORKERS,      # [FIX-T12] = 0
        persistent_workers=False,           # [FIX-T12] irrelevant przy 0, ale jawnie
        pin_memory=use_gpu,
        drop_last=True,
    )
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=batch_size * 2,
        num_workers=TRIAL_NUM_WORKERS,      # [FIX-T12] = 0
        persistent_workers=False,           # [FIX-T12]
        pin_memory=use_gpu,
    )

    # ── 4. Callbacks ──────────────────────────────────────────────────────
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=TRIAL_PATIENCE,
        mode="min",
        min_delta=1e-4,
        check_on_train_epoch_end=False,
    )

    pruning_callback = PyTorchLightningPruningCallback(
        trial, monitor="val_loss"
    )

    # ── 5. Trainer ────────────────────────────────────────────────────────
    # [FIX-T1]  Twarde precision=32 — bf16 powoduje NaN-y przy log-normalizacji
    #           glucose. Identyczna decyzja jak w train_tft_population_v2.py.
    # [FIX-T11] enable_progress_bar=False — 20 triali × 25 epok = 500 renderów
    #           tqdm, zbędny narzut I/O w trybie tuningowym.
    # [FIX-T13] Brak SWA callbacka — patrz docstring modułu i FIX-T13 wyżej.
    accelerator = "cpu" if not use_gpu else "gpu"

    trainer = pl.Trainer(
        max_epochs=TRIAL_EPOCHS,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=gradient_clip_val,        # [FIX-T6] z przestrzeni
        gradient_clip_algorithm="norm",
        precision=32,                               # [FIX-T1] nigdy bf16
        enable_progress_bar=False,                  # [FIX-T11]
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[early_stopping, pruning_callback],
        logger=False,
        deterministic=False,
    )

    # ── 6. Trening ────────────────────────────────────────────────────────
    try:
        trainer.fit(model, train_loader, val_loader)
    except optuna.exceptions.TrialPruned:
        log.info(f"  Trial {trial.number} pruned by HyperbandPruner.")
        raise

    best_val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
    if isinstance(best_val_loss, torch.Tensor):
        best_val_loss = best_val_loss.item()

    log.info(
        f"  Trial {trial.number} finished: val_loss={best_val_loss:.4f}  "
        f"(stopped at epoch {trainer.current_epoch})"
    )

    del model, trainer, train_loader, val_loader
    if use_gpu:
        torch.cuda.empty_cache()
    gc.collect()

    return float(best_val_loss)


# ─────────────────────────────────────────────────────────────────────────────
# SHOW RESULTS
# [FIX-T14] Tabela rozszerzona o lstm_layers i gradient_clip_val
# ─────────────────────────────────────────────────────────────────────────────
def show_results() -> None:
    if not STORAGE_PATH.exists():
        print(f"Brak pliku study: {STORAGE_PATH}")
        print("Uruchom najpierw przynajmniej jeden trial.")
        return

    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
    )

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]

    print(f"\n{'='*80}")
    print(f"Study: {STUDY_NAME}")
    print(f"  Triały: {len(study.trials)} total  "
          f"({len(completed)} complete, {len(pruned)} pruned)")

    if not completed:
        print("  Brak ukończonych triali.")
        return

    print(f"\nNajlepszy trial: #{study.best_trial.number}")
    print(f"  val_loss : {study.best_trial.value:.4f}")
    print(f"  Parametry:")
    for k, v in study.best_trial.params.items():
        print(f"    {k:<30s} = {v}")

    print(f"\nWszystkie ukończone triały (posortowane po val_loss):")
    print(
        f"  {'#':<4} {'val_loss':<10} {'lr':<10} {'hidden':<8} "
        f"{'heads':<7} {'lstm':<6} {'drop':<7} {'hcs':<6} "
        f"{'batch':<7} {'clip'}"
    )
    print(f"  {'-'*80}")

    sorted_trials = sorted(completed, key=lambda t: t.value)
    for t in sorted_trials:
        p = t.params
        lr_val = p.get("lr", float("nan"))
        lr_str = f"{lr_val:.2e}" if isinstance(lr_val, float) else "?"
        print(
            f"  {t.number:<4} {t.value:<10.4f} "
            f"{lr_str:<10} "
            f"{p.get('hidden_size', '?'):<8} "
            f"{p.get('attention_heads', '?'):<7} "
            f"{p.get('lstm_layers', '?'):<6} "
            f"{p.get('dropout', 0.0):<7.2f} "
            f"{p.get('hidden_continuous_size', '?'):<6} "
            f"{p.get('batch_size', '?'):<7} "
            f"{p.get('gradient_clip_val', '?')}"
        )
    print(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if args.show_results:
        show_results()
        return

    pl.seed_everything(args.seed, workers=True)

    use_gpu = torch.cuda.is_available() and not args.no_gpu

    log.info("=" * 60)
    log.info("NMD — Optuna Hyperparameter Search")
    log.info(f"  Study       : {STUDY_NAME}")
    log.info(f"  Storage     : {STORAGE_URL}")
    log.info(f"  N trials    : {args.n_trials} (ta sesja)")
    log.info(f"  Max epochs  : {TRIAL_EPOCHS} per trial")
    log.info(f"  Patience    : {TRIAL_PATIENCE}")
    log.info(f"  Accelerator : {'GPU' if use_gpu else 'CPU'}")
    log.info(f"  Precision   : 32-bit float (bf16 disabled) [FIX-T1]")
    log.info(f"  Datasets    : budowane raz przed trialami [FIX-T2]")
    log.info(f"  Pruner      : HyperbandPruner [FIX-T8]")
    log.info(f"  Sampler     : TPESampler(multivariate=True) [FIX-T9]")
    log.info(f"  Workers     : {TRIAL_NUM_WORKERS} (0=brak wycieku) [FIX-T12]")
    log.info(f"  SWA         : wyłączone podczas triali [FIX-T13]")
    log.info("=" * 60)

    # ── Wczytaj dane i zbuduj datasety RAZ dla wszystkich triali ─────────
    # [FIX-T2] TimeSeriesDataSet budowany tutaj, nie w objective()
    log.info("Wczytywanie danych...")
    train_df, val_df = load_and_preprocess_data()
    log.info(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")

    log.info("Budowanie TimeSeriesDataSets (raz dla wszystkich triali)...")

    class _Args:
        context = MAX_ENCODER_LENGTH
        horizon = MAX_PREDICTION_LENGTH

    training_ds, val_ds = build_datasets(train_df, val_df, _Args())
    log.info(
        f"  Train windows: {len(training_ds):,} | "
        f"Val windows: {len(val_ds):,}"
    )

    # ── Utwórz sampler i pruner ───────────────────────────────────────────

    # [FIX-T9] TPESampler z multivariate=True modeluje zależności między
    # parametrami (np. hidden_size↔lr↔attention_heads) zamiast traktować
    # je jako niezależne rozkłady jednowymiarowe.
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        n_startup_trials=5,   # losowe próbkowanie przez pierwsze 5 triali
    )

    # [FIX-T8] HyperbandPruner zamiast MedianPrunera.
    # MedianPruner wymaga wielu ukończonych triali zanim zaczyna efektywnie
    # przycinać. Hyperband dzieli budżet epok na "rundy" i agresywnie eliminuje
    # słabe konfiguracje już po kilku epokach — znacznie lepsze dla budżetu
    # 20 triali × 25 epok.
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=5,           # nie przycinaj przed epoką 5
        max_resource=TRIAL_EPOCHS,
        reduction_factor=3,
    )

    # ── Utwórz lub wznów study ────────────────────────────────────────────
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        load_if_exists=True,
        sampler=sampler,    # [FIX-T9]
        pruner=pruner,      # [FIX-T8]
    )

    existing = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    log.info(f"Study wznowione — {existing} ukończonych triali w historii.")

    # ── Uruchom trialy ────────────────────────────────────────────────────
    study.optimize(
        lambda trial: objective(trial, training_ds, val_ds, use_gpu),
        n_trials=args.n_trials,
        timeout=None,
        gc_after_trial=True,        # Python GC po każdym trialu
        show_progress_bar=False,
    )

    # ── Podsumowanie ──────────────────────────────────────────────────────
    show_results()

    log.info("=" * 60)
    log.info("Następna sesja:")
    log.info(f"  python ml/scripts/tune_tft.py --n-trials {args.n_trials}")
    log.info("Podgląd wyników bez trenowania:")
    log.info("  python ml/scripts/tune_tft.py --show-results")
    log.info("=" * 60)


if __name__ == "__main__":
    main()