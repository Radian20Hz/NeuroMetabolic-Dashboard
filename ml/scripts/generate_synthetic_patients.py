"""
ml/scripts/generate_synthetic_patients.py
==========================================
Generuje syntetyczne dane T1D z symulatora UVA/Padova (simglucose)
i zapisuje je w formacie kompatybilnym z training.parquet NMD.

Użycie:
    # Instalacja (raz):
    pip install simglucose --break-system-packages

    # Generowanie:
    python ml/scripts/generate_synthetic_patients.py
    python ml/scripts/generate_synthetic_patients.py --days 90 --patients all
    python ml/scripts/generate_synthetic_patients.py --days 30 --patients adolescent

    # Merge z istniejącym parquetem Ohio:
    python ml/scripts/generate_synthetic_patients.py --merge

Strategia:
    1. 30 wirtualnych pacjentów (10 dzieci, 10 dorosłych, 10 nastolatków)
    2. 90 dni symulacji per pacjent z losowymi posiłkami i bolusami
    3. Output: synthetic_population.parquet w tym samym formacie co training.parquet
    4. source_split="synthetic" — training script filtruje osobno

Ważne:
    - simglucose zwraca glucose w mg/dL co 1 minutę — resample do 5min
    - Brak wearables (HR, GSR, temp, steps) → kolumny = 0, flaga wearable_available=0
    - Bolus calculator: CR (carb ratio) i CF (correction factor) z params awatara
    - Basal: stały z params awatara (upd. u dzieci co noc)

Architektura feature pipeline:
    Syntetyczne dane przechodzą przez ten sam process_patient() co Ohio,
    ale z uproszczonym wejściem (brak basis/wearable XML).
    Zamiast tego wywołujemy feature engineering bezpośrednio na DataFrame.
"""
from __future__ import annotations

import argparse
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "ml" / "data" / "processed"

# ─────────────────────────────────────────────────────────────────────────────
# STAŁE SYMULACJI
# ─────────────────────────────────────────────────────────────────────────────

# Wszystkie 30 awatarów simglucose
ALL_PATIENTS = (
    [f"adolescent#{i:03d}" for i in range(1, 11)] +
    [f"adult#{i:03d}"      for i in range(1, 11)] +
    [f"child#{i:03d}"      for i in range(1, 11)]
)

PATIENT_GROUPS = {
    "adolescent": [f"adolescent#{i:03d}" for i in range(1, 11)],
    "adult":      [f"adult#{i:03d}"      for i in range(1, 11)],
    "child":      [f"child#{i:03d}"      for i in range(1, 11)],
    "all":        ALL_PATIENTS,
}

# Meal distributions (g CHO) — truncated normal
MEAL_PARAMS = {
    # (mean_g, std_g, min_g, max_g, base_hour, hour_std)
    "breakfast": (45, 15, 15, 80,  7.5, 0.5),
    "lunch":     (70, 20, 30, 110, 12.5, 0.75),
    "dinner":    (60, 20, 25, 100, 18.5, 0.75),
    "snack":     (25, 10, 10, 50,  None, None),  # losowa godzina
}
SNACK_PROBABILITY = 0.35   # prawdopodobieństwo przekąski danego dnia
SNACK_HOURS       = [10.0, 15.5, 21.0]  # możliwe godziny przekąski

# Bolus calculator
# Correction target: 110 mg/dL (typowy target euglycemia)
CORRECTION_TARGET_MGDL = 110.0

# PK constants (identyczne jak w preprocess_ohiot1dm.py)
IOB_DIA_MIN  = 240.0
IOB_TAU1     = 55.0
IOB_TAU2     = 70.0
COB_LAG_MIN      = 15.0
COB_HALFLIFE_MIN = 30.0
TAU_ACUTE_MIN = 30.0
TAU_EPOC_MIN  = 180.0

WIN_VOLATILITY = 6
WIN_TREND      = 12
WIN_TIR        = 24
WIN_ENTROPY    = 12
WIN_BASELINE   = 12
WIN_RESTING_HR = 96
WIN_STEPS      = 6
WIN_TDD        = 7 * 24 * 12


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generuj syntetyczne dane T1D z symulatora UVA/Padova"
    )
    p.add_argument(
        "--days", type=int, default=90,
        help="Liczba dni symulacji per pacjent (default: 90)"
    )
    p.add_argument(
        "--patients", type=str, default="all",
        choices=list(PATIENT_GROUPS.keys()),
        help="Grupa pacjentów do symulacji (default: all = 30 awatarów)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed losowości (default: 42)"
    )
    p.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help="Katalog wyjściowy"
    )
    p.add_argument(
        "--merge", action="store_true",
        help="Merge synthetic_population.parquet z training.parquet"
    )
    p.add_argument(
        "--no-feature-engineering", action="store_true",
        help="Zapisz surowe dane bez feature engineering (szybszy debug)"
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# BOLUS CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
def compute_meal_bolus(
    carbs_g:     float,
    current_bg:  float,
    carb_ratio:  float,   # g CHO / U insulin
    corr_factor: float,   # mg/dL / U insulin
    iob:         float,   # insulin on board [U]
) -> float:
    """
    Standardowy kalkulator bolusa basal-bolus:
        bolus = carb_bolus + correction_bolus - IOB

    carb_bolus     = carbs_g / carb_ratio
    correction     = (current_bg - target) / corr_factor  (tylko jeśli > target)
    net            = max(0, carb_bolus + correction - iob)
    """
    carb_bolus  = carbs_g / max(carb_ratio, 1.0)
    correction  = max(0.0, (current_bg - CORRECTION_TARGET_MGDL) / max(corr_factor, 1.0))
    net_bolus   = max(0.0, carb_bolus + correction - iob)
    # Ogranicz do rozsądnych wartości (max 20U per bolus)
    return float(np.clip(net_bolus, 0.0, 20.0))


# ─────────────────────────────────────────────────────────────────────────────
# MEAL SCHEDULE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_daily_meals(rng: np.random.Generator, day_idx: int) -> list[dict]:
    """
    Generuje posiłki dla jednego dnia.
    Zwraca listę słowników: {'hour': float, 'carbs_g': float}
    """
    meals = []

    for meal_name, (mean_g, std_g, min_g, max_g, base_hour, hour_std) in MEAL_PARAMS.items():
        if meal_name == "snack":
            continue

        # Czas posiłku z rozkładu normalnego ± 30-45 minut
        hour = float(np.clip(
            rng.normal(base_hour, hour_std),
            base_hour - 1.5,
            base_hour + 1.5,
        ))

        # Ilość węglowodanów — truncated normal
        carbs = float(np.clip(rng.normal(mean_g, std_g), min_g, max_g))

        # Weekendy: śniadanie trochę późniejsze, większy obiad
        if day_idx % 7 in (5, 6):
            if meal_name == "breakfast":
                hour += 1.0
                carbs *= 1.1
            elif meal_name == "lunch":
                carbs *= 1.15

        meals.append({"hour": hour, "carbs_g": round(carbs, 1)})

    # Przekąska (losowo)
    if rng.random() < SNACK_PROBABILITY:
        mean_g, std_g, min_g, max_g, _, _ = MEAL_PARAMS["snack"]
        snack_hour = float(rng.choice(SNACK_HOURS))
        snack_carbs = float(np.clip(rng.normal(mean_g, std_g), min_g, max_g))
        meals.append({"hour": snack_hour, "carbs_g": round(snack_carbs, 1)})

    return sorted(meals, key=lambda m: m["hour"])


# ─────────────────────────────────────────────────────────────────────────────
# CORE SIMULATION — jeden pacjent, N dni
# ─────────────────────────────────────────────────────────────────────────────
def simulate_patient(
    patient_name: str,
    n_days:       int,
    seed:         int,
) -> pd.DataFrame | None:
    """
    Symuluje jednego pacjenta przez n_days dni używając simglucose.
    Używa BBController z wbudowanymi parametrami CR/CF/basal dla każdego awatara.
    """
    try:
        from simglucose.patient.t1dpatient import T1DPatient
        from simglucose.sensor.cgm import CGMSensor
        from simglucose.actuator.pump import InsulinPump
        from simglucose.simulation.env import T1DSimEnv
        from simglucose.simulation.scenario import CustomScenario
        from simglucose.controller.basal_bolus_ctrller import BBController
    except ImportError:
        log.error(
            "simglucose nie jest zainstalowane. "
            "Uruchom: pip install simglucose --break-system-packages"
        )
        return None

    rng = np.random.default_rng(seed)

    # BBController z wbudowanymi parametrami per-awatar (CR, CF, u2ss, BW)
    # target=180 zamiast 140 — redukuje agresywność korekt i hypo rate
    controller = BBController(target=180)

    # Startowy timestamp
    start_date = datetime(2022, 6, 1) + timedelta(days=int(rng.integers(0, 30)))

    all_records = []

    for day_idx in range(n_days):
        day_start = start_date + timedelta(days=day_idx)

        # Generuj posiłki na ten dzień
        daily_meals    = generate_daily_meals(rng, day_idx)
        meal_schedule_min = {int(m["hour"] * 60): m["carbs_g"] for m in daily_meals}

        # Scenariusz dla simglucose — rozłóż każdy posiłek na 15 minut
        # żeby uniknąć jednorazowego spike'u bolusa
        MEAL_DURATION_MIN = 15
        scenario_meals = []
        for m in daily_meals:
            base_min   = int(m["hour"] * 60)
            g_per_min  = m["carbs_g"] / MEAL_DURATION_MIN
            for dm in range(MEAL_DURATION_MIN):
                scenario_meals.append((base_min + dm, g_per_min))

        try:
            scenario      = CustomScenario(start_time=day_start, scenario=scenario_meals)
            sensor        = CGMSensor.withName("Dexcom", seed=seed + day_idx)
            pump          = InsulinPump.withName("Insulet")
            fresh_patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed + day_idx)
            env           = T1DSimEnv(fresh_patient, sensor, pump, scenario)

            obs, _, _, _ = env.reset()
            current_cgm  = float(obs.CGM)

        except Exception as e:
            log.warning(f"  {patient_name} dzień {day_idx}: env reset failed ({e}), pomijam")
            continue

        # Symulacja minutowa — 1440 kroków = 24h
        minute_data       = []
        meal_schedule_min = {int(m["hour"] * 60): m["carbs_g"] for m in daily_meals}
        meal_delivery     = {}  # {minuta: g/min} — rozłożone posiłki
        MEAL_DURATION_MIN = 20  # posiłek rozłożony na 20 minut

        # Przygotuj harmonogram dostawy posiłków [g/min]
        for meal_min, carbs_g in meal_schedule_min.items():
            g_per_min = carbs_g / MEAL_DURATION_MIN
            for bm in range(meal_min, min(meal_min + MEAL_DURATION_MIN, 1440)):
                meal_delivery[bm] = meal_delivery.get(bm, 0.0) + g_per_min

        from collections import namedtuple as _nt

        for minute in range(1440):
            timestamp = day_start + timedelta(minutes=minute)

            # Posiłek w tej minucie [g/min] — rozłożony na MEAL_DURATION_MIN
            meal_g_per_min = meal_delivery.get(minute, 0.0)

            # BBController wylicza basal + bolus z kalibrowanych parametrów awatara
            # Hypo safety guard: nie dawaj bolusa jeśli CGM < 120 mg/dL
            meal_for_controller = meal_g_per_min if current_cgm >= 120.0 else 0.0
            action = controller._bb_policy(
                name=patient_name,
                meal=meal_for_controller,
                glucose=current_cgm,
                env_sample_time=1,  # sample_time=1 min
            )

            try:
                result      = env.step(action)
                current_cgm = float(result.observation.CGM)
                if result.done:
                    break
            except Exception as e:
                log.warning(f"  {patient_name} dzień {day_idx} min {minute}: step failed ({e})")
                break

            carbs_at_minute = meal_g_per_min  # g/min → zsumowane przy resample

            minute_data.append({
                "timestamp":         timestamp,
                "glucose_mg_dl":     current_cgm,
                "bolus_event_raw":   float(action.bolus),
                "meal_event":        carbs_at_minute,
                "basal_rate":        float(action.basal) * 60.0,  # U/min → U/h dla kompatybilności
            })

        if minute_data:
            log.info(f"    dzień {day_idx}: {len(minute_data)} minut → {len(minute_data)//5} oczekiwanych wierszy 5min")
            day_df = pd.DataFrame(minute_data)
            all_records.append(day_df)

    if not all_records:
        log.warning(f"  {patient_name}: brak danych po symulacji")
        return None

    raw_df = pd.concat(all_records, ignore_index=True)

    # ── Resample do 5min (jak w preprocess_ohiot1dm.py) ──────────────────
    raw_df = raw_df.set_index("timestamp")
    freq = "5min"

    # Glucose: średnia z 5 minut (CGM uśrednia)
    glucose_5min = raw_df["glucose_mg_dl"].resample(freq).mean()

    # Bolus: suma w oknie 5min
    bolus_5min = raw_df["bolus_event_raw"].resample(freq).sum()

    # Posiłki: suma w oknie 5min
    meal_5min = raw_df["meal_event"].resample(freq).sum()

    # Basal: ostatnia wartość (stały)
    basal_5min = raw_df["basal_rate"].resample(freq).last().ffill()

    df = pd.DataFrame({
        "timestamp":    glucose_5min.index,
        "glucose_mg_dl": glucose_5min.values,
        "bolus_event":   bolus_5min.values,
        "meal_event":    meal_5min.values,
        "basal_rate":    basal_5min.values,
    })

    # Usuń wiersze z NaN glucose (przerwy symulacji)
    df = df.dropna(subset=["glucose_mg_dl"]).reset_index(drop=True)

    if len(df) < 100:
        log.warning(f"  {patient_name}: za mało danych po resample ({len(df)} wierszy)")
        return None

    # Zerowe kolumny wearable — model obsługuje brak wearables przez fillna(0)
    for col in ["heart_rate", "gsr", "skin_temperature", "steps"]:
        df[col] = np.nan

    df["subject_id"]          = f"synth_{patient_name.replace('#', '_')}"
    df["source_split"]        = "synthetic"
    df["wearable_available"]  = 0.0   # flaga dla modelu

    log.info(f"  {patient_name}: {len(df):,} wierszy po resample do 5min")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (importowane z preprocess_ohiot1dm.py)
# ─────────────────────────────────────────────────────────────────────────────
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplikuje pełny pipeline feature engineering z preprocess_ohiot1dm.py.
    Wearable features będą = 0 (brak danych z symulatora).
    """
    import sys
    sys.path.insert(0, str(ROOT / "ml" / "scripts"))

    try:
        from preprocess_ohiot1dm import (
            add_time_features,
            add_calendar_windows,
            add_glucose_dynamics,
            add_glucose_regularity,
            add_glucose_risk_indices,
            add_glucose_momentum,
            add_glucose_cv,
            add_glucose_dfa,
            add_rolling_insulin_carb,
            compute_iob_cob,
            add_pk_features,
            add_insulin_stacking,
            add_tdd_features,
            add_meal_response_deviation,
            add_wearable_features,
            add_exercise_features,
            add_exercise_insulin_interactions,
            add_nocturnal_features,
            add_circadian_pk_interactions,
            add_glucose_nadir_proximity,
            add_basal_bolus_ratio,
            add_postprandial_phase,
            add_stress_glucose_interaction,
            add_glucose_asymmetry,
            add_step_insulin_offset,
            add_temp_drop_signal,
            add_iob_glucose_coinfall,
            add_meal_bolus_timing,
            add_hr_recovery_slope,
            add_glucose_momentum_divergence,
            add_cgm_gap_flag,
        )
    except ImportError as e:
        log.error(f"Nie można zaimportować preprocess_ohiot1dm: {e}")
        raise

    # Dodaj kolumny wymagane przez pipeline ale nieobecne w syntetykach
    if "exercise_duration_min" not in df.columns:
        df["exercise_duration_min"] = 0.0

    original_glucose = df["glucose_mg_dl"].copy()

    df = add_time_features(df)
    df = add_calendar_windows(df)
    df = add_glucose_dynamics(df)
    df = add_glucose_regularity(df)
    df = add_glucose_risk_indices(df)
    df = add_glucose_momentum(df)
    df = add_glucose_cv(df)
    df = add_glucose_dfa(df)

    df = add_rolling_insulin_carb(df)
    df = compute_iob_cob(df)
    df = add_pk_features(df)

    df = add_insulin_stacking(df)
    df = add_tdd_features(df)
    df = add_meal_response_deviation(df)

    df = add_wearable_features(df)
    df = add_exercise_features(df)
    df = add_exercise_insulin_interactions(df)
    df = add_nocturnal_features(df)
    df = add_circadian_pk_interactions(df)
    df = add_glucose_nadir_proximity(df)
    df = add_basal_bolus_ratio(df)

    df = add_postprandial_phase(df)
    df = add_stress_glucose_interaction(df)
    df = add_glucose_asymmetry(df)
    df = add_step_insulin_offset(df)
    df = add_temp_drop_signal(df)

    df = add_iob_glucose_coinfall(df)
    df = add_meal_bolus_timing(df)
    df = add_hr_recovery_slope(df)
    df = add_glucose_momentum_divergence(df)

    df = add_cgm_gap_flag(df, original_glucose)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Merge mode ────────────────────────────────────────────────────────
    if args.merge:
        ohio_path  = args.out_dir / "training.parquet"
        synth_path = args.out_dir / "synthetic_population.parquet"

        if not ohio_path.exists():
            log.error(f"Brak Ohio parquet: {ohio_path}")
            return
        if not synth_path.exists():
            log.error(f"Brak synthetic parquet: {synth_path}. Uruchom najpierw bez --merge.")
            return

        log.info("Mergowanie synthetic + Ohio...")
        ohio_df  = pd.read_parquet(ohio_path)
        synth_df = pd.read_parquet(synth_path)

        # Upewnij się że kolumny się zgadzają — dodaj brakujące z NaN/0
        for col in ohio_df.columns:
            if col not in synth_df.columns:
                synth_df[col] = 0.0
                log.info(f"  Dodano brakującą kolumnę: {col} = 0.0")

        for col in synth_df.columns:
            if col not in ohio_df.columns:
                ohio_df[col] = 0.0

        combined = pd.concat(
            [ohio_df, synth_df[ohio_df.columns]],
            ignore_index=True,
        ).sort_values(["subject_id", "timestamp"]).reset_index(drop=True)

        merged_path = args.out_dir / "training_with_synthetic.parquet"
        combined.to_parquet(merged_path, index=False, compression="snappy")

        n_ohio   = len(ohio_df)
        n_synth  = len(synth_df)
        log.info(f"Zapisano: {merged_path}")
        log.info(f"  Ohio:      {n_ohio:,} wierszy")
        log.info(f"  Synthetic: {n_synth:,} wierszy")
        log.info(f"  Razem:     {len(combined):,} wierszy")
        log.info(
            "\nTeraz uruchom trening z:\n"
            "  python ml/scripts/train_tft_population_v2.py \\\n"
            "    --epochs 50 --no-resume \\\n"
            "    --hidden-size 256 --attention-heads 8 \\\n"
            "    --dropout 0.25 --hidden-continuous-size 16 \\\n"
            "    --batch-size 128 --lr 7.57e-05\n"
            "\n(Pamiętaj żeby w load_and_preprocess_data() dodać filtr"
            " source_split in ['train', 'synthetic'] dla pretrainingu)"
        )
        return

    # ── Simulation mode ───────────────────────────────────────────────────
    patients_to_sim = PATIENT_GROUPS[args.patients]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("NMD — Synthetic Patient Generator (UVA/Padova via simglucose)")
    log.info(f"  Pacjenci     : {len(patients_to_sim)} ({args.patients})")
    log.info(f"  Dni          : {args.days} per pacjent")
    log.info(f"  Seed         : {args.seed}")
    log.info(f"  Output       : {args.out_dir}")
    log.info(
        f"  Estym. wierszy: ~{len(patients_to_sim) * args.days * 288:,} "
        f"(przed feature eng.)"
    )
    log.info("=" * 65)

    all_dfs = []
    failed  = []

    for idx, patient_name in enumerate(patients_to_sim):
        log.info(
            f"[{idx+1}/{len(patients_to_sim)}] Symulacja: {patient_name} "
            f"({args.days} dni)..."
        )
        patient_seed = args.seed + idx * 1000

        raw_df = simulate_patient(
            patient_name=patient_name,
            n_days=args.days,
            seed=patient_seed,
        )

        if raw_df is None:
            failed.append(patient_name)
            continue

        if not args.no_feature_engineering:
            log.info(f"  Feature engineering dla {patient_name}...")
            try:
                featured_df = apply_feature_engineering(raw_df)
                all_dfs.append(featured_df)
                log.info(
                    f"  {patient_name}: {len(featured_df):,} wierszy, "
                    f"{len(featured_df.columns)} kolumn"
                )
            except Exception as e:
                log.error(f"  Feature engineering failed dla {patient_name}: {e}")
                failed.append(patient_name)
        else:
            all_dfs.append(raw_df)
            log.info(f"  {patient_name}: {len(raw_df):,} wierszy (raw, bez FE)")

    if not all_dfs:
        log.error("Brak danych — wszyscy pacjenci failed.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = (
        combined
        .sort_values(["subject_id", "timestamp"])
        .reset_index(drop=True)
    )

    # ── Statystyki ────────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info(
        f"Synthetic dataset: {len(combined):,} wierszy, "
        f"{len(combined.columns)} kolumn, "
        f"{combined['subject_id'].nunique()} pacjentów"
    )
    if failed:
        log.warning(f"  Pacjenci którzy failed: {failed}")

    for pid in sorted(combined["subject_id"].unique()):
        mask     = combined["subject_id"] == pid
        n        = int(mask.sum())
        hypo_pct = (combined.loc[mask, "glucose_mg_dl"] < 70).mean() * 100
        hyper_pct = (combined.loc[mask, "glucose_mg_dl"] > 180).mean() * 100
        log.info(
            f"  {pid}: {n:,} wierszy | "
            f"hypo={hypo_pct:.1f}% | hyper={hyper_pct:.1f}%"
        )

    # ── Zapis ─────────────────────────────────────────────────────────────
    out_path = args.out_dir / "synthetic_population.parquet"
    combined.to_parquet(out_path, index=False, compression="snappy")
    log.info(f"\nZapisano: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    log.info("=" * 65)
    log.info(
        "\nNastępne kroki:\n"
        "  1. Sprawdź statystyki: otwórz synthetic_population.parquet\n"
        "  2. Merguj z Ohio:  python generate_synthetic_patients.py --merge\n"
        "  3. Uruchom pretraining na training_with_synthetic.parquet\n"
        "     (zmień filtr source_split w load_and_preprocess_data)"
    )
    log.info("=" * 65)


if __name__ == "__main__":
    main()