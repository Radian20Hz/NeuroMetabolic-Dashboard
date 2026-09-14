# Next session — Stage C audit

- **Stage A PASS** — 31 PASS; dane i missingness protocol zamrożone.
- **Stage B PASS** — końcowe38 PASS; CPU strict / CUDA seeded FP32, SWA OFF. Aktualizacja na początku STAGE_B_FINALIZATION_REPORT.md zamyka historyczny FAIL. Nie gwarantujemy bitwise CUDA ani mid-epoch resume.
- **Stage C PLAN READY — EXECUTION NOT STARTED.** W sesji kontraktowej nie wykonano audytu, testów, treningu ani zmian produkcyjnych.

## Canonical artifacts

- `configs/baseline_v1.json`: protocol `nmd-baseline-v1.0-stage-b-2`, context48/horizon12, clinical weighting bez zmian, automatic_test_evaluation=false.
- Dataset: `ml/data/processed/baseline_v1_stage_a_20260913T115538143463Z/`; provenance commit `5ee8d930e80fd7d62d0248c52ed10aaaaaa7f8d0`, git_dirty=false; parquet SHA-256 `d14fe31b1b81971639ca8d5ba17b4882fbd6711569e30785f0da7b91a0c031cf`. Starszy katalog112536 w historycznym raporcie A nie zastępuje tego artefaktu. C-core nie otwiera real parquet.
- Dowody A/B: `docs/BASELINE_AUDIT_STAGE_A_REMEDIATION.md`, `docs/STAGE_B_FINALIZATION_REPORT.md`. CPU final suite: `ml/models/stage_b_remediation/20260913T180459Z/`; CUDA callback restore: `ml/models/stage_b_callback_restore/20260913_callback_01/`.
- Specyfikacje: `docs/STAGE_C_AUDIT_PLAN.md`, `docs/CODEX_STAGE_C_TASK.md`, **`docs/STAGE_C_EXECUTION_CONTRACT.md` (`nmd-stage-c-contract-1`)**. Kontrakt rozstrzyga wcześniejsze D1–D2; nie pytać o nie ponownie.

## Zatwierdzone kontrakty C-core

- Pełne kwalifikowane48/12 windows Stage A; każdy decoder target obserwowany. Częściowo nieobserwowany/krótszy decoder wyklucza całe okno z głównego raportu. TFT i persistence te same subject/window/horizon keys i denominators; awaria modelu nie wybiera korzystnego subsetu.
- Persistence=ostatni rzeczywiście dostępny obserwowany CGM w encoder context, stały na12 h; zgodny z przyczynowym ffill≤6 Stage A. Bez przyszłych wartości lub dodatkowego filtrowania okien.
- MAE, RMSE, bias, MARD; nieważony pinball/hit per q; raw interval coverage/width i crossing. Wszystkie12 horyzontów, micro, macro patient i pełne per-patient z N; strata główne i szczegółowe. Brak progów clinical PASS.
- Strata: y<54;54≤y<70;70≤y≤180;180<y≤250;y>250 mg/dL; główne hypo<70/target70–180/hyper>180. Bez zaokrąglania przed przydziałem.
- Q=.02/.10/.25/.50/.75/.90/.98; central intervals50/80/96%. Brak sorting lub automatycznego calibration fit. ClinicalQuantileLoss bez zmian; analiza wag matematyczna i syntetyczna. Objective variants i real calibration split/metoda pozostają osobnym zadaniem po audycie.
- Audit-only C00–C19: synthetic fixtures, niezależne oracles, mały prawdziwy forward; zero optimizer/backward, real-data evaluation, test access, Optuny i remediation. Budżet/tolerancje według tasku. Findings Cxx-Fyy przed naprawami; brak funkcji lub empirycznych dowodów nie staje się PASS dzięki oracle.

## Dokładnie jeden następny krok

**Uruchomić `docs/CODEX_STAGE_C_TASK.md` w zakresie C-core, z obowiązującym `docs/STAGE_C_EXECUTION_CONTRACT.md`.** Audyt zakończyć raportem findings; nie przechodzić samodzielnie do remediation, C-dev lub test evaluation.
