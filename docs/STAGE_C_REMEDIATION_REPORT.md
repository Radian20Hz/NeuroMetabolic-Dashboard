# Stage C remediation — synthetic-only

2026-09-14. Branch `research/baseline-audit`; HEAD `58c2e36dc108ef3d88de888a3fffc18c63af05bb`. Wejście clean, wynik dirty/uncommitted. Wykonano jawnie aktywowany `CODEX_STAGE_C_REMEDIATION_TASK.md`, z kontraktem `nmd-stage-c-contract-1`. Historyczny audyt i jego czerwone wyniki pozostają niezmienione.

## Werdykty

- **Implementation: PASS w wykonanym zakresie synthetic C-core remediation.** 16 technicznych findings otrzymało poprawki i regresje. Ostatni pełny acceptance: **19 PASS, 0 FAIL, 0 ERROR, 0 SKIP, 0 XFAIL**, rzeczywisty exit 0. Liczba testów oznacza nazwane grupy asercji, nie liczbę findings ani niezależnych eksperymentów.
- **Scientific validity: ograniczona do zgodności implementacji z ustalonym estymandem.** Within-subject temporal, conditional on eligibility; nieważone forecast occurrences, nakładające się okna nie są IID. Real performance, reprezentatywność, calibration/coverage i unseen-patient generalization **NOT EVALUATED**. Nie jest to empiryczny PASS całego Stage C.
- **Clinical interpretation: poprawiony język raportowania; clinical reliability NOT EVALUATED.** Brak automatycznych ocen bezpieczeństwa, progów clinical PASS i gwarancji skalibrowania. Pełna geometria Clarke pozostaje NOT EVALUATED.

C13-F01 pozostaje **ACCEPTED LIMITATION — WEIGHTING UNCHANGED**. C14-F01 pozostaje **NOT IMPLEMENTED/DEFERRED**; wykonano tylko obowiązek jawnych metadata `raw/not_fitted`, bez fitted source/role.

## Zmiana i przyczyna

`train_tft_population_v2.evaluate` deleguje do nowego `evaluation_stage_c` zamiast starej ścieżki z maskami, sanitizacją, zaokrąglaniem i częściowym persistence. Wejściem są jawny context i nowy katalog wyjściowy; wcześniejsze wywołanie bez provenance odmawia. Nie ma automatycznej ewaluacji, loadera danych ani fit kalibratora w tym module.

`qualify` tworzy oddzielną kopię indeksu evaluation W, pełne encoder48/decoder12, rzeczywiste timestamps i source role. `align` sprawdza returned lengths, pełną bijekcję kluczy, całą oś horyzontu, y względem źródła i końcowe CGM rzeczywistego encodera. Permutacja całych rekordów jest poprawna; niezgodny target nie jest zastępowany źródłowym. Uszkodzone obserwowane targety są błędem, a nie kryterium cichego wykluczenia. Role mają odpowiadać jawnie dostarczonej ramce ewaluacji; `source_split` nie może być dowolnie przemianowany w celu ominięcia tej bramki.

`persistence` odtwarza ostatni rzeczywiście obserwowany punkt z encoder context, zapisuje timestamp, age, ffill i mg/dL. Każde brakujące źródło unieważnia pełne porównanie. W raporcie wszystkie12 h, all-h, obie granularności strata, per-patient, micro i equal-patient macro mają point metrics i odpowiednie TFT probabilistic metrics. Persistence pozostaje point-only N/A. Różnice TFT−persistence używają tych samych keys, także per-person.

`reduce_metrics` akumuluje w FP64, bez clamp i selekcji przez predykcję. Finite p≤0 pozostaje w N z licznikiem. Nonfinite, niepoprawny target i overflow redukcji odmawiają. Raw crossings zachowują kolejność kwantyli, interval inversion unieważnia odpowiednią parę/grupę i jej macro. Macro crossing rates i macro maximum magnitude są średnimi odpowiednich statystyk osób; micro maximum magnitude pozostaje maksimum na całej grupie. Liczniki crossing są sumami, nie średnimi.

### C02 — dokładna przyczyna i minimalna korekta

Lokalny PF1.7.0 `_timeseries.py:1091–1148` dopasowuje/wykorzystuje categorical encoders i koduje group IDs **przed** `GroupNormalizer.transform` w okolicy1205. Poprzedni observed-train fit przekazywał surowe string IDs, a PF później integer IDs: lookup wpadał w `missing_` zamiast korzystać z parametrów osoby.

Nowy kod dopasowuje `NaNLabelEncoder` na treningowych identyfikatorach, podaje tę samą mapę do `subject_id` i `__group_id__subject_id`, a observed-train statystyki dopasowuje na zakodowanej kolumnie. Zachowano natywny `GroupNormalizer`, log, centering, standard estimator, ddof/epsilon PF oraz observed-only fitting. Nie zmieniono estymatora ani nie korygowano samego output scale po forward. Unseen subjects i brak observed train targets dla osoby odmawiają.

Rewizje: `observed-train-encoded-subject-v2`, `nmd-evaluation-stage-c-1`, protocol `nmd-baseline-v1.0-stage-c-1`. Nowy config `configs/baseline_v1_stage_c.json`; historyczny `baseline_v1.json` zachowany bajtowo. Parser domyślnie wskazuje nową rewizję i odmawia starego kontraktu; registry wymaga nowej semantyki również dla weights-only. Nie przeprowadzono migracji checkpointów.

**Wpływ naukowy:** poprawione kodowanie zmienia numeryczną funkcję modelu. Dawne wagi/wyniki nie są automatycznie zgodne z nową semantyką. Pełna recertyfikacja treningowa A/B po C02: **NOT RUN, osobne zadanie wymagające zgody na trening**. Nie wolno przedstawiać odziedziczonego CPU/CUDA replay PASS Stage B jako świeżego dowodu po C02.

## Macierz findings i regresji

Dokładne nazwy, statusy i dowody: [run02_results.json](../experiments/stage_c_remediation_20260914/run02_results.json). Stare R1/R2 pozostają immutable; nowe testy badają ten sam kontrakt przez nowy evaluator/helper. Nie przekonwertowano historycznych FAIL na XFAIL.

| Finding | Severity | Wynik | Zmienione symbole / dowód |
|---|---|---|---|
| C01-F01 | BLOCKER | FIXED | `qualify`, `align`; real PF/readback, coherent permutation, corrupt y/groups/time/lengths, nierówny i odwrócony batch |
| C02-F01 | HIGH | FIXED | `create_time_series_dataset`, `build_datasets`, registry contract; niezależne log stats, rzeczywiste encoder_cont/target_scale, poisoned fallback, renamed IDs, observed/assessment sentinels, unseen, owned serialization |
| C03-F01 | BLOCKER | FIXED | `finite`, `align`, `evaluate`; NaN/±Inf w każdym q i y, overflow, natywny model gate i osobna rzeczywista late-PF injection, brak valid publication |
| C03-F02 | BLOCKER | FIXED | `reduce_metrics`; y=.1/20/400/401, p=0/negative pozostają w N; y≤0 odmawia; MAE31/12 i MARD75% |
| C03-F03 | HIGH | FIXED | `qualify`, `align`; 108 candidates→12 pełnych W,96 krótkich wykluczonych; returned encoder24/decoder6 odmawia; role boundary; index Stage A bez mutacji |
| C04-F01 | LOW | FIXED | `reduce_metrics`, JSON publication/readback; sqrt250 bez rounding, FP64 oracle |
| C05-F01 | MEDIUM | FIXED | `report_metrics`, `macro`, `denominators`; nierówne N, micro≠macro RMSE, missing person, empty W/grupy, overlap counters |
| C06-F01 | MEDIUM | FIXED | `HORIZONS`, pełne panels; wszystkie12 h obu modeli, rzeczywiste timestamps origin+h, oracle matrix |
| C07-F01 | MEDIUM | FIXED | `report_metrics`;53.9/54/69.9/70/180/180.1/250/250.1, suma strata=rodzic, wszystkie przekroje |
| C08-F01 | BLOCKER | FIXED | `persistence`, `evaluate`, strict legacy alignment helper; baseline-specific fault przy poprawnym context zachowuje planned144, bez subset rescue |
| C08-F02 | MEDIUM | FIXED | `persistence`, encoder end check w `align`; przebudowane datasets age0/1/6, dokładny source/value/flag; age7/brak kontekstu/brak lookup/mismatch odmawiają |
| C10-F01 | HIGH | FIXED | `quantiles`, `interval_pair`, context/model/output gates; permuted/duplicate/missing/endpoint odmowa, native Q i brak90% |
| C11-F01 | HIGH | FIXED | `reduce_metrics`, `macro`; ties, any/adjacent/magnitude, inverted pair, invalid macro; pełna macierz przekrojów |
| C12-F01 | HIGH | FIXED | unweighted pinball/hit/PICP/width50/80/96, inclusive .75/width2/pinball.375; niezależna pełna macierz, persistence N/A |
| C13-F01 | MEDIUM | ACCEPTED LIMITATION | `INTERPRETATION`; production loss algebra i AST unchanged; weighted49/unweighted70, factor2, atom plateau; brak objective experiment |
| C14-F01 | MEDIUM | NOT IMPLEMENTED/DEFERRED | raw/not_fitted metadata, null fitted source/role; CQR small-n/ties/negative wyłącznie oracle definicji |
| C17-F01 | HIGH | FIXED | nowy evaluator/interpretation; perfect/poor/empty/invalid/raw crossing nie nadają safety labels; pięć legacy interior Clarke examples |
| C18-F01 | HIGH | FIXED w synthetic integration | context/owned receipt, pre-deserialization metadata/hash gates, actual model state, schema/normalizer/Q, input source/code/config, atomic completion chain, tamper i raw→metrics replay |

C18 `registered_best_context` korzysta z istniejącego `Registry.verified(..., "best", ...)`, bez osłabienia ownership lub tworzenia fikcyjnego trained BEST. **Rzeczywisty trained BEST→evaluation NIE BYŁ URUCHOMIONY**. Wykonany roundtrip dotyczy tylko własnego, jawnie `SYNTHETIC_UNTRAINED` state. Gateway do przyszłej jawnej ewaluacji real danych nie stanowi jej autoryzacji ani dowodu performance. Ewaluator w tej rewizji egzekwuje CPU FP32.

## Wykonanie, budżet, guardy

Prerejestracja: [preregistration.json](../experiments/stage_c_remediation_20260914/preregistration.json), SHA `d7bbeb33631687f2e80256393456b3b98748666911b90fc89288ac9ca5c59a23`. Zapis przed pierwszym runem. FP64 atol1e−10/rtol1e−8; FP32 atol1e−5/rtol1e−5, source y cast do FP32; keys/counts exact. Bez zmiany progów po wyniku.

Polecenie wykonane dwukrotnie z root repo:

```bash
python3 ml/tests/run_stage_c_remediation.py
```

Parent uruchamia `.venv/bin/python ml/tests/test_stage_c_remediation.py <nowy run>`, rejestruje source hashes przed importem, CPU strict FP32, seed42, workers0, eval/no_grad. `Popen.wait(timeout=600)` i SIGKILL process group zapewniają zewnętrzny hard watchdog; parent zapisuje rzeczywisty exit/time także przy import error/signal/timeout. Nie uruchamiano drugiego procesu przed ustaleniem zakończenia pierwszego.

| Run | PASS | FAIL | ERROR | SKIP/XFAIL | Forward calls | OS exit | Czas parent |
|---|---:|---:|---:|---|---:|---:|---:|
| run01 |17|1|0|0/0|10|1|7.225s|
| run02 |19|0|0|0/0|10|0|8.177s|

Pierwszy FAIL był w fixture unseen: nadanie obu osobom tej samej nazwy tworzyło duplikaty czasu. Bramka dense timeline odmawiała wcześniej, więc nie był to dowód testu unseen. Poprawiono fixture do jednej osoby z nowym ID; zachowano czerwony wynik. Przy drugim przebiegu dodatkowo rozszerzono pełny niezależny oracle paneli i statycznie domknięto provenance/kompatybilność. Nie zmieniono tolerancji, lossu, architektury ani fixture podstawowego forwardu.

Cała kampania: **20/24 forward batch calls,12/64 unique selected windows,0 backward,0 optimizer steps**.108 candidate index rows to metadane indeksu, a nie okna podane modelowi.144 occurrences,12 windows,34 unique(subject,target_timestamp),2 subjects. Natychmiastowy durable counter zapisuje wywołania, również te prowadzące do odmowy finite gate. Pozostałych4 calls nie wykorzystano.

Python audit hook przed importami odmawia real-data/model open, unlisted data artifacts i zapisów poza własnym runem/campaign ledger. Dodatnia kontrola forbidden parquet/XML/checkpoint oraz main/load_test_data/Trainer.fit/backward/optimizer. Import Optuny zablokowany. State modelu przed/po exact hash zgodny; gradients None. **Audit hook nie jest OS sandbox dla dowolnego natywnego I/O**; nie wywołano natywnych real-data loaderów. Żaden canonical parquet ani trained checkpoint nie został otwarty lub zahashowany. Nie instalowano bibliotek.

Environment: Python3.14.7, torch2.11.0+cu130, Lightning2.6.1, PF1.7.0, NumPy2.4.4, pandas2.3.3; CPU, jeden wątek. Dokładny profil i wersje w [execution_summary.json](../experiments/stage_c_remediation_20260914/execution_summary.json).

## Artefakty i kontrola

Ignorowane binarne/log artifacts: `ml/models/stage_c_remediation_20260914/run01` i `run02`. Każdy run ma source hashes, console, guard events, results, environment; state `.pt` jest synthetic/untrained. `integration/` wiąże `raw.json`, `metrics.json`, `manifest.json` i atomowy `completion.json`. Oddzielne `invalid_finite`, `invalid_baseline`, `native_finite`, `late_finite` zachowują INVALID/planned N, bez completed valid pointer. `empty` nie wywołuje modelu i ma EMPTY, nie quality PASS.

Główne run02 hashes:

| Element | SHA-256 |
|---|---|
| In-memory source fixture | `8af9b6d996f8fbcbac4b5daced18d621e17b9859e00b4092f3583d05a2146683` |
| Ordered keys | `b62692d81096cdccb3f4c34f83a45ad2956c5885e55fec63f9010cf2561ea093` |
| Owned model file | `20dd0c455981933334a359ccb935550c601a8cd60df240c5a336a62d78af114c` |
| Actual model state | `393be948772609de69af7212aa14397e82bbfca1d1a11c72f5053b264adaeaf2` |
| Normalizer semantic state | `b67a22ddc284ef6a0dbff26286b0ab8e460aaf190b3de52aa3c1cbf0956aed40` |
| Raw predictions/keys/lineage | `131d74616321dd5c8509e3f2763f5ad84fac9158497e70bd0872a49ba5d8bc21` |
| Metrics | `b768a4a5591562758763a1e96b5dfbfecbc565486978e2eca7cc7c14412ef7e4` |

Szczegółowe code/config hashes i wyniki w wersjonowanym `experiments/stage_c_remediation_20260914/`. Przed raportem potwierdzono zgodność wszystkich finalnych zmienionych źródeł Python z snapshotem run02, AST `ClinicalQuantileLoss` względem HEAD bez zmian i historyczny config bez zmian. Składnia przez `ast.parse`, rzeczywiste importy w guarded child, `git diff --check` PASS. Brak broad Stage A/B discovery, ponieważ zawiera niedozwolony trening. Nie nadpisano historycznych audit R1/R2 ani wyników.

## Pliki zmienione

- `ml/scripts/train_tft_population_v2.py`: mechaniczne kodowanie normalizatora, nowa jawna rewizja configu, strict legacy alignment, delegacja evaluator.
- `ml/scripts/checkpoint_registry.py`: nowy protocol/normalizer revision i odmowa historycznej semantyki; natywna ownership verification zachowana.
- `ml/scripts/evaluation_stage_c.py`: W/S, context, alignment, persistence, reducers, reporting i atomic artifacts.
- `configs/baseline_v1_stage_c.json`: osobny kontrakt; training recertification NOT RUN.
- `ml/tests/run_stage_c_remediation.py`, `ml/tests/test_stage_c_remediation.py`: watchdog/guards i synthetic acceptance.
- `experiments/stage_c_remediation_20260914/`: preregistration, oba wyniki, execution summary i README.
- `docs/STAGE_C_REMEDIATION_REPORT.md`: ten raport.

Git HEAD bez zmian; branch `research/baseline-audit`; dirty/uncommitted, brak staging/commit/push. Dane pacjentów, modele i logi nie są przeznaczone do commitu.

## Pozostałe ograniczenia i STOP

Brak treningowej recertyfikacji po C02, trained BEST integration, CUDA evaluation, real-data performance/calibration/representativeness, unseen-patient evaluation, pełnych Clarke boundaries i ARIMA convergence. Legacy helper ARIMA pozostaje w kodzie, lecz nowy obowiązkowy raport nie uruchamia opcjonalnego fit; dotyczy TFT i persistence. Prawidłowe testy syntetyczne nie rozstrzygają empirycznego wpływu clinical weighting. Nie wdrożono calibration fit/apply, crossing postprocessing, objective variants, resamplingu ani nowych ról development/calibration.

**STOP na raporcie do review.** Nie rozpoczęto treningu, Optuny, C-dev, real test evaluation, kalibracji ani dalszego etapu.
