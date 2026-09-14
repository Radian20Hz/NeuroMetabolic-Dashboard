# Baseline v1.0 — Stage C C-core audit

2026-09-14. **C-core AUDIT COMPLETE — implementation FAIL. Remediation NOT STARTED.**

Wykonano inspekcję oraz syntetyczne oracles/unit/integration dla C00–C19. **18 findings: 4 BLOCKER, 7 HIGH, 6 MEDIUM, 1 LOW.** Szczegóły i pełne pola również w `findings.json`. Wszystkie propozycje napraw są niewykonane i wymagają osobnego review/zlecenia. Formalne historyczne A/B PASS pozostają zapisami ich ówczesnego zakresu; nowy C02-F01 ujawnia dodatkową lukę na styku datasetu i normalizacji, której tamte dowody nie zamykają.

- **Implementation correctness: FAIL.** Działają natywne predict, dopasowanie source targetów, inverse math, permutacje batchy w ustalonej tolerancji, finite gate wewnątrz modelu i badane baseline helpers. Nie są spełnione kontrakty populacji, normalizatora grupowego, kompletności raportowania ani granic evaluator.
- **Scientific validity: FAIL dla zgodności obecnego raportowania z kontraktem; real performance/calibration NOT EVALUATED.** Weighted objective estymuje inne kwantyle; kwalifikacja okna definiuje warunkową populację. Syntetyczne dowody nie mierzą wielkości rzeczywistego bias, kalibracji lub generalizacji.
- **Clinical interpretation: FAIL dla komunikatów. Clinical reliability NOT EVALUATED.** „clinically SAFE” jest odtwarzalne na wymyślonych dokładnych predykcjach. Nie przeprowadzono walidacji klinicznej.

## Punkt wejścia i aktywacja

Branch `research/baseline-audit`; HEAD `4cc11d8fd7704cf3baeb4301a3b722374aa94fbd`; początkowy `git status --porcelain` pusty. Lokalny status tracking nie wykazał ahead/behind względem zapisanej referencji origin; **nie wykonano fetch**, więc nie jest to świeża weryfikacja serwera. Względem `95c2723316d29d984c4aba2884c7bdb0603e3ba8` wyłącznie cztery nowe dokumenty Stage C, +356 linii, bez zmian produkcji.

D0 aktywowane bezpośrednim zleceniem użytkownika. D1/D2 rozstrzygnięte przez `nmd-stage-c-contract-1`, bez ponownego pytania. CPU strict, FP32, SWA OFF, context48/horizon12, Q=.02/.10/.25/.50/.75/.90/.98. Nie zmieniano baseline config, tolerancji, splitów, lossu, architektury produkcyjnej ani checkpoint semantics.

Przeczytane źródła (hash każdego w configu):
- `AGENTS.md`
- `docs/RESEARCH_WORKFLOW.md`
- `docs/ROADMAP.md`
- `docs/BASELINE_AUDIT_STAGE_A_REMEDIATION.md`
- `docs/BASELINE_STAGE_B_REMEDIATION.md`
- `docs/STAGE_B_FINAL_REVIEW.md`
- `docs/STAGE_B_FINALIZATION_REPORT.md`
- `docs/STAGE_C_AUDIT_PLAN.md`
- `docs/CODEX_STAGE_C_TASK.md`
- `docs/STAGE_C_EXECUTION_CONTRACT.md`
- `docs/NEXT_SESSION_BRIEF.md`
- `configs/baseline_v1.json`

Kanoniczny parquet pozostaje **nieotwarty**. SHA `d14fe31b1b81971639ca8d5ba17b4882fbd6711569e30785f0da7b91a0c031cf`, source provenance `5ee8d930e80fd7d62d0248c52ed10aaaaaa7f8d0`, dirty=false, oraz A31/B38 PASS to **inherited evidence z dokumentów**, nie ponowne testy/hashowanie danych. Historyczna sekcja FAIL w finalizacji B jest zastąpiona aktualizacją callback restore PASS.

## Ścieżka wykonania i skutki uboczne

`create_time_series_dataset` → fitted observed-train `GroupNormalizer(log)` → PF index → `filter_observed_windows` → `assert_observed_evaluation` → `TimeSeriesDataSet.to_dataloader` → prawdziwy `ClinicalTFT.predict(mode=quantiles, return_x/y=True)` → PF PredictCallback → `evaluate` → scalar/quantile-hit/Clarke → persistence/optional ARIMA → płaski dict/log.

PF `_tft.py:670` wywołuje `transform_output`; `_base_model.py:628–658` odwraca normalizator. Callback przy `quantiles` maskuje padding NaN i łączy output/x/y w tej samej kolejności (`_base_model.py:212–335`). Evaluator nie wykonuje drugiego inverse. `x_to_index` jest używane do dekodowania osób dla baseline. To poprawne elementy; nie usuwają potrzeby weryfikacji kompletnego W i zwróconych targetów.

Import modułu rejestruje safe globals, logging/filters i wykonuje `mkdir(exist_ok=True)` istniejącego katalogu modeli; nie czyta checkpointów ani danych. `main` nie jest wywołany; jego treść wyłącznie przeczytana. Certified main ma automatic test evaluation OFF. Brak fit/apply calibration na tej ścieżce stwierdzono przez pełny przepływ wywołań i return, a nie tylko wyszukanie słowa. Backend legacy jest odrębną ścieżką, nie uruchomiono go ani jego ładowania modeli.

## Prerejestracja, fixtures i izolacja

Config `configs/baseline_stage_c_audit.json` zapisany przed pierwszym ML runem; SHA **30318ca5e900c09bbce1ea82efdbe4750009738db23b36f3ffd8a1631f045630**. Zawiera wzory, role, wartości Q/strata, seed42, units, sources i budżet. `followup_preregistration.json` opisuje uzasadnienie dodatkowych probes, przy tych samych tolerancjach.

Fixture: wyłącznie deterministycznie wymyślone dwie osoby, po65 regularnych binów, te same lokalne time_idx0…64, trajektorie `75+.8i` i `180+1.5i`, cechy syntetyczne. Normalizator fit wyłącznie synthetic_train. Assessment to odrębna syntetyczna rola (te same generowane wartości są celowe; brak twierdzenia o niezależnej ocenie jakości).

Produkcyjny validation index ma108 kandydatów, w tym96 krótkich encoderów; jego inspekcja jest dowodem C03-F03. Do forward wybiera się audytowym `dataset.filter` **12 pełnych okien48/12**, nie zmieniając produkcyjnego filtra. Wszystkie runy używają tych samych12 unikalnych okien. Limit64 interpretowano jako okna fixture podawane modelowi; pełną listę108 candidate metadata badano bez forward. Jest to jawne ograniczenie interpretacji budżetu, a nie twierdzenie, że produkcyjny index ma≤64 wiersze.

W:12 windows, S:144 forecast-target occurrences,34 unique `(subject,target_timestamp)`,2 subjects. Manifest keys przechowuje prawdziwe syntetyczne timestamps, window_id i h, bez traktowania offsetu indeksu jako czasu kalendarzowego.

Guard przed importami ML odrzuca Python `open` pod repo `ml/data/` i `ml/models/` oraz nieallowlistowane pliki danych/modeli. Własny OUT jest allowlistą. Dodatnia próba canonical parquet, raw XML i obcego checkpointu kończy się **PermissionError przed otwarciem**. Nie wymagano istnienia dwóch sentineli; ścieżka canonical jest realna i również odrzucona. Po imporcie blokowane main/loaders/Trainer.fit/backward/optimizer oraz import Optuny. Testy dodatnie tych entrypoints są zapisane w guard_events. Synthetic test sentinel nie zmienia audit train/val/cal state ani ponownie dopasowanego train normalizatora; **nie jest to test nieistniejącego produkcyjnego kalibratora**.

Procesy poza sandboxem zachowują te guards; nie twierdzimy, że audit hook jest izolacją OS dla dowolnego natywnego kodu. Nie wywołano natywnych real-data loaderów. Brak rzeczywistych danych w jakimkolwiek fixture/logu. State modelu i brak gradientów potwierdzono przed/po. Nie instalowano zależności.

## Wykonania, błędy środowiska i liczności

| Run | PASS | FAIL | ERROR testów | SKIP/XFAIL | Forward batch calls | Czas runnera | Exit procesu |
|---|---:|---:|---:|---:|---:|---:|---|
| run01 | 23 | 22 | 0 | 0/0 | 8 | 109.835s | 143 (SIGTERM); manifest planował1 |
| run02 | 23 | 22 | 0 | 0/0 | 8 | 4.761s | 1 |
| run03 | 8 | 5 | 0 | 0/0 | 5 | 4.303s | 1 |

Łącznie **103 wykonane checks:54 PASS/49 FAIL,0 ERROR/0 SKIP/0 XFAIL;57 różnych nazw testów (30 PASS/27 FAIL)**. Run02 powtarza45 testów run01; follow-up ma13 checks, w tym powtórzoną integrację. Nie sumować powtórzeń jako nowych niezależnych dowodów. FAIL są zwykłymi asercjami kontraktu, nie xfail i nie zamkniętymi findings.

Pierwszy import torch._C w sandboxie był znacznie opóźniony. Oddzielny diagnostyczny import został przerwany po15s przez faulthandler (exit1,0 forward); import poza sandboxem przeszedł. Zlecono zatrzymanie własnego procesu39493 i powtórkę poza sandboxem. Odczyt końcowy ujawnił, że run01 **zdążył zapisać cały wynik i manifest** przed SIGTERM; końcowy OS exit143 nie zgadza się z planowanym w manifest exit1. Nie przedstawiamy go jako czystego zakończenia ani jako0 forward. Follow-up preregistration podaje wcześniejsze8 calls, ponieważ przed odczytem pełnych artefaktów run01 jego wykonania nie doliczono; rzeczywisty stan to16 przed follow-up +5 =21. Oryginalny dokument zachowano bez przepisywania. Zachowano wszystkie pliki i rzeczywisty licznik. Było to ograniczenie diagnostyki środowiska/instrumentacji, nie naprawa produkcji.

Łącznie **21/24 rzeczywistych forward batch calls**,0 backward,0 optimizer steps, CPU only. Watchdog600s per runner zachowany. FP64 oracle atol1e−10/rtol1e−8; FP32 integration atol1e−5/rtol1e−5, keys/counts exact. Maksymalna różnica predykcji między batch partitions wyniosła9.1552734375e−5 mg/dL; mieści się w pełnej regule atol+rtol·abs(reference) przy tej skali. Nie wymagano bitwise różnych batch partitions i nie zwiększano tolerancji.

Python3.14.7, torch2.11.0+cu130, NumPy2.4.4, pandas2.3.3, Lightning2.6.1, PF1.7.0; jeden wątek, workers0, CUDA_VISIBLE_DEVICES puste, CPU strict i effective numerical profile w manifestach. Statsmodels unavailable; nie instalowano go. ARIMA miała kontrolowany helper success/failure, bez rzeczywistego dopasowania modelu ARIMA.

## Macierz C00–C19

Status w kolumnie runner oznacza asercje; NOT IMPLEMENTED/NOT EVALUATED dotyczą capabilities/dowodów naukowych. Pełna lista metod i tracebacks: `run02/results.json`, `run03/results.json`.

| Obszar | Dowód i kontrakt | Status / finding |
|---|---|---|
| C00 isolation | Dodatnia blokada plików/entrypoints, synthetic sentinel | PASS w zakresie opisanych guards; calibration fit nie istnieje |
| C01 alignment | Real PF keys/targets, batch5 z ostatnim2, reverse batch4; corrupt-y probe | plumbing PASS; boundary FAIL C01-F01 |
| C02 units | Native real-space, tuple weight,×2, analytic inverse; per-group parameters | częściowo PASS; FAIL C02-F01 |
| C03 eligibility/finite | NaN/±Inf, valid-y NaN,≤0 pred,y20/400/401, unobserved flags,short48/12 | unobserved i native model guard PASS; evaluator FAIL C03-F01/F02/F03 |
| C04 scalars | Hand MAE15/RMSE√250/MARD20/bias−5, nearzero/zero,empty,precision | oracle PASS; precision FAIL C04-F01; domain/empty C03-F02/C05-F01 |
| C05 aggregation | Unequal patient N, micro vs macro,empty | oracle PASS; NOT IMPLEMENTED C05-F01 |
| C06 horizons | Unique h error increments, all12 expected | FAIL C06-F01 |
| C07 strata | Floating boundary oracle, actual evaluator inventory | oracle PASS; NOT IMPLEMENTED C07-F01 |
| C08 persistence | Two-subject ramp,ffill age1,1/12 missing,absent decoder index | happy path PASS; FAIL C08-F01; lineage NOT IMPLEMENTED C08-F02 |
| C09 ARIMA | Helper exception→None, noncontiguous success indices1… excluding0/7, exact raw history | controlled tests PASS; actual library/convergence NOT EVALUATED; no new baseline |
| C10 Q | Model-Q mismatch, endpoint oracle | FAIL C10-F01; 50/80/96 oracle PASS |
| C11 crossing | raw inversion,ties,any/adjacent/magnitude | oracle PASS; NOT IMPLEMENTED C11-F01 |
| C12 intervals | inclusive endpoints,hit,width,pinball; production inventory | oracle PASS; NOT IMPLEMENTED C12-F01 |
| C13 weighted objective | Analytic CDF, quadrature, loss factor2, atom plateau,allQ | PASS mathematics; limitation C13-F01; trained effect NOT EVALUATED |
| C14 calibration | Full path inspection,CQR k/small n/ties/negative score | oracle PASS; fit/apply NOT IMPLEMENTED C14-F01; real coverage NOT EVALUATED |
| C15 dependence |144 occurrences vs34 unique targets | oracle PASS; empirical uncertainty NOT EVALUATED; production counters C05-F01 |
| C16 selection |12→6 retained,6 missing occurrences removes72,66 observed discarded | synthetic PASS; real representativeness/bias NOT EVALUATED |
| C17 clinical | Perfect fixture logs,5 interior Clarke examples; primary source review | wording FAIL C17-F01; full Clarke boundaries/clinical reliability NOT EVALUATED |
| C18 provenance | Owned untrained state/hash roundtrip,4 registry mismatch controls, evaluator inventory | helper/roundtrip PASS; linkage NOT IMPLEMENTED C18-F01 |
| C19 integration | Three separate actual runs, no_grad,state unchanged; forced-edge path separate | plumbing PASS; review COMPLETE with FAIL; no remediation |

## Findings — evidence, reproducer i propozycje

W każdym findingu `train_tft_population_v2.py` oznacza `ml/scripts/train_tft_population_v2.py` w HEAD podanym powyżej. Linie PF dotyczą lokalnego stosu1.7.0 w `.venv/lib/python3.14/site-packages/pytorch_forecasting/`. Evidence commit i hashes są również zapisane osobno w JSON. Hash fixture/predictions w findingu identyfikuje bazową fixture; każda deterministyczna perturbacja jest zdefiniowana w wskazanej funkcji zahashowanego runnera. Nie udajemy osobnego raw-prediction capture dla każdej wymuszonej perturbacji.

Reproducer R1 to `tests/run_stage_c.py`, R2 to `tests/followup_stage_c.py`, komendy poniżej. Nazwa testu identyfikuje minimalną funkcję/assertion i fixture perturbation w tym runnerze; nie trzeba uruchamiać żadnego entrypointu treningowego. Wszystkie reproduktory używają wyłącznie syntetycznej fixture. FAIL runtime oznacza exit1 suite przy zwyczajnym zwróceniu wyniku przez evaluate, jeśli nie wskazano inaczej. Dla NOT IMPLEMENTED asercja sprawdza brak capability w pełnym zwracanym wyniku, a mapa kodu potwierdza brak wywołania tej capability.

### C01-F01 — Brak weryfikacji targetów względem kanonicznych keys

**BLOCKER · CONFIRMED · implementation**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:evaluate 2191–2198,2352–2388`. Reproducer: `run03:C01_target_key_mismatch`.

**Expected:** Każdy zwrócony target odpowiada subject/origin/h w źródłowym W; mismatch → INVALID.

**Actual:** Odwrócenie kolejności predictions.y bez zmiany x/output nadal zwraca metryki. Natywny nieuszkodzony PF przeszedł alignment i permutację.

**Scientific impact / uzasadnienie severity:** Wykazano brak ochrony granicy evaluate przed błędną parą. Nie wykazano spontanicznego mieszania targetów przez PF. Taka awaria unieważnia całe porównanie.

**Minimal proposed remediation — NIE WYKONANO:** Jawny manifest keys i walidacja targetów/lengths/source role przed redukcją.

**Future regression / DoD:** Perturbacja tylko y lub keys musi odmówić; prawidłowa permutacja wszystkich pól razem ma zachować wynik.

### C02-F01 — GroupNormalizer używa fallback zamiast statystyk osoby

**HIGH · CONFIRMED · implementation / scientific**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:create_time_series_dataset 1678–1690; PF encoders.py:get_parameters 1295–1306; timeseries/_timeseries.py 2134`. Reproducer: `run02:C02_group_transform`, `run03:C02_subject_normalizer_parameters`, `run03:C02_distinct_inverse_math`.

**Expected:** Dwie znane osoby otrzymują własne obserwowane-train center/scale, zgodne z mapowaniem PF.

**Actual:** Dopasowane A=(4.5997919368,0.1537551768), B=(5.4216211461,0.1267248543). Wszystkie 12 target_scale=(5.0107064247,0.1402400136), tj. missing_ po FP32. norm_ ma string keys, PF przekazuje grupy 0/1. Niezależna matematyka inverse dla dwóch ręcznych skal PASS.

**Scientific impact / uzasadnienie severity:** Kontrakt normalizacji grupowej nie jest realizowany na fixture. Brak dowodu wpływu na historyczne wyniki, nie twierdzimy double inverse ani niewłaściwych jednostek w całym evaluatorze. Znacząca luka na styku A/C.

**Minimal proposed remediation — NIE WYKONANO:** Uzgodnić kodowanie kluczy podczas fit/transform/get_parameters przez wspierane API PF, zachowując observed-train-only fitting; osobny review wpływu na baseline/provenance.

**Future regression / DoD:** Porównać batch target_scale i normalizowane wejścia do niezależnego per-subject oracle; zapewnić brak fallback dla znanych osób i osobne zachowanie unseen. Nie uznawać samego fitted norm_ za dowód.

### C03-F01 — Zastępowanie NaN/Inf i usuwanie uszkodzonych targetów

**BLOCKER · CONFIRMED · implementation**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:evaluate 2205–2226,2247–2256`. Reproducer: `run02:C03_prediction_nan/posinf/neginf`, `run02:C03_target_nan`, `run03:C03_real_predict_late_corruption`, `run03:C03_native_nonfinite_model_gate`.

**Expected:** Nonfinite na ważnej pozycji → INVALID z pełnym N; padding osobno.

**Actual:** NaN→0 i N12→11, MAE60=0; +Inf→400 (MAE23.15); −Inf→40 (MAE6.85); NaN target usuwa wiersz. Real PF z fault hook po FiniteModel: dwa NaN → N10, zwykłe metrics. Fault wewnątrz modelu jest prawidłowo odrzucany przez istniejący FiniteModel.

**Scientific impact / uzasadnienie severity:** Ochrona modelu nie usprawiedliwia sanitizacji na granicy evaluator. Kontrolowana awaria może dać korzystniejszy wynik albo skończoną fikcyjną prognozę. Nie dowiedziono naturalnego występowania tych awarii w treningu.

**Minimal proposed remediation — NIE WYKONANO:** Fail-fast/INVALID przed nan_to_num i redukcją; utrzymać rozróżnienie ważnych pozycji i padding.

**Future regression / DoD:** NaN/±Inf output i target w granicy PF→evaluate muszą unieważnić wynik; existing model gate nadal PASS.

### C03-F02 — Nieuzgodniona selekcja przez y i predykcję

**BLOCKER · CONFIRMED · implementation / scientific**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:evaluate 2247–2256,2282–2291`. Reproducer: `run02:C03_finite_negative_retained`, `run02:C03_targets20_400_401`, `run02:C04_zero_target_invalid`, `run03:C04_nearzero_positive`, `run03:C03_finite_zero_retained`.

**Expected:** Finite p≤0 pozostaje błędem; wszystkie y>0 są w mianowniku; y≤0 → INVALID.

**Actual:** p=−10 lub0: wiersz odrzucony. y=[20,400,401,100×9] → N9 i MAE0 zamiast N12 i MAE31/12. y=.1 odrzucony; y=0 pominięty bez INVALID.

**Scientific impact / uzasadnienie severity:** Model wybiera własną ocenianą populację; błędy na skrajnych wartościach są ukrywane. Dotyczy też mianowników marginal hits i persistence, więc główne porównanie nieważne.

**Minimal proposed remediation — NIE WYKONANO:** Usunąć nieuzgodnione maski z raportowania; walidować y>0; flagować niefizjologiczne finite pred bez usuwania.

**Future regression / DoD:** Exact N i błędy dla p<0,p=0,y=.1/20/400/401; y≤0 odmawia. W i hash keys identyczne dla modeli.

### C03-F03 — Guard ewaluacji nie wymaga pełnego 48/12

**HIGH · CONFIRMED · implementation**. Klasyfikacja: technical bug.

**Evidence:** `observed_windows.py:assert_observed_evaluation 73–78; train_tft_population_v2.py 1683–1684,1691–1693,2244–2245`. Reproducer: `run02:C03_full48_guard`, `run02:C03_decoder_lengths_padding`.

**Expected:** Główna ewaluacja tylko pełny encoder48 i decoder12. Krótsza fixture pomocnicza odrzucona lub poprawnie maskowana.

**Actual:** Produkcyjny builder daje 108 candidate validation index rows, w tym96 encoderów<48; guard je akceptuje. Zmiana returned decoder_lengths[0]=6 nie zapobiega liczeniu padding jako targetów; MAE60=8.3333. Audyt forward używa osobnego pełnego subsetu12.

**Scientific impact / uzasadnienie severity:** Estymand miesza różne długości historii; padding może trafić do metryk. To luka ewaluacyjna, nie zgoda na zmianę krótkich decoderów treningowych Stage A.

**Minimal proposed remediation — NIE WYKONANO:** Egzekwować 48/12 wyłącznie dla W ewaluacji i spójność returned lengths z manifestem.

**Future regression / DoD:** Naturalnie krótkie encodery i controlled short decoder odmówione przed metrykami; główny N nie zależy od padding values.

### C04-F01 — Zaokrąglanie przed zapisem artefaktu metryk

**LOW · CONFIRMED · implementation**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:evaluate 2273–2275,2311,2528–2533`. Reproducer: `run02:C04_reporting_precision`.

**Expected:** Pełna precyzja w danych raportu, rounding tylko prezentacyjny.

**Actual:** RMSE sqrt(250)=15.811388300841896 zapisane jako15.8114; błąd1.1699e−5 przekracza prerejestrowany float64 oracle.

**Scientific impact / uzasadnienie severity:** Utrata precyzji i odtwarzalności kolejnych agregacji. Nie zmienia zasadniczego werdyktu tego audytu.

**Minimal proposed remediation — NIE WYKONANO:** Zwracać surowe scalars, formatować jedynie logi/tabele.

**Future regression / DoD:** Pełnoprecyzyjny zapis ręcznego oracle; osobna prezentacja pozostaje zaokrąglona.

### C05-F01 — Brak per-patient, micro/macro, bias i jawnych pustych grup

**MEDIUM · NOT IMPLEMENTED · implementation / scientific**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:evaluate 2244–2275,2294–2312,2632`. Reproducer: `run02:C05_patient_micro_macro_bias`, `run02:C05_unequal_patient_oracle`, `run02:C04_empty_group`, `run02:C15_overlap_denominators`.

**Expected:** Pełne tabele z N, macro=średnia metryk osób, micro z occurrences; empty→null,N0,reason.

**Actual:** Evaluator zwraca płaski słownik wybranych globalnych mean, bez bias i patient tables; N tylko dla60m. Przy pustej masce pozostają eval_scope/inverse_transform_ok. Oracle: micro RMSE15.811388 vs macro13.660254.

**Scientific impact / uzasadnienie severity:** Nie można ocenić heterogeniczności lub równego udziału osób; nominalne N nie identyfikuje niezależnych pomiarów.

**Minimal proposed remediation — NIE WYKONANO:** Osobny reporting module stosujący zatwierdzone wzory i pełne denominator metadata, bez zmiany lossu.

**Future regression / DoD:** Nierówne liczności osób, missing patient, empty stratum i overlap; wszystkie stosowne metryki per-patient/micro/macro.

### C06-F01 — Raport obejmuje tylko pięć z dwunastu horyzontów

**MEDIUM · NOT IMPLEMENTED · implementation**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:evaluate 2229–2235`. Reproducer: `run02:C06_all12_horizons`.

**Expected:** h=5,10,…,60 dla wszystkich stosownych metrics i persistence.

**Actual:** Obecne TFT h=5/15/30/45/60; pierwszy brak10. Persistence tylko60.

**Scientific impact / uzasadnienie severity:** Brak pełnego obrazu degradacji błędu w czasie; nie wolno traktować tej tabeli jako pełnego kontraktu12h.

**Minimal proposed remediation — NIE WYKONANO:** Raportować wszystkie12, wyróżnienia pozostawić prezentacyjne.

**Future regression / DoD:** Liniowy znacznik h daje dokładne indeksy0…11 i wspólny W dla każdego h.

### C07-F01 — Brak wymaganych zakresów glikemii

**MEDIUM · NOT IMPLEMENTED · implementation / scientific**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:evaluate 2117–2632 (pełna ścieżka zwracanego metrics)`. Reproducer: `run02:C07_strata_reporting`, `run02:C07_strata_boundaries`.

**Expected:** Dwa poziomy target-defined strata z pełną agregacją i N.

**Actual:** Brak stratyfikacji. Niezależny oracle poprawnie rozdziela53.9/54/69.9/70/180/180.1/250/250.1.

**Scientific impact / uzasadnienie severity:** Globalny błąd może maskować hypo/hyper; wynik oracle nie dowodzi istniejącego raportowania.

**Minimal proposed remediation — NIE WYKONANO:** Dodać zatwierdzone strata po observed y, bez dodatkowego filtrowania.

**Future regression / DoD:** Rozłączne/wyczerpujące strata, granice float i empty groups, identyczny podział modeli.

### C08-F01 — Persistence może przejść na niepełny podzbiór

**BLOCKER · CONFIRMED · implementation / scientific**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:_verify_time_idx_alignment 2063–2085; evaluate 2436–2503,2528–2533`. Reproducer: `run02:C08_missing_one_of12`, `run02:C08_missing_decoder_index`.

**Expected:** Brak dowolnego lookup/key → INVALID pełnego porównania, bez zmniejszania W.

**Actual:** Perturbacja jednego origin poza źródło:91.67% przechodzi próg90%, TFT N12, persistence N11, persistence_is_approximate=1, zwykłe metrics. Brak decoder_time_idx tylko pomija baseline.

**Scientific impact / uzasadnienie severity:** Porównanie nie odpowiada pełnemu W; diagnostyczny subset nie może zastępować obligatoryjnej persistence.

**Minimal proposed remediation — NIE WYKONANO:** Wymagać kompletnego dokładnego key set i source lookup; jawnie odmówić pełnego wyniku.

**Future regression / DoD:** Brak1/12, brak klucza, błędny subject/time oraz niezgodne jednostki odmawiają; żadnego progu90%/50%.

### C08-F02 — Brak lineage ostatniej obserwacji persistence

**MEDIUM · NOT IMPLEMENTED · implementation**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:evaluate 2386–2388,2441–2458`. Reproducer: `run02:C08_ffill_origin`, `run02:C08_persistence_two_subject_ramp`.

**Expected:** Zapis s*,age_bins≤6,ffill flag, źródłowe target_observed i zgodność z końcem encodera.

**Actual:** Lookup używa wartości w ostatnim time_idx, bez wyprowadzenia/zapisu s* i wieku. Ramp i dozwolony ffill age1 dają poprawny wynik na fixture; brak śladu pochodzenia nie jest dowodem błędnej wartości.

**Scientific impact / uzasadnienie severity:** Nie można niezależnie zweryfikować dostępności obserwacji w chwili prognozy i limitu wieku z artefaktu ewaluacji.

**Minimal proposed remediation — NIE WYKONANO:** Jawnie wyprowadzić/zapisać ostatnią obserwację w context i porównać z dozwolonym ffill.

**Future regression / DoD:** Age0/1/6, brak source, source poza context i niespójny końcowy CGM; nie dodawać wymogu świeżej obserwacji w origin.

### C10-F01 — Oś kwantyli odczytywana z globalnej listy

**HIGH · CONFIRMED · implementation / scientific**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:evaluate 2218–2225,2308–2311`. Reproducer: `run02:C10_model_quantile_mismatch`.

**Expected:** Q z modelu/artefaktu, identyczne z kontraktem, uporządkowane i zgodne z osią.

**Actual:** Model deklarujący q=.5 na osi0 i .25 na osi3 przyjęty; evaluator czyta index3, MAE30 mimo dokładnego wyjścia rzeczywistego q=.5.

**Scientific impact / uzasadnienie severity:** Błędna point prediction i błędne etykiety hits przy mismatch. Registry Stage B posiada osobne bramki; evaluate nie wymaga ich wywołania.

**Minimal proposed remediation — NIE WYKONANO:** Wymusić kompatybilność Q przed predykcją/redukcją, indeksować po zweryfikowanym model Q.

**Future regression / DoD:** Mismatched/permuted/missing/duplicate Q i wrong output K odmawiają; native Q poprawne.

### C11-F01 — Crossing i odwrócone przedziały bez diagnostyki

**HIGH · NOT IMPLEMENTED · implementation / scientific**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:evaluate 2306–2337`. Reproducer: `run02:C11_crossing_reporting`, `run02:C11_crossing_ties_oracle`.

**Expected:** Raw any/adjacent crossing + magnitude; odwrócona para → INVALID interval summary z pełnym N.

**Actual:** Controlled q25=y+10,q75=y−10 nie daje crossing fields ani INVALID interval. Oracle any1/3, adjacent1/6, magnitude2.

**Scientific impact / uzasadnienie severity:** Nie da się ocenić spójności raw quantiles; marginal hits nie zastępują tej kontroli.

**Minimal proposed remediation — NIE WYKONANO:** Dodać diagnostykę raw oraz walidację par, bez sortowania/clippingu.

**Future regression / DoD:** Ties nie crossing; any i adjacent mają odmienne mianowniki; odwrócone L/U zachowują N_invalid i pierwotny N.

### C12-F01 — Brak PICP, width i nieważonego pinball

**HIGH · NOT IMPLEMENTED · implementation / scientific**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:evaluate 2306–2337`. Reproducer: `run02:C12_interval_reporting`, `run02:C12_interval_pinball_oracle`, `run02:C10_interval_endpoint_oracle`.

**Expected:** Pinball/hit per q i raw50/80/96 PICP/width we wszystkich przekrojach.

**Actual:** Tylko marginal hit per q w60m, zaokrąglany i na masce zależnej od mediany. Żadnego interval summary ani reporting pinball.

**Scientific impact / uzasadnienie severity:** Nie można deklarować empirycznego pokrycia przedziałów ani porównywać ich szerokości; training loss nie jest reporting score.

**Minimal proposed remediation — NIE WYKONANO:** Zaimplementować osobno zatwierdzone reporting metrics, bez factor2/clinical weights.

**Future regression / DoD:** Inclusive endpoints, inverted intervals, full axes/strata/N; odmowa90% bez endpointów. Oracle PASS nie zamyka brakującej funkcji.

### C13-F01 — Clinical weighting zmienia estimand kwantyli

**MEDIUM · CONFIRMED · scientific**. Klasyfikacja: documented limitation / methodological decision.

**Evidence:** `train_tft_population_v2.py:ClinicalQuantileLoss 882–944`. Reproducer: `run02:C13_weighted_objective`, `run02:C13_discrete_quantile_plateau`, `run03:C13_all_q_weighted_CDF`.

**Expected:** Interpretacja wyjść uwzględnia target-dependent ważenie; brak automatycznej tezy o nieważonej medianie/nominalnym coverage.

**Actual:** Loss realizuje2×weighted mean pinball. Uniform[0,140]: q.5 optimum49, nie70; factor2 nie zmienia optimum. Weighted Q=[1.96,9.8,24.5,49,78.75,115.5,135.1].

**Scientific impact / uzasadnienie severity:** Nawet poprawnie zoptymalizowany loss nie identyfikuje ogólnie kwantyli nieważonej populacji. To cecha celu, nie błąd implementacji lossu ani zmierzona wada trained model.

**Minimal proposed remediation — NIE WYKONANO:** Udokumentować estimand w wynikach; późniejsza niezależna ocena nieważonych hits/intervals. Zmiana objective tylko jako osobne zatwierdzone badanie.

**Future regression / DoD:** Zachować analytic/finite oracle i rozdzielenie objective/reporting; przyszły train/validation-only projekt musi mieć odrębne role.

### C14-F01 — Kalibracja fit/apply i jej lineage nie są zaimplementowane w baseline

**MEDIUM · NOT IMPLEMENTED · implementation / scientific**. Klasyfikacja: documented limitation / methodological decision.

**Evidence:** `evaluate 2117–2632; main 2636–2657; build_model 1838–1873; baseline_training.py; checkpoint_registry.py`. Reproducer: `run02:C14_CQR_order_statistic`, `static calibration call-path inventory`.

**Expected:** Dla deklaracji calibrated intervals potrzebne rozdzielone fit/apply i role; brak ma być jawny.

**Actual:** Ścieżka kończy się raw quantiles i marginal hits; nie wywołuje kalibratora. W checkpoint contract brak fitted calibration artifact. CQR jedynie audit oracle.

**Scientific impact / uzasadnienie severity:** Brak dowodu kalibracji; nie jest to blocker samego rozpoczęcia C-core, ani dowód złego empirycznego coverage.

**Minimal proposed remediation — NIE WYKONANO:** Oznaczyć raw/uncalibrated; osobno zatwierdzić calibration method/splits/crossing policy przed przyszłym fit.

**Future regression / DoD:** Fit wyłącznie dozwoloną rolą, perturbacja assessment/test nie zmienia parametrów; zależność czasowa i wymagane założenia jawne.

### C17-F01 — Arbitralne metryki nadają etykietę klinicznego bezpieczeństwa

**HIGH · CONFIRMED · clinical interpretation**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:evaluate 2296–2302,2315–2337`. Reproducer: `run02:C17_clinical_language`.

**Expected:** Brak clinical PASS/SAFE z progu MARD lub Clarke A+B, szczególnie dla synthetic/untrained forecasting.

**Actual:** Dokładne invented predictions wywołują „clinical accuracy target MET” i „clinically SAFE (≥99%)”. Δhit<.05 również dostaje checkmark bez protokołu kalibracji.

**Scientific impact / uzasadnienie severity:** Komunikat wykracza poza dowody: prognoza h>0 nie jest walidacją pomiaru sensora ani bezpieczeństwa klinicznego.

**Minimal proposed remediation — NIE WYKONANO:** Usunąć automatyczne twierdzenia bezpieczeństwa; opisywać miary diagnostyczne i zakres ich dowodu.

**Future regression / DoD:** Przechwycony log dla perfect/poor/empty/crossed fixture nie nadaje safety/confidence na arbitralnych progach.

### C18-F01 — Evaluate nie wiąże provenance modelu, keys i metryk

**HIGH · NOT IMPLEMENTED · implementation**. Klasyfikacja: technical bug.

**Evidence:** `train_tft_population_v2.py:evaluate 2117–2632; checkpoint_registry.py:verified 192–243`. Reproducer: `run02:C18_evaluation_provenance`, `run03:C18_owned_untrained_roundtrip`, `run03:C18_registry_compatibility_controls`.

**Expected:** Evaluation manifest: code/config/schema/Q/normalizer/model role+hash/data role/keys/predictions/definitions/N. Mismatch odmawia.

**Actual:** Evaluate przyjmuje model, dataset,args,frame i oddaje płaski metrics dict, bez manifestu. Audit własnoręcznie zapisuje hashes. Existing registry controls odmówiły4 mismatch, ale nie są wywołane przez evaluate.

**Scientific impact / uzasadnienie severity:** Raportu nie da się powiązać z konkretnym kompletnym eksperymentem wyłącznie przez jego wynik. Registry correctness nie dowodzi downstream evaluation provenance.

**Minimal proposed remediation — NIE WYKONANO:** Wymagać zweryfikowanego evaluation context i zapisywać połączony manifest przy wyniku.

**Future regression / DoD:** Wrong checkpoint/schema/Q/role/normalizer/prediction hash odmawia; owned roundtrip i pełne odtwarzanie metryk z raw predictions.

## Analiza naukowa i odroczone dowody

Dla `L(a)=E[w(Y)ρq(Y−a)|X]`, przy ciągłym rozkładzie `L′(a)=E[w(Y)1{Y≤a}|X]−qE[w(Y)|X]`. Przy atomach warunek subgradientu to `E[w(Y)1{Y<a}] ≤ qE[w(Y)] ≤ E[w(Y)1{Y≤a}]`. Optimum jest więc kwantylem Fw, a dodatni factor2 nie zmienia zbioru minimizerów. Dla uniform[0,140] masa ważona wynosi245, poniżej70 gęstość ważona2.5: mediana `122.5/2.5=49`. Waga1 daje70. Nie zmieniono produkcyjnego lossu ani nie wykonano optimizer-based objective experiments.

CQR oracle używa k=ceil((n+1)(1−α)) i k-tej statystyki, k>n→∞, bez interpolacji i clippingu k. Wyniki obejmują ties i ujemny score. To kandydat metodologiczny, nie wybrany kalibrator. Marginal coverage, conditional/patient-specific coverage, marginal quantile calibration i simultaneous12-step coverage są różnymi celami. Gwarancji z niezależnego calibration sample nie przenosi się automatycznie na nakładające się okna czasowe. Źródła: [Romano et al., CQR](https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf), [Chernozhukov et al., dependent data](https://proceedings.mlr.press/v75/chernozhukov18a.html). Nie wykonano CQR fit na real data.

Synthetic missingness związana z osobą/poziomem usuwa wszystkie6 okien osobyB przy braku tylko jednej obserwacji w jej trajektorii. Od6 missing occurrences przechodzimy do72 odrzuconych occurrences, w tym66 obserwowanych. Symetryczna kontrola usuwa po6 okien obu osób. To demonstracja możliwości selekcji przez pełny decoder; nie estymacja MAR/MNAR, kierunku ani skali real bias. Nie rekonstruowano brakujących real outcomes.

Clarke źródłowo dotyczy porównania systemu SMBG z referencją, nie stanowi samodzielnego dowodu bezpieczeństwa forecast h>0: [Clarke et al.,1987](https://pubmed.ncbi.nlm.nih.gov/3677983/). Przetestowano pięć jednoznacznych interior points A…E. Pełny PDF/figura wydawcy zwracał403, dlatego **geometria wszystkich granic NIE ZOSTAŁA niezależnie certyfikowana**. Nie dorobiono arbitralnych granic ani klinicznego PASS z tych pięciu checks. Zakresy glikemii przyjęto z już zatwierdzonego execution contract; nie dobierano ich do wyników.

Przed C-dev pozostają decyzje: chronologiczne role V_select/V_cal_fit/V_assess, target-boundary purging i checkpoint lineage, calibration/crossing policy, objective variants, kryteria porównania i block/subject resampling. Jeśli checkpoint użył całego val do early stopping, późniejsze wydzielenie jego części nie czyni jej niezależną. Overlap nie uzasadnia IID bootstrap wierszy. Nie ustalano tych decyzji na test-performance ani nie otwierano test setu.

## Dokładne polecenia i artefakty

Pierwotne uruchomienia z katalogu zadania w Codex (cwd widoczne w invocation/summary):

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '/home/radian/NeuroMetabolic Dashboard/.venv/bin/python' outputs/stage_c_synthetic/tests/run_stage_c.py --repo '/home/radian/NeuroMetabolic Dashboard' --out outputs/stage_c_synthetic/run01
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '/home/radian/NeuroMetabolic Dashboard/.venv/bin/python' outputs/stage_c_synthetic/tests/run_stage_c.py --repo '/home/radian/NeuroMetabolic Dashboard' --out outputs/stage_c_synthetic/run02
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '/home/radian/NeuroMetabolic Dashboard/.venv/bin/python' outputs/stage_c_synthetic/tests/followup_stage_c.py --repo '/home/radian/NeuroMetabolic Dashboard' --out outputs/stage_c_synthetic/run03
```

Run02/03 wykonano poza sandboxem z autoryzacją narzędzia i opisanymi Python guards. Console stdout/stderr zachowane w odpowiednich `runNN/console.log`; runner manifest zawiera argv i environment. Nowa reprodukcja wymaga nowego OUT (`exist_ok=False` zapobiega nadpisaniu). Nie uruchamiać szerokiego unittest discovery Stage B: zawiera trening poza zakresem C-core. `--only` nie jest użyte ani rekomendowane w zamrożonych runnerach; odtwarzanie dotyczy pełnego ograniczonego suite.

Pliki repo docelowo: `docs/BASELINE_AUDIT_STAGE_C.md`, `configs/baseline_stage_c_audit.json`; audytowe źródła i kopia prerejestracji w `experiments/stage_c_synthetic_20260914/`; lokalne syntetyczne logi/state/predictions w ignorowanym `ml/models/stage_c_synthetic_20260914/`. Dostarczony pakiet Codex `outputs/stage_c_synthetic/` łączy te części do review; nie zawiera pacjentów. State `.pt` to **SYNTHETIC_UNTRAINED_AUDIT_STATE**, nie BEST/wytrenowany checkpoint ani certyfikowany model inference.

| Artefakt | Znaczenie |
|---|---|
| configs/baseline_stage_c_audit.json | zamrożone definicje i source hashes |
| configs/followup_preregistration.json | powód dodatkowych probes, nie nowy próg |
| tests/run_stage_c.py; tests/followup_stage_c.py | reproduktory, instrumentation tylko audytowa |
| execution_summary.json | rzeczywiste OS exit codes, korekta run01, sumaryczne liczniki |
| findings.json |18 pełnych rekordów Cxx-Fyy |
| runNN/results.json; probe_captures.json | asercje/tracebacks i rzeczywiste metrics/logs |
| runNN/manifest.json | per-run config/source/fixture/window/prediction/state hashes, numerical profile |
| run03/normalizer_probe.json | fitted vs emitted group scales |
| run03/real_corrupt_metrics.json | rzeczywisty PF po late fault, N10 |
| SHA256SUMS.json | końcowy indeks dostarczonych plików; po dodaniu raportu i logów |

Fixture assessment SHA: `7ae240e51cc779a9a8e986a2c4e8dfecc9059403be3b7897c682669ac0a673da`.
Window keys SHA: `7f5ed1a2e76922e6f3fea5958522d15b6b0c1536c6437687595d701f51c780ee`.
Real predictions SHA(run02): `db437b4578812199e18903fb86bbf5a4d2fd35ead393a67817e71b5c3d316af8`.
Frozen R1 SHA: `b87be8716600df18c46176e4f4184ab004a5fcc47e89fcab96431da2a51bc01b`; R2 SHA: `5bbb489950c5c0a3566a5d9a2a883e4c8b2b9f62fa4dc3a232b1131b919fe457`.

## Końcowy zakres i DoD

C00–C19 mają dowód albo jawne ograniczenie; findings mają severity, status, reproducer, expected/actual, scientific impact, minimalną propozycję i regression DoD. Własne oracles nie zastępują brakujących produkcyjnych capabilities. Nie oceniono real calibration/performance, pełnych Clarke boundaries ani rzeczywistego ARIMA convergence. Nie wykazano trained checkpoint mismatch end-to-end w evaluate; istniejący registry gate sprawdzono oddzielnie, a jego powiązania z evaluate brak.

Przed zapisaniem artefaktów repo pozostało czyste, HEAD niezmieniony, `git diff --check` PASS i source hashes zgodne we wszystkich manifestach. Końcowa kontrola po instalacji dokumentuje dokładną listę audit-only plików w `final_verification.json`. Żadnych poprawek produkcyjnych, zmian baseline config, staged files, commitów lub push. Syntetyczne checkpoints/logi trafiają do ignorowanego katalogu; nie przeznaczono ich do commitu.

**Zatrzymano na raporcie. Następny krok: osobne review findings i decyzja użytkownika. Remediation, C-dev, pełny trening, Optuna i test evaluation nie zostały rozpoczęte.**
