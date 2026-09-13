# Baseline v1.0 — Stage B audit mechaniki uczenia

Data: 2026-09-13. **Werdykt: FAIL. Audyt zakończony; baseline nie jest zatwierdzony do pełnego treningu.**

## Change / Reason

Dodano wyłącznie konfigurację audytu, syntetyczne testy, izolowany runner i ten raport. Nie zmieniono produkcyjnego lossu, TFT, schedulera, callbacków, preprocessingu, protokołu Stage A, backendu, frontendu ani XAI. Celem jest weryfikacja mechaniki, nie dobór modelu lub poprawa metryk. Nie wykonano commitu ani remediation.

Przeczytano `AGENTS.md`, `RESEARCH_WORKFLOW.md`, `ROADMAP.md`, oba raporty Stage A, `STAGE_B_AUDIT_PLAN.md`, `CODEX_STAGE_B_TASK.md`, załączone zlecenie i `configs/baseline_stage_a.json`. Audytowano aktywny `ml/scripts/train_tft_population_v2.py` i lokalne implementacje PF/Lightning/Torch. Historyczny `train_tft.py`, serving, XAI i tuning nie otrzymują certyfikacji.

## Kod, dane i środowisko

- Gałąź: `research/baseline-audit`; HEAD: `b83504ff61542fb2e4b2ae805bb79eea3ffa7660`; drzewo **czyste na wejściu**, dirty na wyjściu wyłącznie od nowych plików audytu.
- Kanoniczny dataset: `ml/data/processed/baseline_v1_stage_a_20260913T115538143463Z/`; jego manifest: commit `5ee8d930e80fd7d62d0248c52ed10aaaaaa7f8d0`, `git_dirty=false`.
- SHA-256 parquetu **przeliczono i potwierdzono**: `d14fe31b1b81971639ca8d5ba17b4882fbd6711569e30785f0da7b91a0c031cf`. Sprawdzono też commit/dirty manifestu. Dane nie były regenerowane ani podawane modelowi.
- Historyczny remediation wskazuje wcześniejszy run `112536811560Z` / `b869d85…` / dirty=true. Nie użyto go. Nowszy HEAD audytu nie zmienia produkcyjnego kodu ML względem kanonicznego commitu; różnicę sprawdzono lokalnie przez Git.
- Python 3.14.7; PyTorch 2.11.0+cu130 (metadata pakietu 2.11.0); Lightning 2.6.1; PyTorch Forecasting 1.7.0; NumPy 2.4.4; pandas 2.3.3.
- Referencja: CPU, FP32, jeden wątek, `num_workers=0`, seed 42; kontrola seed 43. CUDA niedostępna w tym procesie (`torch.cuda.is_available() == False`, ostrzeżenie NVML). **CUDA: BLOCKED**, AMP i workery >0: **NOT RUN**. Nie sprawdzano innych platform ani wersji.

## Izolacja i budżet

Runner nie wywołuje produkcyjnego `main()`. Strażniki blokują `main`, `load_test_data`, `evaluate` oraz wejścia Optuny; test kontroli dodatniej wywołuje strażnik i sprawdza odmowę. Optuna nie jest uruchamiana. Żadnych rzeczywistych predykcji ani MARD/MAE/RMSE test setu.

Fixture zawiera wyłącznie wymyślone, deterministyczne XML dwóch syntetycznych osób, po 480 kroków. Korzysta z produkcyjnego preprocessingu, splitu i `build_datasets`. Nie tworzy rzeczywistego ani syntetycznego splitu testowego w Stage B. Do fit wybierane są cztery eligible windows (osiem dla accumulation=2); to ograniczenie **syntetycznego smoke**, a nie zmiana protokołu Stage A. Mały TFT: hidden_size=8, hidden_continuous_size=4, attention_heads=1, LSTM=1, context=48, horizon=12, batch=2, LR=3e-4, clip=1.

Każdy fit: maksymalnie 8 wywołań AdamW, maksymalnie 4 epoki, dwa sanity batches; accumulation=2 ma cztery minibatches na epokę. Zliczane są również wywołania z LR=0. Oddzielny gradient test wykonuje jedną aktualizację TFT z dodatnim LR. Scheduler badano na jednym parametrze przez 160 aktualizacji. Nie wymagano spadku lossu jako dowodu poprawności.

Runner zachowuje produkcyjne tworzenie i kolejność callbacków oraz domyślne SWA w dedykowanych przebiegach. Ogranicza Trainer do powyższego budżetu, używa lokalnego CSVLogger i wyłącza rysowanie predykcji. Dodatkowe callbacki wyłącznie obserwują, zatrzymują przebieg K=4 lub zabezpieczają negatywne reproduktory. W wariancie kontrolnym wyłączono dropout i shuffle; nie przedstawia się go jako konfiguracji domyślnej.

**Jawny wariant diagnostyczny odczytu:** produkcyjne `fit(ckpt_path=…)` ma osobny czerwony test. Do dalszych prób resume runner może przekazać `weights_only=False` wyłącznie dla checkpointu z tego samego syntetycznego runu, po sprawdzeniu zapisanego SHA-256. Nie zmienia to produkcji i nie oznacza naprawy domyślnego resume. Nie zmieniano budżetu schedulera, pozycji samplera ani prywatnego stanu pętli, by wymusić zgodność.

Przy NaN/Inf nie zapisano terminalnego artefaktu jako poprawnego. Input/target probe kończy się błędem backward; gradient probe dochodzi do granicy optimizera i zostaje zatrzymany przez **strażnik audytu**, czego nie zalicza się jako zabezpieczenia produkcyjnego. Zero aktualizacji w tych negatywnych fitach.

## Mapa przepływu i jednostek

1. `build_datasets` → `TimeSeriesDataSet`: GroupNormalizer(log), fit na observed train; val odziedzicza parametry. Target w `y` to mg/dL, a `encoder_cont`/`decoder_cont` zawierają reprezentacje przeskalowane. Stage A dopuszcza krótszy decoder treningowy; val ma pełny horyzont.
2. `ClinicalTFT` dziedziczy PF TFT. Encoder przyjmuje historię, decoder wybiera wyłącznie zadeklarowane known features. Perturbacja przyszłych unknowns i `decoder_target` nie zmienia predykcji; zmiana encodera stanowi dodatnią kontrolę.
3. `output_layer` daje siedem surowych wyjść. PF `TemporalFusionTransformer.forward` → `BaseModel.transform_output` → normalizer odwraca log/skalę **raz**: `exp(raw * scale + center)`. Hook warstwy i zmiana center o log(2) potwierdzają dwukrotne predykcje w mg/dL.
4. `BaseModel.step` pakuje nierówne długości decoderów, wywołuje ClinicalQuantileLoss na predykcjach i targetach mg/dL. PF QuantileLoss używa `2 * max(q*e, (q-1)*e)`. Kod projektu uśrednia Q i mnoży pozycję przez 2.5 dla `target <70`, inaczej 1. MultiHorizonMetric sumuje ważne pozycje i dzieli przez **liczbę pozycji**, nie sumę wag klinicznych. Padding nie zwiększa mianownika.
5. `BaseModel.step` loguje scalar batch loss z `batch_size=len(decoder_target)`. To powoduje B03-F01 przy różnych długościach decoderów; pełnohoryzontowy val nie ma tej rozbieżności.
6. Backward → `GradientNormLogger.on_after_backward` → Lightning `on_before_optimizer_step` → clipping norm → AdamW → scheduler interval=step. Pomiar rzeczywistego wywołania AdamW potwierdza normę po clippingu ≤1.00001, także przy accumulation=2. LR=0 pierwszego kroku nie oznacza zerowego gradientu.
7. Scheduler: `estimated_stepping_batches`, `steps_per_epoch=max(1,total//epochs)`, warmup `max(50,total//20)`, następnie cosine warm restarts, T0=15*steps_per_epoch, Tmult=2, eta_min=1e-6. `monitor='val_loss'` nie przekształca go w ReduceLROnPlateau. W smoke 8 kroków cały przebieg jest w warmupie.
8. `train`: SWA (domyślnie), EarlyStopping(val_loss, min_delta=1e-4, patience=25 z SWA /20 bez), ModelCheckpoint(top3, last), LRMonitor, GradientNormLogger. Lightning umieszcza checkpoint callbacks na końcu. W czteroepokowym smoke SWA wchodzi w epoce o indeksie 2, zastępując SequentialLR(step) przez SWALR(epoch); uśrednia dwa stany. Ostrzeżenie biblioteki o zmianie interwału jest zachowane w logu.
9. `build_model` wczytuje wagi/hparams z checkpointu; `train` ponownie przekazuje go do pełnego `fit`. `main` wybiera wcześniej `find_best_checkpoint()`, a po fit ponownie ładuje best do oceny. W audycie prześledzono to statycznie — `main` nie został wykonany.

Lokalne źródła: `train_tft_population_v2.py:876,940,1020,1682,1756,1815,1856,2667`; PF `metrics/quantile.py:34`, `metrics/base_metrics/_base_metrics.py:862`, `models/base/_base_model.py:628,835,964`; Lightning `callbacks/stochastic_weight_avg.py:136,187,275,341`, `callbacks/model_checkpoint.py:488,535,556`, `trainer/trainer.py:522,1748`. Są to wersje z lokalnej `.venv`, nie założenia z dokumentacji innego wydania.

## Validation — wyniki wykonania

Końcowy run: `20260913T123754Z` w `ml/models/stage_b_audit/20260913T123754Z/`.

| Zestaw / check | PASS | FAIL | ERROR | SKIP | XFAIL | Exit code |
|---|---:|---:|---:|---:|---:|---:|
| Stage B, pełne uruchomienie 32 metod przed korektą jednego kryterium | 17 | 15 | 0 | 0 | 0 | 1 |
| Końcowy bilans 32 metod po focused review (poniżej) | 18 | 14 | 0 | 0 | 0 | FAIL |
| Stage A, świeża regresja | 31 | 0 | 0 | 0 | 0 | 0 |
| Składnia i bezpieczne importy | PASS | — | — | — | — | 0 |
| Git diff/check/protokół/review | PASS | — | — | — | — | 0 |

Pełne uruchomienie raportowało **16 failure events**: test nonfinite loss ma dwie nieudane podpróby (NaN i Inf). W tym uruchomieniu było to **15 metod FAIL**, nie 16 osobnych metod. Po opisanej niżej korekcie kontraktu końcowy bilans wynosi 18 PASS /14 FAIL (15 failure events). `method_counts` i surowe `results` w manifeście rozróżniają obie liczby.

Polecenia wykonano z root repo:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ml.scripts.diagnostics.audit_stage_b
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nmd-stage-a-mpl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s ml/tests -p test_baseline_stage_a.py -v
```

Runner zapisuje rzeczywistą komendę subprocessu `--suite`, komendy poszczególnych workerów, zmienne środowiska i exit codes. Test Stage A: **7.170 s**; końcowy Stage B: **133.730 s unittest / 135.957 s wraz z inicjalizacją**. Watchdog: 600 s.

Końcowy Stage B: **229 synthetic_optimizer_steps = 68 worker TFT + 1 jednostkowy TFT + 160 scalar scheduler**. Każdy worker ≤8; resume wykonujący tylko 2 albo 0 kroków jest jawnie opisanym FAIL, a nie domkniętym budżetem 8. Przebiegi NaN/Inf: zero updates i zero checkpointów. Wszystkie mają `full_training=false`, `optuna=false`, `test_performance=false`.

Zachowano pięć kolejnych unikalnych uruchomień podczas tworzenia/debugowania harnessu: `122820Z`, `122916Z`, `123148Z`, `123446Z`, `123754Z` (wszystkie z prefiksem daty 20260913). Ich sumaryczny czas zestawów to około **494.5 s**, poniżej 10 minut; liczba aktualizacji łącznie **1058 = 800 scalar scheduler + 258 TFT**. Limit 160 scalar steps dotyczył pojedynczego wykonania testu; powtórzenia były potrzebne do poprawienia harnessu i pokrycia nierozstrzygniętych kontraktów, a nie do strojenia wyników. Starsze niekompletne wyniki harnessu nie są wynikiem końcowym. Szczegóły wszystkich prób są w `postrun_review.json`.

SHA-256 głównych dowodów:

- `manifest.json`: `e7735eca7b4a46cbe4e235d3317b28c8734573c3253b2685ad7f4a84fead8db3`.
- `invocation.json`: `f3c4e69c8cb40a77ad9bc24f3c3d12a49c98042f7fc77e7211002c5534432e79`.
- `stage_a_regression.log`: `eefe625ae507bea2910eed26fe04ee500fa021e1468530512606ef937877a665`.

Manifests workerów zawierają hashe checkpointów i parentów, `fixture_sha256`, rzeczywistą schema/q/normalizer/context/horizon, efektywne argumenty oraz provenance kodu. Główny manifest wiąże child manifests. `postrun_review.json` uzupełnia indeks SHA wszystkich lokalnych artefaktów, źródeł audytu i wyniki offline porównania stanów; nie uruchamia treningu. Po pełnym przebiegu skorygowano jeden nieuprawniony warunek testu SWA; focused review ma własny hash źródeł. Finalny diff hash obejmuje tę zmianę i raport.


**Focused review kontraktu SWA:** początkowy test błędnie wymagał, żeby `last` do kontynuacji miał te same wagi co averaged terminal do inferencji. Takiego wymagania nie ma w zleceniu; byłoby sprzeczne z rozróżnieniem ról checkpointów. Usunięto tę nieuprawnioną asercję, pozostawiając kontrolę średniej SWA oraz pełnego stanu optimizera/global_step w last. Nie naprawiono ani nie zamaskowano błędu produkcji. Test `test_B13_swa_weights_and_checkpoint_semantics` wykonano ponownie na **tych samych, niezmienionych checkpointach**: 1 PASS, exit 0, zero dodatkowych aktualizacji. Dowód: `swa_contract_review/review.json`, polecenie:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nmd-stage-b-review-mpl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python ml/models/stage_b_audit/20260913T123754Z/swa_contract_review.py
```

Końcowe 18 PASS /14 FAIL to bilans pełnego zestawu z zastąpieniem tego jednego wyniku focused review; **nie jest deklaracją ponownego wykonania wszystkich 32 metod**. Pozostałe nieudane testy SWA dotyczą realnego resume i nadal FAIL. Różnica wag last/best/terminal jest udokumentowaną właściwością, nie samodzielnym findingiem.

Każdy podany FAIL jest rzeczywistą asercją, bez `expectedFailure`, xfail ani pomijania znalezionego problemu. Przygotowawcze przebiegi runnera pozostawiono lokalnie, wraz z błędami samego harnessu (m.in. ustawienie read-only `log_interval`, zbyt szeroki strażnik importów, niezainicjalizowany estimator). Poprawiano wyłącznie harness. Nie zaliczono tych błędów jako findings produkcji. W końcowych testach nie poluzowano tolerancji replay (atol=1e-6, rtol=1e-5).

### B00–B17

| ID | Status | Dowód i granica wniosku |
|---|---|---|
| B00 | PASS | `test_B00_forbidden_entry_guards_positive_control`; guarded runner, brak wejścia w rzeczywiste test-performance. |
| B01 | PASS | Rzeczywiste PF batch shapes/dtype, 48/12, kolejność time_idx; packed lengths 12/3, ekstremalny padding bez udziału w lossie/mianowniku; regresje Stage A. |
| B02 | PASS | Niezależny scalar oracle, q=.1/.5/.9, predykcje poniżej/równe/powyżej y, 69.9/70/70.1; zero error=0. |
| B03 | FAIL | Transform_output i jednostki/reset PASS; agregacja epoki dla różnych długości FAIL B03-F01. |
| B04 | PASS | Stała lista poprawna, siedem osi, median po wartości .5, round-trip q; crossing wykrywany w fixture, bez sortowania. Nie jest to dowód monotoniczności ani kalibracji. |
| B05 | PASS | Prawdziwy TFT eval/no_grad: przyszłe unknowns nie wpływają; zmiana encodera wpływa; train forward finite. |
| B06 | PASS | Float64 gradcheck i analityczny gradient, rzeczywisty finite backward, niezerowe gradienty aktywnych parametrów, zmiana wag po dodatnim LR, zero_grad. |
| B07 | PASS | AdamW coverage/eps/decay; frozen ma grad=None; normy przed/po clip; 16 backward →8 updates przy accumulation=2; scheduler=8. |
| B08 | FAIL | Trace 0…160, warmup i restart PASS, stan schedulera odczytany po zapisie; max_steps/niepełny batch/accumulation PASS; zero-epoch guard FAIL B08-F01. Pełny replay schedulera w fit niepotwierdzony z powodu B10. |
| B09 | FAIL | Wagi, hparams, q, dataset_parameters, normalizer, optimizer/scheduler i callback payload zapisane; identyczny eval round-trip i odmowa corrupt load PASS. Brak odmowy semantycznie niezgodnego datasetu (B11-F01), pełny fit-load FAIL (B10-F01). |
| B10 | FAIL | Nowe procesy, stały total=8, K=4, default i diagnostic loading; wznowienie nie wykonuje wymaganych czterech dalszych aktualizacji. Callback history nieodtworzona. Brak gwarantowanej granicy exact resume. |
| B11 | FAIL | Dataset context/horizon/normalizer ignorowane przy load; corrupt filename wygrywa wybór; best używany jako wejście continuation. |
| B12 | PASS | Improvement/plateau/worsening/min_delta boundary, sanity ignorowane, NaN/Inf stop, missing val_loss error, best/wait state round-trip; train/test nie zastępują monitora. |
| B13 | FAIL | Rzeczywiste SWA, przejęcie schedulera i średnia stanów zbadane; problemy resume opisane poniżej; role last i averaged terminal rozdzielono, ich nierówność nie jest błędem. Patience=25 może zakończyć run zanim zacznie się SWA (symulacja callbacków). |
| B14 | PASS | Dwa **nowe procesy** seed42 z dropout/shuffle: identyczna inicjalizacja, kolejność, loss, LR i final weights; seed43 różni się. Wyłącznie CPU/1 thread/0 workers/ten stos. CUDA BLOCKED, AMP/workery>0 NOT RUN. |
| B15 | FAIL | Zero/constant target/duża skończona strata PASS; finite FP32 forward/backward PASS. Niepoprawny loss zamieniany na stałą; brak produkcyjnej odmowy NaN gradientu. Negatywne fity zatrzymane, bez poprawnych checkpointów. |
| B16 | PASS | Unikalne run/child manifests, SHA canonical/fixture/config/diff/checkpoints, parent, seed, wersje, efektywne warianty i liczniki actual updates; lokalny indeks wiąże artefakty. Nie należy mylić provenance audytu z brakującym kontraktem produkcji B11. |
| B17 | FAIL | Stage A 31 PASS; składnia/importy/review PASS. Nowe testy mają rzeczywiste FAIL, zatem bramka regresyjna nie jest zielona. Produkcja i protokół niezmienione; nic staged. |

## Potwierdzone findings i osobne taski remediation

Wszystkie poniższe: **status OPEN / CONFIRMED**, audytowany commit `b83504ff61542fb2e4b2ae805bb79eea3ffa7660` (produkcyjny kod identyczny z kanonicznym Stage A). Reproducer to wskazany test w `ml/tests/test_baseline_stage_b.py`, uruchamiany pełnym poleceniem runnera powyżej. Każda propozycja jest taskiem **do osobnej specyfikacji/review**, nie zaimplementowaną zmianą.

### B03-F01 — MEDIUM — inny mianownik lossu epoki

- Plik/symbol: PF `models/base/_base_model.py:BaseModel.step`, używany przez ClinicalTFT.
- Reproducer: `test_B03_epoch_reduction_unequal_lengths_contract`, `aggregation.json`.
- Oczekiwane: średnia po 27 ważnych pozycjach = **12.222222…**. Rzeczywiste: logowane wagi batchy [2,1], batch losses [10,30], agregacja = **16.666666…**.
- Przyczyna: logowanie średniej batchowej ważonej liczbą sekwencji zamiast liczbą ważnych targetów.
- Scientific impact: zniekształcony train_loss przy krótszych decoderach; **nie wykazano tego błędu dla obecnego pełnohoryzontowego val_loss**, więc nie twierdzimy, że ten reproducer zmienia wybór best. Loss pozycji i batchowy gradient mają poprawny wzór.
- Minimalna remediation / DoD: ustalić i jawnie wdrożyć ważenie agregacji ważnymi pozycjami; regresja na różnych batch sizes/lengths oraz pełnohoryzontowa kontrola bez zmiany wyniku. Nie zmieniać clinical weighting.

### B08-F01 — LOW — brak jawnego kontraktu zerowego budżetu epok

- Plik/symbol: `train_tft_population_v2.py:ClinicalTFT.configure_optimizers`.
- Reproducer: `test_B08_invalid_epoch_budget_rejected_explicitly`, `invalid_epochs.json`.
- Oczekiwane: czytelny błąd konfiguracji przed konstrukcją schedulera dla epochs=0. Rzeczywiste: `ZeroDivisionError` w `total_steps // max_epochs`.
- Przyczyna: brak walidacji wejścia tej funkcji. `epochs=-1` jest dozwolone w Lightning z finite max_steps; produkcyjna formuła przechodzi wtedy do steps_per_epoch=1. **Nie klasyfikujemy samego -1 jako błędu.**
- Scientific impact: ograniczona odporność konfiguracji; nie dotyczy domyślnego 60-epokowego budżetu.
- Minimalna remediation / DoD: jawnie zdefiniować wspierane budżety i sprawdzać zero przed obliczeniami; test dla 0 oraz osobny, udokumentowany kontrakt -1/max_steps. Bez strojenia długości warmupu na wynikach.

### B10-F01 — HIGH — produkcyjny full resume odrzuca własny checkpoint

- Plik/symbol: `train_tft_population_v2.py:train` → Lightning `Trainer.fit` / checkpoint load.
- Reproducer: `test_B10_production_resume_loads_own_checkpoint`, `resumed_production/manifest.json`.
- Oczekiwane: odczyt pełnego stanu checkpointu bieżącego syntetycznego runu i cztery dalsze aktualizacje. Rzeczywiste: `UnpicklingError`, unsupported global `pandas.core.frame.DataFrame`, zero nowych updates.
- Przyczyna: domyślny tryb weights_only odczytu fit nie obsługuje serializowanego normalizatora; wcześniejsze PF `load_from_checkpoint` wczytuje ten sam model inną ścieżką. Weight round-trip nie dowodzi fit-resume.
- Scientific impact: kontynuacja treningu nie działa w badanym stosie.
- Minimalna remediation / DoD: określić format i politykę zaufania checkpointów, spójny odczyt pełnego stanu z weryfikacją provenance. Nie zaleca się globalnego wyłączenia ograniczeń dla obcych plików. Regresja: nowy proces odtwarza własny checkpoint; obcy/uszkodzony odrzucany.

### B10-F02 — HIGH — RNG/sampler nie mają kontraktu replay

- Plik/symbol: `build_model`/`train`, standardowy zapis Lightning; seed_everything w wejściu.
- Reproducer: `test_B10_epoch_resume_dropout_shuffle_and_rng`, `resume_stochastic.json` i trace pierwszego resumed batch przy global_step=4.
- Oczekiwane: RNG Python/NumPy/Torch i batch przy tym samym punkcie kontynuacji zgodne z ciągłym przebiegiem. Rzeczywiste: checkpoint nie zawiera jawnego odtworzenia tych trzech stanów; nowe procesy seedują od początku. Przy pierwszym resumed batch (global_step=4) równość stanów Python, NumPy i Torch wynosi odpowiednio false/false/false; kolejność batcha również jest inna. Trace/probes nie odtwarzają stanu kontynuacji.
- Przyczyna: wagi/optimizer nie kodują pozycji RNG/samplera. Python/NumPy mają jawne, syntetyczne probe draws w callbacku; nie sterują targetami ani nie są przywracane przez audit.
- Scientific impact: seed42 i poprawny odczyt wag nie dają exact replay. **Końcowej różnicy wag nie przypisujemy wyłącznie RNG**: B10-F04 zmienia także liczbę wykonanych aktualizacji. N=8 vs4+4 nie został spełniony.
- Minimalna remediation / DoD: dopiero po naprawie B10-F01/F04 ustalić wspieraną granicę resume i zapis/odczyt RNG oraz kolejność danych; nowy proces porównuje pierwszą resumed próbkę, wszystkie cztery dalsze aktualizacje, LR, moments i wagi. Bez prywatnego replay środka epoki.

### B10-F03 — LOW — historia ostrzeżeń gradientu nie jest serializowana

- Plik/symbol: `GradientNormLogger`.
- Reproducer: `test_B10_gradient_callback_state_roundtrip`.
- Oczekiwane: counter=12 po state_dict/load_state_dict. Rzeczywiste: counter=0; callback state jest pusty.
- Przyczyna: własny `_high_grad_consecutive` bez implementacji serializacji.
- Scientific impact: po restarcie zmienia się historia diagnostyki; nie jest to dowód zmiany gradientów modelu.
- Minimalna remediation / DoD: zachować licznik i istotną konfigurację callbacku; regresja kontynuacji ostrzeżeń na granicy patience.

### B10-F04 — HIGH — wznowienie traci aktualizacje przy granicy epoki

- Plik/symbol: `train` i odtwarzanie `fit_loop` Lightning 2.6.1.
- Reproducer: `test_B10_last_checkpoint_epoch_progress`, `test_B10_epoch_resume_no_dropout_no_shuffle`; osobno `last.ckpt` oraz publiczne `Trainer.save_checkpoint` po zatrzymanym fit (`terminal.ckpt`).
- Oczekiwane: total budget=8 niezmieniony, K=4, po wznowieniu 4 updates, global_step=8, cztery epoki łącznie. Rzeczywiste: **2 nowe updates, global_step=6**, koniec na max_epochs=4. Trace zawiera epokę po resume bez batchy. Dotyczy także dropout=0/shuffle=False, więc nie jest tylko problemem stochastic replay.
- Przyczyna zawężona: niespójna kontynuacja licznika postępu epoki i zużytego dataloadera w tej ścieżce lokalnego Lightning; stan jest zapisany, ale nie daje wymaganego odtworzenia. Nie naprawiano prywatnych pól pętli ani nie zwiększano max_epochs.
- Scientific impact: ani last, ani testowany completed-epoch checkpoint nie zapewnia exact resume w badanym stosie. Końcowe porównania optimizer/scheduler/wag nie mogą stanowić testu równego budżetu, skoro ten warunek zawiódł wcześniej.
- Minimalna remediation / DoD: mały upstream reproducer pętli, jawny wspierany punkt zatrzymania i wersja stosu; publiczna kontynuacja 4+4 musi zachować global_step, liczbę batchy, LR i moments. Do tego czasu żadnej deklaracji exact resume.

### B11-F01 — HIGH — brak zgodności semantycznej dataset/checkpoint

- Plik/symbol: `train_tft_population_v2.py:build_model`.
- Reproducer: `test_B11_incompatible_dataset_is_rejected` zmienia jednocześnie context48→24, horizon12→6 i fitted normalizer syntetycznego datasetu przekazanego do build_model.
- Oczekiwane: odmowa przed użyciem checkpointu. Rzeczywiste: model ładuje się, dataset nie jest porównywany; ogólne ostrzeżenie o ignorowaniu CLI nie zastępuje kontroli zgodności.
- Przyczyna: training argument jest używany tylko przy tworzeniu nowego modelu; brak obowiązkowego dataset digest/schema/config manifestu na ścieżce load.
- Scientific impact: możliwe mieszanie modeli z innymi oknami/normalizacją; brakuje kontraktu również dla dataset hash i scheduler budget. Te ostatnie wskazano z inspekcji, nie jako osobne niezależne runtime reproduktory.
- Minimalna remediation / DoD: zweryfikowany fingerprint datasetu, schema, q, fitted normalizer, context/horizon i total scheduler budget; jawna odmowa każdego konfliktu. Wersjonowana migracja tylko po osobnej decyzji. Regresje osobno dla każdej składowej.

### B11-F02 — HIGH — wybór checkpointu opiera się na nazwie, miesza best i continuation

- Plik/symbol: `find_best_checkpoint`; statyczne miejsce wywołania `main:2667`.
- Reproducer: `test_B11_checkpoint_selection_does_not_trust_filename`: plik z tekstem zamiast checkpointu, ale nazwą `…epoch=99-val_loss=0.0001.ckpt`, zostaje wybrany. Kod helpera ignoruje `last` i przy nieparsowalnych nazwach używa mtime. Następnie main przekazuje best do pełnego resume.
- Oczekiwane: zweryfikowany last do continuation, zweryfikowany best do inferencji; nazwa nie jest dowodem wartości metryki ani zgodności. Rzeczywiste: wybór po zaokrąglonym lossie nazwy bez kontroli payload/provenance; best może być starszy od last.
- Przyczyna: wspólny niezweryfikowany helper discovery i brak jawnego resume mode.
- Scientific impact: wznowienie może cofnąć historię, odtworzyć inny eksperyment lub po prostu odrzucić corrupt plik dopiero w późniejszym load.
- Minimalna remediation / DoD: trzy odrębne tryby: last continuation, best inference, weights-only nowy run; manifest/payload jako źródło truth, brak fallbacku mtime w certyfikowanej ścieżce. Test: starszy best i nowszy last, konflikt manifestów, corrupt payload.

### B13-F01 — HIGH — resume po aktywacji SWA nie odtwarza dalszego uczenia

- Plik/symbol: `train`, Lightning `StochasticWeightAveraging.on_train_end`, `ModelCheckpoint` i wznowienie po aktywacji SWA.
- Reproducer: `test_B13_default_swa_scheduler_averaging_resume`, `test_B13_resume_after_swa_activation`. Osobny test `test_B13_swa_weights_and_checkpoint_semantics` potwierdza poprawność samej średniej; `swa_semantics.json` dokumentuje role artefaktów.
- Oczekiwane: po odczycie stanu aktywnego SWA wykonane dwie pozostałe aktualizacje, zachowane scheduler/callback state oraz poprawny końcowy averaged artifact. Średnia dwóch stanów jest obliczona; przejęcie schedulera następuje. **Rzeczywiste:** `n_averaged=2`, terminal parameters są równe average_model; `last_equals_terminal=false`, `best_equals_terminal=false`. Resume przed startem SWA wykonuje 2 zamiast 4 aktualizacji, a z checkpointu epoch=02 (SWA już aktywne) wykonuje 0 zamiast 2, global_step pozostaje 6. Scheduler po tym wznowieniu pozostaje pusty (`state.pt: scheduler=None`). Brak pozostałych updates jest asercją FAIL; nierówność last i averaged terminal **nie** jest błędem sama w sobie.
- Przyczyna zawężona: Lightning SWA usuwa konfigurację schedulera podczas odczytu, aby odtworzyć SWALR w kolejnym `on_train_epoch_start`; w testowanym resume nie następuje już wykonanie właściwej epoki. To interakcja ze stanem pętli B10-F04, nie dowód błędnego wzoru uśredniania. SWA przenosi averaged weights na końcu train, natomiast last pozostaje stanem continuation sprzed transferu. Nie wolno zakładać, że best wybrany po val_loss jest modelem SWA. Po checkpointcie wewnątrz aktywnego SWA odtworzenie pętli dodatkowo podlega B10-F04.
- Scientific impact: nie można zamiennie opisywać best, last i averaged terminal jako tego samego modelu, ani deklarować pełnej zgodności SWA-resume. Sama różnica best vs final nie dowodzi błędu matematycznego SWA.
- Minimalna remediation / DoD: osobno określić checkpoint continuation oraz artefakt SWA do inferencji i jego walidację; testować checkpoint przed/po aktywacji, przerwanie przez ES, optimizer/scheduler/callback state i predykcje. Nie wyłączać SWA tylko po to, by zaliczyć audyt.

### B15-F01 — HIGH — nieskończony loss jest zastępowany stałą

- Plik/symbol: PF `MultiHorizonMetric._update_losses_and_lengths`, wywoływany przez ClinicalQuantileLoss.
- Reproducer: `test_B15_nonfinite_loss_must_raise_before_update`, `test_B15_real_training_nonfinite_input_target_gradient`.
- Oczekiwane: jawny błąd stanu numerycznego przed aktualizacją i brak poprawnego checkpointu. Rzeczywiste: metryka ostrzega i podstawia sumę strat **1e9**; przy dłuższej sekwencji następnie dzieli ją przez liczbę pozycji. W rzeczywistym negative fit backward kończy się wtórnym błędem braku grad_fn, zamiast diagnostyki pierwotnego NaN/Inf.
- Przyczyna: fallback odziedziczony z PF, sanity ClinicalQuantileLoss ostrzega tylko przy pierwszym wywołaniu.
- Scientific impact: pozornie skończona wartość metryki nie dowodzi poprawnego stanu. EarlyStopping(check_finite) nie wykryje pierwotnego NaN, jeśli otrzyma stałą.
- Minimalna remediation / DoD: finite checks target/output/loss na granicy batcha, fail-fast z jasną przyczyną, blokada checkpointu invalid; regresje NaN/Inf osobno oraz duża **skończona** strata bez arbitralnego obcinania.

### B15-F02 — HIGH — NaN gradientu nie przerywa produkcyjnej ścieżki

- Plik/symbol: `GradientNormLogger.on_after_backward` i granica optimizer step.
- Reproducer: `test_B15_nonfinite_gradient_callback_must_abort`; real TFT probe `nonfinite_gradient` w `test_B15_real_training_nonfinite_input_target_gradient`.
- Oczekiwane: produkcyjna odmowa przed AdamW. Rzeczywiste: NaN norm nie przekracza progu w porównaniu `>`; callback nie rzuca błędu. W realnym backward NaN dociera do `on_before_optimizer_step`; zatrzymuje go dopiero containment callback audytu.
- Przyczyna: warunek wielkości gradientu nie jest kontrolą skończoności. Clipping również nie dowodzi finite gradients.
- Scientific impact: brak zabezpieczenia aktualizacji przed stanem numerycznie niepoprawnym. Nie wykonano celowo takiej aktualizacji i nie twierdzimy, że powstał produkcyjny uszkodzony checkpoint.
- Minimalna remediation / DoD: finite check gradientów przed update, błąd i brak valid artifact; test hooks przed/po clip oraz finite dużych gradientów bez mylenia ich z NaN.

## Scientific impact i decyzje Work/użytkownika

**Implementation correctness:** pointwise ClinicalQuantileLoss realizuje obecną deklarowaną funkcję (factor2, średnia Q, próg i waga, redukcja valid positions). Nie stwierdzono future-unknown leakage w testowanym forward. Problem agregacji epoki i brak finite guards są osobnymi, potwierdzonymi uchybieniami.

**Scientific validity:** waga zależna od y zmienia cel estymacji względem nieważonego pinball. Etykieta q=.1 czy q=.9 sama nie daje kwantyla nieważonego rozkładu ani nominalnego pokrycia. Zasadność objective, wagi i kalibracja wymagają decyzji Work/Stage C; Stage B ich nie zatwierdza i nie zmienia. Nie wyciągano wniosków z jakości predykcji syntetycznych ani rzeczywistego test setu.

Do decyzji przed oddzielną remediation:

1. Wspierana semantyka kontynuacji: last pełny stan, best tylko do inferencji, weights-only jako nowy run; wymagany poziom exactness i granica epoki. Aktualnie brak potwierdzonej gwarancji.
2. Format i polityka zaufania checkpointów oraz wymagane fingerprints danych/configu; dopuszczalne migracje i wersja Lightning/PyTorch.
3. Rola SWA: który artefakt podlega walidacji i inferencji oraz co oznacza early stop przed startem/uśrednianiem. Proste wyłączenie SWA nie jest zatwierdzoną naprawą.
4. Kontrakt agregacji lossu i jawna walidacja budżetów; brak zmiany objective na podstawie metryk testowych.
5. Clinical weighting i interpretacja kwantyli — osobna decyzja metodologiczna, bez uruchamiania Stage C w tym zadaniu.

## Remaining risks / granice dowodu

- Nie potwierdzono exact resume ani na granicy epoki; mid-epoch nie implementowano ani nie certyfikowano. Własny RNG/callback probe nie naprawia zapisu stanów.
- Po naprawie deserializacji nadal występuje problem liczby updates; dlatego różnica final weights nie izoluje wpływu samego dropout/shuffle. Artefakty zachowują przebieg porównań i ich niespełnione warunki wstępne.
- Zmiana katalogu ModelCheckpoint w izolowanych child runs powoduje ostrzeżenie Lightning i brak przywrócenia części historii best/top-k. To jawna właściwość izolacji tego audytu, nie dowód uszkodzenia callbacków przy wznowieniu w tym samym katalogu. Round-trip EarlyStopping oraz brak serializacji GradientNormLogger zbadano niezależnie.
- Estimator niepełnego batcha testuje lokalny setup dataloadera bez fit, korzystając z connectora Lightning; nie zmienia stanu replay. Pełny cosine i SWA annealing nie były trenowane na TFT — scheduler sprawdzono tanim trace, SWA w limicie 4 epok.
- Nie testowano długich runów, innych urządzeń, AMP, wielu workerów ani generalizacji. Stały krótki target ma test lossu; nie jest to pełny eksperyment uczenia TFT na stałym sygnale. Crossing nie jest naprawiany sortowaniem.
- B11 ma jeden runtime konflikt złożony; osobne macierze konfliktów hash/q/schema/budget pozostają DoD remediation. Same metadane zapisane w checkpointcie nie są kontrolą zgodności.
- Same-patient temporal forecasting pozostaje jedyną interpretacją Stage A. Żadnych twierdzeń unseen-patient, kalibracji, skuteczności klinicznej lub poprawy MARD.
- Nie zmieniono istniejących historycznych raportów ani roadmapy tak, by ukryć ich wcześniejszy status. Ten raport jest nowym dowodem Stage B.

## Pliki i review

Nowe pliki: `configs/baseline_stage_b_audit.json`, `ml/scripts/diagnostics/audit_stage_b.py`, `ml/tests/test_baseline_stage_b.py`, `docs/BASELINE_AUDIT_STAGE_B.md`. Checkpointy, syntetyczne fixture, logi i JSON wyników są pod ignorowanym `ml/models/stage_b_audit/`. Nic nie zostało staged ani committed. Nie osłabiono `.gitignore`.

Remediation pozostaje osobnym zadaniem. Zamknięcie audytu z FAIL nie oznacza spełnienia bramki gotowości Baseline v1.0.
