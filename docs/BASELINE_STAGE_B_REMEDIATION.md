# Baseline v1.0 — Stage B remediation

Data: 2026-09-13. HEAD: `896c9a5469897ec4938cd4b0cd8665e460780b3f`, branch `research/baseline-audit`; working tree dirty, bez commitu.

## Verdict

**BLOCKED dla pełnej gotowości CPU+CUDA; PASS dla zatwierdzonej ścieżki CPU Stage B.** Na CPU zamknięto mechaniczne findings T0/T1/T2 i potwierdzono epoch-boundary replay. GPU jest widoczne, lecz deterministyczny backward TFT jest niewspierany przez zainstalowany stos. Nie osłabiono deterministyczności, nie zmieniono środowiska i nie rozpoczęto Stage C ani pełnego treningu. To wynik technicznej remediation, nie potwierdzenie jakości lub klinicznej wiarygodności baseline.

## Autoryzacja i zakres

Przeczytano w całości AGENTS.md, RESEARCH_WORKFLOW.md, STAGE_B_AUDIT_PLAN.md, BASELINE_AUDIT_STAGE_B.md, STAGE_B_REMEDIATION_PLAN.md i CODEX_STAGE_B_REMEDIATION_TASK.md. Źródłem aktywacji T0/T1/T2 jest najnowsze polecenie użytkownika w załączniku `7aca0dff-09d7-43d9-a1fd-5aa1243e10f4/pasted-text.txt`, nie dawne rekomendacje oznaczone PENDING.

Zatwierdzone decyzje: D1 — resume wyłącznie po ukończonych batchach epoki i walidacji; D2 — SWA OFF, ON odroczone; D3 — niezmieniona clinical weighting, ocena naukowa w Stage C; D4 — jawne BEST/LAST/WEIGHTS-ONLY i sprawdzanie zgodności bez migracji. Zakres nie obejmuje danych Stage A, splitów, architektury TFT, test-set protocol, backendu, frontendu ani XAI. Nie użyto rzeczywistych test predictions/metrics ani test performance do decyzji. Nie wykonano Optuny.

Kanoniczny Stage A: `ml/data/processed/baseline_v1_stage_a_20260913T115538143463Z/`, SHA-256 `d14fe31b1b81971639ca8d5ba17b4882fbd6711569e30785f0da7b91a0c031cf`, provenance commit `5ee8d930e80fd7d62d0248c52ed10aaaaaa7f8d0`, dirty=false. SHA ponownie sprawdził reproduktor przed zmianami. Wszystkie testy modelu korzystają z syntetycznych fixture; kanoniczne dane nie służą do treningu w tym zadaniu.

## Findings i dowody

| Finding | Status | Zmiana i regresja |
|---|---|---|
| B15-F01 | FIXED CPU | Kontrole wejść, targetu, wyjścia, valid-position loss, sumy, akumulatora i redukcji przed fallbackiem PF do 1e9. Testy NaN/±Inf, overflow, padding, normalizatora i walidacji. |
| B15-F02 | FIXED CPU | Kontrole gradientów po backward, po closure/clipping i przed AdamW; stabilna norma float64; kontrola parametrów i optimizer state po kroku. Złe gradienty: 0 aktualizacji, także accumulation=2. Overflow po kroku: abort po 1 aktualizacji, bez valid checkpointu. |
| B11-F01 | FIXED CPU | Wspólny kontrakt dataset/model/normalizer/q/budget/code; każda składowa osobno perturbowana, odmowa przed deserializacją; kontrola payload kontra manifest. |
| B11-F02 | FIXED CPU | Jawny indeks ról, brak filename/mtime discovery i automatycznej test evaluation; testy corrupt/foreign/partial/role/atomic publication, pełnej precyzji monitora i remisu. |
| B10-F01 | FIXED CPU | Pełny odczyt wyłącznie własnego artefaktu po ownership/hash/compatibility; poprawny odczyt w nowym procesie bez globalnego override deserializacji. |
| B10-F04 | FIXED CPU | Checkpoint przy końcu walidacji, nie train-epoch-end; publiczne wznowienie 8 vs 4+4 przy niezmienionych 4 epokach i budżecie 8. |
| B10-F02 | FIXED CPU | RNG Python/NumPy/Torch i callbacki odtwarzane po inicjalizacji, przed forward; jawny sampler seed+epoch i odrębny generator techniczny dataloadera. Pełny replay bitwise z dropout/shuffle oraz bez nich. |
| B10-F03 | FIXED CPU | Serializacja licznika i konfiguracji GradientNormLogger; 12→12, alert przy patience, zgodny reset, odmowa zmiany konfiguracji. |
| B03-F01 | FIXED CPU | Lightning waży raportowane batch means liczbą ważnych pozycji. Rzeczywisty Trainer.validate: 330/27 = 12.222222, reset kolejnej walidacji do 0; pointwise loss i batch gradient bez zmiany. |
| B08-F01 | FIXED CPU | epochs=0 daje ValueError; profil wymaga dodatniego skończonego budżetu. Dotychczasowy dodatni plan warmup/cosine zachowany i sprawdzony na 160 scalar updates. |
| B13-F01 | DEFERRED / OUT OF SCOPE | SWA ON nie zostało naprawione. Certyfikowana konfiguracja OFF i odmowa ON mają regresję; historyczna ścieżka eksperymentalna została zachowana poza CLI baseline. |
| CUDA deterministic TFT | BLOCKED | `upsample_linear1d_backward_out_cuda` nie ma deterministycznej implementacji w tym stosie; 0 optimizer updates. GPU replay i GPU finite-injection acceptance nie zostały wykonane. |

Wcześniejszy audyt nie dowodził zapisania uszkodzonego checkpointu po NaN ani błędu obecnego pełnohoryzontowego val_loss. Remediation nie dopisuje takich wniosków. B03 zmienia agregację raportowaną przy różnych długościach, nie optymalizowany gradient batcha ani clinical weighting.

## Resume guarantee

Na CPU wykazano **exact epoch-boundary resume w testowanym profilu**: te same końcowe wagi i optimizer state bitwise, scheduler state, Python/NumPy/Torch RNG, callback state, global step/epoch i batch order. Maksymalna różnica wag wynosi 0.0. Test obejmuje pełne cztery dalsze aktualizacje w nowym procesie, także accumulation=2. Seed 43 daje inny przebieg niż 42; dwie świeże próby seed 42 są identyczne.

Granica zapisu: wszystkie train batches, optimizer updates, step scheduler oraz walidacja danej epoki są zakończone; EarlyStopping już odczytał monitor. Lightning kończy własną bookkeeping epoki podczas publicznego resume. Nie zmieniano prywatnego fit_loop, liczby epok ani długości danych. Callback checkpointu aktualizuje własne pola bookkeeping ModelCheckpoint — zależność od konkretnej wersji Lightning pozostaje testowana i wersjonowana.

Minimalny reproduktor skalarny potwierdził: `save_on_train_epoch_end=True` kończył 4+2=6; `False` kończy 4+4=8. Początkowy TFT smoke po poprawie granicy odtworzył budżet, lecz ujawnił inne zużycie generatora dataloadera. Ostateczny `EpochSampler` działa identycznie w obu wariantach; techniczny worker seed jest oddzielony od porządku próbek i RNG modelu. To jawna nowa polityka losowania, zapisana w kontrakcie, nie próba odtwarzania środka starego iteratora.

Gwarancja ogranicza się do tego samego kodu, datasetu, kontraktu, stosu, urządzenia CPU, FP32, liczby wątków oraz `num_workers=0`. Testy wykonano z jednym wątkiem. Nie ma gwarancji mid-epoch, partial-accumulation, cross-device, innej wersji bibliotek, DDP, AMP ani SWA. Run zakończony budżetem/ES nie może być kontynuowany jako LAST. Sekwencje ES i odtworzenie jego stanu sprawdzono oddzielnym testem; nie prowadzono długiego eksperymentu do naturalnego wyczerpania patience=20.

## Checkpoint semantics i bezpieczeństwo

- **BEST:** najmniejsze pełne validation objective w indeksie, remis rozstrzyga wcześniejszy global step; do inference/późniejszej ewaluacji. Brak selekcji na podstawie nazwy lub czasu pliku.
- **LAST:** ostatnia poprawna ukończona granica, jawny run_id i pełny zgodny stan do kontynuacji tego samego runu. Brak fallbacku do BEST.
- **WEIGHTS-ONLY:** jawny eksport tensorów z własnego BEST, nowy model/optimizer/scheduler/RNG i nowy run_id; provenance wskazuje rodzica. Nie jest resume.

Kontrakt obejmuje protocol, canonical oraz actual dataset SHA, schema i kolejność cech/kwantyli, fitted normalizer semantics, window index, context/horizon, architekturę, objective, optimizer, scheduler, całkowity budżet, accumulation, sampler, callbacki, seed, device/precision/threads, biblioteki i hashe kodu. Resume porównuje wszystkie pola; BEST/weights-only nie wymagają identycznego nowego planu optymalizacji, lecz nadal wymagają zgodności danych/modelu. Nie ma automatycznej migracji.

Rejestr sprawdza własność katalogu, wersję, rolę, ścieżkę, symlink, SHA i kontrakt przed pełnym odczytem. Po odczycie sprawdza payload i skończoność. `weights_only=False` jest jawne wyłącznie dla zweryfikowanych własnych plików. Lokalny rejestr jest granicą zaufania, nie kryptograficznym uwierzytelnieniem przeciw osobie mogącej zmienić jednocześnie pliki i indeks. Nie importuje się obcych katalogów/checkpointów.

Publikacja: zapis do pliku tymczasowego, fsync, rename, walidacja payload, atomowa aktualizacja indeksu. Przerwany zapis nie przesuwa valid pointer. Awaria zapisuje osobne `failure-*.json`; poprzednie valid BEST/LAST pozostają. Stan z overflow po kroku zostaje porzucony, nie cofany sztucznie. Częściowych/terminalnych checkpointów poza kontrolowaną granicą nie wolno publikować.

## Testy, wykonania i limity

| Wykonanie / lokalny artefakt | Wynik | Czas zmierzony | Rzeczywiste optimizer calls |
|---|---|---:|---:|
| Przed zmianami: `ml/models/stage_b_audit/20260913T144215Z/` | 32 metody: 18 PASS / 14 FAIL; 15 failure events z subtestami | 131.114 s manifest | 229 = 69 TFT + 160 scalar |
| Minimalny boundary probe: `ml/models/stage_b_remediation/20260913T144500Z/boundary_before.json` | reprodukcja 6 vs 8 | osobny czas nieinstrumentowany | 14 scalar |
| Trzy exploratory TFT smoke w tym samym katalogu | budżet naprawiony, RNG/data order wymagały dalszej poprawki | osobny czas nieinstrumentowany | 16 TFT |
| Pierwszy harness: `20260913T150623Z/` | błąd discovery: zaimportowana historyczna klasa oraz konflikt pyarrow; 17 PASS / 14 FAIL / 2 ERROR events | 43.572 s | 161; stary manifest omyłkowo raportował 0 |
| Pierwszy właściwy suite: `20260913T150820Z/` | 29 PASS / 1 FAIL — błędne oczekiwanie liczby przyszłych raportów awarii | 163.253 s | 237 = 77 TFT + 160 scalar |
| Końcowy suite: `20260913T151632Z/` | **30 PASS / 0 FAIL / 0 ERROR / 0 SKIP / 0 XFAIL** | 160.394 s | 237 = 77 TFT + 160 scalar |
| Dodatkowe końcowe regresje: `final_focused/` | **2 PASS / 0 FAIL** | 7.194 s unittest | 0 |
| Stage A po zmianach + końcowy rerun | **31 PASS**, ponownie **31 PASS** | 7.236 s + 7.100 s | 0 |
| `cuda_final_full/` | BLOCKED przy backward, exit 1 | osobny czas nieinstrumentowany | 0 |

Łącznie końcowy zestaw ma **32 PASS** (30 w pełnym suite + 2 późniejsze focused tests), bez SKIP/XFAIL. Wszystkie finalne testy uruchomiono na końcowym kodzie produkcyjnym. Historyczne testy i raport Stage B pozostawiono bez zmian; nie przemianowano ich FAIL na XFAIL.

Łączny licznik wykonanych aktualizacji w tej sesji: **894 = 240 TFT + 654 scalar**, obejmujący nieudane próby i reproduktory, a nie tylko ostatni zielony suite. Korekta pierwszego harnessu wynika z uruchomionego historycznego gradient oracle (1) i scheduler oracle (160); lokalny manifest zbiorczy jawnie zachowuje tę korektę. Suma zmierzonych czasów tabeli wynosi **519.862 s**; to dolna granica czasu wykonania testów, ponieważ eksploracyjne probe/GPU nie miały osobnego timera. Nie przedstawiam tego jako całkowitego czasu sesji. Każdy fit TFT miał najwyżej 8 aktualizacji i 4 epoki; każdy scheduler test najwyżej 160 scalar updates. Pełne suite miały watchdog 600 s, procesy potomne 75 s. Nie poszerzano budżetu pojedynczego fit, żeby uzyskać replay.

Polecenia reprodukcji:

```bash
.venv/bin/python -m ml.scripts.diagnostics.audit_stage_b
.venv/bin/python -m ml.scripts.diagnostics.remediate_stage_b
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s ml/tests -p test_baseline_stage_a.py -v
.venv/bin/python -m unittest ml.tests.test_baseline_stage_b_remediation.Remediation.test_B16_config_and_callback_contracts ml.tests.test_baseline_stage_b_remediation.Remediation.test_B16_accumulation_nonfinite_and_partial_publication -v
.venv/bin/python -m ml.scripts.diagnostics.remediate_stage_b --worker ml/models/stage_b_remediation/cuda_final_full --registry ml/models/stage_b_remediation/cuda_final_owned --device cuda
```

Runner ustawia lokalne katalogi logów/cache i CPU thread limit; dokładne command, wersje, hashe oraz liczniki są w manifestach. Katalogi worker są immutable: przy powtórce GPU należy wybrać nowy katalog. Syntax check sześciu plików Python PASS; importy sprawdzone podczas testów. `git diff --check` PASS.

### Mapa historycznych testów do acceptance

| Historyczne grupy testów | Nowe acceptance / status |
|---|---|
| B00–B06, finite gradient oracle, units, quantiles, future unknown | Zachowane asercje; B03 dodatkowo rzeczywisty epoch log w Trainer |
| B07 clipping/accumulation/frozen callback order | Stabilny clipping, post-closure gate, accumulation replay i nonfinite microbatch |
| B08 scheduler/budget/estimated steps | Oracle 160 steps + jawny skończony profil i accumulation replay; inne budżety unsupported |
| B09–B11 roundtrip/load/progress/RNG/compatibility/selection | Nowe procesy 8 vs4+4, registry roundtrip, axis/contract perturbations, role/corrupt/atomic tests |
| B12 ES | Zachowane sekwencje i state resume; callback state również w replay |
| B13 SWA | Historyczne reproduktory pozostają; ON DEFERRED, acceptance OFF i odmowa ON |
| B14 seeds | Dwa świeże procesy seed42, kontrola seed43, pełne trace |
| B15 finite | Bezpośrednie loss/gradient/overflow oraz rzeczywisty fit z containment, który nie może być pierwszą ochroną |
| B16 provenance | Wersje, hashe, mode/run lineage; brak wywołania real test evaluation w main |

## CUDA diagnostic

W sandboxie `nvidia-smi` exit 9, torch CUDA unavailable/device_count=0. Po zatwierdzonym wykonaniu poza sandboxem: `nvidia-smi` exit 0, **NVIDIA GeForce RTX 3070, 8192 MiB**, driver **615.71.09**, CUDA UMD **13.4**. Jest to RTX 3070, nie Ti z ogólnego opisu AGENTS.md.

Ta sama `.venv/bin/python`: torch **2.11.0+cu130**, `torch.version.cuda=13.0`, available=true, device_count=1. Operacja `arange(16).square().sum()` na GPU i synchronize: **1240.0, PASS**. Nie instalowano ani nie aktualizowano pakietów/sterowników/toolkitu.

Rzeczywisty syntetyczny TFT FP32 z `deterministic=True` wykonał forward/walidację, ale backward zgłosił: `upsample_linear1d_backward_out_cuda does not have a deterministic implementation`. Zero aktualizacji, brak valid BEST/LAST dla tego runu; powód awarii zapisany. GPU replay pozostaje **BLOCKED**. Nie użyto `warn_only`, nie wyłączono deterministyczności ani nie zmieniono architektury w celu obejścia błędu. Dalszy wybór wsparcia GPU wymaga osobnej decyzji; nie dziedziczy gwarancji CPU.

## Zmiany plików i pozostałe ryzyka

- `ml/scripts/train_tft_population_v2.py`: finite loss/model, optimizer gates, stan callbacku, jawne CLI/config/mode, kanoniczny SHA, brak automatycznej test evaluation; zachowana eksperymentalna ścieżka legacy.
- `ml/scripts/finite_training.py`: kontrole finite i stabilny clipping, adaptacja epokowej agregacji.
- `ml/scripts/checkpoint_registry.py`: kontrakt, własność, role, integralność i publikacja.
- `ml/scripts/baseline_training.py`: sampler/generatory, RNG callback, granica checkpointu i jawne orchestration.
- `ml/scripts/diagnostics/remediate_stage_b.py`: ograniczone syntetyczne subprocessy i provenance.
- `ml/tests/test_baseline_stage_b_remediation.py`: 32 acceptance tests.
- `configs/baseline_v1.json`: wersjonowany profil SWA OFF, kanoniczne dane i jawne założenia.
- Ten raport; manifest zbiorczy lokalnie w `ml/models/stage_b_remediation/remediation_provenance.json`.

Scientific impact: nowy training protocol `nmd-baseline-v1.0-stage-b-1` z SWA OFF, ES patience20/min_delta1e-4, jawnym samplerem i naprawionym epoch loss reporting. Starszych eksperymentów nie wolno traktować jako identycznego protokołu ani automatycznie wznawiać. Architektura, próg 70 mg/dL, hypo weight 2.5 i factor2 pozostają bez zmian. Zasadność clinical weighting, kalibracja kwantyli oraz rzeczywista skuteczność wymagają Stage C; tutaj ich nie oceniano.

Pozostałe ograniczenia: deterministyczne CUDA BLOCKED, SWA ON odroczone, mid-epoch unsupported, brak dowodu skalowania pełnego treningu/zużycia VRAM. Koszt kontroli finite i przechowywania każdej poprawnej granicy wymaga pomiaru w przyszłym jawnie zleconym eksperymencie; nie redukowano kontroli dla wydajności. Integracja PF/Lightning opiera się na testowanym stosie (Lightning 2.6.1, PF 1.7.0, Python 3.14.7, NumPy 2.4.4, pandas 2.3.3) i wymaga ponownej walidacji po aktualizacji.

Na wejściu istniały nieśledzone `docs/CODEX_STAGE_B_REMEDIATION_TASK.md` i `docs/STAGE_B_REMEDIATION_PLAN.md`; pozostawiono je bez edycji. Nic nie staged ani nie commitowano, nie zmieniano `.gitignore`, danych pacjentów ani historycznych raportów. Osobny review gotowości jest nadal wymagany przed pełnym treningiem.
