# Stage B finalization — CPU strict / CUDA seeded

## Aktualizacja 2026-09-13 — callback restore: PASS

**Stage B final verdict: PASS w zakresie certyfikowanej ścieżki Baseline v1.0.**
CPU strict epoch-boundary replay zachowany; CUDA seeded FP32 epoch-boundary resume
przechodzi zatwierdzony kontrakt. SWA OFF. Brak gwarancji bitwise CUDA, mid-epoch,
partial-accumulation, cross-device/cross-stack, AMP, DDP lub pełnego treningu.
Stage C nie rozpoczęto. Poniżej zachowano w całości wcześniejszy wynik FAIL;
ten rozdział dokumentuje osobno jego zatwierdzoną naprawę i nowe dowody.

### Root cause — ustalenia przed naprawą

HEAD na wejściu: `dbea59b06c5a8f6c5eec83dc1050de2f7390cb75`, branch
`research/baseline-audit`, working tree clean. Przeczytano AGENTS.md, raport
remediation, final review, ten raport, aktualne testy oraz lokalną implementację
Lightning2.6.1. Reproducer uruchomiony przed zmianą ponownie dał **FAIL**:
`current_score` po load było None; wynik zachowano w `before.log`.

Lokalny `ModelCheckpoint.state_dict()` zapisuje dziewięć pól: `monitor`,
`best_model_score`, `best_model_path`, `current_score`, `dirpath`, `best_k_models`,
`kth_best_model_path`, `kth_value`, `last_model_path`. Jego `load_state_dict()`
odtwarza historię best/top-k/last przy zgodnym dirpath oraz best_model_path,
lecz nie odczytuje current_score. Monitor i dirpath pozostają konfiguracją
utworzonego callbacku. `current_score` jest ponownie ustawiane dopiero przez
`_update_best_and_save()` z nowego monitora przy kolejnej walidacji.
Źródło: `.venv/lib/python3.14/site-packages/lightning/pytorch/callbacks/model_checkpoint.py`,
`state_dict`/`load_state_dict` około540–577, `_update_best_and_save` około950–975.

Kod biblioteki dowodzi tego pominięcia, ale **nie wyjaśnia intencji autorów**
ani nie udostępnia flagi zapewniającej pełny round-trip current_score. Nie
przedstawiamy tego pola jako natywnej gwarancji Lightning2.6.1. Minimalny natywny
kontrakt operacyjny pozwalałby na current_score=None aż do nowej walidacji;
nie spełniałby jednak jawnego wymagania Stage B pełnego odtworzenia zapisanego
stanu. Zachowano wymaganie użytkownika, rozszerzając tylko publiczny hook odczytu
istniejącego pola; nie osłabiono testu ani nie stworzono wartości zastępczej.

Wykluczono inne przyczyny:

- Orkiestracja tworzy ten sam typ BoundaryCheckpoint: monitor=val_loss, mode=min,
  save_top_k=-1, save_last=false, save_on_train_epoch_end=false, ten sam katalog
  runu. Klucz stanu pozostaje identyczny. Nie ma wymiany callbacku po restore.
- Kolejność: utworzenie callbacków → publiczne Trainer.fit(ckpt_path=...)
  → checkpoint connector `_restore_modules_and_callbacks` → `restore_callbacks`
  → `_call_callbacks_load_state_dict`. Ten dispatcher wyszukuje stan po
  callback.state_key, kopiuje go i wywołuje load_state_dict na instancji znajdującej
  się w trainer.callbacks. Następnie Lightning odtwarza optimizer/scheduler/loops;
  pomiar wykonano w on_train_start po przywróceniu RNG, przed pierwszym forward.
- Źródła lokalne: `trainer/trainer.py` około1000–1075,
  `trainer/connectors/checkpoint_connector.py:restore_callbacks/restore_training_state`,
  `trainer/call.py:_call_callbacks_load_state_dict` około298–311.
- Registry weryfikuje artefakt i przekazuje ścieżkę do natywnego fit. Nie pomija
  callback restore, nie usuwa pola z payloadu i nie zastępuje go manifestem.
- Instrumentacja rzeczywistego CPU i CUDA resume potwierdziła **jedno natywne
  wywołanie loadera oraz identyczność obiektu** z callbackiem odczytanym w
  on_train_start. Porównano jego faktyczne state_dict z natywnym payloadem
  checkpointu, nie z pomocniczo odtworzonymi wartościami.

### Fix

Jedyna zmiana produkcyjna: `BoundaryCheckpoint.load_state_dict` w
`ml/scripts/baseline_training.py`. Po sprawdzeniu zgodności dirpath/monitor,
obecności i finite current_score wywołuje **super().load_state_dict(state_dict)**,
a następnie przypisuje **dokładnie state_dict["current_score"]** z tego samego
natywnego stanu przekazanego przez Lightning. Nie nadpisuje pozostałych pól,
nie zmienia state_key, konfiguracji, kolejności callbacków ani zapisu.

Nie dodano własnej kopii callback state do rejestru lub innego formatu.
Nie odczytuje się nazwy pliku, mtime ani val_loss manifestu i niczego nie
przelicza. To uzupełnienie deserializacji istniejącego pola przez publiczne API,
nie obejście pętli ani ręczne odtwarzanie wyniku naukowego. Site-packages bez zmian.
Brak pola lub inny monitor/katalog daje jawną odmowę zamiast częściowego restore.

W instrumentacji naprawiono również konieczny do oceny resume false negative
HF-F01: obie strony porównania model state_dict są kopiowane tym samym helperem
cpu(). Nie ignoruje się żadnego klucza/tensora; kontrola dodatnia przechodzi,
perturbacja wartości nadal nie przechodzi. Tolerancje pozostają niezmienione.
Oryginalne artefakty i wcześniejsza porównywarka w Git zachowują historyczny FAIL.

### Callback restore test

**PASS.** Zachowano nazwę i pierwotną asercję reproduktora, dodając sprawdzenie
wszystkich dziewięciu pól i dowód wywołania natywnego loadera. Osobno odmowa
niezgodnego/missing state. Unit test nie zastępuje integracji:

| Rzeczywisty stan przed pierwszym resumed forward | CPU strict | CUDA seeded |
|---|---|---|
| best_model_score, best_model_path, current_score | PASS | PASS |
| last_model_path, kth_best_model_path, kth_value, best_k_models | PASS | PASS |
| monitor, dirpath, komplet kluczy stanu | PASS | PASS |
| EarlyStopping: wait_count, best_score, patience, stopped_epoch, stopping_reason/message | PASS | PASS |
| GradientNormLogger: counter, clip_val, warmup_steps | PASS | PASS |
| BoundaryRNG: Python, NumPy, Torch, generatory; CUDA RNG na GPU | PASS | PASS |
| Natywny dispatcher i ta sama instancja callbacku | PASS | PASS |

save_last=false oznacza, że natywne last_model_path jest pustym ciągiem —
sprawdzono jego zgodność. Rola LAST pozostaje jawna w registry i nie jest
utożsamiana z natywnym save_last. save_top_k=-1 używa best_k_models/kth fields;
ich pełna zgodność została sprawdzona, bez pomijania pustych lub zerowych wartości.

### Validation / wyniki

| Wykonanie | Wynik | Aktualizacje | Czas |
|---|---|---:|---:|
| Reproducer przed naprawą | 1 FAIL, zachowany | 0 | 0.001 s |
| Ten sam targeted reproducer po naprawie | 1 PASS | 0 | 0.001 s |
| Wszystkie focused finalization tests, rozszerzone | 6 PASS / 0 FAIL | 0 | 0.090 s |
| Aktualny certyfikowany Stage B remediation suite | 32 PASS / 0 FAIL | 237:77 TFT+160 scalar | 160.012 s manifest |
| Stage A regression | 31 PASS / 0 FAIL | 0 | 6.683 s |
| CUDA P/R, osobne procesy | 4+4 PASS | 8 TFT | P14.136 s, R15.137 s |

**Końcowy Stage B: 38 PASS / 0 FAIL / 0 ERROR / 0 SKIP / 0 XFAIL**
(32 remediation +6 focused; targeted1 wykonano dodatkowo, nie doliczono go
ponownie do liczby unikalnych metod). Stage A31 PASS. Historyczny pre-remediation
`test_baseline_stage_b.py` pozostaje archiwum reproduktorów legacy, w tym SWA ON;
nie uruchamiano eksperymentalnego SWA ani nie przepisywano jego czerwonych wyników.

CPU suite: `ml/models/stage_b_remediation/20260913T180459Z/`.
Strict 8 vs4+4 z dropout/shuffle, bez nich i z accumulation2 pozostaje bitwise
zgodne; max abs weights0, optimizer/scheduler/RNG/order i callbacki PASS.
W każdym wznowieniu nowy test sprawdza też bezpośrednią granicę restore.

CUDA artefakty: `ml/models/stage_b_callback_restore/20260913_callback_01/cuda/`.
Nowa prerejestracja zatwierdzonego ograniczonego wykonania ma SHA-256
`8a9ace2bbfb88dc371e0bfe9d022e001df3f8b14bff4d647070bffff561eb9be`.
Budżet: wyłącznie P4 + R4, każdy proces limit90 s; żadnego nowego C1/C2/C3.
`comparisons.json`: wszystkie restore checks PASS, native_dispatch=true,
same_restored_instance=true, następna epoka/global_step poprawne, RNG bez
zużycia między restore i pierwszym forward. Wszystkie dodatkowe pola sprawdzono
na rzeczywistym callbacku należącym do Trainer.

P+R porównano offline z **każdą** zachowaną referencją C1–C3 z
`20260913_finalization_01`. Dla każdej pary max abs batch/validation loss,
pre/post gradient norm, LR oraz każdego parametru wynosi **0**; exceedances=0,
RMS=0, relativeL2=0, normalized error=0 (LR exact). Dane, inicjalizacja,
kolejność i środowisko są zgodne. Tolerancji nie zmieniono. Jedyny warning
niedeterminizmu nadal dotyczy zaakceptowanego upsample_linear1d_backward_out_cuda.

Ważność wcześniejszych 3×8 potwierdzono przed nowym GPU runem: AST orkiestracji
różni się wyłącznie dodaną metodą load_state_dict; hashe modelu, objective,
finite gates, numerical profile, registry, fixture generatora i configu pozostają
zgodne z dawną prerejestracją. Świeże runy nie wywołują loadera callbacku.
W porównaniu offline dopuszczono wyłącznie jawnie zreviewowaną różnicę code hash;
**nie poluzowano zgodności produkcyjnego resume** i nie migrowano starego
checkpointu. P i R używają tego samego nowego kodu i pełnego zgodnego kontraktu.

Łącznie w tym zadaniu245 optimizer updates (85 TFT+160 scalar), bez dodatkowego
treningu do uzyskania korzystnych wyników. Zachowane historyczne CUDA finite
probes3/3 PASS nadal dotyczą niezmienionych finite gates. Nie powtarzano całej
macierzy GPU ani 3×8. Syntax/import i git diff --check PASS.

Polecenia: istniejący `remediate_stage_b`, unittest dla
`test_baseline_stage_a.py` oraz `ml.tests.test_stage_b_finalization`; targeted
`Finalization.test_checkpoint_current_score_restored_at_boundary` przed/po.
Nowy runner ograniczonej regresji:

```bash
.venv/bin/python -m ml.scripts.diagnostics.check_callback_resume --prepare --root ml/models/stage_b_callback_restore/20260913_callback_01/cuda --reference ml/models/stage_b_finalization/20260913_finalization_01
.venv/bin/python -m ml.scripts.diagnostics.check_callback_resume --root ml/models/stage_b_callback_restore/20260913_callback_01/cuda
```

GPU wykonano poza sandboxem po zatwierdzeniu; jedynie ustawienia procesu
zatwierdzonego profilu, bez instalacji/zmian środowiska. Komendy, exit codes,
stany, hashe, tożsamość callbacku i porównania w lokalnym ignorowanym katalogu;
zbiorcza provenance w `20260913_callback_01/manifest.json`.

### Scientific impact / remaining limitations / Git

Nie zmieniono pytania naukowego, architektury, lossu, danych, splitów, optymalizacji,
ordering, progów lub profilu numerycznego. Naprawiono odczyt istniejącego stanu.
Nowy code hash blokuje niejawne wznawianie starszych checkpointów; nie ma migracji.
Zgodność bitowa zaobserwowana na GPU nie jest gwarancją bitwise CUDA. Obowiązują
pozostałe ograniczenia krótkich testów opisane poniżej. B13-F01 SWA ON pozostaje
DEFERRED poza baseline, nie blocker. Clinical weighting, kalibracja, coverage,
quantile crossing, rzeczywista ewaluacja i persistence pozostają do Stage C.
Nie uruchomiono pełnego treningu, Optuny, Stage C ani test-set performance.

Pliki: produkcyjny `ml/scripts/baseline_training.py`; instrumentacja
`ml/scripts/diagnostics/{callback_restore,check_callback_resume,finalize_stage_b,remediate_stage_b}.py`;
testy `ml/tests/{test_baseline_stage_b_remediation,test_stage_b_finalization}.py`;
ten raport. Nowe pliki to callback_restore.py i check_callback_resume.py.
HEAD pozostaje `dbea59b06c5a8f6c5eec83dc1050de2f7390cb75`, branch
research/baseline-audit, dirty od tych zmian; nic staged/committed/pushed.

## Historyczny wynik finalizacji przed naprawą callbacku


Data: 2026-09-13. **Stage B final verdict: FAIL.**

Trzy świeże CUDA runs oraz trajektoria 4+4 spełniają wszystkie prerejestrowane tolerancje. GPU finite safety: PASS. Dotychczasowa regresja CPU: Stage A 31 PASS, Stage B 32 PASS. Pozostała niespełniona dokładna bramka: `BoundaryCheckpoint` nie przywraca `current_score` na granicy resume. To nie jest problem zmienności CUDA ani SWA, ale nie pozwala zadeklarować pełnego checkpoint/callback restore zgodnie z zatwierdzonym kontraktem.

## Zakres i źródło decyzji

Zadanie uruchomiło bezpośrednie polecenie użytkownika przyjmujące **seeded reproducibility na CUDA, strict replay na CPU**, finalizację według CODEX_STAGE_B_FINALIZATION_TASK.md, wszystkie trzy fresh runs, P/R, trzy finite probes oraz pełną regresję CPU A+B. To polecenie zastępuje status „specyfikacja do aktywacji” w dokumencie tasku. Stage A wykonano ponownie zgodnie z bezpośrednim zleceniem, mimo że specyfikacja dopuszczała odniesienie do niezmienionych historycznych wyników.

Przeczytano w całości AGENTS.md, RESEARCH_WORKFLOW.md, BASELINE_AUDIT_STAGE_B.md, BASELINE_STAGE_B_REMEDIATION.md, STAGE_B_FINAL_REVIEW.md, CODEX_STAGE_B_FINALIZATION_TASK.md oraz configs/baseline_v1.json; sprawdzono ROADMAP.md, kod i lokalne API bibliotek. Brak dodatkowych AGENTS.md w podkatalogach.

HEAD na wejściu: `e548c2f7dce900565be2097250108c5f4cd07b7f`, branch `research/baseline-audit`, clean. Względem review na `84dd507b0888f8f5659f6a0ccd8a74a1ad78de96` doszły wyłącznie dwa dokumenty: final review oraz task finalization. Kod remediation nie uległ zmianie między tymi commitami.

Nie wykonano pełnego treningu, Optuny, Stage C, real test predictions/metrics ani zmian środowiska systemowego. Nie zmieniono ClinicalQuantileLoss, architektury, interpolacji TFT, protokołu Stage A, splitów, scheduler policy, backendu, frontendu ani XAI. SWA ON nie uruchamiano ani nie naprawiano. Zachowano historyczne raporty i ich wcześniejsze wyniki.

## Change / Reason

W aktywnej produkcyjnej ścieżce dodano wersjonowany wybór profilu urządzenia: `cpu-strict` oraz `cuda-seeded`, rewizja `nmd-numerics-2`, protocol `nmd-baseline-v1.0-stage-b-2`. CUDA używa publicznego `Trainer(deterministic="warn", benchmark=False)`, CPU nadal `deterministic=True`. Nie ma automatycznego retry ani fallbacku strict→seeded po błędzie.

FP32, AMP OFF, TF32 OFF ustawiono jednym nowym API PyTorch: globalne/backendowe `fp32_precision="ieee"`; efektywny `get_float32_matmul_precision()` zwraca `highest`. Ustawiono cuDNN deterministic=true, benchmark=false, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Są to zatwierdzone ustawienia procesu. Nie zmieniano pakietów, sterowników, toolkitu ani konfiguracji systemowej.

Profil i flagi są w konfiguracji oraz kontrakcie checkpointu, a hash nowego modułu jest częścią fingerprintu kodu. Produkcja sprawdza efektywne flagi po konstrukcji Trainer i przy zapisie checkpointu; finalizer dodatkowo przed forward/backward. Odmowa niezgodnego profilu jest sprawdzana przed deserializacją. Starszy strict artefakt nie jest automatycznie migrowany do nowego seeded runu.

## Zamrożony protokół i provenance

Artefakty lokalne: `ml/models/stage_b_finalization/20260913_finalization_01/`.

- `preregistration.json` zapisano przed pierwszym seeded GPU runem.
- SHA-256 prerejestracji: `7f66fe63d1efa13559abe07549c259181e11535547681fbb37e9c14ad44e397b`.
- SHA-256 `configs/baseline_v1.json`: `f0fcc65130f5b299c2a3c116da7adefb59bf746e9a6923aa8c53463ef88b57ab`.
- Prerejestracja zawiera pełny config baseline, config fixture, tolerancje, listę par, budżety, HEAD/dirty oraz hashe kodu i final review. Każdy worker sprawdził niezmienność prerejestracji i źródeł. Po wynikach nie zmieniono zamrożonego runnera, źródeł produkcji ani tolerancji.
- Surowe named parameters, buffers, optimizer/scheduler, RNG i stany callbacków: `C1/raw.pt` … `R/raw.pt`; trace i warning inventory w manifestach workerów. Nie zaokrąglano wartości przed porównaniem. Python float zachowuje pełną wartość odczytanego scalaru FP32/FP64.
- `comparisons.json` zawiera wszystkie pary, każdy tensor parametrów, wszystkie miary i per-step differences. `result.json` oraz `invocations.json` zawierają exit codes, czas i actual updates. `readback_review.json` jest osobnym odczytem tych samych, zahashowanych artefaktów; nie nadpisuje oryginalnego FAIL.
- Zbiorczy indeks SHA/provenance: `finalization_manifest.json` w tym samym ignorowanym katalogu.

Fixture jest niezmienioną syntetyczną fixture remediation: context48/horizon12, hidden8/continuous4/head1/LSTM1/dropout0.1, batch2, LR3e-4, clip1, accumulation1, workers0, seed42, FP32, SWA OFF, 4 epoki/8 updates. Nadal wykonuje problematyczny backward interpolacji. Każdy fresh proces niezależnie inicjalizował model. Instrumentacja nie konsumuje RNG; synchronizuje CUDA i obserwuje rzeczywiste gradienty oraz LR użyty przy AdamW.

Prerejestrowana symetryczna reguła: `abs(x-y) <= atol + rtol * max(abs(x), abs(y))`.

| Wielkość | atol | rtol |
|---|---:|---:|
| Każdy batch loss i epoch validation loss | 1e-5 | 1e-4 |
| Pre-/post-clip gradient norm | 1e-5 | 1e-3 |
| Każdy element named parameter | 1e-6 | 1e-4 |
| LR każdego kroku i param group | 0 | 0 |

Są to progi krótkiej regresji inżynierskiej, nie progi kliniczne. Nie zmieniono ich po wynikach.

## Hardware / effective flags / warnings

GPU: **NVIDIA GeForce RTX 3070**, capability8.6, driver615.71.09. PyTorch2.11.0+cu130, CUDA build13.0, cuDNN91900, Lightning2.6.1, PF1.7.0, Python3.14.7, NumPy2.4.4, pandas2.3.3. Jeden wątek CPU w fixture, jeden proces na GPU, num_workers0. GPU uruchomiono poza sandboxem po akceptacji wykonania; wcześniejszy brak dostępu w sandboxie nie został uznany za błąd sterownika.

W każdym pozytywnym workerze po inicjalizacji Trainer potwierdzono: deterministic algorithms=true, warn_only=true, benchmark=false, cuDNN deterministic=true, global/matmul/cuDNN FP32 precision=ieee, matmul precision=highest, workspace=:4096:8, AMP=false. Konfiguracja CPU ma warn_only=false; testy potwierdziły rozdzielenie flag i odrzucenie ich niespójności.

Jedyny warning niedeterminizmu: **`upsample_linear1d_backward_out_cuda`**, zgodny z zaakceptowaną listą. Brak nowych operatorów niedeterministycznych. Ostrzeżenia zachowano w JSON i logach. Pozostałe warningi dotyczą znanych zapisów `loss`/`logging_metrics` w hparams, istniejącego katalogu checkpointów własnego runu, małej liczby workerów oraz deprecacji pytree LeafSpec. Nie zastosowano sugerowanych zmian precision ani liczby workerów. Warn-only nie osłabia finite/checkpoint/data guards.

## CUDA execution i seeded reproducibility

| Proces | Actual optimizer updates | Global step końcowy | Exit | Czas procesu z uruchomieniem |
|---|---:|---:|---:|---:|
| C1 | 8 | 8 | 0 | 37.723 s |
| C2 | 8 | 8 | 0 | 20.396 s |
| C3 | 8 | 8 | 0 | 19.694 s |
| P | 4 | 4 | 0 | 12.983 s |
| R | 4 | 8 | 0 | 13.984 s |
| N_input | 0 | brak update | 1, oczekiwany | 5.973 s |
| N_target | 0 | brak update | 1, oczekiwany | 6.073 s |
| N_gradient | 0 | brak update | 1, oczekiwany | 6.223 s |

Łącznie **32 dodatnie aktualizacje i 0 negatywnych**, jeden zestaw, bez powtórzeń. Cały CUDA runner: **125.085 s**, exit1 z powodu bramki callback restore. Limit600 s/suite i90 s/proces zachowany; każdy fit≤8 updates/4 epoki. LR=0 pierwszego kroku również jest rzeczywistym liczonym wywołaniem AdamW.

Wszystkie dokładne invariants porównania fresh/trajectories: PASS — config/environment fingerprints, inicjalizacja (hash i tensor equality), fixture SHA, window index SHA, pełna kolejność i digest batchy, valid lengths, liczba kroków/epok oraz LR. P+R porównano z **każdym** Ck, bez selekcji referencji.

| Para | Max abs loss | Max abs val loss | Max abs pre norm | Max abs post norm | Max abs LR | Max abs parameter | Przekroczenia tolerancji |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1–C2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C1–C3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C2–C3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C1–(P+R) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C2–(P+R) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C3–(P+R) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Dla każdej pary, każdej wielkości i każdego named tensor: **bitwise equality=true, RMS=0, symmetric relative L2=0, normalized maximum error=0, exceedances=0**. Dla LR normalized error jest nieokreślony przy zerowym progu i nie jest liczony; exact=true. W każdym z ośmiu kroków max abs loss/pre/post wynosi0, a w każdej z czterech walidacji max abs=0. Nie ma wyróżniającego się najgorszego kroku/tensora: wszystkie remisują przy0; indeks0/pierwszy tensor w JSON jest wynikiem argmax remisu.

Norma całej aktualizacji parametrów w C1/C2/C3/P+R: **0.02149386811467695**. Drift/update ratio=0 dla każdej pary. Pełne trace, miary per tensor i indeksy są w lokalnych artefaktach. Nie zastąpiono porównania elementowego średnią modelu.

**CUDA seeded reproducibility C1–C3: PASS.** Zaobserwowana zgodność bitowa jest wynikiem tych prób, nie nową gwarancją bitwise CUDA. N=3 i osiem kroków w warmupie nie stanowią dowodu stabilności pełnego treningu ani rozkładu zmienności.

## CUDA resume — niespełniona bramka

**Wykonanie 4+4 i tolerancje trajektorii: PASS. Pełny boundary restore: FAIL.**

P zapisał własny LAST po walidacji drugiej epoki. R w nowym procesie odtworzył go w tym samym katalogu runu, z tym samym total8/max_epochs4, wykonał dokładnie cztery dalsze aktualizacje i zakończył global_step8/epoch4. Nie ma brakujących/zdublowanych batchy. Pierwszy resumed batch należy do next_epoch z metadanych checkpointu.

Optimizer state, scheduler state, Python/NumPy/Torch CPU/CUDA RNG, global step i zapisany epoch: dokładnie zgodne na granicy. RNG między snapshotem po restore a pierwszym resumed forward pozostał dokładnie zgodny. BoundaryRNG, EarlyStopping i GradientNormLogger state są zgodne. Nie stosowano normalizacji liczników lub metryk callbacków.

### BF-F01 — brak odtworzenia current_score

Potwierdzony problem mechaniczny: `BoundaryCheckpoint` dziedziczy `ModelCheckpoint.load_state_dict`. W Lightning2.6.1 `state_dict()` zapisuje `current_score`, lecz `load_state_dict()` nie przypisuje go z powrotem. Katalog runu jest ten sam, więc nie jest to przypadek odmowy odczytu z powodu zmienionego dirpath.

W checkpointcie P: tensor validation objective około22.8335; po restore R: **None**. Źródło: lokalny `lightning/pytorch/callbacks/model_checkpoint.py`, `state_dict` i `load_state_dict` (linie540–577 tego stosu). Reproducer bez treningu: `test_checkpoint_current_score_restored_at_boundary`, **FAIL**, 0 optimizer updates. Używa syntetycznej wartości1 i tego samego katalogu, więc błąd nie zależy od GPU ani wartości objective.

To pole jest nadpisywane przy następnej walidacji; dlatego wcześniejsze CPU porównanie końcowych callbacków oraz obecna zgodność trajektorii mogą przejść mimo niepełnego stanu bezpośrednio na granicy. Nie wykazano zmiany wag, LR lub wyboru BEST w tych ośmiu krokach. Jednak zadanie wymaga dokładnego callback restore i zabrania wyłączania metryk z porównania. **Nie wolno pominąć tego pola tylko po to, żeby ogłosić PASS.**

Minimalny zakres osobnej naprawy do review: lokalny `BoundaryCheckpoint.load_state_dict` przywracający również `current_score`, z zachowaniem istniejącej walidacji/roli i regresją stanu bezpośrednio po restore. Nie edytować site-packages ani tolerancji. Nie implementowano tego rozszerzenia w zadaniu ograniczonym do finalizacji; nie uruchomiono kolejnego GPU zestawu po wykorzystaniu prerejestrowanego budżetu. Potrzebna byłaby jawna nowa wersja wykonania/review, z zachowaniem obecnego FAIL.

### HF-F01 — fałszywe rozróżnienie kontenera state_dict w finalizerze

Oryginalne `restore_checks.model=false` ma inną przyczynę: helper `cpu()` kopiuje mapping do zwykłego dict, a checkpoint zawiera OrderedDict; `equal()` porównuje typ kontenera przed kluczami/tensorami. To defekt instrumentacji, nie brak odczytu wag.

Oddzielny read-only review sprawdził hash checkpointu i raw exportu, równość kluczy oraz **wszystkie1284 state tensors: bitwise identical**. Potwierdził też, że jedyną różnicą pól callbacków jest `current_score`. `readback_review.py/json` pozostają lokalnym dodatkowym dowodem, 0 updates. Oryginalnego wyniku ani zamrożonej porównywarki nie nadpisano. Przyszły review powinien skorygować reprezentację mappingów w helperze bez poluzowania porównań tensorów, RNG lub callback metrics. Nawet usunięcie tego false negative nie zmienia obecnego werdyktu FAIL przez BF-F01.

## CUDA finite safety

| Probe | Wynik produkcji | Actual updates | Valid BEST/LAST | Audit containment pierwszy |
|---|---|---:|---|---|
| Input NaN | FloatingPointError: model input | 0 | brak | nie |
| Target Inf | FloatingPointError: target | 0 | brak | nie |
| Gradient NaN | FloatingPointError: gradient | 0 | brak | nie |

**3/3 PASS.** Dla każdej próby powód awarii zapisano w osobnym `failure-*.json`. Exit1 jest oczekiwany dla poprawnego fail-fast, nie ukrywanym błędem harnessu. Nie powtarzano pozostałej macierzy CPU na GPU.

## CPU regression i tests

| Zestaw | PASS | FAIL | ERROR | SKIP | XFAIL | Czas | Exit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage A — świeże wykonanie | 31 | 0 | 0 | 0 | 0 | 6.741 s | 0 |
| Stage B remediation — pełny aktualny suite | 32 | 0 | 0 | 0 | 0 | 168.760 s manifest | 0 |
| Focused profile/comparator controls przed GPU | 3 | 0 | 0 | 0 | 0 | 1.282 s unittest | 0 |
| Nowy boundary score reproducer po GPU | 0 | 1 | 0 | 0 | 0 | 0.001 s unittest | 1 |

Bilans metod unittest: **66 PASS / 1 FAIL / 0 ERROR / 0 SKIP / 0 XFAIL**. To cztery jawnie oddzielne wykonania, nie jeden suite. Dodatkowo sześć GPU porównań numerycznych PASS, trzy negative probes PASS; cała finalizacja FAIL na dokładnej bramce restore.

CPU strict 8 vs4+4 z dropout/shuffle, bez dropout/shuffle i z accumulation2 nadal przechodzi; model/optimizer/scheduler bitwise, max abs weights0, końcowe callback states/RNG/order zgodne. Wynik dotyczy dotychczasowej testowanej gwarancji CPU. BF-F01 ujawnia dodatkową lukę bezpośredniego callback restore także we wspólnej klasie CPU — nie przedstawiamy wcześniejszych testów końcowych jako dowodu zaliczenia tej nowej bramki.

CPU runner: `ml/models/stage_b_remediation/20260913T174803Z/`, **237 updates =77 TFT +160 scalar**. CUDA32 TFT. Pozostałe kontrole0. Łącznie w tym zadaniu **269 optimizer updates =109 TFT +160 scalar**, bez rerun-until-green. Suma zmierzonych czasów suite/checks z tabeli i CUDA wynosi około301.869 s; obejmują one częściowo równoległe wykonania, więc nie są czasem ściennym całej sesji.

Syntax/import checks PASS, `git diff --check` PASS. Focused tests potwierdzają efektywne CPU/CUDA flagi po publicznej inicjalizacji Trainer, odmowę zmiany profilu, dodatnią i ujemną kontrolę symetrycznego comparatora oraz dokładny LR. Nie zastosowano skip/xfail do nowego problemu.

Główne polecenia (root repo; logi/cache lokalnie, thread limits1):

```bash
.venv/bin/python -m ml.scripts.diagnostics.finalize_stage_b --prepare --root ml/models/stage_b_finalization/20260913_finalization_01
.venv/bin/python -m ml.scripts.diagnostics.finalize_stage_b --run --root ml/models/stage_b_finalization/20260913_finalization_01
.venv/bin/python -m ml.scripts.diagnostics.remediate_stage_b
.venv/bin/python -m unittest discover -s ml/tests -p test_baseline_stage_a.py -v
.venv/bin/python -m unittest ml.tests.test_stage_b_finalization -v
```

Ostatnie polecenie na końcowych źródłach zbiera trzy kontrole i jawnie czerwony reproducer; w tej sesji trzy kontrole wykonano przed dodaniem reproduktora, a reproducer osobno. Dokładne komendy GPU workerów i exit codes są w invocations.json. Katalog prerejestracji/workerów nie może być użyty ponownie do nadpisania istniejących wyników.

## Oficjalny zakres i remaining limitations

- CPU: strict deterministic replay na zatwierdzonej granicy, w testowanym profilu i stosie. Nowa dokładna kontrola callback score pozostaje FAIL; nie rozszerza się dotychczasowych dowodów końcowego replay na wszystkie chwile życia callbacku.
- CUDA: seeded FP32 reproducibility spełnia tolerancje krótkiej regresji. **Brak gwarancji bitwise CUDA**, mimo zaobserwowanych zerowych różnic.
- **Brak gwarancji mid-epoch exact resume**, partial accumulation, cross-device/cross-stack, AMP, DDP, wielu workerów lub długiego treningu.
- **SWA OFF** dla Baseline v1.0. **B13-F01 SWA ON DEFERRED / OUT OF SCOPE, nie blocker Stage B**. Nie zmieniono go na FIXED.
- Znany nondeterministic CUDA backward jest zaakceptowanym ograniczeniem profilu; nie wymaga przebudowy TFT. Nowe warnings/operator wymagają osobnego review.
- Nie zbadano długookresowego driftu, pełnego cosine na GPU, VRAM pełnego modelu, jakości klinicznej ani generalizacji. Mały smoke kończy się w warmupie.
- Aktywna bramka BF-F01 i false negative HF-F01 wymagają osobnego ograniczonego review/naprawy przed finalnym PASS. Obecnego FAIL nie zastępuje poprawna zgodność samych wag.

## Deferred to Stage C / scientific impact

Do Stage C pozostają: naukowa zasadność clinical weighting, interpretacja kwantyli, calibration, empirical coverage/width, quantile crossing policy, metryki per horizon/patient/glycemic range, porównanie z persistence i właściwy evaluation protocol. Nie wykonano żadnego z tych etapów ani nie użyto test performance do decyzji.

Zmiana dotyczy jawnego kontraktu numerycznego; funkcja modelu, dane i objective pozostały bez zmian. Checkpointy nowego i starego profilu nie są tym samym runem i nie podlegają silent migration. Powtarzalność same-seed nie zastępuje badania wariancji między seedami ani nie dowodzi poprawy predykcji.

## Files modified / Git state

Zmodyfikowano:

- `configs/baseline_v1.json` — wersja i dwa jawne profile.
- `ml/scripts/train_tft_population_v2.py` — walidacja wersji i konfiguracji profili, bez zmian lossu/modelu.
- `ml/scripts/baseline_training.py` — aktywne flagi urządzenia i ich weryfikacja.
- `ml/scripts/checkpoint_registry.py` — numerical profile i hash jego kodu w kontrakcie.

Dodano:

- `ml/scripts/numerical_profile.py`.
- `ml/scripts/diagnostics/finalize_stage_b.py`.
- `ml/tests/test_stage_b_finalization.py` — trzy kontrole i jeden potwierdzony czerwony reproducer.
- `docs/STAGE_B_FINALIZATION_REPORT.md`.

HEAD pozostał `e548c2f7dce900565be2097250108c5f4cd07b7f`; branch `research/baseline-audit`, working tree dirty wyłącznie od powyższych zmian. Nic staged/committed/pushed. Historycznych raportów nie nadpisano. Lokalny manifest, syntetyczne fixture, surowe tensory, checkpointy i logi są pod ignorowanym `ml/models/`. Nie zmieniono .gitignore ani danych pacjentów. Stage C nie rozpoczęto.
