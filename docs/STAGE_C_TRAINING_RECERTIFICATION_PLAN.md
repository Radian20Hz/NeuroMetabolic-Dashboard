# Training recertification after C02 — plan

2026-09-14 · nmd-c02-training-recertification-1 · **PLAN READY / EXECUTION NOT STARTED**

## Decyzja i granica zlecenia

Potrzebny jest krótki controlled fit na syntetycznych danych. Stage C implementation PASS (19 PASS / 0 FAIL / 0 ERROR) obejmuje zero backward i zero optimizer updates: nie sprawdza działania zmienionej skali danych przez backward, AdamW, zapis i rzeczywiste resume. Minimalna kampania zachowująca zakres wcześniejszej gwarancji to CPU strict 8 vs 4+4 (16 updates), następnie CUDA seeded trzy świeże 8-step runs i jedno 4+4 (32 updates). Łącznie 48 updates; nie jest potrzebny fit na danych pacjentów.

Ten dokument i [task wykonawczy](CODEX_STAGE_C_TRAINING_RECERTIFICATION_TASK.md) są specyfikacją do późniejszego jawnego uruchomienia. W sesji przygotowania: zero treningu, forward/backward, test runners i optimizer updates. Nie otwarto test setu ani checkpointów. Kod produkcyjny i konfiguracje pozostają bez zmian. Recertyfikacja nie jest zgodą na pełny Baseline training.

## Podstawa i aktualny stan

Repo `/home/radian/NeuroMetabolic Dashboard`, branch `research/baseline-audit`, HEAD `c2e27a28824cd9ec015c362b7e8df65f3c0902dc`, clean przed utworzeniem dokumentów. Bez fetch; zgodność zdalnego HEAD nie jest świeżo potwierdzona.

Przeczytano raport/plan remediation, final review i finalization Stage B, konfigurację oraz C02 i registry. Aktualny końcowy Stage B to **PASS w pierwszej sekcji finalization po naprawie callback restore**; zachowany niżej FAIL oraz wcześniejszy CUDA PENDING są historyczne. Nie należy traktować ich jako aktualnego nierozwiązanego błędu.

C02 dopasowuje native GroupNormalizer(log, center=True) wyłącznie do obserwowanych train targets, z subject_id zakodowanymi tym samym NaNLabelEncoder, którego używa TimeSeriesDataSet. Wcześniej fit używał surowych stringów, a transform integerów i mógł korzystać z fallback. Teraz nmd_subject_mapping oraz nmd_revision są jawne, unseen subject i brak observed train group powodują odmowę. Zachowano estymator odchylenia/epsilon biblioteki. Zmienia to rzeczywiste encoder_cont i target_scale oraz funkcję numeryczną modelu.

Obowiązują `nmd-baseline-v1.0-stage-c-1`, `observed-train-encoded-subject-v2`, `nmd-evaluation-stage-c-1`, `nmd-numerics-2`. Registry obejmuje normalizer/encoder state i source hashes. Stare checkpointy nie są zgodne także w trybie weights-only; nie migrować ich ani nie dopisywać rewizji do starego obiektu. Nowy baseline startuje fresh.

## Dziedziczenie dowodów A/B

„Ważny” poniżej oznacza historyczny dowód dla niezmienionego komponentu po kontroli źródeł i stosu, nie świeże wykonanie całego Stage A/B.

| Dowód | Status po C02 | Wymagana czynność |
|---|---|---|
| Stage A: causal resampling/alignment, ograniczony ffill, observed masks, features/lags, chronologiczny split i brak leakage upstream | Zachowany warunkowo; C02 nie zmienia preprocessingu ani podziału | Porównać source hashes z końcowym A/B; odczytać tylko istniejące metadata, bez otwierania canonical parquet |
| Stage A: dense time index, kwalifikacja observed windows, placeholder handling | Logika zachowana, integracja z nowym normalizerem wymaga replay | Powtórzyć syntetyczne kontrole luk, flag i window keys przed fit |
| Stage A: dotychczasowe dowody poprawnych grupowych skal/encoder tensors | Nie przenoszą się; stary string-fit mógł przechodzić dzięki fallback | Niezależny log-statistics oracle C02 na rzeczywistych tensorach PF |
| Stage A canonical hash/split metadata | Nadal identyfikuje historyczny dataset; bez świeżej weryfikacji bajtów danych | Zapisać jako inherited, nie jako hash nowej synthetic fixture |
| Stage B: loss algebra, valid-length reduction, finite checks, scheduler policy, sampler i callback implementation | Źródła wspólnych modułów niezmienione przez commit C02; komponentowe dowody zachowane po weryfikacji | Targeted zero-update checks; nie uruchamiać pełnego 237-update suite |
| Stage B: CPU strict trajectory, gradient/optimizer state, 8 vs 4+4 | Historyczny PASS ważny wyłącznie dla starego input contract | Nowe CPU reference i split-run; wszystkie stany porównane |
| Stage B: CUDA C1/C2/C3 oraz resume | Nie certyfikują nowej funkcji modelu | Nowe C1/C2/C3/P/R; nie porównywać nowego R do starych referencji |
| Stage B: callback current_score fix | Mechanizm i focused unit evidence zachowane | Powtórzyć wszystkie pola bezpośrednio na granicy nowego CPU/CUDA resume |
| Stage B: ownership, BEST/LAST, compatibility | Zasady zachowane; schema/protocol i normalizer zmienione | Nowe positive/negative compatibility oraz real synthetic trained checkpoint roundtrip |
| Stage B: strict CPU / seeded CUDA numerical policy | Nadal obowiązuje, ale efektywne flagi wymagają potwierdzenia na urządzeniu wykonania | Preflight i obserwacja flag w każdym workerze |
| Stage C 19 mechanical groups | Dziedziczone, jeśli źródła pozostają zgodne z run02 | Powtórzyć C02 i checkpoint integration; nie deklarować ponownego 19/19 bez wykonania |

Nie unieważniać zbiorczo wszystkich 31 testów Stage A ani nie ogłaszać świeżego A/B PASS na podstawie samej macierzy. Długie scheduler scalar oracles mogą pozostać inherited, ponieważ ich implementacja nie zmieniła się; krótki fit kończy się w warmupie. Zmiana stosu lub kodu od wskazanego HEAD wymaga nowego impact review przed fit, nie automatycznego rozszerzenia tej kampanii.

## R0 — prerejestracja i kontrole bez aktualizacji

1. Zapisz branch/HEAD/dirty, hashe źródeł/configów/test harnessu, Python/dependencies, efektywne urządzenie/profile, tolerancje, fixture spec i pełny budżet przed pierwszym forward. Osobno odziedziczone evidence i nowo wykonane testy. Nie aktualizuj pola training_recertification w configu: wynik będzie osobną atestacją powiązaną z hashem configu.
2. Fixture wytwórz samodzielnie, wyłącznie synthetic, co najmniej dwie grupy o różnych log-średnich i wariancjach. Dodatnie targety obejmują obie strony istniejącego progu70; dodaj kontrolowane nieobserwowane pozycje i assessment sentinel. Brak osób lub wartości skopiowanych z danych pacjentów. Użyj bieżących production dataset builders i observed-window filter.
3. C02 oracle: log mean/std z observed train per encoded group, zgodnie z ddof/epsilon PF1.7, actual encoder_cont oraz target_scale obu grup; train encoder mapping wspólny dla dataset i normalizatora; poison fallback nie zmienia znanych grup. Renaming IDs zachowuje wartości po dopasowaniu kluczy. Sentinel assessment i nieobserwowane targets nie zmieniają fit; unknown subject / no-observed group są odrzucane. Porównanie FP32 do niezależnego FP64 oracle z zamrożonym atol=1e-5, rtol=1e-5, dopasowane jednostki; identities/mapping/keys exact.
4. Powtórz syntetyczne Stage A granice observed/ffill/window kwalifikacji i dense-grid; pełny decoder horizon12. Sprawdź real order/quantiles i target units bez evaluator scoring. Zachowaj production lengths, nie zmieniaj ich dla uzyskania PASS.
5. Targeted B: wszystkie dziewięć pól callback restore, missing/mismatched state refusal, comparator positive i celowa perturbacja negative; LR formula dla ośmiu zaplanowanych kroków bez wywołań optimizer.step; loss factor2, hypo<70, weight2.5 i valid-length reduction bez zmiany objective. Własne fixture manifesty sprawdzają starą rewizję/protocol, mapę, statystyki, feature order, Q, profile i code hash: odmowa zanim nastąpi deserializacja. Brak/obcy/zmodyfikowany artefakt i pomylona rola także odmawiają.

Nie uruchamiać bez selekcji historycznych runners: zawierają stare konfiguracje, dłuższe suite i własne budżety. Można reuse ich helpery po inspekcji; nowe test-only entry point musi jawnie używać bieżącego contract i synthetic_audit=true, z rzeczywistym SHA fixture. Stała canonical Stage A jest inherited protocol identity, a nie SHA syntetycznego pliku.

## R1/R2 — controlled fit

Jedna zamrożona fixture dla wszystkich dodatnich runów: context48/horizon12, hidden8/continuous4/head1/LSTM1/dropout0.1, batch2, cztery pełne training windows (po dwa z każdej grupy), cztery pełne validation windows obejmujące obie grupy. Wybierz klucze deterministycznie przed wynikami; zahashuj populację po filtrowaniu. Nie używaj pierwszych czterech indeksów, jeśli obejmują tylko jedną grupę. Synthetic validation jest wyłącznie instrumentacją treningu, nie oceną performance.

4 epoki ×2 training batches =8 wywołań AdamW, accumulation1, shuffle=true, seed42, FP32, workers0, CPU threads1, LR3e-4, clip1, SWA OFF. Produkcyjne AdamW/clinical loss/scheduler/callbacks i `train_baseline` przez publiczny Trainer.fit. Dropout i shuffle pozostają aktywne. Mały model zachowuje ścieżkę backward interpolacji. Odstępstwa architektury/batch/dropout/epochs od pełnego configu mają być jawne w fixture config; niczego nie wpisywać do baseline configu.

| Faza | Osobne świeże procesy | Actual updates |
|---|---|---:|
| R1 CPU strict | U=8, P=4, R=4 | 16 |
| R2 CUDA seeded | C1=8, C2=8, C3=8, P=4, R=4 | 32 |

P zatrzymuje się po walidacji epoki2 (stop_after_epoch=1) na niekońcowej granicy. Od początku plan total8/max_epochs4, również w P i R. R w nowym procesie, tym samym owned run directory, jawne resume-last. Nie skracaj kontraktu P do dwóch epok. U i P oraz Ck/P inicjalizują się niezależnie tym samym seedem; nie kopiuj im wag referencji.

CPU: U vs P+R exact dla batch order/digests, losses, LR, gradient norms, wszystkich parametrów/buffers, optimizer/scheduler/RNG i końcowych stanów semantycznych. Ścieżki i run IDs między niezależnymi runami są różne: porównaj ich prawidłową rolę/lineage, nie wymagaj tej samej nazwy katalogu. W samym resume wszystkie zapisane ścieżki są exact.

CUDA: wszystkie trzy pary fresh oraz każdy Ck vs P+R. Symetrycznie abs(x-y) ≤ atol + rtol*max(abs(x),abs(y)); każdy element, nie tylko średnia modelu:

| Miara trajektorii CUDA | atol | rtol |
|---|---:|---:|
| Batch loss i epoch synthetic val_loss | 1e-5 | 1e-4 |
| Pre/post-clip gradient norm | 1e-5 | 1e-3 |
| Każdy named parameter | 1e-6 | 1e-4 |
| LR użyty przy każdym update | 0 | 0 |

Keys/counts/order, init tensors i contract exact. Pozostałe floating optimizer/buffer stany po kontynuacji rejestruj w całości i wymagaj finite; nie deklaruj bitwise CUDA continuation. Wszystkie stany restore na samej granicy są exact także na CUDA. Nie porównuj trajektorii CPU z CUDA. Brak wymogu spadku loss lub jego wartości docelowej; nie wybieraj seedu, fixture, checkpointu referencyjnego ani tolerancji według jakości wyniku.

## Checkpoint i finite gates

Po native restore, przed pierwszym resumed forward, porównaj payload i rzeczywisty stan: wszystkie model tensors, optimizer moments/steps, scheduler/LR, global_step/epoch/next batch, Python/NumPy/torch CPU i CUDA RNG, generatory loaderów oraz callbacks. BoundaryCheckpoint: monitor, best_model_score, best_model_path, current_score, dirpath, best_k_models, kth_best_model_path, kth_value, last_model_path. EarlyStopping: wait_count/best_score/patience/stopped_epoch/stopping_reason/message; GradientNormLogger: counter/clip_val/warmup_steps. Potwierdź jedno wywołanie native load_state_dict i tę samą instancję callbacku. Nie pomijaj pustych/zerowych pól ani current_score.

Z nowych syntetycznie trenowanych runów zweryfikuj BEST wybrany z wszystkich finite epoch val_loss (remis: wcześniejszy global_step) oraz LAST na przerwanej granicy. Przez Registry.verified załaduj owned BEST w osobnym procesie bez fit; porównaj wszystkie wagi, Q, normalizer stats/mapping, schema i lineage z zapisem, także tensor normalization po dataset reconstruction. Nie uruchamiaj evaluator ani prediction metrics. Użyj nowych własnych artefaktów; nie otwieraj historycznych trained checkpointów. Weights-only export/load sprawdź bez aktualizacji: zgodne semantyki i nowy optimization run, nie resume. Nie dopuść synthetic checkpointów do pełnego baseline.

Dla CPU i CUDA po trzy osobne negative probes: nonfinite input, target, gradient przed pierwszym update. Wymagaj konkretnego produkcyjnego fail-fast, zero AdamW calls i brak valid BEST/LAST; złapanie przez containment wcześniej niż produkcja jest FAIL. Loguj failure record. Kontrole nie wymagają późnych injection ani post-update corruption, których niezmienione dowody Stage B pozostają inherited.

## Urządzenia i budżet

CPU strict obowiązkowe. CUDA seeded obowiązkowe przed planowanym treningiem na CUDA; brak dostępnej GPU to CUDA BLOCKED/NOT RUN, nigdy CUDA PASS lub silent CPU fallback. CPU-only wynik kwalifikuje wyłącznie cpu-strict; nie zezwala automatycznie na zmianę planowanego urządzenia pełnego runu.

Jeden lokalny proces na GPU, bez DDP/AMP/multiworker. FP32 ieee dla global/matmul/cuDNN, matmul highest, cuDNN deterministic=true, benchmark=false, CUBLAS_WORKSPACE_CONFIG=:4096:8. Deterministic algorithms=true; warn_only=false CPU, true CUDA. Zweryfikuj efektywne flagi po Trainer init i przed forward/backward/checkpoint. Jedyny zaakceptowany nondeterministic operator: upsample_linear1d_backward_out_cuda. Nowy operator/zmiana stosu: stop do review, bez wyłączania ochrony. Nie zmieniaj sterowników lub pakietów.

Historyczny stos: Python3.14.7, torch2.11.0+cu130, Lightning2.6.1, PF1.7.0, numpy2.4.4, pandas2.3.3; CUDA build13.0/cuDNN91900. Raport B wskazuje RTX3070 8GB/driver615.71.09, AGENTS ogólnie RTX3070Ti: przyszły preflight musi zapisać faktyczne urządzenie, bez zgadywania. Ta sesja nie wykonała nowego GPU probe.

**Twarde limity całej aktywowanej kampanii:** 48 optimizer calls łącznie (CPU16 + CUDA32), w tym krok z LR=0; 8 dodatnich fit workers i 6 negative workers; ≤8 updates/4 epoki per fit, P/R po4; negative i R0/roundtrip zero updates. ≤600 sekund wall-clock wykonania po prerejestracji, ≤90 sekund na worker, sekwencyjnie; do600 GPU-sekund konserwatywnego limitu i10 minut CPU wall occupation. Lokalnie, zero cloud spend/zakupów. Nie jest to gwarancja czasu ani wycena energii. Timeout parent musi zakończyć całą process group, zachować partial evidence i oznaczyć FAIL/INCOMPLETE.

Limit obejmuje nieudane próby i wszystkie rzeczywiste AdamW calls, nie tylko zakończone updates. Licz attempt przed delegacją, completion osobno; parent agreguje procesy. Forward ≤256, backward ≤64 łącznie, wlicz sanity validation i probes; fixture candidates ≤1024, jednocześnie ≤8 qualified windows używanych w fit. R0 wolno używać dodatkowych małych fixture do oracles, w tym samym limicie candidates. Nie powiększaj limitów dla automatycznego retry. FAIL/ERROR/timeout przerywa zależne fazy; zachowaj pierwszy wynik i poproś później o nowy zakres, nie rerun-until-green.

## Acceptance / Definition of Done

PASS wymaga kompletnego R0, CPU strict replay, trzech CUDA fresh i wszystkich sześciu CUDA porównań, obu boundary restores, nowego trained synthetic BEST roundtrip, sześciu negative probes i zgodności budżetu. Każda wymagana bramka: PASS; zero FAIL/ERROR, brak SKIP/XFAIL maskujących bramkę. Jawnie osobne CPU status, CUDA status i overall. Niedostępny wymagany GPU lub brak evidence oznacza niepełną recertyfikację.

Zapisz preregistration, effective fixture config, manifest hashes, invocation/exit/timing/counters, surowe traces/state, named comparisons z najgorszym błędem/tensorem/krokiem, warnings, checkpoint ownership/roles i końcowy `docs/STAGE_C_TRAINING_RECERTIFICATION_REPORT.md`. Binaries/logs tylko nowy ignorowany katalog `ml/models/stage_c_training_recertification/<unique-id>/`; lekkie bezpacjentowe metadata w `experiments/`. Report wiąże exact production/config/fixture/environment hashes, odróżnia odziedziczone od nowych dowodów i stwierdza NO PERFORMANCE EVALUATION.

PASS dotyczy wyłącznie krótkiej mechaniki po C02. Nie dowodzi jakości modelu, kalibracji, nieważonej mediany, długiego convergence/drift, pełnego cosine na GPU ani VRAM pełnego hidden64/batch64. Ewentualny pełnowymiarowy capacity smoke należy jawnie zaplanować przy późniejszym launch preflight; nie jest częścią minimalnej recertyfikacji normalizatora. Nadal SWA OFF, brak mid-epoch/partial accumulation/cross-device/cross-stack guarantees. Clinical weighting bez zmian; calibration NOT IMPLEMENTED/DEFERRED.

Po raporcie STOP. Nie uruchamiaj pełnego training, Optuny, real train/validation/test evaluation ani remediation odkrytego błędu. Pełny baseline wymaga osobnego polecenia i świeżego startu z nowym contract.

## Snapshot wejściowych SHA-256

- `docs/STAGE_C_REMEDIATION_REPORT.md`: `8edd3ed80178daaf26c443ffa427c8528deaa74df54ae1e308b870a510d87017`.
- `docs/STAGE_C_REMEDIATION_PLAN.md`: `b6dd6a38a885de02d8e4c8bc11384decb889125d064ce5950817d8ef47986957`.
- `docs/STAGE_B_FINAL_REVIEW.md`: `edfaf0307fcd72b069c4849924d67b5a23d6881f5ffc7743b3c4e0f1913e4d20`.
- `docs/STAGE_B_FINALIZATION_REPORT.md`: `e04b65e07cc455f87c9e5047aba10d71a4c2644851a381002d1df31dd77e53e4`.
- `configs/baseline_v1_stage_c.json`: `a85bb26214253a855af066b590565b3b8b45beebb34302e686cc36bc752e1f35`.
- `ml/scripts/train_tft_population_v2.py`: `dcdb6c738d70958a4a105d81c965f4dd17af0db35143ba77d98ba5b639f24572`.
- `ml/scripts/checkpoint_registry.py`: `cb3b9ca3495faa06ee039c28c2085fb3a9ee1c246f9cdeba766d11707d25df3d`.
- `ml/scripts/baseline_training.py`: `0cdef5a6311e3804cda69dfbc936cafb5fa4b560e12e53c71250c0b569018d74`.
- `ml/scripts/finite_training.py`: `3e0adea81cd1fdef6fec933213b1c1cb84b5fe8172716c7a69acab78e4abbf6a`.
- `ml/scripts/numerical_profile.py`: `5bf981d01877b1900a0c8f0426a5aa27966ec9d6a64ceb9c8639e0306c0e5630`.
