# Stage B — checklista i plan testów

Wersja 1.0, 2026-09-13. Status: SPECYFIKACJA, nie wynik audytu.

## Podstawa

Przeczytano AGENTS.md, docs/ROADMAP.md, docs/BASELINE_AUDIT_STAGE_A.md i docs/BASELINE_AUDIT_STAGE_A_REMEDIATION.md. Dodatkowo sprawdzono configs/baseline_stage_a.json, kanoniczny provenance.json i odpowiednie fragmenty ml/scripts/train_tft_population_v2.py na HEAD `5ee8d930e80fd7d62d0248c52ed10aaaaaa7f8d0`.

Stage A zamknięto w zakresie A01–A07. Stage B sprawdza poprawność mechaniki uczenia i artefaktów, bez oceny jakości modelu. Kalibracja, naukowa ewaluacja i generalizacja należą do dalszych etapów.

Inspekcja wskazuje konkretne punkty do sprawdzenia, nie wyniki testów:

- ClinicalQuantileLoss.loss: średnia po kwantylach, potem mnożnik 2.5 dla target <70 mg/dL, jednorazowe ostrzeżenia zakresu.
- ClinicalTFT.configure_optimizers: AdamW, LambdaLR → CosineAnnealingWarmRestarts przez SequentialLR; minimum 50 kroków warmupu; długości zależą od estimated_stepping_batches i max_epochs.
- find_best_checkpoint: wybór po zaokrąglonym lossie w nazwie i fallback po mtime, bez sprawdzenia manifestu w tym helperze.
- build_model i train: load_from_checkpoint oraz ckpt_path przekazany do fit; trzeba sprawdzić pełne odtworzenie stanu.
- train: precision=32, deterministic=False, SWA domyślnie włączone, val_loss steruje early stopping i zapisem checkpointu.
- main: po treningu uruchamia evaluate na walidacji i na rzeczywistym test secie. Nie wolno używać go jako skrótu do smoke testu Stage B.

## Wspólny protokół testów

Dane syntetyczne, bez kopiowania rekordów pacjentów. Użyj rzeczywistego PF DataLoadera, ClinicalTFT, ClinicalQuantileLoss i produkcyjnych callbacków tam, gdzie badany jest ich kontrakt. Izoluj katalogi, loggery i checkpointy od historycznych runów; wyłącz integracje sieciowe. Nie uruchamiaj main ani tune_tft.

Referencja: CPU, float32, jeden wątek, num_workers=0, seed 42. Drugi seed 43 stanowi kontrolę reakcji inicjalizacji. Jednostkowy gradient check lossu: float64 z punktami poza załamaniami funkcji. Typowy smoke: batch 2–4, mały TFT, context 48/horizon 12; osobno krótkie decodery. Do 8 aktualizacji optymalizatora na przebieg, najwyżej 4 syntetyczne epoki. Test LR: do 160 tanich kroków małego modelu/parametru, bez treningu pełnego TFT. Limit całego zestawu Stage B: 10 minut CPU; przekroczenie raportuj, nie eskaluj do pełnego treningu. Testy Stage A mają odrębny czas wykonania.

Porównania matematyczne float64: proponowane atol=1e-10, rtol=1e-8; deterministyczny replay CPU ma dążyć do identycznych tensorów. Jeśli stos wymaga tolerancji, ustal ją jawnie przed oceną replay (referencja float32 atol=1e-6, rtol=1e-5), przedstaw maksymalną różnicę i przyczynę. Nie podwyższaj tolerancji, żeby ukryć błąd. CUDA tylko jeśli dostępna, z krótkim syntetycznym testem i osobnym wynikiem. Brak GPU nie jest PASS dla GPU.

## Checklista

Wszystkie pozycje początkowo NOT RUN. Każda wymaga wyniku, testu i dowodu plik:symbol/linie.

| ID | Zakres | Test i warunek akceptacji |
|---|---|---|
| B00 | Izolacja audytu | Osobny runner; strażnik rzucający błąd przy próbie load_test_data/evaluate/main/Optuna. Kontrola dodatnia dowodzi działania strażnika. Żadnych pacjenckich testowych predykcji ani metryk. |
| B01 | Kontrakt danych → model | Rzeczywisty batch: kształty, dtype, długości, kolejność horyzontów, maski i target scale. train/val zgodne z Stage A; krótszy decoder i padding nie zwiększają mianownika ani lossu. |
| B02 | Matematyka lossu | Niezależna referencja pinball, redukcji i wag; q poniżej/na/powyżej mediany; predykcja poniżej/równa/powyżej targetu; 69.9/70/70.1 mg/dL. Ustal z lokalnego PF, czy obowiązuje czynnik 2. Równe predykcje i target dają zero; ostrzeżenie sanity nie unieważnia tego przypadku. |
| B03 | Jednostki i redukcja | Prześledź transform_output i target przekazany do lossu: próg 70 stosowany w mg/dL. Test różniących się skal normalizatora, brak podwójnego inverse transform. Agregacja batch/epoka równa niezależnej referencji także przy nierównych batchach i długościach. Reset stanu metryki między etapami. |
| B04 | Kwantyle | Lista q poprawna, unikalna, rosnąca w (0,1); output size zgodny, mediana wybierana po wartości q. Round-trip checkpointu zachowuje q i kolejność osi. Fixture crossing jest wykrywana. Nie sortuj po cichu predykcji; brak ograniczenia monotoniczności raportuj jako ograniczenie, a nie dowód kalibracji. |
| B05 | Forward i leakage end-to-end | eval/no_grad: identyczny encoder i known decoder, perturbacja wyłącznie forbidden future unknowns → identyczne predykcje przy zamrożonym normalizatorze i wagach. Kontrola dodatnia zmienia rzeczywiście używane wejście. train forward daje skończony loss i poprawny kształt. |
| B06 | Backward i gradienty | Niezależny gradient pinball poza punktami nieróżniczkowalności; skończone gradienty, brak przypadkowego detach. Co najmniej odpowiednie aktywne parametry dostają niezerowy gradient i zmieniają się po kroku z dodatnim LR; nie wymagaj tego od każdego parametru w każdym batchu. zero_grad nie kumuluje poprzedniego kroku przypadkowo. |
| B07 | Optimizer, clipping, accumulation | AdamW: parametry dokładnie raz, trainable/frozen prawidłowo, lr/eps/decay zgodne z config. Zmierz normę przed i po clippingu oraz kolejność callbacków. Accumulation daje oczekiwaną liczbę aktualizacji i kroków schedulera; pierwszy LR=0 w warmupie nie jest dowodem zerowych gradientów. |
| B08 | Scheduler | Zapisz dokładny ciąg LR: start, warmup−1/warmup/warmup+1, restart cosinusa i resume. Zweryfikuj estimated_stepping_batches przy max_steps, niepełnych batchach, accumulation. Sprawdź krótki run, warmup dłuższy niż run i skrajny max_epochs. Monitor val_loss nie oznacza automatycznie sterowania cosine przez walidację. |
| B09 | Checkpoint | Save/load prawdziwego małego TFT: wagi, hparams, q, normalizator, schema, optimizer, scheduler, global_step, epoch, callback state. Identyczne predykcje w eval po round-trip. Checkpoint uszkodzony/niekompatybilny → jawna odmowa. Nazwa i mtime nie wystarczają jako dowód kompatybilności. |
| B10 | Resume | Porównaj nieprzerwany N=8 z K=4 + restart procesu + 4 przy tej samej konfiguracji całkowitego N. Porównaj parametry, optimizer moments, LR, kroki, kolejność batchy i callbacki. Pierwszy przebieg zatrzymaj bez zmiany zaplanowanego budżetu schedulera. Sprawdź RNG Python/NumPy/Torch i sampler; osobno dropout/shuffle. Zdefiniuj gwarantowaną granicę resume: epoka lub krok. Mid-epoch bez replay pozycji danych nie może być nazwany exact resume. |
| B11 | Wybór checkpointu i konfiguracja | Rozróżnij kontynuację ostatniego stanu, wybór best do inferencji oraz weights-only warm start jako nowy run. Konflikt CLI/checkpoint, dataset hash, schema, q, normalizator, horizon/context lub budget schedulera musi być jawny. Nie podejmuj automatycznej migracji protokołu. |
| B12 | Early stopping | Syntetyczna sekwencja val_loss: poprawa, plateau, pogorszenie, granica min_delta, NaN/Inf, brak monitora. Sprawdź liczenie patience według walidacji, sanity validation, restart i zachowanie best/wait_count; train/test nie steruje decyzją. |
| B13 | SWA | Ponieważ domyślna ścieżka używa SWA, sprawdź przejęcie schedulera, początek SWA, uśrednione wagi, early stopping, zapis best/last i resume. Krótki test callbacków może symulować późniejsze epoki. Wyłączenie SWA w smoke nie pokrywa domyślnej konfiguracji; nierozstrzygnięta zgodność blokuje PASS tej ścieżki. |
| B14 | Determinizm | Dwa nowe procesy, ten sam seed/config → porównanie inicjalizacji, batchy, lossów, LR i końcowych wag; inny seed jako kontrola. Sprawdź różnicę między seed_everything a ustawieniami deterministic/benchmark. Osobno CPU, CUDA i workery; nie deklaruj gwarancji między wersjami/platformami. |
| B15 | Stabilność | Zerowy błąd, stały sygnał, mały batch, krótkie sekwencje, duże skończone wartości, NaN/Inf w wejściu/target/loss/gradient. Niepoprawny stan zatrzymuje aktualizację i nie zapisuje checkpointu jako poprawnego. Skończony duży loss nie powinien być arbitralnie obcinany. Test FP32; AMP tylko jeśli wspierany, osobno i bez domyślnego włączania. |
| B16 | Provenance | Unikalny run_id; commit/dirty/diff hash; config hash; seed; wersje środowiska i urządzenia; protokół i digest danych/fixture; schema/normalizator/q; checkpoint hash, parent run/checkpoint, resume mode; polecenia, exit codes, wyniki i ograniczenia. Zapisz full_training=false, optuna=false, test_performance=false, synthetic_optimizer_steps z rzeczywistą liczbą. |
| B17 | Regresje i review | Odtwórz zestaw Stage A i nowe testy. Nie zamykaj znanych błędów expectedFailure ani skipem. Sprawdź diff, brak danych/artefaktów w Git, brak zmian protokołu, zgodność raportu i wyników. |

## Interpretacja wyników

Matematyczny oracle lossu: e=y−ŷq; pinball=max(qe,(q−1)e); współczynnik konwencji PF sprawdzany niezależnie; średnia po Q; w(y)=2.5 dla y<70, inaczej 1; następnie udokumentowana redukcja po ważnych pozycjach. Nie normalizuj automatycznie przez sumę wag klinicznych, jeśli kontrakt redukuje przez liczbę pozycji — to zmiana celu uczenia.

Próg i waga opisują obecną implementację, nie nowe zalecenie kliniczne. Work musi odnotować, że ważenie zależne od targetu zmienia cel estymacji względem nieważonego pinball; nazwy q nie gwarantują kwantyli rozkładu nieważonego ani pokrycia przedziałów. W Stage B sprawdzamy realizację tego celu; zasadność wag i kalibrację pozostawiamy do jawnej decyzji/Stage C.

PASS audytu treningu wymaga dowodów dla wszystkich obowiązkowych kontraktów w deklarowanej konfiguracji. Audit complete może mieć werdykt FAIL: wykrycie błędów jest poprawnym wynikiem pracy. Brak potwierdzenia exact resume oznacza brak tej gwarancji, nawet gdy checkpoint się ładuje. PASS CPU nie potwierdza CUDA/SWA/workerów poza przetestowanym zakresem. Nie wymagaj spadku lossu po kilku krokach jako dowodu poprawności całego TFT.

## Kolejność wykonania

B00 i B16 → B01–B06 → B07–B08 → B09–B13 → B14–B15 → B17 i review. Najpierw reproduktory; później osobna specyfikacja minimalnych napraw. Pytania metodologiczne nie blokują niezależnych testów. Nie uruchamiaj kolejnego etapu eksperymentalnego automatycznie.
