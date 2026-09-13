# Stage B — review i plan remediation

Data: 2026-09-13. Status: PROPOZYCJA DO ZATWIERDZENIA. Remediation nie została rozpoczęta. Werdykt Stage B pozostaje FAIL.

## 1. Źródła i granice review

Przeczytano w całości: AGENTS.md, docs/RESEARCH_WORKFLOW.md, docs/ROADMAP.md, docs/STAGE_B_AUDIT_PLAN.md, docs/CODEX_STAGE_B_TASK.md, docs/BASELINE_AUDIT_STAGE_B.md oraz configs/baseline_stage_b_audit.json. Sprawdzono też wskazane testy, runner, produkcyjne symbole oraz lokalne fragmenty PF i Lightning. Review nie uruchamia żadnego testu, fit, CUDA check ani Stage C.

HEAD podczas review: `896c9a5469897ec4938cd4b0cd8665e460780b3f`; drzewo na wejściu czyste. Raport audytuje `b83504ff61542fb2e4b2ae805bb79eea3ffa7660`. Diff do obecnego HEAD dodaje cztery pliki audytu, bez zmian produkcji ML. Źródłem wyników jest raport, nie ponowne wykonanie w tym review.

Stage A: 31 PASS. Stage B: końcowy bilans 18 PASS / 14 FAIL / 0 SKIP / 0 XFAIL. To pełne wykonanie 17 PASS /15 FAIL z zastąpieniem jednego wyniku przez focused review na istniejących artefaktach, a nie drugi pełny zielony/red run. Jedenaście findings nie odpowiada jeden do jednego czternastu nieudanym metodom. CUDA BLOCKED; AMP i workery >0 NOT RUN. Nie ma gwarancji exact resume.

Kanoniczne dane pozostają `ml/data/processed/baseline_v1_stage_a_20260913T115538143463Z/`, provenance commit `5ee8d930e80fd7d62d0248c52ed10aaaaaa7f8d0`, dirty=false; SHA-256 parquetu `d14fe31b1b81971639ca8d5ba17b4882fbd6711569e30785f0da7b91a0c031cf` potwierdzony w audycie. Nie zmieniamy okien, splitów, clinical weighting, ani danych. Dawny zapis „audyt NIE WYKONANY” w RESEARCH_WORKFLOW jest historycznym punktem startowym; późniejszy raport Stage B określa aktualny stan.

## 2. Priorytety

BLOCKER oznacza brak dopuszczenia Baseline v1.0 przy dostępności danej wadliwej ścieżki. HIGH wymaga naprawy przed certyfikacją dotkniętej funkcji; nie każda awaria resume unieważnia matematycznie trening od zera. Przyjęcie kontraktu resume powoduje, że jego niespełnienie blokuje wydanie z tą funkcją.

| Finding | Severity po review | Wpływ | Klasyfikacja | Kolejność |
|---|---|---|---|---|
| B15-F01 | BLOCKER dla Baseline v1.0 | correctness, numerical safety, scientific validity, checkpoint integrity | bug techniczny | R1 |
| B15-F02 | BLOCKER dla Baseline v1.0 | numerical safety, correctness, checkpoint integrity | bug techniczny | R1 |
| B11-F01 | BLOCKER dla Baseline v1.0 | checkpoint integrity, scientific validity, correctness, reproducibility | bug + kontrakt zgodności | R2 |
| B11-F02 | BLOCKER dla Baseline v1.0 w obecnej automatycznej ścieżce | checkpoint integrity, reproducibility, operational reliability | bug + rozdzielenie trybów | R2 |
| B10-F01 | HIGH | operational reliability, reproducibility, checkpoint integrity | bug integracji odczytu + polityka zaufania | R2 |
| B10-F04 | HIGH; blokuje deklarowaną kontynuację | correctness, reproducibility, operational reliability | bug/integracja pętli + granica wsparcia | R3 |
| B10-F02 | HIGH; blokuje deklarację exact | reproducibility, scientific validity | brak kontraktu RNG/danych | R3 |
| B13-F01 | HIGH; blokuje SWA ON z resume | reproducibility, checkpoint integrity, operational reliability | interakcja techniczna + decyzja SWA | D2, potem R3 lub osobne zlecenie |
| B03-F01 | MEDIUM | correctness, scientific validity | bug agregacji raportowanej metryki | R4 |
| B10-F03 | LOW | operational reliability, reproducibility diagnostyki | bug serializacji callbacku | R3 |
| B08-F01 | LOW | operational reliability | bug walidacji konfiguracji | R1, mała niezależna poprawka |

Brak dowodu, że B03-F01 zmienia obecny pełnohoryzontowy val_loss. Brak dowodu, że w audycie zapisano uszkodzony checkpoint po NaN; aktualizację zatrzymał strażnik audytu. Severity B15 wynika z braku ochrony produkcyjnej, nie z dopisanej historii awarii.

## 3. Kontrakty i decyzje

### D1 — granica resume: rekomendacja research leada

Docelowo wspieramy **kontynuację wyłącznie z zatwierdzonej granicy ukończonej epoki**, po wszystkich batchach, aktualizacjach, walidacji oraz aktualizacji callbacków/schedulera należnych tej epoce. Zapis zawiera completed_epoch, next_epoch, global_step i boundary_complete=true. Nie ma oczekujących skumulowanych gradientów. Crash w środku epoki cofa do ostatniej poprawnej granicy; wykonana po niej praca nie staje się częścią zaakceptowanej kontynuacji.

Nie wspieramy step-level ani mid-epoch resume w v1.0. Nie rekonstruujemy prywatnych liczników Lightning i nie zwiększamy max_epochs, by ukryć utracone updates. `on_train_epoch_end` ani `save_checkpoint` po przerwanym fit nie są same w sobie dowodem kompletności granicy. Najpierw mały reproducer wyznacza publiczny, rzeczywiście domknięty punkt na badanym stosie.

Oznaczenie `epoch-boundary exact resume` wolno nadać dopiero po odtworzeniu modelu, optimizer moments/param groups, schedulera i jego planu, RNG Python/NumPy/Torch CPU/CUDA, używanych generatorów, callbacków, stanu postępu i kolejności danych. Referencja certyfikacji: ten sam kod, dane, stos, urządzenie, FP32, jeden proces, num_workers=0; CPU/1 thread najpierw. GPU ma osobną bramkę. Zmiana urządzenia nie dziedziczy gwarancji.

Test podstawowy: 4 epoki ×2 updates =8 nieprzerwanie oraz 2 pełne epoki → zamknięcie procesu →2 dalsze epoki. Oba zachowują pierwotny plan 4 epok/8 updates. Najpierw sprawdź 4+4=8 i brak pustej/zdublowanej epoki; potem trace LR, sample IDs, RNG, callbacks i optimizer; dopiero na końcu wagi/predykcje. Oddziel dropout=0/shuffle=false od dropout>0/shuffle=true. Sprawdź też accumulation, brak checkpointu w środku accumulation oraz checkpoint po decyzji early stopping (nie kontynuuj zatrzymanego eksperymentu po cichu).

Stan dyskretny, RNG, porządek danych i liczniki mają być identyczne. W deterministycznej referencji CPU wymagamy także identyczności tensorów dla deklaracji bitwise exact. Dawne atol=1e-6/rtol=1e-5 wolno raportować jako zgodność numeryczną, ale nie zamieniać jej automatycznie w bitwise exact. Jeśli wszystkie stany i sekwencje są odtworzone, lecz zostają różnice numeryczne, użyj jawnego określenia „pełny stan odtworzony, zgodność numeryczna w podanej tolerancji”, bez etykiety exact.

W przypadku ponownego FAIL na granicy epoki: nie wymuszaj certyfikacji. Udokumentuj reproducer i pozostaw resume niedostępne w certyfikowanej ścieżce. Wariant „v1.0 tylko fresh runs, exact resume odroczone” wymaga osobnego zatwierdzenia użytkownika i jawnego zawężenia specyfikacji; nie jest domyślnym fallbackiem. Zmiana wersji Lightning/PyTorch wymaga osobnej zgody na środowisko.

### D2 — SWA

| Wariant | Konsekwencje naukowe i operacyjne |
|---|---|
| SWA OFF — rekomendowany dla v1.0 | Prostszy, jawny protokół AdamW + istniejący warmup/cosine. Brak przejęcia schedulera i transferu averaged weights na końcu. Zmienia algorytm treningu względem dawnej konfiguracji, więc nowa wersja training protocol; brak twierdzenia o lepszej jakości. B13-F01 pozostaje historycznym defektem poza wspieraną ścieżką. |
| SWA ON | Zachowuje uśrednianie, lecz wymaga naprawienia B10 i osobnego replay przed/po aktywacji SWA, SWALR, n_averaged, average model, ES i artefaktów. Trzeba wcześniej określić, czy inference wybiera zwykły best czy osobno walidowany averaged artifact. To dodatkowy zakres, bez dowodu korzyści w tym audycie. |

Rekomenduję OFF z powodów kontroli metody i złożoności, bez użycia test-performance. Nie wyłączamy SWA wyłącznie w testach, zostawiając domyślne ON. Po zatwierdzeniu OFF certyfikowana konfiguracja i manifest mówią swa=false; jawna próba ON jest odrzucana jako poza profilem v1.0 albo uruchamiana wyłącznie w wyraźnie niecertyfikowanym, osobnym protokole — w tym zleceniu preferowana odmowa.

Przy OFF zachowaj dotychczasowy wariant bez SWA: EarlyStopping patience=20, min_delta=1e-4, monitor val_loss. To jawna część decyzji D2 (ON miało patience=25). Nie dopasowuj cierpliwości na podstawie wyników. Przy ON early stop przed SWA jest dopuszczalną cechą algorytmu, nie bugiem samym w sobie; manifest musi wtedy mówić swa_applied=false. Nie przedłużaj runu tylko po to, aby SWA koniecznie wystąpiło.

### D3 — loss: implementacja i sens naukowy

A. Audyt potwierdza aktualny pointwise/batch objective: `2 * pinball`, średnia Q, waga 2.5 przy y<70 mg/dL, poza tym 1, mianownik liczby ważnych pozycji. Jednostki, padding i gradienty przeszły testy. Nie oznacza to poprawności agregacji epok (B03) ani bezpieczeństwa NaN/Inf (B15).

B. Ważenie zależne od targetu zmienia estymowany rozkład. Wyprowadzenie dla dodatniej wagi w(y): minimalizator E[w(Y)ρq(Y−a)|X=x] jest q-kwantylem rozkładu o CDF `F_w(a|x)=E[w(Y)1{Y≤a}|x]/E[w(Y)|x]`, a nie ogólnie F(a|x). Dodatni czynnik 2 nie zmienia minimizatora. To argument matematyczny, nie wynik kliniczny lub dowód empiryczny. W modelu o ograniczonej pojemności dochodzą dalsze kompromisy wspólnego dopasowania.

Rekomendacja: zachowaj obecny objective w remediation; nie zmieniaj progu, wagi ani mianownika gradientowego. Zasadność clinical weighting, porównanie z nieważonym objective i kalibrację odłóż do jawnej decyzji Stage C, przed twierdzeniami o nominalnym pokryciu. Stage B może uzyskać PASS mechaniki weighted objective bez zatwierdzenia naukowego przedziałów. Nie uruchamiaj obecnie Stage C i nie reklamuj wyjść q jako skalibrowanych przedziałów. Przed pełnym baseline training użytkownik musi świadomie zaakceptować weighted objective jako wariant badawczy albo zlecić odrębny przegląd celu — odroczenie nie jest zgodą na jego kliniczną trafność.

### D4 — checkpointy: role i walidacja

| Tryb | Źródło i semantyka | Stan |
|---|---|---|
| fresh | nowy run, brak automatycznego discovery | nowa inicjalizacja i nowy optimizer/scheduler/RNG/callbacki |
| resume-last | jawny run_id; najnowsza poprawna, kompletna granica epoki tego samego runu | pełny stan; nie wolno wybierać best zamiast last |
| inference-best | best według skończonego val_loss z manifestu/payload, pełna precyzja; jawny monitor/mode i rozstrzyganie remisu (wcześniejszy global_step) | dokładnie wagi ocenione tą metryką + q/schema/normalizator; bez kontynuacji uczenia |
| weights-only new run | jawny plik rodzica i nowy run_id | wyłącznie zgodne wagi; nowy optimizer, scheduler, RNG, callbacki i liczniki, parent hash i tryb zapisane |

`weights-only` jako tryb eksperymentu nie oznacza tego samego co flaga bezpieczeństwa `torch.load(weights_only=...)`. Nigdy nie importuj starego optimizera/schedulera w trybie nowego eksperymentu.

Obowiązkowy fingerprint: dataset parquet SHA-256 + protocol danych/splitów; ordered schema (nazwy, dtype, role known/unknown/static, jednostki, kolejność, encoding kategorii i maski); ordered quantiles; stan dopasowanego normalizatora (klasa, transformacja, grupy, center/scale, semantyczny digest); context/horizon i zasady zmiennych długości; efektywny model config; training protocol version z loss/weighting/SWA. Resume dodatkowo porównuje optimizer/scheduler config i pierwotny total budget, batch/drop_last/accumulation, sampler, precision, callback config, kod i wspierany stos/urządzenie. Serializacja fingerprintu ma być deterministyczna, nie `repr` przypadkowych obiektów.

Wszystkie trzy ścieżki checkpointowe walidują te składowe względem zadeklarowanego kontraktu artefaktu, nie tylko kształtu warstwy. Dla inference dataset hash identyfikuje dane treningowe artefaktu; nie wymagamy, aby nowe dane wejściowe inference miały hash treningowego parquetu. Dla resume bieżący dataset treningowy musi się zgadzać. W tym zakresie weights-only także wymaga zgodności danych/semantyki; transfer do innych danych/normalizatora wymaga osobnego eksperymentu i decyzji, bez silent override.

Minimalna polityka zaufania: pełny odczyt dopuszczalny jedynie dla własnego lokalnego artefaktu z zaufanego rejestru runu, zgodnym hashem i protokołem. Hash i sidecar dostarczone razem przez obcą stronę nie dowodzą zaufania. Weryfikuj pochodzenie i hash przed deserializacją, następnie payload/semantykę; odrzuć brak manifestu, nieznany format, corrupt bytes i konflikt sidecar/payload. Nie skanuj dowolnego folderu i nie odczytuj obcych pickle. Bez globalnego TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD ani automatycznego fallbacku po UnpicklingError. Spójna jawna opcja pełnego odczytu własnych artefaktów może wykorzystać publiczne API obecnego stosu. Ograniczenie dotyczy zarówno build_model, jak i fit, nie tylko drugiego odczytu. [PyTorch 2.11: serialization](https://docs.pytorch.org/docs/2.11/notes/serialization.html).

Preferowany minimalny format v1: własny pełny checkpoint Lightning z wersjonowanym manifestem i jawnie kontrolowanym odczytem; eksport samych tensorów do weights-only osobno. Przepisanie całego normalizatora do nowego formatu nie jest konieczne do tej naprawy. Legacy bez wymaganych metadanych nie otrzymuje automatycznej migracji.

Zapis: niezmienny plik generacyjny + hash i manifest, potem atomowa publikacja wskaźnika last/best. Przerwany zapis nie zastępuje ostatniego poprawnego wskaźnika. Invalid state nie może awansować do valid checkpointu. Zachowaj historyczny poprawny checkpoint; ewentualny dump diagnostyczny ma status invalid, jest poza selekcją i nie służy do resume. Zmiana katalogu callbacku ModelCheckpoint nie może po cichu wyzerować historii best/top-k — minimalnie resume w tym samym katalogu runu. Przenoszenie katalogów poza zakresem do osobnej migracji.

## 4. Review każdego findingu

### B15-F01 — BLOCKER dla Baseline v1.0

**Co jest zepsute:** PF MultiHorizonMetric._update_losses_and_lengths zastępuje nieskończoną sumę lossu przez 1e9, a potem redukuje ją długością. Pozornie skończona metryka może przejść check_finite ES. W negatywnym fit pojawił się wtórny błąd grad_fn, nie prawidłowa obsługa źródła błędu.

**Wpływ:** correctness, numerical safety, scientific validity, checkpoint integrity. **Typ:** jednoznaczny bug techniczny; nie wymaga zmiany celu naukowego. Dokumentacja ostrzeżenia nie zastępuje ochrony.

**Minimalna remediation:** kontrola ważnych wejść/targetów, transformowanych wyjść i elementów lossu przed odziedziczonym fallbackiem, oraz redukcji i akumulacji stanu metryki. Sama kontrola elementów nie wystarcza: suma skończonych elementów także może przepełnić dtype. Sprawdź sumę licznika, mianownik i końcową wartość, zanim fallback PF ukryje Inf. Wprowadź lokalne rozszerzenie/adapter projektu z testem kontraktu biblioteki; nie edytuj site-packages. Zachowaj poprawny gradient i wzór finite lossu.

**Regresja:** NaN/+Inf/−Inf osobno w targetach, wejściach, predykcjach, lossie; przepełnienie sumy przy finite elementach i przepełnienie akumulatora między batchami; niepoprawny normalizer. Production path ma rzucić czytelny błąd przed AdamW i przed aktualizacją monitora/best/last. Obejmuje train i validation oraz wywołanie lossu bez Trainer. Duża skończona strata i zero loss pozostają poprawne. Maskuj wyłącznie udokumentowany padding; nie myl padding NaN w reduction=none z rzeczywistym błędem na ważnej pozycji i nie stosuj nan_to_num.

**DoD:** żaden badany nonfinite nie staje się 1e9 lub inną zastępczą metryką; błąd wskazuje etap bez danych pacjenta; zero updates po wykryciu; brak nowego valid artefaktu, poprzednie valid pozostają. B02/B03 units i finite gradient oracle nadal PASS. Pierwszą odmowę wywołuje produkcja, nie strażnik audytu.

### B15-F02 — BLOCKER dla Baseline v1.0

**Co jest zepsute:** porównanie NaN normy z progiem nie wykrywa błędu. Callback jedynie ostrzega o dużej normie; NaN dociera do granicy optimizera i zatrzymuje go dopiero containment audytu.

**Wpływ:** numerical safety, correctness, checkpoint integrity. **Typ:** bug techniczny, nie zmiana progu klinicznego.

**Minimalna remediation:** niezależny od warmupu i progów alertów check finite gradientów po backward i przed update, również po accumulation/clipping. Clipping nie jest sanitizacją. Dla finite elementów obliczaj normę stabilnie, aby samo przepełnienie obliczenia normy nie zerowało gradientów; jeśli nie można bezpiecznie wykonać kroku, jawny błąd. Po update sprawdź skończoność wag i stanów optimizera, aby nie publikować niepoprawnego checkpointu nawet przy overflow powstałym dopiero w AdamW. Nie trzeba kopiować całego optimizera przed każdym krokiem; błąd po kroku unieważnia bieżący stan i zatrzymuje proces.

**Regresja:** injection NaN/+Inf/−Inf do aktywnego gradientu przed/po warmupie, w jednym mikro-batchu accumulation i na granicy clippingu. Spy na rzeczywistym AdamW potwierdza zero wywołań dla wcześniej niepoprawnego gradientu. Osobny syntetyczny overflow po kroku → abort przed zapisem i kolejnym krokiem. Finite wysokie gradienty przechodzą poprawny clipping; grad=None dla zamrożonych parametrów jest legalne. Nie wyłączaj containment, ale jego przechwycenie musi być FAIL testu produkcyjnej ochrony.

**DoD:** nie wykonuje się optimizer step z nonfinite gradientem; żaden nonfinite stan po update nie jest oznaczany valid; wszystkie ścieżki zapisu, także exception/terminal, przestrzegają bramki. Ochrona nie zależy od historii GradientNormLogger.

### B11-F01 — BLOCKER dla Baseline v1.0

**Co jest zepsute:** build_model pomija porównanie przekazanego datasetu z checkpointem; złożona zmiana context/horizon/normalizatora jest akceptowana. Dla hash/q/schema brak kontroli wynika też z inspekcji, nie z osobnych runtime reproduktorów.

**Wpływ:** checkpoint integrity, scientific validity, correctness, reproducibility. **Typ:** bug techniczny; dopuszczalne migracje są osobną decyzją, obecnie żadnych.

**Minimalna remediation:** wspólna walidacja D4, używana przed inferencją, resume i weights-only; konflikt → jawna odmowa, bez ostrzeżenia „CLI ignored” jako zamiennika walidacji.

**Regresja:** osobna perturbacja każdej składowej D4, w tym ten sam kształt przy innej kolejności cech/q, inny normalizer z tą samą klasą, inny dataset hash, model config/protocol oraz total scheduler budget. Brak pola, nieznana wersja, sidecar/payload mismatch → odmowa przed użyciem modelu. Zgodny round-trip i inference na nowych wejściach według zgodnego kontraktu → PASS.

**DoD:** wszystkie składowe mają niezależne negatywne testy i pozytywną kontrolę; walidacja nie refituje normalizatora i nie czyta wyników test setu; brak silent migration.

### B11-F02 — BLOCKER dla obecnej domyślnej ścieżki Baseline v1.0

**Co jest zepsute:** tekstowy corrupt plik wygrywa selekcję po nazwie z niskim val_loss; last jest ignorowany, fallback opiera się na mtime. main używa best jako continuation, więc może cofać historię do wcześniejszej epoki.

**Wpływ:** checkpoint integrity, reproducibility, operational reliability. **Typ:** bug techniczny i jawny kontrakt ról D4; nie problem skuteczności predykcyjnej.

**Minimalna remediation:** osobne wejścia fresh/resume-last/inference-best/weights-only new run, brak automatycznego wznowienia przy zwykłym fresh. Wybór przez zaufany indeks i pełną wartość monitora, walidacja pliku i roli; przy niedostępnym last błąd, nie fallback do best. W certyfikowanym train entrypoint nie wykonuj automatycznej test-evaluation; Stage C pozostaje osobnym jawnym wejściem, nie częścią remediation.

**Regresja:** starszy best vs nowszy last; corrupt niski filename, zmieniony mtime, remis/zaokrąglenie metryki, obcy run, invalid flag, przerwany zapis. Fresh nie odkrywa plików. Weights-only ma nowy run_id i zerowe nowe stany. Resume z inference-only/weights-only jest odrzucane.

**DoD:** selekcja niezależna od nazwy/mtime; last i best wskazują właściwe role; brak niejawnego rollbacku i wejścia w test-performance; zapis z D4 odporny na przerwanie publikacji.

### B10-F01 — HIGH

**Co jest zepsute:** własny checkpoint przechodzi PF model load, lecz fit-resume kończy się UnpicklingError dla pandas DataFrame normalizatora; zero dalszych updates.

**Wpływ:** operational reliability, reproducibility, checkpoint integrity. **Typ:** bug kompatybilności odczytu; polityka zaufania D4 wymaga jawnego przyjęcia.

**Minimalna remediation:** jeden zweryfikowany kontrakt odczytu dla build_model i fit. Po sprawdzeniu własnego pochodzenia/hash/formatu dopuszczalny jawny pełny odczyt przez publiczną opcję stosu. Nie retry automatycznie dowolnego pliku przez weights_only=False.

**Regresja:** własny syntetyczny checkpoint w nowym procesie odtwarza payload modelu i pełny stan fit; obcy/brak manifestu/corrupt/wadliwy hash odrzucany zanim wykonany zostanie pełny odczyt. B10-F04 nadal osobno sprawdza dalszy budżet.

**DoD:** brak UnpicklingError dla poprawnego własnego formatu, spójny odczyt, brak globalnego osłabienia deserializacji. Naprawa load nie zamyka automatycznie findings RNG/progress.

### B10-F04 — HIGH, bramka dla resume

**Co jest zepsute:** po K=4 z budżetu 8 resumed fit wykonuje tylko 2 updates, kończy na global_step=6 i zawiera epokę bez batchy. Dzieje się to również bez dropout/shuffle i dla testowanych last/terminal. Sam zapis stanu pętli nie daje prawidłowej kontynuacji.

**Wpływ:** correctness, reproducibility, operational reliability. **Typ:** potwierdzony bug/integracja pętli; dokładny upstream root cause pozostaje do izolacji. Nie stwierdzamy ogólnej niemożliwości resume w całym Lightning.

**Minimalna remediation:** mały publiczny reproducer na jednym parametrze i kontrolowanym dataloaderze, potem test rzeczywistego TFT; wyznaczyć boundary D1. Utrzymać ten sam katalog runu dla callbacków, ale osobne logi segmentów/procesów. Wyeliminować zakończenie/zapis w niespójnym punkcie, jeśli publiczne API to umożliwia. Nie patchować prywatnego fit_loop ani powiększać budżetu.

**Regresja:** 8 vs4+4, właściwy next_epoch, brak pustych/zdublowanych epok i batchy; checkpoint podczas normalnego końca epoki, kontrolowanego zatrzymania, przed ES i po ES. Porównanie ciągłych i resumed stanów tylko po przejściu testu równego budżetu.

**DoD:** publiczna kontynuacja na zatwierdzonym punkcie wykonuje dokładnie pozostałe updates i zachowuje plan LR; dokumentuje wersję/granicę. Jeśli nie osiągnięto wyniku: finding OPEN/BLOCKED, resume zabronione, osobna propozycja środowiska lub zawężenia v1.0; żadnego pozornego PASS.

### B10-F02 — HIGH, bramka dla exact

**Co jest zepsute:** nowe procesy seedują od początku, zamiast odtwarzać Python/NumPy/Torch RNG i porządek danych. Pierwszy resumed batch oraz stany RNG różnią się przy global_step=4. Różnica końcowych wag jest współzależna z F04 i nie izoluje wpływu RNG.

**Wpływ:** reproducibility, scientific validity porównań kontynuacji. **Typ:** techniczny brak pełnego stanu; mid-epoch może zostać udokumentowane jako unsupported według D1.

**Minimalna remediation:** snapshot/restore używanych RNG, generatorów dataloadera/samplera i stanu epoki. Użyj jawnego generatora danych, oddzielnego od losowości modelu. Odzyskuj stany po inicjalizacjach zużywających RNG, ale przed konsumpcją danych/forward kolejnej epoki. Walidacja/sanity/logger nie może wprowadzać dodatkowych losowań nieobecnych w ciągłym przebiegu. Nie reseeduj każdej epoki tylko w resumed wariancie.

**Regresja:** po F04, nowy proces i pełna druga połowa trace: sample IDs, RNG przed forward, dropout, optimizer moments, LR, model i callback state. Kontrola seed43 potwierdza czułość. Odrzuć nieobsługiwane worker/device/precision zamiast dziedziczyć certyfikację CPU.

**DoD:** stan i porządek odtworzone według D1, bez ukrytych probe draws asymetrycznych między wariantami; wszystkie 4 dalsze updates porównane. Brak obietnicy exact dla mid-epoch/innego urządzenia lub stosu.

### B10-F03 — LOW

**Co jest zepsute:** GradientNormLogger nie serializuje _high_grad_consecutive; 12 po load staje się 0. Zmienia historię alertów, nie dowodzi zmiany gradientów.

**Wpływ:** operational reliability, reproducibility diagnostyki. **Typ:** jednoznaczny bug techniczny; nie wymaga nowej metody.

**Minimalna remediation:** state_dict/load_state_dict, stabilny state_key i walidacja konfiguracji callbacku. Zapisz licznik i istotne parametry progów/warmupu. Bez zmiany ich wartości. Bezpieczeństwo B15 działa niezależnie.

**Regresja:** 12→12, kontynuacja przy patience−1 emituje alert w tym samym kroku co run ciągły; reset przy normalnej normie identyczny; konflikt config jest jawny.

**DoD:** serializacja i produkcyjny round-trip callbacku PASS; brak podwójnych alertów lub resetowania historii. Jest wymagane do pełnego kontraktu callback state, mimo niskiej severity samodzielnego objawu.

### B13-F01 — HIGH dla SWA ON

**Co jest zepsute:** wznowienie po aktywacji SWA wykonuje 0 zamiast 2 updates, global_step=6, scheduler=None. SWALR ma zostać odtworzony przy następnej epoce, do której wadliwa pętla nie dochodzi. Średnia SWA została policzona poprawnie; last≠averaged terminal i best≠terminal nie są bugiem.

**Wpływ:** reproducibility, checkpoint integrity, operational reliability. **Typ:** interakcja techniczna z F04 oraz decyzja metodologiczna D2. Nie da się rozwiązać jej samym ujednoliceniem wag best/last.

**Minimalna remediation:** po D2=OFF usuń SWA z certyfikowanego profilu i odmów ON w tym profilu; zachowaj finding jako DEFERRED/OUT OF SCOPE, nie FIXED. Po D2=ON wymagane osobne zlecenie implementacyjne po F04: pełny stan SWALR/averaging, jawne role i walidacja averaged artefaktu. Nie utrzymuj ON bez pokrycia.

**Regresja:** OFF: callback SWA nie istnieje, cały LR należy do pierwotnego schedulera, manifest swa=false, ON odrzucone, resume z SWA checkpointu do OFF odrzucone. ON: checkpoints przed aktywacją, po pierwszej średniej i po kolejnej; poprawne pozostałe updates, n_averaged/average_model/scheduler/ES i zgodne predykcje finalnego ocenionego artefaktu. Early stop przed SWA raportuje brak zastosowania.

**DoD:** dla OFF zaakceptowana zmiana protokołu i wyłączona wadliwa ścieżka, bez przepisywania historycznych FAIL; dla ON pełne testy integracji PASS. Samo wyłączenie w harnessie nie spełnia żadnego wariantu.

### B03-F01 — MEDIUM

**Co jest zepsute:** średnie batchowe [10,30] ważone liczbą sekwencji [2,1] dają 16.666666…, a prawidłowa agregacja 24 i 3 ważnych pozycji daje 330/27=12.222222…. Gradient batchowy i pointwise wzór są poprawne.

**Wpływ:** correctness raportowania, scientific validity porównywania train_loss. **Typ:** bug techniczny wobec już przyjętego kontraktu valid positions. Nie ma dowodu błędu obecnego pełnohoryzontowego val_loss.

**Minimalna remediation:** jawny licznik sumy lossów i mianownik liczby ważnych pozycji dla metryki epokowej, z resetem dla train/val/sanity. Można użyć ważenia batch loss przez sum(decoder_lengths), jeśli potwierdzone przez rzeczywistą agregację Lightning. Zachowaj returned loss do backward, gradient accumulation i clinical weights. Nie reinterpretuj tego jako zmiany optymalizatora w celu równego ważenia całego datasetu.

**Regresja:** niezależny oracle przy nierównych batchach/długościach; rzeczywisty log epoch w Trainer, nie tylko przechwycenie self.log. Pełny horizon + ostatni niepełny batch, duży padding, granica epoki, sanity→train/val. Porównaj gradienty i LR przed/po dla tych samych batchy.

**DoD:** epokowe sum/count równe referencji; pełnohoryzontowy val bez zmiany wyniku; bez podwójnego logowania starej i nowej metryki pod tym samym monitorem; clinical weighting i batch gradients niezmienione.

### B08-F01 — LOW

**Co jest zepsute:** epochs=0 dochodzi do dzielenia przez zero w configure_optimizers zamiast błędu konfiguracji. Audit nie dowodzi, że epochs=−1 jest błędem Lightning: finite max_steps może być dozwolone.

**Wpływ:** operational reliability. **Typ:** bug walidacji i zakres wspieranych konfiguracji, nie powód zmiany warmupu.

**Minimalna remediation:** jawna odmowa zerowego budżetu przed konstrukcją optimizera/schedulera. Dla v1.0 rekomendowany ograniczony profil finite positive max_epochs i skończony plan updates, zgodny z D1; epochs=−1 jawnie unsupported w profilu, bez nazywania go niepoprawnym API Lightning. Jeśli użytkownik chce tryb wyłącznie max_steps, wymaga on odrębnej definicji steps_per_epoch i testu LR; nie wybieraj arbitralnej nowej formuły.

**Regresja:** 0 → czytelny ValueError; wartości ujemne/nieskończone lub niespójny plan → jawna odmowa profilu; poprawny dodatni budżet zachowuje istniejący trace warmup/cosine, w tym krótki run i accumulation. Nie zmieniaj warmup min50 tylko dlatego, że smoke ma 8 updates.

**DoD:** brak ZeroDivisionError dla konfiguracji użytkownika; wspierane i niespierane tryby opisane i testowane; matematyka schedulera dla dotychczasowego dodatniego budżetu bez zmian.

## 5. Osobny CUDA environment check — tylko plan

RTX 3070 Ti jest na maszynie użytkownika. CUDA=false/NVML warning w procesie audytu nie dowodzi problemu TFT. Sprawdzenie wykonać przed remediation w terminalu VS Code użytkownika na hoście i w tej samej `.venv`; zapisać, czy proces jest sandbox/container/remote. Jeśli kontekst audytu ma inne uprawnienia urządzeń, porównać rezultaty. Nie instalować sterowników, CUDA ani PyTorch, nie restartować usług i nie zmieniać zmiennych środowiska w ramach checku.

Krótki zestaw do wykonania w Bash, z katalogu repo; nie wykonano go w tym review:

```bash
nvidia-smi
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import json, sys, torch
report = {
    'python_executable': sys.executable,
    'pytorch': torch.__version__,
    'pytorch_cuda_build': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'device_count': torch.cuda.device_count(),
    'cudnn_version': torch.backends.cudnn.version(),
}
print(json.dumps(report, indent=2))
if not report['cuda_available']:
    print('CUDA_ENVIRONMENT_CHECK: BLOCKED; no tensor operation performed')
    raise SystemExit(2)
try:
    print('device_name:', torch.cuda.get_device_name(0))
    x = torch.arange(16, device='cuda', dtype=torch.float32)
    y = x.square().sum()
    torch.cuda.synchronize()
    value = y.item()
    assert torch.isfinite(y).item() and value == 1240.0
    print('tensor_operation:', value, 'PASS')
except Exception as exc:
    print('CUDA_ENVIRONMENT_CHECK: FAIL', type(exc).__name__, str(exc))
    raise SystemExit(1)
PY
```

Zachowaj osobne exit codes nvidia-smi i Pythona, stdout/stderr. „CUDA Version” z nvidia-smi opisuje zdolność sterownika, nie wersję builda torch ani dowód zainstalowanego toolkit. `torch.version.cuda` określa build pakietu. Sukces prostego tensora oznacza dostęp urządzenia, nie certyfikację TFT, deterministyczności ani pamięci pełnego treningu.

nvidia-smi FAIL → zbadaj dostęp urządzenia/sterownik/kontekst. nvidia-smi PASS i torch CUDA=None → sprawdź wybrany interpreter/build. Build CUDA obecny, available=false → diagnostyka inicjalizacji/runtime/dostępu; nie zakładaj konkretnej przyczyny. Niepowodzenie nie blokuje niezależnych napraw CPU, ale blokuje zatwierdzenie treningu CUDA. Po dostępności GPU przyszła remediation wymaga krótkiego syntetycznego FP32 forward/backward/finite check oraz osobnego replay na GPU; bez automatycznej gwarancji na podstawie CPU.

## 6. Zatwierdzenia i kolejność

Niniejsze zlecenie autoryzuje review i dwa dokumenty. Nie autoryzuje implementacji. Task Codexa zawiera bramkę aktywacji, aby nie mylić rekomendacji z decyzją użytkownika.

| Decyzja | Rekomendacja | Status |
|---|---|---|
| A0 | Uruchomić wyłącznie zakres techniczny T0: CUDA check, B15, agregacja B03, walidacja epochs=0 B08, serializacja B10-F03 | PENDING — potrzebne osobne polecenie wykonania |
| D1 | Granica ukończonej epoki według sekcji 3; brak mid-epoch; finite positive epoch budget, FP32/0 workers jako profil początkowy; exact dopiero po dowodzie | PENDING zatwierdzenia profilu |
| D2 | SWA OFF, no-SWA ES patience20/min_delta1e-4, nowa wersja training protocol | PENDING decyzji użytkownika |
| D3 | Zachować weighted loss w remediation; zasadność celu/kalibrację odłożyć do Stage C, bez zatwierdzania pokrycia | PENDING przyjęcia odroczenia; zakaz zmiany wag już obowiązuje |
| D4 | Trzy role checkpointów + fresh, fingerprints i własne zaufane pełne checkpointy; żadnych automatycznych migracji | PENDING przyjęcia kontraktu |
| E1 | Jakakolwiek instalacja/zmiana sterownika lub wersji stosu | NIEZATWIERDZONE, osobna zgoda i osobny plan |

Rekomendowana kolejność: CUDA check (diagnostycznie) → R1 finite safety i walidacja wejścia → R2 manifest/role/zgodność/odczyt → R3 izolacja progress, potem RNG/callback replay na przyjętym profilu SWA → R4 agregacja lossu i pełne regresje → review gotowości. B03 i B10-F03 można wykonać wcześniej niezależnie. D1/D2/D4 ustalić przed zależnymi zmianami. T0 może być osobno zatwierdzone bez rozstrzygania SWA i resume.

## 7. Bramki zakończenia i odroczenia

Raport remediation ma podawać osobno FIXED, OPEN, BLOCKED i DEFERRED/OUT OF SCOPE; ostatnie nie są PASS naprawy. Historyczny audyt i jego 14 FAIL pozostają nienaruszone. Przy zaakceptowaniu OFF oraz braku mid-epoch wolno wersjonować nowy zestaw acceptance z mapowaniem starych testów, ale nie usuwać dowodów, przemianowywać ich na PASS ani używać xfail jako zamknięcia. Nowe testy potwierdzają również odmowę funkcji poza profilem.

Bramka v1.0: wszystkie blockery zamknięte w dostępnych ścieżkach, przyjęte D1–D4, obowiązkowe kontrakty wybranego profilu PASS, Stage A bez regresji, jawny status urządzenia. Nie wystarcza pojedynczy zielony test wczytania. CPU PASS nie zatwierdza GPU. Pełny trening nadal wymaga osobnego zlecenia; ten dokument nie jest uruchomieniem Stage C.

Odroczone: mid-epoch/step resume; migracje checkpointów i katalogów runów; AMP, wielu workerów i inne platformy do osobnej certyfikacji; SWA ON jeśli D2=OFF; clinical weighting jako wybór naukowy, kalibracja, coverage, crossing policy i skuteczność do Stage C; unseen-patient i pozostałe ograniczenia Stage A do właściwych etapów. Wyłączenie uszkodzonego resume w całym v1.0 jest jedynie wariantem awaryjnym do odrębnego zatwierdzenia.

Nie gwarantujemy odtwarzalności między wydaniami PyTorch, platformami ani CPU/GPU. Używane generatory danych wymagają własnego stanu, poza globalnym seedem. [PyTorch 2.11: reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
