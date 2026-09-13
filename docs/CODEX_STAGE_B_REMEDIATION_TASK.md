# Codex — Stage B remediation, zakres kontrolowany

Status: GOTOWA SPECYFIKACJA, NIEAKTYWOWANA. Dokument powstał w zleceniu research review, które zabraniało implementacji. Samo znalezienie tego pliku w repo lub przeczytanie go nie jest poleceniem wykonania.

## 1. Bramka aktywacji

Zanim edytujesz kod, znajdź w aktualnej rozmowie jednoznaczne polecenie użytkownika uruchamiające remediation i określające zakres. Nie proś ponownie, jeśli zostało już wydane. Zapisz w nowym raporcie: zatwierdzony zakres, decyzje D1–D4, źródło decyzji i wyłączenia. Gdy masz zgodę wyłącznie na T0, wykonuj tylko T0. Nie uznawaj zaleceń research leada za zgodę użytkownika na zmianę SWA/metody.

Dopuszczalne pakiety do zatwierdzenia:

- **T0 — naprawy techniczne bez wyboru nowej metody:** opisany CUDA environment check bez zmian środowiska; B15-F01/F02; B03-F01 (wyłącznie metryka epokowa); B08-F01 (epochs=0); B10-F03. Wszystkie stałe clinical weighting, SWA i plan LR pozostają niezmienione.
- **T1 — checkpointy:** B11-F01/F02 i B10-F01 po przyjęciu D4; zależy od bramki finite z T0.
- **T2 — resume v1.0:** B10-F04/F02 i pełny replay callbacków po przyjęciu D1, D2, D4 i po T1. Wariant rekomendowany tego tasku to D2=SWA OFF, wraz z testem odmowy ON w profilu v1.0. Nie obejmuje naprawy SWA ON.
- **D3:** brak zmiany clinical weighting obowiązuje w każdym pakiecie. Odroczenie naukowej oceny do Stage C odnotuj, jeśli przyjęte. Nie uruchamiaj Stage C niezależnie od decyzji.

Jeżeli użytkownik wybierze SWA ON, nie wykonuj części OFF: przygotuj do review doprecyzowany task dla SWA ON według planu, wykonując tylko niezależne zatwierdzone T0/T1. Jeśli zabraknie decyzji zależnej, raportuj dokładny blok i kontynuuj niezależny zatwierdzony zakres. Nie aktywuj T1/T2 automatycznie po zakończeniu T0.

## 2. Wejście i niezmienniki

Repo: `/home/radian/NeuroMetabolic Dashboard`.

Przeczytaj w całości AGENTS.md, instrukcje podkatalogów, docs/RESEARCH_WORKFLOW.md, docs/ROADMAP.md, docs/STAGE_B_AUDIT_PLAN.md, docs/CODEX_STAGE_B_TASK.md, docs/BASELINE_AUDIT_STAGE_B.md, configs/baseline_stage_b_audit.json i docs/STAGE_B_REMEDIATION_PLAN.md. Plan remediation definiuje D1–D4, severity i szczegółowe regresje. W razie sprzeczności zakresu zastosuj najnowszą jawną decyzję użytkownika, nie własne założenie.

Sprawdź HEAD/dirty state i nie usuwaj zmian użytkownika. Review dotyczyło HEAD `896c9a5469897ec4938cd4b0cd8665e460780b3f`; audyt raportował `b83504ff61542fb2e4b2ae805bb79eea3ffa7660` i niezmienioną produkcję. Bieżący kod może być nowszy — powiąż dowody z aktualnym commitem. Pracuj w gałęzi badawczej, nie main. Nie commituj/pushuj bez polecenia.

Stage A ma 31 PASS. Stage B ma bilans 18 PASS/14 FAIL po focused review jednego testu; nie przedstawiaj tego jako nowego wykonania. Kanoniczny dataset: `ml/data/processed/baseline_v1_stage_a_20260913T115538143463Z/`, provenance commit `5ee8d930e80fd7d62d0248c52ed10aaaaaa7f8d0`, dirty=false, SHA-256 `d14fe31b1b81971639ca8d5ba17b4882fbd6711569e30785f0da7b91a0c031cf`.

Nie regeneruj danych, nie zmieniaj splitów, polityki okien, context/horizon naukowego protokołu, normalizacji, wag lossu, progu 70, wagi 2.5 ani sposobu użycia test setu. Nie refituj normalizatora przy load. Nie oceniaj jakości predykcji. Syntetyczne fixture i lokalne metadane/hashe wystarczają do testów.

## 3. Przed naprawami: CUDA environment check

Wykonaj zestaw z sekcji 5 planu w tym samym interpreterze `.venv`, najlepiej na hoście w terminalu VS Code użytkownika. Zapisz nvidia-smi, torch.cuda.is_available(), torch.__version__, torch.version.cuda, device name, device count, prostą operację tensorową i synchronizację, osobne exit codes/stderr. Ustal kontekst procesu: host/sandbox/container/remote. Jeżeli nie masz dostępu do hosta, podaj gotowe polecenia użytkownikowi i oznacz check BLOCKED; nie przedstawiaj wyniku sandboxa jako awarii GPU.

Limit: około 30 s na check, timeout jawny. Nie instaluj, nie aktualizuj pakietów, nie zmieniaj sterowników/usług/uprawnień urządzeń ani ustawień środowiska. CUDA niedostępna nie wstrzymuje niezależnej remediation CPU. Przed późniejszym treningiem GPU wymagany jest PASS środowiska i krótki test rzeczywistego modelu na GPU, nie tylko tensora.

## 4. T0 — techniczne naprawy

### T0.1 — B15-F01/F02: finite safety, priorytet blocker

- Zapobiegaj fallbackowi PF zmieniającemu nonfinite loss w 1e9. Sprawdzaj ważne targety/wyjścia, per-position loss, sumę redukcji i akumulator metryki. Finite elementy nie gwarantują finite sumy. Zachowaj padding contract i finite wzór/gradient.
- Odrzucaj nonfinite gradient niezależnie od warmupu i progu ostrzeżeń, przed rzeczywistym optimizer.step oraz przy accumulation/clipping. Kontroluj też wagę/optimizer state po update i przed publikacją checkpointu. Nie kontynuuj po invalid state.
- Nie poprawiaj problemu przez nan_to_num, clamp lossu, pomijanie batcha, warning-only lub zwiększenie epsilon. Nie edytuj .venv/site-packages; użyj najmniejszego lokalnego rozszerzenia sprawdzonego z bieżącym PF.
- Testy injection NaN/+Inf/−Inf, overflow redukcji przy finite elementach, overflow po update, train/val/direct loss. Spy AdamW i monitor zapisu rozstrzygają, czy ochrona produkcyjna zadziałała pierwsza. Audit containment pozostaje ostatnią ochroną; jego przechwycenie oznacza FAIL produkcji.
- Nie publikuj best/last/terminal z invalid stanu. Nie nadpisuj poprzedniego valid. Finite wysoki gradient ma prawidłowy clipping; finite wysoki loss pozostaje prawdziwą wartością; zero loss jest legalny.

### T0.2 — B08-F01: zerowy budżet

Waliduj epochs=0 przed konstrukcją schedulera; czytelny ValueError zamiast ZeroDivisionError. Zachowaj trace dotychczasowych dodatnich budżetów, warmup min50, cosine i accumulation. W samym T0 nie zmieniaj zachowania epochs=−1: szersze ograniczenie profilu należy do zatwierdzonego D1/T2. Nie traktuj −1 jako ogólnie wadliwego API Lightning.

### T0.3 — B10-F03: stan GradientNormLogger

Dodaj serializację licznika i istotnej konfiguracji, stabilny state_key i kontrolę konfliktu konfiguracji. Testuj 12→12, granicę patience i reset, także poprzez prawdziwy checkpoint. Nie zmieniaj progów i nie uzależniaj finite guard od tego loggera.

### T0.4 — B03-F01: agregacja lossu epoki

Agreguj sumę weighted loss po ważnych pozycjach i dziel przez ich liczbę. Oczekiwany fixture: (24×10+3×30)/27=12.222222…, nie 16.666666…. Sprawdź rzeczywiście opublikowany epoch log w Trainer, nierówne batche/długości/padding/reset, i kontrolę pełnego horizon. Zachowaj returned batch loss do backward i identyczne gradienty; nie normalizuj przez sumę clinical weights. Nie loguj dwa razy pod tym samym monitorem.

## 5. T1 — checkpointy po zatwierdzeniu D4

1. Wprowadź wersjonowany manifest/rejestr własnego runu i wspólną walidację: dataset hash, protocol danych i treningu, ordered schema/role/dtype/units/encodings, ordered q, fitted normalizer, context/horizon/length policy, model config. Resume dodatkowo: optimizer, scheduler/budget, data order, batch/accumulation, callbacks, precision, stos/urządzenie.
2. Hash danych inference dotyczy pochodzenia modelu; nie wymagaj, żeby przyszłe wejście miało hash treningowego parquetu. W tym zakresie nie migruj modelu na inne dane/normalizator także w weights-only.
3. Rozdziel fresh, resume-last tego samego runu, inference-best i weights-only nowy run. Brak checkpointu/zgodności → błąd, nie fallback do best, newest mtime lub świeżego treningu. Best wybieraj z pełnej wartości monitora w zweryfikowanym indeksie/payload, nie z nazwy. Zachowaj tie policy z D4.
4. Pełna deserializacja wyłącznie własnego zaufanego artefaktu po sprawdzeniu pochodzenia i hasha, zarówno na ścieżce build_model, jak i fit. Nie używaj globalnego override weights_only i nie traktuj hasha od obcego dostawcy jako dowodu zaufania. Publiczne fit(weights_only=False) dla własnego zatwierdzonego formatu jest wariantem naprawy load, nie dowodem exact resume.
5. Własny checkpoint w nowym procesie ma wczytać pełny stan. Osobno testuj każdy konflikt składowej (nie tylko złożony context+horizon+normalizer), corrupt bytes, role mismatch, absent manifest, invalid status, podmianę sidecar/payload, obcy run.
6. Zapisuj generacje niezmiennych plików; dopiero po ukończeniu i walidacji publikuj last/best. Symulacja przerwanego zapisu nie zmienia wcześniejszego wskaźnika. Zachowuj callback history w tym samym katalogu runu; zmiana lokalizacji nie jest automatycznym resume.
7. Certyfikowane wejście treningowe nie uruchamia evaluate/test-set performance po fit. Nie zmieniaj metryk Stage C; odseparuj automatyczne wywołanie od ścieżki train. W testach nadal obowiązują strażniki main/load_test_data/evaluate/Optuna.

T1 nie ogłasza resume gotowym: B10-F04 i F02 pozostają do T2.

## 6. T2 — granica epoki, profil SWA OFF

Po zatwierdzeniu D1/D2/D4:

- Profil v1.0: ukończona epoka po walidacji i zmianach callbacków, finite positive max_epochs, zachowany total optimizer/scheduler budget, FP32, pojedynczy proces, num_workers=0 w referencji. Zero oczekujących accumulated gradients przy zapisie. Brak step/mid-epoch resume. Zapis completed_epoch/next_epoch/global_step/boundary_complete.
- SWA OFF w konfiguracji rzeczywistej i manifestach; no-SWA EarlyStopping patience=20/min_delta=1e-4/val_loss. Nowa wersja training protocol. Jawne ON i resume ze SWA artefaktu do OFF odrzucone w tym profilu. Nie przedstawiaj B13-F01 jako naprawionego — DEFERRED/OUT OF SCOPE na mocy decyzji.
- Najpierw minimalny reproducer B10-F04: niezmienione total=8/max_epochs=4, 2 updates/epoka; stop na rzeczywiście domkniętej granicy po 4 updates, nowy proces, 4 dalsze updates. Brak pustej/zdublowanej epoki; używaj publicznych API. Nie manipuluj prywatnym fit_loop, nie zwiększaj max_epochs ani nie zmieniaj train length, aby uzyskać osiem.
- Po poprawnym budżecie dodaj pełny restore RNG/generatorów, samplera, modelu, optimizera, schedulera, EarlyStopping, ModelCheckpoint i GradientNormLogger. Inicjalizacja dataloadera/sanity/logowanie nie mogą konsumować dodatkowego stanu asymetrycznie. Referencja z dropout/shuffle i kontrola bez losowości.
- Porównaj first resumed batch, wszystkie dalsze batch IDs, RNG przed forward, LR, optimizer moments, callbacki, liczniki i wagi/predykcje. Stan dyskretny i porządek identyczne. Bitwise CPU dla etykiety bitwise exact; dawną tolerancję można raportować oddzielnie jako zgodność numeryczną, nigdy jako ukryte poluzowanie exact.
- Zweryfikuj continuation z checkpointu po ES: zakończony run nie podejmuje kolejnych updates bez osobnego nowego eksperymentu. Testuj akumulację i odmowę partial-epoch/partial-accumulation checkpointu. Przerwanie w środku epoki cofa tylko do ostatniej zatwierdzonej granicy z jawnym odnotowaniem odrzuconego segmentu pracy.
- Jeśli publiczna granica na tym stosie nadal nie działa, zakończ tę część jako BLOCKED i podaj reproducer/opcje. Nie zmieniaj środowiska i nie zawężaj samodzielnie v1.0 do fresh-only. Niezależne zatwierdzone naprawy mogą pozostać ukończone.

## 7. Testy, budżet i uczciwość wyników

Używaj rzeczywistych komponentów produkcyjnych, unittest, syntetycznych danych. Maks. 8 aktualizacji TFT na fit i 4 syntetyczne epoki; test schedulera do 160 tanich kroków. Nowy zestaw remediation: watchdog 600 s, Stage A liczony osobno. Re-runy tylko po zmianie/failure lub dla nierozstrzygniętej kontroli, z jawnie raportowanym łącznym czasem i liczbą optimizer steps. Nie zwiększaj budżetu bez osobnej decyzji. Testy publicznej pętli na skalarze nie wymagają treningu TFT na długiej sekwencji.

Jeśli CUDA check PASS i zakres zezwala na testy urządzenia: krótki syntetyczny FP32 forward/backward, finite guards i replay na GPU z tym samym limitem na fit. Oznacz wyniki CUDA osobno. CUDA BLOCKED nie jest PASS, ale nie uniemożliwia raportu T0/T1 CPU. Nie instaluj AMP ani nie włączaj go do profilu.

Uruchom:

- składnia/bezpieczne importy, bez main i side effects treningowych;
- świeże regresje Stage A;
- zatwierdzone testy remediation oraz wymagane istniejące testy Stage B;
- końcowy diff/check i kontrolę braku danych/model artifacts w Git.

Nie zmieniaj historycznego BASELINE_AUDIT_STAGE_B.md. Przy zatwierdzonym zawężeniu zakresu (SWA OFF, brak mid-epoch) zachowaj oryginalne reproduktory i tabelę stare testy→nowe acceptance/DEFERRED z decyzją. Możesz oddzielić diagnostykę nieobsługiwanej ścieżki od acceptance v1.0, ale nie usuwać jej ani nadawać xfail/skip jako sposobu zamknięcia. Nowe testy odmowy nie dowodzą naprawy starego mechanizmu. Przy T0 pozostałe FAIL są oczekiwanym stanem niezatwierdzonego zakresu, nie regresją do ukrycia.

## 8. Wyjścia i Definition of Done

Przygotuj `docs/BASELINE_AUDIT_STAGE_B_REMEDIATION.md` z Change, Reason, Validation, Scientific impact, Remaining risks i tabelą wszystkich 11 findings: severity, zatwierdzony zakres, FIXED/OPEN/BLOCKED/DEFERRED, test/dowód, warunek zamknięcia. Aktualne metodologiczne decyzje podaj dosłownie i ze źródłem zgody. Wskaż commit/dirty/diff hashes oraz granice certyfikacji urządzeń/stosu.

Kod i testy zmieniaj minimalnie w zatwierdzonym zakresie. Dodaj konfigurację remediation/profilu, jeżeli niezbędna; nie przepisuj historycznej konfiguracji audytu tak, jakby używała nowych ustawień. Lokalne manifesty/checkpointy/logi pozostają ignorowane; raport zawiera tylko bezpieczne podsumowania i hashe.

DoD pakietu:

- każdy finding w aktywowanym pakiecie spełnia szczegółowy DoD z planu lub ma jawny BLOCKED; brak domyślnego rozszerzenia scope;
- B15 invalid state odrzucany przez produkcję, bez niepoprawnego update i publikacji valid artefaktu;
- checkpointy, jeśli T1: role/manifest/zgodność/odczyt sprawdzone niezależnymi pozytywnymi i negatywnymi testami;
- resume, jeśli T2: równy budżet, pełny stan i data order udowodnione, nazwa exact odpowiada dowodowi; brak prywatnego obejścia pętli;
- gradient i objective finite niezmienione poza zatwierdzoną metodą, Stage A bez regresji;
- świeże liczby PASS/FAIL/ERROR/SKIP/XFAIL, rzeczywiste polecenia i exit codes, wszystkie focused reruns wyraźnie opisane;
- brak pełnego treningu, Optuny, Stage C, test performance, zmian środowiska i danych; actual synthetic_optimizer_steps zapisane;
- wcześniejsze valid artefakty/historyczne raporty zachowane, diff sprawdzony, brak sekretów/danych/checkpointów staged.

Zakończenie T0 albo T1 nie jest PASS całego Stage B. Gotowość Baseline v1.0 wymaga zamknięcia wszystkich blockerów dostępnych ścieżek, przyjęcia decyzji, przejścia kontraktów profilu i osobnego review. Nawet po PASS nie uruchamiaj pełnego treningu ani Stage C bez osobnego zlecenia.
