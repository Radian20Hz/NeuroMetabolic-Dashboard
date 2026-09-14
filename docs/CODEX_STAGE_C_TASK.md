# Codex task — Stage C audit, bez remediation

Status: SPECIFICATION ONLY / NOT STARTED. Samo utworzenie tego dokumentu nie uruchamia audytu. Wykonuj dopiero po osobnym zleceniu użytkownika. Zakres wykonania to C-core opisany w STAGE_C_AUDIT_PLAN.md; C-dev pozostaje propozycją przyszłego badania.

## Cel i punkt startowy

Zidentyfikuj i odtwórz problemy evaluation pipeline Baseline v1.0, rozdzielając implementation correctness, scientific validity oraz clinical interpretation. Najpierw findings, potem osobna decyzja o naprawach. Stage A i Stage B pozostają formalnie PASS; ich poprawność nie oznacza poprawności ewaluacji.

Przed wykonaniem przeczytaj w całości:

- AGENTS.md oraz instrukcje właściwe dla zmienianych katalogów;
- docs/RESEARCH_WORKFLOW.md i docs/ROADMAP.md;
- docs/BASELINE_AUDIT_STAGE_A_REMEDIATION.md;
- docs/BASELINE_STAGE_B_REMEDIATION.md;
- docs/STAGE_B_FINAL_REVIEW.md i docs/STAGE_B_FINALIZATION_REPORT.md, łącznie z aktualizacją zamykającą wcześniejsze findings;
- configs/baseline_v1.json;
- docs/STAGE_C_AUDIT_PLAN.md i ten task.

Zapisz bieżący commit, status drzewa i różnicę względem punktu projektowania `95c2723316d29d984c4aba2884c7bdb0603e3ba8`. Nie nadpisuj zmian użytkownika. Stosuj repozytoryjne zasady gałęzi. Historyczne FAIL w raportach Stage B nie unieważniają późniejszego formalnego PASS.

Zamrożony protokół: SWA OFF; obecny ClinicalQuantileLoss bez zmian; context48/horizon12; CPU strict/CUDA seeded. Kanoniczny dataset i jego hashe identyfikuje plan. Nie otwieraj rzeczywistych danych w celu potwierdzania hashy w C-core: wykorzystaj istniejące manifesty/raporty, oznacz ich pochodzenie jako inherited evidence. Nie przedstawiaj tego jako świeżej walidacji pliku.

## Dopuszczone i zabronione działania

Po aktywacji wolno czytać kod i dokumentację, dodawać audytowe fixtures/testy/runner oraz raporty. Wszystkie nowe artefakty wykonania mają trafiać do osobnego katalogu Stage C oznaczonego synthetic, bez nadpisywania baseline artefacts. Nie zmieniaj plików produkcyjnych ani baseline config. Instrumentacja i monkeypatch mogą żyć wyłącznie w testach; nie mogą usuwać badanego błędu.

Zabronione: remediation, instalowanie/aktualizowanie bibliotek, pełny trening, backward/optimizer steps, Optuna, rzeczywiste prognozy i metryki train/validation/test, fit kalibratora na rzeczywistych danych, objective experiments, zmiana lossu/features/architektury/splitów/filtering/checkpoint semantics. Nie uruchamiaj entrypointu treningowego ani automatycznej ewaluacji BEST na test. GPU nie jest wymagane; nie zmieniaj środowiska, aby je udostępnić.

Dopuszczony jest rzeczywisty forward nietrenowanego małego TFT i produkcyjna funkcja evaluate **wyłącznie na zweryfikowanej syntetycznej fixture**. Syntetyczny split nazwany test służy testowi izolacji, nie jest rzeczywistym test setem. Nie używaj istniejącego trained checkpointu. Jeśli integration wymaga checkpointu, utwórz należący do audytu checkpoint niezoptymalizowanego modelu, jawnie opisany jako synthetic/untrained i niecertyfikowany do inference.

Nie buduj brakującego produkcyjnego modułu kalibracji lub nowej metryki. Niezależny testowy oracle jest dozwolony; jego PASS nie dowodzi poprawności nieistniejącej funkcji produkcyjnej.

## Kolejność pracy

1. Sprawdź aktywację i uzgodniony kontrakt D0–D2 z planu. Zapisz decyzje bez ponawiania pytań już rozstrzygniętych przez użytkownika. Rozbieżności metodologiczne oznacz jako OPEN DECISION; kontynuuj niezależną inspekcję, nie naprawiaj produkcji.
2. Zmapuj drogę dataset/index→PF predict→inverse transform→masks→metrics/baselines→logs/artifacts. Wskaż rzeczywiste funkcje, biblioteki i side effects. Sprawdź importy przed uruchomieniem; zależności o brakującym środowisku raportuj bez instalacji.
3. Utwórz audytowy config zawierający zatwierdzone wzory, quantiles, role danych, budżet i tolerancje **przed pierwszym runem**. Wersjonuj go i zachowaj hash. Nie dostrajaj tolerancji do wyniku.
4. Najpierw uruchom guard izolacji: allowlist plików syntetycznych, blokada rzeczywistych loaderów i nieautoryzowanych entrypointów przed otwarciem danych. Kontrolowana próba dostępu do zabronionej ścieżki ma zostać odrzucona. Sam brak wywołania w logu nie dowodzi skuteczności guardu. Syntetyczny sentinel zmieniany w roli test nie może wpływać na decyzje/fit train/val/cal.
5. Wykonaj C00–C19 z planu przez niezależne ręczne/numeryczne oracles i małe fixtures. Priorytet: prediction/target alignment, jednostki, NaN/Inf, populacja/mianowniki i persistence, potem metryki/kwantyle/interpretacja/provenance.
6. Zrealizuj jeden integration przez prawdziwe PF predict i evaluate, oraz kontrolowane exact predictions do porównania metryk. Nie zastępuj całego badanego pipeline stubem. Rejestruj osobno test rzeczywistego plumbing i testy wymuszonych edge cases.
7. Opracuj matematyczną analizę weighted pinball, przegląd założeń kalibracji, reprezentatywności i języka klinicznego. Dla pytań wymagających danych/treningu zapisz NOT EVALUATED oraz konkretny przyszły dowód; nie wykonuj C-dev.
8. Zapisz findings i trzy odrębne werdykty. Sprawdź diff: wyłącznie audytowe pliki i dokumentacja, żadnych napraw produkcji. Zakończ i przekaż raport do research review.

## Minimalne pokrycie testowe

Macierz C00–C19 w planie jest obowiązująca. Każdy ID musi mieć mapowanie na reproducer/test albo uzasadnione ograniczenie. Szczególnie:

- TFT i persistence mają identyczną pełną listę kwalifikowanych keys dla każdego horyzontu; brak baseline lookup jest jawnym błędem pełnego porównania. Nie uznawaj korzystnego wspólnego subsetu za spełnienie tego wymogu.
- Dwie osoby z nakładającym się time_idx, nierówny ostatni batch, przestawione batche, krótkie decodery i różne normalizer scales ujawniają pomyłki indeksów i jednostek.
- Niezależne MAE/RMSE/MARD/bias; per-patient/micro/macro; wszystkie12 horyzontów; wartości na granicach strata i empty groups. Nie używaj produkcyjnej funkcji jako własnego oracle.
- NaN/Inf na ważnych pozycjach powoduje jawny invalid evaluation. Padding ma osobną maskę. Finite błędnych predykcji nie usuwa się z metryk. Testuj obecne nan_to_num i maski również poprzez realną ścieżkę wywołania.
- Quantiles pochodzą z artefaktu/modelu, nie niezweryfikowanego indeksu globalnej listy. Dostępne central intervals to50/80/96%; brak endpointów dla90% nie może zostać przemilczany.
- Crossing raw outputs, ties, quantile hits, PICP i width mają niezależne fixtures. Bez samoczynnego sortowania lub pomijania odwróconych przedziałów.
- Weighted quantile oracle z planu musi wyjaśnić zmianę estimandu; nie uruchamiaj porównania trenowanych objectives.
- Kalibracja: audytuj istniejące fit/apply oraz ich lineage. Jeśli nie istnieją, raportuj brak. CQR order-statistic oracle sprawdza definicję k, small n i ties; nie stanowi wdrożenia ani empirycznej certyfikacji kalibracji.
- Syntetyczna selekcja przez missingness ujawnia różnicę populations; brak identyfikacji rzeczywistego selection bias jest osobnym ograniczeniem.
- Logi nie mogą nadawać clinical safety ani confidence na podstawie arbitralnego MARD/Clarke threshold; odtwórz komunikat zamiast zmieniać go w audycie.
- Provenance łączy checkpoint, dane, keys, predictions, metric definitions, config, normalizer, quantiles, protocol i role kalibracji. Mismatch musi dać jawny błąd lub finding.

Budżet: około35–55 krótkich testów jako estymacja, nie wymuszona liczba; do64 synthetic windows oraz24 rzeczywistych forward batch calls łącznie, zero optimizer steps; watchdog600 s. Oracles float64: atol1e-10/rtol1e-8; integration FP32: atol1e-5/rtol1e-5; keys/counts exact. Nie wymagaj bitwise różnych batch partitions FP32. Zmiana budżetu/tolerancji wymaga uzasadnienia przed kolejnym runem i zachowania pierwotnego wyniku, bez konwersji błędu na PASS dla wygody. Brak możliwości uruchomienia to ERROR/BLOCKED, nie dowód poprawności.

## Wyniki wykonania

Utwórz `docs/BASELINE_AUDIT_STAGE_C.md`, audytowy config (np. `configs/baseline_stage_c_audit.json`), izolowane tests/runner zgodne ze strukturą repo oraz manifest/logi wyników w osobnym katalogu. Nie zmieniaj plików planu, aby dopasować oczekiwania do zastanego wyniku.

Raport zawiera:

1. Commit/dirty state, dokładne komendy, environment, budżet, config hash, source provenance oraz listę przeczytanych plików.
2. Macierz C00–C19: warstwa, expected contract, dowód, status, finding IDs. Rozdziel wynik runnera PASS/FAIL/ERROR/SKIP/XFAIL od oceny naukowej NOT EVALUATED/OPEN DECISION. Nie dodawaj xfail, by ukrywać potwierdzony błąd; pokaż rzeczywisty wynik i exit code.
3. Każdy finding w formacie poniżej, z priorytetem i statusem dowodu. Podejrzenia bez odtworzenia są SUSPECTED, a nie CONFIRMED. Brak implementacji potwierdzony inspekcją jest NOT IMPLEMENTED.
4. Oddzielne wnioski dla implementation correctness, scientific validity i clinical interpretation. Audit COMPLETE może mieć FAIL; empiryczna calibration/clinical reliability nie otrzymują PASS z testów syntetycznych.
5. Rejestr decyzji, odroczony C-dev, minimalne propozycje remediation i przyszłe testy zamknięcia, bez ich wdrożenia.
6. Deklarację ograniczeń wraz z dowodem guardów, że nie użyto rzeczywistego testu, treningu, Optuny ani zmian produkcyjnych. Końcowy diff i lista własnych artefaktów.

### Format findingu

- ID: Cxx-Fyy, np. C03-F01; xx odpowiada obszarowi planu, yy to kolejny finding w tym obszarze.
- Title; layer; CONFIRMED / SUSPECTED / NOT IMPLEMENTED.
- Severity: BLOCKER / HIGH / MEDIUM / LOW z uzasadnieniem zasięgu.
- Evidence: commit, plik/symbol/linie, fixture hash i dokładny minimalny reproducer.
- Expected versus actual, z wartościami/logiem i exit code tam, gdzie wykonalne.
- Scientific impact: jaki estimand, porównanie lub wniosek staje się błędny; oddziel potencjalny wpływ od wykazanego.
- Classification: technical bug / methodological decision / documented limitation; możliwe osobne składowe.
- Minimal proposed remediation: propozycja, nie wykonanie; potrzebna decyzja użytkownika.
- Future regression test i Definition of Done zamknięcia findingu.

## Definition of Done

Audyt jest zakończony, gdy każdy obszar ma śledzalny wynik lub jawne ograniczenie, kluczowe kontrakty mają niezależne reproduktory, wszystkie findings są kompletne i rozdzielono trzy warstwy oceny. Test isolation ma dodatnią kontrolę. Weighted objective i conditional-on-eligibility estimand są wyjaśnione bez automatycznych zmian. Raport nie myli PASS testowego oracle z poprawnością produkcji ani empiryczną walidacją. Nic z remediation/C-dev/test evaluation nie zostało wykonane. Kolejny krok to review findings i osobna decyzja użytkownika; nie rozpoczynaj go samodzielnie.
