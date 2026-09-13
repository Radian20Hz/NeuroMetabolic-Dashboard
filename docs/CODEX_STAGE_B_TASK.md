# Task dla Codexa — NMD Stage B audit

Wykonaj audyt mechaniki uczenia Baseline v1.0 zgodnie z załączonym STAGE_B_AUDIT_PLAN.md (B00–B17) oraz RESEARCH_WORKFLOW.md. Celem są reproduktory, testy i raport; nie pełny trening ani poprawa metryk.

## Wejście

Repo lokalne: `/home/radian/NeuroMetabolic Dashboard`.

Najpierw przeczytaj AGENTS.md, instrukcje właściwych podkatalogów, docs/ROADMAP.md, docs/BASELINE_AUDIT_STAGE_A.md, docs/BASELINE_AUDIT_STAGE_A_REMEDIATION.md i aktualny kod. Zweryfikuj HEAD i dirty state bez usuwania zmian użytkownika. Pracuj na gałęzi badawczej, nie main.

Kanoniczne dane: `ml/data/processed/baseline_v1_stage_a_20260913T115538143463Z/`, provenance git_commit=`5ee8d930e80fd7d62d0248c52ed10aaaaaa7f8d0`, git_dirty=false. Nie używaj starszego runu wskazanego w historycznym remediation. Sprawdź manifest; hash danych można zweryfikować lokalnie bez obliczania metryk i bez udostępniania rekordów. Nie regeneruj danych.

Zakres kodu: ClinicalQuantileLoss, ClinicalTFT.configure_optimizers, build_model, train, find_best_checkpoint, GradientNormLogger, konfiguracja datasetów/normalizatora oraz właściwe lokalne implementacje PyTorch Forecasting/Lightning. Starszy train_tft.py, backend, frontend i XAI poza certyfikowanym zakresem. tune_tft wolno czytać wyłącznie dla zależności, nie uruchamiać.

## Wykonanie

1. Sporządź mapę przepływu targetów i predykcji, jednostek, redukcji lossu, kroków optymalizatora/schedulera, callbacków i checkpointów. Zanotuj dokładne wersje bibliotek; sprawdzaj ich lokalny kod, nie zakładaj zachowania z innej wersji.
2. Dodaj izolowane syntetyczne testy B00–B17. Preferuj istniejący unittest; nie instaluj pytest tylko dla tego zadania. Użyj produkcyjnych komponentów, niezależnych referencji i kontroli dodatnich.
3. Zachowaj limity planu: CPU float32, mały TFT, maks. 8 aktualizacji/run, maks. 4 syntetyczne epoki, do 160 tanich kroków testu schedulera, 10 minut dla nowych testów. Nie uruchamiaj głównego skryptu treningowego: uruchamia on także ewaluację testową. Resume test używa wyłącznie checkpointu utworzonego przez bieżący syntetyczny fixture.
4. Nie zmieniaj produkcyjnej metody w audycie. Minimalne wydzielenie punktu wejścia lub konfiguracji dla testowalności jest dopuszczalne, jeśli nie zmienia domyślnej semantyki i ma regresję. Błędy ujawnij testem oraz findingiem. Zmiana lossu, wag, scheduler policy, SWA, reguł resume czy danych wymaga osobnej specyfikacji napraw; nie naprawiaj przez dostosowanie oczekiwań testu.
5. Uruchom kontrole składni/importów bez startowania main, istniejące testy Stage A i nowy zestaw. Zapisz faktyczne polecenia i wyniki. Test wykrywający błąd może kończyć się FAIL; raport nie może przedstawiać takiego audytu jako PASS.
6. Przygotuj raport i listę tasków naprawczych z DoD. Nie rozpoczynaj tych tasków automatycznie.

## Wyjścia

Proponowane nowe pliki w repo:

- `docs/BASELINE_AUDIT_STAGE_B.md`: zakres, wersja kodu, środowisko, B00–B17 z dowodami, findings, wpływ naukowy, ograniczenia, werdykt i następne kroki.
- `ml/tests/test_baseline_stage_b.py` lub logicznie rozdzielony zestaw.
- `configs/baseline_stage_b_audit.json`: zamrożony budżet i parametry testów.
- W razie potrzeby `ml/scripts/diagnostics/audit_stage_b.py`: bezpieczny syntetyczny runner bez ewaluacji rzeczywistego test setu.
- Lokalny, ignorowany przez Git manifest audytu i wyniki maszynowe. Raport wskazuje ich hashe; checkpointy i szczegółowe logi pozostają poza commitem.

Każdy finding: ID Bxx-Fnn, ważność, status, commit/plik/symbol, reproducer, oczekiwane vs rzeczywiste, przyczyna, wpływ naukowy, minimalna propozycja naprawy i test regresyjny. Nie nazywaj hipotezy potwierdzonym błędem bez dowodu.

## Definition of Done

- Wszystkie cztery dokumenty wejściowe przeczytane, kanoniczny manifest zweryfikowany, różnica względem historycznego remediation jawna.
- B00–B17 ma status i dowód albo precyzyjny powód BLOCKED/NOT RUN/NOT APPLICABLE; żaden brak wykonania nie jest PASS.
- Loss ma niezależny oracle, test jednostek i redukcji; forward/backward i perturbacje badają rzeczywisty model.
- Scheduler, checkpoint, early stopping, SWA i resume mają testy zachowania lub udokumentowany reproducer ograniczenia/błędu.
- Determinizm opisany w granicach przetestowanego środowiska; odróżniono wczytanie wag od pełnej kontynuacji.
- Wykonano regresje Stage A; raport podaje liczbę PASS/FAIL/SKIP/XFAIL i exit codes zamiast oczekiwanej liczby 31 jako nowego wyniku.
- Brak pełnego treningu, Optuny, odczytu test-performance i zmiany protokołu Stage A; synthetic_optimizer_steps zapisane uczciwie.
- Raport zawiera Change, Reason, Validation, Scientific impact, Remaining risks oraz decyzje wymagające Work/użytkownika.
- Diff sprawdzony; żadne dane pacjentów, checkpointy, sekrety ani lokalne logi nie są staged.

Zakończenie tasku oznacza kompletny audyt z dowodami, również gdy werdykt brzmi FAIL. Nie ogłaszaj gotowości treningu Baseline v1.0, dopóki obowiązkowe kontrakty nie mają PASS po osobno przeprowadzonej remediation i review.
