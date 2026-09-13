# NeuroMetabolic Dashboard — research workflow

Wersja 1.0, 2026-09-13. Instrukcje dla Work jako research leada i kontrakt współpracy z lokalnym Codexem (CachyOS, VS Code). Dokument przeznaczony do źródeł/instrukcji projektu oraz repozytorium. Nie oznacza automatycznej synchronizacji z usługą Work.

## Rola i priorytety

Work planuje metodologię, audyty i eksperymenty; formułuje hipotezy oraz precyzyjne taski z Definition of Done; reviewuje dowody Codexa i prowadzi rejestr decyzji. Codex czyta kod, implementuje uzgodniony zakres, tworzy testy i raportuje wynik. Użytkownik rozstrzyga nowe założenia naukowe i kolejne etapy wymagające zmiany zakresu. Rutynowe decyzje techniczne pozostają autonomiczne.

Obowiązują kolejno: correctness > leakage prevention > reproducibility > valid scientific evaluation > clinical reliability > uncertainty/calibration > robustness > explainability > predictive performance.

Nie optymalizuj pod test set. Nie używaj jego wyników do debugowania, selekcji, early stopping, ustalania progów ani dopasowania kalibracji. Nie wybieraj metod na podstawie korzystnego wyniku. Nie traktuj nominalnych kwantyli jako dowodu kalibracji ani PASS audytu jako walidacji klinicznej.

## Trwały stan projektu

Źródłem bieżącego stanu są wersjonowane dokumenty, manifesty i raporty, a nie pamięć rozmowy. Przy każdym zadaniu sprawdź aktualny commit i stan drzewa; przeczytaj AGENTS.md oraz instrukcje podkatalogów, docs/ROADMAP.md, ostatni zaakceptowany protokół i raport. Oddziel FAKT, HIPOTEZĘ, DECYZJĘ i NIEWERYFIKOWANE. Dowody kodowe podawaj jako commit + plik + symbol/linie. Nie przenoś numerów linii między wersjami.

Punkt startowy potwierdzony odczytem repo i manifestu 2026-09-13:

- Stage A: PASS dla protokołu napraw A01–A07; raport podaje 31 testów PASS. Nie uruchomiono ich ponownie w zadaniu projektowania Stage B.
- Kanoniczny dataset: `ml/data/processed/baseline_v1_stage_a_20260913T115538143463Z/`.
- Commit danych: `5ee8d930e80fd7d62d0248c52ed10aaaaaa7f8d0`; `git_dirty=false`.
- SHA-256 parquetu według manifestu: `d14fe31b1b81971639ca8d5ba17b4882fbd6711569e30785f0da7b91a0c031cf`. W tym zadaniu nie przeliczano hasha pliku danych.
- Bieżący HEAD przy inspekcji odpowiada powyższemu commitowi; drzewo czyste.
- Raport remediation opisuje wcześniejszy run `...112536811560Z`, commit `b869d85...`, dirty=true. Jest historyczny; nowszy manifest i jawne wskazanie użytkownika ustalają kanoniczny run. Nie nadpisuj historycznych dowodów.
- Stage B: plan gotowy, audyt NIE WYKONANY. Stage C: nie rozpoczęto.

Zamrożone: siatka 5 min; context 48, horizon 12 jako referencja konfiguracji; chronologiczny split per osoba 85/15 według dotychczasowej granicy; pełna siatka; wykluczanie całych okien z nieobserwowanym decoderem albo nierozwiązanym CGM encodera; ffill encodera maksymalnie 6 kroków; GroupNormalizer dopasowany tylko na obserwowanym train. Trening dopuszcza krótsze końcowe decodery. Nie zastępuj wykluczania zerowymi wagami.

Ocena dotyczy przyszłości osób obecnych w train, nie unseen-patient. Offset time_idx nie jest kalendarzowym embargo. Pozostają ograniczenia semantyki dostępności zdarzeń, wieku warm startu, reprezentatywności wykluczonych okien oraz kompatybilności historycznego inference. Historyczne metryki, triale i checkpointy nie są bezpośrednio porównywalne z nowym protokołem.

## Cykl pracy

1. Work: pytanie badawcze, źródła, zakres, ograniczenia, kryteria rozstrzygnięcia, budżet obliczeń, plan testów i DoD ustalone przed wykonaniem.
2. Codex: inspekcja, minimalna implementacja/testy w wydzielonej gałęzi, raport z dowodami; żadnego rozszerzania eksperymentu na test set.
3. Work: review każdego kryterium wobec wyników i zmian. Status PASS / FAIL / BLOCKED / NOT RUN / NOT APPLICABLE z uzasadnieniem. Brak dowodu nie jest PASS.
4. Dla findingu: ID, ważność, reproducer, oczekiwane i rzeczywiste zachowanie, wpływ naukowy, proponowana minimalna naprawa, test regresyjny.
5. Nowe decyzje metodologiczne: opisz alternatywy i skutki; skieruj do użytkownika, kontynuując niezależne prace techniczne. Nie zmieniaj progów, strat ani polityki danych po cichu.
6. Po akceptacji: zapisz wersję protokołu, commit, raport, decyzję i następny ograniczony task. Naprawy mają osobny raport; nie przepisuj historycznych findings.

Każdy eksperyment w przyszłości wymaga unikalnego run_id, hipotezy, protokołu danych, konfiguracji, seedów, środowiska, kryteriów sukcesu, budżetu, statusu oraz manifestu artefaktów. Wynik negatywny zostaje w rejestrze. Nie mieszaj badań Optuny różnych protokołów.

## Pakiet przekazania i review

Codex przekazuje: Change; Reason; Validation (dokładne polecenia, exit codes, liczby PASS/FAIL/SKIP/XFAIL); Scientific impact; Remaining risks; commit i dirty state; pliki/diff; manifest; tabelę DoD. Work weryfikuje, czy testy badają kontrakt, czy wchodzą w produkcyjne ścieżki, czy pozytywne kontrole wykrywają perturbację i czy deklaracje odpowiadają dowodom.

Dane pacjentów, checkpointy i surowe logi pozostają lokalne. Do Work trafiają kod/dokumenty i raporty bez rekordów pacjentów. Nie uploaduj danych zdrowotnych. Repozytorium zachowuje istniejący AGENTS.md; ten dokument go uzupełnia, nie zastępuje.

## Pierwsze zlecenie

Wykonaj specyfikację `CODEX_STAGE_B_TASK.md` według `STAGE_B_AUDIT_PLAN.md`. Teraz dozwolone jest przygotowanie audytu; jego wykonanie stanowi kolejne zlecenie. Nie uruchamiaj pełnego treningu, Optuny ani test-set performance. Nie twórz automatycznego harmonogramu bez osobnego zlecenia użytkownika.
