# Baseline v1.0 — Stage A remediation

Data: 2026-09-13. Ten dokument opisuje naprawy; pierwotny
[audyt](BASELINE_AUDIT_STAGE_A.md) pozostaje historycznym zapisem stanu sprzed zmian.

**Status: PASS dla zatwierdzonego protokołu napraw A01–A07.** Spełniono siedem
kryteriów zakończenia zlecenia, w tym numeryczny test lossu i metadane finalnej
regeneracji. Nie oznacza to zatwierdzenia modelu do zastosowań klinicznych ani
zakończenia etapów B/C.

## Zatwierdzony zakres

Implementacja A01–A07 zgodnie ze zleceniem użytkownika i doprecyzowaniem A07:
brak etykiety correction bolus; używamy binarnego `bolus_event` i cechy
`bolus_event_prior`. Nie zmieniono backendu, XAI ani architektury TFT. Dodano
zmienne dostępności lagów w konfiguracji datasetu. Nie uruchomiono treningu,
Optuny ani oceny predykcji na rzeczywistym test secie.

## Zmiany przed / po

| Problem | Przed | Po |
| --- | --- | --- |
| A01 | Konwolucja trendu zawierała przyszłe próbki | Pełne trailing window, dewiacja od trendu dopasowanego do przeszłego okna; brak kompletnego okna → neutralna dewiacja jak wcześniej |
| A02 | Pierwszy przyszły HR wypełniał wcześniejszy prefiks | Do pierwszej obserwacji NaN lub historyczny warm start; brak stałej fizjologicznej; także sentinel w parquet jest NaN |
| A03 | Usuwanie pustych CGM przed feature engineering kompresowało czas i usuwało zdarzenia | Pełna siatka 5 min zachowana w parquet i po splitach; zdarzenia istnieją niezależnie od CGM |
| A04 | PF powtarzał target z poprzedniego wiersza na brakujących krokach | Wymagana pełna siatka i `allow_missing_timesteps=False`; walidacja indeksu przed utworzeniem datasetu |
| A05 | Imputowany CGM uczestniczył w lossie jak obserwacja | `target_observed` sprzed ffill; wykluczenie każdego okna z nieobserwowanym targetem decodera |
| A06 | `.any()` po całym szeregu wybierało formułę wcześniejszych cech | Dostępność bieżących wartości po przyczynowym resamplingu; row-wise wybór formuły autonomic stress |
| A07 | Inne definicje prioru w train i val/test; nazwa sugerowała nieistniejącą etykietę korekty | Jeden helper: rolling mean binarnego zdarzenia przez maks. 288 kroków, włącznie z bieżącym, `min_periods=1` |

Pliki produkcyjne:

- `ml/scripts/preprocess_ohiot1dm.py`: siatka, obserwowalność targetu, trend,
  zachowanie modalności także przy braku HR, dawka i binarny bolus.
- `ml/scripts/temporal_protocol.py`: wspólna walidacja/odtwarzanie regularnego
  czasu dla jednej osoby i jednego splitu, walidacja `target_observed`.
- `ml/scripts/train_tft_population_v2.py`: przyczynowy HR, dostępność sensorów,
  lagi i maski dostępności, wspólny prior, dataset oraz zabezpieczenie ewaluacji.
- `ml/scripts/observed_windows.py`: niezależne sprawdzanie encodera i decodera,
  filtrowanie przez wspierane API `TimeSeriesDataSet.filter`, statystyki wykluczeń.
- `ml/scripts/tune_tft.py`: przekazanie `--data-dir` i odrębna nazwa bazy/study
  dla zmienionego protokołu. Nie wykonano żadnego trialu.
- `ml/scripts/diagnostics/regenerate_stage_a.py`: regeneracja, metadane,
  konstrukcja datasetów i statystyki bez tworzenia modelu.
- `configs/baseline_stage_a.json`: konfiguracja odtwarzania protokołu.

## Czas i granice splitów

Siatka obejmuje wszystkie kroki od `ceil(first_CGM, 5min)` do
`ceil(last_CGM, 5min)`. Początek zaokrąglany wcześniej w dół dodawał pusty bin
przed pierwszym CGM testowego XML. Regeneracja wykryła nakładanie takich binów
z ogonem train; warm-start guard zatrzymał przebieg. Naprawa usuwa wyłącznie
ten sztuczny prefiks, nie przesuwa przyszłej obserwacji do przeszłości.
Regresja sprawdza sąsiadujące XML rozpoczynające się poza pełnymi pięcioma minutami.

Zachowano kryterium granicy 85/15: jest wyznaczana na wierszach z dostępnym
CGM po maksymalnie sześciu krokach ffill. Następnie wszystkie regularne biny
przypisywane są do train lub validation względem tego timestampu. Nie zmieniono
tego na 85% wszystkich nowych, gęstych wierszy. Rzeczywiste źródłowe train/test
pozostają oddzielne; warm start może korzystać tylko z wcześniejszej historii
tej samej osoby. Offsety indeksów między splitami nie są kalendarzowym embargo.

`bolus_dose` przechowuje sumę dawki w jednostkach insuliny. `bolus_event` wynosi
1, gdy ta dawka jest dodatnia, a 0 w przeciwnym przypadku. IOB, sumy dawek,
TDD i basal/bolus ratio korzystają z dawki; częstość zdarzeń korzysta ze wskaźnika.
Nie klasyfikujemy bolusów jako meal/correction.

Brakujący lag oznacza brak wartości dokładnie k×5 minut wcześniej, a nie ostatni
zachowany rekord. Reprezentacja modelowa takiego lagu to 0 z osobną zmienną
`glucose_lag_k_available=0`. Ta liczba nie jest traktowana jako obserwacja glukozy.
W przypadku dostępnego lagu maska wynosi 1. Dodano sześć takich masek.

## Polityka okien i lossu

Sprawdzono lokalny PF **1.7.0**, w szczególności:

- `TimeSeriesDataSet.__getitem__`: powtarza `target` przy lukach time_idx, jeśli
  włączone jest `allow_missing_timesteps`;
- `TimeSeriesDataSet.filter`: wspierane API filtrowania indeksu okien;
- `MultiHorizonMetric.update/reduce_loss`: sposób zastosowania wag i mianownika.

Wybrano **filtrowanie całych okien**, a nie ważenie. Pozwala zachować istniejącą
funkcję `ClinicalQuantileLoss` bez zmiany redukcji i mieć wszystkie zwracane
targety rzeczywiście obserwowane. To jawnie zmienia populację okien.

Encoder: każda glukoza musi być dostępna jako obserwacja albo dotychczas
dozwolony ffill ≤6 kroków; pozostaje `cgm_gap_flag`. Nierozwiązany NaN
w encoderze wyklucza okno. Decoder: **każdy** krok musi mieć
`target_observed=True`. Częściowo obserwowane decodery są wykluczane w całości,
również z obserwowanymi punktami w tym samym oknie. Ten koszt jest raportowany.

W gęstym parquet nierozwiązany CGM pozostaje NaN. PF nie akceptuje takich
wartości nawet przy budowie indeksu, więc na prywatnej kopii wejścia do datasetu
używany jest dodatni placeholder 1.0. Nigdy nie jest zwracany w dopuszczonym
encoderze ani decoderze. Nie jest fizjologicznym założeniem ani targetem.
GroupNormalizer jest jawnie dopasowywany **tylko do obserwowanych CGM train**
przed przekazaniem do PF; lokalny PF nie refituje dopasowanego normalizatora.
Val/test dziedziczą ten stan. Pozostałe niekompletne cechy mają reprezentację
liczbową po obliczeniu przyczynowych cech, nie zastępują brakujących etykiet.

Przed ewaluacją `assert_observed_evaluation` ponownie sprawdza okna względem
ramki źródłowej, zanim nastąpi wywołanie modelu. Odmawia oceny datasetu z
nieobserwowanym decoderem lub nierozwiązanym encoderem. Nie uruchamiano
rzeczywistej ewaluacji w ramach tego zadania.

Test integracyjny zmienia nieobserwowaną etykietę z 35 na 390 mg/dL,
tworzy rzeczywiste PF DataLoadery i wylicza produkcyjny `ClinicalQuantileLoss`
przy identycznych predykcjach. Indeksy i loss są identyczne liczbowo (zerowa
tolerancja). Kontrola dodatnia: zmiana obserwowanej etykiety zmienia loss.
Predykcje są celowo stałe, aby badać wpływ etykiety na nadzór; dozwolona
imputacja encodera nadal może wpływać na predykcję w innych oknach.

## Testy i weryfikacja

`ml/tests/test_baseline_stage_a.py`: **31 testów, wszystkie PASS, 0 expected
failures, 0 pominiętych testów**. Obejmuje:

- niezależną referencję trendu przez least squares;
- niezmienność prefiksu dla przyszłych wartości i przyszłej dostępności sensorów;
- brak przyszłej inicjalizacji HR i dopuszczony historyczny warm start;
- IOB/rolling po wielogodzinnych lukach, zachowanie zdarzeń podczas braku CGM;
- dokładne lagi z maską dostępności;
- identyczną definicję bolus prior we wszystkich splitach;
- odrzucanie nieobserwowanych targetów, niezależność lossu od ich wartości;
- izolację pacjentów/splitów, chronologię, zachowanie time_idx;
- guard przed niepoprawną ewaluacją;
- pojedynczy forward małego TFT na syntetycznych danych: poprawny kształt
  i skończone wartości, bez `fit`, backpropagation lub aktualizacji wag.

Uruchomienie:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nmd-stage-a-mpl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s ml/tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nmd-stage-a-mpl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ml.scripts.diagnostics.regenerate_stage_a
```

Kontrole składni nie zapisują bytecode; `git diff --check` przechodzi.

## Wpływ naukowy i kompatybilność

Zmiana dotyczy definicji cech, dostępności wejść i populacji nadzorowanych
okien. **Historyczne metryki, checkpointy i triale nie są porównywalne z tym
protokołem.** Nazwa cechy bolusa i dodatkowe maski lagów zmieniają schemat
datasetu. Starszy parquet bez `target_observed` jest jawnie odrzucany.

Modele tego protokołu mają odrębny katalog `ml/models/baseline_v1_stage_a/`
i prefiks `tft-stage-a-v1`; Optuna ma osobną nazwę study i bazę. Nie wznawiano
starych eksperymentów. Główne skrypty przyjmują `--data-dir` dla nowego katalogu;
historyczne `ml/data/processed/training.parquet` pozostaje nienaruszone.

Nie jest to dowód generalizacji na nowych pacjentów: nadal oceniamy przyszłość
osób reprezentowanych w train. Nie zweryfikowano skuteczności modelu,
kalibracji ani interpretacji klinicznej. Źródłowa dostępność opisów czasu trwania
wysiłku i całkowitej dawki na timestampie zdarzenia nie została niezależnie
zwalidowana; zachowano dotychczasową interpretację rejestrowanych zdarzeń.
Dotychczasowy limit dopasowania CGM wstecz 2 min 30 s i wiek historycznych
stanów warm start nie były optymalizowane na podstawie liczby wykluczeń.

Finalna identyfikacja danych, statystyki i wynik kontroli metadanych znajdują się
w sekcji wyniku regeneracji poniżej.

## Wynik regeneracji i pochodzenie danych

Finalny, zaakceptowany katalog lokalny:
`ml/data/processed/baseline_v1_stage_a_20260913T112536811560Z/`.
Zawiera `training.parquet` (źródłowe train i test rozróżnione kolumną splitu),
`provenance.json` oraz lokalny `preprocessing.log`. Dane i log pozostają
ignorowane przez Git. Starsze katalogi przebiegów są zachowane, ale nie są
finalnym artefaktem; pierwszy zatrzymał się na błędzie brzegów XML, kolejny
został zastąpiony nowym przebiegiem po doprecyzowaniu sentinela HR jako NaN.
Nie nadpisano ich ani historycznego głównego parquetu.

Regeneracja użyła 12 bieżących raw XML. Manifest zawiera:

- HEAD `b869d85c5d1d5a59d25c8e8a0c72caa1a162ef90` i `git_dirty=true`;
- SHA-256 wszystkich skryptów ML/testów, ponieważ sam commit nie opisuje zmian;
- konfigurację, zbiorczy digest źródłowych XML, SHA-256 wynikowego parquetu;
- wersje zależności, agregaty metadanych i statystyki okien;
- jawne `model_training_performed=false`, `model_performance_inspected=false`.

SHA-256 finalnego parquetu:
`d14fe31b1b81971639ca8d5ba17b4882fbd6711569e30785f0da7b91a0c031cf`.
Skrypt sprawdza zgodność hashy kodu i raw przed/po wykonaniu; zmiana w trakcie
przebiegu unieważnia manifest i wymaga nowego katalogu.

Środowisko: Python 3.14.7, NumPy 2.4.4, pandas 2.3.3, SciPy 1.17.1,
PyArrow 23.0.1, PyTorch 2.11.0, Lightning 2.6.1, PF 1.7.0; CPU, jeden wątek
OpenMP/OpenBLAS. W tym zadaniu nie wykonano commita.

Kontrola finalnych metadanych:

| Własność | Wynik |
| --- | --- |
| Osoby z train/test | 6/6 |
| Rzeczywiste granice train/val uporządkowane | 6/6 |
| Źródłowe train/test uporządkowane | 6/6 |
| Nakładające się timestampy splitów | 0 |
| Duplikaty w osobie/splicie | 0 |
| Braki subject/split/timestamp | 0 |
| Timestampy poza siatką 5 min | 0 |
| Luki timestampów >1 h w pełnej siatce | 0 |

Ostatni wiersz **nie oznacza braku przerw w pomiarze CGM**. Przerwy są teraz
zachowane jako wiersze z flagami/NaN, zamiast znikających timestampów.

### Wykluczenia okien

| Split | Kandydaci | Zachowane | Wykluczone | Udział wykluczonych | Niepoprawny encoder | Nieobserwowany decoder |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 63 394 | 35 902 | 27 492 | 43,37% | 25 644 | 23 998 |
| Validation | 12 359 | 6 499 | 5 860 | 47,41% | 5 602 | 5 428 |
| Test | 16 828 | 9 592 | 7 236 | 43,00% | 6 858 | 6 513 |

Powody odrzucenia mogą współwystępować, dlatego dwie ostatnie kolumny nie sumują
się do liczby wykluczonych. Kandydaci to okna wygenerowane przez PF przed filtrem,
z zachowaniem dotychczasowych długości: train może mieć krótszy końcowy decoder,
val/test mają 12 kroków. Nie są to wszystkie teoretyczne kombinacje długości okien.

### Wykluczenia wystąpień targetów

| Split | Wystąpienia w kandydatach | Nieobserwowane | Udział nieobserwowanych | Usunięte z całymi oknami | Zachowane obserwowane |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 759 936 | 272 316 | 35,83% | 329 706 | 430 230 |
| Validation | 148 308 | 63 288 | 42,67% | 70 320 | 77 988 |
| Test | 201 936 | 74 922 | 37,10% | 86 832 | 115 104 |

Są to **wystąpienia w nakładających się oknach**, nie liczba unikalnych pomiarów.
Nie stosowano zerowych wag (0 wystąpień); wybrano odrzucanie okien. Usunięte
wystąpienia obejmują też obserwowane punkty w oknach odrzuconych z innego powodu.

| Split | Wiersze pełnej siatki | Obserwowane CGM | Nierozwiązany brak CGM encodera |
| --- | ---: | ---: | ---: |
| Train | 63 328 | 40 589 | 22 147 |
| Validation | 12 425 | 7 184 | 5 155 |
| Test | 16 894 | 10 610 | 6 167 |

Wszystkie powyższe liczby są kontrolą dostępności danych i konstrukcji okien.
Nie obliczano ani nie oglądano testowych predykcji, MARD, lossu czy innych
wyników modelu. Polityka została ustalona przed uzyskaniem tych statystyk.

## Pozostałe ograniczenia

Nie pozostają otwarte błędy z A01–A07 w testowanym protokole PF 1.7.0.
Wymagana dalsza ocena obejmuje interpretację czasu dostępności źródłowych
adnotacji zdarzeń, trening/checkpointy (B), metryki i kalibrację (C), zgodność
starszych ścieżek inference oraz wpływ wykluczeń na reprezentatywność danych.
Backend, XAI i historyczny generator danych syntetycznych nie zostały przeniesione
na nowy schemat. Nie należy podłączać ich starych artefaktów do nowego treningu.
Żadne twierdzenie o poprawie jakości predykcji nie wynika z tego etapu.
