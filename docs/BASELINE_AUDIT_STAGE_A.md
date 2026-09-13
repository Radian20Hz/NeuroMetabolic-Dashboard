# Baseline v1.0 — audyt etapu A

Data: 2026-09-13. Status: **etap A nie daje zgody na uznanie baseline'u za wolny od leakage**.
Potwierdzono poprawne elementy izolacji splitów i indeksowania, ale także siedem
problemów opisanych poniżej. Nie zmieniono logiki produkcyjnej, modelu ani backendu.

## Zakres i źródła dowodów

Audyt obejmuje aktualne `preprocess_ohiot1dm.py` oraz przygotowanie danych,
split, warm start i `TimeSeriesDataSet` w `train_tft_population_v2.py`.
`tune_tft.py` importuje te same funkcje przygotowania danych i datasetów.
Starszy `train_tft.py` nie jest tym samym protokołem i nie otrzymuje certyfikacji
na podstawie tych testów. Backend, jakość predykcji, trening i kalibracja są poza etapem A.

Dowody:

1. Inspekcja kodu repozytorium i lokalnego kodu PyTorch Forecasting 1.7.0.
2. Testy jednostkowe i integracyjne na wymyślonych sygnałach: XML → preprocessing
   → tymczasowy parquet → loader train/val/test → rzeczywiste TimeSeriesDataSet
   → odczyt okien i tensorów. Żaden test nie wywołuje `fit()`, `predict()` ani `evaluate()`.
3. Odczyt wyłącznie `subject_id`, `source_split`, `timestamp` z lokalnego parquetu.
   Raport zawiera tylko agregaty, bez identyfikatorów i rekordów pacjentów.
   Nie odczytywano wartości glukozy w kontroli lokalnych danych; nie wyliczano
   wyników modelu na test secie ani nie używano ich do decyzji.

Nie regenerowano lokalnego parquetu ani nie porównywano go rekord po rekordzie
z surowymi XML. Dlatego poprawna chronologia lokalnego pliku nie dowodzi, że
powstał on z aktualnej wersji preprocessingu. Nie budowano datasetu z rzeczywistych
wartości pacjentów. Testy okien dotyczą pełnego przebiegu syntetycznego.

## Wersja audytowanego kodu i środowiska

Gałąź: `research/baseline-audit`.
HEAD: `f96ae05d97376334eacafede3c9b5ff98634f814`.
Na początku pracy istniały niezacommitowane zmiany w trzech skryptach ML.
Audyt dotyczy drzewa roboczego, a nie samego HEAD. SHA-256 plików produkcyjnych:

| Plik | SHA-256 |
| --- | --- |
| `ml/scripts/preprocess_ohiot1dm.py` | `f3d95e62634b3dd6f93c4d4d91f49f6b80dafd89d0f693839d0aa73f9a99f64f` |
| `ml/scripts/train_tft_population_v2.py` | `4cb8b60bcdf0e33a0ee51a4202f8c673c99803f9403407011170026966621f94` |
| `ml/scripts/tune_tft.py` | `cb66a532c54e1fba146ef0ca8aa3ae07b3f8a0760ec133d3ed4c4b5be81a3500` |

Środowisko: Python 3.14.7, NumPy 2.4.4, pandas 2.3.3, SciPy 1.17.1,
PyTorch 2.11.0+cu130, Lightning 2.6.1, PyTorch Forecasting 1.7.0.
Obliczenia testowe: CPU, jeden wątek OpenMP/OpenBLAS, bez modelu i bez GPU.
Sygnały testowe są deterministyczne; nie losują danych.
Lokalne środowisko nie ma pytest, dlatego testy używają standardowego `unittest`
(są również zbieralne przez pytest). Wynik dotyczy tych wersji bibliotek,
nie przypiętego w backendzie PyTorch Forecasting 1.1.1.

## Wynik kontroli rzeczywistych metadanych

| Kontrola | Wynik |
| --- | --- |
| Liczba pacjentów | 6 |
| Pacjenci z train i test | 6/6 |
| Chronologia train → validation po podziale 85/15 | 6/6 |
| Chronologia całego źródłowego train → test | 6/6 |
| Wspólne timestampy train/test dla tej samej osoby | 0 |
| Duplikaty `(subject_id, source_split, timestamp)` | 0 |
| Brakujące subject/split/timestamp | 0 |
| Etykiety splitu | wyłącznie `train`, `test` |
| Timestampy poza siatką 5 minut | 0 |
| Odstępy > 1 h wewnątrz pacjenta i splitu | 55 |

Podział 85/15 jest liczony według liczby zachowanych wierszy, nie 85% czasu
kalendarzowego. To opis aktualnego protokołu, nie stwierdzenie jego optymalności.
Przetwarzanie raw rozpoznaje test po ciągu `testing` w nazwie XML; pozostałe nazwy
traktuje jako train. Lokalne etykiety są poprawnie rozdzielone czasowo, ale
walidacja nazw i kompletności plików źródłowych pozostaje osobną kontrolą pochodzenia.

## Potwierdzone problemy i minimalne poprawki do rozważenia

Poprawek poniżej **nie zaimplementowano**. Audyt dostarcza reproducerów;
zmiany definicji cech lub populacji targetów wymagają jawnego opisania nowego
protokołu i nie pozwalają bezpośrednio porównywać historycznych wyników.

### A01 — trend glukozy używa przyszłych próbek

Miejsce: `preprocess_ohiot1dm.py::_rolling_linear_deviation`, ok. linii 623–645.

Po `fftconvolve(g_vals, w[::-1], mode="full")` kod wybiera fragment
`[win - 1 : n + win - 1]`. Wartość dla indeksu t otrzymuje ważoną sumę
z okna rozpoczynającego się w t, zamiast kończącego się w t. Średnia rolling
pozostaje prawostronna, więc zestawiane są też statystyki różnych okien.
Zmiana glukozy od indeksu 40 zmienia cechę już w indeksie 35 dla win=12.

Dowód: `test_A01_trend_feature_must_not_read_future` oraz test numeryczny
`test_known_defects_have_numeric_reproducers`.

Wpływ: bezpośredni temporal leakage w kolumnie `glucose_deviation_from_trend`
w parquet. **Kolumna nie jest obecnie aktywną cechą v2**; nie przypisujemy jej
wpływu na wynik aktualnego modelu bez takiego połączenia.

Minimalna poprawka: wyrównać konwolucję do prawego końca okna (`full[:n]`)
i porównać wynik z niezależnym referencyjnym dopasowaniem trendu do przeszłego
okna. To lokalna korekta indeksowania, niewymagająca zmiany modelu.

### A02 — inicjalizacja resting HR pobiera pierwszą przyszłą obserwację

Miejsce: `train_tft_population_v2.py::_compute_hr_resting_causal`, ok. linii 358–399.

`hr.dropna().iloc[0]` uzupełnia również wiersze poprzedzające pierwszą obserwację.
Dla 30 brakujących wartości i późniejszego HR=123 wcześniejszy wynik wynosi 123.
Zmiana przyszłego pierwszego HR zmienia cały brakujący prefiks.

Dowód: `test_A02_resting_hr_before_first_observation_must_not_read_future`.

Wpływ: nieprzyczynowa wartość pomocnicza. `hr_resting_estimate` nie jest bezpośrednio
w liście aktywnych unknown reals v2; test nie dowodzi, że każda zależna aktywna
cecha zmienia się przed pierwszym HR, ponieważ niektóre są wtedy zerowane.

Minimalna poprawka: przed pierwszą dostępną obserwacją używać wyłącznie
rzeczywistego wcześniejszego warm startu albo oznaczenia braku. Wybór stałego
fallbacku zamiast braku jest decyzją o semantyce cechy, nie automatyczną naprawą.

### A03 — cechy liczone po wierszach kompresują luki, mimo poprawnego time_idx

Miejsca: `process_patient` usuwa nieuzupełnione wiersze przed feature engineering;
`add_rolling_insulin_carb`, `compute_iob_cob`, `add_meal_bolus_timing`;
przeliczanie lagów i rolling w `_compute_long_window_features`.

Bolus sprzed 3 h 5 min nadal występuje w `bolus_last_1h`, gdy między nim
a bieżącym wierszem nie ma zachowanych rekordów. IOB po sześciogodzinnej przerwie
jest traktowane jak IOB po jednym kroku, mimo skończonej długości kernela.
`shift(1)` oznacza poprzedni rekord, który nie musi być sprzed pięciu minut.
Usunięcie wierszy bez CGM może również usunąć zdarzenia insuliny/posiłków
z tych wierszy przed obliczeniem ich późniejszego wpływu.

Dowody: dwa testy `test_A03_*`. Lokalny parquet ma 55 luk > 1 h, zatem
nie jest to wyłącznie hipotetyczny kształt danych; nie mierzono wpływu na predykcje.

Wpływ: błędna semantyka czasu, nie samo użycie przyszłości. Dotyczy aktywnych cech.

Minimalne warianty: liczyć cechy na pełnej siatce przed odrzucaniem wierszy CGM
albo rozdzielać ciągłe segmenty i resetować stan; rolling można też definiować
czasowo. Segmentacja odcina dostępną starszą historię, pełna siatka wymaga jawnej
obsługi braków CGM i zachowania niezależnych zdarzeń. Wybór wymaga decyzji
metodologicznej. Sama korekta `time_idx` nie naprawia wcześniej policzonych cech.

### A04 — dataset tworzy nieobserwowane targety wewnątrz luk

Miejsce: `create_time_series_dataset`, `allow_missing_timesteps=True`.

Na syntetycznej dwugodzinnej luce dataset generuje okna, w których część decoder
targetów nie ma odpowiadających wierszy źródłowych. W PF 1.7.0 `__getitem__`
powtarza poprzedni rekord, w tym `target`; `weight` pozostaje None.
Test sprawdza rzeczywisty tensor i potwierdza skopiowanie poprzedniej glukozy.

Dowody: `test_A04_decoder_targets_must_not_be_synthesized_inside_missing_gap`
i `test_known_defects_have_numeric_reproducers`. Źródło biblioteczne:
`pytorch_forecasting/data/timeseries/_timeseries.py`, blok `repetitions`/`indices`
w `__getitem__` (lokalnie ok. linii 2140–2180).

Wpływ: syntetyczne etykiety mogą wejść do lossu i ewaluacji z pełną wagą.
To nie jest przekroczenie splitu ani przyszła informacja; to naruszenie
interpretacji targetu jako obserwacji. Powtarzane są również cechy, w tym
potencjalnie nieaktualne zmienne kalendarzowe i flagi braków.

Minimalne warianty: odrzucać okna z nieobserwowanym decoderem lub przenosić
jawną maskę obserwacji do lossu i metryk. Trzeba oddzielnie określić dopuszczalne
braki encodera. Samo `allow_missing_timesteps=False` spowoduje błąd dla obecnego
parquetu, nie definiuje jeszcze poprawnego protokołu okien.

### A05 — także jawnie uzupełnione CGM są targetami z pełną wagą

Miejsca: `process_patient` (`ffill(limit=6)`, `cgm_gap_flag`), konstrukcja datasetu.

Forward fill jest przyczynowy, ale wartości uzupełnione pozostają w kolumnie
targetowej. `cgm_gap_flag` jest cechą unknown, nie maską etykiet. Dla takiego
wiersza w decoderze `dataset[i]` zwraca `weight=None`. Sama flaga nie odróżnia
obserwacji i imputacji przy obliczaniu funkcji straty.

Dowód: `test_A05_imputed_cgm_targets_must_be_excluded_or_zero_weighted`.

Minimalna poprawka do uzgodnienia wspólnie z A04: wykluczenie lub zerowa waga
nieobserwowanych targetów, z raportowaniem pokrycia okien. To zmienia populację
oceny; nie należy wybierać wariantu na podstawie lepszego MARD.

### A06 — dostępność przyszłego czujnika zmienia aktywną cechę w przeszłości

Miejsce: `_compute_exercise_features_causal`, `has_hr/has_gsr/has_temp = ...any()`
i warunek wyliczający `autonomic_stress_index` (ok. linii 434–580).

Sprawdzenie dostępności czujnika dotyczy całego przekazanego szeregu. Gdy temperatura
pojawia się dopiero od indeksu 300, wcześniejsze wartości autonomic stress są
liczone inną formułą niż dla szeregu bez temperatury: zamiast fallbacku
`composite_stress_index` wybierana jest gałąź trzech składowych.

Dowód: `test_A06_future_sensor_availability_must_not_change_active_past_feature`:
identyczny prefiks, temperatura zmieniona tylko w sufiksie, różne wcześniejsze wyniki.

Wpływ: **potwierdzony temporal leakage aktywnej cechy v2**. Nie trzeba zmieniać
przyszłego targetu; wystarcza przyszła dostępność modalności.

Minimalna poprawka: wybierać formułę na podstawie maski dostępności znanej
w danym momencie, nie globalnego `.any()`. Trzeba ustalić, czy dostępność oznacza
bieżący pomiar, ostatni pomiar o dopuszczalnym wieku, czy dostępny prefiks.
Każdy wariant powinien zachować niezmienność prefiksu.

### A07 — correction_bolus_prior ma różną definicję między splitami

Miejsce: `load_and_preprocess_data`, ok. linii 1402–1429,
oraz `_compute_correction_bolus_prior`.

Train liczy rolling mean wskaźnika pojedynczego bolusa (minimum 1 próbka).
Walidacja i test używają rolling max przez 6 kroków, potem rolling mean 288
kroków (minimum 12 próbek i fallback 0.1). Identyczna historia daje różne
wartości zależnie od ścieżki przygotowania splitu.

Dowód: `test_A07_correction_prior_has_same_definition_in_train_and_validation`
porównuje cechę rzeczywiście zwróconą przez loader z helperem używanym w val/test.

Wpływ: niespójna definicja aktywnej cechy, nie bezpośredni leakage.
Minimalna poprawka: jedna definicja i wspólny helper dla wszystkich splitów.
Należy najpierw zdecydować, czy cecha oznacza częstość zdarzeń bolusa,
czy częstość stanów „bolus w poprzednich 30 minutach”; te wielkości są różne.

## Co potwierdzono jako poprawne

- Bolus, posiłek i wpis wysiłku o 00:04 pojawiają się o 00:05, nie o 00:00.
  Dopasowanie CGM i wearable w badanym przypadku używa wyłącznie wcześniejszych danych.
- Krótkie braki CGM są wypełniane ostatnią przeszłą wartością i oznaczane flagą.
  Dalsza przyszła obserwacja nie jest używana do interpolacji.
- Causal IOB/COB, rolling insuliny/węglowodanów i konwolucje wysiłku są
  niezmienne po perturbacji przyszłych zdarzeń na regularnej siatce.
- Aktywne cechy samego preprocessingu na fixture z kompletnymi sensorami
  przechodzą test perturbacji sufiksu. To nie obejmuje A01 ani A06 po recompute.
- Cechy po podziale przechodzą test perturbacji sufiksu przy obserwowanym początku
  i niezmiennej dostępności modalności. Ten warunek jest istotny wobec A02/A06.
- Loader v2 wyklucza `source_split='test'` z train i walidacji; perturbacja
  syntetycznej glukozy w test nie zmienia żadnego wiersza zwróconego train/val.
- Podział 85/15 jest osobny dla każdej osoby; grupą datasetu jest `subject_id`.
- Dla wszystkich okien syntetycznych train/val/test sprawdzono przynależność
  całego zakresu do właściwej osoby i splitu oraz zgodność targetów z wierszami.
- `time_idx` zachowuje trzygodzinną lukę jako różnicę 37 kroków, nie 1;
  odrzuca duplikaty, odwrócony czas i timestampy poza siatką.
- Walidacja używa wielu kolejnych okien, a nie jednego końcowego okna na osobę;
  decoder walidacji/testu ma 12 kroków. Trening dopuszcza krótsze końcowe horyzonty.
- GroupNormalizer w val/test zachowuje statystyki dopasowane na train.
- Przyszłe glukoza, insulin/basal i posiłki nie należą do known reals.
  Perturbacja decoder unknown values przy zamrożonym datasetcie referencyjnym
  nie zmienia encodera ani znanych wejść decodera.
  PF przechowuje unknown columns także w tensorze decoder_cont; nie jest to
  samo w sobie leakage. Lokalny TFT wybiera `decoder_variables`. Nie wykonywano
  forward modelu; end-to-end dowód niezmienności predykcji jest poza etapem A.
- Warm start z nakładającej się lub przyszłej historii jest odrzucany.
  Przy ciągłej historii lagi pierwszego wiersza val odpowiadają rzeczywistemu
  ogonowi train. Korzystanie z dostępnej przeszłości nie jest target leakage.

## Rozróżnienia metodologiczne i otwarte kwestie

1. **Target ma pochodzić z przyszłości względem forecast origin.** Błędem byłoby
   udostępnienie go wejściom encodera lub nieznanym przyszłym zmiennym modelu.
   Testy sprawdzają wyrównanie targetów i izolację wejść, nie zakaz przyszłych etykiet.
2. **Embargo w kodzie jest offsetem indeksu.** Pierwszy val jest realnie 5 minut
   po ostatnim train w fixture, ale indeks różni się o 110. Żadne rekordy nie są
   usuwane. Okna rozdzielają osobne datasety, nie udowodnione 545-minutowe embargo
   kalendarzowe. Offset nie jest błędem sam w sobie; nie należy tak go opisywać.
3. Komentarze sugerujące, że trailing `rolling()` przed splitem z definicji
   używa przyszłej walidacji, są niepoprawne. Decydują faktyczne zależności czasowe.
4. Test warm start używa całego źródłowego train, czyli również okresu walidacji
   85/15. Ponieważ okres ten poprzedza test, jest to historia dostępna czasowo;
   trzeba jawnie zapisać taki protokół. Statystyki normalizatora nadal pochodzą
   z treningowego 85%, a nie z testu.
5. Podział i normalizacja są **within-subject temporal**, nie unseen-patient.
6. Dostępność czasu trwania wysiłku na timestampie początku oraz pełnej dawki
   bolusa na `ts_begin` wymaga sprawdzenia semantyki zdarzeń źródłowych.
   Kod parsuje całkowite `duration`/dawkę w tym momencie. Jeśli oznaczają dane
   znane dopiero po zakończeniu, powstaje leakage dostępności. Etap A nie ustala
   arbitralnie, że są planem znanym na początku; to otwarta decyzja semantyczna.
7. Warm start dopuszcza granicę do 10 minut, a fallback dla większych przerw
   przenosi część statystyk skalarnych. Causal nie oznacza świeże ani równoważne
   pełnej historii. Dokładna polityka przeterminowania i luk wymaga ustalenia.
8. Testy nie dowodzą braku wszystkich możliwych błędów ani kompletności raw.
   Nie rozstrzygają opóźnienia rejestracji zdarzeń, stref czasowych czy
   nieobserwowanego pochodzenia już istniejących artefaktów.

## Testy i odtworzenie wyniku

Pliki:

- `ml/tests/test_baseline_stage_a.py` — 24 testy, syntetyczne fixture w katalogu
  tymczasowym usuwanym po zakończeniu.
- `ml/tests/audit_stage_a_metadata.py` — powtarzalna kontrola tylko trzech kolumn
  lokalnego parquetu, wynik zagregowany na stdout; osobny test sprawdza wykrywanie
  celowo dodanego nakładania splitów i duplikatu.

Uruchomienie z katalogu głównego:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nmd-stage-a-mpl PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s ml/tests -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ml/tests/audit_stage_a_metadata.py
```

Wynik: **24 testy, 16 spełnionych, 8 expected failures**, bez nieoczekiwanych
błędów. Osiem expected failures dokumentuje siedem problemów A01–A07 (A03 ma
dwa reproduktory). Są to testy oczekiwanych poprawnych własności, które obecna
produkcja narusza, a nie testy uznające wadliwe zachowanie za prawidłowe.
`unexpected success` wymaga usunięcia oznaczenia i przeglądu naprawy.
Zielony exit code zestawu **nie oznacza akceptacji baseline'u**.

Kontrola składni nowych plików przez `compile()` nie zapisuje bytecode.
`git diff --check` całego drzewa wykrywa istniejący wcześniej trailing whitespace
w `train_tft_population_v2.py:1400`; nie poprawiano go w ramach audytu.
Nowe pliki sprawdzono osobno; nie zmieniono trzech audytowanych skryptów.

## Wpływ naukowy i status zakończenia

Dodano wyłącznie testy i raport, zatem aktualna metoda treningu i ewaluacji
nie została zmieniona. Nie wykonano treningu, benchmarku ani selekcji na test secie.
Raport nie potwierdza historycznych metryk.

Przed baseline v1.0 wymagają rozstrzygnięcia aktywny leakage A06, czasowa semantyka
cech A03, polityka obserwowanych targetów A04/A05 i spójność A07. A01/A02
pozostają potwierdzonymi nieprzyczynowymi obliczeniami, z ograniczeniami wpływu
opisanymi wyżej. Minimalne poprawki przedstawiono do oceny, ale żadnej nie
wdrożono bez ustalenia odpowiedniej semantyki.
