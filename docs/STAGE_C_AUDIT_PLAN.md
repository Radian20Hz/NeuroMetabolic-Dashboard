# Baseline v1.0 — Stage C audit plan

Data: 2026-09-14. Status: PLAN, EXECUTION NOT STARTED. Wyłącznie audyt: najpierw findings, osobne review, następnie ewentualna remediation. Ten dokument nie aktywuje wykonania ani nie zmienia produkcji.

## 1. Punkt startowy i źródła

Przeczytano AGENTS.md, RESEARCH_WORKFLOW.md, ROADMAP.md, BASELINE_AUDIT_STAGE_A_REMEDIATION.md, BASELINE_STAGE_B_REMEDIATION.md, STAGE_B_FINAL_REVIEW.md, STAGE_B_FINALIZATION_REPORT.md i configs/baseline_v1.json. Dodatkowo wykonano statyczny odczyt evaluate i helperów persistence/ARIMA w ml/scripts/train_tft_population_v2.py. Żadnych importów modelu, testów, forward, treningu ani obliczania metryk w zadaniu projektowym.

HEAD inspekcji: `95c2723316d29d984c4aba2884c7bdb0603e3ba8`; drzewo czyste na wejściu. A i B formalnie PASS. Aktualizacja na początku STAGE_B_FINALIZATION_REPORT zamyka również callback restore; wcześniejsze FAIL/BLOCKED w dalszej części i stary status RESEARCH_WORKFLOW są historyczne. Stage A:31 PASS, końcowy Stage B:38 PASS (32 remediation +6 focused). Nie uruchomiono ich ponownie tutaj.

Zamrożone: dataset `ml/data/processed/baseline_v1_stage_a_20260913T115538143463Z/`; provenance commit `5ee8d930e80fd7d62d0248c52ed10aaaaaa7f8d0`, dirty=false, parquet SHA-256 `d14fe31b1b81971639ca8d5ba17b4882fbd6711569e30785f0da7b91a0c031cf`. Protocol `nmd-baseline-v1.0-stage-b-2`, SWA OFF, CPU strict/CUDA seeded, context48/horizon12, co 5 min, within-subject temporal. Nie zmieniamy Stage A window filtering, normalizatora, splitów, lossu, architektury ani checkpoint semantics.

## 2. Trzy warstwy i dwa zakresy pracy

| Warstwa | Pytanie | Dowód możliwy bez treningu |
|---|---|---|
| Implementation correctness | Czy indeksy, maski, wzory, redukcje, baseline’y i artefakty realizują zapisany kontrakt? | Niezależne oracles, unit/synthetic i prawdziwy PF→predict→evaluate integration na fixture. |
| Scientific validity | Jaki estimand mierzymy, czy porównanie jest uczciwe, co znaczy weighted quantile i czy założenia kalibracji/inferencji są uzasadnione? | Analiza matematyczna, testy kontraktów i plan identyfikacji. Faktycznej jakości/kalibracji nie dowodzi nietrenowany model. |
| Clinical interpretation | Co można uczciwie powiedzieć o znaczeniu błędów i niepewności? | Review etykiet, jednostek, zakresów, ograniczeń. Brak walidacji klinicznej i decyzji terapeutycznych na podstawie smoke testu. |

**C-core — rekomendowany pierwszy task:** inspekcja, syntetyczne unit/integration, matematyczna analiza i raport luk. Zero optimizer steps, zero real-data prediction, zero calibration fit na rzeczywistych danych, zero dostępu do rzeczywistego test setu. Brak produkcyjnego modułu kalibracji/metryki raportuj jako NOT IMPLEMENTED/OPEN; nie implementuj go w ramach audytu.

**C-dev — osobny przyszły zakres, nieaktywowany tym taskiem:** analiza reprezentatywności train/validation, rzeczywista ocena jakości/kalibracji i ewentualne objective variants wyłącznie na zatwierdzonych train/validation/calibration partitions. Wymaga osobnego zlecenia, checkpointu i protokołu z sekcji 6. Test pozostaje zamrożony także w C-dev.

C-core może być kompletnym audytem z werdyktem FAIL lub częściowo NOT EVALUATED. Nie uznawaj rzeczywistej kalibracji, skuteczności ani całego Stage C za PASS tylko dlatego, że implementacyjne testy są zielone.

## 3. Kontrakt ewaluacji do zatwierdzenia przed C-core

### Jednostka i wspólny zbiór

Jednostka: `(subject, forecast_origin_timestamp, target_timestamp, horizon_minutes)` oraz unikalny window_id. target_timestamp=origin+h, h=5,10,…,60 min. Wersjonowany manifest kwalifikowanych okien Stage A stanowi bazę **S**; dla TFT i persistence ta sama ordered lista, obserwowane targety, wagi i mianowniki. Timestamps rzeczywiste, nie interpretacja sztucznego offsetu time_idx jako czasu kalendarzowego.

Model nie wybiera populacji przez swoje predykcje. Brak lookup persistence, zła kolejność, nonfinite na ważnej pozycji lub brak predykcji ma dać jawny invalid evaluation/finding; nie wolno milcząco liczyć porównania na przecięciu pozostałych wyników. Diagnostyczny paired subset może zostać osobno opisany, lecz nie zastępuje porównania na pełnym S. Padding wyłączony tylko przez prawdziwe decoder_lengths. Uszkodzony target i nieobserwowany target to odrębne przypadki.

Persistence: ostatnia dostępna dla modelu glukoza na forecast origin, w mg/dL, stała dla wszystkich h. Może być dozwolonym przez Stage A ffill≤6 kroków; zapisz observation age/ffill flag. Nie wymaga świeższego pomiaru niż TFT, nie pobiera najbliższej przyszłej próbki, nie odtwarza mg/dL heurystyką ze znormalizowanego encoder_cont. Zgodność z końcem dopuszczonego encodera jest obowiązkowa.

Wystąpienia tego samego target_timestamp w różnych origin/horizon są poprawnymi zadaniami prognozowania, nie niezależnymi pacjentami/pomiarami. Raportuj N forecast-target occurrences, N unique windows, N unique target timestamps i N subjects. Nie deduplikuj arbitralnie przed porównaniem.

### Metryki

Dla e=pred−y, na tym samym S_h:

- MAE=mean(abs(e)), RMSE=sqrt(mean(e²)), bias=mean(e), w mg/dL.
- MARD=100×mean(abs(e)/abs(y)) dla obserwowanych y>0; nie jest ilorazem średnich ani medianą ARD. Zero/ujemny target: jawny błąd/niezdefiniowany wynik z liczbą przypadków, bez clamp mianownika do1. Małe dodatnie y zachowują swój wkład; nie ukrywaj czułości MARD na niską glikemię.
- Finite nieprawidłowa fizjologicznie predykcja pozostaje błędem modelu w metrykach i dostaje osobną flagę; nie odrzucaj jej dlatego, że pred≤0. Nie wprowadzaj po cichu maski y∈(20,400). Ewentualna polityka jakości sensorów wymaga niezależnej uprzedniej decyzji i identycznego zastosowania do modeli.
- Wszystkie 12 horyzontów, z wyróżnionymi 15/30/60 min; bez wyboru najlepszego. Raportuj pooled micro oraz per-patient i macro (równa waga osób z danymi), N przy każdym wyniku. Macro RMSE=średnia RMSE osób, odrębna od pooled RMSE. All-horizon summary tylko jako wtórna, jawnie zdefiniowana średnia; nie zastępuje krzywej horyzontów.
- Empty group → null/undefined, count0, powód; nigdy0% błędu,100% coverage ani brakująca osoba po cichu. Rounding tylko prezentacyjne, pełna precyzja w artefakcie.
- Raportowanie zakresów według **obserwowanego targetu**: hypo<70, in-range70≤y≤180, hyper>180 mg/dL; wtórnie y<54 i y>250. Testuj dokładne granice. To proposed reporting strata, nie nowe kryterium wykluczenia/terapii i nie indywidualne cele leczenia. Stratyfikacja po origin glucose, jeśli dodana, ma osobną nazwę i odpowiada innemu pytaniu.

Zakres 70–180 mg/dL oparto na konsensusie CGM; jego użycie do tabel błędów nie czyni odsetka forecast occurrences miarą time-in-range ani predykcji klinicznie bezpieczną. [International Consensus on Time in Range](https://diabetesjournals.org/care/article/42/8/1593/36184/Clinical-Targets-for-Continuous-Glucose-Monitoring).

Nie wyprowadzaj progu „clinical accuracy” z MARD<10/15%, Clarke A+B≥99% lub samego niskiego błędu. Jeżeli kod raportuje Clarke grid, audytuj geometrię i granice względem niezależnej implementacji/źródła oryginalnego, ale oddziel poprawność klasyfikacji od zasadności klinicznej dla prognozy o h>0. Nie myl błędu prognozowania z dokładnością sensora względem referencyjnego badania laboratoryjnego.

### Dodatkowy baseline

Persistence jest obowiązkowy. Pierwszy audyt nie potrzebuje nowego baseline’u. Istniejące ARIMA(1,1,0) audytuj pod kątem causal input, jednostek, zależności, convergence/failure i odrębnych podzbiorów; co najmniej test kontrolowanej awarii helpera. Refit na samym historycznym encoderze nie jest uczeniem na przyszłych targetach; uzupełnianie wewnątrz całej dostępnej historii oceniaj względem forecast origin, nie automatycznie jako future-target leakage. Polityka imputacji nadal musi być jawna.

Opcjonalny przyszły prosty comparator: średnia ostatnich6 dostępnych pięciominutowych wartości encodera (30-min trailing mean), stała po horizon, bez clippingu lub strojenia. Jest w pełni zdefiniowany na kwalifikowanych encoderach. Jego dodanie do produkcji/porównania jakości wymaga osobnej decyzji; audit-only oracle może zbadać semantykę. Nie instaluj statsmodels ani nie wybieraj baseline’u na podstawie performance.

## 4. Checklista C00–C19 i testy

Wszystkie statusy początkowo NOT RUN. „Oczekiwane” jest kontraktem audytu, nie twierdzeniem o aktualnej produkcji.

| ID | Warstwa i zakres | Minimalny dowód/reproducer i oczekiwane zachowanie |
|---|---|---|
| C00 | Implementation: isolation | Allowlist syntetycznych plików, guards real test loader/main/train/Optuna, dodatnia próba zakazanego dostępu daje odmowę. Synthetic test-labelled sentinel nie wpływa na train/val/cal decisions; real test files nieotwierane. |
| C01 | Implementation: prediction/index alignment | Minimum2 syntetyczne osoby, ten sam lokalny time_idx, różne trajektorie; permutacja batchy, nierówny ostatni batch, zmienne długości. Keys i target timestamps zgodne z PF x_to_index, żadnego zip po niezweryfikowanej pozycji. |
| C02 | Implementation: units/target transform | Różne normalizer scales/centers, oracle mg/dL, zmiana skali×2. Dokładnie jedna inverse transform, 0.5 po wartości q z artefaktu; tuple(target,weight) nie jest target_scale. |
| C03 | Implementation: eligibility/finite | Perturbacja padding, observed flags, predNaN/±Inf, valid targetNaN, finite pred≤0, y20/400/401. Bez nan_to_num i nowych pred-dependent masks. Persistence/TFT identyczny S i SHA window keys. |
| C04 | Implementation: scalar metrics | Ręczny przykład y=[50,100], pred=[60,80]: MAE15, RMSEsqrt250, MARD20%, bias−5. Perfect prediction, asymmetric errors, near-zero/zero, empty i extreme finite; batch partition nie zmienia wyniku. |
| C05 | Implementation/scientific: patient aggregation | Osoby o nierównym N/błędach: micro≠macro zgodnie z oracle, jawny N i missing groups, brak mieszania grup lub średniej batchy. |
| C06 | Implementation: horizons | Sygnał liniowy z unikalnym znacznikiem każdego h; 5→index0,15→2,30→5,60→11. Wszystkie12 h, real timestamps, krótkie decodery/padding poprawnie. |
| C07 | Three layers: glycemic strata | y=53.9/54/69.9/70/180/180.1/250/250.1; rozłączne główne grupy sumują się do S; subgroup metrics i N poprawne; brak pred-dependent bins i deklaracji bezpieczeństwa. |
| C08 | Implementation/scientific: persistence | Stały/ramp signal, ffill przy origin, różne osoby z tym samym time_idx. Pred=g(origin) w mg/dL dla każdego h. Brak lookup ma unieważnić pełne porównanie, nie tworzyć korzystnego subsetu. |
| C09 | Implementation/scientific: dodatkowy baseline | ARIMA success/failure mock na nieprzyległych window IDs, kontrola wejścia≤origin i mg/dL; błędy/skips jawne. Wynik optional-only nie zastępuje pełnego matched persistence/TFT. Trailing mean tylko spec/oracle, bez produkcji. |
| C10 | Implementation/scientific: quantile semantics | Q=[.02,.10,.25,.50,.75,.90,.98] z modelu/checkpointu, ordered/unique/in(0,1), output axis i median zgodne; brak pomylenia mean, median, percentile i probability. Brakujący endpoint → odmowa, nie indeks z globalnej listy. |
| C11 | Implementation/scientific: crossing | Ręczne noncrossing/tied/crossed tensors. adjacent crossing rate oraz any-crossing per window/h, max magnitude w mg/dL. Ties nie są crossing; bez sortowania lub przypisania starych etykiet nowej kolejności. |
| C12 | Implementation: coverage/width | Quantile hit mean(1{y≤qhat}); interval PICP=mean(1{L≤y≤U}), width=U−L. Fixtures na końcach i poza nimi. Pary .25/.75=50%, .10/.90=80%, .02/.98=96%; żadnego „90%” z nieistniejących .05/.95. Wszystkie h/osoby/strata, te same denominator keys. |
| C13 | Scientific: weighted objective | Niezależna analityczna CDF dla target-dependent wagi i deterministyczna quadrature/finite sample oracle, bez optimizer. Waga1 odzyskuje zwykłe kwantyle, waga2.5 poniżej70 przesuwa cel; test factor2 nie zmienia minimizatora. |
| C14 | Implementation/scientific: calibration | Mapa istniejącego kodu fit/apply i jego split lineage, jeśli istnieje. Na syntetycznych danych osobne fit/apply; perturbacja assessment/test sentinel nie zmienia fitted parameters. Jeśli brak modułu: dokumentuj brak, tylko niezależny audit oracle kontraktu CQR, bez implementacji produkcji. |
| C15 | Scientific: dependence/uncertainty | Fixture duplicate/overlapping origins: liczniki rozróżniają occurrences i unique timestamps. Udowodnij, że nominalne N okien nie jest liczbą niezależnych jednostek; blueprint block/subject uncertainty, nie IID bootstrap wierszy. |
| C16 | Scientific: representativeness | Synthetic missingness zależna od pory/poziomu i kontrola niezależna: flow candidates→retained/excluded, overlap powodów, unique vs occurrences. Rekonstrukcja nieobserwowanych y zabroniona; pełna skala bias na real data pozostaje C-dev. |
| C17 | Clinical: uncertainty language | Przechwyć logi/raporty na fixture o „dobrych” metrykach: brak automatycznego clinically SAFE, confidence HIGH/LOW lub gwarancji coverage z q. Oddziel predictive interval, confidence interval metryki i ryzyko zdarzenia. |
| C18 | Implementation: evaluation provenance | Synthetic owned checkpoint/manifest→prediction→metrics: SHA model/dataset/window list/normalizer/config/numeric profile, split role i code. Wrong checkpoint/schema/q/calibration source→odmowa; brak automatycznego latest selection. |
| C19 | Integration/review | Jeden mały TFT eval/no_grad przez prawdziwy PF predict i produkcyjny evaluate na fixture, obok niezależnego oracle i stubowanych exact predictions. Batch-size/permutation invariance, no optimizer, no production edits. Raport obejmuje wszystkie IDs i trzy osobne warstwy werdyktu. |

Po static inspection priorytet reproduktorów mają evaluate::nan_to_num, maski (20,400)/pred>0, baseline lookup subsets, global QUANTILES, coverage tylko t+60 i komunikaty kliniczne. Nie nadaj im statusu CONFIRMED runtime bez odtworzenia; brak funkcjonalności potwierdź mapą wywołań, nie samym brakiem pasującego słowa.

## 5. Kwantyle, crossing i kalibracja

Nie waż podstawowych metryk błędu/coverage clinical weights. Weighted training loss raportuj osobno, z nazwą i redukcją. Empiryczne quantile hit i interval coverage liczy się na rzeczywistej **nieważonej** populacji kwalifikowanych forecast targets; micro/macro jasno oddzielone.

Dla w(y)>0 minimalizator E[w(Y)ρq(Y−a)|X=x] jest kwantylem rozkładu `F_w(a|x)=E[w(Y)1{Y≤a}|x]/E[w(Y)|x]`, nie ogólnie F(a|x). Factor2 jest dodatnią stałą. Własne wyprowadzenie przez jednostronne pochodne/subgradient przy atomach należy do C13. Deterministyczny oracle: Y uniform[0,140], w=2.5 dla y<70, w=1 powyżej; nieważona mediana70, weighted mediana49. Dla dyskretnych fixtures kwantyl może być nieunikalny — sprawdź warunki minimizacji zamiast wymuszać jedną wartość na plateau.

Zatem poprawny ClinicalQuantileLoss nie dowodzi nominalnego coverage lub właściwej nieważonej mediany. Nie zmieniamy wagi w C-core. Niskie ważone val_loss nie uzasadnia nazwania przedziałów skalibrowanymi. Quantile calibration, interval coverage i conditional/patient-specific coverage są różnymi właściwościami; pooled coverage może maskować złą kalibrację hypo i poszczególnych osób.

Crossing: zachowaj surowe wyjścia i raportuj częstość/magnitude. Dla par z L>U wynik interval summary oznacz INVALID z N_invalid i pełnym mianownikiem, bez ujemnej „średniej szerokości” albo milczącego usunięcia tych próbek. Można dodatkowo podać formalny indicator coverage=0 dla pustego przedziału, ale nie przedstawiać tego jako poprawnej kalibracji. Poprawne pary/horyzonty raportuj odrębnie. Sorting/rearrangement/isotonic projection to osobna metoda postprocessingu do przyszłego zatwierdzenia, z raw i processed wynikami na tych samych oknach.

Kandydat do przyszłego calibration audit: split conformal/CQR, nie automatycznie wybrana metoda wdrożenia. Dla uporządkowanego przedziału score_i=max(L_i−y_i,y_i−U_i), k=ceil((n+1)(1−alpha)); użyj k-tej statystyki porządkowej, nie domyślnego interpolowanego percentile. k>n → brak skończonego certyfikowanego promienia (jawne insufficient data/+infinity w oracle), bez przycięcia k do n. Score może być ujemny; zasada shrinkage lub nonnegative expansion musi być jawnie wybrana przed real fit. Oracle obejmuje ties, small n i missing endpoints.

CQR może poprawiać marginal interval coverage także przy niedoskonałych kwantylach; nie naprawia automatycznie celu każdego q, conditional coverage ani clinical interpretation. Standardowe gwarancje wymagają odpowiednich założeń wymienności. Szeregi i nakładające się okna nie są IID; odstęp czasowy sam nie przywraca wymienności. Techniki block/time-series mogą wymagać odrębnych założeń i nie dają automatycznie gwarancji w OhioT1DM. [Romano et al., CQR](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html), [Chernozhukov et al., dependent-data conformal](https://proceedings.mlr.press/v75/chernozhukov18a.html).

## 6. C-dev — zaprojektowane, ale odroczone badania bez test setu

### Data roles przed real fit

Zachowaj zewnętrzne granice Stage A. W obrębie deweloperskich danych zaprojektuj per-person chronologiczne V_select→V_cal_fit→V_assess. V_select służy early stopping/model selection, V_cal_fit wyłącznie dopasowaniu wcześniej wybranej kalibracji, V_assess ocenie zamrożonej metody. Dokładne timestamp boundaries/liczności i zasadę minimalnych liczności zatwierdzić przed performance. Nie wybieraj ich według pokrycia/błędu.

Zakresy **target timestamps** tych ról muszą być rozłączne: okna przekraczające granicę decoderem są purge’owane. Dopuszczony encoder wcześniejszej historii jest zgodny z real forecast, ale trzeba zapisać zakres i nie traktować wspólnej historii jako niezależności statystycznej. H=60 min określa zakres targetów; nie deklaruj arbitralnego time_idx gap jako embargo. Dodatkowe bloki/gapy zależności wymagają osobnej decyzji i analizy tylko development data.

Jeśli istniejący model użył całego val do ES, później wydzielone V_cal_fit/V_assess nie stają się automatycznie niezależne. Należy zaprojektować nowy eksperyment z ograniczonym V_select i lineage albo uczciwie traktować wynik jako exploratory; nie przepisywać historii checkpointu. Jeżeli danych brak na niezależne role, raportuj ograniczenie lub zaproponuj z góry temporal cross-fitting. Nie pożyczaj testu. Ta zmiana wewnętrznych ról jest nowym protokołem eksperymentu, nie remediation Stage A.

Jeśli wybieramy metodę kalibracji na V_assess, ta część staje się danymi wyboru i nie jest już niezależnym końcowym assessment. Trzeba osobnego późniejszego bloku/cross-fitting z uprzednim projektem. Ostateczny test może być oceniony tylko w osobnym zleceniu po zamrożeniu wszystkich decyzji; obecny plan nie daje takiej zgody.

### Objective variants — tylko propozycja eksperymentu

W razie potrzeby porównaj obecny weighted objective z unweighted pinball przy tym samym factor2/redukcji, architekturze, inicjalizacjach sparowanych seeds (propozycja42/43/44), danych/oknach, budżecie i profilu numerycznym. Każdy wariant nowy run/protocol; BEST wybierany według jawnej reguły w V_select. Raportuj faktyczne liczby updates i ES; różne momenty ES nie są „tym samym wykonanym budżetem”.

Ocena przed/po kalibracji na oddzielnych development roles: nieważony pinball per q, MAE/RMSE/MARD, coverage/width, crossing, osoby/horyzonty/ranges. Bez nowego primary endpoint wybranego po wynikach. Przed eksperymentem zatwierdź kompromis point error–coverage–width i kryteria porównania; nie udawaj, że jedna metryka rozstrzyga wszystkie pytania. C-core nie trenuje tych wariantów ani nie wybiera zwycięzcy.

### Reprezentatywność filtrowania

Dokument Stage A podaje około43.37% wykluczeń train i47.41% validation. To sygnał do analizy, nie dowód rozmiaru/kierunku bias ani powód zmiany filtra. Estimand bieżącej ewaluacji to błąd **warunkowy na kwalifikację okna**, w tym przyszłą obserwowalność całego decodera; nie jakość przez całą dobę.

Plan C-dev: wyłącznie train/validation, osobno per person/pora dnia/okres/age ostatniego CGM/dostępność sensorów i historyczne zdarzenia; retention rates oraz rozkłady origin features przed/po filtrze, powody wykluczeń z nakładaniem, windows vs unique observations/time. Glikemię targetową porównuj wyłącznie tam, gdzie naprawdę obserwowana, z denominator i brakami; nie imputuj nieobserwowanych outcomes, aby ocenić błędy na odrzuconych oknach. Odrzucenie częściowo obserwowanego okna usuwa również obserwowane cele — opisz je osobno. Nie wnioskuj MAR/MNAR lub wielkości bias bez identyfikowalności. IPW/reweighting/sensitivity scenarios wymagają nowej decyzji; nie zmieniaj metryk ani filtra automatycznie.

Uncertainty of estimates: paired differences TFT−persistence na tych samych keys, per-person wyniki, a ewentualne CI przez uprzednio ustalone resampling całych osób/czasowych bloków. Mała liczba6 osób ogranicza uogólnienia; IID bootstrap nakładających się wierszy i testy udające tysiące niezależnych obserwacji są niedopuszczalne. Długość bloków i liczba replik wymagają prerejestracji na danych deweloperskich przed końcowym assessment. Coverage pointwise nie jest jednoczesnym pokryciem 12-step trajektorii.

## 7. Provenance i uczciwe findings

Artefakt ewaluacji musi wiązać: run_id/parent, checkpoint role BEST i hash, code/dirty/diff, numerical profile, dataset/protocol/schema/normalizer/quantiles, role splitów i ich boundaries, window-key hash, predictions hash, maskę/denominators, baseline definition, units, metric version, calibration method/config/fitted-source hash, seed/środowisko, polecenia/exit codes i exclusion/failure counts. Bez dowodu pochodzenia brak porównania. Nie nadpisuj historycznych wyników; brak trained checkpointu jest ograniczeniem real evaluation, nie powodem użycia legacy artefaktu.

Finding schema: `Cxx-Fyy`, title, severity (BLOCKER/HIGH/MEDIUM/LOW), layer, status (CONFIRMED/SUSPECTED/NOT IMPLEMENTED), commit+plik+symbol/linie, minimalny reproducer/polecenie, expected/actual, scientific impact, minimal proposed remediation, test przyszłego zamknięcia i decyzja potrzebna. Findings numeruj po dowodzie; przykłady w planie nie są wynikami audytu.

BLOCKER: test leakage, zła para prediction/target, ciche naprawianie NaN/Inf lub ukryta selekcja populacji unieważniająca główne porównanie. HIGH: niewłaściwa skala/kwantyle/coverage, fałszywa gwarancja kalibracji/bezpieczeństwa lub brak niezależnych ról. MEDIUM: błędna agregacja/reprezentacja ograniczająca konkretny wniosek bez naruszenia całego pipeline. LOW: śledzalny problem prezentacji bez zmiany wyniku. Severity zawsze uzasadnić; luka empiryczna nie jest automatycznie bugiem kodu.

## 8. Decyzje, zakres i Definition of Done

Przed C-core: (D0) osobno aktywować audit-only; (D1) przyjąć kontrakt keys/metryk/denominators, wszystkich12 horizonów, micro+macro i strata z sekcji3 jako oczekiwania audytu; (D2) obowiązkowa persistence na pełnym S, ARIMA jako legacy audit, bez nowego baseline’u produkcyjnego. Brak zatwierdzenia odmiennej interpretacji → raport decyzji, nie samodzielne przepisywanie produkcji.

Przed C-dev, nieblokujące synthetic C-core: dokładne role/calibration boundaries i niezależność checkpointu; metoda/crossing policy i poziomy50/80/96; objective variants/budżet/seeds/selection criteria; zgoda na analizę rzeczywistych train/validation i pola danych; resampling i minimalne liczności klinicznych podgrup. Żadna decyzja nie może opierać się na test-performance.

Szacowany C-core:20 obszarów, około35–55 krótkich testów zachowania (liczba orientacyjna, nie kryterium sukcesu), jeden izolowany runner, raport findings i rejestr decyzji. Unit CPU float64 oracle atol1e-10/rtol1e-8; integracja FP32 atol1e-5/rtol1e-5 ustalone przed runem, IDs/counts exact. Synthetic real-TFT do64 windows i24 forward batch calls na suite, zero backward/optimizer. Watchdog600 s; brak powiększania budżetu bez wyjaśnienia. GPU nie jest potrzebne do kontraktu matematyki ewaluacji. Real model forward bada plumbing, nie jakość predykcji.

DoD audytu: wszystkie C00–C19 mają status i dowód/ograniczenie; produkcyjne zachowanie porównano z niezależnymi oracles; matched S dla TFT/persistence zbadany; izolacja testu potwierdzona kontrolą dodatnią; weighted objective wyjaśniony matematycznie bez zmiany; scientific/clinical questions oddzielone od testów implementacji; nowe findings kompletne, bez naprawiania/xfail jako zamknięcia; manifest i dokładne PASS/FAIL/ERROR/SKIP/XFAIL/exit codes; wszystkie NOT RUN/NOT IMPLEMENTED jawne; brak zmian produkcyjnych/treningu/real test access. Raport zakończenia audytu może brzmieć FAIL. Potem osobna decyzja o remediation oraz C-dev; żadnego automatycznego otwarcia test setu.
