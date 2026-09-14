# Codex task — Stage C mechaniczna remediation, synthetic-only

2026-09-14 · `nmd-stage-c-remediation-task-1` · **SPECIFICATION ONLY / IMPLEMENTATION NOT STARTED**.

Ten dokument powstał w zadaniu review. **Nie uruchamia implementacji.** Wykonać dopiero po osobnym, jawnym zleceniu użytkownika. Po przyszłej implementacji zatrzymać się na raporcie; bez automatycznego treningu, C-dev i test evaluation.

## 1. Cel i źródło wymagań

Naprawić 16 technicznych findings Stage C i prawidłowo opisać dwa ograniczenia metodologiczne. Zachować istniejący clinical loss i naukowy kontrakt ewaluacji. Rozstrzygający dokument: [STAGE_C_EXECUTION_CONTRACT.md](STAGE_C_EXECUTION_CONTRACT.md); szczegółowe review severity, zależności, minimalne naprawy i per-finding DoD: [STAGE_C_REMEDIATION_PLAN.md](STAGE_C_REMEDIATION_PLAN.md). Historyczny [BASELINE_AUDIT_STAGE_C.md](BASELINE_AUDIT_STAGE_C.md) i jego czerwone dowody pozostają niezmienione.

Przed pracą przeczytać AGENTS.md i instrukcje podkatalogów, powyższe trzy dokumenty, CODEX_STAGE_C_TASK.md, STAGE_C_AUDIT_PLAN.md, NEXT_SESSION_BRIEF.md, configs/baseline_v1.json oraz wskazane audit R1/R2, findings, config, wyniki i manifesty. Dokumenty A/B są źródłem inherited evidence, nie zastępują regresji zmienionego styku A/C.

HEAD review: `abdcb1abd51b48728972e45aca095468de17ef55`, branch `research/baseline-audit`, clean przed utworzeniem planu/tasku. Oryginalny kod audytu: `4cc11d8fd7704cf3baeb4301a3b722374aa94fbd`. Między nimi jedynie11 plików audytu; hashe produkcji i wskazanych lokalnych źródeł PF zgodne. Na wejściu przyszłego tasku ponownie sprawdzić HEAD/branch/status/diff i nie nadpisywać zmian użytkownika. Nie zakładać, że wtedy nadal będzie ten sam HEAD.

## 2. Zakres po aktywacji i granice

Dozwolone po osobnym zleceniu: minimalne zmiany ewaluacji/helperów, mechaniczna korekta kodowania GroupNormalizer, jawna kompatybilność/provenance, nowe synthetic acceptance/runner i raport. Preferowane obszary: `ml/scripts/train_tft_population_v2.py`, ewaluacyjne funkcje `observed_windows.py`, wąskie zmiany kontraktu/load integration w `checkpoint_registry.py`/`baseline_training.py` tylko gdy niezbędne; można wydzielić mały moduł ewaluacji. Żadnej szerokiej przebudowy treningu. Wybór nazw nowych plików jest detalem technicznym; każdy zmieniony symbol powiązać z findingiem.

Zabronione:

- pełny i krótki trening, `fit`, backward/optimizer steps, Optuna, optimizer-based objective experiments;
- jakiekolwiek real-data predictions/metrics, real train/validation/test loaders, canonical parquet i trained checkpoints; żadnego hashowania canonical parquet w tym tasku;
- instalacja/aktualizacja zależności, modyfikacja site-packages lub środowiska dla GPU; CPU wystarcza;
- zmiana ClinicalQuantileLoss, factor2, hypo threshold70, weight2.5, redukcji lossu, cech, architektury produkcyjnej, splitów, treningowego window filtering albo early-stopping population;
- implementacja/fit kalibratora, wybór nowej metody lub crossing postprocessing, sortowanie raw quantiles, nowy baseline, IID CI/resampling;
- cicha migracja dawnych modeli, przepisywanie historycznego configu/raportu tak, by stare wyniki wyglądały na zgodne z poprawioną semantyką;
- backend/frontend/XAI, publikacja, commit/push bez osobnego polecenia.

Dopuszczony prawdziwy forward **nietrenowanego małego TFT** i produkcyjny evaluate wyłącznie na syntetycznej fixture; checkpoint/state do integracji wyłącznie własny, jawnie synthetic/untrained, nigdy oznaczany jako wytrenowany BEST. Nie osłabiać produkcyjnej weryfikacji BEST, aby testowy artefakt przeszedł.

## 3. Kolejność wykonania

1. **G0:** przygotować osobny immutable run directory, acceptance config z hashami, budżetem, tolerancjami, fixture schema i typami błędów przed pierwszym runem. Zweryfikować guards dodatnią kontrolą. Zachować audit R1/R2 bez edycji; nowe acceptance badają nowe zachowanie, a nie tylko stare nazwy pól.
2. **G1a:** wprowadzić input evaluation context/provenance, canonical W/S i bramki source role, keys, targets i48/12 — C18/input, C01-F01, C03-F03.
3. **G1b:** usunąć grupowy fallback dla znanych osób zgodnie z przyjętym normalizatorem oraz walidować model/Q/schema — C02-F01, C10-F01, kompatybilność C18. Nie „naprawiać” skali dopiero na output. Treningowy index i observed-train fit policy pozostają niezmienione.
4. **G1c:** bezwzględna kontrola nonfinite/domain, brak selekcji po predykcjach i pełna persistence z lineage — C03-F01/F02, C08-F01/F02. Wzajemne bramki nie mogą maskować błędów w testach.
5. **G1d:** raw crossing i wymagane probabilistic metrics — C11/C12. C17 usunięcie nieuprawnionych komunikatów można wykonać wcześniej niezależnie. Nie nadawać intermediate PASS całej ewaluacji.
6. **G2:** pełnoprecyzyjne reductions, wszystkie h, micro/macro/per-patient/per-range — C04/C05/C06/C07; domknąć pełne przekroje C11/C12, finalny manifest C18 oraz interpretację C17/C13/C14. Proste wspólne primitives G2 mogą powstać już przy G1d. HIGH uncertainty nie jest zamknięte po samym h60.
7. **G3:** rzeczywista integracja, odczyt artefaktów i macierz zamknięcia. Zatrzymać się przed dalszymi eksperymentami.

Pierwszeństwo ryzyka: C01-F01,C03-F01,C03-F02,C08-F01 oraz pozostałe7 HIGH. Kolejność zależności jest ważniejsza od mechanicznego sortowania po severity. C18 ma osobne input/output etapy, bez cyklu zależności. Szczegóły grafu w planie.

## 4. Kontrakt wdrożenia

### W, keys, role i valid/invalid

W ustalić **przed predykcjami**, ze źródła/indeksu Stage A odpowiedniej roli: regularna siatka5min, jedna osoba, encoder48 z dozwolonym causal ffill≤6, wszystkie12 decoder targets observed, target timestamps w roli ewaluacji, bez duplikatów. Nieobserwowany decoder wyklucza całe okno przed predict; uszkodzony observed target nie jest „missingness” do cichego odrzucenia. Treningowe krótsze okna pozostają bez zmian. Oddzielna warstwa evaluation W nie zmienia validation population early stopping Stage B.

S = W × {5,10,…,60}; key=(subject,window_id,origin_timestamp,target_timestamp,horizon_minutes). Reprezentować i weryfikować kanoniczny ordered key set, bez zip po niezweryfikowanej pozycji. Zmiana batch order wymaga prawidłowego dopasowania rekordów; reject missing/extra/duplicate keys, mismatch y/source i długości. Timestamp to rzeczywisty czas fixture/source, nie sztuczny offset time_idx. Nie zastępować błędnego returned y etykietą ze źródła po cichu.

Nonfinite na wymaganej pozycji, y≤0, wrong units/schema/Q/role/keys albo brak persistence → jawna odmowa/INVALID odpowiedniego wyniku lub pełnego porównania, z planowanym N i powodem. Nie publikować valid metric artifact po częściowej awarii. W nie zmienia się przez awarię modelu. Finite p≤0 pozostaje we wszystkich stosownych metrykach z flagą, nie służy do selekcji. Brak clamp y do1 albo20…400. Dla ekstremalnego finite wejścia sprawdzić także wynik redukcji; overflow unieważnia obliczenie, a nie uruchamia clamp.

Prawdziwie pusta grupa ma null,N0,reason; p=0 nie oznacza pustej grupy. Empty W raportować jawnie bez model forward, nie jako sukces jakości. Wspólne timestampy różnych origin/h to odrębne zadania, nie niezależne pomiary; raportować occurrences,windows,unique(subject,target_timestamp),subjects.

### Normalizer i artefakty modelu

Jedna spójna mapa raw/encoded subject IDs przy observed-train-only fit, transform i get_parameters. W lokalnym PF1.7.0 categorical encoding następuje przed target transform; nie przekazywać uprzednio dopasowanych string-index statistics do integer lookup bez zgodnej mapy. Zachować istniejący typ/log transform/centering/estymator skali. Fitted state oraz faktyczny batch target_scale i encoder input muszą zgadzać się z niezależnym per-subject oracle. Val/assessment nie refitują; perturbacje nieobserwowanych labels nie wpływają na fitted statistics; unseen-patient odmawia zgodnie z baseline.

Ta korekta zmienia semantykę numeryczną i wymaga nowej jawnej rewizji kontraktu. Brak cichego użycia starego checkpointu pod poprawionym normalizatorem, również przez weights-only. Własne synthetic artifacts obejmują roundtrip mapy/normalizatora. Jeśli konieczna jest zmiana poza mechanicznym kodowaniem — np. estimator/windows/splits — zgłosić decyzję i zatrzymać zależną część zamiast zgadywać.

Q modelu/artefaktu zgodne z ordered [.02,.10,.25,.50,.75,.90,.98], unikalne i w(0,1), output shape zgodny. Nie sortować listy Q lub output, aby wymusić zgodność. Point output to zweryfikowane q=.5. PF quantiles/target są już w native mg/dL; zero drugiej inverse i heurystyk po wielkości liczb. Tuple(target,weight) nie jest target_scale.

### Persistence i metryki

Persistence = ostatnie observed g(s*) w48-bin encoderze, dostępne w origin, stałe dla12 h; age≤6, s*,ffill flag i zgodność z końcowym CGM TFT zapisane. Brak source lub lookup unieważnia comparison na pełnym W; żadnych progów90%/50%,nearest-future,pożyczania innej osoby,subset rescue. Stan poprawnych standalone TFT metrics można odróżnić od invalid comparison, ale nie ogłaszać ukończonego pełnego porównania.

Wszystkie metryki raportowe **nieważone**, occurrence weight1:

| Metryka | Definicja |
|---|---|
| MAE | Σabs(p−y)/N, mg/dL |
| RMSE | sqrt(Σ(p−y)²/N), mg/dL; sqrt po agregacji |
| Bias | Σ(p−y)/N, dodatni=przeszacowanie |
| MARD | 100 Σ(abs(p−y)/y)/N, y>0, bez clamp |
| Pinball per q | mean((y−zq)(q−1{y<zq})), mg/dL; bez factor2/clinical weights |
| Hit per q | mean(y≤zq) i hit−q; proportion, różnica pp po×100 |
| PICP | mean(L≤y≤U), inclusive |
| Width | mean(U−L), mg/dL, tylko valid pair/group |
| Any crossing | Σ1{istnieje z_j>z_j+1}/N |
| Adjacent crossing | Σ_iΣ_j1{z_ij>z_i,j+1}/[N(K−1)] |

Dodatkowo raw maximum crossing magnitude mg/dL. Ties nie crossing. L>U → interval summary danej pary/grupy INVALID z N_invalid i pierwotnym N; bez ujemnej width, sorting lub usunięcia wiersza. Pooled/macro aggregate zawierający invalid interval nie może pomijać go i udawać valid. Finite pinball/hit/crossing pozostają definiowalne. Pary (.25,.75),(.10,.90),(.02,.98) to50/80/96%;90% odmawia bez nowo zatwierdzonej metody. Persistence probabilistic metrics=N/A, bez syntetycznych identycznych kwantyli.

Wszystkie stosowne miary: wszystkie12h, pełne per-patient×h, micro, macro equal-patient, oba poziomy strata z N. Strata3: y<70,70≤y≤180,y>180; strata5: y<54,54≤y<70,70≤y≤180,180<y≤250,y>250. Tylko observed y, przed rounding. Macro=średnia metryk osób z n>0; nie zero za missing. Macro RMSE=mean(patient RMSE). W stratum jawna lista osób bez danych. All-horizon micro liczone bezpośrednio na S; all-horizon macro najpierw metryka ze wszystkich occurrences osoby, potem średnia osób. Różnice TFT−persistence point metrics tylko po tych samych keys. Brak średnich batch means, selection najlepszego h lub arbitrary clinical PASS. Pełna precyzja w artefakcie, rounding tylko prezentacyjny.

### Provenance i komunikaty

Input context musi wiązać kod/dirty/config/units/schema/role, model state/role/hash, normalizer+encoders i Q przed predict. Niezgodność wymaganej metadanej ma zatrzymać load/predict przed wykonaniem, gdzie jest to sprawdzalne bez deserializacji. Nie naruszać istniejącej ownership/compatibility boundary registry. State in-memory używany do forward również zgodny z context; sam poprawny hash niewykorzystanego pliku nie wystarcza.

Output manifest atomowo wiąże ordered keys, source-fixture/data hash, model/normalizer/config/definitions/numerical profile, raw predictions hash, metrics/denominators/failure counts, calibration status i role. Osobny failed/invalid record, nigdy valid pointer po awarii. Nie implementować arbitralnego nowego pełnego registry; wykorzystać istniejące primitives, rozszerzyć wąsko do evaluation context. W ramach tego tasku wyłącznie synthetic owned state, bez wczytania historical BEST. Brak latest/mtime fallback i implicit test access.

C17: usunąć SAFE/clinical accuracy target MET/acceptable i checkmarki kalibracji oparte na arbitralnych progach; liczby diagnostyczne mogą zostać z ograniczeniami. C13: weighting unchanged, raw q.5 nie gwarantuje nieważonej mediany. C14: raw/not_fitted, brak fitted-source; nie tworzyć kalibratora, nie deklarować rzeczywistego pokrycia. Nie wprowadzać nowych confidence labels lub progów bezpieczeństwa.

## 5. Obowiązkowa macierz acceptance

Każde ID zachowuje severity i per-finding DoD z planu. Minimalne testy poniżej trzeba przeprowadzić na rzeczywistej nowej funkcji, nie tylko na własnym oracle.

| Finding | Minimalny regression i warunek zamknięcia |
|---|---|
| C01-F01 BLOCKER | Corrupt y/keys/lengths→alignment error; coherent permutation i uneven batch PASS; exact bijection S. |
| C02-F01 HIGH | Dwie osoby/różne skale, niezależne observed-train log stats, real PF inputs/scale, no known-subject fallback; unseen odmowa; serialized mapping i nowa compatibility revision. |
| C03-F01 BLOCKER | NaN/±Inf y i każdy q; model gate oraz late PF fault osobno; finite-specific invalid przed publikacją, planned N zachowany. |
| C03-F02 BLOCKER | Coherent y=.1/20/400/401,p≤0 w N; y≤0 target-domain odmowa; helper MAE31/12 i MARD75% w odpowiednich fixture. |
| C03-F03 HIGH | Naturalne krótkie encoder/decoder, role boundary i returned lengths odmawiają; osobny evaluation W bez zmiany training index. |
| C04-F01 LOW | √250 w pełnej precyzji po zapisie/odczycie; formatter niezależny. |
| C05-F01 MEDIUM | Nierówne N,micro≠macro,missing patient,prawdziwe empty group; pełne point/probabilistic tabele i liczniki overlap. |
| C06-F01 MEDIUM | Wszystkie12h dla obu modeli; znaczniki i timestamps zgodne, wspólny W. |
| C07-F01 MEDIUM | Floating boundaries53.9/54/69.9/70/180/180.1/250/250.1; suma N strata=rodzic, empty groups jawne. |
| C08-F01 BLOCKER | Brak1 lookup przy poza tym spójnym context→baseline-specific invalid comparison; brak favorable subset. |
| C08-F02 MEDIUM | Coherent przebudowany encoder,age0/1/6 i invalid7/source missing/future; exact s*/value/flags, brak wymogu fresh origin. |
| C10-F01 HIGH | Q z modelu/artefaktu/output mismatch,duplicate,missing.5/endpoint→Q-specific odmowa; native Q PASS. |
| C11-F01 HIGH | Ties,any vs adjacent,magnitude,inverted pair; group/macro INVALID bez selekcji; pełne przekroje. |
| C12-F01 HIGH | Inclusive PICP.75,width2,hit.75,pinball_q25=.375;50/80/96,90 odmowa; wszystkie h/osoby/strata, N/A persistence. |
| C13-F01 MEDIUM limitation | Loss algebra bez backward,uniform49 vs70,factor2 i atom plateau; opis Fw,weighting bez zmian; status ACCEPTED LIMITATION. |
| C14-F01 MEDIUM deferred | Raw/not_fitted metadata i brak claimed calibration; CQR oracle zachowany tylko jako definicja; status NOT IMPLEMENTED/DEFERRED. |
| C17-F01 HIGH | Log/raport perfect,poor,empty,invalid,crossed bez safety/calibration guarantees; ograniczenia przy Clarke. |
| C18-F01 HIGH | Context/role/schema/Q/normalizer/hash mismatch z kontrolą etapu odmowy, owned state roundtrip, atomic invalid publication, raw→metric readback. |

Nie kopiować bezmyślnie starych testów: po C01 wiele fake y nie pasuje do source; after-fix mogą odmawiać z niewłaściwego powodu. Pure scalar tests używają helperów lub coherent fixtures, integration używa pełnej ścieżki. `must_reject` wymaga konkretnej przyczyny. Nie używać unconditional-fail `full_context`, p=0 jako empty ani `--only` starego runnera. Nie zmieniać historycznych wyników na xfail. Gdy nazwy/API się zmienią, udokumentować mapowanie stary reproducer→nowy test o tym samym expected behavior.

## 6. Budżet i bezpieczeństwo testów

Przed runem nowa jawna prerejestracja: CPU strict FP32,seed42,workers0,model.eval/no_grad,≤64 unikalne wybrane okna rzeczywiście podane modelowi,≤24 forward batch calls **łącznie na acceptance campaign**,0 backward/optimizer, watchdog600s per child. Reruns/aborted calls wliczać; zliczać także oddzielnie candidate index rows. Orientacyjna liczba unit checks nie jest celem sama w sobie. Nowy budżet wymaga aktywacji tego tasku; nie odziedziczyć pozornie niewykorzystanych calls starego audytu.

Float64 oracle atol1e−10/rtol1e−8; FP32 integration atol1e−5/rtol1e−5; keys/counts exact. Prerejestrować dtype porównania y source↔PF i unikać bitwise wymagań wobec źródłowego float64. Nie zwiększać tolerancji dla korzystnego wyniku; konieczna zmiana wymaga wcześniejszego uzasadnienia/akceptacji i zachowania wyniku pierwotnego.

Runner nadrzędny ma zapisywać rzeczywiste exit codes/times i durable counters również przy import error/signal/timeout. Nie uruchamiać drugiego procesu dopóki stan pierwszego nie jest ustalony. Hard watchdog ma zakończyć child w razie blokady natywnej; sam SIGALRM w Pythonie nie wystarcza jako gwarancja. Oddzielić error infrastruktury od production FAIL.

Guards real-data open i entrypoints przed ich użyciem; allowlist tylko synthetic owned fixtures/artifacts, dodatnia próba forbidden path odmawia przed open; zero zapisów poza własnym runem/testową konfiguracją. Nie czytać canonical parquet w celu sprawdzenia jego SHA — inherited evidence wystarcza. Zablokować main/train/load_test_data/Optuna/backward/optimizer. Sprawdzić state/grad przed/po. Role sentinel ma badać faktyczną izolację normalizer fit; calibration production nie istnieje, więc nie udawać tego testu.

Pure oracles/helper tests bez modelu, ograniczona prawdziwa integracja PF→evaluate→artefakt, coherent exact predictions jako osobna ścieżka. Nie broad discovery historycznych Stage B suites; zawierają optimizer steps. Dodatkowa regresja treningowa związana z C02, jeśli potrzebna do pełnej certyfikacji, jest osobnym taskiem/NOT RUN, bez potajemnego fit. Brak nowych testów GPU lub real performance.

## 7. Deliverables i Definition of Done całego tasku

Dostarczyć produkcyjne minimalne zmiany przypisane do ID, osobne acceptance tests/runner, prereg config oraz immutable synthetic run outputs. Przygotować `docs/STAGE_C_REMEDIATION_REPORT.md` z macierzą wszystkich18 findings, komendami, environment, commit/dirty/diff, config/source/fixture/key/prediction/model hashes, actual exits/counters, PASS/FAIL/ERROR/SKIP/XFAIL i remaining risks. Umieścić nowe źródła testów w repo, binarne/log artifacts w osobnym ignorowanym katalogu oznaczonym Stage C synthetic. Nie nadpisywać audytu ani planu, żeby dopasować kryteria.

Zamknięcie mechaniczne wymaga:

1. Wszystkie16 technical findings spełniają per-ID DoD z planu; czerwony test lub niewykonana kluczowa integracja oznacza OPEN/FAIL/BLOCKED, nie FIXED.
2. C13 opisany jako ACCEPTED LIMITATION, loss unchanged. C14 pozostaje NOT IMPLEMENTED/DEFERRED; jawne metadata DONE. Nie wymaga się real calibration i nie udaje jej wdrożenia.
3. Wspólny W/S, source semantics i provenance zachowane end-to-end; raw metrics z artefaktu odtworzone bez forward; brak test masking przez wcześniejszą inną bramkę.
4. Brak produkcyjnego nan_to_num/selective filtering/safety claims; wszystkie stosowne przekroje i statusy, brak silent checkpoint migration.
5. Guard positive controls,0 updates,state unchanged,budżet i rzeczywiste exit codes udokumentowane. Składnia/importy oraz stosowne testy i `git diff --check` wykonane w dozwolonym zakresie. Data/files/staged review bez pacjentów lub checkpointów przeznaczonych do commitu.
6. Trzy osobne werdykty: implementation, scientific validity, clinical interpretation. Poprawna implementacja i język nie dają empirycznego calibration/reliability PASS ani pełnej recertyfikacji training Stage B po zmianie normalizatora.

**Następnie STOP.** Przekazać raport do review. Nie uruchamiać treningu, Optuny, C-dev, kalibracji, real test evaluation ani kolejnego etapu bez osobnej decyzji użytkownika.
