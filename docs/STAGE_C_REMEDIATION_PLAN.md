# Stage C — remediation plan i review 18 findings

2026-09-14 · `nmd-stage-c-remediation-plan-1` · **PLAN READY / IMPLEMENTATION NOT STARTED**.

Zlecenie tej sesji obejmuje review i dwa dokumenty, nie implementację. Ten plan oraz [task](CODEX_STAGE_C_REMEDIATION_TASK.md) wymagają późniejszej jawnej aktywacji przez użytkownika. Nie zmieniono kodu produkcyjnego ani clinical weighting.

## 1. Wynik review i provenance

Review utrzymuje **18 findings: 4 BLOCKER, 7 HIGH, 6 MEDIUM, 1 LOW**. To **16 technical bugs/capability gaps** oraz **2 methodological/accepted scope limitations (C13-F01,C14-F01)**. Brak funkcji wymaganej przez zaakceptowany reporting contract jest luką techniczną; brak kalibratora nie jest powodem do jego automatycznego wdrożenia.

Źródło: [BASELINE_AUDIT_STAGE_C.md](BASELINE_AUDIT_STAGE_C.md), wiążący [execution contract](STAGE_C_EXECUTION_CONTRACT.md), plan i task C-core, AGENTS.md, baseline config i artefakty wskazane poniżej. Bieżący branch `research/baseline-audit`, HEAD `abdcb1abd51b48728972e45aca095468de17ef55`, clean na wejściu. Względem audytowego `4cc11d8fd7704cf3baeb4301a3b722374aa94fbd` przybyło11 wersjonowanych plików audytu (+1781 linii); kod produkcyjny bez zmian. Nie wykonano fetch ani ponownej walidacji zdalnego HEAD.

Odczytano obydwa pełne reproduktory,18 records findings, prerejestracje, execution summary, final verification oraz wszystkie evidence JSON/logs w trzech katalogach run. Zweryfikowano każdy hash artefaktu zapisany w per-run manifestach: **0 mismatch**; także production code hashes i wskazane lokalne PF source hashes: **0 mismatch**. Odczytano wszystkie tablice NPZ z `allow_pickle=False` i syntetyczne CSV; targety źródłowe zgadzają się z zapisanym y w FP32 (max abs3.051757815342171e−6 mg/dL). Keys/counts są exact, nie mylić tego z bitową równością źródłowego float64 i PF float32. `.pt` zweryfikowano bajtowo oraz strukturę archiwum; **nie deserializowano modelu ponownie**, wcześniejszy owned roundtrip pozostaje evidence run03.

W tej sesji **0 forward,0 backward,0 optimizer steps,0 nowych przebiegów testowego runnera**. To read-only review wcześniejszych dowodów, nie ponowny audyt wykonaniowy. Canonical parquet i trained checkpoints nieotwarte. Historyczne A/B PASS nie są świeżymi dowodami zamknięcia C02.

### Indeks dowodów

- `experiments/stage_c_synthetic_20260914/tests/run_stage_c.py` = **R1**, SHA `b87be8716600df18c46176e4f4184ab004a5fcc47e89fcab96431da2a51bc01b`.
- `experiments/stage_c_synthetic_20260914/tests/followup_stage_c.py` = **R2**, SHA `5bbb489950c5c0a3566a5d9a2a883e4c8b2b9f62fa4dc3a232b1131b919fe457`.
- `experiments/stage_c_synthetic_20260914/findings.json`: pełne ID/evidence/source locations/fixture-parent hashes.
- `experiments/stage_c_synthetic_20260914/configs/baseline_stage_c_audit.json`, `configs/followup_preregistration.json`, `execution_summary.json`, `static_source_hashes.json`, `final_verification.json` i README.
- `ml/models/stage_c_synthetic_20260914/run01`, `run02`, `run03`: manifest, results/partial, probe_captures, guard_events, production/console logs, real_metrics, window_keys, real_predictions.npz, synthetic_train/assessment.csv, synthetic_untrained.pt. Dodatkowo run03: normalizer_probe, real_corrupt_metrics, registry_compatibility_source. To **wyłącznie istniejące synthetic evidence**, mimo słowa „real” w nazwie oznaczającego rzeczywiste API PF.
- Pakiet poprzedniej sesji `outputs/stage_c_synthetic/SHA256SUMS.json` pozostaje indeksem paczki; bieżący review opiera kontrolę repo na per-run manifestach i source hashes. Nie zmieniono historycznych artefaktów.

Raport ma SHA `fe2e2ac40df28c62086d96a4cc095cb4c5b900ff10d9cfe670beffd09ca73a24`; kontrakt ma SHA `a7a631d8d1cd36cff1847d44878380ec2ca97680115abe08be9d6d42eb1d2610`. Ścieżki repo w tym planie są względne wobec `/home/radian/NeuroMetabolic Dashboard`.

## 2. Co dowody potwierdzają i czego nie wolno z nich wywnioskować

C01-F01 jest **brakiem ochrony granicy**, nie dowodem spontanicznego pomieszania PF predictions. C03-F01 ma zarówno sztuczne zwroty predict, jak i rzeczywistą injekcję po FiniteModel; naturalna finite gate modelu przechodzi. C08-F01 i C01-F01 współdzielą część reproduktora, ale mają odrębne obowiązki: alignment całego wyniku oraz kompletność baseline. C02-F01 jest potwierdzony emitted target_scale i lokalną ścieżką PF: categorical IDs są kodowane przed normalizacją, a dopasowany wcześniej normalizer ma string keys. Sama poprawność inverse math lub istnienie norm_ nie zamyka tego problemu.

NOT IMPLEMENTED dla C05/C06/C07/C11/C12/C18 potwierdza odczyt całej ścieżki evaluate→metrics, nie tylko nieobecność wybranych kluczy w teście. Przyszła implementacja może mieć inny schemat niż stare nazwy top-level; acceptance ma badać semantykę i liczby. C11 nie udowodnił, że produkcja raportuje ujemną width — width nie istnieje. C13 nie dowodzi błędu implementacji lossu ani wielkości empirycznej miscalibration. C14 nie wymaga calibrated model do zakończenia C-core.

### Ograniczenia starego harnessu wymagające nowego acceptance

1. `must_reject` łapie dowolny ValueError/RuntimeError: po dodaniu C01 test nonfinite, Q lub persistence może fałszywie przejść z błędem keys. W nowych testach sprawdzać **przyczynę, etap i brak publikacji**, a pozostałe wejścia utrzymać spójne.
2. R1 `full_context` kończy się bezwarunkowym AssertionError; po poprawnym filtrowaniu sam reproduktor też może upaść na założeniu, że short.any(). Zachować stary plik, napisać właściwy test nowego zachowania z niepełną fixture i kontrolą odmowy.
3. `empty_mask` używa finite p=0. Zgodnie z kontraktem powinno pozostać N12, nie być empty group. Oddzielić regression dla p=0 od prawdziwego pustego W/stratum.
4. Wiele fixed-prediction probes zmienia y bez przebudowy source frame/datasetu. To poprawne dla izolacji starego scoringu, ale po C01 trzeba użyć **spójnej source fixture albo bezpośredniego testu reducerów**, a osobno świadomie uszkodzonego alignment case.
5. Ffill probe zmienia source frame, nie buduje ponownie encoder input i sprawdza N, nie pełną lineage. Nie traktować go jako zamknięcia age/value consistency. Registry probe używa prostego dict z `resume-last`; nie testuje pełnej drogi BEST→evaluate ani calibrated lineage.
6. Stary test-sentinel to oracle separacji słowników i niezależny refit na tym samym train; nie dowodzi szczelności produkcyjnego fit/apply, którego brak. Nowa regresja ma rzeczywiście perturbować dane nieuprawnione do dopasowania normalizatora; bez fikcyjnego kalibratora.
7. Run01 zapisał wyniki, następnie miał OS exit143; run02 exit1, run03 exit1. Łącznie103 wykonane checks54 PASS/49 FAIL,57 różnych nazw30 PASS/27 FAIL;21 forwardów, nie13. Licznik follow-up pierwotnie pominął8 calls run01; korektę zachować. Nie przedstawiać powtórzeń jako nowych dowodów.
8. Python audit-hook nie jest gwarancją izolacji OS dla dowolnego natywnego loadera. Budget64 dotyczył12 wybranych okien modelu, a nie108 candidate index rows. W nowej prerejestracji rozdzielić te liczniki, egzekwować łączny budżet procesu nadrzędnego i actual exit, także przy błędzie importu/timeout. Nie używać uszkodzonego `--only` starych runnerów jako bramki.

Te uwagi nie zamykają ani nie unieważniają potwierdzonych findings. Są wymaganiami wiarygodnej regresji; stare czerwone wyniki pozostają nietknięte.

## 3. Kolejność i zależności

**Pierwsza grupa ryzyka G1:** C01-F01,C03-F01,C03-F02,C08-F01 oraz wszystkie HIGH: C02-F01,C03-F03,C10-F01,C11-F01,C12-F01,C17-F01,C18-F01. C08-F02 MEDIUM wchodzi wcześniej jako zależność persistence. Proste reducers/iteratory/strata można budować jako zależności HIGH; severity nie jest nakazem odkładania potrzebnej infrastruktury.

| Krok | Zakres i warunek przejścia |
|---|---|
| G0 — przygotowanie | Zamrozić nowy acceptance config, fixture schema, powody odmowy, budżet i provenance; sklasyfikować C13/C14 jako ograniczenia; guard positive controls. Bez zmiany lossu. |
| G1a — integralność wejścia | Szkielet input context C18, canonical W/S C01, długości/rola C03-F03. Zaplanować kontrakt modelu/Q/normalizatora. Żadnych metryk przed walidacją. |
| G1b — semantyka modelu | C02 normalization i C10 Q; rewizja kompatybilności C18, bez migracji. Przejście dopiero po zgodnych rzeczywistych PF wejściach/skalach/Q. |
| G1c — ważność pełnego porównania | C03-F01 finite,C03-F02 brak selekcji,C08-F01 pełna persistence z C08-F02 lineage. Nie wybierać pozytywnego podzbioru po awarii. |
| G1d — uncertainty | C11 raw crossing i C12 pinball/hit/PICP/width z poprawnymi podstawowymi reducerami. C17 usuwanie SAFE może być zrobione wcześniej niezależnie; końcowe teksty zweryfikować tu. |
| G2 — domknięcie raportu | C04 precision,C05 aggregation,C06 wszystkie h,C07 strata; **pełne przekroje C11/C12**, finalny output manifest C18 i język C17; dokumentacyjna część C13/C14. |
| G3 — review zamknięcia | End-to-end acceptance i readback artefaktu;16 technical findings muszą mieć dowód albo pozostać OPEN. Osobne statusy scientific/clinical, stop przed treningiem/C-dev. |

C11/C12 i C18 nie są FIXED po samym szkielecie G1: do zamknięcia wymagają G2. G1 ma pierwszeństwo w pracy i blokuje wiarygodne wyniki, ale **nie ma release/publikacji valid final evaluation między G1 a G2**. To jeden spójny task mechanicznej remediation, z małymi przeglądalnymi zmianami, nie sekwencja eksperymentów performance.

Zależności bez cyklu: `input schema → W/keys/lengths → normalizer/Q → finite/domain/persistence → reducers → all-horizon/patient/range + uncertainty → final manifest`. C18 podzielono na input i output, dlatego jego zależność od wyników nie blokuje początku. Model normalization i Q można opracować niezależnie od samych wzorów, lecz acceptance integracyjne musi spiąć obie strony. C17 zależy od semantyki opisu, nie od osiągnięcia dobrych metryk.

## 4. Review każdego findingu

Severity potwierdza wpływ w kontrakcie C-core, nie częstość naturalnych awarii ani bezpieczeństwo kliniczne. Status CONFIRMED/NOT IMPLEMENTED odziedziczony po zweryfikowanych dowodach. Każda poniższa remediation jest **propozycją do przyszłej aktywacji**, nie wykonaną zmianą.

### C01-F01 — Brak weryfikacji targetów względem kanonicznych keys

**Review severity:** BLOCKER utrzymany: brak bramki integralności może unieważnić całe porównanie. Fault injection potwierdza brak odmowy, nie spontaniczną awarię PF.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `CONFIRMED`. **Kolejność:** G1a. **Zależności:** C03-F03; C18-F01/input.

**Evidence:** `train_tft_population_v2.py:evaluate 2191–2198,2352–2388`; `run03:C01_target_key_mismatch`.

**Minimal remediation:** Przed predict utworzyć kanoniczny W i pełne S; po predict zweryfikować jednoznaczny, kompletny bijektywny związek keys↔x↔y↔output. Dopasować kolejność po keys, odrzucić duplicates/missing/extra i niezgodne targets/timestamps/lengths; nie korygować danych przez zamianę y na wygodną etykietę.

**Regression test:** R1 C01_uneven_batch/permuted_batches i R2 C01_target_key_mismatch. Nowe: jedna błędna y, zamiana wyłącznie y, brak/duplikat/obcy key, shift h, cross-subject przy tym samym time_idx; kontrola dodatnia permutuje cały rekord. Sprawdzać kod błędu alignment, nie dowolny ValueError.

**Definition of Done:** Każdy poprawny element S jest obecny raz; source y zgadza się w ustalonej tolerancji reprezentacji; keys/counts exact. Nieprawidłowy batch nie publikuje valid metrics. Real PF i nierówny batch przechodzą.

### C02-F01 — GroupNormalizer używa fallback zamiast statystyk osoby

**Review severity:** HIGH utrzymany: rzeczywisty PF używa wspólnego fallback dla znanych osób. Brak dowodu utraty jednostek lub jakości trained model; nie obniża to niezgodności kontraktu grupowego.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `CONFIRMED`. **Kolejność:** G1b. **Zależności:** C18-F01/input; wspólny model/dataset schema.

**Evidence:** `train_tft_population_v2.py:create_time_series_dataset 1678–1690; PF encoders.py:get_parameters 1295–1306; timeseries/_timeseries.py 2134`; `run02:C02_group_transform`, `run03:C02_subject_normalizer_parameters`, `run03:C02_distinct_inverse_math`.

**Minimal remediation:** Uzgodnić mapę raw subject→encoded group używaną przez fit, transform i get_parameters, z observed-train-only fitting. Wybrać najmniejszy wspierany przez PF sposób przekazania dopasowanych encoders/normalizatora; nie edytować site-packages ani naprawiać wyłącznie target_scale po forward. Sprawdzić także encoder_cont, dodane scale features i decoder inverse. Zachować log/center/estymator skali i brak refit na val.

**Regression test:** R1 C02_group_transform oraz R2 C02_subject_normalizer_parameters/distinct_inverse_math. Niezależne statystyki log(y) per osoba z właściwym ddof/epsilon tego stosu, dwie odmienne skale, permutacje nazw/kolejności/ID, observed-false sentinel i val sentinel; porównanie rzeczywistych PF wejść/target_scale. Fallback spy nie może zostać wywołany dla znanej osoby; unseen odmawia zgodnie z baseline.

**Definition of Done:** Własne statystyki każdej znanej osoby w danych PF, dokładnie jedna inverse i brak fit na brakujących/val danych. State/hash normalizatora+encoders przechodzi roundtrip. Nowa rewizja kompatybilności odmawia użycia starej semantyki; bez migracji/reinterpretacji starego checkpointu. Wpływ naukowy opisany.

### C03-F01 — Zastępowanie NaN/Inf i usuwanie uszkodzonych targetów

**Review severity:** BLOCKER utrzymany: output sanitization i usuwanie nonfinite targetów faktycznie zmienia wynik. Native FiniteModel działa; obejście po jego wyjściu pozostaje odrębną granicą.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `CONFIRMED`. **Kolejność:** G1c. **Zależności:** C01-F01; C03-F03; C10-F01.

**Evidence:** `train_tft_population_v2.py:evaluate 2205–2226,2247–2256`; `run02:C03_prediction_nan/posinf/neginf`, `run02:C03_target_nan`, `run03:C03_real_predict_late_corruption`, `run03:C03_native_nonfinite_model_gate`.

**Minimal remediation:** Usunąć nan_to_num z ewaluacji. Kontrolować finite całego wymaganego raw output i observed y przed redukcją; invalid sample unieważnia wynik modelu/porównanie, zachowując planowane N. Rozdzielić brak target_observed od uszkodzonego observed target. Kontrole finite modelu zachować.

**Regression test:** R1 prediction_nan/posinf/neginf,target_nan; R2 native_nonfinite_model_gate i real_predict_late_corruption. Każdy q, nie tylko q.5; NaN/±Inf target na ważnej pozycji; źródło i keys poza injekcją zgodne. Asercja finite-specific i brak valid artefaktu, przy pełnym planned N.

**Definition of Done:** Żadnej sanitizacji lub usuwania wierszy dla nonfinite; trwały invalid/failure record. Native gate i downstream gate osobno potwierdzone. Prawdziwy padding testowany wyłącznie w dozwolonym pomocniczym helperze; główny W pełny.

### C03-F02 — Nieuzgodniona selekcja przez y i predykcję

**Review severity:** BLOCKER utrzymany: pred-dependent masks i y20/400 skrywają błędy również w zdrowych finite tensorach.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `CONFIRMED`. **Kolejność:** G1c. **Zależności:** C01-F01; C03-F01; C03-F03.

**Evidence:** `train_tft_population_v2.py:evaluate 2247–2256,2282–2291`; `run02:C03_finite_negative_retained`, `run02:C03_targets20_400_401`, `run02:C04_zero_target_invalid`, `run03:C04_nearzero_positive`, `run03:C03_finite_zero_retained`.

**Minimal remediation:** Zastąpić maski pred>0 i y∈(20,400) stałą kwalifikacją W. Wszystkie finite p≤0 pozostają błędem z flagą; każde observed y>0 wnosi wkład bez clamp; y≤0 unieważnia ewaluację, nie redukuje W.

**Regression test:** R1 negative_retained,targets20_400_401,zero_target_invalid; R2 nearzero_positive,finite_zero_retained. Spójne fixtures source/returned y i osobne unit reducers: y=.1/20/400/401, p<0/0. Dla [20,400,401,100×9] i [30,390,390,100×9]: N12,MAE31/12. Nearzero: MARD75%; zero/ujemny y→target-domain error.

**Definition of Done:** W i mianowniki pozostają niezależne od predykcji i zakresów; TFT/persistence korzystają z tego samego S. Wyeliminowane clamps i ukryte selekcje w point i probabilistic path. Empty stratum nie utożsamiany z usunięciem złych prognoz.

### C03-F03 — Guard ewaluacji nie wymaga pełnego 48/12

**Review severity:** HIGH utrzymany. Naturalny index zawiera96/108 krótkich encoderów; controlled short decoder dowodzi braku kontroli returned lengths. Nie wykazano, że PF samo tworzy błędny padding dla pełnego W.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `CONFIRMED`. **Kolejność:** G1a. **Zależności:** C18-F01/input.

**Evidence:** `observed_windows.py:assert_observed_evaluation 73–78; train_tft_population_v2.py 1683–1684,1691–1693,2244–2245`; `run02:C03_full48_guard`, `run02:C03_decoder_lengths_padding`.

**Minimal remediation:** Wydzielić kwalifikację głównej ewaluacji: encoder48,decoder12,regular timestamps,observed decoder i właściwa rola. Odrzucenia przed predict jawnie policzyć. Guard głównego evaluator odmawia datasetu/indexu lub returned lengths niezgodnych z W. Nie zmieniać treningowego Stage A window filter/min lengths ani budowania val do early stopping bez osobnego protokołu.

**Regression test:** R1 full48_guard i decoder_lengths_padding wymagają nowych asercji zachowania. Sprawdzić naturalne krótkie encodery, krótkie decodery i boundary crossing. Oddzielny test potwierdza niezmienność treningowego indexu przed/po. Perturbacja padding nie zmienia pomocniczej masked metric, lecz nie rozszerza W.

**Definition of Done:** W zawiera wyłącznie48/12 i znany flow candidates→eligible/excluded. Niepełne returned lengths odmawiają. Pełny W spełnia Stage A; nie wprowadzono ostrzejszego filtra train ani nowego splitu.

### C04-F01 — Zaokrąglanie przed zapisem artefaktu metryk

**Review severity:** LOW utrzymany: rounding zmienia precyzję artefaktu, nie główny werdykt. Naprawić razem z reducerami, bez osobnego dużego refaktoru.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `CONFIRMED`. **Kolejność:** G2/prymitywy mogą powstać w G1d. **Zależności:** C03-F02; C18-F01/output.

**Evidence:** `train_tft_population_v2.py:evaluate 2273–2275,2311,2528–2533`; `run02:C04_reporting_precision`.

**Minimal remediation:** Zwracać/zapisywać pełnoprecyzyjne scalars; rounding wyłącznie w formatterze. Akumulować raportowe miary w stabilnej precyzji, z kontrolą overflow, bez zmiany training loss.

**Regression test:** R1 reporting_precision: RMSE√250=15.811388300841896 w roundtrip JSON, nie15.8114. Sprawdzić też MARD i proportions, podział batchy oraz oddzielnie prezentację.

**Definition of Done:** Raw artefakt spełnia tolerancję float64 oracle; ponowny odczyt/obliczenie zachowuje wynik. Żaden pośredni round nie wpływa na sumy, macro ani differences.

### C05-F01 — Brak per-patient, micro/macro, bias i jawnych pustych grup

**Review severity:** MEDIUM utrzymany jako brak wymaganej funkcji raportowej; nie klasyfikować jako accepted limitation. Pełne przekroje są częścią bramki zamknięcia uncertainty HIGH.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `NOT IMPLEMENTED`. **Kolejność:** G2/agregator potrzebny do G1d. **Zależności:** C01-F01; C03-F02; C04-F01; dla uncertainty także C11-F01/C12-F01.

**Evidence:** `train_tft_population_v2.py:evaluate 2244–2275,2294–2312,2632`; `run02:C05_patient_micro_macro_bias`, `run02:C05_unequal_patient_oracle`, `run02:C04_empty_group`, `run02:C15_overlap_denominators`.

**Minimal remediation:** Wspólny agregator MAE/RMSE/bias/MARD i stosownych probabilistic metrics: per-patient, micro/global, equal-patient macro, wszystkie h/strata; dokładne N occurrences/windows/unique(subject,target_timestamp)/subjects. Empty→null,N0,reason,lista missing osób. Macro RMSE=mean(patient RMSE).

**Regression test:** R1 unequal_patient_oracle,patient_micro_macro_bias,overlap_denominators. Nierówne liczności, różne błędy i brakująca osoba/stratum; mikro RMSE√250 vs macro(10+√300)/2. Nowy prawdziwy empty-group test — nie stara fixture p=0. Porównać sumy na całym zbiorze z różnymi batch partitions.

**Definition of Done:** Pełne tabele z count i denominators, bez średnich batch means lub zer za brakujące osoby; all-horizon podsumowanie wtórne. Per-patient probabilistic metrics tak samo kompletne; żaden brakujący panel nie znika.

### C06-F01 — Raport obejmuje tylko pięć z dwunastu horyzontów

**Review severity:** MEDIUM utrzymany: brak7 horyzontów i persistence poza60m ogranicza kontrakt raportowania.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `NOT IMPLEMENTED`. **Kolejność:** G2/iteracja h potrzebna do G1d. **Zależności:** C01-F01; C03-F03; C05-F01.

**Evidence:** `train_tft_population_v2.py:evaluate 2229–2235`; `run02:C06_all12_horizons`.

**Minimal remediation:** Iterować po wszystkich12 h=5…60 na tym samym W, obliczać point metrics obu modeli i probabilistic TFT. Wyróżnienia15/30/60 wyłącznie prezentacyjne.

**Regression test:** R1 all12_horizons z unikalnym znacznikiem h: index0/2/5/11→5/15/30/60; wszystkie12 pól/wierszy. Timestamps muszą odpowiadać origin+h; wspólne N obu modeli.

**Definition of Done:** Każdy h obecny dla każdej stosownej agregacji; brak globalnej stałej h ukrywającej niezgodny dataset; all-horizon nie zastępuje krzywej.

### C07-F01 — Brak wymaganych zakresów glikemii

**Review severity:** MEDIUM utrzymany: brak strata to luka implementacji zaakceptowanego kontraktu, nie wybór progów do ponownego strojenia.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `NOT IMPLEMENTED`. **Kolejność:** G2/grupy potrzebne do G1d. **Zależności:** C03-F02; C05-F01; C06-F01.

**Evidence:** `train_tft_population_v2.py:evaluate 2117–2632 (pełna ścieżka zwracanego metrics)`; `run02:C07_strata_reporting`, `run02:C07_strata_boundaries`.

**Minimal remediation:** Zaimplementować dwa zatwierdzone poziomy strata wyłącznie według observed y, przed rounding; bez usuwania skrajnych y. Dodać każdy wymagany przekrój i N.

**Regression test:** R1 strata_boundaries/strata_reporting:53.9/54/69.9/70/180/180.1/250/250.1, małe dodatnie y; rozłączność i wyczerpanie S dla3 i5 grup, identyczny przydział modeli i empty stratum.

**Definition of Done:** Suma N strata równa rodzicowi; wszystkie point/probabilistic metrics i osoby/horyzonty zdefiniowane zgodnie z null/invalid policy. Brak klinicznego progu PASS.

### C08-F01 — Persistence może przejść na niepełny podzbiór

**Review severity:** BLOCKER utrzymany; częściowo wspólna przyczyna z C01-F01. Stary probe przesuwa returned origin poza W, więc nowa bramka alignment może zatrzymać go wcześniej i nie dowiedzie naprawy samego persistence resolvera.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `CONFIRMED`. **Kolejność:** G1c. **Zależności:** C01-F01; C03-F03; C03-F02; C08-F02.

**Evidence:** `train_tft_population_v2.py:_verify_time_idx_alignment 2063–2085; evaluate 2436–2503,2528–2533`; `run02:C08_missing_one_of12`, `run02:C08_missing_decoder_index`.

**Minimal remediation:** Persistence ma kompletny key set W i12h; każdy brak source lub wynik lookup unieważnia pełne porównanie. Usunąć progi90%/50% i ratowanie raportu korzystnym intersection. Ewentualny istniejący diagnostyczny subset oznaczyć odrębnie; nie budować nowego subset reporting jako zakresu naprawy.

**Regression test:** R1 missing_one_of12/missing_decoder_index; dodać odrębną kontrolę błędu persistence resolver przy poprawnych kanonicznych keys/source envelope, aby C01 nie maskował ścieżki. Assert baseline-specific error. Ramp/stały sygnał/ffill i dwie osoby mają N i ordered-key hash identyczne z TFT.

**Definition of Done:** Brak1 lookup→comparison INVALID z planned N, bez normalnego improvement. Brak baseline nie usuwa eligible window; poprawny baseline przechodzi całą ścieżkę i wszystkie h.

### C08-F02 — Brak lineage ostatniej obserwacji persistence

**Review severity:** MEDIUM utrzymany: brak provenance źródłowej obserwacji, nie dowód błędnej wartości dla poprawnego ffill. Wcześniejszy test ffill sprawdzał zasadniczo N, nie pełną zgodność encoder input.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `NOT IMPLEMENTED`. **Kolejność:** G1c — zależność blockera. **Zależności:** C03-F03; C01-F01.

**Evidence:** `train_tft_population_v2.py:evaluate 2386–2388,2441–2458`; `run02:C08_ffill_origin`, `run02:C08_persistence_two_subject_ramp`.

**Minimal remediation:** Z source frame wyznaczyć s*=ostatnie observed CGM w48-bin context dostępne w origin; zapisać s*,age_bins,ffill. Wymagać age≤6 i zgodności wartości z dozwolonym końcem encodera; bez future/nearest lookup i bez wymagania age0.

**Regression test:** R1 ffill_origin/persistence_two_subject_ramp zastąpić rozszerzeniem używającym ponownie zbudowanego spójnego datasetu. Age0/1/6, brak source,age7,out-of-context,future pomiar, inna osoba, niespójny CGM encodera. Potwierdzić exact value i s*, nie tylko N.

**Definition of Done:** Każde w∈W ma udokumentowany source i12 identycznych point predictions mg/dL. Dozwolony ffill nie zmniejsza W; invalid lineage blokuje comparison.

### C10-F01 — Oś kwantyli odczytywana z globalnej listy

**Review severity:** HIGH utrzymany: metryka może użyć niewłaściwej osi jako q.5. Natywna poprawna lista nie jest dowodem odmowy mismatch.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `CONFIRMED`. **Kolejność:** G1b. **Zależności:** C18-F01/input; model/dataset contract C02-F01.

**Evidence:** `train_tft_population_v2.py:evaluate 2218–2225,2308–2311`; `run02:C10_model_quantile_mismatch`.

**Minimal remediation:** Odczytać Q z zweryfikowanego modelu/artefaktu i porównać z kontraktem/output K przed scoring. Wymagać ordered/unique/in(0,1), dokładnie zatwierdzonego Q; brakujące/permuted Q odmawia, nie sortuje. Point index wyznaczony po wartości. Unit label sprawdzony z provenance, bez heurystyk po amplitudzie.

**Regression test:** R1 model_quantile_mismatch/interval_endpoint_oracle; model Q, checkpoint Q,output axis width, brak.5,duplicate,permutation,out-of-range,wymagany endpoint. Positive native Q, dokładnie jedna inverse, tuple weight nie target_scale.

**Definition of Done:** Żadne relabelowanie ani globalny index bez weryfikacji; mismatched model nie produkuje metryk. Tylko central50/80/96,90% odmówione bez nowej metody.

### C11-F01 — Crossing i odwrócone przedziały bez diagnostyki

**Review severity:** HIGH utrzymany jako brak diagnostyki niezbędnej do interpretacji raw intervals. Brak produkcyjnych intervals nie oznacza, że stwierdzono ujemną raportowaną width.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `NOT IMPLEMENTED`. **Kolejność:** G1d; zamknięcie pełnych przekrojów w G2. **Zależności:** C01-F01; C03-F01/F02; C10-F01; C05-F01.

**Evidence:** `train_tft_population_v2.py:evaluate 2306–2337`; `run02:C11_crossing_reporting`, `run02:C11_crossing_ties_oracle`.

**Minimal remediation:** Raw any/adjacent crossing i max magnitude mg/dL; ties nie crossing. Zachować kolejność kwantyli. Dla grupy/pary z L>U interval summary INVALID z N_invalid i oryginalnym N; nie sortować, nie usuwać tych wierszy. Pozostałe definiowalne finite scores pozostają dostępne.

**Regression test:** R1 crossing_ties_oracle/crossing_reporting; noncrossing,ties,jeden/wiele crossed adjacent,przecięcie tylko innej pary niż badana. Pary z inversion oraz grupy zawierające choć jeden inversion; zgodność licznika any vs adjacent.

**Definition of Done:** Wyniki i statusy dla wszystkich h/osób/strata/micro/macro. Invalid patient interval nie znika po cichu ze średniej macro; agregat obejmujący nieważny składnik również jawnie INVALID, diagnostyka z pełnymi licznikami.

### C12-F01 — Brak PICP, width i nieważonego pinball

**Review severity:** HIGH utrzymany: obecne marginal hits nie stanowią interval coverage/width ani reporting pinball. Wdrożenie uzgodnionych wzorów nie wymaga wyboru kalibratora.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `NOT IMPLEMENTED`. **Kolejność:** G1d; zamknięcie pełnych przekrojów w G2. **Zależności:** C10-F01; C11-F01; C03-F01/F02; C04-F01; C05-F01/C06-F01/C07-F01.

**Evidence:** `train_tft_population_v2.py:evaluate 2306–2337`; `run02:C12_interval_reporting`, `run02:C12_interval_pinball_oracle`, `run02:C10_interval_endpoint_oracle`.

**Minimal remediation:** Dodać nieważony pinball bez factor2 per q,hit i hit−q oraz PICP/width dla50/80/96 z inclusive endpoints. Stosować S i wspólny agregator; persistence point-only N/A. Jawnie raw,nie calibrated.

**Regression test:** R1 interval_pinball_oracle/interval_reporting/endpoints: inclusive PICP.75,width2,hit.75,pinball_q25=.375; poza endpoints,inverted pair,empty. Loss training bez zmiany. Niewielkie fixture full crossing h×patient×range porównane do niezależnego float64 oracle.

**Definition of Done:** Kompletny reporting matrix, raw+status+units+N i roundtrip zgodny z oracle; bez udawanej90%, sortowania i calibration fit. Partial implementation tylko h60 nie zamyka HIGH.

### C13-F01 — Clinical weighting zmienia estimand kwantyli

**Review severity:** MEDIUM zachowany jako waga ograniczenia interpretacyjnego, nie severity defektu lossu. ACCEPTED LIMITATION obecnego protokołu: poprawne target-dependent weighting nie jest nominalnym nieważonym quantile.

**Klasyfikacja:** methodological / accepted limitation; audit evidence `CONFIRMED`. **Kolejność:** G0 ustalenie/G2 opis; badanie objective DEFERRED. **Zależności:** C17-F01; C12-F01 dla unweighted reporting.

**Evidence:** `train_tft_population_v2.py:ClinicalQuantileLoss 882–944`; `run02:C13_weighted_objective`, `run02:C13_discrete_quantile_plateau`, `run03:C13_all_q_weighted_CDF`.

**Minimal remediation:** Zachować factor2,hypo<70,weight2.5 i redukcję. W metadata/raporcie opisać Fw oraz raw q.5 bez gwarancji nieważonej mediany. Brak zmiany objective lub weight; future variant tylko po osobnym zleceniu.

**Regression test:** R1 weighted_objective/discrete_quantile_plateau i R2 all_q_weighted_CDF zachować. Assert loss algebra bez backward, uniform mediana49 vs70,factor2 zachowuje optimum, atom plateau. Metadata nie nazywa wyniku calibrated median.

**Definition of Done:** Część dokumentacyjna może mieć DONE, ale finding ma ACCEPTED LIMITATION — WEIGHTING UNCHANGED, nie FIXED przez zmianę lossu. Empiryczny wpływ i objective comparison NOT EVALUATED.

### C14-F01 — Kalibracja fit/apply i jej lineage nie są zaimplementowane w baseline

**Review severity:** MEDIUM zachowany jako jawna luka capability. NOT IMPLEMENTED / DEFERRED jest akceptowanym ograniczeniem zakresu C-core, nie technicznym blockerem mechanicznej remediation; nie oznacza empirycznie wystarczającej kalibracji.

**Klasyfikacja:** methodological / accepted limitation; audit evidence `NOT IMPLEMENTED`. **Kolejność:** G0 status/G2 jawne metadata; calibration DEFERRED. **Zależności:** C18-F01/output; C17-F01; przyszłe oddzielne role danych.

**Evidence:** `evaluate 2117–2632; main 2636–2657; build_model 1838–1873; baseline_training.py; checkpoint_registry.py`; `run02:C14_CQR_order_statistic`, `static calibration call-path inventory`.

**Minimal remediation:** Wyłącznie jawny calibration status raw/not_fitted i brak fitted source/role; nie tworzyć kalibratora. Oddzielny przyszły plan metody,split lineage,crossing policy i oceny dependent data po akceptacji.

**Regression test:** Zachować CQR order statistic oracle (k>n→∞,ties,negative score) jako test definicji, nie produkcji. Nowy test wymaga zgodnego raw metadata i braku deklaracji fitted/calibrated. Brak kalibratora nie może być testowany fikcyjnym PASS fit/apply.

**Definition of Done:** Zamknięty obowiązek jawnego statusu; implementacja kalibracji pozostaje NOT IMPLEMENTED/DEFERRED, real coverage NOT EVALUATED. Nie wymagać real calibration do zamknięcia technicznych16 findings.

### C17-F01 — Arbitralne metryki nadają etykietę klinicznego bezpieczeństwa

**Review severity:** HIGH utrzymany: odtwarzalne „clinically SAFE” na perfect invented predictions to nieuprawniona deklaracja; to błąd komunikacji, a nie nowa decyzja terapeutyczna.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `CONFIRMED`. **Kolejność:** G1 — może rozpocząć się niezależnie; integracja G2. **Zależności:** C13-F01/C14-F01 interpretacja; nie czekać na metryki, aby usunąć SAFE.

**Evidence:** `train_tft_population_v2.py:evaluate 2296–2302,2315–2337`; `run02:C17_clinical_language`.

**Minimal remediation:** Usunąć automatyczne clinical SAFE/accuracy target MET/acceptable i kalibracyjne checkmarki oparte na arbitralnych progach. Pozostawić surowe diagnostyczne liczby, jednostki i ograniczenia; nie zastąpić ich innym progiem. Clarke jako legacy diagnostic bez certyfikacji pełnych granic.

**Regression test:** R1 clinical_language i zachowane5 interior Clarke examples. Przechwycić log/raport/metadata dla perfect,poor,empty,nonfinite,crossed; żadnych gwarancji coverage/safety/confidence. Assert także poprawne raw/within-subject/conditional-on-eligibility labels.

**Definition of Done:** Wszystkie kanały nowej ewaluacji bez nieuzasadnionych safety/calibration claims. Full Clarke geometry oraz clinical reliability nadal NOT EVALUATED, nie „naprawione” zmianą tekstu.

### C18-F01 — Evaluate nie wiąże provenance modelu, keys i metryk

**Review severity:** HIGH utrzymany: registry helper i owned roundtrip nie dowodzą linked evaluation provenance. Naprawa musi być funkcjonalną bramką, nie dodaniem kilku pól do dict.

**Klasyfikacja:** technical bug / required capability gap; audit evidence `NOT IMPLEMENTED`. **Kolejność:** G1a input / G2 final artifact — przekrojowe. **Zależności:** Input: niezależny schema szkic; final: C01,C02,C03,C08,C10,C11,C12 oraz G2.

**Evidence:** `train_tft_population_v2.py:evaluate 2117–2632; checkpoint_registry.py:verified 192–243`; `run02:C18_evaluation_provenance`, `run03:C18_owned_untrained_roundtrip`, `run03:C18_registry_compatibility_controls`.

**Minimal remediation:** Wymagać zweryfikowanego evaluation context przed predict: code/schema/units/protocol/role/Q/normalizer+encoders/model ownership+hash i źródło danych. Po scoring atomowo związać raw predictions,ordered keys,config/definitions,counts/status i wynik. Własny synthetic-untrained context oddzielić od produkcyjnej roli BEST; brak latest fallback lub cichego przyjmowania unknown/mismatch.

**Regression test:** R1 evaluation_provenance,R2 owned_untrained_roundtrip/registry_compatibility_controls są punktem wyjścia. Dodać rzeczywistą ścieżkę loader/verified context przed deserializacją (counter), perturbacje pojedynczych pól/hash/schema/Q/role/normalizer oraz output tamper; roundtrip raw→metrics bez model forward. Nie polegać wyłącznie na arbitralnym dict w resume-last.

**Definition of Done:** Invalid context zatrzymany przed predict/deserializacją tam, gdzie możliwe; invalid output przed publikacją valid. Artefakt wiąże realny model state i parametry użyte w predict, a nie obcą sidecar. Statusy/denominators i hashes odtwarzalne; historical semantics odmawiają bez silent migration.

## 5. Niezmienne kontrakty i decyzje metodologiczne

Nie wybierać ponownie definicji ustalonych w execution contract: within-subject temporal, W pełne48/12 na pięciominutowej siatce, wszystkie decoder targets obserwowane, ffill≤6 w encoderze, S=W×12 h, per-target forecast occurrences, raw Q, unweighted reporting, zakresy glikemii. Liczniki zależności (occurrences vs unique timestamps) służą opisowi estymandu; nie dodawać IID CI/p-values ani resamplingu.

Normalizacja C02 jest **mechaniczną korektą realizacji przyjętego GroupNormalizer**, nie decyzją o nowym typie normalizatora. Wdrożenie zmienia jednak liczbową funkcję modelu i kompatybilność dawnych artefaktów. Minimalny plan: nowa jawna rewizja normalizer/evaluation contract, brak silent migration i brak użycia starych wag pod poprawioną semantyką. Dokładna nazwa wersji jest detalem technicznym. Jeśli naprawa wymaga zmiany estymatora, obserwowanych danych fit, architektury, training windows lub splitów, zatrzymać tę część i przedstawić decyzję; kontynuować niezależne prace. Nie zmieniać baseline_v1 historycznie w miejscu, żeby ukryć różnicę; nową konfigurację wersjonować jawnie dopiero w aktywowanej remediation.

Kwalifikacja ewaluacji48/12 nie daje zgody na zmianę walidacyjnego objective/early stopping Stage B albo ograniczenie train do48/12. Evaluation W należy budować osobno od treningowego indexu; porównanie indexów przed/po musi wykazać tę separację. Nie twierdzić, że po zmianie wyniki są bezpośrednio porównywalne z dawnymi.

C13: pozostaje `ClinicalQuantileLoss`, factor2, threshold70 i weight2.5. Udokumentowany Fw ma ważoną medianę49 dla uniform[0,140], nieważona70; nie uruchamiać objective variants. C14: raw intervals mają jawny status not_fitted. Wybór CQR/innej metody, shrinkage, crossing postprocessing, calibration/selection/assessment boundaries oraz wymienność/bloki pozostają **DEFERRED — osobna decyzja użytkownika**, bez fit na real data.

Clarke full boundaries, empiryczna kalibracja/reliability, real representativeness/bias i unseen-patient performance pozostają NOT EVALUATED. Istniejący optional ARIMA nie jest wymaganym nowym modułem: zachować causal history i jawny N/failure; nie instalować statsmodels, nie uruchamiać real fit ani wymagać ARIMA do zamknięcia persistence. Nie modyfikować backend/frontend/XAI w tym tasku.

## 6. Przyszła walidacja i bramka zakończenia

Nowe acceptance mają trzy warstwy: (A) niezależne float64 oracles dla wzorów i grup, (B) produkcyjne helpery z coherent fixtures i fault-specific errors, (C) prawdziwe PF→predict→evaluate→artifact na nietrenowanym małym TFT, bez stubbingu całej ścieżki. Unit test matematyk nie ma tworzyć TFT. Pure helper tests nie zastępują integracji.

Przed pierwszym runem nowa prerejestracja: FP64 atol1e−10/rtol1e−8, integration FP32 atol1e−5/rtol1e−5, keys/counts exact;≤64 unikalne **wybrane okna podawane modelowi**,≤24 forward batch calls łącznie na kampanię acceptance (także aborted runs),0 backward/optimizer; watchdog600s na child + twarde zakończenie i actual exit z parent. Liczbę candidate index rows, kolejne przebiegi i wszystkie wykorzystane calls raportować odrębnie. To proponowany nowy budżet przyszłego tasku, nie pozostałe3 calls starego audytu. Zwiększenie budżetu/progów wymaga wcześniejszego jawnego uzasadnienia/akceptacji, nie rerun-until-green.

Guard izolacji przed loaderami, dodatnia próba odmowy real data, blokada main/train/Optuna/backward/optimizer, state hash przed/po, no_grad/eval i CPU strict. Nie uruchamiać broad A/B test discovery, bo obejmuje trening. Jeśli wymagane po C02 training replay/checkpoint training-regression wykracza poza0 optimizer, zapisać NOT RUN i osobny task; mechaniczny PASS C-core nie odnawia pełnej certyfikacji treningowej Stage B.

Finalny raport remediation ma zawierać macierz wszystkich18 ID:16 technical FIXED tylko po dowodzie, C13 ACCEPTED LIMITATION, C14 NOT IMPLEMENTED/DEFERRED. Dokumentacyjne obowiązki ograniczeń mogą być DONE bez zmiany statusu empirical. Trzy werdykty oddzielnie: implementation PASS/FAIL, scientific claims ograniczone estymandem i brakiem real evaluation, clinical interpretation poprawność języka przy reliability NOT EVALUATED. Nie ogłaszać całego Stage C empirycznie PASS.

Dostarczyć dokładne komendy/source commit+dirty/diff/hashes, prereg, fixtures z pojedynczymi perturbacjami, per-test status i przyczynę odmowy, modele/keys/predictions/metrics/config hashes, actual subprocess exit/time/counters, listę plików i remaining risks. Oryginalne audit docs/config/runners/logs pozostają immutable. Żadnego automatycznego commitu/push ani otwarcia testu. Zatrzymać się na raporcie do research review.

## 7. Wynik tej sesji

Przygotowano wyłącznie ten plan i `CODEX_STAGE_C_REMEDIATION_TASK.md`. Potwierdzenie review ma charakter odczytu i kontroli artefaktów, nie świeżego wykonania testów. Implementacja, trening, Optuna, real-data evaluation i calibration fit **nie zostały rozpoczęte**.
