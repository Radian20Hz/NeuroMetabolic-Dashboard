# Stage C — execution contract

2026-09-14 · `nmd-stage-c-contract-1` · CONTRACTS RESOLVED / EXECUTION NOT STARTED.

Użytkownik zlecił zamknięcie decyzji przed audytem. Niniejszy dokument rozstrzyga D1–D2 z STAGE_C_AUDIT_PLAN.md i jest wiążącym oczekiwaniem dla CODEX_STAGE_C_TASK.md. Nie twierdzi, że produkcja już spełnia kontrakt. Odstępstwa mają zostać findings, nie automatyczną remediation. Następna sesja może wykonać C-core bez ponownego wyboru poniższych zasad; ta sesja nie aktywuje wykonania (D0).

Przeczytano plan, task, BASELINE_AUDIT_STAGE_A_REMEDIATION, STAGE_B_FINALIZATION_REPORT i configs/baseline_v1.json. Stage A PASS; Stage B PASS według aktualizacji zamykającej callback restore, a nie zachowanego historycznego FAIL. Nie uruchomiono żadnych testów lub modelu podczas ustalania kontraktu.

## 1. Populacja ewaluacji

Oficjalna ewaluacja Baseline v1.0 dotyczy within-subject temporal forecasting, context48 regularnych kroków pięciominutowych i pełnego decodera12 kroków (h=5,10,…,60 min). To ocena warunkowa na kwalifikację Stage A, nie dowód jakości na całym strumieniu ani generalizacji na nowych pacjentów.

Zbiór W ustalamy przed predykcjami z indeksu Stage A i danych źródłowych właściwej roli splitu:

- jedna osoba, jednoznaczny origin t będący końcem encodera, regularne timestamps, bez duplikatów;
- kompletny encoder zgodny z Stage A: CGM obserwowany lub przyczynowy ffill≤6 kroków, bez nierozwiązanych braków;
- wszystkie12 decoder targets rzeczywiście obserwowane: target_observed sprzed ffill=True; żaden placeholder, ffill lub inna imputacja nie jest etykietą;
- wszystkie target timestamps t+h należą do zadeklarowanej roli ewaluacji; brak przecinania jej granicy decoderem.

Jeżeli choć jeden decoder target jest nieobserwowany, wyklucza się **całe okno**, również jego obserwowane punkty, dla obu modeli. Decoder krótszy niż12 nie wchodzi do głównej ewaluacji. Krótsze decodery dozwolone w treningu Stage A pozostają bez zmian; syntetyczne testy długości/padding sprawdzają odmowę lub poprawne maskowanie w pomocniczych funkcjach, nie rozszerzają W. Nie wolno ratować niepełnych okien odrębnymi mianownikami horyzontów w głównym raporcie.

Jednostka S: (subject, window_id, origin_timestamp, target_timestamp, horizon_minutes), gdzie S=W×{5,…,60}. Każdy model ma dokładnie te same keys, targety i N. Kolejność manifestu kanoniczna; różna kolejność batchy wymaga jawnego dopasowania po keys. Wystąpienia wspólnego timestampu z różnych origin/h są odrębnymi zadaniami prognostycznymi; nie są niezależnymi pomiarami. Raportuj N occurrences, N windows, N unique (subject,target_timestamp) i N subjects.

Nie twórz nowych filtrów według błędu, predykcji lub zakresu glikemii. Finite pred≤0 nadal wnosi błąd i ma flagę fizjologicznej niewiarygodności. NaN/Inf, niepoprawny target≤0, brak wyniku lub lookup persistence, mismatch keys/units oznacza INVALID odpowiedniego wyniku/porównania i finding, bez nan_to_num, clamp lub usuwania próbki. Liczebność W nie zmienia się przez awarię modelu. Diagnostyczny subset musi być wyraźnie oddzielony i nigdy nie zastępuje oficjalnego porównania na pełnym W.

## 2. Metryki obowiązkowe i agregacja

Dla grupy G⊆S, n=|G|, e_i=p_i−y_i. Point prediction TFT to surowe wyjście q=.5 (nie gwarantujemy nieważonej mediany populacji); persistence definiuje sekcja4. Brak clinical weights w metrykach raportowych; każde wystąpienie ma wagę1.

| Metryka | Definicja i mianownik | Jednostka |
|---|---|---|
| MAE | Σ abs(e_i)/n | mg/dL |
| RMSE | sqrt(Σ e_i²/n), pierwiastek po agregacji | mg/dL |
| Bias / signed error | Σ e_i/n; dodatni=przeszacowanie | mg/dL |
| MARD | 100 Σ(abs(e_i)/y_i)/n, wszystkie y_i>0; bez clamp | % |
| Quantile pinball per q | Σ ρ_q(y_i−z_iq)/n; ρ_q(u)=u(q−1{u<0}) | mg/dL |
| Quantile hit per q | Σ 1{y_i≤z_iq}/n; dodatkowo hit−q | proportion; różnica w pp po ×100 |
| Interval coverage (PICP) | Σ 1{L_i≤y_i≤U_i}/n, obie granice włączone | proportion / % |
| Mean interval width | Σ(U_i−L_i)/n | mg/dL |
| Any quantile crossing rate | Σ 1{istnieje j: z_ij>z_i,j+1}/n | proportion / % |
| Adjacent crossing rate | Σ_i Σ_j 1{z_ij>z_i,j+1}/[n(K−1)] | proportion / % |

Minimalne central intervals są ustalone: Q=(.02,.10,.25,.50,.75,.90,.98), K=7; pary (.25,.75),(.10,.90),(.02,.98) mają nominalnie50%,80%,96%. Nie produkuj90% bez endpointów .05/.95 ani interpolacji. Niezgodność Q z checkpointem jest findingiem/odmową, nie powodem relabelowania osi. Pinball to **nieważony, bez factor2** reporting score; pomocniczy mean over q=Σ_q pinball_q/K ma jawną nazwę. Nie mylić go z training/validation ClinicalQuantileLoss.

Ties nie są crossing. Zachowaj raw quantiles, bez sorting/clipping/rearrangement. Gdy L>U, interval summary tej pary/grupy jest INVALID z N_invalid i oryginalnym n; nie raportuj ujemnej średniej szerokości, nie usuwaj tych wierszy. Quantile hit/pinball i crossing pozostają definiowalne dla finite raw outputs. Metryki probabilistyczne obowiązują TFT; persistence ma point-only N/A, nie udawane identyczne kwantyle lub zerową niepewność.

Agregacja obowiązkowa dla każdej stosownej metryki:

1. **Per-horizon**: wszystkie12 h, każdy na W; wyróżnienie15/30/60 jest prezentacyjne, bez wyboru najlepszego h.
2. **Per-patient**: metryka dla każdej osoby i każdego h; publikuj pełną tabelę oraz N. Brak danych danej osoby pozostaje widoczny.
3. **Micro/global**: wzór z tabeli na wszystkich occurrences danej grupy; global oznacza micro i musi mieć tę etykietę.
4. **Macro patient**: średnia odpowiednich metryk pacjentów z n>0, równa waga każdej osoby. Macro RMSE=mean(patient RMSE), nie sqrt(mean(patient MSE)). Podaj liczbę osób uwzględnionych/ogółem; bez zastępowania wyniku osoby zerem.
5. **Per-range**: dla obu granularności z sekcji3, per-horizon, micro, macro i per-patient z N. Strata zależą tylko od targetu, wspólne dla modeli. Macro w stratum używa osób z obserwacjami w tym stratum, z jawną listą brakujących.

All-horizon micro i macro są wtórnym podsumowaniem: micro stosuje wzór bezpośrednio do S; macro najpierw wzór na wszystkich occurrences osoby, potem średnia osób. Nie zastępują wyników per-horizon. Nie uśredniaj średnich batchy. Empty group → null, N=0, reason; małe n raportuj bez progu odrzucania i bez deklaracji wiarygodności. Rozkład per-patient jest obowiązkowy także dla coverage/width/crossing; nie wystarczy średnia globalna. Rounding tylko na prezentacji.

Różnice TFT−persistence dla point metrics liczyć na wspólnych keys, bez selekcji zwycięskiego zakresu. CI, testy istotności i resampling nie są wymaganiem C-core: nie zakładaj IID nakładających się okien. Obowiązkowy zestaw jest wybrany według typu błędu i własności niepewności, nie według performance. Brak arbitralnego progu MARD/coverage dającego clinical PASS.

## 3. Zakresy glikemii

Wszystkie granice w mg/dL, według **obserwowanego y**, bez zaokrąglania przed przydziałem.

| Główna strata | Dokładny zakres | Drobniejsza strata |
|---|---|---|
| Hypo | y<70 | y<54 oraz 54≤y<70 |
| Target range | 70≤y≤180 | 70≤y≤180 |
| Hyper | y>180 | 180<y≤250 oraz y>250 |

Oba poziomy obowiązkowe. Etykiety „54–69” i „181–250” są poprawnym skrótem dla liczb całkowitych; dla wartości ciągłych używaj nierówności powyżej (np.69.9=hypo,180.1=hyper). 54 należy do drugiej strata,70 i180 do target range,250 do niższej hyper. Trzy główne/pięć drobnych strata są rozłączne i wyczerpują poprawne y>0.

To utrwalone zakresy raportowania CGM przyjęte wcześniej w planie, rozdzielające skrajne poziomy bez doboru na wynikach. Źródło metodologiczne: International Consensus on Time in Range (2019), wskazane w STAGE_C_AUDIT_PLAN.md. Nie są indywidualnymi celami terapeutycznymi, filtrem danych ani dowodem bezpieczeństwa predykcji. Udział forecast occurrences nie jest time-in-range; wyniki dla prognozy nie są oceną dokładności sensora wobec laboratoryjnej referencji.

## 4. Persistence zgodna z Stage A

Niech E_w obejmuje48 binów encodera kończących się w t. Niech s*=max{s∈E_w: CGM jest rzeczywiście obserwowany i dostępny w chwili t}. Wtedy p_w(h)=g(s*) dla każdego h=5,…,60, w natywnych mg/dL. Dostępność i przyczynowe przypisanie binów zgodne z istniejącym Stage A; nie wolno użyć przyszłego pomiaru, nearest-future lookup, targetów decodera lub innej osoby.

Stage A pozwala ffill≤6 kroków. Jeśli g(t) jest takim ffill, jego wartość jest równa ostatniej rzeczywistej obserwacji g(s*), a nie sztucznie wybranemu targetowi. Pełny48-krokowy kwalifikowany encoder i limit6 oznaczają, że obserwacja źródłowa dla końca encodera powinna leżeć w E_w. Dlatego wybór ostatniej rzeczywistej obserwacji jest zgodny z protokołem, nie dodaje wymogu świeżego pomiaru w t i nie zmniejsza W.

Implementacyjnie: lookup po subject i rzeczywistym timestampie w źródłowej ramce, target_observed sprzed ffill; zapisz s*, age_bins=t−s* w krokach5 min, flagę ffill oraz zgodność z końcowym CGM dostępnym TFT. Age≤6, żadnej wartości z placeholdera lub heurystycznej inverse transform encoder_cont. Jeżeli audyt wykaże brak s* lub nierówność z dozwolonym CGM encodera, zgłoś finding i invalid comparison; nie szukaj poza context i nie zmieniaj populacji. Jednostki/normalizer TFT podlegają osobnej kontroli dokładnie jednej inverse transform.

Persistence jest jedynym wymaganym baseline’em porównania. ARIMA pozostaje przedmiotem inspekcji legacy/failure handling w C09; nowy trailing-mean baseline nie jest wymagany ani wdrażany. Nie ma decyzji o dodatkowym baseline blokującej audyt.

## 5. Izolacja, weighting i granice wykonania

C-core: wyłącznie statyczna inspekcja i synthetic/unit/integration z CODEX_STAGE_C_TASK.md, w tym syntetyczne eksperymenty/oracles pokazujące działanie wag. Zero rzeczywistych danych/predykcji/metryk testowych, zero treningu/backward/optimizer, zero Optuny, zero zmian produkcyjnych. Guard ma pozytywną kontrolę odmowy otwarcia real data. Nie otwieraj wspólnego canonical parquet, który zawiera także source test; używaj dokumentacji/manifestów jako inherited evidence.

Test labels/performance nie wybierają metryk, metody/fit kalibracji, interval levels/thresholdów, clinical ranges, loss/objective variant, features, architektury, early stopping ani hyperparameters. Test pozostaje całkowicie zamrożony również po zakończeniu C-core.

ClinicalQuantileLoss pozostaje factor2, hypo_threshold70, hypo_weight2.5. Target-dependent weighting minimalizuje pinball dla F_w(a|x)=E[w(Y)1{Y≤a}|x]/E[w(Y)|x], nie ogólnie F(a|x). C13 ma sprawdzić wyprowadzenie oraz syntetyczny oracle (uniform[0,140]: weighted median49, unweighted70), bez optimizer. Empiryczny wpływ na wytrenowany model pozostaje NOT EVALUATED. Poprawny loss nie dowodzi nominalnego coverage, calibrated median ani clinical reliability.

Przyszłe objective variants są osobnym validation-only eksperymentem: trening na train, decyzje/ocena na oddzielonych development roles, nigdy test. Osobny calibration split, jego boundaries, wybór metody, crossing postprocessingu i resampling pozostają **decyzjami po audycie**, nie warunkami rozpoczęcia C-core. Nie wykonuj fit na rzeczywistych danych ani nie twórz produkcyjnego kalibratora; brak implementacji raportuj jawnie. Surowe50/80/96 intervals i powyższe reporting metrics są już ustalone niezależnie od późniejszej kalibracji.

## 6. Provenance i zakończenie

Do audytowych artefaktów dołącz wersję/hash tego kontraktu, planu, tasku i audit config; commit/dirty; fixture/window-key/prediction hashes; schema, units, Q, normalizer, context/horizon, checkpoint role/hash, numerical profile i denominators. C-core zachowuje budżet/tolerancje i format findings Cxx-Fyy z tasku. Nie zamykaj findings przez xfail lub naprawy.

Nie ma otwartych decyzji metodologicznych blokujących C-core. DoD to kompletne dowody/ograniczenia C00–C19 w trzech warstwach, z findings i propozycjami remediation. Audit COMPLETE może mieć FAIL; nie jest certyfikacją empirycznej kalibracji ani całego Stage C. Po audycie osobne review. Obecnie Stage C PLAN READY, execution nie rozpoczęto.
