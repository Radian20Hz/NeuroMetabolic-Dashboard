# Stage B — finalne review research leada

Data: 2026-09-13. Stan: **CPU PASS; CUDA finalization PENDING**. Ten dokument definiuje rekomendowany kontrakt i warunki końcowego PASS. Nie jest deklaracją wykonania nowych testów ani poleceniem implementacji.

## 1. Podstawa decyzji

Sprawdzono AGENTS.md, docs/BASELINE_STAGE_B_REMEDIATION.md, configs/baseline_v1.json, konfigurację audytu, aktywną orkiestrację baseline_training.py, runner remediation, lokalną interpolację PF i manifest cuda_final_full. Uwzględniono wcześniejsze plany/review i najnowsze decyzje użytkownika. HEAD: `84dd507b0888f8f5659f6a0ccd8a74a1ad78de96`, drzewo na wejściu czyste. Raport remediation opisuje przedcommitowy stan na `896c9a...`; obecny commit zawiera zmiany i raport. Nie przeprowadzono ponownie testów i nie badano test-performance.

Akceptuję dowody CPU w ich zadeklarowanym zakresie:

- Stage A: 31 PASS. Stage B remediation: 32 PASS — 30 w pełnym suite i 2 focused na końcowym kodzie, nie jeden ponownie wykonany zestaw 32.
- B03-F01, B08-F01, B10-F01–F04, B11-F01/F02, B15-F01/F02 zamknięte na CPU. Kontrole finite oraz role i zgodność checkpointów są częścią aktywnej ścieżki.
- CPU: 8 updates versus 4+4 w nowym procesie daje identyczne wagi, optimizer/scheduler, RNG, callbacki i kolejność danych w testowanym profilu; nie jest to gwarancja innych urządzeń, bibliotek ani długich eksperymentów.
- B13-F01: świadomie DEFERRED / OUT OF SCOPE. SWA OFF zapisane w profilu, ON odrzucane w certyfikowanej ścieżce; wzór uśredniania SWA nie został uznany za błędny.
- Architektura i clinical weighting bez zmian; cel naukowy weighting odłożony do Stage C.

CUDA na hoście: NVIDIA GeForce **RTX 3070, 8 GB**, driver 615.71.09, CUDA UMD 13.4; PyTorch 2.11.0+cu130, torch.version.cuda=13.0. To odczyt diagnostyczny ma pierwszeństwo przed ogólnym opisem „3070 Ti” w AGENTS.md. Tensor operation PASS; rzeczywisty TFT forward działa. Manifest potwierdza błąd `upsample_linear1d_backward_out_cuda` i **0 optimizer updates**. To nie jest wynik niestabilnego treningu ani potwierdzenie GPU finite-injection/replay.

## 2. Porównanie A/B/C

| Kryterium | A — strict bitwise CUDA | B — seeded reproducibility z pomiarem różnic | C — eliminacja problematycznej operacji |
|---|---|---|---|
| Scientific defensibility | Mocny kontrakt mechaniczny, lecz nie warunek konieczny rzetelnej nauki ani dowód jakości modelu. Obecny stos nie realizuje go dla tego TFT. | Uzasadniony, gdy kontrolowane są dane/metoda/losowość, odchylenia mierzone bez selekcji korzystnych runów, a zakres gwarancji jawny. | Uzasadniony po dowodzie równoważności funkcji i gradientów. Bez niego zmieniamy badaną metodę. |
| Reproducibility | Potencjalna identyczność bitowa na tym samym stosie/urządzeniu; w tym przypadku brak wykonalnego backward. | Kontrola pełnego stanu i empirycznie ograniczonych rozbieżności, bez obietnicy identycznych bitów czy wyników długiego treningu. | Może umożliwić A; sama zamiana nie gwarantuje usunięcia wszystkich źródeł niedeterminizmu. |
| Engineering complexity | Niska konfiguracja flag, wysoki/nieznany koszt doprowadzenia całego stosu do wsparcia operacji. | Niska–umiarkowana: jawny profil numeryczny, manifest i mały test w kilku procesach. | Umiarkowana–wysoka: lokalizacja wszystkich wywołań, zgodny operator/autograd, testy graniczne i utrzymanie kompatybilności. |
| Ryzyko zmiany modelu | Brak przy samym strict flag; obejścia lub nowe wydanie biblioteki wymagają ponownej walidacji. | Nie zmienia architektury/objective; zmienia gwarancję numeryczną, którą należy wersjonować. | Zależne od rozwiązania; zamiana linear na nearest, zmiana hidden size lub usunięcie warstwy zmienia funkcję/model. |
| Koszt obliczeniowy | Deterministyczne algorytmy mogą być wolniejsze; obecnie run kończy się przed update. | Kilkadziesiąt syntetycznych updates przy zamknięciu; później jawne powtórzenia w protokole badawczym. | Koszt implementacji i testów; np. fallback CPU może powodować transfery i spowolnienie. |
| Wpływ na dalsze badania | Może niepotrzebnie blokować GPU bez korzyści dla pytania badawczego. | Zachowuje bieżący model i pozwala mierzyć zmienność zamiast ją ukrywać. Dalsze efekty należy porównywać ze skalą zmienności. | Może wymagać nowej wersji architektury i utrudniać porównywalność, jeśli równoważność nie jest wykazana. |

**Rekomendacja: B dla Baseline v1.0 CUDA.** A pozostaje trybem diagnostycznym, CPU zachowuje osobny strict kontrakt. C nie jest wymaganą remediation i nie jest częścią zamknięcia Stage B.

Lokalny PF TimeDistributedInterpolation używa `F.interpolate(..., mode='linear', align_corners=True)` przy zmianie rozmiaru reprezentacji. Nie proponuję zamiany na nearest, sztucznego wyrównania hidden sizes ani pomijania gradientu. Ewentualny przyszły odpowiednik C musi zachować współczynniki interpolacji, align_corners, rozmiary brzegowe, dtype i analityczny backward oraz przejść niezależny oracle/gradcheck. Dopiero wtedy można oceniać, czy to zmiana implementacji czy modelu. Nie zakładamy bez testu, że fallback CPU lub nowa wersja biblioteki rozwiązuje problem.

PyTorch dokumentuje możliwość niedeterministycznych gradientów interpolacji CUDA; strict mode zgłasza błąd, gdy nie ma deterministycznego wariantu. Nie jest to samo co wykrycie NaN/Inf. [Interpolacja, PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.interpolate.html), [deterministic algorithms, PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html).

## 3. Oficjalny kontrakt proponowany do przyjęcia

Nazwa: **CUDA seeded reproducibility, FP32, bounded numerical variability**. Obowiązuje dopiero po zatwierdzeniu profilu i przejściu finalization; nie opisuje obecnej twardo strict ścieżki jako już naprawionej.

1. Ten sam seed, snapshot danych/okien/normalizatora, kolejność batchy, architektura, config, sampler, optimizer/scheduler, callbacki, biblioteki, urządzenie i profil numeryczny. Zapis rzeczywistych ustawień, nie tylko żądanych flag.
2. CPU pozostaje strict, FP32, num_workers=0, jednoznaczna granica epoki; dotychczasowego bitwise CPU resume nie osłabiamy.
3. CUDA: preferuj deterministyczne algorytmy dostępne w stosie, dopuszczając ostrzeżenie dla znanej niedeterministycznej operacji. Minimalny wariant obecnego Lightning: `deterministic='warn'`, `benchmark=False`, odpowiadający `torch.use_deterministic_algorithms(True, warn_only=True)`. Bez automatycznego retry z deterministic=False po dowolnym błędzie.
4. Jawny profil FP32, AMP OFF, TF32 OFF, num_workers=0, jeden proces/GPU, jeden wątek CPU dla fixture. Użyj jednego wspieranego sposobu ustawienia TF32 w tym stosie; nie mieszaj sprzecznych nowych i starych API. Zapisz efektywny matmul precision, cuDNN, cuBLAS/workspace configuration oraz stan deterministic/warn_only po inicjalizacji Trainer i przed backward. Ustawienia dotyczą procesu, nie instalacji środowiska.
5. Zachowaj wszystkie finite guards, clipping, walidację danych i checkpointów. Tryb warning dotyczy deterministyczności, nie pomijania błędów numerycznych. Znany operator `upsample_linear1d_backward_out_cuda` jest jedynym obecnie zaakceptowanym ostrzeżeniem niedeterminizmu. Inny operator → HOLD do review, nie automatyczne rozszerzenie listy. Ostrzeżenia zapisuj, nie filtruj w ciszy.
6. Profil reprodukowalności i wszystkie efektywne flagi mają wejść do configu, fingerprintu checkpointu i provenance. Zmianę wersjonuj jako nową rewizję profilu numerycznego; nie pozwól traktować starego strict artefaktu jako tego samego seeded runu. CPU strict i CUDA seeded to różne certyfikowane profile.
7. CUDA resume: wyłącznie po ukończonej epoce i walidacji; odtworzenie optimizer/scheduler/RNG/callbacków i porządku danych wymagane dokładnie na granicy. Dalsza trajektoria ma spełniać ten sam z góry ustalony kontrakt odchyleń co świeże runy; **bez deklaracji bitwise CUDA resume**. Cross-device resume, mid-epoch i partial accumulation pozostają unsupported.
8. Gwarancja dotyczy testowanego stosu i małego regression workload. Nie obiecuje identycznych końcowych wag ani statystycznej stabilności pełnego treningu. Powtórzenia same-seed badają zmienność numeryczną, nie wariancję różnych inicjalizacji/seeds ani generalizację.

Globalny seed nie usuwa wszystkich źródeł niedeterminizmu; PyTorch nie gwarantuje identyczności między wersjami, platformami i CPU/GPU. Nasz kontrakt celowo ma węższy, testowalny zakres. [Reproducibility, PyTorch 2.11](https://docs.pytorch.org/docs/2.11/notes/randomness.html).

## 4. Minimalny CUDA regression protocol — zamrożony przed wykonaniem

### Przebiegi i budżet

- Zapisz konfigurację, progi z tej sekcji i ich SHA-256 **przed pierwszym seeded CUDA runem**. Nie dobieraj ich według wyników.
- Użyj obecnej syntetycznej fixture remediation: context48/horizon12, hidden8, hidden_continuous4, heads1, LSTM1, dropout0.1, batch2, LR3e-4, clip1, accumulation1, num_workers0, seed42, SWA OFF. Te wymiary mają zachować wywołanie problematycznej interpolacji. Nie zmieniaj ich, aby wyeliminować warning.
- Trzy świeże, niezależne procesy C1/C2/C3 uruchomione kolejno na tej samej GPU; każdy dokładnie 8 rzeczywistych optimizer steps w 4 syntetycznych epokach. Nie współdzielą obiektów RNG, modelu ani checkpointu inicjalizacji. Te same seedy muszą wytworzyć identyczną inicjalizację; sprawdź hash i tensor equality. Hash fixture, indeksów okien i wszystkich uporządkowanych batchy identyczny.
- Dodaj tylko jeden split-run: proces P wykonuje 4 updates i zapisuje prawidłowy LAST; proces R odtwarza go i wykonuje kolejne 4, przy tym samym planie total=8/max_epochs=4. Porównaj P+R z każdą z C1/C2/C3. To minimalna kontrola CUDA continuation, nie szeroki ponowny audyt checkpointów.
- Trzy małe GPU negative probes: nonfinite input, target, gradient; każdy w osobnym izolowanym runie ma zakończyć się produkcyjnym błędem przed pierwszym optimizer step, bez valid checkpointu. Użyj istniejących injection i containment; jeżeli containment pierwszy łapie błąd, test FAIL. To domyka nieprzetestowaną ochronę GPU bez powtarzania całej macierzy CPU.
- Budżet pozytywny łącznie **32 updates** (24 +4+4); negatywne probes 0. Każdy fit ≤8 steps/4 epoki. Limit finalizacji CUDA 600 s, pojedynczy proces 90 s; timeout raportuj i nie powiększaj automatycznie. Od osobnych regresji CPU wymagaj tylko właściwego zakresu zmian. Nie uruchamiaj pełnego treningu ani Optuny.

Osiem steps jest w obecnym warmupie; ten test nie certyfikuje długiego cosine, długookresowego dryfu, jakości klinicznej ani VRAM pełnego modelu. Istniejące CPU testy schedulera pozostają źródłem dowodu dla jego mechaniki.

### Co zapisać

Dla każdej aktualizacji: global step, epoch, uporządkowane IDs/digest batcha i valid lengths, loss przed backward, LR **użyty do update** dla wszystkich param groups, stabilna globalna norma gradientów przed i po clippingu, liczba aktualizacji, finite status. Norma float64 z rzeczywistych gradientów, w tym samym hooku we wszystkich procesach. Synchronizuj CUDA przy pomiarze i finalnym eksporcie; logowanie nie losuje niczego i nie zmienia kolejności.

Finalnie zapisz named parameters, potrzebne buffers oddzielnie, optimizer/scheduler state, identyfikatory/hashes checkpointów, RNG CPU/CUDA/generatorów, callback state i środowisko. Do porównań eksportuj surowe tensory, nie zaokrąglone JSON-y. Epoch val_loss też raportuj: mały dryf może zmienić BEST przy bliskim remisie i nie musi sam w sobie oznaczać błędu selekcji.

### Reguły porównania ustalone a priori

Dla wszystkich par C1–C2, C1–C3, C2–C3 i dla każdej pary Ck–(P+R): bez wybierania najlepszego runu jako referencji.

Warunki dokładne: seed/config/environment fingerprints, initial parameters, pełna kolejność danych, liczba updates/epok i sekwencja LR są identyczne. Dla P/R stan na granicy jest dokładną rekonstrukcją **własnego** checkpointu P, łącznie z CUDA RNG, optimizer/scheduler i callbackami; nie musi być bitwise równy granicy odrębnego Ck. Po restore nie ma nowej inicjalizacji lub dodatkowego zużycia RNG przed pierwszym resumed forward. Brakujące/dodatkowe kroki to FAIL, nie odchylenie numeryczne.

Dla finite wartości x,y użyj symetrycznej reguły:

`abs(x-y) <= atol + rtol * max(abs(x), abs(y))`.

| Wielkość | atol | rtol | Jednostka/granularność |
|---|---:|---:|---|
| Każdy batch loss i epoch validation loss | 1e-5 | 1e-4 | skala obecnego weighted objective, każdy krok/epoka |
| Norma gradientu przed i po clip | 1e-5 | 1e-3 | każda aktualizacja, oddzielne pre/post |
| Każdy element finalnego named parameter | 1e-6 | 1e-4 | każdy tensor/element, żadnego ukrywania w średniej całego modelu |
| LR | 0 | 0 | każdy krok i param group |

To **proponowane konserwatywne progi inżynierskie dla 8-step FP32 regression**, nie kliniczne progi istotności ani uniwersalna gwarancja. Nie zostały dopasowane do wyników seeded CUDA, których nie wykonano. Większy rtol dla norm uwzględnia czułość gradientów; atol zapobiega nieokreślonej względnej różnicy przy zerze. Surowe różnice i skala aktualizacji zostają widoczne mimo progu.

Raportuj dla każdej pary i wielkości: bitwise equality, max absolute difference, RMS absolute difference, symetryczny relative L2 `||x-y||2/max(||x||2,||y||2,1e-12)`, maksymalny znormalizowany błąd `abs(x-y)/(atol+rtol*max(abs(x),abs(y)))`, liczbę przekroczeń oraz najgorszy krok/tensor/indeks. Przy LR pomiń dzielenie przez zerowy próg, podaj exact boolean i max abs. Dla parametrów: wynik per tensor i najgorszy tensor, nie tylko globalna norma. Dodatkowo podaj normę całej aktualizacji `||theta_final-theta_initial||2` i stosunek rozbieżności między runami do większej z norm ich aktualizacji (floor1e-12); ta metryka opisowa pokazuje, czy mały drift jest duży wobec małego uczenia, nie służy do dobierania progu po fakcie.

Przedstaw pełne trace i max różnic w kolejnych krokach. Słowo „stabilne” oznacza tutaj: wszystkie trzy niezależne runy i split-run kończą pełny budżet, wszystkie próbki spełniają zarejestrowane granice, brak nonfinite i nowych ostrzeżeń. Nie wnioskuj o rozkładzie prawdopodobieństwa, p-value lub długiej stabilności z n=3. Bitwise equality może wystąpić i jest raportowana, ale nie staje się nową obietnicą dla CUDA.

### Werdykt testu

- **PASS:** wszystkie dokładne invariants i bramki finite/checkpoint spełnione, każdy porównywany element w zamrożonej tolerancji, wszystkie runy zakończone, wyłącznie zaakceptowany warning.
- **FAIL / HOLD:** przekroczenie progu, brak poprawnego restore/budżetu, nowa operacja niedeterministyczna, nonfinite lub brak ochrony produkcyjnej. Wyjaśnij rodzaj: variability, execution, safety, provenance albo unsupported environment. Nie każda taka awaria oznacza konieczność zmiany architektury.
- Nie powtarzaj tylko niekorzystnych runów aż do PASS; zachowaj wszystkie próby. Uzasadniona zmiana testu/progu wymaga nowej wersji protokołu, niezależnego review i pełnego nowego zestawu, z zachowaniem wcześniejszego FAIL. Nie dobieraj tolerancji do największego zaobserwowanego błędu.

## 5. CUDA blocker status i SWA

**CUDA blocker status: ACCEPTED LIMITATION.** Brak strict deterministycznego backward dla tej operacji nie wymaga naprawy architektury Baseline v1.0. Jest zgodny z rekomendowanym kontraktem B, o ile jego testy przejdą. Obecna implementacja nadal zatrzymuje się na strict flag i nie ma danych o seeded CUDA variability; integracja jawnego profilu oraz test finalization są pozostałą bramką Stage B. Nie oznacza to, że CUDA już otrzymało PASS.

Nie akceptujemy w ramach tego ograniczenia NaN/Inf, ukrywania błędów, pomijania updates, niezgodnych checkpointów, utraty RNG lub różnej kolejności danych. Ich wystąpienie byłoby aktywnym blockerem niezależnie od wyboru A/B/C.

**SWA status: OFF w Baseline v1.0; B13-F01 DEFERRED / OUT OF SCOPE, nie aktywny blocker.** To zatwierdzona przez użytkownika zmiana protokołu z regresją odmowy ON. ON pozostaje eksperymentalną przyszłą ścieżką, bez certyfikacji i bez dziedziczenia PASS v1.0. Nie zmieniamy historycznego statusu findingu na FIXED.

## 6. Minimalne pozostałe prace Stage B

1. Zatwierdzić rekomendowany profil B i prerejestrowany protokół powyżej; ta rozmowa autoryzuje dokumenty, nie ich wykonanie.
2. Codex dodaje wyłącznie jawny wybór profilu CUDA seeded/CPU strict oraz jego fingerprint, warnings evidence i ograniczony test finalization. Obecny aktywny train_baseline ma `deterministic=True`; zmiana tylko w harnessie nie wystarczy do certyfikacji działającej ścieżki. Bez zmian operatorów, architektury, lossu, danych i instalacji środowiska.
3. Wykonać C1–C3, P/R i trzy finite probes. Potwierdzić rzeczywiste flagi produkcji oraz manifest/checkpoint round-trip. Sprawdzić odrzucenie resume z innym profilem numerycznym.
4. Uruchomić jeden aktualny CPU acceptance suite (32 testy) po zmianie wspólnej orkiestracji/kontraktu, z potwierdzeniem CPU bitwise resume. Stage A nie trzeba ponownie wykonywać, jeśli jej kod/dane/fixture nie zmieniły się i wcześniejszy wynik jest powiązany z niezmienionymi źródłami; jeśli finalizer zmieni wspólne komponenty danych, należy ją powtórzyć. Bez nowego szerokiego audytu B00–B17.
5. Nowy raport końcowy: config hash, commit/dirty, środowisko/hardware, wszystkie procesy/exit codes/updates/tolerancje/różnice, status CPU, CUDA seeded i SWA osobno. Wyniki historyczne pozostają bez nadpisywania. GPU negative probes bez valid artefaktów, pozytywne z poprawnymi rolami/provenance. Review zgodności z kryteriami zamyka etap.

Po spełnieniu tych warunków oficjalny werdykt może brzmieć:

**Stage B PASS — CPU strict epoch-boundary replay; CUDA seeded FP32 reproducibility w zadeklarowanych granicach krótkiej regresji; SWA OFF.**

**Can Stage B be closed after that work? YES**, jeśli wszystkie wymagane bramki mają PASS. Do tego czasu CPU pozostaje PASS, CUDA PENDING FINALIZATION. Zakończenie Stage B nie uruchamia treningu, Optuny ani Stage C automatycznie i nie jest walidacją kliniczną.

## 7. Intentionally deferred to Stage C

Naukowa zasadność clinical weighting, docelowa interpretacja kwantyli, kalibracja i empiryczne coverage/width, polityka quantile crossing, jakość predykcji, metryki/horyzonty/grupy i glycemic ranges, porównanie z persistence oraz poprawny protokół ewaluacji pozostają do Stage C. Ważony pinball estymuje kwantyle rozkładu przeważonego target-dependent wagą; sama poprawność implementacji nie nadaje nominalnego coverage nieważonemu rozkładowi.

W przyszłych badaniach trzeba oddzielić zmienność numeryczną przy tym samym seedzie od zmienności między seedami oraz statystycznej niepewności wyników; mały efekt porównywalny z tymi źródłami nie wystarcza do twierdzenia o poprawie. Nie oznacza to zlecenia tych eksperymentów teraz. Test set pozostaje wyłączony z decyzji rozwojowych.

SWA ON, usuwanie interpolacji, mid-epoch/cross-device resume, AMP, wielu workerów/DDP i gwarancje między stosami są osobnymi przyszłymi rozszerzeniami inżynierskimi, nie wymaganym elementem Stage C ani powodem do ponownego otwarcia poprawnych kontraktów CPU.
