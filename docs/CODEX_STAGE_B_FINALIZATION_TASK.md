# Codex — minimalne zamknięcie Stage B CUDA/reproducibility

Status: SPECYFIKACJA DO AKTYWACJI. Nie wykonuj jej wyłącznie dlatego, że znajduje się w repo. Wymagane jest osobne polecenie użytkownika przyjmujące profil B i uruchamiające finalization. Nie powtarzaj prośby, jeżeli takie polecenie już istnieje w rozmowie.

## Cel i źródła

Zamknąć jedyną pozostałą bramkę CUDA przy zaakceptowanym ograniczeniu braku strict bitwise GPU determinism, bez ponownego szerokiego audytu Stage B. Przeczytaj AGENTS.md, docs/STAGE_B_FINAL_REVIEW.md w całości, docs/BASELINE_STAGE_B_REMEDIATION.md, configs/baseline_v1.json i bieżące baseline_training/checkpoint_registry/runner/testy remediation oraz właściwe instrukcje podkatalogów.

Review wykonano na `84dd507b0888f8f5659f6a0ccd8a74a1ad78de96`. Potwierdź bieżący HEAD, dirty state i różnice od tego commitu; nie naruszaj pracy użytkownika. Zachowaj historyczne raporty, 31 PASS Stage A i 32 PASS CPU jako wyniki historyczne, nie świeżo wykonane.

Stan zaakceptowany: SWA OFF, B13-F01 DEFERRED; CPU bitwise epoch-boundary 8 vs4+4; BEST/LAST/WEIGHTS-ONLY rozdzielone; clinical weighting niezmienione. CUDA tensor i forward działają na RTX 3070; strict backward zgłasza `upsample_linear1d_backward_out_cuda`, zero updates. Nie jest to powód do instalacji lub zmiany modelu.

## Jedyny dozwolony zakres zmian po aktywacji

1. Jawny profil reprodukowalności w aktywnej ścieżce: CPU strict zachowany, CUDA seeded FP32 z deterministic='warn', benchmark=False, AMP/TF32 OFF, num_workers=0. Sprawdź publiczne API obecnego stosu i efektywne flagi po konstrukcji Trainer. Bez automatycznego fallbacku z strict na seeded po błędzie.
2. Profil/flag settings w configu, provenance i kontrakcie checkpointu; wersjonowana rewizja profilu, odmowa niezgodności podczas resume. Nie osłabiaj pozostałych fingerprints ani finite gates. CPU i CUDA mają osobne profile; zmiana profilu nie jest kontynuacją starego runu.
3. Ograniczony runner/test i raport zgodne z sekcją 4 final review. Reuse istniejącej fixture, orkiestracji i injection. Dodaj wyłącznie brakujące trace gradient norms, CUDA RNG i porównań; trace nie może konsumować losowości. W razie konieczności zapisu/licznika generatorów zapewnij obserwację bez zmiany metody.

Poza zakresem: zmiana interpolacji/hidden sizes/architektury, loss/weighting, scheduler policy, danych i splitów, SWA ON, prywatna ingerencja w pętlę, AMP, DDP, wielu workerów, migracje checkpointów, pełny trening, Optuna, Stage C i dowolne test-performance. Nie edytuj site-packages, nie aktualizuj bibliotek/sterowników. Ustawienia procesu dla zatwierdzonego profilu nie są zgodą na zmianę środowiska systemowego.

## Wykonanie

- Najpierw zapisz i zahashuj konfigurację oraz progi z final review. Nie uruchamiaj seeded GPU przed ich zamrożeniem. Środowisko tylko potwierdź krótkim istniejącym CUDA check, jeśli zmieniło się od diagnostyki lub aktualny proces nie ma dostępu. Nie reinterpretuj braku GPU w sandboxie jako problemu modelu.
- C1, C2, C3: trzy niezależne procesy, sequential na jednej GPU, seed42, identyczna syntetyczna fixture/okna/order. Mały TFT hidden8/continuous4/head1/dropout0.1, context48/horizon12, batch2, LR3e-4, clip1, FP32, SWA OFF, 4 epoki/8 updates. Zachowaj problematyczną operację; nie zamieniaj modelu, żeby warning zniknął.
- P/R: osobny proces 4 updates, LAST po walidacji drugiej epoki, nowy proces 4 updates z tym samym total=8/epochs=4. Dokładny restore własnego boundary state, wszystkie cztery dalsze updates, brak obietnicy bitwise końcowej trajektorii GPU.
- Negative GPU probes: input NaN, target Inf, gradient NaN, według istniejącego harnessu. Produkcyjne zabezpieczenie ma zadziałać przed AdamW; zero updates i valid checkpointów. Pierwsza interwencja audit containment oznacza FAIL. Nie powtarzaj całej macierzy finite CPU.
- Łącznie 32 pozytywne optimizer updates i 0 w negatywnych probes; limit CUDA suite 600 s/proces 90 s, fit≤8 updates/4 epoki. Żadnego rozszerzania budżetu ani rerun-until-green.

Wszystkie ostrzeżenia niedeterminizmu zapisz z nazwą operatora. Jedyny obecnie zaakceptowany operator to upsample_linear1d_backward_out_cuda. Dodatkowy warning → HOLD do review. Warn-only nie może dotyczyć finite error, walidacji danych ani checkpointu.

## Porównania i prerejestrowane bramki

Pełna definicja miar znajduje się w final review; zastosuj ją bez zmian. Porównaj wszystkie trzy pary C1/C2/C3 oraz P+R z każdym Ck. Nie wybieraj referencji na podstawie korzystnego wyniku.

Dokładne: initial weights, config/data/order/environment, update counts, LR sequence; dla P/R pełny restore optimizer/scheduler/RNG CPU+CUDA/callbacków/order na granicy względem checkpointu P. Do porównań stanów callbacków normalizuj wyłącznie nieistotne unikalne ścieżki/run IDs, nigdy liczniki/metryki.

Symetryczny elementwise limit: `abs(x-y) <= atol + rtol*max(abs(x),abs(y))`:

- batch i epoch validation losses: atol1e-5, rtol1e-4;
- pre/post-clip gradient norms: atol1e-5, rtol1e-3;
- final named parameters: atol1e-6, rtol1e-4, każdy element i tensor;
- LR sequence: exact, atol0/rtol0.

Raportuj bitwise equality, max abs, RMS, relative L2 z floor1e-12, normalized maximum error, exceedance count, najgorszy krok/tensor. Dodaj normę aktualizacji i relację drift/update jako opis, bez nowego dobranego post hoc progu. Surowe tensory i trace lokalnie; JSON porównawczy nie może opierać się na zaokrąglonych liczbach. LR odpowiada krokowi rzeczywiście wykonanemu, nie następnej wartości po scheduler.step.

Każdy element musi spełniać zamrożony próg. FAIL zachowaj; nie podnoś tolerancji, nie zmieniaj seedu, nie kasuj niekorzystnych procesów. Gdy przyczyna wymaga nowej metody/środowiska, zatrzymaj tylko zależną część i przedstaw osobny task. Nie rozpoczynaj wariantu C samodzielnie.

## Sprawdzenia końcowe i wyjścia

Po zmianie wspólnej orkiestracji/kontraktu wykonaj raz bieżący CPU acceptance suite 32, z dowodem zachowania CPU bitwise resume. Pełny historyczny audyt B00–B17 nie jest wymagany. Stage A powtarzaj tylko przy zmianie jej danych/kodu/wspólnej fixture; inaczej powiąż istniejące 31 PASS z niezmienionymi źródłami. Dodaj focused test zgodności profilu checkpointu i osobnych ustawień CPU/CUDA. Zweryfikuj round-trip GPU LAST w P/R, pozostałe role CPU już mają acceptance.

Utwórz `docs/STAGE_B_FINALIZATION_REPORT.md` oraz lokalny manifest/porównania w ignorowanym katalogu unikalnego runu. Raport: commit/dirty/hash config, rzeczywisty hardware/driver/build/biblioteki/flags, warning inventory, polecenia/exit codes/czas/actual updates, tabela wszystkich par i miar, CPU strict/CUDA seeded/finite/SWA status. Nie nadpisuj historycznego remediation. Nie commituj danych, checkpointów ani surowych lokalnych logów. Commit/push tylko na polecenie użytkownika.

Definition of Done:

- profil numeryczny zaakceptowany i rzeczywiście użyty przez aktywną produkcyjną ścieżkę, a nie tylko patch testowego Trainer;
- trzy fresh i jeden split-run spełniają invariants, prerejestrowane tolerancje oraz budżet;
- GPU finite probes zatrzymane przez produkcję, nie containment; valid checkpointy wyłącznie dla poprawnych runów;
- CUDA resume przywraca pełny stan i order, ale nie jest opisane jako bitwise; CPU strict replay nadal PASS;
- brak nieznanych zaakceptowanych po cichu warnings i brak niezgodnych profili checkpointów;
- SWA OFF/ON rejected zachowane, weighting/architektura/data protocol niezmienione;
- dowody, focused testy i diff review kompletne; raport nie wyciąga wniosków o pełnym treningu, coverage ani klinicznej jakości.

Jeśli wszystko przejdzie, zaproponuj do final review werdykt: „Stage B PASS — CPU strict epoch-boundary replay; CUDA seeded FP32 reproducibility w zakresie krótkiej regresji; SWA OFF”. Jeśli nie przejdzie, raportuj konkretny FAIL/HOLD/BLOCKED. Nawet po PASS nie uruchamiaj treningu, Optuny ani Stage C.
