# Codex task — training recertification after C02

**READY FOR LATER ACTIVATION / DO NOT EXECUTE FROM THIS DOCUMENT ALONE**

To osobne zadanie wykonawcze. Obecne zlecenie obejmuje tylko jego przygotowanie. Po późniejszym jawnym poleceniu użytkownika wykonaj [wiążący plan](STAGE_C_TRAINING_RECERTIFICATION_PLAN.md), rewizja nmd-c02-training-recertification-1. Nie rozpoczynaj pełnego Baseline training po jego zaliczeniu.

## Cel

Sprawdź na danych syntetycznych, czy observed-train-encoded-subject-v2 zachowuje poprawne tensors, finite training, CPU strict replay, CUDA seeded reproducibility oraz pełny checkpoint/resume dla nowego nmd-baseline-v1.0-stage-c-1. Nie oceniaj jakości predykcji ani clinical performance.

## Najpierw

- Przeczytaj AGENTS.md, plan, STAGE_C_REMEDIATION_REPORT.md, STAGE_C_REMEDIATION_PLAN.md, STAGE_B_FINAL_REVIEW.md, aktualizację callback PASS w STAGE_B_FINALIZATION_REPORT.md, baseline_v1_stage_c.json oraz C02/registry/baseline_training/finite/numerical sources.
- Zweryfikuj branch/HEAD/dirty. Punkt odniesienia c2e27a28824cd9ec015c362b7e8df65f3c0902dc. Dokumentacyjne późniejsze commity są dopuszczalne po diff; production/config/stack drift wymaga impact review przed fit. Nie kasuj cudzych zmian. Nie pracuj na main.
- Utwórz nowy test-only harness, inspection/import bez uruchamiania produkcyjnego main. Nie edytuj produkcji, baseline configu ani historycznych testów/artefaktów. Historyczne helpery można reuse po kontroli, ale nie uruchamiaj całego Stage B suite (237 updates).
- Przerejestruj exact hashes, fixture, profile, pary porównań, tolerancje, liczniki, limity i spodziewane odmowy zanim ruszy pierwszy forward. Record synthetic_audit=true i prawdziwy SHA fixture. Nie podszywaj fixture pod canonical dataset.

## Sekwencja bramek

1. **R0, zero updates:** inherited evidence matrix ze sprawdzeniem niezmienionych źródeł; C02 independent log stats oracle na actual encoder_cont/target_scale obu grup; fallback poison, renaming, observed-only/assessment sentinel, unknown/no-observed refusal; Stage A synthetic grids/masks/window keys; loss/length/LR algebra; dziewięć pól callback restore i comparator perturbation. Normalizer FP64 oracle vs FP32 atol=1e-5/rtol=1e-5; keys/mapping exact. Wszystkie compatibility mismatch odrzucone przed deserializacją.
2. **CPU:** U8, P4 i R4 w osobnych procesach. Nowy production builder, dwie synthetic grupy o różnych skalach, po dwa wybrane training windows i po dwa validation windows per group; context48/horizon12. Hidden8/continuous4/head1/LSTM1/dropout.1/batch2, seed42, shuffle, accumulation1, workers0, FP32, LR3e-4, clip1, SWA OFF. CPU threads1. Model/training loop/optimizer/callbacks produkcyjne. Porównaj U z P+R exact zgodnie z planem.
3. **CUDA:** dopiero po CPU PASS; świeże C1/C2/C3 po8, P4/R4, osobne sekwencyjne procesy na jednej GPU. Te same fixture i architektura. Nie reuse starych B weights/reference trajectories. Wszystkie trzy fresh pary i każdy Ck vs P+R; atol/rtol loss1e-5/1e-4, gradient norms1e-5/1e-3, każdy parameter1e-6/1e-4, LR exact. Symmetric elementwise rule z planu; nie dobieraj tolerancji do wyników.
4. **Resume:** P stop_after_epoch=1, po walidacji, total8/max_epochs4 pozostaje ten sam w P/R. Owned LAST i ten sam run directory; publiczny Trainer.fit(ckpt_path=...). Przed pierwszym resumed forward exact model/optimizer/scheduler/RNG/generators/counters/callback states także CUDA, z current_score i wszystkimi dziewięcioma polami. Potwierdź native dispatcher raz i tę samą instancję callbacku. Zachowaj trace następnego batchu, bez pominięć/powtórzeń. Ścieżki między niezależnymi fresh runs porównuj przez role, w resume dokładnie.
5. **Finite:** CPU i CUDA po trzy izolowane probes input NaN / target Inf / gradient NaN, przed pierwszym AdamW. Oczekiwany produkcyjny błąd, zero calls i brak valid checkpointu. Catch przez audit containment jako pierwszy = FAIL. Bez late/post-update injections.
6. **Owned serialization, zero updates:** z nowych synthetic trained artefaktów BEST/niekońcowy LAST, niezależny oracle BEST z wszystkich epoch val_loss i tie-break; nowy proces Registry.verified BEST load, wszystkie tensors/Q/map/stats/schema/lineage zgodne, reconstructed normalization zgodna. Weights-only export/load bez fit, nowa optymalizacja a nie resume. Stare contracty także weights-only odmawiają; negatywne manifesty własne, nie historyczne trained pliki. Nie wywołuj evaluator scoring.
7. **Raport i stop:** zapisz komplet evidence, oddziel historyczne źródła dowodu od wykonanych testów, wszystkie statusy i actual budgets. Brak wymaganej CUDA = BLOCKED/NOT RUN, nie pełny PASS. Nigdy nie uruchamiaj kolejnego etapu automatycznie.

## Profile i containment

Użyj nmd-numerics-2: CPU deterministic strict, CUDA deterministic warn-only seeded; FP32/ieee/highest, AMP/TF32 OFF, cuDNN deterministic=true/benchmark=false, CUBLAS_WORKSPACE_CONFIG=:4096:8. Sprawdź efektywne flagi po Trainer init i przy wykonaniu. Jedyny zaakceptowany nondeterministic warning to upsample_linear1d_backward_out_cuda. Nie zmieniaj interpolacji, pakietów, drivera ani tolerancji. Zapisz faktyczne GPU i stack; nie zakładaj Ti na podstawie AGENTS.

Dopuszczone ścieżki danych to wyłącznie nowe owned synthetic fixture. Nie czytaj real/canonical parquet nawet po to, by filtrować train: plik może zawierać test. Odczyt historycznych metadata/hash manifests dozwolony, ale oznacz go inherited, nie fresh byte verification. Zablokuj produkcyjny main, evaluator, Optuna, dostęp do real danych i historycznych model artifacts w harnessie; opisz faktyczny poziom containment bez udawania izolacji systemowej.

## Nieprzekraczalny budżet

- CPU16 + CUDA32 = **48 AdamW calls**, wlicz LR=0 i failed calls; count attempt przed delegacją, completions osobno, parent aggregate.
- 8 positive fit workers, 6 negative; każdy dodatni ≤8 steps/4 epochs, P/R4; R0/serialization/negative zero calls.
- Cała prerejestrowana execution ≤600 s wall-clock; każdy worker≤90 s. Parent watchdog zabija process group i zachowuje evidence.
- Wszystkie forwards≤256, backwards≤64, generated candidates≤1024, fit używa4 train+4 validation windows. Sanity validation i negative probes wliczane.
- Lokalnie; zero płatnego cloud. Bez retry, rozszerzania budżetu lub tolerancji. FAIL/ERROR/timeout: przerwij zależne fazy, raportuj i zatrzymaj się przed poprawkami. CPU-only ogranicza aktualizacje do16 i nie certyfikuje GPU.

## Deliverables i DoD

Nowy ignorowany `ml/models/stage_c_training_recertification/<unique-id>/`: raw states/traces/checkpoints/logs. Lekkie synthetic metadata w `experiments/`: preregistration, effective config, hashes, invocation/exit/counters/timing, comparisons/warnings, ownership. Raport `docs/STAGE_C_TRAINING_RECERTIFICATION_REPORT.md`: CPU/CUDA/overall verdict, pełna macierz bramek bez ukrytych SKIP/XFAIL, exact source/config/environment identity, inheritance matrix i remaining limits. Nie aktualizuj training_recertification w baseline configu w tej kampanii; raport stanowi osobną atestację.

PASS wymaga wszystkich R0–R2/finite/serialization/restore bramek oraz zgodności liczników. Synthetic val_loss jest instrumentacją do replay/checkpoint selection, nie performance evaluation; brak kryterium poprawy loss. Nie twórz MARD/MAE/RMSE/PICP, klinicznych wniosków lub rankingów jakości. Zachowaj clinical weighting2.5/hypo70/factor2, raw quantile interpretation i calibration DEFERRED.

Na końcu sprawdź diff: wyłącznie nowe test-only/metadata/report artefakty w autoryzowanym zakresie; production/config/dane bez zmian. Podaj faktyczny koszt i ograniczenia: smoke nie certyfikuje długiego treningu, pełnej architektury/VRAM, mid-epoch, AMP, DDP ani cross-device. **STOP przed pełnym Baseline training i przed jakąkolwiek remediation.**
