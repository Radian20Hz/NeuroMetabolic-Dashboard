# Stage C training recertification — execution stopped

2026-09-15 · plan `nmd-c02-training-recertification-1` · **ERROR / INCOMPLETE — recertification NOT PASSED**.

## Wynik

Uruchomienie zakończyło się błędem infrastrukturalnym w preflight procesu nadrzędnego, **przed zapisem prerejestracji, R0, utworzeniem fixture i uruchomieniem pierwszego workera**. Wykonano **0/48 prób AdamW,0 zakończonych aktualizacji,0 forward,0 backward**. Nie ma wyniku pozwalającego potwierdzić albo odrzucić poprawność treningu po C02.

- **CPU strict: NOT RUN.**
- **CUDA seeded: NOT RUN.**
- **Overall: ERROR / INCOMPLETE**, nie PASS i nie potwierdzony błąd produkcyjnej mechaniki ML.
- **NO PERFORMANCE EVALUATION.**

Zastosowano no-automatic-retry: jedno uruchomienie, bez ponowienia, poprawiania harnessu po błędzie, zwiększania budżetu, zmian tolerancji lub uruchamiania zależnych faz. Zatrzymano na raporcie. Niewykorzystane48 aktualizacji nie stanowi automatycznej zgody na kolejną próbę.

## Git i impact review

Branch `research/baseline-audit`; HEAD `2fed05028aaa303ab06362cee96d02c3221fece8`. Repo czyste przed przygotowaniem harnessu. Względem punktu odniesienia `c2e27a28824cd9ec015c362b7e8df65f3c0902dc` zmieniono wyłącznie dwa dokumenty recertyfikacji (+166 linii), bez driftu produkcji/configu. Nie wykonano fetch.

Przeczytano AGENTS.md, task i plan recertyfikacji, raport/plan remediation Stage C, final review Stage B i aktualizację callback PASS, config Stage C oraz C02, registry, baseline_training, finite i numerical sources. Odczytano lokalny `ModelCheckpoint.state_dict/load_state_dict` i historyczne helpery instrumentacji; nie uruchomiono historycznych runners/suites. Dziewięć natywnych pól i istniejąca korekta current_score pozostają wymaganiem, którego nie przetestowano ponownie w tej próbie.

Świeże porównanie wszystkich10 SHA-256 wymienionych w planie: **10/10 zgodne**. Pełne wartości expected/actual znajdują się w [launch_failure.json](../experiments/stage_c_training_recertification/20260915T140937Z/launch_failure.json). To statyczna kontrola plików, nie wykonane R0 ani świeży Stage A/B PASS.

## Dokładny błąd i polecenie

Uruchomiono raz, z root repo:

```bash
python3 ml/tests/stage_c_recertification/campaign.py
```

Parent utworzył puste katalogi runu `20260915T140937Z`. Podczas budowania prerejestracji wykonał:

```text
subprocess.check_output([
  'nvidia-smi',
  '--query-gpu=name,driver_version,memory.total',
  '--format=csv,noheader'
], text=True)
```

Wywołanie zakończyło się `subprocess.CalledProcessError`, **exit9 subprocessu / exit1 parenta**, w `parent()` przy pobieraniu `cuda_preflight`. Nie osiągnięto `save(...preregistration.json...)`, inicjalizacji ledger ani pętli workers.

Wcześniejszy osobny odczyt inventory tym samym poleceniem narzędzia zakończył się exit0 i zwrócił:

```text
NVIDIA GeForce RTX 3070, 615.71.09, 8192 MiB
```

Nie pozwala to uznać CUDA za sprawne wewnątrz workera ani za niedostępne ogólnie. **Przyczyna różnicy między bezpośrednim odczytem a subprocess parenta pozostaje NIEUSTALONA.** Output dziecka był przechwycony przez `check_output`, ale nie został trwale zapisany przez harness przed wyjątkiem. Nie przypisujemy błędu driverowi, PyTorch, sandboxowi lub C02 bez dowodu. Nie ponawiano probe w celu diagnozy.

Narzędzie podało około0.1s wall time nieudanego polecenia. Dokładny czas parenta nie został zapisany przed wyjątkiem; worker wall time i czas fit wynoszą0, ponieważ żaden worker nie został uruchomiony. Zewnętrzne watchdogi90/600s nie zostały osiągnięte ani sprawdzone wykonaniowo.

## Budżet i macierz bramek

| Element | Limit / plan | Faktyczne wykonanie |
|---|---|---|
| AdamW attempts, także LR0 i failed calls |48|0|
| AdamW completions |48|0|
| CPU updates |16|0|
| CUDA updates |32|0|
| Positive fit workers |8|0|
| Negative workers |6|0|
| Forward, także sanity/probes |≤256|0|
| Backward |≤64|0|
| Generated candidate windows |≤1024|0|
| Qualified fit windows |4 train+4 validation|0, fixture nieutworzona|
| Worker runtime |≤90s|0; workers nieuruchomione|
| Execution watchdog |≤600s|nieosiągnięty; brak powtórki|
| Cloud spend |0|0, wyłącznie lokalne polecenia|

| Bramka | Status | Dowód / ograniczenie |
|---|---|---|
| Branch/HEAD i production/config drift |PASS, static|wyłącznie dokumentacyjne commity;10 hashes planu zgodne|
| Składnia harnessu |PASS, static|`ast.parse`, bez importów ML|
| Parent GPU preflight |ERROR|subprocess exit9; parent exit1|
| Prerejestracja i fixture hashes |NOT RUN / NOT WRITTEN|wyjątek przed zapisem; brak fixture|
| R0 containment positive controls |NOT RUN|worker nieuruchomiony|
| R0 C02 log stats, mapping, encoder_cont/target_scale |NOT RUN|brak importu ML/builders w execution|
| R0 observed-only/assessment/rename/unknown/no-observed |NOT RUN|jw.|
| R0 Stage A grids/masks/window keys |NOT RUN|jw.|
| R0 loss/valid lengths/LR algebra |NOT RUN|jw.|
| R0 callback dziewięć pól/comparator |NOT RUN|jw.|
| R0 pre-deserialization compatibility |NOT RUN|jw.|
| CPU U8/P4/R4, strict compare |NOT RUN|zależne od R0|
| CUDA C1/C2/C3/P/R, sześć porównań |NOT RUN|zależne od CPU PASS|
| CPU/CUDA exact boundary restore i dispatcher |NOT RUN|brak checkpointu/fit|
| Sześć finite probes |NOT RUN|brak workers|
| Owned synthetic trained BEST/LAST roundtrip |NOT RUN|brak nowych trenowanych artefaktów|
| Weights-only export/load |NOT RUN|jw.|
| `git diff --check` i końcowy scope review |PASS, static|nowe test-only/metadata/report, produkcja i config bez zmian|

Nie ma SKIP/XFAIL maskujących wymagane bramki. Brak evidence jest jawnie NOT RUN; nie zaliczono żadnego testu ML. Zero aktualizacji wynika ze ścieżki wykonania parenta zakończonej przed startem workers, a nie z odczytu nieistniejącego ledgera.

## Dziedziczone dowody

| Obszar | Co pozostaje dziedziczone | Czego ta próba nie potwierdza |
|---|---|---|
| Stage A upstream preprocessing/splits/causality |historyczne dowody dla niezmienionej logiki; brak zmian produkcyjnych w tasku|świeża regresja grid/mask/tensor integration|
| Stage A canonical dataset |historyczna identyfikacja z dokumentacji|fresh byte hash, otwarcie danych; niczego takiego nie wykonano|
| Stage B loss/finite/scheduler/sampler/callback implementation |niezmienione źródła, historyczne komponentowe wyniki|nowa trajektoria CPU/CUDA po C02|
| Stage B current_score correction |historyczny PASS opisany na początku finalization, nie stary niższy FAIL|ponowne dziewięć pól/dispatcher na nowym resume|
| Registry ownership/protocol |bieżące źródło i snapshot zgodne z planem|nowy synthetic trained BEST/LAST/weights-only|
| Stage C19 mechanical checks |historyczne19 PASS; nie powtarzano|training/backward/optimizer po normalizer revision|
| CPU strict / CUDA seeded profile |zatwierdzona polityka nmd-numerics-2|efektywne flagi w nowym Trainer/worker|

Historyczny stos z planu: Python3.14.7, torch2.11.0+cu130, Lightning2.6.1, PF1.7.0, NumPy2.4.4, pandas2.3.3, CUDA build13.0/cuDNN91900. **Nie jest to świeży import/stack verification tej próby.** Żaden ML worker nie doszedł do kontroli wersji lub flag. Bezpośredni GPU inventory opisano osobno powyżej.

## Artefakty i zmiany

- `ml/tests/stage_c_recertification/campaign.py`: nowy test-only harness; przygotowany i sprawdzony składniowo, **niezwalidowany wykonaniowo**, z ujawnionym niezabezpieczonym błędem parent preflight. Zachowany bez poprawki po wyniku.
- `experiments/stage_c_training_recertification/20260915T140937Z/launch_failure.json`: lekkie metadata faktycznego błędu, source hashes, invocation i zerowe liczniki. To **post-failure record**, nie prerejestracja.
- `ml/models/stage_c_training_recertification/20260915T140937Z/launch_failure.json`: lokalna ignorowana kopia; brak model state/checkpoint/fixture.
- `docs/STAGE_C_TRAINING_RECERTIFICATION_REPORT.md`: ten raport.

Brak zmian produkcji, baseline configu, historycznych testów/artefaktów, danych, clinical loss, CUDA/PyTorch i tolerancji. Nie zaktualizowano `training_recertification` w configu. Brak staging/commit/push.

## Ograniczenia i STOP

Recertyfikacja pozostaje nieukończona. Następna próba wymaga osobnej decyzji obejmującej diagnozę preflight, review harnessu i nową prerejestrację; w tej sesji nie wykonano remediation ani retry. Nie ma podstaw do nowych twierdzeń o CPU/CUDA training, calibration, clinical reliability lub performance. Clinical weighting pozostaje2.5/hypo70/factor2, SWA OFF, calibration DEFERRED. Długi/full-size trening, mid-epoch, AMP, DDP, cross-device i cross-stack są poza zakresem.

**STOP.** Nie uruchomiono pełnego Baseline training, Optuny, real-data evaluation ani kolejnego etapu.
