# Codex execution task — launch preflight and full-model capacity only

2026-09-15 · nmd-baseline-launch-preflight-1 · **NOT ACTIVATED**

Execute only after an explicit user instruction activates this task. Deliver a launch readiness report and synthetic capacity evidence, then STOP. This is not authorization for full training, real data access, test performance, Optuna, weights-only or production fixes.

## Inputs and preflight

Read AGENTS.md, BASELINE_V1_TRAINING_LAUNCH_PLAN.md, current C02 recertification report and closure review, Stage B final review/finalization, Stage C execution contract, baseline_v1_stage_c.json and current training/registry/numerical/finite code. Reference HEAD4f0cf189b54cd2cfc2ad288da1b6f8c85461b66d on research/baseline-audit. Check HEAD/dirty and all plan hashes before any GPU work; classify docs-only drift, stop for unreviewed production/config/stack drift. Preserve existing evidence byte-for-byte.

Create only a new test-only capacity harness and new owned synthetic artifacts/metadata/report. No production/config changes, package installation, old checkpoint deserialization, registry relocation, export_weights or writes to old registries. Do not call production main, real loader, load_test_data or evaluator. Do not read/hash canonical parquet. Freeze exact synthetic inputs/config/harness/profile/tolerances/budget before first forward. Attach the narrowed C02 PASS report hash to explain the stale config recertification string without editing it.

Inspect real RTX3070/8192MiB, UUID, driver, torch/CUDA/cuDNN/dependencies, RAM, free disk, GPU baseline usage/other processes. One GPU worker, CPU threads1/workers0. No competing compute jobs; do not kill them. Require numerical profile cuda-seeded nmd-numerics-2 with FP32 ieee/highest, AMP/TF32 OFF, deterministic warn-only, cuDNN deterministic/no benchmark, CUBLAS_WORKSPACE_CONFIG=:4096:8. Historical stack is a reference to verify, not permission to update it.

## Exact smoke

Use production dataset builders, GroupNormalizer observed-train-encoded-subject-v2 and model/training orchestration. Config architecture hidden64/continuous16/heads4/LSTM1/dropout.3, context48/horizon12, Q7, full ordered features. Match actual production embedding/cardinality shapes from safe existing schema metadata; if unavailable without patient access, mark CONDITIONAL/HOLD and do not claim exact production capacity. Do not silently use tiny recertification schema.

Generate up to4096 synthetic candidate windows, deterministic seed42, positive targets with distinct subject scales, all required features/masks; choose128 train and128 validation full-length windows before execution. Train64, validation128, accumulation1, shuffle train, no validation dropping. Exactly one epoch/two base AdamW calls through native production Trainer. Declare epochs1/total2 as smoke-only budget overrides; every other model/optimizer/loss/numerics parameter matches baseline. Count LR0 call; require nonzero LR and actual update on second, AdamW moments present. Do not alter warmup. Allow native fresh sanity validation and final full128 validation.

Measure actual tensors/shapes/parameter count, finite loss/gradients/state and clip≤1+1e-5. Synchronize CUDA and capture allocated/reserved/peak memory, driver free/used/total/process usage at baseline, construction, each forward/backward/update, validation and save; poll at least1Hz. Keep optimizer state resident during validation. Do not empty_cache between phases. Record RSS, available RAM, timestamps, checkpoint byte size and write duration. Save via production boundary/registry, verify synthetic completed BEST at its original owned location; no inference scoring, export or resume campaign.

## Acceptance and hard resource budget

CAPACITY PASS only if exact production shapes are established; actual RTX3070 8GB; exactly2 AdamW attempts/completions and2 backwards; finite states, expected nonzero second update, complete128 validation, verified full checkpoint, no OOM/errors/unreviewed nondeterministic operators; sampled GPU free≥1024MiB and peak reserved+baseline non-process usage≤total−1024MiB. Sampling limitations must be reported. Allowed inherited nondeterministic operator is upsample_linear1d_backward_out_cuda only. Do not require loss improvement.

Maximum one GPU smoke worker, no retry; ≤2 optimizer attempts/completions, ≤2 backwards, ≤6 model forwards including sanity/validation, ≤4096 generated candidates,256 selected windows; ≤300s GPU worker, ≤600s total execution after preregistration. Parent watchdog terminates whole process group and records partial counters/evidence. ≤2GiB newly written smoke artifacts and require≥5GiB free initially; checkpoint-size excess ends INCOMPLETE, no automatic enlargement. Local only, zero cloud purchases. Count failed calls before delegation and successful calls separately. No CPU fallback or batch search.

OOM/headroom failure => FAIL/HOLD. No automatic32 batch, accumulation2, AMP or shortened context. Those need a separately approved versioned variant and targeted certification. Missing cardinality proof/device/evidence => INCOMPLETE/HOLD. Preserve every failure, no rerun-until-green or production remediation.

## Launch review deliverable

Write docs/BASELINE_V1_LAUNCH_PREFLIGHT_REPORT.md and lightweight experiment metadata under a new unique experiments/baseline_v1_launch_preflight ID; binaries/logs only ignored ml/models/baseline_v1_launch_preflight/<ID>. Include preregistration, hashes, effective config/profile/device, memory phase table/peaks/margin, counters/time/exit, parameter shapes, artifact hashes, checkpoint verification, exceptions and full unrounded evidence.

Report separately CAPACITY_STATUS and FULL_LAUNCH_READINESS. Full launch remains HOLD while data-access authorization for the shared parquet, safe source/schema freeze, monitoring/recovery journal, disk sizing and actual N/B/T/runtime budget are unresolved. Do not read real data to compute those counts in this task. Provide formulas B=floor(N/64), T=60B, validation ceil(Nval/128), measured checkpoint size S and proposed full-run disk reserve2×60×S+max(5GiB,log allowance). Do not confuse synthetic timing with guaranteed real runtime.

Inspect current recovery limitations statically: no directory fsync, no multi-file transaction, potential orphan generation collision, no automatic older-LAST fallback, failure events do not update run.status. Retain all60 generations for first run; never add pruning or repair pointers. Describe requirements for external monitoring and later explicit recovery approval.

Review diff and source hashes at exit: only authorized new harness/report/metadata; unchanged source/config/historical evidence. End with NO FULL TRAINING / NO REAL DATA / NO TEST EVALUATION / NO OPTUNA. **STOP even if capacity passes.**
