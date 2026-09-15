# Stage C training recertification — authorized host-GPU attempt

2026-09-15 · plan `nmd-c02-training-recertification-1` · branch `research/baseline-audit` · HEAD `2fed05028aaa303ab06362cee96d02c3221fece8`.

## Verdict

**Overall: FAIL / INCOMPLETE.** Host GPU preflight passed. R0, CPU strict replay and CUDA seeded reproducibility/resume passed. The campaign stopped at **CPU_N_target FAIL**; remaining finite probes and owned serialization were not run. No automatic retry or remediation followed the failure.

- **CPU trajectory + epoch-boundary resume: PASS**, U8 versus P4+R4 exact. CPU finite-safety suite: **INCOMPLETE / one FAIL**.
- **CUDA trajectory + epoch-boundary resume: PASS**, C1/C2/C3 and all six registered comparisons. CUDA finite-safety: **NOT RUN**. This is not full CUDA recertification PASS.
- **Budget: 48/48 optimizer attempts, 48 completions**, CPU16 + CUDA32. No optimizer calls in either executed negative probe.
- **NO PERFORMANCE EVALUATION.** Synthetic validation was replay/checkpoint instrumentation only.

## Two distinct attempts and preserved history

| Attempt | Context | Result | Updates |
|---|---|---|---:|
| `20260915T140937Z` | restricted Codex sandbox | preflight ERROR before R0; nvidia-smi exit9, parent exit1 | 0/48 |
| `20260915T143739Z` | newly authorized host GPU access | GPU preflight PASS; later CPU_N_target FAIL | 48/48 |

The original report was preserved byte-for-byte before this update at [report_original.md](../experiments/stage_c_training_recertification/20260915T140937Z/report_original.md), SHA256 `397648d9a283aac0405fac033c48adcb164169eb942ba9b9cc04dd1aaccb4b8e`. Original launch_failure.json and README were not changed. Infrastructure diagnosis established that the sandbox blocked GPU access; the host driver worked. This campaign is a new explicit authorization, not an automatic retry of the old campaign.

New unique directories, created with exist_ok=False:
- Ignored raw artifacts: `ml/models/stage_c_training_recertification/20260915T143739Z/`.
- Lightweight evidence: [20260915T143739Z](../experiments/stage_c_training_recertification/20260915T143739Z/).

Do not reuse/overwrite either attempt directory. The report is the updated index; saved attempt evidence remains historical.

## Minimal diagnostic change and preflight

Only test harness `ml/tests/stage_c_recertification/campaign.py` changed: separate bounded preflight subprocess, structured result capture, required GPU assertions and explicit failure record before workers. No production code, configuration, architecture, loss, driver or packages changed. CPU/CUDA fit, comparison tolerances and negative probe implementations were left unchanged. Static ast.parse passed before execution.

Invocation once from repository root, explicitly outside the restricted sandbox:

```bash
python3 ml/tests/stage_c_recertification/campaign.py
```

Preflight used repository `.venv/bin/python -B`, resolved `/usr/bin/nvidia-smi`, captured stdout/stderr/exit and visible `/dev/nvidia*`. Result: exit0, stdout `NVIDIA GeForce RTX 3070, 615.71.09, 8192 MiB`, stderr empty. `/dev/nvidia0`, nvidiactl, modeset, uvm, uvm-tools and caps visible. PyTorch CUDA available=true, count=1, exact device name `NVIDIA GeForce RTX 3070`; no silent CPU fallback. CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES/LD_LIBRARY_PATH/LD_PRELOAD unset. Full PATH/cwd/invocation in [preflight.json](../experiments/stage_c_training_recertification/20260915T143739Z/preflight.json).

Preflight exit0 in 11.780s, before R0/forward/updates. Both initial flags and configured effective flags recorded. Configured profile: deterministic algorithms=true; CUDA warn_only=true, CPU warn_only=false; cuDNN deterministic=true/benchmark=false; global/matmul/cuDNN precision=ieee, matmul=highest, AMP off, precision32-true, CUBLAS_WORKSPACE_CONFIG=:4096:8. Actual per-worker profiles and runtime verifications are in worker results. These are process settings required by the existing contract, not package/system changes.

Fresh worker stack: Python3.14.7, torch2.11.0+cu130, Lightning2.6.1, PF1.7.0, NumPy2.4.4, pandas2.3.3, CUDA build13.0, cuDNN91900. Matches the plan. GPU is RTX3070, not assumed Ti.

## Identity and preregistration

All 10 plan snapshot hashes matched before execution. Relative to reference c2e27a28824cd9ec015c362b7e8df65f3c0902dc, HEAD adds only the two recertification documents; no production/config drift. Initial dirty files were the previous report, harness and attempt evidence; none discarded.

Preregisration SHA256: `ffad05facb098dc2e350f407c0d095b9deb6314396700c12e475ae26dc66e9e2`. Fixture SHA256: `3334c2a6eb4f44e3990da08e8ac24578dd716a794f9941192485d2712127683c`. Ordered window hash: `dd4380903969d185cc37734e4a29d06fa760e125170776642ad7a2dfa1ba8fd4`. Owned serialized dataset hash: `94499485b4daf0653d50a691eee19925aac453971376478c3ded244a95fc1b6b`. These are actual synthetic artifacts; canonical Stage A hash remains inherited protocol identity, not a fresh data read.

Fixture: two synthetic subjects, 61 regular bins each; A=55+.8i and B=180+1.5i; bin10 unobserved with causal ffill. Current production builders and observed-window qualification, then deterministic full decoder starts48/49 per subject: four train/four assessment windows, encoder48/decoder12. Hidden8, continuous4, head1, LSTM1, dropout.1, batch2, seed42, shuffled training, accumulation1, workers0, FP32, LR3e-4, clip1, SWA OFF, CPU threads1. Four epochs/eight updates; P stops after epoch index1, retaining full eight-step schedule. No baseline config rewrite.

| Source | SHA256 |
|---|---|
| `ml/scripts/train_tft_population_v2.py` | `dcdb6c738d70958a4a105d81c965f4dd17af0db35143ba77d98ba5b639f24572` |
| `ml/scripts/baseline_training.py` | `0cdef5a6311e3804cda69dfbc936cafb5fa4b560e12e53c71250c0b569018d74` |
| `ml/scripts/checkpoint_registry.py` | `cb3b9ca3495faa06ee039c28c2085fb3a9ee1c246f9cdeba766d11707d25df3d` |
| `ml/scripts/finite_training.py` | `3e0adea81cd1fdef6fec933213b1c1cb84b5fe8172716c7a69acab78e4abbf6a` |
| `ml/scripts/numerical_profile.py` | `5bf981d01877b1900a0c8f0426a5aa27966ec9d6a64ceb9c8639e0306c0e5630` |
| `ml/scripts/observed_windows.py` | `f530d31ff6c41f50c28b86e02ad70af5c749b763799377ab11df872d916417a9` |
| `ml/scripts/temporal_protocol.py` | `e0f0661387dffe1563669284fcfeb009ad7a6bab15712011e6f26ef9637889f0` |
| `ml/scripts/diagnostics/callback_restore.py` | `50f5071efade402dd946b808546bea0b0fb9d50ef3f2acfe81ae96ea96e5b7c3` |
| `configs/baseline_v1_stage_c.json` | `a85bb26214253a855af066b590565b3b8b45beebb34302e686cc36bc752e1f35` |
| `docs/STAGE_C_TRAINING_RECERTIFICATION_PLAN.md` | `86101d4782444af3be5dc2297b3b2c6f0c2abbf320352fe66735fe4726996c71` |
| `docs/CODEX_STAGE_C_TRAINING_RECERTIFICATION_TASK.md` | `d9ba228fac8a9075ce8c44530fc23a9ad8a368e16b781bb3f9dd00e770268f04` |
| `ml/tests/stage_c_recertification/campaign.py` | `56afa2d5adcf7b33d59cc6b2dda826f9e515f9d84c34f9eb133261a132034192` |

## Gate matrix

| Gate | Actual result |
|---|---|
| Host GPU access / expected device / numerical preflight | PASS |
| R0 source inheritance | PASS |
| R0 C02 independent log-statistics oracle, actual encoder_cont/target_scale, poisoned fallback | PASS |
| R0 observed/assessment sentinels, renaming, unknown/no-observed refusal | PASS |
| R0 Stage A synthetic grid/masks/window keys | PASS |
| R0 callback nine fields, comparator perturbation | PASS |
| R0 loss/length/LR algebra without updates | PASS |
| R0 compatibility refusal before deserialize | PASS |
| CPU U8/P4/R4 and strict comparison | PASS |
| CPU exact native epoch-boundary restore | PASS |
| CUDA C1/C2/C3/P4/R4 and six comparisons | PASS |
| CUDA exact native epoch-boundary restore | PASS |
| CPU input NaN | PASS, production model-input refusal, zero updates, no valid BEST/LAST |
| CPU target Inf | FAIL, production model-input refusal disagrees with target-specific test expectation |
| CPU gradient NaN | NOT RUN after FAIL |
| CUDA input NaN / target Inf / gradient NaN | NOT RUN after FAIL |
| Owned trained BEST/LAST serialization, independent BEST oracle | NOT RUN after FAIL |
| Weights-only export/load | NOT RUN after FAIL |

Executed workers:12 exit0,1 exit1; remaining5 workers NOT RUN. R0 has seven named PASS checks. No required gate hidden as SKIP/XFAIL. Training wrote synthetic checkpoints and resume loaded its own LAST, but this does not replace the unexecuted standalone serialization gate.

## CPU and CUDA comparisons

CPU U versus P+R: exact batch keys/digests/order, initialization, loss/LR/pre/post gradient norms/validation, all1122 named parameter tensors and complete final semantic state (model/optimizer/scheduler/RNG/callbacks). Independent run path prefixes normalized only for semantic comparison; paths at own checkpoint restore exact.

CUDA pairs C1–C2, C1–C3, C2–C3, C1–PR, C2–PR, C3–PR: **all PASS**. Maximum observed absolute difference is **0** for every recorded loss, validation value, LR, pre/post gradient norm and each of1122 parameter tensors. Zero exceedances and zero normalized error for nonzero tolerances. Full per-tensor results retained, compact [comparison_summary.json](../experiments/stage_c_training_recertification/20260915T143739Z/comparison_summary.json). Observed equality does not upgrade seeded CUDA to a bitwise guarantee.

Frozen symmetric rule: abs(x-y) <= atol + rtol*max(abs(x),abs(y)); loss/validation1e-5/1e-4, gradient norms1e-5/1e-3, parameters1e-6/1e-4, LR exact. No thresholds adjusted after results.

Both CPU/CUDA restore: model/optimizer/scheduler/counters/RNG/generators exact before first resumed forward. BoundaryCheckpoint monitor, best_model_score, best_model_path, current_score, dirpath, best_k_models, kth_best_model_path, kth_value, last_model_path all true. EarlyStopping and GradientNormLogger fields all true. One native dispatcher call, same restored callback instance. Evidence: CPU_R_restore.json and CUDA_R_restore.json. No skipped/repeated batch in replay.

Only recorded nondeterministic-operator warning: accepted upsample_linear1d_backward_out_cuda. Other retained warnings include dataloader worker count, existing owned checkpoint directory, TreeSpec deprecation and six modules in eval mode at resume; no changes made in response. Warnings are retained in raw logs/results, not suppressed to obtain PASS.

## First failure: CPU_N_target

Harness injection at on_train_batch_start modifies `y[0][0,0]` to Inf. It expects a production FloatingPointError containing `target`. Actual production exception is `Nonfinite state: model input` from FiniteModel.forward → require_finite(x), then harness assertion at campaign.py:295 fails. The failure happened before backward and before AdamW delegation; ledger stayed48/48. The negative run's registry has best=null and last=null.

Local PF `_timeseries.py` collate returns the same target tensor in x['decoder_target'] and y[0] (lines2533/2539). Thus in-place target injection also contaminates model input; the earlier production input gate rejects it. This is a probe/expectation mismatch, not evidence of an Inf optimizer update. It remains **FAIL** under the registered test; no reclassification or repair was applied. No additional runtime probe was executed after failure.

## Actual budget and exits

| Counter | Actual | Limit |
|---|---:|---:|
| AdamW attempts |48|48|
| AdamW completions |48|48|
| CPU positive updates |16|16|
| CUDA positive updates |32|32|
| Forward calls |82|256|
| Backward calls |48|64|
| Candidate windows |458|1024|
| Positive fit workers |8|8|
| Negative workers executed |2|6|
| Execution after preregistration, seconds |210.735|600|
| Longest worker, seconds |55.985|90|

Parent exit1. Preflight11.780s separately; total measured preflight + campaign approximately222.515s. All attempts/completions counted before/after underlying AdamW respectively, including LR0; negative failures add no AdamW calls. No retries, extra optimizer steps, tolerance changes or budget extensions.

| Worker | Exit | Seconds |
|---|---:|---:|
| R0 | 0 | 55.985 |
| CPU_U | 0 | 14.883 |
| CPU_P | 0 | 6.723 |
| CPU_R | 0 | 8.576 |
| CPU_compare | 0 | 4.971 |
| CUDA_C1 | 0 | 36.611 |
| CUDA_C2 | 0 | 20.291 |
| CUDA_C3 | 0 | 20.291 |
| CUDA_P | 0 | 12.831 |
| CUDA_R | 0 | 14.283 |
| CUDA_compare | 0 | 6.122 |
| CPU_N_input | 0 | 4.570 |
| CPU_N_target | 1 | 4.520 |

## Inherited evidence and limitations

Stage A upstream preprocessing/splits/causality and canonical dataset identity remain historical evidence for unchanged sources, not newly executed full31-test regression or a canonical parquet byte check. Fresh synthetic integration is limited to the R0 checks above. Historical Stage B long scheduler/protection evidence and callback implementation remain inherited; fresh short CPU/CUDA trajectories were executed here. Stage C mechanical19 PASS remains historical; not rerun or relabelled. Clinical weighting2.5/hypo70/factor2 unchanged; calibration NOT IMPLEMENTED/DEFERRED.

Containment: Python audit hook and blocked production entrypoints/Optuna/real-data and historical artifact opens; not an OS sandbox for arbitrary native I/O. Host GPU execution does not authorize patient-data access. No real dataset or historical trained checkpoint read, no evaluator scoring, Optuna, calibration fit or full Baseline training. No long-run, full hidden64/batch64 VRAM, mid-epoch, partial accumulation, AMP, DDP, cross-device/cross-stack or clinical/calibration claims.

## Modified files and final state

- `ml/tests/stage_c_recertification/campaign.py`: minimal preflight recording/gate only.
- `experiments/stage_c_training_recertification/20260915T140937Z/report_original.md`: immutable byte-copy of previous report.
- `experiments/stage_c_training_recertification/20260915T143739Z/`: new attempt metadata, comparisons and restore evidence.
- `docs/STAGE_C_TRAINING_RECERTIFICATION_REPORT.md`: this updated report.
- Ignored new raw run under ml/models, containing only owned synthetic artifacts/logs.

Production/config changes: none; historical launch evidence unchanged. Git HEAD unchanged, branch research/baseline-audit, dirty/untracked report/harness/experiments, nothing staged, no commit/push. Final syntax/source hash and git diff --check verification recorded in final_verification.json.

**STOP.** No repair, retry, full baseline training or subsequent stage is authorized by this result.
