# Update 2026-09-15 — accepted narrowed C02 closure scope

**Final overall verdict: FAIL / INCOMPLETE.** New authorized attempt `20260915T152140Z-closure` completed read-only trained BEST checks, then failed its CPU target-Inf acceptance assertion. Remaining negative probes were not run. No repair/retry followed. Original48 optimizer updates were not repeated; this closure added **0 base AdamW calls**.

**Weights-only: NOT RECERTIFIED AFTER C02 / DEFERRED.** The user explicitly accepted the closure review amendment. No export/load was performed and no weights-only PASS is claimed. Registry semantics and ownership remain unchanged.

## Final CPU/CUDA status and evidence boundaries

| Evidence | Status | Origin |
|---|---|---|
| CPU strict U8 versus P4+R4, full boundary restore | PASS, inherited | 20260915T143739Z; not repeated |
| CUDA C1/C2/C3 and P4+R4, six comparisons | PASS, inherited; recorded discrepancies0 | same original campaign; seeded contract only |
| CPU_R/CUDA_R step4 native LAST restore | PASS, inherited | original successful runtime restore; no manifest rewind |
| CPU input NaN | PASS, inherited | original campaign; not repeated |
| Current CPU_U/CUDA_C1 BEST verification | Assertions PASS, new | original registry read-only |
| Trained BEST reconstruction and C02 normalization | Assertions PASS, new | exact tensor/schema checks and independent oracle |
| Current completed LAST refusal | PASS, new | both BEST source runs rejected LAST resume as expected |
| CPU target Inf | FAIL | acceptance sees an earlier containment flag |
| CPU gradient NaN | NOT RUN | stop after FAIL |
| CUDA input/target/gradient | NOT RUN | stop after FAIL |
| Full CPU recertification | INCOMPLETE | finite gate not closed |
| Full CUDA recertification | INCOMPLETE | finite gates not closed |
| Overall narrowed C02 recertification | FAIL / INCOMPLETE | no automatic retry |

Serialization worker returned exit0/status PASS and all explicit checks completed, but its report also contains an unexplained filesystem-containment flag. Retain that qualification; do not describe the whole worker environment as clean. This does not establish a failed tensor comparison, but requires review before treating the runner as fully validated.

## Preregistration, lineage and execution

[preregistration.json](../experiments/stage_c_training_recertification/20260915T152140Z-closure/preregistration.json), SHA256 `6068d339100fe39b01c3306242ca7ce9b67dad16bfd407aadaa773e9afe1f76e`, records source hashes, full original campaign inventory (including registry), relative paths/types/sizes/SHA256, exact read allowlist, config, bounds and inherited evidence. New raw directory: `ml/models/stage_c_training_recertification/20260915T152140Z-closure/`. No relocation/copy of registry, no symlinks, no rewritten ownership paths or historical manifests.

Single command, authorized host GPU context:

```bash
python3 ml/tests/stage_c_recertification/closure.py
```

GPU preflight ran inside the first of six planned workers: nvidia-smi exit0, RTX3070/615.71.09/8192MiB, CUDA available=true/count1. Recorded PyTorch2.11.0+cu130, Lightning2.6.1, PF1.7.0, NumPy2.4.4, pandas2.3.3, Python3.14.7, CUDA13.0/cuDNN91900. CPU strict and CUDA seeded FP32 IEEE profiles retained, AMP off, cuDNN deterministic, benchmark off. No driver/package change.

The existing original registry index, identity/path and checkpoint SHA checks preceded workers. Complete original inventory matched after EACH executed worker and at final exit. Original files remained byte-identical; no additions/removals. The previous report is preserved byte-for-byte at [report_before.md](../experiments/stage_c_training_recertification/20260915T152140Z-closure/report_before.md) and unchanged below. Earlier attempt reports/results remain history.

## New read-only BEST evidence

Only `Registry(original_owned_root).verified(run_id,'best',expected)` and `manifest` were used against existing owned state. Expected contract was reconstructed from the saved R0 dataset and current code, including original CPU/CUDA device contract, and compared with saved campaign contract. Full-model reconstruction was CPU deserialization, not a device-contract rewrite.

| Source | Run ID | Selected BEST | SHA256 |
|---|---|---:|---|
| CPU_U |3106ee7c00854db8a21c01b0389213b6|step8|73f0f8b70aee27a67bc3fda13d8893407da57b685d32e20c5c75e8e6f60c7079|
| CUDA_C1 |0b4cba625cd94552a44026bfaf87438d|step8|f17b9735ca1d57f26ce5890e421481ecefab94bd0f58220955e837e27be8ce72|

Both independent BEST oracles used all four saved finite epoch values and earliest-step tie-break. Manifest/generation/record agreed. Every loaded state tensor equalled payload and independently saved final trajectory model state. Ordered Q/features, categorical encoders, complete dataset parameters, GroupNormalizer fitted semantic state/mapping/revision matched saved fixture. Reconstructed validation windows had exact keys; encoder_cont and target_scale matched R0 tensors. The glucose encoder component and target_scale additionally matched independent FP64 observed-train log means/std (ddof1 plus PF float16 epsilon), with frozen FP32 atol1e-5/rtol1e-5. Total reconstruction candidates288.

Both completed-run LAST requests returned the exact expected refusal. No attempt to expose historical step4 as current LAST. No model forward, fit, backward or optimizer was needed for these read-only checks. Full evidence in serialization.json. No performance evaluation or calibration.

## First worker FAIL and instrumentation limitation

CPU target probe used a NEW empty probe registry initialized by production APIs; it did not import runs. Injection evidence: target_inf=true, input_target_inf=true. The recorded stack reaches production FiniteModel.forward → require_finite and raises **FloatingPointError: Nonfinite state: model input** before backward/base AdamW, as expected for aliased decoder target.

The harness then fails `assert not report.get('containment_first')`. The flag already contains **filesystem mutation outside closure**. The audit hook sets this flag for every blocked filesystem operation, not only a first rejection of injected nonfinite data. The operation's path/event/stack was not recorded, so this run cannot establish exactly which attempted operation set it or its ordering relative to injection. The serialization worker also carries this flag despite completing its assertions. A blocked operation appears to have been handled by library code; the runner did not fail immediately on that event. This is a limitation of this harness's strict-first-error enforcement, not proof of a new production finite defect.

Do not relabel CPU_N_target PASS: the registered acceptance assertion failed, and causal evidence for the containment flag is incomplete. The production stack does show safe input rejection; no evidence of an unsafe optimizer update. Read-only post-failure manifest inspection confirms no valid BEST/LAST in the probe registry. No diagnostic runtime rerun or repair was performed.

Harness containment uses Python audit hooks and a function-call profile observer, without replacing production/Registry methods. Base optimizer implementation entry is forbidden; forward calls are counted at FiniteModel.forward. This is Python-level containment, not OS isolation of arbitrary native I/O. The original registry constructor's mkdir(exist_ok=True) on its already existing root was permitted; no index was created. Unexplained filesystem flags are retained rather than hidden.

## Budgets and actual exits

| Counter | Actual | Bound |
|---|---:|---:|
| Base AdamW attempts/completions |0/0|0/0|
| Model forwards |2|32|
| Backwards |0|2, gradient probes only|
| Candidates |288|300|
| Workers started |2|6|
| Total seconds |19.186|300|
| Longest worker seconds |12.631|90|

Serialization exit0 (12.631s); CPU_N_target exit1 (6.472s); parent exit1. Four remaining workers NOT RUN. No positive replay workers. Aggregate historical optimizer usage remains48/48. No repair-and-retry, tolerance adjustment or budget extension.

## Files and STOP

New test-only `ml/tests/stage_c_recertification/closure.py`; new attempt metadata/logs; this report update. Production/config hashes unchanged. No baseline config recertification flag change, weights-only certification, registry mutation, real-data evaluation, Optuna or full Baseline training. Historical reports/artifacts preserved. Static syntax/diff checks in final_verification.json. Git branch research/baseline-audit, HEAD2fed05028aaa303ab06362cee96d02c3221fece8; dirty/untracked outputs, nothing committed or pushed.

**STOP.** This failed closure does not authorize a further attempt or stage.

---

Historical report text follows unchanged:

# Update 2026-09-15 — zero-update supplement

**Final overall recertification verdict: FAIL / INCOMPLETE.** The explicitly authorized supplement stopped at its first worker error, before target injection. No retry or remediation was performed after the error. The completed CPU/CUDA replay evidence remains valid, but missing finite/serialization gates prevent overall PASS.

## Final status by scope

| Scope | Final status |
|---|---|
| Original sandbox attempt `20260915T140937Z` | Historical preflight ERROR,0 updates |
| Original host campaign `20260915T143739Z` | Historical FAIL/INCOMPLETE from overspecified target assertion,48 updates |
| CPU strict replay and epoch-boundary restore | PASS, inherited unchanged from host campaign; not repeated |
| CUDA C1/C2/C3 and seeded resume comparisons | PASS, inherited unchanged from host campaign; not repeated |
| CPU input NaN probe | PASS, inherited unchanged |
| CPU target Inf supplemental gate | Worker FAIL at registry setup; injection/acceptance NOT REACHED |
| CPU gradient NaN | NOT RUN |
| CUDA input NaN, target Inf, gradient NaN | NOT RUN |
| BEST/LAST verification and trained BEST serialization | NOT RUN |
| Weights-only export/load | NOT RUN |
| Supplemental completion | FAIL / INCOMPLETE |

No full CPU/CUDA recertification PASS is claimed. Original positive trajectory results have maximum observed comparison differences0, as documented below, without creating a bitwise CUDA guarantee.

## Root-cause review of the original target failure

Classification **1 — HARNESS OVERSPECIFIED**. PF collate shares target between y[0] and x['decoder_target']. In-place Inf injection therefore contaminated the input dictionary. Production FiniteModel.forward called require_finite(x, 'model input'), raising FloatingPointError: Nonfinite state: model input before loss/backward/base AdamW. The harness then incorrectly required the substring target. The binding plan requires a concrete production fail-fast, zero base AdamW calls and no valid BEST/LAST, not a specific later guard. The safe production behavior required no production changes. Original FAIL evidence remains unchanged.

## Test-only correction

The target probe now accepts only exact production messages `Nonfinite state: target`, `Nonfinite state: valid target` or `Nonfinite state: model input`. It records target Inf injection, input decoder-target Inf and storage identity; requires the traceback's originating frame to be production finite_training.py/require_finite, and, for earlier input rejection, requires the production forward frame and contaminated input target. Containment must not reject first. Zero base AdamW attempts and absence of valid BEST/LAST remain required. The base AdamW guard records attempted entry and refuses delegation; non-gradient backward is prohibited in the supplement.

This correction passed static syntax checking but **was not validated at runtime**: the new failure preceded injection. No arbitrary-exception acceptance or production/configuration modification was introduced.

## Supplemental preregistration and lineage

New immutable attempt: **20260915T145551Z-supplement**, revision nmd-c02-zero-update-supplement-1. Executed once outside restricted sandbox:

```bash
python3 ml/tests/stage_c_recertification/supplement.py
```

Preregisration SHA256 `d7625fd70b51899cf1c71d07dd926de5e8a58e8340f3943878fa85ecb5beee1d`. [preregistration.json](../experiments/stage_c_training_recertification/20260915T145551Z-supplement/preregistration.json) records exact original artifact inventory hashes, source hashes, config/tolerances, lineage, allowed exceptions, budgets and order before workers. [copies.json](../experiments/stage_c_training_recertification/20260915T145551Z-supplement/copies.json) records byte-verified selected checkpoints and saved per-worker manifest snapshots. Selected synthetic artifacts came only from the host campaign; no real data or unrelated historical models were accessed.

The source host campaign's complete raw artifact inventory was rehashed at exit and is **unchanged**. Previous report and harness were archived before editing under `supplement_preparation/report_before.md` and `campaign_before.py.txt`; report archive SHA256 `60a5955347e1bac958d2fa903d89b3f067f885f4b70b1fea6fc878e718ff4798`. Original campaign metadata/results were not changed. Supplemental raw evidence remains ignored under matching ml/models directory.

## Actual first failure — incomplete registry copy

Host preflight PASS: nvidia-smi exit0, /usr/bin/nvidia-smi, RTX3070/615.71.09/8192MiB; torch2.11.0+cu130, CUDA13.0 available=true, device_count=1. Expected numerical flags verified: CUDA seeded warn-only, deterministic algorithms enabled, FP32 IEEE/highest, AMP off, cuDNN deterministic=true/benchmark=false. CPU worker stack/profile matched the existing contract.

Then CPU_N_target raised:

```text
ValueError: Refusing to import an existing directory as a trusted run registry
```

Stack: campaign.worker → Registry(args.registry_dir) → checkpoint_registry.py:162. The supplemental preparer copied selected run directories/manifests/checkpoint bytes into `owned`, but omitted the registry-level **owned_runs.json** index. Registry.__init__ correctly refuses a nonempty directory without that index. This is a **supplement harness preparation error**, not a target finite rejection, not a driver failure and not a production defect. The copied boundary registry has the same incomplete preparation; it was never opened by serialization.

No model build, target injection, model forward, backward or base AdamW execution was reached. Thus this failure supplies no new finite-safety PASS and cannot validate the test assertion change. Copied checkpoints exist as inputs; no new valid checkpoint was produced by a negative probe. No exports were performed. The omission was not repaired after the result, respecting STOP-on-first-failure.

## Actual budgets and timing

| Counter | Supplement actual | Preregistered maximum |
|---|---:|---:|
| Base AdamW attempts |0|0|
| Base AdamW completions |0|0|
| Model forward |0|32|
| Backward |0|2, gradient probes only|
| Generated candidate windows |0|300|
| Positive fit workers |0|0|
| Negative workers started |1, failed before fit|5|
| Serialization workers started |0|1|
| Execution seconds, including preflight |6.139|300|
| Per-worker seconds |max4.270|90|

Preflight exit0 in1.816s; CPU_N_target exit1 in4.270s; parent exit1. Exactly one attempt, no retry. Aggregate optimizer use across sandbox attempt + host campaign + supplement remains **48 attempts/48 completions**, with no additional optimizer updates. The supplemental limit is separate and remains zero.

## Files and remaining scope

- `ml/tests/stage_c_recertification/campaign.py`: test-only finite-probe assertion/evidence and supplemental zero-update guards.
- `ml/tests/stage_c_recertification/supplement.py`: new bounded supplemental runner, explicit allowlist excluding positive runs; incomplete copy setup retained as failed evidence.
- `experiments/stage_c_training_recertification/supplement_preparation/`: archived report/harness.
- `experiments/stage_c_training_recertification/20260915T145551Z-supplement/`: new preregistration, copies, preflight, failure, counters and verification.
- `docs/STAGE_C_TRAINING_RECERTIFICATION_REPORT.md`: this update, with prior report preserved below and separately byte-for-byte.

Production code, baseline configs, clinical loss, tolerances, packages/drivers and original evidence unchanged. No positive replay rerun, real-data evaluation, Optuna or full Baseline training. No commit/push. Branch research/baseline-audit, HEAD2fed05028aaa303ab06362cee96d02c3221fece8, dirty/untracked test/report/evidence. Static syntax, unchanged-source hashes and git diff --check verified separately in final_verification.json.

Remaining gates are still the five finite probes (corrected CPU target plus four pending probes) and serialization. Existing replay evidence may be inherited in a separately authorized future scope; this report does not authorize a retry. SWA OFF; calibration deferred; no mid-epoch, cross-device/cross-stack, full-model or clinical-performance guarantees.

**STOP.** No repair or further run after the first error.

---

The following is the unchanged report text for the original host campaign, retained as historical evidence. Its statuses describe that attempt.

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
