# Stage C training recertification — closure review

2026-09-15 · review only · research/baseline-audit.

**ADDITIONAL ZERO-UPDATE TESTING IS REQUIRED AND POSSIBLE WITHOUT CONTRACT CHANGES**

This verdict concerns the production ownership/normalizer/training contract. It is **not recertification PASS**. The original written DoD explicitly includes a fresh weights-only export/load. This review recommends a documented scope amendment deferring that independent capability; it does not silently mark the unexecuted test satisfied. Until that disposition is accepted and the necessary remaining tests pass, the original FAIL/INCOMPLETE remains current. No registry redesign is necessary to verify C02's fresh training, full checkpoint and epoch-boundary resume path.

## Evidence and classification

Categories: **1** satisfied by executed evidence; **2** possible read-only at the original registry; **3** necessary capability blocked by API; **4** overspecified for certifying C02's fresh/full-checkpoint path, proposed deferral or removal of duplicate testing.

| Requirement | Category | Decision and boundary |
|---|---|---|
| CPU strict 8 vs4+4, CUDA C1/C2/C3 and4+4 | 1 | PASS from 20260915T143739Z;48 updates, recorded discrepancies0. Do not repeat. CUDA remains seeded, not a bitwise guarantee. |
| Historical step4 LAST verification and boundary restore | 1 | CPU_R/CUDA_R used Registry.verified on their own then-current LAST and public Trainer.fit. Exact model/optimizer/scheduler/RNG/generators/counters/callback restore before forward, native dispatcher and same instance PASS. Successful continuation/order comparison completes this evidence. |
| Re-exposing step4 as current LAST after R finishes | 4 | Not required for the above guarantee. Current manifests correctly advanced to step8. Rewinding them or importing snapshots is unnecessary and unsupported. |
| Completed-run LAST semantics | 2 | Read current manifest and require terminated=true. Optional exact expected refusal from verified(...,'last',...) confirms that a completed run cannot resume. This is safe refusal, not a missing checkpoint or failed restore. |
| Current trained BEST and independent BEST selection | 2 | Mandatory fresh positive integration still pending. verified(...,'best',expected) is read-only when the existing complete registry is used. Compute oracle from all saved finite epoch validation values, tie-break earliest step; compare record, generation and bytes. No prediction scoring. |
| Normalizer/encoder mapping roundtrip | 1 + 2 | R0 independent observed-log oracle and actual tensors PASS; resume PASS. Still require trained BEST reconstruction, all fitted mapping/stats/schema/Q and actual encoder_cont/target_scale versus saved fixture. C02 changes these semantics, so this must not be waived. |
| Checkpoint ownership/compatibility | 1 + 2 | R0 negative pre-deserialization checks and executed LAST restore PASS. Add positive current BEST verification at its registered absolute path. Never bypass ownership or reuse old-protocol models. |
| Fresh weights-only export/load and new optimization lineage | 4, proposed deferral | Explicit original DoD item, but independent of C02 fresh training/full checkpoint resume. It has not passed after C02. Stage B evidence supports unchanged mechanics only; it does not certify new C02 weights-only semantics. Exclude weights-only initialization from closure claims until separately tested. |
| Relocated registry clone to enable weights export | 4 | An implementation workaround introduced by the supplement, not a necessary C02 safety guarantee. Current API deliberately rejects relocation; no need to weaken it. |

No category3 capability is necessary for the narrowed C02 fresh/full-checkpoint certification. If fresh weights-only certification remains mandatory, its execution is blocked **under the current immutability constraints**, not by a nonexistent exporter: export_weights exists but writes to the owned run. It cannot be run read-only or on a relocated byte-copy. Do not claim the original unamended DoD has passed.

## Actual registry contract

[checkpoint_registry.py](../ml/scripts/checkpoint_registry.py), Registry.__init__/manifest/verified/export_weights:

- Existing owned_runs.json binds each run to an absolute path and contract digest. manifest rejects relocation and symlinks.
- verified reads the index, run.json and checkpoint; checks role/compatibility/hash before deserialization, then full payload semantics and finite state. It does not update pointers or write exports.
- Constructor can create an index for an empty directory. Therefore preflight must require the original directory AND index already exist; never instantiate it on an incomplete copy.
- Completed LAST is rejected before deserialization. Historical step4 restore was verified when step4 really was LAST; current pointer advancement does not invalidate that evidence.
- export_weights loads BEST, writes weights-<uuid>.pt and updates run.json. create(mode='weights-only') writes a new identity/index entry. Neither is a read-only verification API or an import/clone API.

Source evidence: [CPU_R_restore.json](../experiments/stage_c_training_recertification/20260915T143739Z/CPU_R_restore.json), [CUDA_R_restore.json](../experiments/stage_c_training_recertification/20260915T143739Z/CUDA_R_restore.json), [R0.json](../experiments/stage_c_training_recertification/20260915T143739Z/R0.json). Original [plan](STAGE_C_TRAINING_RECERTIFICATION_PLAN.md) and [task](CODEX_STAGE_C_TRAINING_RECERTIFICATION_TASK.md) explicitly require finite gates and trained BEST reconstruction; neither is replaced by this review.

## Proposed zero-update closure procedure — NOT EXECUTED

Prerequisite: explicitly accept the weights-only deferral above in the closure scope, keeping the original specification/report and failed runs as history. No production contract change, no automatic retry authorization follows from this document.

1. Preregister a NEW unique run directory and exact source/config/stack hashes, inherited evidence, original artifact allowlist, expected refusals and hard budgets. Preserve48 historical updates; supplement allows **zero base AdamW calls**, including failed/LR0 calls. No positive fit/replay workers.
2. Record a complete relative-path/type/size/SHA256 inventory of original campaign and registry, reject symlinks and concurrent writers, and enforce read-only access to that source. Output/log/cache paths belong only to the new run. Check inventory after each phase and on failure/final exit; differences, additions or removals mean FAIL.
3. Check original owned_runs.json/version, run IDs, absolute paths, contract digests, existing manifest/checkpoint files and their hashes before imports/deserialization. Recheck production hashes and effective CPU/CUDA profiles; CUDA preflight must pass on the expected host GPU before any finite injection.
4. In a fresh verification process, instantiate Registry at the **original** path:
   `ml/models/stage_c_training_recertification/20260915T143739Z/owned`.
   Call `verified(run_id, 'best', expected)` for CPU_U `3106ee7c00854db8a21c01b0389213b6` and CUDA_C1 `0b4cba625cd94552a44026bfaf87438d`. Validate expected against frozen fixture/current code, not merely against an unchecked manifest. Inherit its numerical device contract even when deserializing on CPU. Check oracle min((value,step)) from the complete saved validation sequence and equality with the corresponding generation/record. Verify lineage and source hashes.
5. Load those verified full checkpoints without fit; compare every state tensor to payload and selected original evidence, Q, ordered features, categorical encoder mapping, normalizer fitted stats/revision and dataset parameters. Reconstruct from saved dataset parameters and the explicitly allowlisted R0 synthetic fixture; compare actual encoder_cont/target_scale and exact window keys to the saved fixture and the independent observed-log oracle. Freeze FP32 atol1e-5/rtol1e-5; identities exact. No model prediction/evaluator metrics or calibration.
6. Retain CPU_R/CUDA_R step4 restore evidence; do not rewrite current LAST. Read current terminated manifests, optionally assert exact completed-LAST refusal. Do not call export_weights, publish, create or failure on the original registry.
7. Finite safety still needs the corrected CPU target and CPU gradient probes plus CUDA input/target/gradient probes. These use a **separate initially nonexistent/empty new probe registry**, initialized normally by Registry and Registry.create through production orchestration. Never preload copied run directories there. Reuse verified synthetic fixture, not trained checkpoint initialization. Only the gradient probes may backward; no base AdamW delegation. Exact production-origin refusal, demonstrated injection, containment not first, zero attempts/completions and no valid BEST/LAST required. CPU input PASS remains inherited. Negative probes may enter Trainer.fit but must fail before any update; this is not a repeat of positive training.
8. Suggested preregistered limits:6 workers (five negative, one serialization),90s each,300s total;32 forward calls,2 backwards,300 candidates,0 base AdamW calls. Count all actual calls, including failures. GPU preflight/metadata work included in wall budget. Stop at first FAIL/ERROR/timeout; no repairs or retries within that authorization.
9. Report inherited versus new evidence and exact hashes. Overall closure only after necessary gates pass and the weights-only exclusion is explicit. Preserve earlier FAIL/INCOMPLETE results. Weights-only remains **NOT RECERTIFIED AFTER C02 / DEFERRED**, not a synthetic PASS.

Original copied-registry failures do not invalidate successful CPU/CUDA trajectories. Conversely, safe original target rejection alone does not supply the unexecuted gradient/CUDA finite gates. Those protections and positive trained BEST normalization roundtrip remain necessary.

## This review's actions and STOP

Read source and existing JSON evidence only; no checkpoint deserialization, fixture execution, Registry instantiation, GPU probe, training, backward or optimizer call. Created only this decision document. Current report, production/configs, registries and historical artifacts unchanged. No symlinks, ownership rewrite, monkeypatch, commit or push. **STOP before implementation and test execution.**
