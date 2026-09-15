# Baseline v1.0 — corrected training launch plan

2026-09-15 · nmd-baseline-launch-plan-1 · **PLAN ONLY / NOT EXECUTED**

## Decision and evidence

Proceed next only to the separately activated [launch preflight task](CODEX_BASELINE_V1_LAUNCH_PREFLIGHT_TASK.md): static launch checks and a synthetic full-model capacity smoke. Full real-data training requires a subsequent explicit authorization and a frozen launch manifest. Capacity PASS alone is not launch authorization.

Reviewed source branch research/baseline-audit, HEAD **4f0cf189b54cd2cfc2ad288da1b6f8c85461b66d**, clean at entry; no fetch. The first update in STAGE_C_TRAINING_RECERTIFICATION_REPORT.md establishes the accepted narrowed PASS, superseding the earlier closure review's pending verdict and preserved failed attempts. CPU strict replay, CUDA seeded replay, complete boundary restore, trained BEST reconstruction/encoded-subject normalization, finite gates and ownership passed. Original evidence was preserved. Do not repeat those campaigns for this launch. Weights-only remains **NOT RECERTIFIED AFTER C02 / DEFERRED**.

Stage A temporal/leakage evidence is inherited for unchanged upstream components; Stage B mechanics are historical evidence complemented by fresh C02 recertification. None establishes full architecture capacity, real-data predictive quality, calibration or clinical reliability. This planning session performed source/document reads only: no data/model loading, GPU probe, test execution or optimization.

## Fresh-run contract and freeze

| Field | Required value |
|---|---|
| Protocol | nmd-baseline-v1.0-stage-c-1 |
| Data protocol | baseline_v1_stage_a |
| Normalizer | observed-train-encoded-subject-v2 |
| Evaluation | nmd-evaluation-stage-c-1 |
| Numerics | nmd-numerics-2 / cuda-seeded |
| Initialization | fresh, no checkpoint, no parent; synthetic_audit=false for real run |
| Quantiles | [.02,.10,.25,.50,.75,.90,.98], ordered exactly |
| Canonical Stage A SHA256 | d14fe31b1b81971639ca8d5ba17b4882fbd6711569e30785f0da7b91a0c031cf |

Old-protocol/normalizer checkpoints are incompatible: no resume, conversion, manual revision tagging or weights-only transfer. New synthetic smoke weights are also forbidden as initialization. Resume is only continuation of the same newly created real run from its own compatible nonterminal LAST.

Freeze before launch: exact clean source commit plus the byte hashes below; config bytes and resolved CLI/effective arguments; launch wrapper/monitor/test source hashes; dependency versions and lockfiles if present; Python, GPU UUID/name/VRAM/driver, CUDA/cuDNN; effective numerical flags and CPU thread count (1); data provenance and actual dataset hash once authorized; train/validation window index hashes/counts, feature order/dtypes, Q, subject encoder mapping and train-fitted normalizer state; full registry contract/digest, owned absolute path and run ID. Store patient-specific state only in protected local artifacts, not public reports. Record environment/package manifest separately from Git. Changed production/config/stack triggers impact review; documentation-only commits can inherit unchanged source hashes.

The config's training_recertification string still says NOT RUN. This is stale descriptive metadata, not evidence negating the later signed-off report. Preserve config bytes in preflight; attach report/hash and explicit narrowed-scope attestation in launch manifest. Any later config wording update changes its hash and must be frozen anew. Never pretend the config itself already records PASS. Registry fingerprints only five training modules; supplementary manifest must also hash evaluator, upstream preprocessing/window logic, config and launch tooling.

## Data-access gate before real launch

Current main hashes the canonical training.parquet, then load_and_preprocess_data reads the entire file before filtering source_split='train'; load_test_data is separate and must never be called. This is not a train-only physical reader. The original C-core contract forbids opening this shared file because it contains source test. Preflight remains synthetic and does not open or hash the parquet; metadata hashes are inherited.

Before real training, explicitly resolve this boundary in the separate launch authorization: either authorize the existing canonical read solely for hashing and deterministic train selection, with test rows never passed to fit/validation/scoring/logs, or commission an isolated train-only loader/artifact with provenance and targeted regression. The latter changes source/data compatibility and must not be implemented silently or supplied with a forged canonical hash. Preferred stricter isolation is a separate reviewed change if physical test-row access remains prohibited. Until the selected policy is explicit, **DATA_ACCESS_GATE=HOLD**; do not issue a ready-to-run real main command.

After this gate is resolved, verify unchanged Stage A split/window construction and observed-only fit, full rolling validation, nonempty train/validation, no unseen validation groups, qualified counts and finite model inputs. Do not recompute splits or select windows by performance. Training's variable decoder lengths remain supported; full evaluation W is a later distinct population.

## Full-model capacity preflight

Use the exact production model: hidden64, hidden_continuous16, attention heads4, LSTM1, dropout.3, context48/horizon12, Q7, production ordered features and categorical schema; batch64 training and **batch128 validation** (production uses batch_size*2). Native GroupNormalizer and encoders through production dataset/model builders, not hand-built random tensors or tiny recertification architecture.

Generate synthetic positive targets, observed flags and all required features with maximum encoder48/decoder12, multiple subject scales, and the production categorical cardinalities/embedding sizes established from existing non-patient schema metadata or saved synthetic schema evidence. Prove equality of model parameter shapes to expected production shapes. If metadata cannot establish production cardinalities without real data access, report capacity CONDITIONAL and launch HOLD; do not assert exact full-model feasibility from a smaller embedding table. Do not inspect patient data to fill that gap in the first task.

Select exactly128 training windows and128 validation windows in advance, full lengths and balanced synthetic groups; 2 complete training batches and 1 complete validation batch. Production orchestration with a declared smoke-only override epochs=1, total_steps=2; other model/optimizer/loss/numerics settings unchanged. This small schedule tests state allocation, not the real 60-epoch LR trajectory. Two AdamW calls include initial LR=0 and a nonzero subsequent LR; assert moments allocated and nonzero parameter update on the second. Do not change warmup to manufacture this result. Production fresh sanity validation is included. Smoke artifacts are SYNTHETIC_CAPACITY_ONLY, never real candidate models.

Use one fresh local CUDA worker on the actual **RTX3070, 8192MiB** (verify UUID/model/capacity, not AGENTS' generic Ti assumption). No competing compute workload; do not terminate other users' processes. Capture PyTorch allocated/reserved and peak allocated/reserved after cuda synchronization/reset_peak_memory_stats; driver total/used/free and process GPU usage sampled at least1Hz; baseline, model construction, each forward/backward/update, full128 validation, checkpoint serialization, final state. Also host RSS, available RAM, elapsed step/validation/save time and artifact sizes. Keep AdamW state alive during validation and save measurement. Do not empty_cache between phases to disguise the true peak.

Acceptance: complete exactly2 optimizer calls, finite loss/gradients/parameters/moments, clip norm≤1+1e-5, correct flags/shapes, complete full128 validation and verified full checkpoint; no OOM/allocator error; sampled device free memory never below **1024MiB**, and peak process reserved memory plus measured baseline non-process usage ≤ total−1024MiB. The1GiB is a preregistered operational headroom requirement, not a clinical or universal hardware rule. Record sampling limitations; short smoke cannot prove absence of long-run leaks. Any OOM or inadequate headroom => CAPACITY FAIL/HOLD; missing memory evidence => INCOMPLETE. If desktop GPU usage makes this infeasible, report it, do not relax the margin after results.

**No automatic batch reduction, accumulation, AMP, shorter context/horizon or architecture reduction.** A proposed32×accumulation2 is not numerically equivalent to64×1 because dropout, loss reduction and update scheduling can change. OOM ends this task; such a variant needs separate approval, versioned effective config, capacity test and targeted mechanics/replay review. No silent retries or CPU fallback.

## Full training execution contract (later task only)

Let N be actual qualified training windows. B=floor(N/64), must be≥1. With accumulation1, planned total updates T=60B; drop_last=true each epoch. Freeze N/B/T after authorized dataset construction and verify Trainer.estimated_stepping_batches agrees. Do not guess a numerical T from row counts. Validation batches=ceil(N_val/128), drop_last=false, shuffle=false, all qualified rolling full-horizon validation windows every epoch; fresh sanity validation up to2 batches, resume0. No limit_train_batches or reduced validation coverage.

Maximum60 complete epochs, or native EarlyStopping(val_loss,min,patience20,min_delta1e-4,check_on_train_epoch_end=false). No arbitrary loss target or best seed search. ClinicalQuantileLoss factor2, hypo threshold70, weight2.5, valid-position reduction unchanged. CheckedAdamW LR3e-4, eps1e-7, weight_decay.01, current library defaults otherwise frozen. Step scheduler SequentialLR: linear LambdaLR warmup max(50,T//20), then CosineAnnealingWarmRestarts T_0=15B, T_mult2, eta_min1e-6, frequency1. Original total budget remains unchanged through resume; never extend epochs or restart the scheduler to improve results.

Global norm clip1.0 via finite FP64 norm accumulation. Seed42, independent fresh initialization via seed_everything; epoch-seeded shuffled sampler, workers0, CPU threads1, single GPU, FP32, SWA OFF. cuda-seeded: deterministic algorithms=true/warn_only=true, cuDNN deterministic=true/benchmark=false, global/matmul/cuDNN ieee, highest matmul precision, CUBLAS_WORKSPACE_CONFIG=:4096:8, AMP/TF32 OFF. CPU strict evidence is not a GPU bitwise promise. Record warnings; only known upsample_linear1d_backward_out_cuda nondeterminism is inherited. No cross-stack/device resume or mid-epoch/partial-accumulation guarantees.

Estimate upper runtime from measured capacity timings and actual N/B/N_val, marking synthetic timing approximate. Before full launch, freeze a wall-time allowance and disk reservation in its separately approved manifest. This plan does not invent a reliable hours estimate without those measurements. Budget T counts the uninterrupted logical run; any replayed work after crash is separately counted, not hidden. No automatic recovery loops or additional compute beyond the later approved allowance.

## Checkpoints, retention and crash recovery

Current BoundaryCheckpoint saves after every complete epoch's training and validation, with optimizer/scheduler/loops/callback/RNG/encoder state. Registry LAST is the latest generation, not Lightning save_last (which is false). BEST is minimum (finite val_loss, global_step), earlier step on exact tie; min_delta affects early stopping, not this BEST comparison. Terminal completion/early stop sets terminated=true; terminal LAST is intentionally refused for continuation.

Current writes: checkpoint temp → successful serialization/file fsync → os.replace → registry verifies payload/finite/hash → atomically replace run.json; JSON temp flush/fsync before replace. There is no single transaction spanning bytes/index/manifest and **no parent-directory fsync**, so do not claim power-loss durability or automatic rollback. Temp/orphan files may remain after abrupt crash. A final generation written before pointer advance can collide on replay because overwrites are refused.

Retention decision for first run: **retain all epoch generations (up to60)** plus BEST/LAST pointers, index, manifests, segment logs/failure events and temporary evidence. This automatically retains at least the most recent two completed boundaries where available. No rotating deletion, pointer rewind, registry cloning/relocation or filename-based recovery. No automatic fallback to an older retained generation: current API verifies current registered LAST, not arbitrary history. Older files aid manual investigation, not certified fallback.

After crash/power loss: mark interruption in external launch journal; process absence with status=running is not success. Read index/manifest at original absolute path, verify current LAST bytes, ownership, contract, nonterminal complete boundary and finite full state. Restore only through production verified LAST/public Trainer.fit and same environment/total budget. Work after that boundary is discarded and epoch recomputed; no promise of mid-epoch continuation. First-epoch crash without LAST => no certified resume; a new fresh run needs explicit approval. Corrupt/missing manifest/LAST, orphan collision, terminal LAST or incompatible environment => HOLD for read-only recovery review, never silently promote an older checkpoint or use BEST/weights-only. Existing valid pointers may survive an exception, but failure() only writes invalid_segment event and does not change run.status. Reconcile these facts in external segment journal.

Reserve disk before full launch: at least 2×60×S + max(5GiB, measured log allowance), where S is measured full smoke checkpoint size; re-evaluate against actual full model metadata size. Retain extra capacity for temp+final concurrent writes and logs. Before each epoch require remaining reserve ≥2×remaining_epochs×S +5GiB. Treat this as conservative provisional sizing, check actual sizes as run proceeds. No checkpoint pruning to keep a failing disk alive. If durable power-loss recovery beyond validated current LAST is required, commission a separate implementation/test; it is not already guaranteed.

## Monitoring and stopping

| Trigger | Required response |
|---|---|
| NaN/Inf valid input/target/prediction/loss/gradient/parameters/moments | Production fail-fast, invalidate segment; no sanitizing/dropping; review before recovery |
| OOM | Stop and preserve logs/last valid boundary; no batch/profile retry |
| Hash/ownership/schema/profile mismatch, corrupt checkpoint | Stop before load; no fallback or sidecar edits |
| Missing/nonfinite val_loss, empty/changing unexpected validation population, incomplete epoch | Fail/HOLD; no publication or success claim |
| Finite validation plateau/deterioration | Native frozen early stopping; no threshold tuning or test consultation |
| Unexpected abrupt finite validation change | Flag for review with counts/LR/data/source integrity; do not invent a quality cutoff or auto-change LR |
| Sustained pre-clip norm>100 after step200 for30 consecutive steps | Existing logger only warns/reset counter; external supervisor flags and pauses after next valid boundary for review, without changing clipping/LR |
| Post-clip norm>1+1e-5 or new numerical warning | Stop/HOLD for instrumentation/numerics review |
| Low disk, failed save/fsync/replace, missing log heartbeat/process crash | Halt segment, external failure journal even if registry.failure cannot write; never mark completed |

The external monitor/journal and boundary pause behavior are launch-tool requirements, not claims about current automatic protections. Prepare/review them before full launch; do not add production callbacks silently. An emergency kill loses current partial epoch, using the above recovery policy. Low disk forecast should stop before the next epoch; native stop_after_epoch can be used only with its existing nonterminal boundary semantics and frozen deployment configuration. Every intervention is recorded. No unattended auto-resume.

## Training DoD and post-training sequence

TRAINING_COMPLETED requires normal terminal full boundary due60 epochs or native early stopping, verified terminal LAST and independent BEST oracle over all saved finite epoch values, verified BEST full reconstruction, complete updates/validation/segment history, correct source/config/data/profile hashes, no unresolved invalid segment affecting the accepted trajectory, and intact artifact hashes. Externally interrupted/paused/failed runs are not completed even if a usable BEST exists. Record terminal reason explicitly in external report; registry completed alone is insufficient after an ambiguous crash.

The registered full checkpoint BEST becomes **candidate BEST**, immutable path/hash/run ID; no weights-only export requirement. First inspect mechanical training/validation evidence: losses and denominators over all epochs, LR/gradient traces, stopping reason, population continuity, finite flags, checkpoint selection/lineage. Good execution does not mean the model performs well. No test metrics, calibration or clinical reliability claims.

Next, separately authorize Stage C real **validation-only** evaluation after freezing candidate BEST and evaluation input roles/W/keys under STAGE_C_EXECUTION_CONTRACT.md and corrected evaluator revision. Use trained model's normalization once, raw Q, shared TFT/persistence keys, all required horizons/persons/ranges/aggregations. Training weighted val_loss is distinct from unweighted reporting. Validation used for stopping is development evidence, not untouched generalization evidence. No automatic calibration fit or quantile rearrangement.

Only a later explicit final-test authorization, after methodology/model/thresholds are frozen and a role-specific test contract is reviewed, may open test for evaluation. The C-core contract by itself does not authorize real test evaluation. Do not repeatedly consult test to revise the model. Calibration and clinical reliability remain NOT EVALUATED; weights-only remains DEFERRED.

## Exact planning snapshot hashes

These are observed byte hashes, not substitutes for launch-time revalidation. The broader scripts inventory intentionally includes unchanged upstream files; source changes need classification, not automatic recertification of unrelated legacy paths.

- `AGENTS.md`: `18694ef1a11969e0e0bb6dbaba36b1b95b07178f45f4366841068c54e177d0e6`
- `configs/baseline_v1_stage_c.json`: `a85bb26214253a855af066b590565b3b8b45beebb34302e686cc36bc752e1f35`
- `docs/STAGE_C_TRAINING_RECERTIFICATION_REPORT.md`: `e32b2216cf46278a749986818d8428545992cb877bc6dd98efe4e350e7969082`
- `docs/STAGE_C_TRAINING_RECERTIFICATION_CLOSURE_REVIEW.md`: `73bd07a6fe9405c3ec924184810f2afee1153acad906c1a0c31bc60cc4a565fb`
- `docs/STAGE_B_FINAL_REVIEW.md`: `edfaf0307fcd72b069c4849924d67b5a23d6881f5ffc7743b3c4e0f1913e4d20`
- `docs/STAGE_B_FINALIZATION_REPORT.md`: `e04b65e07cc455f87c9e5047aba10d71a4c2644851a381002d1df31dd77e53e4`
- `docs/STAGE_C_EXECUTION_CONTRACT.md`: `a7a631d8d1cd36cff1847d44878380ec2ca97680115abe08be9d6d42eb1d2610`
- `ml/scripts/baseline_training.py`: `0cdef5a6311e3804cda69dfbc936cafb5fa4b560e12e53c71250c0b569018d74`
- `ml/scripts/checkpoint_registry.py`: `cb3b9ca3495faa06ee039c28c2085fb3a9ee1c246f9cdeba766d11707d25df3d`
- `ml/scripts/evaluation_stage_c.py`: `971ef31ae41b568f61c9583ae086b63fb09b98843786e6dd1f8a8cde3d725545`
- `ml/scripts/export_onnx.py`: `0d57045cfac19e29c3cfd6babf3299ac53d53c4bfb57130850888a65603fda30`
- `ml/scripts/finite_training.py`: `3e0adea81cd1fdef6fec933213b1c1cb84b5fe8172716c7a69acab78e4abbf6a`
- `ml/scripts/generate_synthetic_patients.py`: `288f30894c01c0a04d659e1268475b05916c34434c0815c06e450959da9d9bd4`
- `ml/scripts/numerical_profile.py`: `5bf981d01877b1900a0c8f0426a5aa27966ec9d6a64ceb9c8639e0306c0e5630`
- `ml/scripts/observed_windows.py`: `f530d31ff6c41f50c28b86e02ad70af5c749b763799377ab11df872d916417a9`
- `ml/scripts/preprocess_ohiot1dm.py`: `9e654c63fe1dc1bb4f189302b91e06281c39aff0a48e6f4410d2f3e5d1b463d3`
- `ml/scripts/temporal_protocol.py`: `e0f0661387dffe1563669284fcfeb009ad7a6bab15712011e6f26ef9637889f0`
- `ml/scripts/train_tft.py`: `2a9263cc7f63f66a6d493e5489a431528bdd6e32f06c2fc3a6af46a48a485e31`
- `ml/scripts/train_tft_population_v2.py`: `dcdb6c738d70958a4a105d81c965f4dd17af0db35143ba77d98ba5b639f24572`
- `ml/scripts/tune_tft.py`: `853b38fcc94c4cc6250e1dac6421541b471c92c28fe61a92845f2ded50fc7e4a`
