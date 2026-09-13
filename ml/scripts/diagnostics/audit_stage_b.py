"""Bounded synthetic Stage B audit, never the production training main.

Run: .venv/bin/python -m ml.scripts.diagnostics.audit_stage_b
Artifacts are unique and ignored under ml/models/stage_b_audit. Known production
bugs remain unittest failures. Child processes exercise real Lightning resume.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
import unittest
from contextlib import ExitStack
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/baseline_stage_b_audit.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def config():
    return json.loads(CONFIG.read_text())


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)
    status = git("status", "--porcelain").decode()
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    sources = {p: sha(ROOT / p) for p in untracked if p and (ROOT / p).is_file()}
    delta = git("diff", "HEAD", "--binary") + json.dumps(sources, sort_keys=True).encode()
    return dict(git_commit=git("rev-parse", "HEAD").decode().strip(),
                git_dirty=bool(status), git_status=status,
                diff_sha256=hashlib.sha256(delta).hexdigest(), untracked_sha256=sources,
                config_sha256=sha(CONFIG), configuration=config(), python=platform.python_version(),
                versions={p: importlib.metadata.version(p) for p in
                          ["torch", "lightning", "pytorch-forecasting", "numpy", "pandas"]},
                device="cpu", precision="32-true", seed=config()["seed"],
                full_training=False, optuna=False, test_performance=False,
                canonical_stage_a_sha256=config()["canonical_parquet_sha256"],
                canonical_data_usage="hash only; never loaded as training/evaluation data")


def forbidden(*args, **kwargs):
    raise RuntimeError("Stage B isolation guard: forbidden production/test/Optuna entry")


def guards():
    from ml.scripts import train_tft_population_v2 as p
    stack = ExitStack()
    for symbol in ("main", "load_test_data", "evaluate"):
        stack.enter_context(patch.object(p, symbol, forbidden))
    module = ModuleType("optuna")
    def guarded_attribute(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return forbidden
    module.__getattr__ = guarded_attribute
    stack.enter_context(patch.dict(sys.modules, {"optuna": module}))
    return stack


def fixture(directory):
    """Two invented patients, production preprocessing/split/dataset; no test rows."""
    import pandas as pd
    from ml.tests.test_baseline_stage_a import write_xml, pre, pipeline as p
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for subject, shift in [("synthetic_A", 0), ("synthetic_B", 80)]:
        path = directory / f"{subject}.xml"
        write_xml(path, n=480, shift=shift)
        frames.append(pre.process_patient(path, subject, source_split="train"))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(directory / "training.parquet", index=False)
    tr, va = p.load_and_preprocess_data(directory)
    training, validation = p.build_datasets(tr, va, SimpleNamespace(context=48, horizon=12))
    digest = hashlib.sha256(b"".join((directory / f"{s}.xml").read_bytes()
                                   for s in ["synthetic_A", "synthetic_B"])).hexdigest()
    return training, validation, dict(fixture_sha256=digest,
        actual_data="deterministic synthetic XML, two invented subjects; no real records",
        schema=dict(reals=training.reals, categoricals=training.flat_categoricals,
                    groups=training.group_ids, target=training.target),
        normalizer=repr(training.target_normalizer), quantiles=p.QUANTILES,
        context=48, horizon=12)


def arguments(dropout=None, swa=False):
    c = config()
    return SimpleNamespace(**{k: c[k] for k in ("context", "horizon", "hidden_size",
        "hidden_continuous_size", "attention_heads", "lstm_layers", "lr", "gradient_clip",
        "batch_size", "epochs", "num_workers")}, dropout=c["dropout"] if dropout is None else dropout,
        no_gpu=True, no_resume=False, no_swa=not swa)


def tensor_digest(state):
    import torch
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if isinstance(value, torch.Tensor):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_digest():
    import torch
    import numpy as np
    return {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
            "torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}


def worker(directory, seed=42, parent=None, stop=False, stochastic=False, swa=False, accumulation=1, nonfinite=None, trusted_resume=False):
    import logging
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import CSVLogger
    from ml.scripts import train_tft_population_v2 as p
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1)
    pl.seed_everything(seed, workers=True)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    manifest = provenance()
    manifest.update(run_id=out.name, seed=seed, parent_checkpoint=parent,
                    parent_run=Path(parent).parent.name if parent else None,
                    parent_checkpoint_sha256=sha(parent) if parent else None,
                    resume_mode="epoch-boundary full fit ckpt_path" if parent else "fresh",
                    synthetic_optimizer_steps=0, command=sys.argv, stochastic=stochastic, swa=swa,
                    accumulation=accumulation, nonfinite_injection=nonfinite,
                    trusted_resume_diagnostic=trusted_resume)
    trace = []
    trainer = None
    class Trace(pl.Callback):
        def on_train_start(self, trainer, module):
            manifest["start_rng"] = rng_digest()
            manifest["estimated_stepping_batches"] = trainer.estimated_stepping_batches
        def on_train_epoch_start(self, trainer, module):
            cfg = trainer.lr_scheduler_configs[0]
            trace.append(dict(event="epoch", epoch=trainer.current_epoch, step=trainer.global_step,
                              scheduler=type(cfg.scheduler).__name__, interval=cfg.interval))
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            import numpy as np
            x, y = batch
            if nonfinite == "input":
                x["encoder_cont"][0, 0, 0] = float("nan")
            elif nonfinite == "target":
                y[0][0, 0] = float("inf")
            trace.append(dict(event="batch", epoch=trainer.current_epoch, step=trainer.global_step,
                time_idx=x["decoder_time_idx"].tolist(), groups=x["groups"].tolist(), rng=rng_digest(),
                python_probe=random.random(), numpy_probe=float(np.random.random())))
        def on_after_backward(self, trainer, module):
            trace.append(dict(event="backward", step=trainer.global_step,
                norm=float(torch.stack([p.grad.norm() for p in module.parameters() if p.grad is not None]).norm())))
        def on_before_optimizer_step(self, trainer, module, optimizer):
            # Negative probes stop here if production has allowed invalid gradients
            # to reach the optimizer boundary. This is an AUDIT guard, not a fix.
            if any(not torch.isfinite(v.grad).all() for v in module.parameters() if v.grad is not None):
                manifest["audit_guard_intercepted_invalid_gradient"] = True
                raise RuntimeError("AUDIT containment: production reached optimizer with nonfinite gradient")
            # This hook is before Lightning clipping; actual optimizer hook below is after.
            trace.append(dict(event="before_clip", step=trainer.global_step,
                norm=float(torch.stack([p.grad.norm() for p in module.parameters() if p.grad is not None]).norm())))
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            trace.append(dict(event="loss", step=trainer.global_step, loss=float(outputs["loss"])))
        def on_train_epoch_end(self, trainer, module):
            if stop and trainer.global_step >= config()["resume_boundary_steps"]:
                trainer.should_stop = True
    real_trainer = pl.Trainer
    def bounded_trainer(**kwargs):
        kwargs.update(max_steps=8, max_epochs=4, enable_progress_bar=False,
                      enable_model_summary=False, logger=CSVLogger(out, name="csv"),
                      accumulate_grad_batches=accumulation)
        kwargs["callbacks"].append(Trace())
        instance = real_trainer(**kwargs)
        if trusted_resume:
            # Bypass ONLY the independently reproduced deserialization blocker,
            # for an exact checkpoint digest from this same synthetic audit run.
            parent_path = Path(parent).resolve()
            parent_manifest = json.loads((parent_path.parent / "manifest.json").read_text())
            if parent_path.parent.parent != out.parent.resolve() or parent_manifest["checkpoints"].get(parent_path.name) != sha(parent_path):
                raise RuntimeError("Trusted diagnostic resume requires a same-run recorded synthetic checkpoint")
            original_fit = instance.fit
            def diagnostic_fit(*a, **kw):
                return original_fit(*a, **kw, weights_only=False)
            instance.fit = diagnostic_fit
        return instance
    try:
        with guards():
            training, validation, meta = fixture(out / "fixture")
            manifest.update(meta)
            # Only a tiny deterministic subset of eligible SYNTHETIC windows.
            # Four batches for accumulation=2, two otherwise; 2 updates per epoch.
            training = training.filter(lambda x: x.index < 4 * accumulation)
            validation = validation.filter(lambda x: x.index < 4)
            real_loader = training.to_dataloader
            def loader(**kwargs):
                kwargs["shuffle"] = stochastic
                return real_loader(**kwargs)
            args = arguments(dropout=0.1 if stochastic else 0.0, swa=swa)
            manifest["effective_model_arguments"] = vars(args)
            model = p.build_model(training, args, parent)
            if nonfinite == "gradient":
                next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
            model.hparams.log_interval = -1
            model.hparams.log_val_interval = -1
            manifest["initial_weights_sha256"] = tensor_digest(model.state_dict())
            # Hook actual AdamW calls: counts updates even if a later callback raises.
            original_step = torch.optim.AdamW.step
            def step(optimizer, *a, **kw):
                result = original_step(optimizer, *a, **kw)
                manifest["synthetic_optimizer_steps"] += 1
                trace.append(dict(event="optimizer", lr=optimizer.param_groups[0]["lr"],
                    norm=float(torch.stack([v.grad.norm() for g in optimizer.param_groups
                                            for v in g["params"] if v.grad is not None]).norm())))
                return result
            with patch.object(p, "MODEL_DIR", out), patch.object(p.pl, "Trainer", bounded_trainer), \
                 patch.object(training, "to_dataloader", loader), patch.object(torch.optim.AdamW, "step", step):
                trainer = p.train(model, training, validation, args, parent)
            model.eval()
            x, _ = next(iter(validation.to_dataloader(train=False, batch_size=2, num_workers=0)))
            with torch.no_grad():
                prediction = model(x)["prediction"]
            if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
                raise RuntimeError("AUDIT containment: refusing terminal artifact with nonfinite weights")
            checkpoint = out / "terminal.ckpt"
            trainer.save_checkpoint(checkpoint)
            torch.save(dict(weights=model.state_dict(), prediction=prediction,
                            optimizer=trainer.optimizers[0].state_dict(),
                            scheduler=trainer.lr_scheduler_configs[0].scheduler.state_dict() if trainer.lr_scheduler_configs else None), out / "state.pt")
            manifest["final_weights_finite"] = all(torch.isfinite(v).all().item() for v in model.state_dict().values())
            manifest.update(exit_code=0, global_step=trainer.global_step, epoch=trainer.current_epoch,
                final_weights_sha256=tensor_digest(model.state_dict()), final_rng=rng_digest(),
                best_checkpoint=trainer.checkpoint_callback.best_model_path,
                last_checkpoint=trainer.checkpoint_callback.last_model_path,
                terminal_checkpoint=str(checkpoint),
                callbacks={c.state_key: c.state_dict() for c in trainer.callbacks if c.state_dict()},
                deterministic=trainer._accelerator_connector._deterministic_mode
                    if hasattr(trainer._accelerator_connector, "_deterministic_mode") else "production False",
                cudnn_benchmark=torch.backends.cudnn.benchmark)
    except Exception as exc:
        # Audit evidence, not error recovery: worker returns nonzero, suite asserts it.
        manifest.update(exit_code=1, error=repr(exc), traceback=traceback.format_exc())
    finally:
        manifest["checkpoints"] = {str(p.relative_to(out)): sha(p) for p in out.glob("*.ckpt")}
        # Callback states can be tensors/models; checkpoint is authoritative, JSON overview only.
        manifest["trace"] = trace
        dump(out / "manifest.json", manifest)
    return manifest["exit_code"]


def run_suite(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["NMD_STAGE_B_RUN"] = str(root)
    import torch
    torch.set_num_threads(1)
    manifest = provenance()
    manifest.update(run_id=root.name, command=sys.argv)
    canonical = ROOT / config()["canonical_stage_a"]
    data = list(canonical.glob("*.parquet"))
    if len(data) != 1 or sha(data[0]) != config()["canonical_parquet_sha256"]:
        raise RuntimeError("Canonical Stage A parquet hash mismatch")
    canonical_meta = json.loads((canonical / "provenance.json").read_text())
    if canonical_meta["git_commit"] != "5ee8d930e80fd7d62d0248c52ed10aaaaaa7f8d0" or canonical_meta["git_dirty"]:
        raise RuntimeError("Canonical Stage A provenance mismatch")
    manifest["canonical_manifest_sha256"] = sha(canonical / "provenance.json")
    records = []
    class Result(unittest.TextTestResult):
        def addSuccess(self, test):
            super().addSuccess(test); records.append(dict(test=test.id(), status="PASS"))
        def addFailure(self, test, err):
            super().addFailure(test, err); records.append(dict(test=test.id(), status="FAIL", detail=self._exc_info_to_string(err, test)))
        def addError(self, test, err):
            super().addError(test, err); records.append(dict(test=test.id(), status="ERROR", detail=self._exc_info_to_string(err, test)))
        def addSubTest(self, test, subtest, err):
            super().addSubTest(test, subtest, err)
            if err is not None:
                status = "FAIL" if issubclass(err[0], test.failureException) else "ERROR"
                records.append(dict(test=subtest.id(), status=status, detail=self._exc_info_to_string(err, test)))
        def addSkip(self, test, reason):
            super().addSkip(test, reason); records.append(dict(test=test.id(), status="SKIP", detail=reason))
    started = time.monotonic()
    with guards():
        suite = unittest.defaultTestLoader.discover(str(ROOT / "ml/tests"), pattern="test_baseline_stage_b.py")
        result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
    children = [json.loads(p.read_text()) for p in root.glob("*/manifest.json")]
    counts = json.loads((root / "unit_counts.json").read_text()) if (root / "unit_counts.json").exists() else {}
    method_status = {}
    for record in records:
        method = record["test"].split(" (")[0]
        status = record["status"]
        if method not in method_status or status in ("FAIL", "ERROR"):
            method_status[method] = status
    manifest.update(elapsed_seconds=time.monotonic()-started, results=records,
        method_counts={s: list(method_status.values()).count(s) for s in ["PASS", "FAIL", "ERROR", "SKIP", "XFAIL"]},
        counts={s: sum(r["status"] == s for r in records) for s in ["PASS", "FAIL", "ERROR", "SKIP", "XFAIL"]},
        synthetic_optimizer_steps=sum(c["synthetic_optimizer_steps"] for c in children)+sum(counts.values()),
        unit_optimizer_counts=counts, fixture=json.loads((root / "fixture.json").read_text()), child_manifest_sha256={str(p.relative_to(root)): sha(p) for p in root.glob("*/manifest.json")},
        exit_code=0 if result.wasSuccessful() else 1,
        limitations=["CUDA, AMP, multiple workers not exercised", "No mid-epoch replay guarantee"])
    dump(root / "manifest.json", manifest)
    return manifest["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parent")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--nonfinite", choices=["input", "target", "gradient"])
    parser.add_argument("--trusted-resume", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.seed, args.parent, args.stop, args.stochastic, args.swa, args.accumulation, args.nonfinite, args.trusted_resume)
    if args.suite:
        return run_suite(args.suite)
    directory = ROOT / "ml/models/stage_b_audit" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               MPLCONFIGDIR=str(directory / "mpl"))
    command = [sys.executable, "-m", "ml.scripts.diagnostics.audit_stage_b", "--suite", str(directory)]
    started = time.monotonic()
    try:
        with (directory / "console.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=config()["suite_timeout_seconds"])
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
    dump(directory / "invocation.json", dict(command=command, environment={k: env[k] for k in
        ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "MPLCONFIGDIR"]},
        exit_code=exit_code, elapsed_seconds=time.monotonic()-started))
    print(directory)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
