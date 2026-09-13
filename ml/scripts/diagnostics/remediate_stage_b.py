"""Bounded synthetic remediation acceptance; never production main/evaluation."""
import argparse
import copy
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import unittest
from unittest.mock import patch

import numpy as np
import torch
import lightning.pytorch as pl
from ml.scripts import train_tft_population_v2 as p
from ml.scripts.diagnostics import audit_stage_b as old
from ml.scripts.checkpoint_registry import Registry,contract,sha256,atomic_json
from ml.scripts.diagnostics.callback_restore import observe_restore, snapshot as callback_snapshot


def settings(metadata, directory, seed=42, stochastic=True, accumulation=1, device="cpu"):
    args=old.arguments(dropout=.1 if stochastic else 0.,swa=False)
    args.seed=seed;args.shuffle=stochastic;args.accumulate_grad_batches=accumulation
    args.dataset_sha256=metadata["parquet_sha256"]
    args.synthetic_audit=True;args.registry_dir=Path(directory);args.mode="fresh"
    args.no_gpu=device=="cpu";args.max_steps=8;args.stop_after_epoch=None
    args.run_id=None;args.parent_run_id=None
    return args


def small_fixture(path,accumulation=1):
    training,validation,meta=old.fixture(path)
    meta["parquet_sha256"]=sha256(Path(path)/"training.parquet")
    return training.filter(lambda x:x.index<4*accumulation),validation.filter(lambda x:x.index<4),meta


def worker(out,registry,parent=None,seed=42,stochastic=True,stop=False,accumulation=1,injection=None,mode="fresh",device="cpu"):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);pl.seed_everything(seed,workers=True)
    logging.getLogger().setLevel(logging.ERROR)
    report=old.provenance();report.update(run_id=out.name,command=sys.argv,
        synthetic_optimizer_steps=0,device=device,precision="32-true",parent_run=parent,mode=mode,seed=seed,
        source_sha256={q.name:sha256(q) for q in [Path(p.__file__),Path(__file__)]})
    trace=[];trainer=None
    class Trace(pl.Callback):
        def on_train_start(self,t,m):
            if parent_payload is not None:
                report["callback_restore"]=callback_snapshot(t,parent_payload,restore_calls)
        def on_train_batch_start(self,t,m,b,i):
            x,y=b
            if injection=="input":x["encoder_cont"][0,0,0]=float("nan")
            if injection=="target":y[0][0,0]=float("inf")
            trace.append(dict(event="batch",epoch=t.current_epoch,step=t.global_step,
                time=x["decoder_time_idx"].cpu().tolist(),rng=old.rng_digest(),
                python_probe=random.random(),numpy_probe=float(np.random.random())))
        def on_before_optimizer_step(self,t,m,op):
            if any(not torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None):
                report["audit_containment"]=True
                raise RuntimeError("AUDIT containment detected missed production finite gate")
        def on_train_batch_end(self,t,m,outputs,b,i):
            trace.append(dict(event="loss",step=t.global_step,value=float(outputs["loss"])))
        def on_train_epoch_start(self,t,m):
            trace.append(dict(event="epoch",epoch=t.current_epoch,step=t.global_step))
    original_step=torch.optim.AdamW.step
    def observed_step(op,*a,**kw):
        result=original_step(op,*a,**kw)
        report["synthetic_optimizer_steps"]+=1
        trace.append(dict(event="optimizer",lr=op.param_groups[0]["lr"]))
        if injection=="parameter":
            with torch.no_grad():op.param_groups[0]["params"][0].fill_(float("inf"))
        if injection=="optimizer":
            next(iter(op.state.values()))["exp_avg"].fill_(float("inf"))
        return result
    try:
        with old.guards():
            training,validation,meta=small_fixture(out/"fixture",accumulation)
            report.update(meta)
            args=settings(meta,registry,seed,stochastic,accumulation,device)
            args.stop_after_epoch=1 if stop else None;args.mode=mode;args.run_id=parent
            reg=Registry(registry);expected=contract(training,args,p.QUANTILES)
            checkpoint=None;parent_payload=None
            if mode=="resume-last":checkpoint,parent_payload,_=reg.verified(parent,"last",expected)
            if mode=="weights-only":args.parent_run_id=parent
            model=p.build_model(training,args,checkpoint)
            report["initial_weights"]=old.tensor_digest(model.state_dict())
            if injection in ("gradient","gradient_late"):
                def bad_gradient(gradient):
                    if injection=="gradient" or model.global_step>=2:
                        return torch.full_like(gradient,float("nan"))
                    return gradient
                next(model.parameters()).register_hook(bad_gradient)
            with observe_restore() as restore_calls, patch.object(torch.optim.AdamW,"step",observed_step):
                trainer=p.train(model,training,validation,args,checkpoint,extra_callbacks=[Trace()])
            own=trainer.nmd_registry.manifest(trainer.nmd_run_id)
            report.update(exit_code=0,owned_run_id=trainer.nmd_run_id,contract=expected,
                          global_step=trainer.global_step,epoch=trainer.current_epoch,registry_manifest=own,
                          final_weights=old.tensor_digest(model.state_dict()),final_rng=old.rng_digest())
            torch.save(dict(weights=model.state_dict(),optimizer=trainer.optimizers[0].state_dict(),
                            scheduler=trainer.lr_scheduler_configs[0].scheduler.state_dict()),out/"state.pt")
    except Exception as exc:
        report.update(exit_code=1,error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
    finally:
        report["owned_run_id"]=getattr(locals().get("model"),"_nmd_run_id",None)
        report["trace"]=trace
        report["registry_records"]={q.parent.name:json.loads(q.read_text()) for q in Path(registry).glob("*/run.json")}
        atomic_json(out/"manifest.json",report)
    return report["exit_code"]


def suite(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    os.environ["NMD_REMEDIATION_RUN"]=str(root)
    records=[]
    class Result(unittest.TextTestResult):
        def addSuccess(self,test):super().addSuccess(test);records.append(dict(test=test.id(),status="PASS"))
        def addFailure(self,test,err):super().addFailure(test,err);records.append(dict(test=test.id(),status="FAIL",detail=self._exc_info_to_string(err,test)))
        def addError(self,test,err):super().addError(test,err);records.append(dict(test=test.id(),status="ERROR",detail=self._exc_info_to_string(err,test)))
        def addSubTest(self,test,subtest,err):
            super().addSubTest(test,subtest,err)
            if err:records.append(dict(test=subtest.id(),status="FAIL",detail=self._exc_info_to_string(err,test)))
    start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(unittest.defaultTestLoader.discover("ml/tests",pattern="test_baseline_stage_b_remediation.py"))
    report=old.provenance()
    children=[json.loads(f.read_text()) for f in root.glob("*/manifest.json")]
    unit=json.loads((root/"unit_counts.json").read_text()) if (root/"unit_counts.json").exists() else {}
    report.update(run_id=root.name,results=records,elapsed_seconds=time.monotonic()-start,unit_optimizer_counts=unit,
        counts={s:sum(r["status"]==s for r in records) for s in ["PASS","FAIL","ERROR","SKIP","XFAIL"]},
        synthetic_optimizer_steps=sum(r["synthetic_optimizer_steps"] for r in children)+sum(unit.values()),
        child_manifest_sha256={str(f.relative_to(root)):sha256(f) for f in root.glob("*/manifest.json")},
        exit_code=0 if result.wasSuccessful() else 1)
    atomic_json(root/"manifest.json",report)
    return report["exit_code"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",type=Path);parser.add_argument("--registry",type=Path)
    parser.add_argument("--parent");parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--no-stochastic",action="store_true");parser.add_argument("--stop",action="store_true")
    parser.add_argument("--accumulation",type=int,default=1)
    parser.add_argument("--injection",choices=["input","target","gradient","gradient_late","parameter","optimizer"])
    parser.add_argument("--mode",choices=["fresh","resume-last","weights-only"],default="fresh")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--suite",type=Path)
    args=parser.parse_args()
    if args.worker:return worker(args.worker,args.registry,args.parent,args.seed,not args.no_stochastic,args.stop,args.accumulation,args.injection,args.mode,args.device)
    if args.suite:return suite(args.suite)
    root=old.ROOT/"ml/models/stage_b_remediation"/time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())
    root.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,"-m","ml.scripts.diagnostics.remediate_stage_b","--suite",str(root)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1",MPLCONFIGDIR=str(root/"mpl"))
    start=time.monotonic()
    try:
        with (root/"console.log").open("w") as stream:
            result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
        code=result.returncode
    except subprocess.TimeoutExpired:code=124
    atomic_json(root/"invocation.json",dict(command=command,exit_code=code,elapsed_seconds=time.monotonic()-start))
    print(root)
    return code


if __name__=="__main__":raise SystemExit(main())
