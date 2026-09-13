"""Preregistered Stage B CUDA finalization. Synthetic data only, 32 total updates."""
import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import warnings
from unittest.mock import patch

import numpy as np
import torch
import lightning.pytorch as pl
from ml.scripts import train_tft_population_v2 as p
from ml.scripts import numerical_profile as numerics
from ml.scripts.diagnostics import audit_stage_b as old
from ml.scripts.diagnostics.remediate_stage_b import settings, small_fixture
from ml.scripts.checkpoint_registry import Registry, contract, sha256, digest, atomic_json
from ml.scripts.finite_training import gradient_norm

ROOT=Path(__file__).resolve().parents[3]
TOLERANCES={"loss": {"atol":1e-5,"rtol":1e-4},
            "validation": {"atol":1e-5,"rtol":1e-4},
            "pre_clip": {"atol":1e-5,"rtol":1e-3},
            "post_clip": {"atol":1e-5,"rtol":1e-3},
            "parameters": {"atol":1e-6,"rtol":1e-4},
            "lr": {"atol":0.,"rtol":0.}}
CONFIG=dict(seed=42,context=48,horizon=12,hidden_size=8,hidden_continuous_size=4,
            attention_heads=1,lstm_layers=1,dropout=.1,batch_size=2,lr=3e-4,
            gradient_clip=1.,accumulate_grad_batches=1,num_workers=0,epochs=4,max_steps=8,
            precision="32-true",swa=False,device="cuda")
SOURCES=["ml/scripts/diagnostics/finalize_stage_b.py","ml/scripts/numerical_profile.py",
         "ml/scripts/baseline_training.py","ml/scripts/checkpoint_registry.py",
         "ml/scripts/train_tft_population_v2.py","ml/scripts/finite_training.py",
         "ml/scripts/diagnostics/remediate_stage_b.py","ml/scripts/diagnostics/audit_stage_b.py",
         "configs/baseline_v1.json","docs/STAGE_B_FINAL_REVIEW.md"]


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def equal(a,b):
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray) and isinstance(b,np.ndarray):return np.array_equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def rng():
    return cpu(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all()))


def state(trainer,model):
    return cpu(dict(model=model.state_dict(),optimizer=trainer.optimizers[0].state_dict(),
        scheduler=trainer.lr_scheduler_configs[0].scheduler.state_dict(),rng=rng(),
        callbacks={c.state_key:c.state_dict() for c in trainer.callbacks if c.state_dict()},
        global_step=trainer.global_step,epoch=trainer.current_epoch))


def preregister(root):
    root.mkdir(parents=True,exist_ok=False)
    spec=dict(version="stage-b-finalization-1",config=CONFIG,tolerances=TOLERANCES,
              pairs=["C1-C2","C1-C3","C2-C3","C1-PR","C2-PR","C3-PR"],
              positive_updates=32,negative_updates=0,suite_timeout_seconds=600,process_timeout_seconds=90,
              allowed_nondeterministic_operators=["upsample_linear1d_backward_out_cuda"],
              baseline_config=json.loads((ROOT/"configs/baseline_v1.json").read_text()),
              source_sha256={s:sha256(ROOT/s) for s in SOURCES},
              head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              git_status=subprocess.check_output(["git","status","--short"],text=True),
              frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    atomic_json(root/"preregistration.json",spec)
    (root/"preregistration.sha256").write_text(sha256(root/"preregistration.json")+"\n")


def verify_preregistration(root):
    path=root/"preregistration.json"
    if sha256(path)!=(root/"preregistration.sha256").read_text().strip():raise ValueError("Preregistration changed")
    spec=json.loads(path.read_text())
    if spec["tolerances"]!=TOLERANCES or spec["config"]!=CONFIG:raise ValueError("Frozen protocol changed")
    for name,expected in spec["source_sha256"].items():
        if sha256(ROOT/name)!=expected:raise ValueError(f"Frozen source changed: {name}")
    return spec


def worker(root,name):
    verify_preregistration(root)
    out=root/name;out.mkdir(exist_ok=False)
    start=time.monotonic();report=dict(name=name,actual_updates=0,exit_code=1,trace=[],validation=[])
    raw={};model=None;trainer=None
    injection={"N_input":"input","N_target":"target","N_gradient":"gradient"}.get(name)
    captured=[]
    try:
        with warnings.catch_warnings(record=True) as captured,old.guards():
            warnings.simplefilter("always")
            torch.set_num_threads(1);pl.seed_everything(42,workers=True)
            numerics.configure("cuda")
            if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable in worker")
            training,validation,metadata=small_fixture(out/"fixture")
            args=settings(metadata,root/"owned",device="cuda")
            for key,value in CONFIG.items():
                if hasattr(args,key) and getattr(args,key)!=value:raise ValueError(f"Fixture config mismatch: {key}")
            args.stop_after_epoch=1 if name=="P" else None
            registry=Registry(root/"owned");expected=contract(training,args,p.QUANTILES)
            checkpoint=None;parent_payload=None
            if name=="R":
                args.mode="resume-last"
                args.run_id=json.loads((root/"P/manifest.json").read_text())["run_id"]
                checkpoint,parent_payload,parent_record=registry.verified(args.run_id,"last",expected)
                report["parent_checkpoint"]=parent_record
            model=p.build_model(training,args,checkpoint)
            raw["initial_parameters"]=cpu(dict(model.named_parameters()))
            report.update(initial_sha256=old.tensor_digest(raw["initial_parameters"]),contract=expected,
                          fixture_sha256=metadata["parquet_sha256"],window_index_sha256=expected["window_index_sha256"])
            if injection=="gradient":
                next(model.parameters()).register_hook(lambda g:torch.full_like(g,float("nan")))
            class Trace(pl.Callback):
                def on_train_start(self,t,m):
                    torch.cuda.synchronize()
                    report["flags"]=numerics.verify("cuda")
                    raw["start"]=state(t,m)
                    report["environment"]=dict(torch=torch.__version__,cuda=torch.version.cuda,
                        cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(0),
                        capability=list(torch.cuda.get_device_capability(0)),threads=torch.get_num_threads(),
                        stack=expected["stack"],python=expected["python"],flags=report["flags"],
                        driver=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).strip())
                    if parent_payload is not None:
                        saved_rng=next(v for k,v in parent_payload["callbacks"].items() if "BoundaryRNG" in k)
                        checks=dict(model=equal(raw["start"]["model"],parent_payload["state_dict"]),
                            optimizer=equal(raw["start"]["optimizer"],parent_payload["optimizer_states"][0]),
                            scheduler=equal(raw["start"]["scheduler"],parent_payload["lr_schedulers"][0]),
                            rng=equal(raw["start"]["rng"],{k:saved_rng[k] for k in ("python","numpy","torch","cuda")}),
                            callbacks=equal(raw["start"]["callbacks"],parent_payload["callbacks"]),
                            global_step=t.global_step==parent_payload["global_step"],
                            epoch=t.current_epoch==parent_payload["epoch"])
                        report["restore_checks"]=checks
                        # Evidence is captured before any forward; failed checks remain a gate.
                def on_train_batch_start(self,t,m,b,i):
                    torch.cuda.synchronize();numerics.verify("cuda")
                    x,y=b
                    ids={k:cpu(x[k]).tolist() for k in ("groups","decoder_time_idx","encoder_lengths","decoder_lengths")}
                    report["trace"].append(dict(step=t.global_step,epoch=t.current_epoch,ids=ids,
                        batch_sha256=old.tensor_digest({**x,"audit_target":y[0]})))
                    raw.setdefault("batch_rng",[]).append(rng())
                    if injection=="input":x["encoder_cont"][0,0,0]=float("nan")
                    if injection=="target":y[0][0,0]=float("inf")
                def on_before_backward(self,t,m,loss):
                    torch.cuda.synchronize();report["trace"][-1]["loss"]=float(loss.detach())
                def on_before_optimizer_step(self,t,m,op):
                    if any(not torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None):
                        report["audit_containment"]=True
                        raise RuntimeError("AUDIT containment: production missed nonfinite gradient")
                    torch.cuda.synchronize();report["trace"][-1]["pre_clip"]=float(gradient_norm(m.parameters()))
                def on_validation_end(self,t,m):
                    if not t.sanity_checking:
                        report["validation"].append(dict(epoch=t.current_epoch,step=t.global_step,
                                                       value=float(t.callback_metrics["val_loss"])))
            original=torch.optim.AdamW.step
            def observed(op,*a,**kw):
                torch.cuda.synchronize()
                row=report["trace"][-1]
                row["post_clip"]=float(gradient_norm(model.parameters()))
                row["lr"]=[group["lr"] for group in op.param_groups]
                result=original(op,*a,**kw)
                torch.cuda.synchronize();report["actual_updates"]+=1;row["finite"]=True
                return result
            with patch.object(torch.optim.AdamW,"step",observed):
                trainer=p.train(model,training,validation,args,checkpoint,extra_callbacks=[Trace()])
            torch.cuda.synchronize()
            raw["final"]=state(trainer,model)
            raw["parameters"]=cpu(dict(model.named_parameters()));raw["buffers"]=cpu(dict(model.named_buffers()))
            report.update(exit_code=0,global_step=trainer.global_step,epoch=trainer.current_epoch)
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
    finally:
        report["warnings"]=[dict(message=str(w.message),category=w.category.__name__,filename=w.filename,lineno=w.lineno) for w in captured]
        for w in report["warnings"]:print(f'{w["category"]}: {w["message"]}',file=sys.stderr)
        report["run_id"]=getattr(model,"_nmd_run_id",None)
        if report["run_id"]:
            report["registry"]=Registry(root/"owned").manifest(report["run_id"])
        report["elapsed_seconds"]=time.monotonic()-start
        torch.save(raw,out/"raw.pt")
        report["raw_sha256"]=sha256(out/"raw.pt")
        atomic_json(out/"manifest.json",report)
    return report["exit_code"]


def differences(a,b,tolerance):
    a=torch.as_tensor(a).detach().cpu().double();b=torch.as_tensor(b).detach().cpu().double()
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Invalid comparison shape/nonfinite")
    delta=(a-b).abs();atol=tolerance["atol"];rtol=tolerance["rtol"]
    limit=atol+rtol*torch.maximum(a.abs(),b.abs())
    flat=delta.flatten();worst=int(flat.argmax()) if flat.numel() else None
    result=dict(bitwise_equal=torch.equal(a,b),max_abs=float(flat.max()) if flat.numel() else 0.,
        rms=float(delta.square().mean().sqrt()) if flat.numel() else 0.,
        relative_l2=float(delta.norm()/max(float(a.norm()),float(b.norm()),1e-12)),
        exceedances=int((delta>limit).sum()),worst_flat_index=worst)
    if atol or rtol:
        ratio=delta/limit;result["max_normalized_error"]=float(ratio.max()) if flat.numel() else 0.
        result["worst_normalized_flat_index"]=int(ratio.flatten().argmax()) if flat.numel() else None
    return result


def compare(root):
    verify_preregistration(root)
    names=["C1","C2","C3","P","R","N_input","N_target","N_gradient"]
    reports={n:json.loads((root/n/"manifest.json").read_text()) for n in names}
    raws={n:torch.load(root/n/"raw.pt",map_location="cpu",weights_only=False) for n in names[:5]}
    checks={}
    for n in names:
        r=reports[n];positive=not n.startswith("N_")
        checks[n+"_execution"]=r["exit_code"]==(0 if positive else 1)
        checks[n+"_updates"]=r["actual_updates"]==((4 if n in ("P","R") else 8) if positive else 0)
        checks[n+"_warnings"]=all("upsample_linear1d_backward_out_cuda" in w["message"] for w in r["warnings"]
            if "deterministic" in w["message"].lower() and "implementation" in w["message"].lower())
        if positive:
            checks[n+"_finite"]=all(row.get("finite",False) for row in r["trace"])
            checks[n+"_checkpoint"]=bool(r.get("registry",{}).get("last"))
        else:
            checks[n+"_production_finite"]=r.get("error","").startswith("FloatingPointError:") and not r.get("audit_containment",False)
            checks[n+"_no_valid_checkpoint"]=r.get("registry",{}).get("last") is None and r.get("registry",{}).get("best") is None
    if not all(checks.values()):
        atomic_json(root/"comparisons.json",dict(verdict="FAIL",checks=checks,pairs={}))
        return False
    checks["resume_next_epoch"]=reports["R"]["trace"][0]["epoch"]==reports["R"]["parent_checkpoint"]["metadata"]["next_epoch"]
    checks["own_boundary_restore"]=all(reports["R"].get("restore_checks",{}).values()) and bool(reports["R"].get("restore_checks"))
    # Restoring the checkpoint must not consume CUDA/CPU random draws before forward.
    checks["resume_rng_before_first_forward"]=equal(raws["R"]["start"]["rng"],raws["R"]["batch_rng"][0])
    reports["PR"]={**reports["R"],"trace":reports["P"]["trace"]+reports["R"]["trace"],
                   "validation":reports["P"]["validation"]+reports["R"]["validation"]}
    raws["PR"]={**raws["R"],"initial_parameters":raws["P"]["initial_parameters"]}
    results={}
    for left,right in list(itertools.combinations(["C1","C2","C3"],2))+[(n,"PR") for n in ["C1","C2","C3"]]:
        a,b=reports[left],reports[right];ar,br=raws[left],raws[right]
        invariants=dict(initial=equal(ar["initial_parameters"],br["initial_parameters"]),
            contract=a["contract"]==b["contract"],environment=a["environment"]==b["environment"],
            fixture=a["fixture_sha256"]==b["fixture_sha256"],windows=a["window_index_sha256"]==b["window_index_sha256"],
            order=[{k:r[k] for k in ("step","epoch","ids","batch_sha256")} for r in a["trace"]]==[{k:r[k] for k in ("step","epoch","ids","batch_sha256")} for r in b["trace"]],
            progress=a["global_step"]==b["global_step"]==8 and a["epoch"]==b["epoch"]==4,
            validation_order=[(r["epoch"],r["step"]) for r in a["validation"]]==[(r["epoch"],r["step"]) for r in b["validation"]])
        metrics={key:differences([r[key] for r in a["trace"]],[r[key] for r in b["trace"]],TOLERANCES[key]) for key in ("loss","lr","pre_clip","post_clip")}
        metrics["validation"]=differences([r["value"] for r in a["validation"]],[r["value"] for r in b["validation"]],TOLERANCES["validation"])
        params={key:differences(value,br["parameters"][key],TOLERANCES["parameters"]) for key,value in ar["parameters"].items()}
        def flatten(values):return torch.cat([v.flatten().double() for v in values.values()])
        av,bv=flatten(ar["parameters"]),flatten(br["parameters"])
        updates=[float((flatten(r["parameters"])-flatten(r["initial_parameters"])).norm()) for r in (ar,br)]
        per_step=[{key:abs(a["trace"][i][key]-b["trace"][i][key]) for key in ("loss","pre_clip","post_clip")} for i in range(8)]
        passed=all(invariants.values()) and all(m["exceedances"]==0 for m in [*metrics.values(),*params.values()])
        results[left+"-"+right]=dict(pass_gate=passed,invariants=invariants,metrics=metrics,parameters=params,
            worst_parameter=max(params,key=lambda k:params[k]["max_normalized_error"]),
            parameter_aggregate=differences(av,bv,TOLERANCES["parameters"]),per_step_max_abs=per_step,
            update_norms=updates,drift_over_update=float((av-bv).norm())/max(*updates,1e-12))
    passed=all(checks.values()) and all(r["pass_gate"] for r in results.values())
    atomic_json(root/"comparisons.json",dict(verdict="PASS" if passed else "FAIL",checks=checks,pairs=results))
    return passed


def run(root):
    verify_preregistration(root);start=time.monotonic();invocations=[]
    for name in ["C1","C2","C3","P","R","N_input","N_target","N_gradient"]:
        remaining=600-(time.monotonic()-start)
        if remaining<=0:break
        command=[sys.executable,"-m","ml.scripts.diagnostics.finalize_stage_b","--worker",name,"--root",str(root)]
        began=time.monotonic()
        with (root/(name+".log")).open("w") as stream:
            try:code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=min(90,remaining)).returncode
            except subprocess.TimeoutExpired:code=124
        invocations.append(dict(name=name,command=command,exit_code=code,elapsed_seconds=time.monotonic()-began))
        atomic_json(root/"invocations.json",invocations)
        print(name,code,flush=True)
        if code!= (1 if name.startswith("N_") else 0):break
    complete=len(invocations)==8
    passed=compare(root) if complete else False
    atomic_json(root/"result.json",dict(verdict="PASS" if passed else "FAIL",elapsed_seconds=time.monotonic()-start,
                complete=complete,preregistration_sha256=sha256(root/"preregistration.json"),invocations=invocations,
                actual_updates=sum(json.loads(f.read_text())["actual_updates"] for f in root.glob("*/manifest.json"))))
    return 0 if passed else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--prepare",action="store_true");parser.add_argument("--run",action="store_true")
    parser.add_argument("--worker",choices=["C1","C2","C3","P","R","N_input","N_target","N_gradient"])
    args=parser.parse_args();root=args.root.resolve()
    if args.prepare:preregister(root);return 0
    if args.worker:return worker(root,args.worker)
    if args.run:return run(root)
    parser.error("Choose prepare, run or worker")


if __name__=="__main__":raise SystemExit(main())
