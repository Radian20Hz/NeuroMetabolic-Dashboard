"""Authorized callback-only CUDA regression: new P/R, old C1-C3 references, no new 3x8."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from ml.scripts.diagnostics import finalize_stage_b as f
from ml.scripts.checkpoint_registry import atomic_json,sha256


def prepare(root,reference):
    # Prove the only production orchestration edit is the added native load hook.
    before=subprocess.check_output(["git","show","HEAD:ml/scripts/baseline_training.py"],text=True)
    old=ast.parse(before);new=ast.parse((f.ROOT/"ml/scripts/baseline_training.py").read_text())
    cls=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=="BoundaryCheckpoint")
    cls.body=[n for n in cls.body if getattr(n,"name",None)!="load_state_dict"]
    if ast.dump(old)!=ast.dump(new):raise ValueError("Production change exceeds callback restore scope")
    historical=json.loads((reference/"preregistration.json").read_text())
    # These unchanged sources determine model/optimizer/ordering/data/numerics.
    for name in ("ml/scripts/train_tft_population_v2.py","ml/scripts/finite_training.py",
                 "ml/scripts/numerical_profile.py","ml/scripts/checkpoint_registry.py",
                 "ml/scripts/diagnostics/audit_stage_b.py","configs/baseline_v1.json"):
        if sha256(f.ROOT/name)!=historical["source_sha256"][name]:raise ValueError(f"Reference invalidated: {name}")
    if historical["config"]!=f.CONFIG or historical["tolerances"]!=f.TOLERANCES:
        raise ValueError("Reference configuration/tolerances changed")
    f.preregister(root)
    spec=json.loads((root/"preregistration.json").read_text())
    spec.update(version="callback-restore-regression-1",positive_updates=8,negative_updates=0,
                execution_scope="P=4 then R=4; compare with retained C1/C2/C3; no new fresh runs",
                pairs=["C1-PR","C2-PR","C3-PR"],reference=str(reference),
                reference_sha256={str(p.relative_to(reference)):sha256(p) for name in ("C1","C2","C3")
                    for p in (reference/name/"manifest.json",reference/name/"raw.pt")},
                production_change="Only BoundaryCheckpoint.load_state_dict added; AST verified",
                retained_reference_preregistration_sha256=sha256(reference/"preregistration.json"))
    spec["source_sha256"]["ml/scripts/diagnostics/check_callback_resume.py"]=sha256(Path(__file__))
    atomic_json(root/"preregistration.json",spec)
    (root/"preregistration.sha256").write_text(sha256(root/"preregistration.json")+"\n")


def compare(root):
    spec=f.verify_preregistration(root);reference=Path(spec["reference"])
    for name,expected in spec["reference_sha256"].items():
        if sha256(reference/name)!=expected:raise ValueError("Historical reference changed")
    p=json.loads((root/"P/manifest.json").read_text());r=json.loads((root/"R/manifest.json").read_text())
    pr=torch.load(root/"P/raw.pt",map_location="cpu",weights_only=False)
    rr=torch.load(root/"R/raw.pt",map_location="cpu",weights_only=False)
    trace=p["trace"]+r["trace"];validation=p["validation"]+r["validation"]
    checks=dict(updates=p["actual_updates"]==r["actual_updates"]==4,
        progress=p["global_step"]==4 and r["global_step"]==8 and r["epoch"]==4,
        same_run=p["run_id"]==r["run_id"],restore=bool(r.get("restore_checks")) and all(r["restore_checks"].values()),
        rng_before_forward=f.equal(rr["start"]["rng"],rr["batch_rng"][0]),
        next_epoch=r["trace"][0]["epoch"]==r["parent_checkpoint"]["metadata"]["next_epoch"],
        finite=all(row.get("finite",False) for row in trace),
        warnings=all("upsample_linear1d_backward_out_cuda" in w["message"] for run in (p,r) for w in run["warnings"]
            if "deterministic" in w["message"].lower() and "implementation" in w["message"].lower()))
    pairs={}
    for name in ("C1","C2","C3"):
        a=json.loads((reference/name/"manifest.json").read_text());ar=torch.load(reference/name/"raw.pt",map_location="cpu",weights_only=False)
        # This is offline comparison, NOT checkpoint migration; source-only delta was reviewed at prepare.
        ac={k:v for k,v in a["contract"].items() if k!="code_sha256"}
        rc={k:v for k,v in r["contract"].items() if k!="code_sha256"}
        exact=dict(contract_except_reviewed_code=ac==rc,environment=a["environment"]==r["environment"],
            initial=f.equal(ar["initial_parameters"],pr["initial_parameters"]),
            batch_order=[{k:row[k] for k in ("step","epoch","ids","batch_sha256")} for row in a["trace"]]==
                        [{k:row[k] for k in ("step","epoch","ids","batch_sha256")} for row in trace],
            validation_order=[(v["epoch"],v["step"]) for v in a["validation"]]==[(v["epoch"],v["step"]) for v in validation])
        metrics={key:f.differences([row[key] for row in a["trace"]],[row[key] for row in trace],f.TOLERANCES[key])
                 for key in ("loss","lr","pre_clip","post_clip")}
        metrics["validation"]=f.differences([v["value"] for v in a["validation"]],[v["value"] for v in validation],f.TOLERANCES["validation"])
        parameters={key:f.differences(value,rr["parameters"][key],f.TOLERANCES["parameters"]) for key,value in ar["parameters"].items()}
        passed=all(exact.values()) and all(v["exceedances"]==0 for v in [*metrics.values(),*parameters.values()])
        pairs[name+"-PR"]=dict(pass_gate=passed,exact=exact,metrics=metrics,parameters=parameters)
    passed=all(checks.values()) and all(v["pass_gate"] for v in pairs.values())
    atomic_json(root/"comparisons.json",dict(verdict="PASS" if passed else "FAIL",checks=checks,pairs=pairs,
                callback_restore=r["callback_restore"],preregistration_sha256=sha256(root/"preregistration.json")))
    return passed


def run(root):
    f.verify_preregistration(root);start=time.monotonic();invocations=[]
    for name in ("P","R"):
        command=[sys.executable,"-m","ml.scripts.diagnostics.finalize_stage_b","--worker",name,"--root",str(root)]
        began=time.monotonic()
        with (root/(name+".log")).open("w") as stream:
            try:code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=90).returncode
            except subprocess.TimeoutExpired:code=124
        invocations.append(dict(name=name,command=command,exit_code=code,elapsed_seconds=time.monotonic()-began))
        atomic_json(root/"invocations.json",invocations)
        print(name,code,flush=True)
        if code:break
    passed=compare(root) if len(invocations)==2 and not any(v["exit_code"] for v in invocations) else False
    atomic_json(root/"result.json",dict(verdict="PASS" if passed else "FAIL",invocations=invocations,
        elapsed_seconds=time.monotonic()-start,actual_updates=sum(json.loads(p.read_text())["actual_updates"] for p in root.glob("*/manifest.json"))))
    return 0 if passed else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--reference",type=Path);parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args();root=args.root.resolve()
    if args.prepare:prepare(root,args.reference.resolve());return 0
    return run(root)


if __name__=="__main__":raise SystemExit(main())
