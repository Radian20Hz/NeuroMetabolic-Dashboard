"""Versioned local checkpoint ownership, compatibility and atomic publication.

The registry is a local trust boundary, not authentication for downloaded files.
Only runs CREATED here may be loaded. No folder scanning, legacy import or
sidecar-based registration of foreign pickle files is supported.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import uuid

import numpy as np
import pandas as pd
import torch

try:
    from .finite_training import require_finite
except ImportError:
    from finite_training import require_finite

try:
    from .numerical_profile import profile as numerical_profile
except ImportError:
    from numerical_profile import profile as numerical_profile

PROTOCOL = "nmd-baseline-v1.0-stage-c-1"
CANONICAL_STAGE_A_SHA256 = "d14fe31b1b81971639ca8d5ba17b4882fbd6711569e30785f0da7b91a0c031cf"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic(value):
    """Deterministic fitted-state representation, never object repr/address."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError("Nonfinite checkpoint contract")
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, torch.Tensor):
        require_finite(value, "contract tensor")
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.detach().cpu().tolist()}
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": semantic(value.tolist())}
    if isinstance(value, pd.DataFrame):
        return {"columns": semantic(value.columns.tolist()), "index": semantic(value.index.tolist()),
                "dtypes": list(map(str,value.dtypes)), "values": semantic(value.to_numpy())}
    if isinstance(value, pd.Series):
        return {"index": semantic(value.index.tolist()), "values": semantic(value.to_numpy())}
    if isinstance(value, dict):
        # Preserve key type: category integer 1 is distinct from string "1".
        return {"mapping": sorted([[semantic(k), semantic(v)] for k,v in value.items()], key=lambda pair: json.dumps(pair[0],sort_keys=True))}
    if isinstance(value, (list, tuple)):
        return [semantic(v) for v in value]
    if hasattr(value, "__dict__") and type(value).__module__.startswith(("sklearn.", "pytorch_forecasting.data")):
        return {"class": type(value).__module__+"."+type(value).__qualname__, "state": semantic(vars(value))}
    raise TypeError(f"Unsupported checkpoint contract type: {type(value).__qualname__}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def contract(dataset, args, quantiles):
    if getattr(dataset.target_normalizer, "nmd_revision", None) != "observed-train-encoded-subject-v2":
        raise ValueError("Checkpoint compatibility: historical normalizer semantics")
    data_hash = getattr(args, "dataset_sha256", None)
    if not isinstance(data_hash,str) or len(data_hash) != 64 or any(c not in "0123456789abcdef" for c in data_hash):
        raise ValueError("Verified dataset_sha256 is required for baseline training")
    if args.epochs <= 0 or not math.isfinite(args.epochs):
        raise ValueError("Baseline requires finite positive epoch budget")
    if not getattr(args, "no_swa", True):
        raise ValueError("SWA ON is outside the certified baseline protocol")
    if getattr(args, "num_workers", 0) != 0:
        raise ValueError("Baseline resume profile supports num_workers=0 only")
    parameters = dataset.get_parameters()
    schema_keys = ["time_idx", "target", "group_ids", "weight", "static_categoricals", "static_reals",
        "time_varying_known_categoricals", "time_varying_known_reals", "time_varying_unknown_categoricals",
        "time_varying_unknown_reals", "variable_groups", "categorical_encoders", "scalers", "lags",
        "add_relative_time_idx", "add_target_scales", "add_encoder_length", "allow_missing_timesteps"]
    length_keys = ["max_encoder_length", "min_encoder_length", "max_prediction_length", "min_prediction_length", "randomize_length"]
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    batches = len(dataset) // args.batch_size
    accumulation = getattr(args, "accumulate_grad_batches", 1)
    if args.batch_size <= 0 or accumulation <= 0 or batches < 1:
        raise ValueError("Baseline requires a positive complete training batch and accumulation")
    steps_per_epoch = math.ceil(batches / accumulation)
    planned = steps_per_epoch * args.epochs
    if getattr(args, "max_steps", planned) not in (None, -1, planned):
        raise ValueError("max_steps must equal the original complete-epoch training budget")
    model_keys = ["hidden_size", "hidden_continuous_size", "attention_heads", "dropout", "lstm_layers"]
    architecture = {k:getattr(args,k) for k in model_keys}
    result = dict(protocol=PROTOCOL, numerical_profile=numerical_profile("cpu" if args.no_gpu else "cuda"), canonical_stage_a_sha256=CANONICAL_STAGE_A_SHA256,
        dataset_sha256=data_hash, data_protocol="baseline_v1_stage_a", schema=semantic({k:parameters[k] for k in schema_keys}),
        window_index_sha256=digest(semantic(dataset.index)),
        synthetic_audit=getattr(args,"synthetic_audit",False),ordered_reals=dataset.reals, ordered_categoricals=dataset.flat_categoricals,
        tensor_dtypes={k:str(v.dtype) for k,v in dataset.data.items() if isinstance(v,torch.Tensor)},
        units={"target":"mg/dL", "time_idx":"5 minutes", "other_features":"unchanged Stage A feature definitions"},
        normalizer_revision="observed-train-encoded-subject-v2", normalizer=semantic(dataset.target_normalizer), lengths=semantic({k:parameters[k] for k in length_keys}),
        context=args.context, horizon=args.horizon, quantiles=list(quantiles), architecture=architecture,
        loss=dict(name="ClinicalQuantileLoss", factor=2, hypo_threshold=70., hypo_weight=2.5, reduction="valid_positions_mean"),
        optimizer=dict(name="AdamW", lr=args.lr, eps=1e-7, weight_decay=.01),
        scheduler=dict(name="LambdaLR_SequentialLR_CosineAnnealingWarmRestarts", interval="step", warmup=max(50,planned//20),
                       T_0=15*steps_per_epoch, T_mult=2, eta_min=1e-6),
        budget=dict(epochs=args.epochs,total_steps=planned,steps_per_epoch=steps_per_epoch),
        data_order=dict(batch_size=args.batch_size,drop_last=True,accumulation=accumulation,
                        shuffle=getattr(args,"shuffle",True),sampler="epoch_seeded_v1",loader_worker_seed="reset_each_iterator_workers_0",num_workers=0,train_windows=len(dataset)),
        callbacks=dict(early_stopping_patience=20,min_delta=.0001,monitor="val_loss",mode="min",swa=False,
                       gradient_clip=args.gradient_clip,gradient_warmup=200,gradient_alert_patience=30),
        precision="32-true", device="cpu" if args.no_gpu else "cuda", threads=torch.get_num_threads(),
        seed=getattr(args,"seed",42),
        stack={k:importlib.metadata.version(k) for k in ["torch","lightning","pytorch-forecasting","numpy","pandas"]},
        python=platform.python_version(),
        code_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name("finite_training.py"),Path(__file__).with_name("train_tft_population_v2.py"),Path(__file__).with_name("baseline_training.py"),Path(__file__).with_name("numerical_profile.py")]})
    return result


def check_compatible(saved, expected, mode="resume-last"):
    # Weights-only is a NEW optimization run, but retains all data/model semantics.
    optimization = {"optimizer","scheduler","budget","data_order","callbacks","seed"}
    keys = set(expected) if mode == "resume-last" else set(expected)-optimization
    if set(saved) != set(expected):
        raise ValueError("Checkpoint compatibility: missing/unknown contract fields")
    for key in sorted(keys):
        if saved[key] != expected[key]:
            raise ValueError(f"Checkpoint compatibility mismatch: {key}")


def atomic_json(path, obj):
    path=Path(path)
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with tmp.open("x") as stream:
            json.dump(obj,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


class Registry:
    def __init__(self, root):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.index=self.root/"owned_runs.json"
        if not self.index.exists():
            if any(self.root.iterdir()):
                raise ValueError("Refusing to import an existing directory as a trusted run registry")
            atomic_json(self.index,{"version":1,"runs":{}})

    def create(self, expected, mode="fresh", parent=None):
        if mode not in ("fresh","weights-only"):
            raise ValueError("New runs require fresh or weights-only mode")
        run_id=uuid.uuid4().hex
        directory=self.root/run_id;directory.mkdir()
        manifest=dict(run_id=run_id,version=1,status="running",contract=expected,
                      lineage=dict(mode=mode,parent=parent),last=None,best=None,weights_only=None,generations=[])
        atomic_json(directory/"run.json",manifest)
        index=json.loads(self.index.read_text());index["runs"][run_id]=dict(contract_sha256=digest(expected),path=str(directory))
        atomic_json(self.index,index)
        return run_id

    def manifest(self,run_id):
        index=json.loads(self.index.read_text())
        if index.get("version") != 1:
            raise ValueError("Unknown registry format version")
        if run_id not in index["runs"]:
            raise ValueError("Untrusted or unknown run_id")
        entry=index["runs"][run_id]
        directory=self.root/run_id
        if directory.is_symlink() or str(directory.resolve()) != entry["path"]:
            raise ValueError("Run relocation is not supported")
        value=json.loads((directory/"run.json").read_text())
        if value.get("version") != 1:
            raise ValueError("Unknown run format version")
        if value["run_id"] != run_id or digest(value["contract"]) != entry["contract_sha256"]:
            raise ValueError("Registry/manifest contract mismatch")
        return value

    def verified(self,run_id,role,expected):
        if role not in ("last","best","weights_only"):
            raise ValueError("Unknown checkpoint role")
        manifest=self.manifest(run_id)
        check_compatible(manifest["contract"],expected,"resume-last" if role=="last" else role)
        record=manifest.get(role)
        if not record or record["status"] != "valid":
            raise ValueError(f"No valid {role} checkpoint")
        wanted_kind = "weights-only" if role=="weights_only" else "epoch-boundary"
        if record["metadata"].get("kind") != wanted_kind:
            raise ValueError("Checkpoint role mismatch")
        if wanted_kind == "epoch-boundary":
            for key in ("boundary_complete", "terminated", "global_step", "val_loss"):
                if record.get(key) != record["metadata"].get(key):
                    raise ValueError("Checkpoint sidecar fields disagree")
        if role=="last" and not record["boundary_complete"]:
            raise ValueError("Mid-epoch/partial accumulation resume is unsupported")
        if role=="last" and record["terminated"]:
            raise ValueError("Completed/early-stopped run cannot resume; use a new weights-only run")
        path=self.root/run_id/record["file"]
        if path.is_symlink() or path.resolve().parent != self.root/run_id or not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError("Checkpoint file/hash/ownership mismatch")
        # Only reached after local ownership, compatibility and bytes verification.
        payload=torch.load(path,map_location="cpu",weights_only=(role=="weights_only"))
        if payload.get("nmd_checkpoint") != record["metadata"]:
            raise ValueError("Checkpoint sidecar/payload mismatch")
        if record["metadata"]["contract_sha256"] != digest(expected) and role=="last":
            raise ValueError("Checkpoint payload contract mismatch")
        if wanted_kind == "epoch-boundary":
            for key in ("optimizer_states", "lr_schedulers", "loops", "callbacks", "global_step", "epoch"):
                if key not in payload:
                    raise ValueError("Incomplete full-state checkpoint")
            if payload["global_step"] != record["global_step"] or payload["epoch"] != record["metadata"]["completed_epoch"]:
                raise ValueError("Checkpoint progress metadata mismatch")
        if wanted_kind == "epoch-boundary":
            parameters = payload.get("dataset_parameters")
            hparams = payload.get("hyper_parameters")
            if not isinstance(parameters, dict) or not isinstance(hparams, dict):
                raise ValueError("Missing model/dataset checkpoint semantics")
            if semantic(parameters["target_normalizer"]) != manifest["contract"]["normalizer"]:
                raise ValueError("Payload normalizer mismatch")
            for key, value in manifest["contract"]["architecture"].items():
                checkpoint_key = "attention_head_size" if key == "attention_heads" else key
                if hparams.get(checkpoint_key) != value:
                    raise ValueError("Payload architecture mismatch")
            if hparams.get("x_reals") != manifest["contract"]["ordered_reals"]:
                raise ValueError("Payload feature order mismatch")
            if list(hparams["loss"].quantiles) != manifest["contract"]["quantiles"]:
                raise ValueError("Payload quantile order mismatch")
        require_finite(payload["state_dict"],"loaded checkpoint parameters")
        require_finite(payload.get("optimizer_states",[]),"loaded checkpoint optimizer")
        return path,payload,record

    def publish(self,run_id,path,metadata,val_loss):
        require_finite(float(val_loss),"checkpoint monitor")
        manifest=self.manifest(run_id)
        if metadata["contract_sha256"] != digest(manifest["contract"]):
            raise ValueError("Publication contract mismatch")
        path=Path(path)
        if path.is_symlink() or path.resolve().parent != self.root/run_id:
            raise ValueError("Publication outside owned run")
        # Validate our just-written generation before advancing any valid pointer.
        payload=torch.load(path,map_location="cpu",weights_only=False)
        if payload.get("nmd_checkpoint") != metadata or metadata.get("run_id") != run_id:
            raise ValueError("Publication payload metadata mismatch")
        if not metadata.get("boundary_complete") or metadata.get("kind") != "epoch-boundary":
            raise ValueError("Only completed epoch checkpoints may be published")
        if payload.get("global_step") != metadata["global_step"] or payload.get("epoch") != metadata["completed_epoch"]:
            raise ValueError("Publication progress mismatch")
        if metadata.get("val_loss") != float(val_loss):
            raise ValueError("Publication monitor mismatch")
        require_finite(payload["state_dict"], "published parameters")
        require_finite(payload["optimizer_states"], "published optimizer state")
        record=dict(file=path.name,sha256=sha256(path),status="valid",metadata=metadata,
                    boundary_complete=metadata["boundary_complete"],terminated=metadata["terminated"],
                    global_step=metadata["global_step"],val_loss=float(val_loss))
        manifest["generations"].append(record)
        manifest["last"]=record
        best=manifest["best"]
        if best is None or (record["val_loss"],record["global_step"]) < (best["val_loss"],best["global_step"]):
            manifest["best"]=record
        manifest["status"]="completed" if record["terminated"] else "running"
        atomic_json(self.root/run_id/"run.json",manifest)
        return record

    def export_weights(self,run_id,expected):
        _,payload,parent=self.verified(run_id,"best",expected)
        manifest=self.manifest(run_id)
        path=self.root/run_id/("weights-"+uuid.uuid4().hex+".pt")
        metadata=dict(kind="weights-only",run_id=run_id,contract_sha256=digest(manifest["contract"]),parent_sha256=parent["sha256"])
        torch.save(dict(state_dict=payload["state_dict"],nmd_checkpoint=metadata),path)
        manifest["weights_only"]=dict(file=path.name,sha256=sha256(path),status="valid",metadata=metadata,
                                      boundary_complete=False,terminated=False)
        atomic_json(self.root/run_id/"run.json",manifest)
        return path

    def failure(self,run_id,reason):
        # Existing valid pointers remain intact. Failure is a separate segment event.
        path=self.root/run_id/("failure-"+uuid.uuid4().hex+".json")
        atomic_json(path,dict(status="invalid_segment",reason=str(reason)))
