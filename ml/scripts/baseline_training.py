"""Certified baseline training orchestration, completed-validation boundaries.

Public Lightning APIs only. SWA experiments and mid-epoch replay are not part of
this protocol. Training never invokes evaluation on the real test split.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

try:
    from .checkpoint_registry import Registry,contract,digest,atomic_json
    from .finite_training import optimizer_finite,require_finite
except ImportError:
    from checkpoint_registry import Registry,contract,digest,atomic_json
    from finite_training import optimizer_finite,require_finite


class EpochSampler(torch.utils.data.Sampler):
    """A public set_epoch sampler: same permutation for an epoch in every process."""
    def __init__(self, dataset, seed, shuffle):
        self.dataset=dataset;self.seed=seed;self.shuffle=shuffle;self.epoch=0
    def __len__(self):return len(self.dataset)
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def __iter__(self):
        if self.shuffle:
            generator=torch.Generator().manual_seed(self.seed+self.epoch)
            return iter(torch.randperm(len(self.dataset),generator=generator).tolist())
        return iter(range(len(self.dataset)))


class BoundaryDataLoader(torch.utils.data.DataLoader):
    """Worker seed has no stateful role in the workers=0 profile.

    Keep it separate from both the epoch sampler and model RNG. Each iterator
    uses the same worker-seed generator initialization in fresh AND resumed runs;
    Lightning's extra setup iterator therefore cannot advance the data-order RNG.
    This is an explicit loader policy, not replay of a private sampler position.
    """
    def __iter__(self):
        self.generator.manual_seed(self.worker_seed)
        return super().__iter__()


def boundary_loader(dataset,batch_size,seed,shuffle,drop_last):
    sampler=EpochSampler(dataset,seed,shuffle)
    generator=torch.Generator().manual_seed(seed+10000)
    base=dataset.to_dataloader(train=drop_last,batch_size=batch_size,num_workers=0,
        shuffle=False,sampler=sampler,drop_last=drop_last,generator=generator)
    loader=BoundaryDataLoader(dataset,batch_sampler=base.batch_sampler,collate_fn=base.collate_fn,
                              num_workers=0,generator=generator)
    loader.worker_seed=seed+10000
    return loader


class BoundaryRNG(pl.Callback):
    def __init__(self,train_generator,val_generator):
        self.train_generator=train_generator;self.val_generator=val_generator
        self.pending=None

    def state_dict(self):
        return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    train_generator=self.train_generator.get_state(),val_generator=self.val_generator.get_state())

    def load_state_dict(self,state):
        # Lightning restores callbacks BEFORE optimizer configuration/data setup.
        # Restore generators now, before constructing the next epoch iterator.
        self.train_generator.set_state(state["train_generator"])
        self.val_generator.set_state(state["val_generator"])
        self.pending=state

    def on_train_start(self,trainer,module):
        if self.pending is not None:
            state=self.pending
            random.setstate(state["python"]);np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
            if state["cuda"]:
                torch.cuda.set_rng_state_all(state["cuda"])
            self.pending=None


class BoundaryCheckpoint(ModelCheckpoint):
    """Save after validation/ES, before Lightning's epoch-end transition.

    The epoch's batches, optimizer steps, step-scheduler and validation are
    complete. Lightning publicly resumes the remaining bookkeeping, then enters
    the next epoch; we neither rewrite its counters nor change the total budget.
    """
    def __init__(self,registry,run_id,expected,stop_after_epoch=None):
        super().__init__(dirpath=registry.root/run_id,filename="epoch-{epoch:03d}-step-{step}",
            monitor="val_loss",mode="min",save_top_k=-1,save_last=False,
            save_on_train_epoch_end=False,auto_insert_metric_name=False,enable_version_counter=False)
        self.registry=registry;self.run_id=run_id;self.expected=expected
        self.stop_after_epoch=stop_after_epoch;self.publication=None

    def _save_checkpoint(self,trainer,filepath):
        budget=self.expected["budget"]
        next_epoch=trainer.current_epoch+1
        if trainer.global_step != next_epoch*budget["steps_per_epoch"]:
            raise ValueError("Refusing partial-epoch/partial-accumulation checkpoint")
        if "val_loss" not in trainer.callback_metrics:
            raise ValueError("Completed validation is required before checkpoint publication")
        for optimizer in trainer.optimizers:
            optimizer_finite(optimizer)
        val_loss=float(trainer.callback_metrics["val_loss"])
        require_finite(val_loss,"checkpoint validation objective")
        metadata=dict(kind="epoch-boundary",run_id=self.run_id,contract_sha256=digest(self.expected),
            boundary_complete=True,completed_epoch=trainer.current_epoch,next_epoch=next_epoch,
            global_step=trainer.global_step,terminated=bool(trainer.should_stop or next_epoch>=budget["epochs"]),
            monitor="val_loss",mode="min",val_loss=val_loss)
        path=Path(filepath)
        if path.exists():
            raise ValueError("Refusing to overwrite a checkpoint generation")
        tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
        self.publication=metadata
        try:
            trainer.save_checkpoint(tmp,weights_only=False)
            # Full serialization succeeded; publish bytes first, pointer last.
            with tmp.open("rb") as stream:os.fsync(stream.fileno())
            os.replace(tmp,path)
            self.registry.publish(self.run_id,path,metadata,val_loss)
        finally:
            self.publication=None
            tmp.unlink(missing_ok=True)
        # Maintain public ModelCheckpoint state/notification without a second save.
        self._last_global_step_saved=trainer.global_step
        self._last_checkpoint_saved=str(path)

    def on_save_checkpoint(self,trainer,module,checkpoint):
        if self.publication is None:
            raise ValueError("Only validated epoch boundaries may publish baseline checkpoints")
        checkpoint["nmd_checkpoint"]=self.publication

    def on_validation_end(self,trainer,module):
        super().on_validation_end(trainer,module)
        if not trainer.sanity_checking and self.stop_after_epoch is not None and trainer.current_epoch==self.stop_after_epoch:
            trainer.should_stop=True


def registry_for(args,default_root):
    return Registry(getattr(args,"registry_dir",None) or default_root)


def load_model(model_cls,training,args,checkpoint,registry,expected,mode,run_id):
    role={"resume-last":"last","inference-best":"best","weights-only":"weights_only"}[mode]
    path,payload,record=registry.verified(run_id,role,expected)
    if checkpoint is not None and Path(checkpoint).resolve()!=path.resolve():
        raise ValueError("Checkpoint must be the explicit compatible registered role")
    if mode=="weights-only":
        return payload["state_dict"],path,record
    model=model_cls.load_from_checkpoint(path,map_location="cpu",weights_only=False)
    return model,path,record


def train_baseline(model,training,validation,args,checkpoint,model_root,gradient_callback,extra_callbacks=()):
    expected=contract(training,args,model.loss.quantiles)
    if expected["device"]=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this process; select CPU explicitly")
    registry=registry_for(args,model_root)
    mode=getattr(args,"mode","fresh")
    if mode=="inference-best":
        raise ValueError("Inference BEST cannot be used to continue training")
    if mode=="resume-last":
        run_id=getattr(args,"run_id",None)
        path,_,record=registry.verified(run_id,"last",expected)
        if checkpoint is None or Path(checkpoint).resolve()!=path.resolve():
            raise ValueError("Resume requires the explicit registered LAST checkpoint")
    else:
        if checkpoint is not None:
            raise ValueError("Fresh training never discovers or resumes a checkpoint")
        lineage=getattr(model,"_weights_only_parent",None)
        if mode=="weights-only" and lineage is None:
            raise ValueError("Weights-only requires a verified tensor checkpoint parent")
        run_id=registry.create(expected,mode,parent=lineage)
        path=None
    train_loader=boundary_loader(training,args.batch_size,args.seed,getattr(args,"shuffle",True),True)
    val_loader=boundary_loader(validation,args.batch_size*2,args.seed+1,False,False)
    train_generator=train_loader.generator
    val_generator=val_loader.generator
    callbacks=[BoundaryRNG(train_generator,val_generator),
        EarlyStopping(monitor="val_loss",mode="min",patience=20,min_delta=1e-4,check_on_train_epoch_end=False),
        gradient_callback(clip_val=args.gradient_clip,warmup_steps=200),*extra_callbacks,
        BoundaryCheckpoint(registry,run_id,expected,getattr(args,"stop_after_epoch",None))]
    trainer=pl.Trainer(max_epochs=args.epochs,max_steps=expected["budget"]["total_steps"],
        accelerator=expected["device"].replace("cuda","gpu"),devices=1,precision="32-true",
        accumulate_grad_batches=expected["data_order"]["accumulation"],gradient_clip_val=args.gradient_clip,
        gradient_clip_algorithm="norm",callbacks=callbacks,deterministic=True,benchmark=False,
        num_sanity_val_steps=0 if path else 2,logger=CSVLogger(registry.root/run_id,name="segments",version=uuid.uuid4().hex),
        enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=1)
    model._nmd_run_id=run_id
    trainer.nmd_registry=registry;trainer.nmd_run_id=run_id;trainer.nmd_contract=expected
    try:
        trainer.fit(model,train_loader,val_loader,ckpt_path=path,weights_only=False if path else None)
    except Exception as exc:
        # Record the failure and re-raise; never label an invalid segment successful.
        registry.failure(run_id,f"{type(exc).__name__}: {exc}")
        raise
    return trainer
