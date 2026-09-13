"""Acceptance for the approved SWA-OFF epoch-boundary profile.

Historical Stage B reproductions remain in test_baseline_stage_b.py and Git.
This suite maps supported contracts to real regression tests, not xfails/skips.
Run via: .venv/bin/python -m ml.scripts.diagnostics.remediate_stage_b
"""
import copy
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch,PropertyMock

import numpy as np
import torch
import lightning.pytorch as pl
from torch.nn.utils.rnn import pack_padded_sequence
from ml.scripts import train_tft_population_v2 as p
from ml.scripts import baseline_training as training_code
from ml.scripts.checkpoint_registry import Registry,contract,check_compatible,atomic_json,digest,sha256
from ml.scripts.finite_training import CheckedAdamW,gradient_norm
from ml.scripts.diagnostics import audit_stage_b as old
from ml.scripts.diagnostics import remediate_stage_b as harness
from ml.tests.test_baseline_stage_b import StageB as Historical,oracle,nested_equal


def same(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


class Remediation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger().setLevel(logging.ERROR);torch.set_num_threads(1);pl.seed_everything(42,workers=True)
        cls.tmp=tempfile.TemporaryDirectory(prefix="nmd-remediation-")
        cls.root=Path(os.environ.get("NMD_REMEDIATION_RUN",cls.tmp.name))
        cls.guard=old.guards()
        cls.training,cls.validation,cls.meta=harness.small_fixture(cls.root/"unit_fixture")
        cls.args=harness.settings(cls.meta,cls.root/"owned")
        cls.model=p.build_model(cls.training,cls.args,None).eval()
        cls.x,cls.y=next(iter(cls.validation.to_dataloader(train=False,batch_size=2,num_workers=0)))
        cls.children={};cls.counts={"scalar_scheduler":0,"unit_tft":0,"other_scalar":0}
        cls.expected=contract(cls.training,cls.args,p.QUANTILES)

    @classmethod
    def tearDownClass(cls):
        atomic_json(cls.root/"unit_counts.json",cls.counts)
        cls.guard.close();cls.tmp.cleanup()

    def child(self,name,**kwargs):
        if name not in self.children:
            command=[sys.executable,"-m","ml.scripts.diagnostics.remediate_stage_b","--worker",str(self.root/name),"--registry",str(self.root/"owned")]
            for key,value in kwargs.items():
                flag="--"+key.replace("_","-")
                if value is True:command.append(flag)
                elif value is not False and value is not None:command.extend([flag,str(value)])
            with (self.root/(name+".log")).open("w") as stream:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=75)
            value=json.loads((self.root/name/"manifest.json").read_text())
            value["observed_exit_code"]=result.returncode
            self.children[name]=value
        return self.children[name]

    def good(self,name,**kwargs):
        child=self.child(name,**kwargs)
        self.assertEqual(child["observed_exit_code"],0,child.get("traceback"))
        self.assertLessEqual(child["synthetic_optimizer_steps"],8)
        return child

    def state(self,name):return torch.load(self.root/name/"state.pt",weights_only=False,map_location="cpu")

    # Existing finite/causality/math contracts, unchanged assertions.
    test_B00_isolation=Historical.test_B00_forbidden_entry_guards_positive_control
    test_B01_batch_and_padding=Historical.test_B01_real_batch_contract_and_short_decoder_mask
    test_B02_oracle=Historical.test_B02_independent_pinball_threshold_oracle
    test_B03_units_reset=Historical.test_B03_transform_output_once_in_mgdl_and_reset
    test_B04_quantiles=Historical.test_B04_quantile_axis_median_and_crossing_detection
    test_B05_future_unknowns=Historical.test_B05_real_forward_future_unknown_invariance_positive_control
    test_B06_gradient_oracle=Historical.test_B06_analytic_gradient_gradcheck_real_backward
    test_B08_lr_trace=Historical.test_B08_scheduler_boundaries_restart_and_state_resume
    test_B08_zero_budget=Historical.test_B08_invalid_epoch_budget_rejected_explicitly
    test_B10_gradient_callback=Historical.test_B10_gradient_callback_state_roundtrip
    test_B12_early_stopping=Historical.test_B12_early_stopping_sequences_and_resume
    test_B15_large_finite_zero=Historical.test_B15_large_finite_and_constant_zero_loss

    def test_B03_real_lightning_epoch_aggregation(self):
        # Use the inherited TFT.step/log and real Lightning reduction, with fixed
        # predictions to isolate the denominator. No optimizer updates.
        model=p.build_model(self.training,self.args,None)
        losses=[];counts=[]
        batches=[]
        for size,length,error in [(2,12,10.),(1,3,30.)]:
            x={k:v[:size].clone() for k,v in self.x.items()}
            x["decoder_lengths"][:]=length
            x["decoder_target"]=x["decoder_target"][:,:length]
            target=self.y[0][:size,:length]
            x["fixed_error"]=torch.tensor(error)
            batches.append((x,(target,None)))
            losses.append(error*size*length);counts.append(size*length)
        def forward(x,**kwargs):
            return {"prediction":x["decoder_target"][...,None].expand(-1,-1,7)+x["fixed_error"]}
        trainer=pl.Trainer(logger=False,enable_checkpointing=False,enable_progress_bar=False,accelerator="cpu")
        # Bind model itself: override only validation_step to omit PF visualization,
        # keep step + finite log adapter + Lightning epoch aggregation intact.
        import types
        model.validation_step=types.MethodType(lambda m,b,i:m.step(b[0],b[1],i)[0],model)
        model.on_validation_epoch_end=types.MethodType(lambda m:None,model)
        loader=torch.utils.data.DataLoader(batches,batch_size=None,collate_fn=lambda b:b)
        with patch.object(model,"forward",forward):
            result=trainer.validate(model,dataloaders=loader,verbose=False)
        self.assertAlmostEqual(result[0]["val_loss"],sum(losses)/sum(counts),places=5)
        # Second validation resets epoch metric state.
        with patch.object(model,"forward",lambda x,**kw:{"prediction":x["decoder_target"][...,None].expand(-1,-1,7)}):
            result=trainer.validate(model,dataloaders=loader,verbose=False)
        self.assertEqual(result[0]["val_loss"],0.)

    def test_B16_config_and_callback_contracts(self):
        configured=json.loads((old.ROOT/"configs/baseline_v1.json").read_text())
        for field in ["protocol","clinical_loss","precision","early_stopping","automatic_test_evaluation"]:
            bad=copy.deepcopy(configured);bad[field]="unsupported"
            path=self.root/("bad-config-"+field+".json");atomic_json(path,bad)
            with self.assertRaisesRegex(ValueError,field):p.parse_args(["--config",str(path)])
        args=p.parse_args([])
        self.assertEqual(args.data_dir.name,"baseline_v1_stage_a_20260913T115538143463Z")
        left=p.GradientNormLogger();left._high_grad_consecutive=left.EXPLOSION_PATIENCE-1
        right=p.GradientNormLogger();right.load_state_dict(left.state_dict())
        model=torch.nn.Linear(1,1);model.log=lambda *a,**k:None
        trainer=type("Trainer",(),{"global_step":201})()
        for parameter in model.parameters():parameter.grad=torch.full_like(parameter,1000.)
        with self.assertLogs(p.log,level="WARNING") as first:left.on_after_backward(trainer,model)
        with self.assertLogs(p.log,level="WARNING") as second:right.on_after_backward(trainer,model)
        self.assertEqual(first.output,second.output)
        for parameter in model.parameters():parameter.grad.zero_()
        left.on_after_backward(trainer,model);right.on_after_backward(trainer,model)
        self.assertEqual(left.state_dict(),right.state_dict())
        self.assertEqual(right._high_grad_consecutive,0)
        with self.assertRaises(ValueError):p.GradientNormLogger(clip_val=2.).load_state_dict(left.state_dict())

    def test_B16_accumulation_nonfinite_and_partial_publication(self):
        run=self.child("bad_accumulated",injection="gradient",accumulation=2)
        self.assertEqual(run["observed_exit_code"],1)
        self.assertEqual(run["synthetic_optimizer_steps"],0)
        self.assertIn("FloatingPointError",run["error"])
        self.assertFalse(run.get("audit_containment",False))
        self.assertIsNone(run["registry_records"][run["owned_run_id"]]["last"])
        registry=Registry(self.root/"owned");rid=registry.create(self.expected)
        callback=training_code.BoundaryCheckpoint(registry,rid,self.expected)
        with self.assertRaisesRegex(ValueError,"validated epoch"):
            callback.on_save_checkpoint(None,None,{})
        fake=type("Trainer",(),{"current_epoch":0,"global_step":1})()
        with self.assertRaisesRegex(ValueError,"partial-epoch"):
            callback._save_checkpoint(fake,str(self.root/"never.ckpt"))

    def test_B07_stable_clip_and_post_clip_gate(self):
        model=torch.nn.Linear(2,1)
        for parameter in model.parameters():parameter.grad=torch.full_like(parameter,1e30)
        optimizer=CheckedAdamW(model.parameters(),lr=.0003)
        from ml.scripts.finite_training import FiniteModel
        FiniteModel.configure_gradient_clipping(model,optimizer,1.,"norm")
        self.assertLessEqual(float(gradient_norm(model.parameters())),1.000001)
        parameter=next(model.parameters())
        # Closure represents the point AFTER clipping; actual AdamW spy must not run.
        def closure():parameter.grad.fill_(float("nan"))
        with patch.object(torch.optim.AdamW,"step") as step:
            with self.assertRaises(FloatingPointError):optimizer.step(closure)
            step.assert_not_called()

    def test_B10_new_process_full_state_replay(self):
        full=self.good("full")
        partial=self.good("partial",stop=True)
        resumed=self.good("resumed",mode="resume-last",parent=partial["owned_run_id"])
        self.assertEqual((partial["synthetic_optimizer_steps"],resumed["synthetic_optimizer_steps"],resumed["global_step"]),(4,4,8))
        a,b=self.state("full"),self.state("resumed")
        maxdiff=max(float((a["weights"][k]-b["weights"][k]).abs().max()) for k in a["weights"])
        atomic_json(self.root/"replay.json",dict(max_abs_weight_difference=maxdiff,
            optimizer_equal=same(a["optimizer"],b["optimizer"]),scheduler_equal=same(a["scheduler"],b["scheduler"])))
        self.assertTrue(same(a,b),f"Bitwise state mismatch, max weight diff={maxdiff}")
        for event in ("batch","loss","optimizer"):
            self.assertEqual([e for e in full["trace"] if e["event"]==event][4:],
                             [e for e in resumed["trace"] if e["event"]==event],event)
        self.assertEqual(full["final_rng"],resumed["final_rng"])
        # Compare callback state without equating paths of independent runs.
        pa=torch.load(self.root/"owned"/full["owned_run_id"]/full["registry_manifest"]["last"]["file"],weights_only=False)
        pb=torch.load(self.root/"owned"/resumed["owned_run_id"]/resumed["registry_manifest"]["last"]["file"],weights_only=False)
        for key in pa["callbacks"]:
            if "BoundaryCheckpoint" in key:
                def paths_relative(value,run):
                    if isinstance(value,str):return value.replace(str(self.root/"owned"/run),"<same-run-role>")
                    if isinstance(value,dict):return {paths_relative(k,run):paths_relative(v,run) for k,v in value.items()}
                    return value
                left=paths_relative(pa["callbacks"][key],full["owned_run_id"])
                right=paths_relative(pb["callbacks"][key],resumed["owned_run_id"])
            else:left=pa["callbacks"][key];right=pb["callbacks"][key]
            self.assertTrue(same(left,right),key)

    def test_B10_no_dropout_no_shuffle_replay(self):
        self.good("plain_full",no_stochastic=True)
        half=self.good("plain_partial",no_stochastic=True,stop=True)
        resumed=self.good("plain_resumed",no_stochastic=True,mode="resume-last",parent=half["owned_run_id"])
        self.assertEqual(resumed["global_step"],8)
        self.assertEqual(resumed["synthetic_optimizer_steps"],4)
        self.assertTrue(same(self.state("plain_full"),self.state("plain_resumed")))

    def test_B07_accumulation_replay(self):
        self.good("acc_full",accumulation=2)
        half=self.good("acc_partial",accumulation=2,stop=True)
        last=self.good("acc_resumed",accumulation=2,mode="resume-last",parent=half["owned_run_id"])
        self.assertEqual(last["synthetic_optimizer_steps"],4)
        self.assertTrue(same(self.state("acc_full"),self.state("acc_resumed")))

    def test_B11_each_contract_field_rejected_before_deserialization(self):
        partial=self.good("compat_partial",stop=True)
        expected=partial["contract"];registry=Registry(self.root/"owned")
        run=partial["owned_run_id"]
        path,payload,record=registry.verified(run,"last",expected)
        self.assertEqual(payload["global_step"],4)
        for key in expected:
            with self.subTest(field=key):
                changed=copy.deepcopy(expected);changed[key]="deliberate incompatible value"
                with patch("torch.load") as load:
                    with self.assertRaisesRegex(ValueError,key):registry.verified(run,"last",changed)
                    load.assert_not_called()
        missing=copy.deepcopy(expected);del missing["protocol"]
        with self.assertRaisesRegex(ValueError,"fields"):registry.verified(run,"last",missing)
        # Explicit ordered-axis perturbations of equal shape.
        for key in ["ordered_reals","quantiles"]:
            changed=copy.deepcopy(expected);changed[key]=list(reversed(changed[key]))
            with self.assertRaisesRegex(ValueError,key):registry.verified(run,"last",changed)

    def test_B11_roles_weights_only_new_run_and_completed_refusal(self):
        full=self.good("full");registry=Registry(self.root/"owned");run=full["owned_run_id"]
        expected=full["contract"]
        with self.assertRaisesRegex(ValueError,"Completed"):registry.verified(run,"last",expected)
        registry.export_weights(run,expected)
        child=self.good("weights_child",mode="weights-only",parent=run,stop=True)
        self.assertNotEqual(run,child["owned_run_id"])
        self.assertEqual(child["synthetic_optimizer_steps"],4)
        manifest=registry.manifest(child["owned_run_id"])
        self.assertEqual(manifest["lineage"]["mode"],"weights-only")
        self.assertEqual(manifest["lineage"]["parent"]["run_id"],run)
        self.assertEqual([e for e in child["trace"] if e["event"]=="optimizer"][0]["lr"],0.)
        self.assertEqual([e for e in child["trace"] if e["event"]=="batch"][0]["step"],0)
        self.assertEqual(child["initial_weights"],old.tensor_digest(registry.verified(run,"best",expected)[1]["state_dict"]))
        with self.assertRaisesRegex(ValueError,"discovery"):p.find_best_checkpoint()
        self.assertEqual(Path(p.find_best_checkpoint(registry=registry,run_id=run,expected=expected)).name,registry.manifest(run)["best"]["file"])

    def test_B11_tamper_corrupt_foreign_role_and_partial_refusal(self):
        half=self.good("compat_partial",stop=True);registry=Registry(self.root/"owned")
        expected=half["contract"]
        with self.assertRaisesRegex(ValueError,"Untrusted"):
            registry.verified("foreign", "last", expected)
        # Dedicated OWNED clone for corruption tests; never modify the original run.
        test_run=registry.create(expected)
        src=self.root/"owned"/half["owned_run_id"]
        dest=self.root/"owned"/test_run
        source_record=registry.manifest(half["owned_run_id"])["last"]
        original=torch.load(src/source_record["file"],weights_only=False)
        metadata=copy.deepcopy(original["nmd_checkpoint"]);metadata["run_id"]=test_run
        original["nmd_checkpoint"]=metadata
        file=dest/"owned-fixture.ckpt";torch.save(original,file)
        registry.publish(test_run,file,metadata,source_record["val_loss"])
        registry.verified(test_run,"last",expected)
        untouched=file.read_bytes()
        file.write_bytes(b"corrupt")
        with patch("torch.load") as load:
            with self.assertRaisesRegex(ValueError,"hash"):registry.verified(test_run,"last",expected)
            load.assert_not_called()
        file.write_bytes(untouched)
        manifest=registry.manifest(test_run);saved=copy.deepcopy(manifest)
        for change in ("partial","role","invalid","sidecar"):
            bad=copy.deepcopy(saved)
            if change=="partial":bad["last"]["boundary_complete"]=False;bad["last"]["metadata"]["boundary_complete"]=False
            if change=="role":bad["last"]["metadata"]["kind"]="weights-only"
            if change=="invalid":bad["last"]["status"]="invalid"
            if change=="sidecar":bad["last"]["metadata"]["monitor"]="unexpected"
            atomic_json(dest/"run.json",bad)
            with self.subTest(change=change),self.assertRaises(ValueError):registry.verified(test_run,"last",expected)
        atomic_json(dest/"run.json",saved)

    def test_B11_atomic_publication_keeps_previous_pointers(self):
        half=self.good("compat_partial",stop=True);registry=Registry(self.root/"owned")
        source=registry.manifest(half["owned_run_id"])
        run=registry.create(half["contract"]);directory=self.root/"owned"/run
        state=torch.load(self.root/"owned"/half["owned_run_id"]/source["last"]["file"],weights_only=False)
        metadata=copy.deepcopy(state["nmd_checkpoint"]);metadata["run_id"]=run
        state["nmd_checkpoint"]=metadata
        first=directory/"one.ckpt";torch.save(state,first)
        registry.publish(run,first,metadata,source["last"]["val_loss"])
        before=(directory/"run.json").read_bytes()
        second=directory/"two.ckpt";torch.save(state,second)
        with patch("ml.scripts.checkpoint_registry.os.replace",side_effect=OSError("simulated publication interruption")):
            with self.assertRaises(OSError):registry.publish(run,second,metadata,metadata["val_loss"])
        self.assertEqual((directory/"run.json").read_bytes(),before)
        registry.verified(run,"last",half["contract"])

    def test_B11_best_full_precision_and_earlier_step_tie(self):
        half=self.good("compat_partial",stop=True);registry=Registry(self.root/"owned")
        run=registry.create(half["contract"]);directory=self.root/"owned"/run
        parent=Registry(self.root/"owned").manifest(half["owned_run_id"])["last"]
        base=torch.load(self.root/"owned"/half["owned_run_id"]/parent["file"],weights_only=False)
        for step,loss in [(2,10.00002),(4,10.00001),(6,10.00001)]:
            path=directory/(f"arbitrary-{step}.ckpt")
            metadata=dict(contract_sha256=digest(half["contract"]),boundary_complete=True,terminated=False,
                run_id=run,global_step=step,kind="epoch-boundary",val_loss=loss,completed_epoch=step//2-1,next_epoch=step//2)
            payload=copy.deepcopy(base);payload["nmd_checkpoint"]=metadata
            payload["global_step"]=step;payload["epoch"]=step//2-1
            torch.save(payload,path)
            registry.publish(run,path,metadata,loss)
        result=registry.manifest(run)
        self.assertEqual(result["last"]["global_step"],6)
        self.assertEqual(result["best"]["global_step"],4)
        self.assertEqual(result["best"]["val_loss"],10.00001)

    def test_B13_off_profile_rejects_swa(self):
        full=self.good("full")
        self.assertFalse(full["contract"]["callbacks"]["swa"])
        self.assertEqual(full["contract"]["callbacks"]["early_stopping_patience"],20)
        self.assertEqual(self.state("full")["scheduler"]["last_epoch"],8)
        args=copy.copy(self.args);args.no_swa=False
        with self.assertRaisesRegex(ValueError,"SWA"):contract(self.training,args,p.QUANTILES)
        args=copy.copy(self.args);args.epochs=-1
        with self.assertRaisesRegex(ValueError,"positive epoch"):contract(self.training,args,p.QUANTILES)
        args=copy.copy(self.args);args.num_workers=2
        with self.assertRaisesRegex(ValueError,"num_workers"):contract(self.training,args,p.QUANTILES)

    def test_B14_independent_process_reproducibility_seed_control(self):
        first=self.good("full");second=self.good("repeat");other=self.good("seed43",seed=43)
        self.assertEqual(first["trace"],second["trace"])
        self.assertEqual(first["final_weights"],second["final_weights"])
        self.assertNotEqual(first["initial_weights"],other["initial_weights"])
        self.assertNotEqual(first["final_weights"],other["final_weights"])

    def test_B15_nonfinite_all_signs_direct_loss_target_prediction(self):
        for value in [float("nan"),float("inf"),float("-inf")]:
            for where in ["target","prediction"]:
                with self.subTest(value=value,where=where):
                    target=torch.full((1,2),100.);pred=torch.full((1,2,7),110.)
                    if where=="target":target[0,0]=value
                    else:pred[0,0,0]=value
                    metric=p.ClinicalQuantileLoss(quantiles=p.QUANTILES)
                    with self.assertRaises(FloatingPointError):metric(pred,target)
        # Padding, and only padding, may be nonfinite without contributing.
        target=torch.full((2,3),100.);pred=torch.full((2,3,7),110.);pred[1,1:]=float("nan")
        packed=pack_padded_sequence(target,[3,1],batch_first=True,enforce_sorted=False)
        self.assertAlmostEqual(float(p.ClinicalQuantileLoss(quantiles=p.QUANTILES)(pred,(packed,None))),10.,places=5)

    def test_B15_reduction_and_accumulator_overflow(self):
        metric=p.ClinicalQuantileLoss(quantiles=p.QUANTILES)
        with self.assertRaisesRegex(FloatingPointError,"reduction"):
            metric._update_losses_and_lengths(torch.full((1,2),2e38),torch.tensor([2]))
        metric=p.ClinicalQuantileLoss(quantiles=p.QUANTILES)
        metric._update_losses_and_lengths(torch.full((1,1),2e38),torch.tensor([1]))
        before=metric.losses.clone()
        with self.assertRaisesRegex(FloatingPointError,"accumulator"):
            metric._update_losses_and_lengths(torch.full((1,1),2e38),torch.tensor([1]))
        self.assertTrue(torch.equal(before,metric.losses))

    def test_B15_production_rejects_before_update(self):
        for injection in ["input","target","gradient"]:
            run=self.child("bad_"+injection,injection=injection)
            self.assertEqual(run["observed_exit_code"],1)
            self.assertIn("FloatingPointError",run["error"])
            self.assertEqual(run["synthetic_optimizer_steps"],0)
            self.assertFalse(run.get("audit_containment",False))
        for value in [float("nan"),float("inf"),float("-inf")]:
            model=torch.nn.Linear(1,1);model.log=lambda *a,**k:None
            for param in model.parameters():param.grad=torch.full_like(param,value)
            for step in [0,201]:
                with self.assertRaises(FloatingPointError):
                    p.GradientNormLogger().on_after_backward(type("Trainer",(),{"global_step":step})(),model)

    def test_B15_post_step_overflow_and_previous_valid_checkpoint(self):
        for injection in ["parameter","optimizer"]:
            run=self.child("bad_"+injection,injection=injection)
            self.assertEqual(run["observed_exit_code"],1)
            self.assertEqual(run["synthetic_optimizer_steps"],1)
            self.assertIn("FloatingPointError",run["error"])
            self.assertIsNone(run["registry_records"][run["owned_run_id"]]["last"])
        late=self.child("bad_late",injection="gradient_late")
        self.assertEqual(late["synthetic_optimizer_steps"],2)
        failures=list((self.root/"owned").glob("*/failure-*.json"))
        self.assertGreaterEqual(len(failures),3)
        # The new invalid segment must not replace the valid first boundary.
        manifests=late["registry_records"]
        preserved=[manifests[late["owned_run_id"]]]
        self.assertEqual(preserved[0]["last"]["global_step"],2)
        self.assertTrue(preserved)
        for manifest in preserved:
            self.assertTrue((self.root/"owned"/manifest["run_id"]/manifest["last"]["file"]).exists())

    def test_B15_invalid_normalizer_and_validation_prediction(self):
        x={k:v.clone() for k,v in self.x.items()};x["target_scale"][0,1]=float("nan")
        with self.assertRaises(FloatingPointError):self.model(x)
        x["target_scale"][0,1]=0.
        with self.assertRaises(ValueError):self.model(x)
        target=self.y[0].clone();target[0,0]=float("nan")
        with patch.object(p.ClinicalTFT,"current_stage",new_callable=PropertyMock,return_value="val"):
            with self.assertRaises(FloatingPointError):self.model.step(self.x,(target,None),0)

    def test_B16_provenance_and_no_automatic_test_evaluation(self):
        full=self.good("full")
        for key in ["canonical_stage_a_sha256","dataset_sha256","schema","normalizer","architecture","budget","code_sha256","stack","data_order"]:
            self.assertIn(key,full["contract"])
        self.assertNotEqual(full["contract"]["dataset_sha256"],full["contract"]["canonical_stage_a_sha256"])
        # Main remains guarded; inspect the certified entry to ensure no evaluate call.
        import ast
        tree=ast.parse(Path(p.__file__).read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        calls=[n.func.id for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertNotIn("evaluate",calls);self.assertNotIn("load_test_data",calls)
        with patch.object(sys,"argv",["train"]):
            args=p.parse_args()
        self.assertTrue(args.no_swa);self.assertEqual(args.mode,"fresh")


def load_tests(loader, tests, pattern):
    # Historical is imported only to reuse unchanged assertions, not to execute
    # its legacy orchestration or deferred SWA paths as v1.0 acceptance.
    return loader.loadTestsFromTestCase(Remediation)


if __name__=="__main__":unittest.main(verbosity=2)
