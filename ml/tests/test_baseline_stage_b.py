"""Synthetic behavioral contracts. Known violations deliberately FAIL, never xfail.

Use the bounded runner for complete manifests and a 600-second suite timeout:
.venv/bin/python -m ml.scripts.diagnostics.audit_stage_b
"""
import copy
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, PropertyMock
import warnings

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from ml.scripts import train_tft_population_v2 as p
from ml.scripts.diagnostics import audit_stage_b as audit


def oracle(pred, target, qs):
    """Independent scalar loops, mg/dL, PF factor 2, average Q, target weights."""
    return torch.tensor([[sum(2 * max(q * (float(target[b, t])-float(pred[b, t, k])),
                                      (q-1) * (float(target[b, t])-float(pred[b, t, k])))
                              for k, q in enumerate(qs)) / len(qs) *
                          (2.5 if float(target[b, t]) < 70 else 1.0)
                          for t in range(target.size(1))] for b in range(target.size(0))],
                        dtype=pred.dtype)


def opt_config(total=8, epochs=4):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    dummy = SimpleNamespace(parameters=lambda: [parameter], hparams=SimpleNamespace(learning_rate=3e-4),
                            trainer=SimpleNamespace(estimated_stepping_batches=total, max_epochs=epochs),
                            COSINE_EPOCHS=p.ClinicalTFT.COSINE_EPOCHS)
    result = p.ClinicalTFT.configure_optimizers(dummy)
    return parameter, result["optimizer"], result["lr_scheduler"]["scheduler"]


def nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(nested_equal(x, y) for x, y in zip(a, b))
    return a == b


class StageB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger().setLevel(logging.ERROR)
        torch.set_num_threads(1)
        pl.seed_everything(42, workers=True)
        cls.tmp = tempfile.TemporaryDirectory(prefix="nmd-stage-b-")
        cls.root = Path(os.environ.get("NMD_STAGE_B_RUN", cls.tmp.name))
        cls.guard = audit.guards()
        cls.training, cls.validation, cls.meta = audit.fixture(cls.root / "unit_fixture")
        cls.model = p.build_model(cls.training, audit.arguments(dropout=0), None)
        cls.model.eval()
        cls.x, cls.y = next(iter(cls.validation.to_dataloader(train=False, batch_size=2, num_workers=0)))
        cls.children = {}
        cls.counts = {"scalar_scheduler": 0, "unit_tft": 0}
        audit.dump(cls.root / "fixture.json", cls.meta)

    @classmethod
    def tearDownClass(cls):
        audit.dump(cls.root / "unit_counts.json", cls.counts)
        cls.guard.close()
        cls.tmp.cleanup()

    @classmethod
    def child(cls, name, **options):
        if name not in cls.children:
            path = cls.root / name
            command = [sys.executable, "-m", "ml.scripts.diagnostics.audit_stage_b", "--worker", str(path)]
            for key, value in options.items():
                if value is True:
                    command.append("--" + key.replace("_", "-"))
                elif value is not False and value is not None:
                    command.extend(["--" + key.replace("_", "-"), str(value)])
            with (cls.root / f"{name}.log").open("w") as log:
                result = subprocess.run(command, cwd=audit.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=75, env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["observed_subprocess_exit_code"] = result.returncode
            cls.children[name] = manifest
        return cls.children[name]

    def good_child(self, name, **options):
        result = self.child(name, **options)
        self.assertEqual(result["observed_subprocess_exit_code"], 0, result.get("traceback"))
        self.assertLessEqual(result["synthetic_optimizer_steps"], 8)
        return result

    def state(self, name):
        return torch.load(self.root / name / "state.pt", weights_only=False, map_location="cpu")

    def test_B00_forbidden_entry_guards_positive_control(self):
        for fn in [p.main, p.load_test_data, p.evaluate]:
            with self.assertRaisesRegex(RuntimeError, "isolation guard"):
                fn()
        import optuna
        with self.assertRaisesRegex(RuntimeError, "isolation guard"):
            optuna.create_study()

    def test_B01_real_batch_contract_and_short_decoder_mask(self):
        x, y = self.x, self.y
        self.assertEqual(y[0].shape, (2, 12))
        self.assertEqual(x["encoder_cont"].shape[:2], (2, 48))
        self.assertEqual(x["decoder_cont"].dtype, torch.float32)
        self.assertEqual(x["decoder_lengths"].dtype, torch.int64)
        self.assertTrue(torch.all(x["decoder_time_idx"].diff(dim=1) == 1))
        self.assertTrue(torch.isfinite(x["target_scale"]).all())
        lengths = torch.tensor([12, 3])
        target = y[0].clone()
        pred = target[..., None].expand(-1, -1, 7).clone() + 10
        pred[1, 3:] = 99999
        packed = pack_padded_sequence(target, lengths, batch_first=True, enforce_sorted=False)
        loss = p.ClinicalQuantileLoss(quantiles=p.QUANTILES)(pred, (packed, None))
        ref = oracle(pred, target, p.QUANTILES)
        expected = (ref[0].sum()+ref[1, :3].sum()) / 15
        torch.testing.assert_close(loss, expected)
        self.assertTrue((self.training.decoded_index.time_idx_last >= self.training.decoded_index.time_idx_first_prediction).all())

    def test_B02_independent_pinball_threshold_oracle(self):
        target = torch.tensor([[69.9, 70., 70.1]], dtype=torch.float64)
        qs = [.1, .5, .9]
        for offset in [-10., 0., 10.]:
            pred = target[..., None].expand(-1, -1, 3) + offset
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                actual = p.ClinicalQuantileLoss(quantiles=qs).loss(pred, target)
            torch.testing.assert_close(actual, oracle(pred, target, qs), atol=1e-10, rtol=1e-8)

    def test_B03_transform_output_once_in_mgdl_and_reset(self):
        raw = []
        handle = self.model.output_layer.register_forward_hook(lambda m, a, output: raw.append(output.detach().clone()))
        with torch.no_grad():
            prediction = self.model(self.x)["prediction"]
        handle.remove()
        scale = self.x["target_scale"]
        # Fitted GroupNormalizer uses log transform and exp inverse (no second inverse).
        expected = torch.exp(raw[0]*scale[:, None, 1:2]+scale[:, None, 0:1])
        torch.testing.assert_close(prediction, expected)
        altered = {k: v.clone() for k, v in self.x.items()}
        altered["target_scale"][:, 0] += math.log(2)
        with torch.no_grad():
            doubled = self.model(altered)["prediction"]
        torch.testing.assert_close(doubled, prediction*2)
        metric = p.ClinicalQuantileLoss(quantiles=p.QUANTILES)
        observed = metric(prediction, self.y)
        torch.testing.assert_close(observed, oracle(prediction, self.y[0], p.QUANTILES).mean())
        metric.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zero = metric(self.y[0][..., None].expand(-1, -1, 7), self.y)
        self.assertEqual(float(zero), 0)

    def test_B03_epoch_reduction_unequal_lengths_contract(self):
        # Execute BaseModel.step and capture its actual logging denominator.
        # Fixed predictions isolate aggregation from neural numerical effects.
        batches, values, weights, sums, positions = [], [], [], [], []
        for size, length, error in [(2, 12, 10.), (1, 3, 30.)]:
            x = {k: v[:size].clone() for k, v in self.x.items()}
            x["decoder_lengths"][:] = length
            x["decoder_target"] = x["decoder_target"][:, :length]
            target = self.y[0][:size, :length]
            pred = target[..., None].expand(-1, -1, 7)+error
            with patch.object(self.model, "forward", return_value={"prediction": pred}), \
                 patch.object(self.model, "log") as logged, \
                 patch.object(p.ClinicalTFT, "current_stage", new_callable=PropertyMock, return_value="val"):
                result, _ = self.model.step(x, (target, None), 0)
            values.append(float(result["loss"]))
            weights.append(logged.call_args.kwargs["batch_size"])
            reference = oracle(pred, target, p.QUANTILES)
            sums.append(float(reference.sum())); positions.append(target.numel())
        reported = sum(v*w for v, w in zip(values, weights))/sum(weights)
        reference = sum(sums)/sum(positions)
        audit.dump(self.root / "aggregation.json", dict(batch_losses=values, logged_weights=weights,
                   reported=reported, valid_position_reference=reference))
        self.assertAlmostEqual(reported, reference, places=5,
                               msg="B03-F01 epoch weights samples rather than valid decoder positions")

    def test_B04_quantile_axis_median_and_crossing_detection(self):
        qs = p.QUANTILES
        self.assertEqual(qs, sorted(set(qs)))
        self.assertTrue(all(0 < q < 1 for q in qs))
        self.assertEqual(self.model.hparams.output_size, len(qs))
        pred = torch.arange(7.).reshape(1, 1, 7)
        self.assertEqual(float(self.model.loss.to_prediction(pred)), qs.index(.5))
        crossed = pred.flip(-1)
        self.assertTrue(torch.any(crossed.diff(dim=-1) < 0))
        self.assertTrue(torch.equal(self.model.loss.to_quantiles(crossed), crossed))

    def test_B05_real_forward_future_unknown_invariance_positive_control(self):
        model = self.model
        x = {k: v.clone() for k, v in self.x.items()}
        with torch.no_grad():
            original = model(x)["prediction"]
            forbidden_indices = [i for i, name in enumerate(model.hparams.x_reals)
                                 if name not in model.decoder_variables]
            self.assertTrue(forbidden_indices)
            x["decoder_cont"][:, :, forbidden_indices] += 1000
            x["decoder_target"] += 1000
            changed = model(x)["prediction"]
            torch.testing.assert_close(original, changed, atol=0, rtol=0)
            x["encoder_cont"] += 1
            positive = model(x)["prediction"]
        self.assertGreater(float((positive-original).abs().max()), 1e-5)
        model.train()
        prediction = model(self.x)["prediction"]
        self.assertTrue(torch.isfinite(model.loss(prediction, self.y)))
        model.eval()

    def test_B06_analytic_gradient_gradcheck_real_backward(self):
        target = torch.tensor([[60., 100.]], dtype=torch.float64)
        qs = [.1, .5, .9]
        pred = (target[..., None].expand(-1, -1, 3)+3).clone().requires_grad_(True)
        metric = p.ClinicalQuantileLoss(quantiles=qs)
        self.assertTrue(torch.autograd.gradcheck(lambda z: metric.loss(z, target), (pred,), eps=1e-6, atol=1e-8))
        metric.loss(pred, target).sum().backward()
        expected = torch.tensor([[[2*(1-q)/3*2.5 for q in qs], [2*(1-q)/3 for q in qs]]], dtype=torch.float64)
        torch.testing.assert_close(pred.grad, expected, atol=1e-10, rtol=1e-8)
        model = p.build_model(self.training, audit.arguments(dropout=0), None).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        before = copy.deepcopy(model.state_dict())
        loss = model.loss(model(self.x)["prediction"], self.y)
        loss.backward()
        gradients = [v.grad for v in model.parameters() if v.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(torch.count_nonzero(g) for g in gradients))
        optimizer.step(); self.counts["unit_tft"] += 1
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(all(v.grad is None for v in model.parameters()))

    def test_B07_optimizer_clip_accumulation(self):
        run = self.good_child("accumulation", accumulation=2)
        self.assertEqual(run["synthetic_optimizer_steps"], 8)
        self.assertEqual(run["estimated_stepping_batches"], 8)
        optim = [e for e in run["trace"] if e["event"] == "optimizer"]
        backward = [e for e in run["trace"] if e["event"] == "backward"]
        self.assertEqual(len(backward), 16)
        self.assertTrue(all(e["norm"] <= 1.00001 for e in optim))
        self.assertTrue(any(e["norm"] > 1 for e in backward))
        self.assertEqual(optim[0]["lr"], 0)
        self.assertGreater(backward[0]["norm"], 0)
        state = self.state("accumulation")
        group = state["optimizer"]["param_groups"][0]
        self.assertEqual(group["eps"], 1e-7); self.assertEqual(group["weight_decay"], .01)
        self.assertEqual(len(group["params"]), len(set(group["params"])))
        self.assertEqual(len(group["params"]), len(list(self.model.parameters())))
        self.assertEqual(state["scheduler"]["last_epoch"], 8)

    def test_B08_scheduler_boundaries_restart_and_state_resume(self):
        param, optimizer, scheduler = opt_config(total=160, epochs=160)
        lrs = [optimizer.param_groups[0]["lr"]]
        for step in range(160):
            param.grad = torch.ones_like(param)
            optimizer.step(); self.counts["scalar_scheduler"] += 1
            scheduler.step(); lrs.append(optimizer.param_groups[0]["lr"])
            if step == 73:
                snapshot = copy.deepcopy((optimizer.state_dict(), scheduler.state_dict()))
        self.assertEqual(lrs[0], 0)
        self.assertAlmostEqual(lrs[49], 3e-4*49/50)
        self.assertAlmostEqual(lrs[50], 3e-4)
        self.assertLess(lrs[51], lrs[50])
        self.assertAlmostEqual(lrs[65], 3e-4)
        _, op2, sc2 = opt_config(total=160, epochs=160)
        op2.load_state_dict(snapshot[0]); sc2.load_state_dict(snapshot[1])
        self.assertEqual(sc2.state_dict(), snapshot[1])
        self.assertAlmostEqual(op2.param_groups[0]["lr"], lrs[74])
        audit.dump(self.root / "scheduler.json", dict(lrs=lrs, warmup=50, first_restart=65))

    def test_B08_invalid_epoch_budget_rejected_explicitly(self):
        outcomes = {}
        for epochs in [0, -1]:
            try:
                opt_config(total=8, epochs=epochs)
                outcomes[epochs] = "accepted"
            except (ZeroDivisionError, ValueError, RuntimeError) as exc:
                outcomes[epochs] = type(exc).__name__
        audit.dump(self.root / "invalid_epochs.json", outcomes)
        # Lightning permits max_epochs=-1 with finite max_steps; record its fallback,
        # but require an explicit configuration error only for zero epochs.
        self.assertIn(outcomes[0], ("ValueError", "RuntimeError"),
                      f"B08-F01 zero-epoch budget lacks explicit validation: {outcomes}")

    def test_B09_checkpoint_roundtrip(self):
        run = self.good_child("continuous")
        checkpoint = run["terminal_checkpoint"]
        loaded = p.ClinicalTFT.load_from_checkpoint(checkpoint, map_location="cpu").eval()
        state = self.state("continuous")
        self.assertTrue(nested_equal(loaded.state_dict(), state["weights"]))
        with torch.no_grad():
            prediction = loaded(self.x)["prediction"]
        torch.testing.assert_close(prediction, state["prediction"], atol=0, rtol=0)
        self.assertEqual(loaded.loss.quantiles, p.QUANTILES)
        payload = torch.load(checkpoint, weights_only=False)
        for key in ["hyper_parameters", "optimizer_states", "lr_schedulers", "global_step", "epoch", "callbacks"]:
            self.assertIn(key, payload)
        self.assertEqual(payload["global_step"], 8)
        self.assertIn("dataset_parameters", payload)
        broken = self.root / "broken.ckpt"
        broken.write_bytes(b"corrupted synthetic checkpoint")
        with self.assertRaises(Exception):
            p.ClinicalTFT.load_from_checkpoint(broken, map_location="cpu")

    def test_B10_production_resume_loads_own_checkpoint(self):
        partial = self.good_child("partial", stop=True)
        self.good_child("resumed_production", parent=partial["last_checkpoint"])

    def test_B10_epoch_resume_no_dropout_no_shuffle(self):
        continuous = self.good_child("continuous")
        partial = self.good_child("partial", stop=True)
        resumed = self.good_child("resumed_trusted", parent=partial["terminal_checkpoint"], trusted_resume=True)
        self.assertEqual(partial["synthetic_optimizer_steps"], 4)
        self.assertEqual(resumed["synthetic_optimizer_steps"], 4)
        self.assertEqual(resumed["estimated_stepping_batches"], 8)
        a, b = self.state("continuous"), self.state("resumed_trusted")
        diff = max(float((a["weights"][k]-b["weights"][k]).abs().max()) for k in a["weights"])
        audit.dump(self.root / "resume_deterministic.json", dict(max_abs_weight_difference=diff,
            optimizer_equal=nested_equal(a["optimizer"], b["optimizer"]), scheduler_equal=nested_equal(a["scheduler"], b["scheduler"])))
        self.assertTrue(nested_equal(a, b), f"B10 deterministic replay discrepancy max={diff}")
        self.assertEqual(continuous["global_step"], resumed["global_step"])
        for event in ["optimizer", "loss"]:
            self.assertEqual([e for e in continuous["trace"] if e["event"] == event][4:],
                             [e for e in resumed["trace"] if e["event"] == event])

    def test_B10_epoch_resume_dropout_shuffle_and_rng(self):
        full = self.good_child("stochastic", stochastic=True)
        partial = self.good_child("stochastic_partial", stochastic=True, stop=True)
        resumed = self.good_child("stochastic_resumed_trusted", stochastic=True, parent=partial["terminal_checkpoint"], trusted_resume=True)
        a, b = self.state("stochastic"), self.state("stochastic_resumed_trusted")
        diff = max(float((a["weights"][k]-b["weights"][k]).abs().max()) for k in a["weights"])
        audit.dump(self.root / "resume_stochastic.json", dict(max_abs_weight_difference=diff,
            optimizer_equal=nested_equal(a["optimizer"], b["optimizer"]), scheduler_equal=nested_equal(a["scheduler"], b["scheduler"]),
            full_rng=full["final_rng"], resumed_rng=resumed["final_rng"],
            full_batches=[e for e in full["trace"] if e["event"] == "batch"],
            resumed_batches=[e for e in resumed["trace"] if e["event"] == "batch"]))
        for k in a["weights"]:
            torch.testing.assert_close(a["weights"][k], b["weights"][k], atol=1e-6, rtol=1e-5,
                                       msg="B10-F01 dropout/shuffle epoch resume does not replay weights")
        self.assertEqual(full["final_rng"], resumed["final_rng"])

    def test_B10_last_checkpoint_epoch_progress(self):
        partial = self.good_child("partial", stop=True)
        resumed = self.good_child("resumed_last_diagnostic", parent=partial["last_checkpoint"], trusted_resume=True)
        self.assertEqual(resumed["synthetic_optimizer_steps"], 4,
                         "B10-F04 last checkpoint loses remaining epoch progress")
        self.assertEqual(resumed["global_step"], 8)

    def test_B10_gradient_callback_state_roundtrip(self):
        callback = p.GradientNormLogger()
        callback._high_grad_consecutive = 12
        restored = p.GradientNormLogger()
        restored.load_state_dict(callback.state_dict())
        self.assertEqual(restored._high_grad_consecutive, 12,
                         "B10-F03 gradient alert history absent from callback checkpoint state")

    def test_B11_incompatible_dataset_is_rejected(self):
        run = self.good_child("continuous")
        incompatible = copy.deepcopy(self.training)
        incompatible.max_prediction_length = 6
        incompatible.max_encoder_length = 24
        incompatible.target_normalizer.norm_.iloc[:, :] += 1
        with self.assertRaisesRegex((ValueError, RuntimeError), "incompat|schema|dataset|horizon|context"):
            p.build_model(incompatible, audit.arguments(), run["terminal_checkpoint"])

    def test_B11_checkpoint_selection_does_not_trust_filename(self):
        directory = self.root / "selection"
        directory.mkdir(exist_ok=True)
        corrupt = directory / "tft-stage-a-v1-epoch=99-val_loss=0.0001.ckpt"
        corrupt.write_bytes(b"not a checkpoint")
        with patch.object(p, "MODEL_DIR", directory):
            selected = p.find_best_checkpoint()
        self.assertIsNone(selected, "B11-F02 corrupt checkpoint selected by filename alone")

    def test_B12_early_stopping_sequences_and_resume(self):
        def trainer(value=1., sanity=False):
            return SimpleNamespace(callback_metrics={"val_loss": torch.tensor(value)}, fast_dev_run=False,
                current_epoch=0, should_stop=False, sanity_checking=sanity,
                state=SimpleNamespace(fn=pl.trainer.states.TrainerFn.FITTING),
                strategy=SimpleNamespace(reduce_boolean_decision=lambda x, **kw: x), world_size=1, global_rank=0)
        es = EarlyStopping("val_loss", mode="min", patience=2, min_delta=1e-4, check_on_train_epoch_end=False)
        t = trainer(10, sanity=True)
        es.on_validation_end(t, None)
        self.assertEqual(es.wait_count, 0); self.assertTrue(torch.isinf(es.best_score))
        for value, wait in [(10., 0), (9., 0), (9.-1e-4, 1), (10., 2)]:
            t = trainer(value); es.on_validation_end(t, None)
            self.assertEqual(es.wait_count, wait)
        self.assertTrue(t.should_stop)
        restored = EarlyStopping("val_loss", mode="min", patience=2)
        restored.load_state_dict(es.state_dict())
        self.assertTrue(nested_equal(es.state_dict(), restored.state_dict()))
        for value in [float("nan"), float("inf")]:
            fresh = EarlyStopping("val_loss", check_on_train_epoch_end=False)
            t = trainer(value); fresh.on_validation_end(t, None)
            self.assertTrue(t.should_stop)
        t = trainer(); t.callback_metrics = {"test_loss": torch.tensor(0.), "train_loss": torch.tensor(0.)}
        with self.assertRaisesRegex(RuntimeError, "val_loss"):
            es.on_validation_end(t, None)

    def test_B13_swa_weights_and_checkpoint_semantics(self):
        full = self.good_child("swa", swa=True)
        terminal = torch.load(full["terminal_checkpoint"], weights_only=False)
        last = torch.load(full["last_checkpoint"], weights_only=False)
        best = torch.load(full["best_checkpoint"], weights_only=False)
        state = next(v for k, v in terminal["callbacks"].items() if "StochasticWeightAveraging" in k)
        # Compare parameters only: buffers/metric states are not SWA averaged.
        names = dict(self.model.named_parameters())
        equal_average = all(torch.equal(terminal["state_dict"][k], state["average_model_state"][k]) for k in names)
        evidence = dict(n_averaged=int(state["n_averaged"]), final_equals_average=equal_average,
            last_equals_terminal=nested_equal(last["state_dict"], terminal["state_dict"]),
            best_equals_terminal=nested_equal(best["state_dict"], terminal["state_dict"]),
            scheduler_events=[e for e in full["trace"] if e["event"] == "epoch"])
        audit.dump(self.root / "swa_semantics.json", evidence)
        self.assertTrue(equal_average)
        self.assertEqual(evidence["n_averaged"], 2)
        # Last is a continuation state; it need not equal averaged inference weights.
        # Require its full optimizer state, and record (do not silently conflate) roles.
        self.assertTrue(last["optimizer_states"])
        self.assertEqual(last["global_step"], 8)

    def test_B13_early_stop_can_precede_swa(self):
        # Cheap callback simulation, no optimizer steps and no real-data decision.
        es = EarlyStopping("val_loss", mode="min", patience=25, check_on_train_epoch_end=False)
        swa = p.StochasticWeightAveraging(swa_epoch_start=int(60*.75), swa_lrs=1.5e-5)
        for epoch in range(26):
            trainer = SimpleNamespace(callback_metrics={"val_loss": torch.tensor(10.)}, fast_dev_run=False,
                current_epoch=epoch, should_stop=False, sanity_checking=False,
                state=SimpleNamespace(fn=pl.trainer.states.TrainerFn.FITTING),
                strategy=SimpleNamespace(reduce_boolean_decision=lambda x, **kw: x), world_size=1, global_rank=0)
            es.on_validation_end(trainer, None)
        self.assertTrue(trainer.should_stop)
        self.assertLess(epoch, swa.swa_start)

    def test_B13_default_swa_scheduler_averaging_resume(self):
        full = self.good_child("swa", swa=True)
        partial = self.good_child("swa_partial", swa=True, stop=True)
        resumed = self.good_child("swa_resumed_trusted", swa=True, parent=partial["terminal_checkpoint"], trusted_resume=True)
        epochs = [e for e in full["trace"] if e["event"] == "epoch"]
        self.assertEqual(epochs[0]["scheduler"], "SequentialLR")
        self.assertEqual(epochs[-1]["scheduler"], "SWALR")
        self.assertEqual(epochs[-1]["interval"], "epoch")
        self.assertTrue(nested_equal(self.state("swa"), self.state("swa_resumed_trusted")))
        payload = torch.load(full["terminal_checkpoint"], weights_only=False)
        swa = next(v for k, v in payload["callbacks"].items() if "StochasticWeightAveraging" in k)
        self.assertGreater(int(swa["n_averaged"]), 0)
        self.assertIsNotNone(swa["average_model_state"])
        # Last retains the optimizer continuation state; averaged terminal is a
        # distinct inference artifact. Equality of their weights is not required.
        last = torch.load(full["last_checkpoint"], weights_only=False)
        self.assertTrue(last["optimizer_states"])

    def test_B14_new_process_cpu_determinism_seed_control(self):
        a = self.good_child("stochastic", stochastic=True)
        b = self.good_child("repeat", stochastic=True)
        c = self.good_child("seed43", stochastic=True, seed=43)
        self.assertEqual(a["initial_weights_sha256"], b["initial_weights_sha256"])
        self.assertEqual(a["final_weights_sha256"], b["final_weights_sha256"])
        self.assertEqual(a["trace"], b["trace"])
        self.assertNotEqual(a["initial_weights_sha256"], c["initial_weights_sha256"])
        self.assertNotEqual(a["final_weights_sha256"], c["final_weights_sha256"])

    def test_B15_large_finite_and_constant_zero_loss(self):
        target = torch.full((1, 1), 100.)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = p.ClinicalQuantileLoss(quantiles=p.QUANTILES)
            self.assertEqual(float(loss(target[..., None].expand(-1, -1, 7), target)), 0)
            large = loss(torch.full((1, 1, 7), 1e8), target)
        self.assertTrue(torch.isfinite(large)); self.assertGreater(float(large), 1e7)

    def test_B15_nonfinite_loss_must_raise_before_update(self):
        for value in [float("nan"), float("inf")]:
            with self.subTest(value=value), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metric = p.ClinicalQuantileLoss(quantiles=p.QUANTILES)
                with self.assertRaises((ValueError, RuntimeError, FloatingPointError)):
                    metric(torch.full((1, 1, 7), value), torch.tensor([[100.]]))

    def test_B15_nonfinite_gradient_callback_must_abort(self):
        model = torch.nn.Linear(1, 1)
        model.log = lambda *a, **k: None
        for param in model.parameters():
            param.grad = torch.full_like(param, float("nan"))
        callback = p.GradientNormLogger()
        with self.assertRaises((ValueError, RuntimeError, FloatingPointError)):
            callback.on_after_backward(SimpleNamespace(global_step=201), model)

    def test_B15_real_training_nonfinite_input_target_gradient(self):
        outcomes = {}
        for kind in ["input", "target", "gradient"]:
            run = self.child("nonfinite_" + kind, nonfinite=kind)
            outcomes[kind] = dict(exit_code=run["exit_code"], updates=run["synthetic_optimizer_steps"],
                                  checkpoints=run["checkpoints"], error=run.get("error"),
                                  final_weights_finite=run.get("final_weights_finite"),
                                  audit_guard_intercepted=run.get("audit_guard_intercepted_invalid_gradient", False))
        audit.dump(self.root / "nonfinite.json", outcomes)
        self.assertTrue(all(r["exit_code"] != 0 and r["updates"] == 0 and not r["checkpoints"] and not r["audit_guard_intercepted"]
                            for r in outcomes.values()),
                        f"B15-F01 invalid state was not stopped before update/checkpoint: {outcomes}")

    def test_B13_resume_after_swa_activation(self):
        full = self.good_child("swa", swa=True)
        candidates = list((self.root / "swa").glob("*-epoch=02-*.ckpt"))
        self.assertEqual(len(candidates), 1)
        resumed = self.good_child("swa_after_activation", swa=True, parent=str(candidates[0]), trusted_resume=True)
        self.assertEqual(resumed["synthetic_optimizer_steps"], 2)
        self.assertTrue(nested_equal(self.state("swa"), self.state("swa_after_activation")),
                        "B13 resume after SWA activation changed final state")

    def test_B07_frozen_parameter_and_callback_order(self):
        model = p.build_model(self.training, audit.arguments(dropout=0), None)
        frozen = next(model.parameters()); frozen.requires_grad_(False)
        model._trainer = SimpleNamespace(estimated_stepping_batches=8, max_epochs=4)
        result = model.configure_optimizers()
        optimizer = result["optimizer"]
        included = [v for group in optimizer.param_groups for v in group["params"]]
        self.assertEqual({id(v) for v in included}, {id(v) for v in model.parameters()})
        # AdamW includes frozen parameters but skips them when grad is None.
        self.assertIsNone(frozen.grad)
        self.assertFalse(frozen.requires_grad)
        run = self.good_child("accumulation", accumulation=2)
        events = [e["event"] for e in run["trace"]]
        index = events.index("optimizer")
        self.assertEqual(events[index-1], "before_clip")
        self.assertIn("backward", events[:index])

    def test_B08_estimated_steps_incomplete_batch_and_max_steps(self):
        from torch.utils.data import DataLoader, TensorDataset
        class Tiny(pl.LightningModule):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
            def training_step(self, batch, batch_idx):
                return self.weight.sum()
            def train_dataloader(self):
                return DataLoader(TensorDataset(torch.arange(5.)), batch_size=2, drop_last=False)
        trainer = pl.Trainer(max_epochs=4, max_steps=8, accumulate_grad_batches=2,
            logger=False, enable_checkpointing=False, enable_progress_bar=False, accelerator="cpu")
        model = Tiny()
        trainer.strategy.connect(model)
        trainer._data_connector.attach_data(model)
        self.assertEqual(trainer.estimated_stepping_batches, 8)  # ceil(ceil(5/2)/2)*4 capped8
        self.assertEqual(len(trainer.train_dataloader), 3)

    def test_B16_artifact_provenance(self):
        run = self.good_child("continuous")
        for key in ["run_id", "git_commit", "git_dirty", "diff_sha256", "config_sha256", "seed", "versions",
                    "device", "precision", "schema", "quantiles", "normalizer", "context", "horizon",
                    "parent_checkpoint", "parent_run", "resume_mode", "command", "exit_code", "fixture_sha256"]:
            self.assertIn(key, run)
        self.assertEqual(run["canonical_stage_a_sha256"], audit.config()["canonical_parquet_sha256"])
        for key in ["full_training", "optuna", "test_performance"]:
            self.assertFalse(run[key])
        self.assertTrue(run["checkpoints"])
        self.assertEqual(run["synthetic_optimizer_steps"], 8)
        audit.dump(self.root / "environment.json", dict(cuda_available=torch.cuda.is_available(),
            cuda_exercised=False, amp_exercised=False, workers_exercised=[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
