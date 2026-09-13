"""Focused numerical-profile and frozen comparator controls; no GPU fits."""
import copy
import unittest
from unittest.mock import patch
import torch
import lightning.pytorch as pl
from ml.scripts import numerical_profile as n
from ml.scripts.checkpoint_registry import check_compatible
from ml.scripts.diagnostics.finalize_stage_b import differences,TOLERANCES


class Finalization(unittest.TestCase):
    def tearDown(self):n.configure("cpu")

    def test_distinct_effective_profiles_after_trainer(self):
        for device,flag in [("cpu",True),("cuda","warn")]:
            n.configure(device)
            # No fit and no GPU request: verify public Trainer flag setup on CPU.
            pl.Trainer(accelerator="cpu",logger=False,enable_checkpointing=False,
                       deterministic=flag,benchmark=False)
            self.assertEqual(n.verify(device)["warn_only"],device=="cuda")
            self.assertEqual(n.effective()["matmul_precision"],"highest")
        self.assertNotEqual(n.profile("cpu"),n.profile("cuda"))

    def test_profile_mismatch_refused_before_loading(self):
        expected={"numerical_profile":n.profile("cuda")}
        bad=copy.deepcopy(expected);bad["numerical_profile"]=n.profile("cpu")
        with patch("torch.load") as load:
            with self.assertRaisesRegex(ValueError,"numerical_profile"):
                check_compatible(bad,expected)
            load.assert_not_called()
        check_compatible(expected,expected)
        n.configure("cpu");torch.use_deterministic_algorithms(True,warn_only=True)
        with self.assertRaisesRegex(ValueError,"warn_only"):n.verify("cpu")

    def test_checkpoint_current_score_restored_at_boundary(self):
        # Diagnostic reproducer added after the frozen GPU trial found a mismatch.
        # No fit, no threshold change and no claim that the original trial passed.
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from ml.scripts.baseline_training import BoundaryCheckpoint
        with tempfile.TemporaryDirectory() as directory:
            registry=SimpleNamespace(root=Path(directory))
            before=BoundaryCheckpoint(registry,"same-run",{})
            before.current_score=torch.tensor(1.)
            before.best_model_score=torch.tensor(.5)
            before.best_model_path=str(Path(directory)/"same-run/best.ckpt")
            before.best_k_models={before.best_model_path:torch.tensor(.5)}
            before.kth_best_model_path=before.best_model_path
            before.kth_value=torch.tensor(.5)
            saved=before.state_dict()
            after=BoundaryCheckpoint(registry,"same-run",{})
            original=pl.callbacks.ModelCheckpoint.load_state_dict
            with patch.object(pl.callbacks.ModelCheckpoint,"load_state_dict",autospec=True,side_effect=original) as native:
                after.load_state_dict(saved)
                native.assert_called_once_with(after,saved)
            self.assertIsNotNone(after.current_score,"Serialized current_score was not restored")
            torch.testing.assert_close(after.current_score,saved["current_score"],rtol=0,atol=0)
            from ml.scripts.diagnostics.callback_restore import same
            for field,value in saved.items():
                self.assertTrue(same(after.state_dict()[field],value),field)

    def test_callback_restore_rejects_incompatible_or_missing_state(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from ml.scripts.baseline_training import BoundaryCheckpoint
        with tempfile.TemporaryDirectory() as directory:
            callback=BoundaryCheckpoint(SimpleNamespace(root=Path(directory)),"same-run",{})
            saved=callback.state_dict()
            for field in ("dirpath","monitor","current_score"):
                bad=copy.deepcopy(saved)
                if field=="current_score":del bad[field]
                else:bad[field]="incompatible"
                with self.assertRaises(ValueError):callback.load_state_dict(bad)

    def test_boundary_tensor_container_canonicalization(self):
        from collections import OrderedDict
        from ml.scripts.diagnostics.finalize_stage_b import cpu,equal
        original=OrderedDict(weight=torch.tensor([1.,2.]))
        captured=cpu(original)
        self.assertTrue(equal(captured,cpu(original)))
        changed=OrderedDict(weight=torch.tensor([1.,3.]))
        self.assertFalse(equal(captured,cpu(changed)))

    def test_symmetric_tolerances_positive_and_negative_control(self):
        tol=TOLERANCES["parameters"]
        a=torch.tensor([0.,1.],dtype=torch.float64)
        b=torch.tensor([1e-7,1.+1e-5],dtype=torch.float64)
        self.assertEqual(differences(a,b,tol)["exceedances"],0)
        c=torch.tensor([2e-6,1.],dtype=torch.float64)
        self.assertEqual(differences(a,c,tol)["exceedances"],1)
        self.assertEqual(differences(c,a,tol)["max_normalized_error"],differences(a,c,tol)["max_normalized_error"])
        self.assertEqual(differences(a,a,TOLERANCES["lr"])["max_abs"],0)
        self.assertEqual(differences(a,b,TOLERANCES["lr"])["exceedances"],2)
        with self.assertRaises(ValueError):differences(a,torch.tensor([float("nan"),1.]),tol)


if __name__=="__main__":unittest.main()
