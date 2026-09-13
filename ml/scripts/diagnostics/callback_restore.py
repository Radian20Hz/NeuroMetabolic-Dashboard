"""Read-only instrumentation of native callback restore; no training-state writes."""
from contextlib import contextmanager
from copy import deepcopy
import inspect
from unittest.mock import patch
import numpy as np
import torch
from ml.scripts.baseline_training import BoundaryCheckpoint


def same(a,b):
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
        return a.dtype==b.dtype and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,np.ndarray) and isinstance(b,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if type(a) is not type(b):return False
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


@contextmanager
def observe_restore():
    calls=[]
    original=BoundaryCheckpoint.load_state_dict
    def observed(callback,state):
        functions=[frame.function for frame in inspect.stack()]
        original(callback,state)
        calls.append(dict(instance_id=id(callback),state_key=callback.state_key,
            native_dispatch="_call_callbacks_load_state_dict" in functions and "restore_callbacks" in functions,
            fields={key:same(callback.state_dict().get(key),value) for key,value in state.items()}))
    with patch.object(BoundaryCheckpoint,"load_state_dict",observed):yield calls


def snapshot(trainer,checkpoint,calls):
    """Called after native fit restore and RNG restoration, before first forward."""
    actual={c.state_key:c for c in trainer.callbacks if c.state_dict()}
    expected=checkpoint["callbacks"]
    fields={key:{field:same(actual[key].state_dict().get(field),value) for field,value in saved.items()}
            for key,saved in expected.items() if key in actual}
    checkpoint_callbacks=[c for c in trainer.callbacks if isinstance(c,BoundaryCheckpoint)]
    matched=len(checkpoint_callbacks)==1 and len(calls)==1 and id(checkpoint_callbacks[0])==calls[0]["instance_id"]
    config={k:getattr(checkpoint_callbacks[0],k) for k in
            ("dirpath","monitor","mode","save_top_k","save_last","_save_on_train_epoch_end")}
    result=dict(state_keys_equal=actual.keys()==expected.keys(),fields=fields,
                same_restored_instance=matched,native_dispatch=bool(calls) and all(c["native_dispatch"] for c in calls),
                restore_calls=deepcopy(calls),checkpoint_config=config)
    result["pass"]=result["state_keys_equal"] and matched and result["native_dispatch"] and all(all(v.values()) for v in fields.values())
    return result
