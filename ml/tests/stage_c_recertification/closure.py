"""Narrow C02 closure: original registry read-only; no patched production methods."""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import campaign as c

ORIGINAL=c.ROOT/'ml/models/stage_c_training_recertification/20260915T143739Z'
ORDER=['serialization','CPU_N_target','CPU_N_gradient','CUDA_N_input','CUDA_N_target','CUDA_N_gradient']
LIMITS=dict(attempts=0,completions=0,forward=32,backward=2,candidates=300)

def inventory(root):
    found={}
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink(),str(p)
        found[str(p.relative_to(root))]=dict(type='directory') if p.is_dir() else dict(type='file',size=p.stat().st_size,sha256=c.sha(p))
    return found


def worker(root,name):
    spec=json.loads((root/'preregistration.json').read_text());out=root/name
    report=dict(name=name,status='ERROR',checks=[]);ledger=root/'ledger.json'
    def count(key):
        d=json.loads(ledger.read_text());d[key]+=1;c.save(ledger,d)
        if d[key]>LIMITS[key]:raise RuntimeError('containment budget '+key)
    def deny(label):
        report['containment_first']=label;raise PermissionError(label)
    def audit(event,args):
        if event=='import' and args[0].split('.')[0]=='optuna':deny('Optuna')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
            write=(isinstance(mode,str) and any(k in mode for k in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR))
            if write and not p.is_relative_to(root) and str(p)!='/dev/null':deny('write outside closure')
            if not write and p.is_relative_to(c.ROOT/'ml/data'):deny('real data')
            if not write and p.is_relative_to(c.ROOT/'ml/models') and not p.is_relative_to(root):
                rel=str(p.relative_to(ORIGINAL)) if p.is_relative_to(ORIGINAL) else None
                if rel not in spec['read_allowlist']:deny('unlisted original artifact')
        if event=='os.mkdir' and Path(os.fsdecode(args[0])).resolve()==ORIGINAL/'owned' and (ORIGINAL/'owned').is_dir():
            return  # Registry's exist_ok=True on an already existing directory; no mutation.
        if event in ('os.remove','os.rmdir','os.mkdir','os.rename','os.link','os.symlink'):
            paths=args[:2] if event in ('os.rename','os.link','os.symlink') else args[:1]
            for path in paths:
                if isinstance(path,(str,bytes,os.PathLike)) and not Path(os.fsdecode(path)).resolve().is_relative_to(root):deny('filesystem mutation outside closure')
    sys.addaudithook(audit)
    try:
        for path,h in spec['source_hashes'].items():assert c.sha(c.ROOT/path)==h,path
        if name=='serialization':
            assert c.gpu_preflight(root)==0,'GPU preflight'
        import numpy as np
        import torch
        import lightning.pytorch as pl
        import pytorch_forecasting as pf
        import pandas as pd
        from types import SimpleNamespace as NS
        sys.path.insert(0,str(c.ROOT))
        from ml.scripts import train_tft_population_v2 as p,checkpoint_registry as r,numerical_profile as n,finite_training as f
        from ml.scripts.diagnostics.callback_restore import same
        torch.set_num_threads(1);pl.seed_everything(42,workers=True)
        device='cuda' if name.startswith('CUDA') else 'cpu';n.configure(device)
        report['environment']=dict(python=sys.version,torch=torch.__version__,lightning=pl.__version__,pf=pf.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),profile=n.verify(device))
        assert (torch.__version__,pl.__version__,pf.__version__,np.__version__,pd.__version__)==('2.11.0+cu130','2.6.1','1.7.0','2.4.4','2.3.3')
        def eq(a,b,label):assert same(a,b),label
        def close(a,b):np.testing.assert_allclose(a.detach().cpu().numpy() if isinstance(a,torch.Tensor) else a,b.detach().cpu().numpy() if isinstance(b,torch.Tensor) else b,atol=1e-5,rtol=1e-5)
        base_codes={inspect.unwrap(torch.optim.AdamW.step).__code__}
        # Observe calls without replacing any method. A forbidden base entry fails before body execution.
        forward_code=f.FiniteModel.forward.__code__
        forbidden_codes={p.main.__code__,p.evaluate.__code__,p.load_and_preprocess_data.__code__,p.load_test_data.__code__}
        if name=='serialization':forbidden_codes.add(pl.Trainer.fit.__code__)
        def observe(frame,event,arg):
            if event!='call':return
            code=frame.f_code
            if code in base_codes:
                count('attempts');deny('base AdamW entry')
            if code in forbidden_codes:deny('forbidden production entrypoint')
            if code is forward_code:
                if name=='serialization':deny('serialization model forward')
                count('forward');n.verify(device)
        sys.setprofile(observe)
        fixture=torch.load(ORIGINAL/'R0/fixture.pt',map_location='cpu',weights_only=False)
        training,validation=fixture['training'],fixture['validation']
        assert len(training)==len(validation)==4
        args=NS(**c.CONFIG,no_gpu=device=='cpu',mode='fresh',run_id=None,parent_run_id=None,stop_after_epoch=None,registry_dir=root/'probe_registry',dataset_sha256=c.sha(ORIGINAL/'R0/fixture.json'))
        if name=='serialization':
            assert (ORIGINAL/'owned/owned_runs.json').is_file()
            reg=r.Registry(ORIGINAL/'owned');evidence={}
            for label in ('CPU_U','CUDA_C1'):
                saved=json.loads((ORIGINAL/label/'result.json').read_text());rid=saved['run_id']
                args.no_gpu=label=='CPU_U';n.configure('cpu' if args.no_gpu else 'cuda')
                expected=r.contract(training,args,p.QUANTILES);assert expected==saved['contract']
                path,payload,record=reg.verified(rid,'best',expected)
                manifest=reg.manifest(rid);assert manifest==saved['registry']
                values=saved['validation'];assert len(values)==4 and all(math.isfinite(v['value']) for v in values)
                oracle=min(values,key=lambda v:(v['value'],v['step']))
                assert (record['val_loss'],record['global_step'])==(oracle['value'],oracle['step'])
                assert record in manifest['generations'] and c.sha(path)==record['sha256']
                model=p.ClinicalTFT.load_from_checkpoint(path,map_location='cpu',weights_only=False)
                eq(model.state_dict(),payload['state_dict'],'model tensors')
                # Current selected BEST is final step8; compare to independently saved trajectory state.
                raw=torch.load(ORIGINAL/label/'raw.pt',map_location='cpu',weights_only=False)
                assert record['global_step']==raw['final']['global_step']
                eq(model.state_dict(),raw['final']['model'],'trajectory tensors')
                eq(r.semantic(model.dataset_parameters),r.semantic(payload['dataset_parameters']),'model dataset params')
                eq(r.semantic(payload['dataset_parameters']),r.semantic(training.get_parameters()),'fixture dataset params')
                eq(r.semantic(model.output_transformer),r.semantic(training.target_normalizer),'normalizer stats/map')
                assert list(model.loss.quantiles)==p.QUANTILES
                assert model.hparams.x_reals==training.reals and model.hparams.x_categoricals==training.flat_categoricals
                assert model.output_transformer.nmd_revision=='observed-train-encoded-subject-v2'
                rebuilt=pf.TimeSeriesDataSet.from_parameters(payload['dataset_parameters'],fixture['val_frame'],stop_randomization=True)
                d=json.loads(ledger.read_text());d['candidates']+=len(rebuilt.decoded_index);c.save(ledger,d);assert d['candidates']<=300
                rebuilt=rebuilt.filter(lambda d:(d.time_idx_first_prediction-d.time_idx_first==48)&(d.time_idx_last-d.time_idx_first_prediction==11)&d.time_idx_first_prediction.isin([48,49]))
                eq(rebuilt.decoded_index.to_dict('records'),validation.decoded_index.to_dict('records'),'window keys')
                eq(r.semantic(rebuilt.get_parameters()['categorical_encoders']),r.semantic(training.get_parameters()['categorical_encoders']),'encoder mapping')
                x,y=next(iter(rebuilt.to_dataloader(train=False,batch_size=4,num_workers=0,shuffle=False)))
                ref,_=next(iter(validation.to_dataloader(train=False,batch_size=4,num_workers=0,shuffle=False)))
                close(x['encoder_cont'],ref['encoder_cont']);close(x['target_scale'],ref['target_scale'])
                train=fixture['train_frame'];val=fixture['val_frame'];scales=[]
                for j,row in enumerate(rebuilt.x_to_index(x).itertuples(index=False)):
                    observed=train[(train.subject_id==row.subject_id)&train.target_observed].glucose_mg_dl.to_numpy(dtype=np.float64)
                    logs=np.log(observed);mu=logs.mean();sigma=logs.std(ddof=1)+np.finfo(np.float16).eps;scales.append([mu,sigma])
                    enc=val[(val.subject_id==row.subject_id)&val.time_idx.between(row.time_idx-48,row.time_idx-1)].glucose_mg_dl.to_numpy(dtype=np.float64)
                    close(x['encoder_cont'][j,:,rebuilt.reals.index('glucose_mg_dl')],(np.log(enc)-mu)/sigma)
                    encoded=rebuilt.target_normalizer.nmd_subject_mapping[row.subject_id]
                    close(rebuilt.target_normalizer.get_parameters([encoded]),[mu,sigma])
                close(x['target_scale'],scales)
                try:reg.verified(rid,'last',expected)
                except ValueError as ex:assert str(ex)=='Completed/early-stopped run cannot resume; use a new weights-only run'
                else:raise AssertionError('completed LAST accepted')
                evidence[label]=dict(run_id=rid,best_sha256=record['sha256'],best_step=record['global_step'],checks=['oracle','ownership','all tensors','Q/features','dataset parameters','normalizer/map','encoder_cont','target_scale','window keys','completed LAST refusal'])
            report['serialization']=evidence
        else:
            injection=name.split('_N_')[1]
            model=p.build_model(training,args,None)
            def inject_gradient(g):
                report['gradient_injected']=True
                return torch.full_like(g,float('nan'))
            if injection=='gradient':next(model.parameters()).register_hook(inject_gradient)
            class Probe(pl.Callback):
                def on_train_start(self,t,m):n.verify(device)
                def on_train_batch_start(self,t,m,batch,i):
                    assert t.global_step==0 and i==0
                    x,y=batch
                    if injection=='target':
                        y[0][0,0]=float('inf')
                        report['injection']=dict(target_inf=bool(torch.isposinf(y[0][0,0])),input_target_inf=bool(torch.isposinf(x['decoder_target'][0,0])))
                        assert report['injection']['target_inf']
                    if injection=='input':
                        x['encoder_cont'][0,0,0]=float('nan');report['input_injected']=bool(torch.isnan(x['encoder_cont'][0,0,0]))
                def on_before_backward(self,t,m,loss):
                    if injection!='gradient':deny('unexpected backward')
                    count('backward');n.verify(device)
                def on_before_optimizer_step(self,t,m,optimizer):deny('containment before base AdamW')
            try:p.train(model,training,validation,args,None,extra_callbacks=[Probe()])
            except FloatingPointError as ex:
                frames=traceback.extract_tb(ex.__traceback__)
                assert not report.get('containment_first')
                assert Path(frames[-1].filename).resolve()==Path(f.__file__).resolve() and frames[-1].name=='require_finite'
                allowed={'target':{'Nonfinite state: model input','Nonfinite state: target','Nonfinite state: valid target'},'input':{'Nonfinite state: model input'},'gradient':{'Nonfinite state: gradient'}}
                assert str(ex) in allowed[injection]
                if injection=='target':
                    assert report['injection']['target_inf']
                    if str(ex)=='Nonfinite state: model input':assert report['injection']['input_target_inf']
                if injection=='gradient':assert report['gradient_injected']
                if injection=='input':assert report['input_injected']
                report['production_failure']=dict(type=type(ex).__name__,message=str(ex),stack=traceback.format_exc())
            else:raise AssertionError('negative probe succeeded')
            reg=r.Registry(args.registry_dir);manifest=reg.manifest(model._nmd_run_id)
            assert manifest['best'] is None and manifest['last'] is None
            assert json.loads(ledger.read_text())['attempts']==0
            report.update(run_id=model._nmd_run_id,no_valid_best_last=True)
        report['status']='PASS';return 0
    except Exception:
        report.update(status='FAIL',traceback=traceback.format_exc());return 1
    finally:
        sys.setprofile(None);report['counters']=json.loads(ledger.read_text());c.save(out/'result.json',report)


def parent():
    started=time.monotonic()
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='research/baseline-audit'
    run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-closure'
    root=c.ROOT/'ml/models/stage_c_training_recertification'/run_id;meta=c.ROOT/'experiments/stage_c_training_recertification'/run_id
    root.mkdir(exist_ok=False);meta.mkdir(exist_ok=False)
    (meta/'report_before.md').write_bytes((c.ROOT/'docs/STAGE_C_TRAINING_RECERTIFICATION_REPORT.md').read_bytes())
    original=inventory(ORIGINAL);old=json.loads((ORIGINAL/'preregistration.json').read_text())
    for p,h in old['source_hashes'].items():
        if p!='ml/tests/stage_c_recertification/campaign.py':assert c.sha(c.ROOT/p)==h,p
    allow=['R0/fixture.pt','R0/fixture.json','owned/owned_runs.json']
    index=json.loads((ORIGINAL/'owned/owned_runs.json').read_text());assert index['version']==1
    for label in ('CPU_U','CUDA_C1'):
        d=json.loads((ORIGINAL/label/'result.json').read_text());rid=d['run_id'];directory=ORIGINAL/'owned'/rid
        assert index['runs'][rid]['path']==str(directory)
        manifest=json.loads((directory/'run.json').read_text());assert manifest==d['registry']
        for role in ('best','last'):
            rec=manifest[role];path=directory/rec['file'];assert path.resolve().parent==directory and c.sha(path)==rec['sha256']
            allow.append(str(path.relative_to(ORIGINAL)))
        allow.extend([str((directory/'run.json').relative_to(ORIGINAL)),label+'/result.json',label+'/raw.pt'])
    source_files=c.SOURCES+[str(Path(__file__).relative_to(c.ROOT)),'docs/STAGE_C_TRAINING_RECERTIFICATION_CLOSURE_REVIEW.md']
    spec=dict(revision='nmd-c02-narrow-closure-1',original=str(ORIGINAL),original_inventory=original,read_allowlist=sorted(set(allow)),
        source_hashes={p:c.sha(c.ROOT/p) for p in source_files},limits=LIMITS,workers=ORDER,worker_seconds=90,total_seconds=300,
        config=c.CONFIG,tolerance=dict(atol=1e-5,rtol=1e-5),weights_only='NOT RECERTIFIED AFTER C02 / DEFERRED',
        inherited=['R0','CPU replay/restore','CUDA C1/C2/C3/replay/restore','CPU input NaN'],
        guard='Python audit I/O and profile call observer; no method replacement. Base AdamW entry is forbidden and counted as failure.',
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        initial_status=subprocess.check_output(['git','status','--short'],text=True),original_preregistration_sha256=c.sha(ORIGINAL/'preregistration.json'))
    c.save(root/'preregistration.json',spec);c.save(meta/'preregistration.json',spec)
    c.save(root/'ledger.json',dict(attempts=0,completions=0,forward=0,backward=0,candidates=0))
    status='PASS';calls=[]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(root/'mpl'),TMPDIR=str(root),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for name in ORDER:
        remaining=300-(time.monotonic()-started)
        if remaining<=0:status='TIMEOUT';break
        out=root/name;out.mkdir();command=[str(c.ROOT/'.venv/bin/python'),'-B',str(Path(__file__).resolve()),'--root',str(root),'--worker',name]
        begin=time.monotonic()
        with (out/'console.log').open('w') as stream:
            proc=subprocess.Popen(command,cwd=c.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=min(90,remaining))
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124;status='TIMEOUT'
        unchanged=inventory(ORIGINAL)==original
        calls.append(dict(name=name,command=command,exit_code=code,seconds=time.monotonic()-begin,original_unchanged=unchanged))
        c.save(root/'invocations.json',calls);c.save(meta/'invocations.json',calls)
        if (out/'result.json').exists():c.save(meta/(name+'.json'),json.loads((out/'result.json').read_text()))
        print(name,code,flush=True)
        if code or not unchanged:
            if status!='TIMEOUT':status='FAIL'
            break
    result=dict(verdict=status,counters=json.loads((root/'ledger.json').read_text()),invocations=calls,seconds=time.monotonic()-started,original_unchanged=inventory(ORIGINAL)==original,completed=len(calls)==6,root=str(root),preregistration_sha256=c.sha(root/'preregistration.json'))
    c.save(root/'result.json',result);c.save(meta/'result.json',result)
    if (root/'preflight.json').exists():c.save(meta/'preflight.json',json.loads((root/'preflight.json').read_text()))
    print(json.dumps(result,indent=2));return 0 if status=='PASS' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path);parser.add_argument('--worker',choices=ORDER);a=parser.parse_args()
    raise SystemExit(worker(a.root,a.worker) if a.worker else parent())
