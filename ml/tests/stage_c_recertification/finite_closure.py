"""Two-phase finite closure with ordered containment evidence; no production patches."""
import argparse
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import campaign as c
from closure import inventory, ORIGINAL

ORDER=['diagnostic','CPU_N_target','CPU_N_gradient','CUDA_N_input','CUDA_N_target','CUDA_N_gradient']
LIMITS=dict(attempts=0,completions=0,forward=32,backward=2,candidates=0)
KNOWN_DIR=c.ROOT/'ml/models/baseline_v1_stage_a'

def known_benign(row):
    return (row['audit_event']=='os.mkdir' and row['operation']=='os.mkdir' and
        row['resource']==str(KNOWN_DIR) and row['existed_directory'] and
        row['injection_seq'] is None and row['escaped_immediate_library_call'] is False and
        not row['became_worker_final_uncaught_exception'] and
        row['immediate_library_call']['function']=='mkdir' and
        any(fr['file']==str(c.ROOT/'ml/scripts/train_tft_population_v2.py') and fr['function']=='<module>' for fr in row['stack']))

def dir_identity():
    st=KNOWN_DIR.stat()
    return dict(device=st.st_dev,inode=st.st_ino,mode=st.st_mode,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)



def worker(root,name):
    out=root/name;ledger=root/'ledger.json';spec=json.loads((root/'preregistration.json').read_text())
    report=dict(name=name,status='ERROR');events=[];blocked={};live={};injection_seq=None;final_exception=None
    def event(kind,**values):
        row=dict(seq=len(events)+1,monotonic_ns=time.monotonic_ns(),kind=kind,**values);events.append(row);return row
    def count(key):
        d=json.loads(ledger.read_text());d[key]+=1;c.save(ledger,d)
        if d[key]>LIMITS[key]:raise RuntimeError('budget '+key)
    def deny(audit_name,resource,operation):
        caller=sys._getframe(2)
        path=Path(resource) if isinstance(resource,str) and resource.startswith('/') else None
        row=event('containment',audit_event=audit_name,resource=resource,operation=operation,
            stack=[dict(file=f.filename,line=f.lineno,function=f.name) for f in traceback.extract_stack()[:-1]],
            injection_seq=injection_seq,existed_directory=bool(path and path.is_dir()),
            immediate_library_call=dict(file=caller.f_code.co_filename,function=caller.f_code.co_name,line=caller.f_lineno),
            escaped_immediate_library_call=None,became_worker_final_uncaught_exception=False,propagation=[])
        ex=PermissionError('containment '+audit_name+' '+str(resource))
        blocked[id(ex)]=row;live[id(ex)]=(ex,caller,caller.f_back)
        raise ex
    def audit(a,args):
        if a=='import' and args[0].split('.')[0]=='optuna':deny(a,'optuna','import')
        if a=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
            write=(isinstance(mode,str) and any(k in mode for k in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR))
            if write and not path.is_relative_to(root) and str(path)!='/dev/null':deny(a,str(path),'write')
            if not write and path.is_relative_to(c.ROOT/'ml/data'):deny(a,str(path),'read real data')
            if not write and path.is_relative_to(c.ROOT/'ml/models') and not path.is_relative_to(root):
                if path not in (ORIGINAL/'R0/fixture.pt',ORIGINAL/'R0/fixture.json'):deny(a,str(path),'read unlisted artifact')
        if a in ('os.remove','os.rmdir','os.mkdir','os.rename','os.link','os.symlink'):
            paths=args[:2] if a in ('os.rename','os.link','os.symlink') else args[:1]
            for value in paths:
                if isinstance(value,(str,bytes,os.PathLike)):
                    path=Path(os.fsdecode(value)).resolve()
                    if not path.is_relative_to(root):deny(a,str(path),a)
    finite_file=str(c.ROOT/'ml/scripts/finite_training.py')
    def trace(frame,kind,arg):
        frame.f_trace_lines=False
        if kind=='exception':
            typ,ex,tb=arg
            if id(ex) in blocked:
                row=blocked[id(ex)];row['propagation'].append(dict(file=frame.f_code.co_filename,function=frame.f_code.co_name,line=frame.f_lineno))
                if frame is live[id(ex)][2]:row['escaped_immediate_library_call']=True
            if isinstance(ex,FloatingPointError) and frame.f_code.co_filename==finite_file and frame.f_code.co_name=='require_finite':
                event('production_finite_exception',type=type(ex).__name__,message=str(ex),injection_seq=injection_seq)
        if kind=='return':
            for key,(_,immediate,parent) in live.items():
                if frame is immediate and blocked[key]['escaped_immediate_library_call'] is None:blocked[key]['escaped_immediate_library_call']=False
        return trace
    sys.settrace(trace);sys.addaudithook(audit)
    try:
        for file,h in spec['source_hashes'].items():assert c.sha(c.ROOT/file)==h,file
        import numpy as np
        import torch
        import lightning.pytorch as pl
        import pytorch_forecasting as pf
        import pandas as pd
        from types import SimpleNamespace as NS
        sys.path.insert(0,str(c.ROOT))
        from ml.scripts import train_tft_population_v2 as p,numerical_profile as n,finite_training as f,checkpoint_registry as r
        torch.set_num_threads(1);pl.seed_everything(42,workers=True)
        device='cuda' if name.startswith('CUDA') else 'cpu';n.configure(device)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3070','host GPU unavailable'
        report['environment']=dict(python=sys.version,torch=torch.__version__,lightning=pl.__version__,pf=pf.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),profile=n.verify(device))
        assert (torch.__version__,pl.__version__,pf.__version__,np.__version__,pd.__version__)==('2.11.0+cu130','2.6.1','1.7.0','2.4.4','2.3.3')
        base=inspect.unwrap(torch.optim.AdamW.step).__code__
        def observe(frame,kind,arg):
            if kind!='call':return
            code=frame.f_code
            if code is base:
                event('base_adamw_delegation');count('attempts');raise RuntimeError('base AdamW forbidden')
            if code is f.FiniteModel.forward.__code__:
                event('forward',injection_seq=injection_seq)
                if name=='diagnostic':raise RuntimeError('diagnostic forward forbidden')
                count('forward');n.verify(device)
            if code is f.require_finite.__code__:
                event('require_finite_call',stage=frame.f_locals.get('stage'),injection_seq=injection_seq)
            if code is pl.Trainer.fit.__code__ and name=='diagnostic':raise RuntimeError('diagnostic fit forbidden')
            if code in (p.main.__code__,p.evaluate.__code__,p.load_and_preprocess_data.__code__,p.load_test_data.__code__):raise RuntimeError('forbidden entrypoint')
        sys.setprofile(observe)
        fixture=torch.load(ORIGINAL/'R0/fixture.pt',map_location='cpu',weights_only=False)
        training,validation=fixture['training'],fixture['validation']
        assert len(training)==len(validation)==4
        args=NS(**c.CONFIG,no_gpu=device=='cpu',mode='fresh',run_id=None,parent_run_id=None,stop_after_epoch=None,registry_dir=root/'probes',dataset_sha256=c.sha(ORIGINAL/'R0/fixture.json'))
        model=p.build_model(training,args,None)
        if name=='diagnostic':
            # Match imports, numerical settings, fixture load, model construction and empty registry bootstrap; no fit.
            r.Registry(args.registry_dir)
            seen=list(blocked.values());assert seen,'previous containment event not reproduced'
            for row in seen:
                # Only harmless mkdir(exist_ok) attempts on already existing nonsource directories qualify.
                path=Path(row['resource'])
                benign=known_benign(row)
                row['classification']='caught existing-directory mkdir attempt' if benign else 'UNEXPLAINED_OR_BOUNDARY_VIOLATION'
                assert benign,'Phase A containment needs review'
            report['phase_a_explained']=True
        else:
            injection=name.split('_N_')[1]
            def gradient(g):
                nonlocal injection_seq
                injection_seq=event('finite_injection',type='gradient NaN')['seq'];report['injection_proven']=True
                return torch.full_like(g,float('nan'))
            if injection=='gradient':next(model.parameters()).register_hook(gradient)
            class Probe(pl.Callback):
                def on_train_start(self,t,m):n.verify(device)
                def on_train_batch_start(self,t,m,batch,i):
                    nonlocal injection_seq
                    assert t.global_step==0 and i==0
                    x,y=batch
                    if injection=='target':
                        y[0][0,0]=float('inf');report['injection_proven']=bool(torch.isposinf(y[0][0,0]));report['input_target_inf']=bool(torch.isposinf(x['decoder_target'][0,0]))
                        injection_seq=event('finite_injection',type='target Inf')['seq']
                    elif injection=='input':
                        x['encoder_cont'][0,0,0]=float('nan');report['injection_proven']=bool(torch.isnan(x['encoder_cont'][0,0,0]));injection_seq=event('finite_injection',type='input NaN')['seq']
                def on_before_backward(self,t,m,loss):
                    event('backward',injection_seq=injection_seq)
                    assert injection=='gradient';count('backward');n.verify(device)
                def on_before_optimizer_step(self,t,m,op):raise RuntimeError('containment before base AdamW')
            try:p.train(model,training,validation,args,None,extra_callbacks=[Probe()])
            except FloatingPointError as ex:
                frames=traceback.extract_tb(ex.__traceback__)
                event('probe_uncaught_rejection',type=type(ex).__name__,message=str(ex),origin='production')
                assert report['injection_proven'] and injection_seq is not None
                assert frames[-1].filename==finite_file and frames[-1].name=='require_finite'
                allowed={'target':{'Nonfinite state: model input','Nonfinite state: target','Nonfinite state: valid target'},'input':{'Nonfinite state: model input'},'gradient':{'Nonfinite state: gradient'}}
                assert str(ex) in allowed[injection]
                if injection=='target' and str(ex)=='Nonfinite state: model input':assert report['input_target_inf']
                assert any(e['kind']=='production_finite_exception' and e['seq']>injection_seq for e in events)
                report['production_failure']=dict(type=type(ex).__name__,message=str(ex),stack=traceback.format_exc())
            else:raise AssertionError('probe not rejected')
            manifest=r.Registry(args.registry_dir).manifest(model._nmd_run_id)
            assert manifest['best'] is None and manifest['last'] is None
            assert json.loads(ledger.read_text())['attempts']==0
            report['no_valid_best_last']=True
        for row in blocked.values():
            path=Path(row['resource'])
            benign=known_benign(row)
            row['classification']='caught existing-directory mkdir attempt' if benign else 'UNEXPLAINED_OR_BOUNDARY_VIOLATION'
            assert benign,'unexplained containment event'
        report['status']='PASS';return 0
    except Exception as ex:
        final_exception=ex;report.update(status='FAIL',traceback=traceback.format_exc());return 1
    finally:
        sys.setprofile(None);sys.settrace(None)
        chain=[];ex=final_exception
        while ex is not None and id(ex) not in chain:
            chain.append(id(ex));ex=ex.__cause__ or ex.__context__
        for key,row in blocked.items():
            row['became_worker_final_uncaught_exception']=key in chain
        report['containment_event_seen']=bool(blocked)
        report['containment_uncaught_rejection']=any(row['became_worker_final_uncaught_exception'] for row in blocked.values())
        report['containment_was_first_uncaught_rejection_after_injection']=bool(injection_seq and any(row['seq']>injection_seq and row['became_worker_final_uncaught_exception'] for row in blocked.values()))
        report['counters']=json.loads(ledger.read_text());c.save(out/'events.json',events);c.save(out/'result.json',report)


def parent():
    started=time.monotonic();assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='research/baseline-audit'
    run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-finite';root=c.ROOT/'ml/models/stage_c_training_recertification'/run_id;meta=c.ROOT/'experiments/stage_c_training_recertification'/run_id
    root.mkdir(exist_ok=False);meta.mkdir(exist_ok=False)
    (meta/'report_before.md').write_bytes((c.ROOT/'docs/STAGE_C_TRAINING_RECERTIFICATION_REPORT.md').read_bytes())
    known_before=dir_identity();original=inventory(ORIGINAL);old=json.loads((ORIGINAL/'preregistration.json').read_text())
    for p,h in old['source_hashes'].items():
        if p!='ml/tests/stage_c_recertification/campaign.py':assert c.sha(c.ROOT/p)==h,p
    files=c.SOURCES+[str(Path(__file__).relative_to(c.ROOT)),'ml/tests/stage_c_recertification/closure.py']
    spec=dict(revision='nmd-c02-finite-containment-2',source_hashes={p:c.sha(c.ROOT/p) for p in files},original_inventory=original,known_directory_identity=known_before,
        phases=dict(A=['diagnostic'],B=ORDER[1:]),limits=LIMITS,worker_seconds=90,total_seconds=300,
        phase_a_acceptance='Exact blocked mkdir for existing ml/models/baseline_v1_stage_a from train_tft_population_v2 module import before injection, handled by pathlib.mkdir; unchanged directory identity and original inventory. Any other event STOP.',
        phase_b_acceptance='Proven injection; exact production require_finite origin/message; no uncaught containment rejection caused by injection; zero base AdamW; no valid BEST/LAST.',
        inherited=['CPU strict','CUDA C1/C2/C3/resume','LAST restore','BEST reconstruction','completed LAST refusal','CPU input NaN'],weights_only='NOT RECERTIFIED AFTER C02 / DEFERRED',head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    c.save(root/'preregistration.json',spec);c.save(meta/'preregistration.json',spec);c.save(root/'ledger.json',dict(attempts=0,completions=0,forward=0,backward=0,candidates=0))
    calls=[];status='PASS'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(root/'mpl'),TMPDIR=str(root),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for name in ORDER:
        remaining=300-(time.monotonic()-started)
        if remaining<=0:status='TIMEOUT';break
        out=root/name;out.mkdir();cmd=[str(c.ROOT/'.venv/bin/python'),'-B',str(Path(__file__).resolve()),'--root',str(root),'--worker',name];begin=time.monotonic()
        with (out/'console.log').open('w') as stream:
            proc=subprocess.Popen(cmd,cwd=c.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=min(90,remaining))
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124;status='TIMEOUT'
        unchanged=inventory(ORIGINAL)==original and dir_identity()==known_before;calls.append(dict(name=name,command=cmd,exit_code=code,seconds=time.monotonic()-begin,original_unchanged=unchanged))
        for file in ('result.json','events.json'):
            if (out/file).exists():c.save(meta/(name+'_'+file),json.loads((out/file).read_text()))
        c.save(meta/'invocations.json',calls);print(name,code,flush=True)
        if code or not unchanged:
            if status!='TIMEOUT':status='FAIL'
            break
    result=dict(verdict=status,invocations=calls,counters=json.loads((root/'ledger.json').read_text()),seconds=time.monotonic()-started,original_unchanged=inventory(ORIGINAL)==original,known_directory_unchanged=dir_identity()==known_before,completed=len(calls)==6,root=str(root),preregistration_sha256=c.sha(root/'preregistration.json'))
    c.save(root/'result.json',result);c.save(meta/'result.json',result);print(json.dumps(result,indent=2));return 0 if status=='PASS' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path);parser.add_argument('--worker',choices=ORDER);a=parser.parse_args()
    raise SystemExit(worker(a.root,a.worker) if a.worker else parent())
