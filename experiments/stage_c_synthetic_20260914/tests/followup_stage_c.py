"""Synthetic audit ONLY. No production remediation. Run --repo PATH --out NEW_DIR.
Contract assertions deliberately fail on existing production defects; no xfail.
"""
import argparse, contextlib, copy, hashlib, io, json, math, os, signal, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--out',required=True); ap.add_argument('--only',default=''); opts=ap.parse_args()
REPO=Path(opts.repo).resolve(); OUT=Path(opts.out).resolve(); OUT.mkdir(parents=True,exist_ok=False)
CONFIG=Path(__file__).resolve().parents[1]/'configs/baseline_stage_c_audit.json'
cfg=json.loads(CONFIG.read_text()); start=time.monotonic(); events=[]; results=[]; captures={}; forwards=0
os.environ.update(MPLCONFIGDIR=str(OUT/'mpl'),TMPDIR=str(OUT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str,allow_nan=True)+'\n')
def digest(b): return hashlib.sha256(b).hexdigest()
def deny(label):
    def blocked(*a,**k): events.append({'blocked':label}); raise PermissionError('C-core guard: '+label)
    return blocked
def audit(event,args):
    if event=='import' and (args[0]=='optuna' or args[0].startswith('optuna.')): deny('Optuna import')()
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(OUT): return
        forbidden=p.is_relative_to(REPO/'ml/data') or p.is_relative_to(REPO/'ml/models')
        forbidden |= p.suffix.lower() in {'.parquet','.xml','.csv','.ckpt','.pt','.pth','.npy','.npz'} and not p.is_relative_to(REPO/'.venv')
        if forbidden: deny('real/unlisted data open: '+str(p))()
sys.addaudithook(audit)
signal.signal(signal.SIGALRM,lambda *a: (_ for _ in ()).throw(TimeoutError('600s watchdog'))); signal.alarm(cfg['budget']['watchdog_seconds'])
# Positive refusal before any ML import or fixture creation.
for path in [REPO/'ml/data/processed/baseline_v1_stage_a_20260913T115538143463Z/training.parquet',REPO/'ml/data/raw/forbidden.xml',REPO/'ml/models/forbidden.ckpt']:
    try: open(path,'rb'); raise AssertionError('Guard allowed forbidden access')
    except PermissionError: pass
sys.path.insert(0,str(REPO))
import logging
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
import pytorch_forecasting as pf
from torch.utils.data import DataLoader
from ml.scripts import train_tft_population_v2 as p
from ml.scripts import observed_windows as ow
from ml.scripts import numerical_profile as numerics
from ml.scripts import checkpoint_registry as registry
for name in ['main','train','train_experimental_legacy','train_baseline','load_and_preprocess_data','load_test_data']:
    setattr(p,name,deny(name))
pl.Trainer.fit=deny('Trainer.fit'); torch.Tensor.backward=deny('Tensor.backward'); torch.autograd.backward=deny('autograd.backward')
for cl in [torch.optim.Optimizer,torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD,p.CheckedAdamW]: cl.step=deny(cl.__name__+'.step')
torch.set_num_threads(1); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); numerics.configure('cpu')
logging.getLogger().setLevel(logging.WARNING)
logfile=logging.FileHandler(OUT/'production.log'); p.log.addHandler(logfile); p.log.setLevel(logging.INFO)

def check(condition,detail='contract not satisfied'):
    if not condition: raise AssertionError(detail)
def close(a,b,fp32=False):
    tol=cfg['tolerances']['integration' if fp32 else 'oracle']; np.testing.assert_allclose(a,b,**tol)
def test(name,kind,fn):
    if opts.only and opts.only not in name: return
    t=time.monotonic(); row={'id':name,'kind':kind}
    try:
        value=fn(); row.update(status='PASS',evidence=value)
    except AssertionError as e: row.update(status='FAIL',error=str(e),traceback=traceback.format_exc())
    except Exception as e: row.update(status='ERROR',error=repr(e),traceback=traceback.format_exc())
    row['seconds']=time.monotonic()-t; results.append(row); dump('results.partial.json',results); print(name,row['status'],flush=True)

def frame(role):
    rows=[]
    # 65 regular bins each: exactly 6 full 48/12 rolling windows per person.
    for sid,base,slope in [('synthetic_A',75.,.8),('synthetic_B',180.,1.5)]:
        for i in range(65):
            d={col:float((i%7)/7) for col in set(p.TIME_VARYING_KNOWN_REALS+p.TIME_VARYING_UNKNOWN_REALS)}
            d.update({f'glucose_lag_{lag}_available':1. for lag in [1,2,3,6,12,24]})
            d.update(subject_id=sid,time_idx=i,timestamp=pd.Timestamp('2024-01-01')+pd.Timedelta(minutes=5*i),source_split=role,target_observed=True,glucose_mg_dl=base+slope*i)
            rows.append(d)
    return pd.DataFrame(rows)
train_df=frame('synthetic_train'); val_df=frame('synthetic_assessment')
train_ds,unfiltered=p.build_datasets(train_df,val_df,SimpleNamespace(context=48,horizon=12))
# Audit fixture restriction only; retain production unfiltered dataset for C03 probe.
full=lambda d: (d.time_idx_first_prediction-d.time_idx_first==48)&(d.time_idx_last-d.time_idx_first_prediction==11)
ds=unfiltered.filter(full); check(len(ds)==12,'fixture must have 12 full windows')
train_df.to_csv(OUT/'synthetic_train.csv',index=False); val_df.to_csv(OUT/'synthetic_assessment.csv',index=False)
args=SimpleNamespace(batch_size=3,no_gpu=True)
model=p.ClinicalTFT.from_dataset(train_ds,hidden_size=8,hidden_continuous_size=4,attention_head_size=1,lstm_layers=1,dropout=.1,loss=p.ClinicalQuantileLoss(quantiles=cfg['quantiles']),output_size=7,log_interval=-1)
model.eval(); orig_forward=model.forward

def counted(*a,**k):
    global forwards
    check(not torch.is_grad_enabled(),'forward must be no_grad/inference'); forwards+=1
    check(forwards<=24,'forward budget exceeded'); numerics.verify('cpu')
    return orig_forward(*a,**k)
model.forward=counted
initial={k:v.detach().clone() for k,v in model.state_dict().items()}
# Owned synthetic untrained state only; not an inference-certified BEST checkpoint.
torch.save({'role':'SYNTHETIC_UNTRAINED_AUDIT_STATE','state_dict':initial,'quantiles':cfg['quantiles']},OUT/'synthetic_untrained.pt')
real_predict=model.predict

def capture_predict(*a,**k):
    r=real_predict(*a,**k); captures['real']=r; return r
model.predict=capture_predict

def integration():
    with torch.no_grad(),patch.object(p,'_STATSMODELS_AVAILABLE',False):
        m=p.evaluate(model,ds,args,val_df)
    captures['real_metrics']=m
    r=captures['real']; check(r.output.shape==(12,12,7)); check(torch.isfinite(r.output).all())
    ix=ds.x_to_index(r.x); lookup=val_df.set_index(['subject_id','time_idx']).glucose_mg_dl
    expected=np.array([[lookup.loc[(sid,int(t)+h)] for h in range(12)] for sid,t in zip(ix.subject_id,ix.time_idx)])
    close(r.y[0].numpy(),expected,True)
    # Raw reporting values are rounded to 4 decimals: separate precision finding.
    for h in [5,15,30,45,60]:
        j=h//5-1; y=expected[:,j]; pred=r.output[:,j,3].numpy(); mask=(pred>0)&(y>20)&(y<400)
        close(m[f'val_mae_{h}m_mg_dl'],round(float(np.abs(pred[mask]-y[mask]).mean()),4),True)
    dump('real_metrics.json',m)
    np.savez(OUT/'real_predictions.npz',output=r.output.numpy(),target=r.y[0].numpy(),**{k:v.numpy() for k,v in r.x.items() if isinstance(v,torch.Tensor)})
    keys=[{'subject':str(s),'window_id':f'{s}:{int(t)-1}','origin_timestamp':str(pd.Timestamp('2024-01-01')+pd.Timedelta(minutes=5*(int(t)-1))),'target_timestamp':str(pd.Timestamp('2024-01-01')+pd.Timedelta(minutes=5*(int(t)+h))),'horizon_minutes':(h+1)*5} for s,t in zip(ix.subject_id,ix.time_idx) for h in range(12)]
    dump('window_keys.json',keys); captures['keys']=keys
    return {'shape':list(r.output.shape),'target_alignment':'exact source mg/dL','metrics_match':'existing five horizons, existing masks, rounded','windows':12}
test('C19_real_PF_predict_evaluate','production integration',integration)
if 'real' not in captures: dump('results.json',results); sys.exit(2)
base=captures['real']

def payload(y=None,pred=None,q=None):
    y=base.y[0].double().clone() if y is None else torch.as_tensor(y,dtype=torch.float64)
    if y.ndim==1: y=y[:,None].repeat(1,12)
    if pred is None: output=y[:,:,None].repeat(1,1,7)
    else:
        v=torch.as_tensor(pred,dtype=torch.float64)
        if v.ndim==1: v=v[:,None].repeat(1,12)
        output=v[:,:,None].repeat(1,1,7) if v.ndim==2 else v
    return SimpleNamespace(y=(y,None),output=output,x={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in base.x.items()})

def evaluate(r=None,frame_=None,dataset=None,quantiles=None,arima=None,label='probe'):
    r=payload() if r is None else r
    fake=SimpleNamespace(predict=lambda *a,**k:r,loss=SimpleNamespace(quantiles=quantiles or cfg['quantiles']))
    stream=io.StringIO(); handler=logging.StreamHandler(stream); p.log.addHandler(handler)
    try:
        with patch.object(p,'_STATSMODELS_AVAILABLE',arima is not None),patch.object(p,'_compute_arima_baseline',arima or (lambda *a:None)):
            m=p.evaluate(fake,dataset or ds,args,val_df if frame_ is None else frame_)
    finally: p.log.removeHandler(handler)
    captures[label]={'metrics':m,'log':stream.getvalue()}; return m,stream.getvalue()
def must_reject(fn):
    try: fn()
    except (ValueError,RuntimeError,FloatingPointError): return 'production refused'
    raise AssertionError('Expected INVALID/refusal; production returned metrics')
def isolation():
    for fn in [p.main,p.load_test_data,pl.Trainer.fit,torch.optim.AdamW.step,torch.Tensor.backward]:
        try: fn(); raise AssertionError('entry allowed')
        except PermissionError: pass
    return events


# Focused follow-up, preregistered after run02. No production changes.
def normalizer_fallback():
    ix=ds.x_to_index(base.x); wanted=ds.target_normalizer.get_norm(ix[['subject_id']]); actual=base.x['target_scale'].numpy(); missing=ds.target_normalizer.missing_
    dump('normalizer_probe.json',{'fitted':registry.semantic(ds.target_normalizer.norm_),'missing':missing,'expected':wanted.tolist(),'actual':actual.tolist(),'encoded_groups':base.x['groups'].tolist()})
    close(actual,wanted,True)
test('C02_subject_normalizer_parameters','production targeted follow-up',normalizer_fallback)
def transform_two_scales():
    scales=torch.tensor([[4.,.1],[5.,.5]],dtype=torch.float64); z=torch.ones((2,12,7),dtype=torch.float64)
    out=model.transform_output(z,target_scale=scales).detach().numpy(); close(out,np.broadcast_to(np.exp([4.1,5.5])[:,None,None],(2,12,7)))
    return 'Native inverse math correct on independently supplied distinct centers/scales'
test('C02_distinct_inverse_math','production + oracle',transform_two_scales)
def finite_inside_model():
    def bad(module,inputs,out): return out*float('nan')
    handle=model.output_layer.register_forward_hook(bad)
    try:
        with torch.no_grad(),patch.object(p,'_STATSMODELS_AVAILABLE',False):
            return must_reject(lambda:p.evaluate(model,ds,args,val_df))
    finally: handle.remove()
test('C03_native_nonfinite_model_gate','real PF integration',finite_inside_model)
def corrupt_after_model():
    # Fault injected AFTER production FiniteModel.forward validation, before PF predict callback.
    # Tests evaluate's own boundary, not a claim that native model NaN evades its guard.
    def bad(module,inputs,out):
        out.prediction[0,11,3]=float('nan'); return out
    handle=model.register_forward_hook(bad)
    try:
        with torch.no_grad(),patch.object(p,'_STATSMODELS_AVAILABLE',False):
            m=p.evaluate(model,ds,args,val_df); dump('real_corrupt_metrics.json',m)
            raise AssertionError(f'Real PF→evaluate returned metrics after output fault; N={m.get("n_valid_samples")} instead of INVALID')
    finally: handle.remove()
test('C03_real_predict_late_corruption','real PF integration fault injection',corrupt_after_model)
def target_alignment_boundary():
    r=payload(); r.y=(r.y[0].flip(0),None)
    return must_reject(lambda:evaluate(r,label='swapped_targets'))
test('C01_target_key_mismatch','controlled prediction boundary',target_alignment_boundary)
def owned_roundtrip():
    artifact=OUT/'synthetic_untrained.pt'; before=digest(artifact.read_bytes()); value=torch.load(artifact,weights_only=True)
    check(value['role']=='SYNTHETIC_UNTRAINED_AUDIT_STATE'); check(value['quantiles']==cfg['quantiles']); check(digest(artifact.read_bytes())==before)
    check(all(torch.equal(initial[k],v) for k,v in value['state_dict'].items())); return {'sha256':before,'role':value['role'],'roundtrip':'exact','not_BEST':True}
test('C18_owned_untrained_roundtrip','audit artifact',owned_roundtrip)
def registry_mismatch():
    # Existing compatibility gate, no checkpoint read or training; production evaluate does not invoke it.
    import inspect
    source=inspect.getsource(registry.check_compatible); dump('registry_compatibility_source.json',source)
    original={'protocol':'synthetic','quantiles':[.1,.5,.9],'normalizer':{'center':1},'schema':'synthetic','calibration':'none'}
    variants=[]
    for field,val in [('quantiles',[.5,.1,.9]),('normalizer',{'center':2}),('schema','wrong'),('calibration','test')]:
        altered=copy.deepcopy(original); altered[field]=val
        try: registry.check_compatible(original,altered,'resume-last')
        except ValueError: variants.append(field)
    check(len(variants)==4); return {'refused':variants,'evaluation_gate':'not wired to evaluate'}
test('C18_registry_compatibility_controls','existing production helper',registry_mismatch)
def nearzero():
    r=payload(y=[.1]+[100]*11,pred=[1.]+[100]*11); m,_=evaluate(r,label='nearzero'); check(m.get('n_valid_samples')==12,'small positive target excluded'); close(m['val_mard_60m_pct'],75.)
test('C04_nearzero_positive','production contract',nearzero)
def exact_zero_pred():
    r=payload(y=[100]*12,pred=[0]+[100]*11); m,_=evaluate(r,label='zero_prediction'); check(m.get('n_valid_samples')==12,'finite zero prediction excluded'); close(m['val_mae_60m_mg_dl'],100/12)
test('C03_finite_zero_retained','production contract',exact_zero_pred)
def pure_clarke():
    # Geometrically interior points only; complete boundary certification is NOT EVALUATED.
    pairs=[(100,100,'A'),(100,150,'B'),(100,250,'C'),(300,100,'D'),(50,250,'E')]
    for y,z,zone in pairs:
        result=p.clarke_error_grid(torch.tensor([y]),torch.tensor([z])); check(result[zone]==100,str((y,z,result)))
    return '5 interior zone examples; primary full figure inaccessible HTTP403; boundary sweep NOT EVALUATED'
test('C17_Clarke_interior_examples','limited geometry probe',pure_clarke)
def loss_unweighted_quantiles():
    # Analytic uniform weighted CDF inverse for all Q, plus deterministic CDF quadrature.
    y=(np.arange(14000)+.5)/100; w=np.where(y<70,2.5,1.); out=[]
    for q in cfg['quantiles']:
        a=q*245/2.5 if q<=175/245 else 70+q*245-175
        close(np.sum(w[y<=a])/w.sum(),q); out.append(a)
    return {'weighted_targets':out,'nominal_q':cfg['quantiles']}
test('C13_all_q_weighted_CDF','oracle',loss_unweighted_quantiles)
def no_updates():
    check(all(torch.equal(initial[k],v) for k,v in model.state_dict().items())); check(all(v.grad is None for v in model.parameters())); return {'forward_calls':forwards,'state_unchanged':True}
test('C19_followup_no_updates','audit isolation',no_updates)
# Preserve exact observations and all failures, not expected-failure annotations.
dump('probe_captures.json',captures | {'real':'stored in real_predictions.npz','keys':'window_keys.json'})
dump('results.json',results); dump('guard_events.json',events)
counts={s:sum(r['status']==s for r in results) for s in ['PASS','FAIL','ERROR','SKIP','XFAIL']}
manifest={'scope':'SYNTHETIC C-core','config_sha256':digest(CONFIG.read_bytes()),'runner_sha256':digest(Path(__file__).read_bytes()),'source_commit':cfg['commit'],'command':sys.argv,'environment':{'python':sys.version,'torch':torch.__version__,'pandas':pd.__version__,'numpy':np.__version__,'lightning':pl.__version__,'pytorch_forecasting':pf.__version__,'statsmodels_available':p._STATSMODELS_AVAILABLE},'counts':counts,'forward_calls':forwards,'full_windows_used':len(ds),'optimizer_steps':0,'backward':0,'elapsed_seconds':time.monotonic()-start,'real_data_opened':False,'model_state_unchanged':all(torch.equal(initial[k],v) for k,v in model.state_dict().items()),'normalizer':registry.semantic(train_ds.target_normalizer),'numerical_profile':numerics.effective(),'exit_code':1 if counts['FAIL'] or counts['ERROR'] else 0,'source_hashes_match':all(digest((REPO/k).read_bytes())==v for k,v in cfg['code_hashes'].items())}
manifest['artifact_hashes']={str(f.relative_to(OUT)):digest(f.read_bytes()) for f in OUT.iterdir() if f.is_file()}
dump('manifest.json',manifest); print(json.dumps(manifest['counts']),flush=True); signal.alarm(0); sys.exit(manifest['exit_code'])
