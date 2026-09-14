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

test('C00_positive_guards','audit isolation',isolation)
def sentinel():
    def dev(data): return registry.semantic(data['train']),registry.semantic(data['val']),registry.semantic(data['cal'])
    d={'train':[1,2],'val':[3],'cal':[4],'test':[5]}; before=dev(d); d['test']=[-1e9]; check(dev(d)==before)
    # Real normalizer fit path: only synthetic training frame is supplied.
    norm=copy.deepcopy(train_ds.target_normalizer); a=norm.norm_.copy(); norm.fit(train_df.glucose_mg_dl,train_df); pd.testing.assert_frame_equal(a,norm.norm_)
    return 'audit role separation only; no production calibration fit exists'
test('C00_test_sentinel','oracle / normalizer',sentinel)

def invariance(batch,reverse=False):
    sampler=list(reversed(range(len(ds)))) if reverse else None
    loader=ds.to_dataloader(train=False,batch_size=batch,num_workers=0,shuffle=False,sampler=sampler)
    with torch.no_grad(): r=real_predict(loader,return_x=True,return_y=True,mode='quantiles',trainer_kwargs={'accelerator':'cpu','logger':False})
    def keyed(obj):
        ix=ds.x_to_index(obj.x); return {(str(s),int(t)):(obj.output[i].numpy(),obj.y[0][i].numpy()) for i,(s,t) in enumerate(zip(ix.subject_id,ix.time_idx))}
    a,b=keyed(base),keyed(r); check(a.keys()==b.keys())
    for k in a: close(a[k][0],b[k][0],True); close(a[k][1],b[k][1],True)
    return {'batch_size':batch,'reverse':reverse,'keys':len(a),'max_abs':max(float(np.max(np.abs(a[k][0]-b[k][0]))) for k in a)}
test('C01_uneven_batch','production integration',lambda:invariance(5))
test('C01_permuted_batches','production integration',lambda:invariance(4,True))

def units():
    r=payload(y=[50,100]*6,pred=[60,80]*6); m,_=evaluate(r,label='hand_metrics')
    close(m['val_mae_60m_mg_dl'],15); close(m['val_mard_60m_pct'],20)
    r.y=(r.y[0]*2,torch.full_like(r.y[0],13)); r.output*=2; n,_=evaluate(r,label='scale2')
    close(n['val_mae_60m_mg_dl'],30); close(n['val_mard_60m_pct'],20)
    return 'native units x2; tuple weight13 does not act as target_scale'
test('C02_native_units_tuple_scale2','production',units)

def raw_transform():
    # Local PF convention exp(output*scale+center) independently for log normalizer.
    scale=base.x['target_scale'].double(); norm=torch.zeros((12,12,7),dtype=torch.float64)
    out=model.transform_output(norm,target_scale=scale)
    close(out.detach().numpy(),np.broadcast_to(np.exp(scale[:,0].numpy())[:,None,None],(12,12,7)))
    check(scale[:,0].unique().numel()==2); check(scale[:,1].unique().numel()==2)
    return {'centers':scale[:,0].unique().tolist(),'scales':scale[:,1].unique().tolist()}
test('C02_group_transform','production + independent oracle',raw_transform)
for tag,value in [('nan',float('nan')),('posinf',float('inf')),('neginf',-float('inf'))]:
    def probe(v=value,t=tag):
        r=payload(); r.output[0,11,3]=v; return must_reject(lambda:evaluate(r,label='nonfinite_'+t))
    test('C03_prediction_'+tag,'production contract',probe)
def target_bad():
    r=payload(); r.y[0][0,11]=float('nan'); return must_reject(lambda:evaluate(r,label='target_nan'))
test('C03_target_nan','production contract',target_bad)
def finite_negative():
    r=payload(); r.output[0,:,3]=-10; m,_=evaluate(r,label='negative_prediction')
    check(m.get('n_valid_samples')==12,f"N={m.get('n_valid_samples')}; expected12"); close(m['val_mae_60m_mg_dl'],float(torch.abs(r.output[:,11,3]-r.y[0][:,11]).mean()))
test('C03_finite_negative_retained','production contract',finite_negative)
def extremes():
    r=payload(y=[20,400,401]+[100]*9,pred=[30,390,390]+[100]*9); m,_=evaluate(r,label='extreme_targets'); check(m.get('n_valid_samples')==12,f"N={m.get('n_valid_samples')}; expected12")
test('C03_targets20_400_401','production contract',extremes)
def observed():
    f=val_df.copy(); f.loc[(f.subject_id=='synthetic_A')&(f.time_idx==59),'target_observed']=False
    return must_reject(lambda:evaluate(frame_=f,label='unobserved'))
test('C03_unobserved_decoder','production guard',observed)
def full_context():
    ix=unfiltered.decoded_index; short=ix.time_idx_first_prediction-ix.time_idx_first<48
    check(short.any(),'fixture must include short encoders'); ow.assert_observed_evaluation(unfiltered,val_df)
    raise AssertionError(f'Production evaluation guard accepts {int(short.sum())}/{len(ix)} encoders shorter than48')
test('C03_full48_guard','production contract',full_context)
def short_decoder():
    r=payload(); r.x['decoder_lengths'][0]=6; r.y[0][0,6:]=100; r.output[0,6:,:]=200
    return must_reject(lambda:evaluate(r,label='short_decoder'))
test('C03_decoder_lengths_padding','controlled prediction boundary',short_decoder)

def scalar():
    y=np.array([50,100.]); z=np.array([60,80.]); e=z-y
    close([np.abs(e).mean(),np.sqrt((e*e).mean()),100*np.mean(np.abs(e)/y),e.mean()],[15,math.sqrt(250),20,-5])
    for n in [1,2]: close(sum(np.abs(e[i:i+n]).sum() for i in range(0,2,n))/2,15)
    return {'MAE':15,'RMSE':math.sqrt(250),'MARD':20,'bias':-5}
test('C04_independent_scalar_oracle','oracle',scalar)
def precision():
    m,_=evaluate(payload(y=[50,100]*6,pred=[60,80]*6),label='precision'); close(m['val_rmse_60m_mg_dl'],math.sqrt(250))
test('C04_reporting_precision','production contract',precision)
def zero_target(): return must_reject(lambda:evaluate(payload(y=[0]+[100]*11),label='zero_target'))
test('C04_zero_target_invalid','production contract',zero_target)
def empty():
    m,_=evaluate(payload(y=[100]*12,pred=[0]*12),label='empty_mask'); check('n_valid_samples' in m,'No explicit N=0/null/reason; invalid selection returns only flags')
test('C04_empty_group','production contract',empty)
def aggregation():
    m,_=evaluate(label='perfect'); missing=[k for k in ['micro','macro_patient','per_patient','bias','denominators'] if k not in m]; check(not missing,'Missing reporting: '+str(missing))
test('C05_patient_micro_macro_bias','production capability',aggregation)
def macro():
    errors=[np.array([10.]),np.array([0.,0.,30.])]; micro=np.sqrt(np.mean(np.concatenate(errors)**2)); macro=np.mean([np.sqrt(np.mean(e**2)) for e in errors]); close(micro,math.sqrt(250)); close(macro,(10+math.sqrt(300))/2); check(micro!=macro)
    return {'micro_RMSE':micro,'macro_RMSE':macro,'patient_N':[1,3],'missing_patient':'null,N=0'}
test('C05_unequal_patient_oracle','oracle',macro)
def horizons():
    r=payload(); r.output+=torch.arange(1,13,dtype=torch.float64)[None,:,None]; m,_=evaluate(r,label='horizons')
    for h in range(5,61,5): check(f'val_mae_{h}m_mg_dl' in m,f'Missing horizon {h}'); close(m[f'val_mae_{h}m_mg_dl'],h/5)
test('C06_all12_horizons','production contract',horizons)
def strata():
    y=np.array([53.9,54,69.9,70,180,180.1,250,250.1]); bins=np.select([y<54,y<70,y<=180,y<=250],[0,1,2,3],4); check(bins.tolist()==[0,1,1,2,2,3,3,4]); return {'values':y.tolist(),'fine_strata':bins.tolist(),'N':[1,2,2,2,1]}
test('C07_strata_boundaries','oracle',strata)
test('C07_strata_reporting','production capability',lambda:check('per_range' in evaluate(label='strata')[0],'No target-defined strata or N in evaluator return'))

def persistence():
    m,_=evaluate(label='persistence_ramp'); ix=ds.x_to_index(base.x); lookup=val_df.set_index(['subject_id','time_idx']).glucose_mg_dl
    pred=np.array([lookup.loc[(str(s),int(t)-1)] for s,t in zip(ix.subject_id,ix.time_idx)]); y=base.y[0][:,11].numpy(); close(m['persistence_mae_60m_mg_dl'],np.mean(np.abs(pred-y)),True)
    return {'N':m['n_persistence_lookups_used'],'raw_mgdl_MAE':m['persistence_mae_60m_mg_dl']}
test('C08_persistence_two_subject_ramp','production',persistence)
def ffill():
    f=val_df.copy(); mask=(f.subject_id=='synthetic_A')&(f.time_idx==47); previous=float(f.loc[(f.subject_id=='synthetic_A')&(f.time_idx==46),'glucose_mg_dl'].iloc[0]); f.loc[mask,'glucose_mg_dl']=previous; f.loc[mask,'target_observed']=False
    m,_=evaluate(frame_=f,label='ffill'); check(m['n_persistence_lookups_used']==12); return {'N':12,'origin_age':1,'source_value':previous,'age_not_exported':True}
test('C08_ffill_origin','production',ffill)
def missing_lookup():
    # Perturb only returned prediction index: valid source frame retains its dense grid.
    r=payload(); r.x['decoder_time_idx'][0]+=1000
    return must_reject(lambda:evaluate(r,label='missing_lookup'))
test('C08_missing_one_of12','controlled prediction boundary',missing_lookup)
def no_index():
    r=payload(); del r.x['decoder_time_idx']; return must_reject(lambda:evaluate(r,label='missing_index'))
test('C08_missing_decoder_index','controlled prediction boundary',no_index)

def arima_failure():
    with patch.object(p,'_STATSMODELS_AVAILABLE',True),patch.object(p,'_ARIMA',side_effect=RuntimeError('synthetic fail'),create=True):
        check(p._compute_arima_baseline(np.arange(48.),12) is None)
    return 'helper returns None on controlled error (reason/convergence not exported)'
test('C09_arima_helper_failure','production',arima_failure)
def arima_subset():
    calls=[]; r=payload(); r.output[0]+=30; r.output[7]+=20
    def mock(window,h):
        i=len(calls); calls.append(window.copy()); return None if i in [0,7] else np.repeat(window[-1],h)
    m,_=evaluate(r,arima=mock,label='arima_subset'); check(m.get('n_arima_samples')==10); close(m['model_mard_arima_subset'],0)
    ix=ds.x_to_index(r.x)
    for i,(sid,t) in enumerate(zip(ix.subject_id,ix.time_idx)):
        wanted=val_df.loc[(val_df.subject_id==sid)&val_df.time_idx.between(int(t)-48,int(t)-1),'glucose_mg_dl'].to_numpy(); close(calls[i],wanted)
    return {'successful_indices':[i for i in range(12) if i not in [0,7]],'causal_mgdl':True}
test('C09_arima_noncontiguous_success','production with helper mock',arima_subset)

def quantile_axis():
    r=payload(); q=[.5,.02,.1,.25,.75,.9,.98]; r.output[:,:,0]=r.y[0]; r.output[:,:,3]=r.y[0]+30
    return must_reject(lambda:evaluate(r,quantiles=q,label='wrong_quantiles'))
test('C10_model_quantile_mismatch','production contract',quantile_axis)
def endpoints():
    q=cfg['quantiles']; check(.05 not in q and .95 not in q); close([b-a for a,b,_ in cfg['intervals']],[.5,.8,.96]); return '50/80/96%; 90% unavailable without new method'
test('C10_interval_endpoint_oracle','oracle',endpoints)
def crossing():
    z=np.array([[0,1,2],[1,1,2],[3,1,2.]]); c=z[:,:-1]>z[:,1:]; close(c.any(1).mean(),1/3); close(c.mean(),1/6); close(np.maximum(z[:,:-1]-z[:,1:],0).max(),2); return {'any':1/3,'adjacent':1/6,'max_magnitude':2}
test('C11_crossing_ties_oracle','oracle',crossing)
def crossing_prod():
    r=payload(); r.output[:,:,2]=r.y[0]+10; r.output[:,:,4]=r.y[0]-10; m,_=evaluate(r,label='crossed'); check(any('crossing' in k for k in m),'Raw crossing unreported; inverted interval not invalidated')
test('C11_crossing_reporting','production capability',crossing_prod)
def intervals():
    y=np.array([1,2,3,4.]); lo=np.array([1,1,1,1.]); hi=np.array([3,3,3,3.]); close(np.mean((lo<=y)&(y<=hi)),.75); close(np.mean(hi-lo),2); close(np.mean(y<=hi),.75)
    z=np.array([2,2,2,2.]); u=y-z; close(np.mean(u*(.25-(u<0))),.375); return {'inclusive_PICP':.75,'width':2,'hit':.75,'pinball_q25':.375,'inverted_pair':'INVALID, original denominator retained'}
test('C12_interval_pinball_oracle','oracle',intervals)
test('C12_interval_reporting','production capability',lambda:check(any('picp' in k.lower() or 'interval_width' in k for k in evaluate(label='intervals')[0]),'Only marginal q hits at60m; PICP/width/pinball missing'))

def weighted():
    y=(np.arange(14000,dtype=float)+.5)/100; w=np.where(y<70,2.5,1.)
    def objective(a,weights=w,factor=1): return np.mean(weights*np.abs(y-a)*.5)*factor
    grid=np.arange(0,141,dtype=float); weighted_arg=int(np.argmin([objective(a) for a in grid])); plain_arg=int(np.argmin([objective(a,np.ones_like(w)) for a in grid])); factor_arg=int(np.argmin([objective(a,factor=2) for a in grid])); check((weighted_arg,plain_arg,factor_arg)==(49,70,49))
    target=torch.tensor([[50.,70.,100.]],dtype=torch.float64); pred=torch.tensor([[[60.]*7,[80.]*7,[80.]*7]],dtype=torch.float64); u=target.numpy()[:,:,None]-pred.numpy(); pin=np.maximum(np.array(cfg['quantiles'])*u,(np.array(cfg['quantiles'])-1)*u).mean(-1); expected=2*pin*np.where(target.numpy()<70,2.5,1); close(p.ClinicalQuantileLoss(quantiles=cfg['quantiles']).loss(pred,target).numpy(),expected)
    return {'weighted_median':49,'unweighted_median':70,'factor2_median':49,'loss_matches':True,'empirical_trained_effect':'NOT EVALUATED'}
test('C13_weighted_objective','oracle + production loss',weighted)
def atoms():
    y=np.array([0.,1.]); value=lambda a:np.abs(y-a).mean()*.5; close([value(a) for a in [0,.25,.5,.75,1]],[.25]*5); return 'atom plateau is a set of minimizers, not one forced quantile'
test('C13_discrete_quantile_plateau','oracle',atoms)
def cqr():
    def radius(scores,alpha):
        k=math.ceil((len(scores)+1)*(1-alpha)); return math.inf if k>len(scores) else sorted(scores)[k-1]
    check(math.isinf(radius([1,2],.1))); close(radius([-2,-1,-1,3],.4),-1); close(radius([1,1,1,1],.4),1)
    return {'small_n':'infinity','ties':1,'negative_radius':-1,'method':'oracle only; no production fit'}
test('C14_CQR_order_statistic','oracle',cqr)
def dependence():
    keys=captures['keys']; unique={(k['subject'],k['target_timestamp']) for k in keys}; check(len(keys)==144 and len(unique)==34); return {'occurrences':144,'windows':12,'unique_subject_target_timestamps':34,'subjects':2,'IID':False}
test('C15_overlap_denominators','oracle on real PF keys',dependence)
def selection():
    decoded=ds.decoded_index
    def run(indices):
        f=val_df.copy(); f.loc[(f.subject_id=='synthetic_B')&f.time_idx.isin(indices),'target_observed']=False; return ow.window_validity(f,decoded)
    keep,stats=run([59]); check(stats['retained_windows']==6); check(stats['excluded_target_occurrences']==72); check(stats['nonobserved_target_occurrences']==6)
    independent=val_df.copy(); independent.loc[independent.time_idx==59,'target_observed']=False; ik,ist=ow.window_validity(independent,decoded); check(not ik.any())
    return {'level_dependent_missingness':stats,'symmetric_control':ist,'observed_targets_discarded':66,'real_bias':'NOT EVALUATED'}
test('C16_missingness_selection','synthetic production window filter',selection)
def clinical():
    m,log=evaluate(label='clinical'); check('clinically SAFE' not in log and 'clinical accuracy target MET' not in log,'Perfect invented predictions trigger clinical accuracy target MET and clinically SAFE')
test('C17_clinical_language','production logs',clinical)
def provenance():
    m,_=evaluate(label='provenance'); check(all(k in m for k in ['checkpoint_sha256','window_keys_sha256','prediction_sha256','schema','quantiles','split_role']),'evaluate accepts arbitrary model/args, returns no linked provenance manifest')
test('C18_evaluation_provenance','production capability',provenance)
def invariant_weights():
    check(all(torch.equal(initial[k],v) for k,v in model.state_dict().items()),'model state changed'); check(all(v.grad is None for v in model.parameters())); return {'forward_calls':forwards,'optimizer_steps':0,'backward':0,'state_unchanged':True,'effective_profile':numerics.verify('cpu')}
test('C19_no_updates_final','audit isolation',invariant_weights)
# Preserve exact observations and all failures, not expected-failure annotations.
dump('probe_captures.json',captures | {'real':'stored in real_predictions.npz','keys':'window_keys.json'})
dump('results.json',results); dump('guard_events.json',events)
counts={s:sum(r['status']==s for r in results) for s in ['PASS','FAIL','ERROR','SKIP','XFAIL']}
manifest={'scope':'SYNTHETIC C-core','config_sha256':digest(CONFIG.read_bytes()),'runner_sha256':digest(Path(__file__).read_bytes()),'source_commit':cfg['commit'],'command':sys.argv,'environment':{'python':sys.version,'torch':torch.__version__,'pandas':pd.__version__,'numpy':np.__version__,'lightning':pl.__version__,'pytorch_forecasting':pf.__version__,'statsmodels_available':p._STATSMODELS_AVAILABLE},'counts':counts,'forward_calls':forwards,'full_windows_used':len(ds),'optimizer_steps':0,'backward':0,'elapsed_seconds':time.monotonic()-start,'real_data_opened':False,'model_state_unchanged':all(torch.equal(initial[k],v) for k,v in model.state_dict().items()),'normalizer':registry.semantic(train_ds.target_normalizer),'numerical_profile':numerics.effective(),'exit_code':1 if counts['FAIL'] or counts['ERROR'] else 0,'source_hashes_match':all(digest((REPO/k).read_bytes())==v for k,v in cfg['code_hashes'].items())}
manifest['artifact_hashes']={str(f.relative_to(OUT)):digest(f.read_bytes()) for f in OUT.iterdir() if f.is_file()}
dump('manifest.json',manifest); print(json.dumps(manifest['counts']),flush=True); signal.alarm(0); sys.exit(manifest['exit_code'])
