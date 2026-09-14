"""Guarded synthetic acceptance. Run only via run_stage_c_remediation.py.

No historical runner edits, no unittest discovery of training suites.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]).resolve()
assert OUT.parent == ROOT/'ml/models/stage_c_remediation_20260914'
CAMPAIGN = OUT.parent/'campaign.json'
EVENTS = []

def dump(name, value):
    with (OUT/name).open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())

def deny(label):
    def blocked(*args, **kwargs):
        EVENTS.append(label); dump('guards.json', EVENTS)
        raise PermissionError(label)
    return blocked

def guard(event, args):
    if event == 'import' and args[0].split('.')[0] == 'optuna':
        deny('Optuna')()
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(OUT) or path == CAMPAIGN:
            return
        if path.is_relative_to(ROOT/'ml/data') or path.is_relative_to(ROOT/'ml/models'):
            deny('forbidden data/model open')()
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR)):
            if str(path) not in ('/dev/null',):
                deny('write outside owned run')()
        if path.suffix.lower() in ('.parquet', '.xml', '.csv', '.pt', '.ckpt', '.npz', '.npy') and not path.is_relative_to(ROOT/'.venv'):
            deny('unlisted data artifact')()

sys.addaudithook(guard)
for name in ('ml/data/processed/forbidden.parquet', 'ml/data/raw/forbidden.xml', 'ml/models/foreign.ckpt'):
    try:
        open(ROOT/name, 'rb')
        raise AssertionError('guard allowed open')
    except PermissionError:
        pass
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
import pytorch_forecasting as pf
from ml.scripts import train_tft_population_v2 as production
from ml.scripts import evaluation_stage_c as e
from ml.scripts import numerical_profile as numerical
from ml.scripts import checkpoint_registry as registry

for name in ('main', 'train', 'train_experimental_legacy', 'train_baseline', 'load_and_preprocess_data', 'load_test_data'):
    setattr(production, name, deny(name))
pl.Trainer.fit = deny('Trainer.fit')
torch.Tensor.backward = deny('Tensor.backward')
torch.autograd.backward = deny('autograd.backward')
for cls in (torch.optim.Optimizer, torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD, production.CheckedAdamW):
    cls.step = deny(cls.__name__+'.step')
torch.set_num_threads(1); torch.manual_seed(42); np.random.seed(42); numerical.configure('cpu')
RESULTS = []
def test(name, fn):
    start = time.monotonic()
    try:
        evidence = fn(); row = dict(id=name, status='PASS', evidence=evidence)
    except AssertionError:
        row = dict(id=name, status='FAIL', traceback=traceback.format_exc())
    except Exception:
        row = dict(id=name, status='ERROR', traceback=traceback.format_exc())
    row['seconds'] = time.monotonic()-start; RESULTS.append(row)
    dump('results.json', RESULTS); print(name, row['status'], flush=True)

def close(a, b, fp32=False):
    np.testing.assert_allclose(a, b, atol=1e-5 if fp32 else 1e-10, rtol=1e-5 if fp32 else 1e-8)

def rejects(code, fn, stage=None):
    try: fn()
    except e.EvaluationError as ex:
        assert ex.code == code, str(ex)
        if stage: assert ex.stage == stage, str(ex)
        return
    raise AssertionError('expected '+code)


def guards():
    for fn in (production.main, production.load_test_data, pl.Trainer.fit, torch.Tensor.backward, torch.optim.AdamW.step):
        try: fn(); raise AssertionError('guard failed')
        except PermissionError: pass
    return EVENTS

test('C00_guards', guards)


def frame(role):
    rows = []
    for sid, base, slope in [('synthetic_A',75.,.8), ('synthetic_B',180.,1.5)]:
        for i in range(65):
            row = {col: float(i%7/7) for col in sorted(set(production.TIME_VARYING_KNOWN_REALS+production.TIME_VARYING_UNKNOWN_REALS))}
            row.update({f'glucose_lag_{lag}_available': 1. for lag in (1,2,3,6,12,24)})
            row.update(subject_id=sid,time_idx=i,timestamp=pd.Timestamp('2024-01-01')+pd.Timedelta(minutes=5*i), source_split=role,target_observed=True,glucose_mg_dl=base+slope*i)
            rows.append(row)
    return pd.DataFrame(rows)

train = frame('synthetic_train'); assessment = frame('synthetic_assessment')
args = SimpleNamespace(context=48, horizon=12, batch_size=5, no_gpu=True)
training, candidate = production.build_datasets(train, assessment, args)
original_index = candidate.index.copy()
selected, windows, keys, flow = e.qualify(candidate, assessment, 'synthetic_assessment')
model = production.ClinicalTFT.from_dataset(training, hidden_size=8, hidden_continuous_size=4, attention_head_size=1, lstm_layers=1, dropout=.1, loss=production.ClinicalQuantileLoss(quantiles=list(e.Q)), output_size=7, log_interval=-1)
model.eval()
original_forward = model.forward

def counted(*a, **kw):
    assert not torch.is_grad_enabled() and not model.training
    ledger = json.loads(CAMPAIGN.read_text())
    ledger['forward_batch_calls'] += 1
    actual = selected.x_to_index(a[0])
    used = {f'{r.subject_id}:{r.time_idx}' for r in actual.itertuples(index=False)}
    ledger['selected_window_keys'] = sorted(set(ledger['selected_window_keys']) | used)
    with CAMPAIGN.open('w') as f:
        json.dump(ledger, f, indent=2); f.flush(); os.fsync(f.fileno())
    assert ledger['forward_batch_calls'] <= 24 and len(ledger['selected_window_keys']) <= 64
    return original_forward(*a, **kw)
model.forward = counted
config = {'revision':e.REVISION, 'seed':42, 'context':48, 'horizon':12, 'quantiles':list(e.Q)}
receipt = e.own_synthetic_state(model, training, OUT/'synthetic_untrained.pt', config=config, code={'head':'58c2e36dc108ef3d88de888a3fffc18c63af05bb', 'dirty':True})
context = e.synthetic_context(receipt, model, candidate, assessment, split_role='synthetic_assessment', config=config)
initial = e.state_hash(model)
captured = {}
real_predict = model.predict

def capture(*a, **kw):
    result = real_predict(*a, **kw); captured['prediction'] = result; return result
model.predict = capture

def integration():
    metrics = production.evaluate(model, candidate, args, assessment, context=context, output_dir=OUT/'integration')
    assert metrics['denominators'] == {'occurrences':144,'windows':12,'unique_subject_target_timestamps':34,'subjects':2}
    assert e.readback(OUT/'integration') == metrics
    pd.testing.assert_frame_equal(candidate.index, original_index)
    return {'flow':flow, 'denominators':metrics['denominators']}

test('C01_C03_C18_real_PF_evaluate_readback', integration)


def normalizer():
    batch = next(iter(selected.to_dataloader(train=False,batch_size=12,num_workers=0,shuffle=False)))[0]
    ix = selected.x_to_index(batch)
    expected = []
    for i, row in enumerate(ix.itertuples(index=False)):
        values = np.log(train.loc[train.subject_id == row.subject_id, 'glucose_mg_dl'].to_numpy())
        stats = [values.mean(),values.std(ddof=1)+np.finfo(np.float16).eps]
        expected.append(stats)
        source = assessment[(assessment.subject_id == row.subject_id) & assessment.time_idx.between(row.time_idx-48,row.time_idx-1)]
        close(batch['encoder_cont'][i,:,selected.reals.index('glucose_mg_dl')], (np.log(source.glucose_mg_dl)-stats[0])/stats[1], True)
    close(batch['target_scale'],expected,True)
    assert len(np.unique(np.asarray(expected)[:,0])) == 2
    before = registry.semantic(training.target_normalizer)
    sentinel = train.copy(); sentinel.loc[sentinel.time_idx == 0,'target_observed'] = False
    a = production.create_time_series_dataset(sentinel)
    sentinel.loc[sentinel.time_idx == 0,'glucose_mg_dl'] = 9999
    b = production.create_time_series_dataset(sentinel)
    assert registry.semantic(a.target_normalizer) == registry.semantic(b.target_normalizer)
    altered = assessment.copy(); altered.glucose_mg_dl *= 3
    production.create_time_series_dataset(altered, reference_dataset=training)
    assert before == registry.semantic(training.target_normalizer)
    unseen = assessment[assessment.subject_id == 'synthetic_A'].copy(); unseen.subject_id = 'unseen'
    try: production.create_time_series_dataset(unseen, reference_dataset=training)
    except ValueError as ex: assert 'unseen' in str(ex)
    else: raise AssertionError('unseen accepted')
    parameters = e.load_owned_state(receipt, model, training, expected=receipt.metadata)
    assert registry.semantic(parameters['target_normalizer']) == before
    for mode in ('resume-last','weights-only'):
        rejects_legacy = dict(receipt.metadata, revision='old')
        try: registry.check_compatible(rejects_legacy, receipt.metadata, mode)
        except ValueError: pass
        else: raise AssertionError('legacy accepted')
    # Make the fallback nonfinite: known encoded IDs must still return their
    # own scales, including through actual PF __getitem__.
    norm = selected.target_normalizer
    saved_missing = copy.deepcopy(norm.missing_)
    norm.missing_ = {k: float('nan') for k in norm.missing_}
    try:
        actual = next(iter(selected.to_dataloader(train=False,batch_size=12,num_workers=0,shuffle=False)))[0]
        close(actual['target_scale'], expected, True)
    finally:
        norm.missing_ = saved_missing
    renamed = train.copy()
    renamed.subject_id = renamed.subject_id.map({'synthetic_A':'zeta','synthetic_B':'alpha'})
    renamed = renamed.iloc[::-1].copy()
    renamed_ds = production.create_time_series_dataset(renamed)
    for sid, group in renamed.groupby('subject_id'):
        encoded = renamed_ds.target_normalizer.nmd_subject_mapping[sid]
        values = np.log(group.glucose_mg_dl.to_numpy())
        close(renamed_ds.target_normalizer.get_parameters([encoded]), [values.mean(),values.std(ddof=1)+np.finfo(np.float16).eps])
    return {'scales':expected,'mapping':training.target_normalizer.nmd_subject_mapping}

test('C02_normalizer_real_batch_sentinels_roundtrip', normalizer)


def payload():
    r = captured['prediction']
    return SimpleNamespace(x={k:v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in r.x.items()}, y=(r.y[0].clone(),None), output=r.output.clone())

def alignment():
    r = payload(); r.y[0][0,0] += 1
    rejects('alignment', lambda:e.align(r,selected,windows))
    for field in ('groups','decoder_time_idx'):
        r=payload(); r.x[field][0] = r.x[field][1]
        # groups may coincide within a patient; explicitly use the other patient.
        if field == 'groups': r.x[field][0] = r.x[field][-1]
        rejects('alignment',lambda:e.align(r,selected,windows))
    r=payload();r.x['decoder_lengths'][0]=6
    rejects('length',lambda:e.align(r,selected,windows))
    r=payload();r.x['encoder_lengths'][0]=24
    rejects('length',lambda:e.align(r,selected,windows))
    r=payload();r.output=r.output[:-1]
    rejects('alignment',lambda:e.align(r,selected,windows))
    r=payload();r.x['decoder_time_idx'][0,4]+=1
    rejects('alignment',lambda:e.align(r,selected,windows))
    r=payload(); y,z=e.align(r,selected,windows)
    r.y=(r.y[0].flip(0),torch.ones_like(r.y[0])*13);r.output=r.output.flip(0)
    r.x={k:v.flip(0) if isinstance(v,torch.Tensor) else v for k,v in r.x.items()}
    yy,zz=e.align(r,selected,windows);close(yy,y);close(zz,z)

test('C01_keys_y_lengths_coherent_permutation', alignment)


def source_guards():
    assert flow['excluded']['length']==96 and len(windows)==12
    f=assessment.copy();f.loc[(f.subject_id=='synthetic_B') & (f.time_idx==59),'target_observed']=False
    _,w,k,s=e.qualify(candidate,f,'synthetic_assessment');assert len(w)==6 and len(k)==72
    f=assessment.copy();f.loc[f.time_idx==59,'source_split']='other'
    rejects('role',lambda:e.qualify(candidate,f,'synthetic_assessment'))
    f=assessment.copy();f.loc[f.time_idx==59,'timestamp']+=pd.Timedelta(hours=2)
    rejects('alignment',lambda:e.qualify(candidate,f,'synthetic_assessment'))
    for value in (float('nan'),float('inf'),-float('inf')):
        f=assessment.copy();f.loc[f.time_idx==59,'glucose_mg_dl']=value
        rejects('finite',lambda:e.qualify(candidate,f,'synthetic_assessment'))
    f=assessment.copy();f.loc[f.time_idx==59,'glucose_mg_dl']=0
    rejects('target_domain',lambda:e.qualify(candidate,f,'synthetic_assessment'))
    pd.testing.assert_frame_equal(candidate.index,original_index)

test('C03_source_roles_missingness_full48',source_guards)


def finite_boundaries():
    for value in (float('nan'),float('inf'),-float('inf')):
        for j in range(7):
            r=payload();r.output[0,0,j]=value
            rejects('finite',lambda:e.align(r,selected,windows))
        r=payload();r.y[0][0,0]=value
        rejects('finite',lambda:e.align(r,selected,windows))
    rejects('finite',lambda:e.reduce_metrics([1.],[1e308]))

test('C03_all_q_finite_domain',finite_boundaries)


def scalar_metrics():
    m=e.reduce_metrics([50,100],[60,80]);close(list(m['point'].values()),[15,math.sqrt(250),-5,20])
    e.atomic_json(OUT/'precision.json',m);assert json.loads((OUT/'precision.json').read_text())==m
    close(e.reduce_metrics([20,400,401]+[100]*9,[30,390,390]+[100]*9)['point']['mae'],31/12)
    close(e.reduce_metrics([.1]+[100]*11,[1]+[100]*11)['point']['mard'],75)
    for p in (0,-10):
        m=e.reduce_metrics([100],[p]);assert m['N']==1 and m['nonpositive_predictions']==1
    for y in (0,-1): rejects('target_domain',lambda:e.reduce_metrics([y],[1]))
    assert e.reduce_metrics([],[])['status']=='EMPTY'

test('C03_C04_scalar_full_precision',scalar_metrics)


def baseline_lineage():
    for age in (0,1,6):
        f=assessment.copy();w=copy.deepcopy(windows[0]);origin=w['decoder']-1
        value=float(f[(f.subject_id==w['subject']) & (f.time_idx==origin-age)].glucose_mg_dl.iloc[0])
        mask=(f.subject_id==w['subject']) & f.time_idx.between(origin-age+1,origin)
        f.loc[mask,'target_observed']=False;f.loc[mask,'glucose_mg_dl']=value
        ds=production.create_time_series_dataset(f,reference_dataset=training)
        _,ww,_,_=e.qualify(ds,f,'synthetic_assessment')
        rebuilt=next(x for x in ww if x['window_id']==w['window_id'])
        b,l=e.persistence([rebuilt],f);close(b,[value]*12);assert l[0]['age_bins']==age and l[0]['ffill']==bool(age)
        assert pd.Timestamp(l[0]['source_timestamp'])==pd.Timestamp(w['origin_timestamp'])-pd.Timedelta(minutes=5*age)
    f=assessment[~((assessment.subject_id==windows[0]['subject']) & (assessment.time_idx==windows[0]['decoder']-1))]
    rejects('persistence',lambda:e.persistence(windows,f))
    for age in (7,48):
        f=assessment.copy();w=copy.deepcopy(windows[0]);origin=w['decoder']-1
        mask=(f.subject_id==w['subject']) & f.time_idx.between(max(0,origin-age+1),origin)
        f.loc[mask,'target_observed']=False
        rejects('persistence',lambda:e.persistence([w],f))
    w=copy.deepcopy(windows[0]);w['encoder_end']+=1
    rejects('persistence',lambda:e.persistence([w],assessment))

test('C08_complete_persistence_lineage',baseline_lineage)


def probability():
    y=np.array([1,2,3,4.]);z=np.tile([0,1,2,2,3,3,5.],(4,1))
    m=e.reduce_metrics(y,z[:,3],z)['probabilistic']
    close(m['quantiles']['0.25']['pinball'],.375);close(m['quantiles']['0.75']['hit'],.75)
    close(m['intervals']['80']['picp'],.75);close(m['intervals']['80']['width'],2)
    for level in (50,80,96): assert m['intervals'][str(level)]['N']==4
    rejects('quantiles',lambda:e.interval_pair(90))
    for q in ([.5,*e.Q[1:]],list(reversed(e.Q)),[0,*e.Q[1:]],e.Q[:-1]): rejects('quantiles',lambda:e.quantiles(q))
    z[:]=2;z[0,2]=4;z[0,4]=1
    m=e.reduce_metrics(y,z[:,3],z)['probabilistic']
    close(m['crossing']['any_rate'],.25);close(m['crossing']['adjacent_rate'],2/24);close(m['crossing']['max_magnitude'],2)
    assert m['intervals']['50']['status']=='INVALID' and m['intervals']['50']['N_invalid']==1

test('C10_C11_C12_quantiles_intervals_crossing',probability)


def aggregation():
    kk=[dict(subject=s,window_id=f'{s}:{i}',origin_timestamp='2024-01-01T00:00:00',target_timestamp='2024-01-01T00:05:00',horizon_minutes=5) for i,s in enumerate(['A','B','B','B'])]
    y=np.ones(4)*100;z=np.repeat((y+[10,0,0,30])[:,None],7,axis=1)
    m=e.report_metrics(kk,y,z,y,['A','B','missing'])
    p=m['panels']['5']['all']['tft']
    close(p['micro']['point']['rmse'],math.sqrt(250));close(p['macro_patient']['point']['rmse'],(10+math.sqrt(300))/2)
    assert p['macro_patient']['missing_subjects']==['missing']
    assert m['denominators']['occurrences']==4 and m['denominators']['unique_subject_target_timestamps']==2
    z[0,2]=200;z[0,4]=50
    m=e.report_metrics(kk,y,z,y,['A','B','missing'])
    assert m['panels']['5']['all']['tft']['macro_patient']['probabilistic']['intervals']['50']['status']=='INVALID'
    bounds=np.array([53.9,54,69.9,70,180,180.1,250,250.1])
    kk=[dict(subject='A',window_id=str(i),origin_timestamp='t',target_timestamp=str(i),horizon_minutes=h) for h in e.HORIZONS for i in range(8)]
    y=np.tile(bounds,12);z=np.repeat(y[:,None],7,axis=1)
    m=e.report_metrics(kk,y,z,y,['A','missing'])
    for h in e.HORIZONS:
        panels=m['panels'][str(h)]
        assert [panels[n]['denominators']['occurrences'] for n in ['five:lt54','five:54to70','five:70to180','five:180to250','five:gt250']]==[1,2,2,2,1]
        assert sum(p['denominators']['occurrences'] for n,p in panels.items() if n.startswith('three:'))==8
        for panel in panels.values():
            assert panel['tft']['micro']['N']==panel['persistence']['micro']['N']
            assert len(panel['tft']['micro']['probabilistic']['quantiles'])==7
            assert panel['persistence']['micro']['probabilistic']['status']=='N/A'

test('C05_C06_C07_all_panels_macro_overlap_ranges',aggregation)



def complete_oracle_matrix():
    records = [(sid, i, h, y) for sid, ys in [('A',[53.9,54,69.9,70,180,180.1,250,250.1]),('B',[54,100,100,251])]
               for i,y in enumerate(ys) for h in e.HORIZONS]
    kk = [dict(subject=s,window_id=f'{s}:{i}',origin_timestamp='origin',target_timestamp=f'{i}:{h}',horizon_minutes=h) for s,i,h,y in records]
    yy = np.array([y for s,i,h,y in records]); offsets=np.array([h/5+i for s,i,h,y in records])
    zz=yy[:,None]+offsets[:,None]+np.array([-20,-10,-5,0,5,10,20])
    base=yy+3
    report=e.report_metrics(kk,yy,zz,base,['A','B','absent'])
    def oracle(indices, model_name):
        if not indices:return None
        y=yy[indices];z=zz[indices];p=z[:,3] if model_name=='tft' else base[indices]
        errors=[float(a-b) for a,b in zip(p,y)]
        point={'mae':sum(abs(v) for v in errors)/len(errors),'rmse':math.sqrt(sum(v*v for v in errors)/len(errors)),
               'bias':sum(errors)/len(errors),'mard':100*sum(abs(v)/float(t) for v,t in zip(errors,y))/len(errors)}
        return point
    def in_range(name,y):
        return {'all':True,'three:lt70':y<70,'three:70to180':70<=y<=180,'three:gt180':y>180,
                'five:lt54':y<54,'five:54to70':54<=y<70,'five:70to180':70<=y<=180,'five:180to250':180<y<=250,'five:gt250':y>250}[name]
    for h, panels in report['panels'].items():
        for name,panel in panels.items():
            indices=[i for i,(s,w,hor,y) in enumerate(records) if (h=='all' or hor==int(h)) and in_range(name,y)]
            for model_name in ('tft','persistence'):
                expected=oracle(indices,model_name)
                if expected:
                    for key,value in expected.items():close(panel[model_name]['micro']['point'][key],value)
                for sid in ('A','B','absent'):
                    subset=[i for i in indices if records[i][0]==sid]
                    want=oracle(subset,model_name)
                    if want:
                        for key,value in want.items():close(panel[model_name]['per_patient'][sid]['point'][key],value)
                    else:assert panel[model_name]['per_patient'][sid]['status']=='EMPTY'
                if indices:
                    persons=[oracle([i for i in indices if records[i][0]==sid],model_name) for sid in ('A','B')]
                    persons=[p for p in persons if p]
                    for key in expected:close(panel[model_name]['macro_patient']['point'][key],sum(p[key] for p in persons)/len(persons))
            # Independent raw pinball/hit/PICP/width at every h/range/person.
            for sid in (None,'A','B','absent'):
                ix=indices if sid is None else [i for i in indices if records[i][0]==sid]
                actual=panel['tft']['micro' if sid is None else 'per_patient']
                actual=actual if sid is None else actual[sid]
                if not ix:continue
                for j,q in enumerate(e.Q):
                    errors=[float(yy[i]-zz[i,j]) for i in ix]
                    close(actual['probabilistic']['quantiles'][str(q)]['pinball'],sum(v*(q-int(v<0)) for v in errors)/len(ix))
                    close(actual['probabilistic']['quantiles'][str(q)]['hit'],sum(yy[i]<=zz[i,j] for i in ix)/len(ix))
                for level,a,b in [(50,2,4),(80,1,5),(96,0,6)]:
                    cell=actual['probabilistic']['intervals'][str(level)]
                    close(cell['picp'],sum(zz[i,a]<=yy[i]<=zz[i,b] for i in ix)/len(ix))
                    close(cell['width'],sum(float(zz[i,b]-zz[i,a]) for i in ix)/len(ix))
    return {'panels':13*9,'subjects':['A','B','absent'],'oracle':'independent FP64 loops'}

test('C04_C05_C06_C07_C11_C12_complete_oracle_matrix',complete_oracle_matrix)


def provenance():
    for field,value,code in [('units','mmol/L','schema'),('role','BEST','role'),('quantiles',list(reversed(e.Q)),'quantiles'),('revision','old','compatibility')]:
        modified=copy.deepcopy(receipt);modified.metadata[field]=value
        c=e.EvaluationContext(modified,context.source_sha256,context.split_role,context.schema,context.normalizer,context.config_sha256)
        rejects(code,lambda:e.validate_context(c,model,candidate,assessment))
    metadata=copy.deepcopy(receipt.metadata);metadata['units']='mmol/L'
    with patch.object(torch,'load',side_effect=AssertionError('deserialized mismatch')):
        rejects('compatibility',lambda:e.load_owned_state(receipt,model,training,expected=metadata),'preload')
    c=e.EvaluationContext(receipt,context.source_sha256,'test',context.schema,context.normalizer,context.config_sha256)
    rejects('role',lambda:e.validate_context(c,model,candidate,assessment))
    c=e.EvaluationContext(receipt,context.source_sha256,context.split_role,{},context.normalizer,context.config_sha256)
    rejects('schema',lambda:e.validate_context(c,model,candidate,assessment))
    c=e.EvaluationContext(receipt,context.source_sha256,context.split_role,context.schema,{},context.config_sha256)
    rejects('compatibility',lambda:e.validate_context(c,model,candidate,assessment))
    c=e.EvaluationContext(receipt,'bad',context.split_role,context.schema,context.normalizer,context.config_sha256)
    rejects('context',lambda:e.validate_context(c,model,candidate,assessment))
    param=next(model.parameters());saved=param.detach().clone()
    with torch.no_grad():param.add_(1)
    rejects('context',lambda:e.validate_context(context,model,candidate,assessment))
    with torch.no_grad():param.copy_(saved)
    raw=OUT/'integration/raw.json';saved=raw.read_bytes();raw.write_bytes(saved+b' ')
    rejects('artifact',lambda:e.readback(OUT/'integration'))
    raw.write_bytes(saved)
    e.readback(OUT/'integration')

test('C18_context_preload_state_output_tamper',provenance)


def invalid_publication():
    r=payload();r.output[0,0,0]=float('nan')
    with patch.object(model,'predict',return_value=r):
        rejects('finite',lambda:production.evaluate(model,candidate,args,assessment,context=context,output_dir=OUT/'invalid_finite'))
    invalid=json.loads((OUT/'invalid_finite/invalid.json').read_text())
    assert invalid['planned_N']==144 and not (OUT/'invalid_finite/manifest.json').exists()
    with patch.object(e,'persistence',side_effect=e.EvaluationError('persistence','baseline','one lookup missing')),patch.object(model,'predict',return_value=payload()):
        rejects('persistence',lambda:production.evaluate(model,candidate,args,assessment,context=context,output_dir=OUT/'invalid_baseline'))
    invalid=json.loads((OUT/'invalid_baseline/invalid.json').read_text())
    assert invalid['planned_N']==144 and not (OUT/'invalid_baseline/metrics.json').exists()

test('C03_C08_invalid_publication_planned_N',invalid_publication)


def real_permuted():
    loader=selected.to_dataloader(train=False,batch_size=4,num_workers=0,shuffle=False,sampler=list(reversed(range(12))))
    with torch.no_grad():
        r=real_predict(loader,return_x=True,return_y=True,mode='quantiles',trainer_kwargs={'accelerator':'cpu','logger':False,'enable_progress_bar':False})
    y,z=e.align(r,selected,windows);yy,zz=e.align(captured['prediction'],selected,windows)
    close(y,yy,True);close(z,zz,True)
    return {'max_abs':float(np.abs(z-zz).max()),'batches':3}

test('C01_real_uneven_reversed_batches',real_permuted)


def native_finite():
    handle=model.output_layer.register_forward_hook(lambda module,inputs,out:out*float('nan'))
    try:
        try:production.evaluate(model,candidate,args,assessment,context=context,output_dir=OUT/'native_finite')
        except FloatingPointError:pass
        else:raise AssertionError('native gate accepted nonfinite')
    finally:handle.remove()
    invalid=json.loads((OUT/'native_finite/invalid.json').read_text())
    assert invalid['error_code']=='finite' and invalid['planned_N']==144

test('C03_native_model_finite_gate',native_finite)


def late_finite():
    def bad(module,inputs,out):
        out.prediction[0,0,0]=float('nan');return out
    handle=model.register_forward_hook(bad)
    try:
        rejects('finite',lambda:production.evaluate(model,candidate,args,assessment,context=context,output_dir=OUT/'late_finite'))
    finally:handle.remove()
    assert not (OUT/'late_finite/manifest.json').exists()

test('C03_real_PF_late_fault',late_finite)


def language_empty():
    f=assessment.copy();f.loc[f.time_idx==59,'target_observed']=False
    c=e.synthetic_context(receipt,model,candidate,f,split_role='synthetic_assessment',config=config)
    with patch.object(model,'predict',side_effect=AssertionError('empty forward')):
        m=production.evaluate(model,candidate,args,f,context=c,output_dir=OUT/'empty')
    assert m['denominators']['occurrences']==0 and e.readback(OUT/'empty')==m
    text=json.dumps(m)
    for claim in ('clinically SAFE','accuracy target MET','acceptable','✓'):
        assert claim not in text
    assert m['interpretation']['calibration']=={'status':'raw/not_fitted','fitted_source':None,'fitted_role':None}
    for target,pred in [(100,100),(100,-1)]:
        r=e.report_metrics(keys,np.full(144,target),np.full((144,7),pred),np.full(144,100),['synthetic_A','synthetic_B'])
        for claim in ('clinically SAFE','accuracy target MET','acceptable','✓'):assert claim not in json.dumps(r)
    for y,p,zone in [(100,100,'A'),(100,150,'B'),(100,250,'C'),(300,100,'D'),(50,250,'E')]:
        assert production.clarke_error_grid(torch.tensor([y]),torch.tensor([p]))[zone]==100

test('C14_C17_raw_language_empty_Clarke',language_empty)


def limitations():
    y=(np.arange(14000)+.5)/100;w=np.where(y<70,2.5,1)
    grid=np.arange(141)
    losses=np.array([np.mean(w*np.abs(y-a)*.5) for a in grid])
    assert grid[losses.argmin()]==49 and grid[(2*losses).argmin()]==49
    assert grid[np.array([np.mean(np.abs(y-a)*.5) for a in grid]).argmin()]==70
    for a in (0,.25,.5,.75,1):close(np.abs(np.array([0,1])-a).mean()*.5,.25)
    target=torch.tensor([[50.,70.,100.]],dtype=torch.float64)
    pred=torch.tensor([[[60.]*7,[80.]*7,[80.]*7]],dtype=torch.float64)
    u=target.numpy()[:,:,None]-pred.numpy()
    expected=2*np.mean(u*(np.asarray(e.Q)-(u<0)),axis=-1)*np.where(target.numpy()<70,2.5,1)
    close(production.ClinicalQuantileLoss(quantiles=list(e.Q)).loss(pred,target).numpy(),expected)
    def radius(scores,alpha):
        k=math.ceil((len(scores)+1)*(1-alpha));return math.inf if k>len(scores) else sorted(scores)[k-1]
    assert math.isinf(radius([1,2],.1));assert radius([-2,-1,-1,3],.4)==-1;assert radius([1,1,1,1],.4)==1
    return {'C13-F01':'ACCEPTED LIMITATION','C14-F01':'NOT IMPLEMENTED/DEFERRED','calibration_fit':'NOT IMPLEMENTED'}

test('C13_C14_loss_algebra_CQR_definition_only',limitations)

def no_updates():
    assert initial==e.state_hash(model)
    assert all(p.grad is None for p in model.parameters())
    return {'state_sha256':initial,'state_unchanged':True,'backward':0,'optimizer_steps':0}

test('C19_state_no_updates',no_updates)
dump('environment.json',{'python':sys.version,'torch':torch.__version__,'lightning':pl.__version__,'pytorch_forecasting':pf.__version__,'numpy':np.__version__,'pandas':pd.__version__,'numerics':numerical.verify('cpu'),'candidate_index_rows':len(candidate)})
counts={status:sum(r['status']==status for r in RESULTS) for status in ('PASS','FAIL','ERROR','SKIP','XFAIL')}
dump('summary.json',counts)
sys.exit(int(bool(counts['FAIL'] or counts['ERROR'])))
