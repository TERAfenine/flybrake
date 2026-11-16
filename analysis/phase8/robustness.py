"""Fixed-checkpoint fault-frequency evaluation. No fitting or threshold tuning."""
from pathlib import Path
import sys,json,time,platform,importlib.metadata,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch,joblib
from src.model import SignedConnectome
from src.data import load_occ
from src.evaluation import decision_metrics
from src.audit import sha256

BUNDLE=ROOT/'results/phase7_20261105'
DEFAULT_OUT=ROOT/'results/phase8_20261116'
MODELS=['logistic','banc_signed','rewired_signed']
RATES=[0.,.1,.2,.3]
FAULTS=['missing_light','co2_spike']
METRICS=['false_alarm_per_empty','unsafe_false_empty_per_occupied','coverage','abstain_rate','f1_decided_only','recall_decided_only','false_alarm_decided_empty','miss_decided_occupied','empty_coverage','occupied_coverage']

def inject(x,fault,rate,seed):
    if fault not in FAULTS or rate not in RATES:raise ValueError('Unsupported condition')
    mask=np.zeros(len(x),bool);mask[np.random.default_rng(seed).permutation(len(x))[:int(np.floor(rate*len(x)))]]=True
    raw=x.copy()
    if fault=='missing_light':raw[mask,2]=np.nan
    else:raw[mask,3]+=3000
    return raw,mask

def decisions(score,invalid,gate):
    return np.where(invalid,-1,np.where(score>=.8,1,np.where(score<=.2,0,-1))) if gate else (score>=.5).astype(int)

def run(out,models=MODELS,train_seeds=(42,43,44,45,46),attack_seeds=(700,701,702),faults=FAULTS,rates=RATES,progress=None):
    out=Path(out).resolve()
    if not out.is_relative_to(ROOT/'results') or out.exists():raise ValueError('Use a new directory under results')
    if not models or not set(models)<=set(MODELS) or not set(train_seeds)<=set(range(42,47)) or not attack_seeds:raise ValueError('Invalid selection')
    meta=json.loads((BUNDLE/'manifest.json').read_text())
    recorded={k.replace('\\','/'):v for k,v in meta['files'].items()}
    for package in ['scikit-learn','numpy','torch','joblib']:
        if meta['environment']['versions'][package]!=importlib.metadata.version(package):raise RuntimeError(f'Checkpoint environment mismatch: {package}')
    for f in ['data/datatraining.txt','data/datatest2.txt','data/banc_graph.npz','src/model.py','src/data.py','src/evaluation.py']:
        if sha256(ROOT/f)!=recorded[f]:raise RuntimeError('Source changed: '+f)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    x,y=load_occ(ROOT/'data/datatest2.txt');prep=joblib.load(BUNDLE/'sensor_prep.joblib')
    graphs=np.load(BUNDLE/'graphs.npz',allow_pickle=False)
    jobs=[(model,s,r) for model in models for s in train_seeds for r in ([1042,1043,1044] if model=='rewired_signed' else [None])]
    out.mkdir(parents=True);rows=[];checkpoints={};start=time.perf_counter();normal_checks=0
    old=pd.read_csv(BUNDLE/'metrics.csv')
    for j,(name,s,r) in enumerate(jobs):
        rid=f'{name}_t{s}'+(f'_r{r}' if r else '')
        checkpoint=BUNDLE/rid/('model.joblib' if name=='logistic' else 'model.pt')
        checkpoints[rid]=sha256(checkpoint)
        if name=='logistic':model=joblib.load(checkpoint)
        else:
            model=SignedConnectome(graphs['mask'] if r is None else graphs[f'rewired_{r}'],graphs['signs'])
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
        arrays={}
        for attack in attack_seeds:
            for fault in faults:
                for rate in rates:
                    raw,mask=inject(x,fault,rate,attack);frame,invalid=prep.transform(raw)
                    with torch.no_grad():score=model.predict_proba(frame)[:,1] if name=='logistic' else torch.sigmoid(model(torch.as_tensor(frame))).numpy()
                    key=f'a{attack}_{fault}_p{int(rate*100)}'
                    arrays[key+'_score']=score;arrays[key+'_mask']=mask
                    for gate in [False,True]:
                        d=decisions(score,invalid,gate);arrays[key+f'_decision{int(gate)}']=d.astype(np.int8)
                        m=decision_metrics(y,d)
                        if rate==0:
                            prior=old[old.run_id.eq(rid)&old.scenario.eq('normal')&old.safety_gate.eq(gate)].iloc[0]
                            for metric in METRICS:assert np.isclose(m[metric],prior[metric],atol=1e-12,equal_nan=True)
                            normal_checks+=1
                        rows.append(dict(model=name,train_seed=s,rewire_seed=r,attack_seed=attack,fault=fault,fault_rate=rate,
                            actual_fault_rate=mask.mean(),safety_gate=gate,n_total=len(y),n_corrupted=int(mask.sum()),
                            n_abstained=int((d<0).sum()),n_decided_empty=int(((y==0)&(d>=0)).sum()),n_decided_occupied=int(((y==1)&(d>=0)).sum()),
                            mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),checkpoint_sha256=checkpoints[rid],**m))
        np.savez_compressed(out/(rid+'_predictions.npz'),y_true=y,row_index=np.arange(len(y)),**arrays)
        if progress:progress((j+1)/len(jobs),rid)
    df=pd.DataFrame(rows);df.to_csv(out/'metrics.csv',index=False)
    manifest=dict(source_bundle=str(BUNDLE.relative_to(ROOT)),models=list(models),train_seeds=list(train_seeds),attack_seeds=list(attack_seeds),
        faults=list(faults),rates=list(rates),rewire_seeds=[1042,1043,1044],selection='nested permutation prefix floor(rate*N)',
        test_sha256=sha256(ROOT/'data/datatest2.txt'),prep_sha256=sha256(BUNDLE/'sensor_prep.joblib'),checkpoints=checkpoints,
        script_sha256=sha256(__file__),python=platform.python_version(),versions=meta['environment']['versions'],
        elapsed_seconds=time.perf_counter()-start,normal_checks=normal_checks,finished_utc=pd.Timestamp.now(tz='UTC').isoformat())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out/'COMPLETE.json').write_text(json.dumps(dict(rows=len(df),runs=len(jobs))),encoding='utf-8')
    return df

def summarize(df):
    # Average rewires, then fault masks within each train seed; SD across train seeds.
    keys=['model','fault','fault_rate','safety_gate']
    within=df.groupby(keys+['train_seed'])[METRICS].agg(lambda v:v.mean() if v.notna().all() else np.nan).reset_index()
    rows=[]
    for group,g in within.groupby(keys):
        for metric in METRICS:
            v=g[metric];ok=v.notna().all()
            rows.append(dict(zip(keys,group),metric=metric,mean=v.mean() if ok else np.nan,std=v.std(ddof=1) if ok and len(v)>1 else np.nan,n_valid=int(v.notna().sum()),n_train=len(v)))
    return pd.DataFrame(rows)

def analyze(df,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summary=summarize(df);summary.to_csv(out/'summary.csv',index=False)
    keys=['model','train_seed','rewire_seed','attack_seed','fault','safety_gate']
    base=df[df.fault_rate.eq(0)]
    joined=df.merge(base,on=keys,suffixes=('','_normal'),validate='many_to_one')
    delta=joined[keys+['fault_rate']].copy()
    for m in METRICS:delta[m]=100*(joined[m]-joined[m+'_normal'])
    delta.to_csv(out/'normal_deltas_pp.csv',index=False)
    for attack,g in df.groupby('attack_seed'):
        summarize(g).assign(attack_seed=attack).to_csv(out/f'attack_{attack}_summary.csv',index=False)
    pairs=[]
    a=df[df.model.eq('banc_signed')]
    for other in ['logistic','rewired_signed']:
        b=df[df.model.eq(other)]
        if a.empty or b.empty:continue
        joined=a.merge(b,on=['train_seed','attack_seed','fault','fault_rate','safety_gate'],suffixes=('_a','_b'),validate='one_to_many')
        p=joined[['train_seed','attack_seed','fault','fault_rate','safety_gate']].copy();p['other']=other;p['rewire_seed']=joined.rewire_seed_b
        for m in METRICS:p[m]=100*(joined[m+'_a']-joined[m+'_b'])
        pairs.append(p)
    if pairs:pd.concat(pairs).to_csv(out/'paired_differences_pp.csv',index=False)
    return summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(DEFAULT_OUT));a=p.parse_args()
    frame=run(a.out,progress=lambda v,r:print(f'{v:.0%} {r}',flush=True))
    analyze(frame,ROOT/'analysis/phase8/tables')
