"""Isolated Phase 7 CPU experiment; immutable original data/code, explicit seeds."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, platform, sys, time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
import joblib
from src.model import SignedConnectome
from src.data import load_occ, SensorPrep, augment, sensor_attack
from src.connectome import rewire_preserving_degrees
from src.evaluation import decision_metrics, select_exact
from src.audit import audit_split, sha256

SCENARIOS=['normal','missing_light','co2_spike']
TARGETS=[.25,.5,.65,.75,.9,1.]
CONFIG=dict(protocol='phase7_exploratory_v1',train_seeds=[42,43,44,45,46],
    rewire_seeds=[1042,1043,1044],graph_seed=None,graph_selection='fixed original NPZ; graph_seed not used',
    augmentation_seed=4242,attack_seed=42,validation_attack_seed=43,tie_seed=2026,
    batch_seed_offset=2000,epochs=18,patience=5,min_delta=.0001,batch_size=256,
    learning_rate=.006,weight_decay=.0001,clip_grad_norm=2.,torch_threads=4,
    mlp_hidden=196,activation='tanh',device='cpu')


class MatchedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(8,196),nn.Tanh(),nn.Linear(196,1))
    def forward(self,x):return self.layers(x).squeeze(-1)


def capacities(mask,steps=4):
    """Structural paths to final readout, allowing learned nonzero recurrent bias."""
    n=len(mask);nin=nout=max(4,n//4)
    output=np.zeros(n,bool);output[-nout:]=True
    reach=[output]
    for _ in range(steps-1):reach.append(reach[-1] | (mask[reach[-1],:]>0).any(axis=0))
    # h0=0: recurrent weights can contribute at steps 2..T only.
    recurrent=int((mask[reach[steps-2],:]>0).sum()) if steps>1 else 0
    bias=int(reach[-1].sum());inputs=int(reach[-1][:nin].sum())
    return dict(allocated=n*n+n+nin*9+nout+1,mask_active=int(mask.sum())+n+nin*9+nout+1,
                output_effective=recurrent+bias+inputs*9+nout+1,
                effective_recurrent=recurrent,effective_bias=bias,effective_input_neurons=inputs,
                backward_reach_sizes=[int(x.sum()) for x in reach],
                dense_matmul_macs=8*nin+steps*n*n+nout,
                masked_matmul_macs=8*nin+steps*int(mask.sum())+nout)


def initialize(model_name,train_seed,mask,signs):
    torch.manual_seed(train_seed)
    return MatchedMLP() if model_name=='mlp_matched' else SignedConnectome(mask,signs)


def load_training():
    X,y=load_occ(ROOT/'data/datatraining.txt');cut=int(.8*len(X))
    prep=SensorPrep().fit(X[:cut])
    Ftr,_=prep.transform(augment(X[:cut],np.random.default_rng(CONFIG['augmentation_seed'])))
    Fval,_=prep.transform(X[cut:])
    return prep,Ftr,y[:cut],Fval,y[cut:],X[cut:]


def fit_network(model,Ftr,ytr,Fval,yval,seed,epochs=18):
    start=time.perf_counter()
    optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(max(1,int((ytr==0).sum())/max(int((ytr==1).sum()),1)),dtype=torch.float32))
    x=torch.as_tensor(Ftr);y=torch.tensor(ytr,dtype=torch.float32)
    vx=torch.as_tensor(Fval);vy=torch.tensor(yval,dtype=torch.float32)
    rng=np.random.default_rng(seed+CONFIG['batch_seed_offset'])
    best=float('inf');stale=0;history=[]
    for epoch in range(epochs):
        model.train();order=rng.permutation(len(Ftr))
        for j in range(0,len(order),CONFIG['batch_size']):
            idx=order[j:j+CONFIG['batch_size']];optimizer.zero_grad()
            loss=loss_fn(model(x[idx]),y[idx]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),CONFIG['clip_grad_norm']);optimizer.step()
        model.eval()
        with torch.no_grad():v=loss_fn(model(vx),vy).item()
        history.append(v)
        if v<best-CONFIG['min_delta']:
            best=v;best_epoch=epoch+1;state={k:t.detach().clone() for k,t in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=CONFIG['patience']:break
    model.load_state_dict(state);model.eval()
    return dict(best_epoch=best_epoch,epochs_run=epoch+1,validation_bce=best,
                validation_bce_history=history,training_seconds=time.perf_counter()-start)


def predict(model,frame):
    if isinstance(model,LogisticRegression):return model.predict_proba(frame)[:,1]
    with torch.no_grad():return torch.sigmoid(model(torch.as_tensor(frame))).numpy()


def basic_decisions(scores,invalid,gate):
    return np.where(invalid,-1,np.where(scores>=.8,1,np.where(scores<=.2,0,-1))) if gate else (scores>=.5).astype(int)


def enrich(y,decision):
    d=decision_metrics(y,decision)
    d.update(n_decided_empty=int(((y==0)&(decision>=0)).sum()),n_decided_occupied=int(((y==1)&(decision>=0)).sum()))
    return d


def evaluate(model,prep,Xval,yval,Xt,yt,dates,run_id):
    metrics=[];coverage=[];thresholds=[];predictions=[];validation=[]
    for scenario in SCENARIOS:
        raw=sensor_attack(Xt,scenario,CONFIG['attack_seed']);frame,invalid=prep.transform(raw)
        t0=time.perf_counter();score=predict(model,frame);elapsed=time.perf_counter()-t0
        vp,vi=prep.transform(sensor_attack(Xval,scenario,CONFIG['validation_attack_seed']))
        vs=predict(model,vp)
        assert np.isfinite(score).all() and np.isfinite(vs).all()
        pred=pd.DataFrame(dict(run_id=run_id,row_index=np.arange(len(yt)),timestamp=dates,y_true=yt,scenario=scenario,
            score=score,invalid=invalid,corrupted=(np.random.default_rng(CONFIG['attack_seed']).random(len(yt))<.25) if scenario!='normal' else False,
            decision_off=basic_decisions(score,invalid,False),decision_on=basic_decisions(score,invalid,True)))
        for gate in [False,True]:
            metrics.append(dict(scenario=scenario,safety_gate=gate,inference_batch_seconds=elapsed,**enrich(yt,pred['decision_on' if gate else 'decision_off'].to_numpy())))
            for target in TARGETS:
                d=select_exact(score,invalid,target,gate,CONFIG['tie_seed'])
                row=dict(scenario=scenario,safety_gate=gate,target_coverage=target,feasible=d is not None,method='test_score_top_k_no_labels')
                col=f'posthoc_guard{int(gate)}_target{int(target*100)}'
                pred[col]=d if d is not None else np.nan
                if d is not None:row.update(enrich(yt,d))
                coverage.append(row)
        for target in TARGETS:
            threshold=float(np.quantile(np.abs(vs-.5),1-target))
            d=np.where(np.abs(score-.5)>=threshold,(score>=.5).astype(int),-1)
            pred[f'validation_target{int(target*100)}']=d
            thresholds.append(dict(scenario=scenario,safety_gate=False,target_coverage=target,
                method='validation_threshold_test_coverage_not_guaranteed',validation_confidence_threshold=threshold,**enrich(yt,d)))
        predictions.append(pred)
        validation.append(pd.DataFrame(dict(row_index=np.arange(len(yval)),y_true=yval,scenario=scenario,score=vs,invalid=vi)))
    return dict(metrics=pd.DataFrame(metrics),coverage_metrics=pd.DataFrame(coverage),
        validation_threshold_metrics=pd.DataFrame(thresholds),predictions=pd.concat(predictions,ignore_index=True),
        validation_predictions=pd.concat(validation,ignore_index=True))


def prepare_output(out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(f'Refusing to overwrite existing output: {out}')
    if not out.is_relative_to(ROOT/'results') and not out.is_relative_to(ROOT/'analysis/phase7'):
        raise ValueError('Output must be new directory under results or analysis/phase7')
    out.mkdir(parents=True)
    return out


def environment():
    return dict(python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),
        versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','torch','scikit-learn','joblib']},
        torch_threads=torch.get_num_threads(),cuda_available=torch.cuda.is_available())


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--pilot',action='store_true')
    a=p.parse_args();out=prepare_output(a.out)
    torch.set_num_threads(CONFIG['torch_threads']);torch.use_deterministic_algorithms(True)
    g=np.load(ROOT/'data/banc_graph.npz',allow_pickle=False);mask,signs=g['mask'],g['signs']
    prep,Ftr,ytr,Fval,yval,Xval=load_training()
    if a.pilot:
        timings={}
        for name in ['banc_signed','mlp_matched']:
            model=initialize(name,42,mask,signs)
            log=fit_network(model,Ftr,ytr,Fval,yval,42,epochs=2)
            timings[name]={'epochs':2,'training_seconds':log['training_seconds']}
        (out/'timing_only.json').write_text(json.dumps(dict(environment=environment(),timings=timings,test_data_used=False),indent=2),encoding='utf-8')
        print(json.dumps(timings),flush=True);return
    plan=ROOT/'analysis/phase7/PROTOCOL_KO.md';scale=ROOT/'analysis/phase7/EXECUTION_SCALE_KO.md'
    assert plan.exists() and scale.exists(),'Record protocol and timing-based scale before training'
    provenance=dict(config=CONFIG,environment=environment(),protocol_sha256=sha256(plan),scale_sha256=sha256(scale),
        files={str(f.relative_to(ROOT)):sha256(f) for f in [Path(__file__),ROOT/'src/model.py',ROOT/'src/data.py',ROOT/'src/evaluation.py',ROOT/'src/connectome.py',ROOT/'data/banc_graph.npz',ROOT/'data/datatraining.txt',ROOT/'data/datatest2.txt']},
        split_audit=audit_split(ROOT/'data/datatraining.txt',ROOT/'data/datatest2.txt'))
    assert provenance['split_audit']['passes_basic_checks']
    (out/'manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    joblib.dump(prep,out/'sensor_prep.joblib')
    Xt,yt=load_occ(ROOT/'data/datatest2.txt');dates=pd.read_csv(ROOT/'data/datatest2.txt').date.to_numpy()
    masks={s:rewire_preserving_degrees(mask,s) for s in CONFIG['rewire_seeds']}
    np.savez_compressed(out/'graphs.npz',**{k:g[k] for k in g.files},**{f'rewired_{s}':v for s,v in masks.items()})
    arms=[]
    for seed in CONFIG['train_seeds']:
        arms.extend([('banc_signed',seed,None),('mlp_matched',seed,None),('logistic',seed,None)])
        arms.extend(('rewired_signed',seed,r) for r in CONFIG['rewire_seeds'])
    all_metrics={k:[] for k in ['metrics','coverage_metrics','validation_threshold_metrics']};logs=[];replays=[]
    for name,seed,rseed in arms:
        run_id=f'{name}_t{seed}'+(f'_r{rseed}' if rseed is not None else '')
        dest=out/run_id;dest.mkdir()
        topology=mask if rseed is None else masks[rseed]
        if name=='logistic':
            t0=time.perf_counter();model=LogisticRegression(max_iter=2000,class_weight='balanced',random_state=seed).fit(Ftr,ytr)
            log=dict(training_seconds=time.perf_counter()-t0,epochs_run=None,best_epoch=None,validation_bce=None)
            cap=dict(allocated=9,mask_active=9,output_effective=9,dense_matmul_macs=8,masked_matmul_macs=8)
            joblib.dump(model,dest/'model.joblib')
        else:
            model=initialize(name,seed,topology,signs);log=fit_network(model,Ftr,ytr,Fval,yval,seed)
            cap=capacities(topology) if name!='mlp_matched' else dict(allocated=1961,mask_active=1961,output_effective=1961,dense_matmul_macs=1764,masked_matmul_macs=1764)
            torch.save(model.state_dict(),dest/'model.pt')
            # Prespecified reproducibility check: repeat t42 BANC and MLP, no model choice.
            if seed==42 and name in ['banc_signed','mlp_matched']:
                again=initialize(name,seed,topology,signs);rep=fit_network(again,Ftr,ytr,Fval,yval,seed)
                exact=all(torch.equal(t,again.state_dict()[k]) for k,t in model.state_dict().items())
                scores_equal=np.array_equal(predict(model,Fval),predict(again,Fval))
                replays.append(dict(run_id=run_id,state_exact=exact,validation_scores_exact=scores_equal,repeat_training_seconds=rep['training_seconds']))
                assert exact and scores_equal,'Deterministic replay failed'
        idcols=dict(run_id=run_id,model=name,train_seed=seed,rewire_seed=rseed,graph_id='original_banc_96',graph_seed=None)
        result=evaluate(model,prep,Xval,yval,Xt,yt,dates,run_id)
        for key,frame in result.items():
            for col,val in idcols.items():frame[col]=val
            if key.endswith('predictions'):frame.to_csv(dest/(key+'.csv.gz'),index=False,compression='gzip',float_format='%.17g')
            else:
                frame.to_csv(dest/(key+'.csv'),index=False);all_metrics[key].append(frame)
        logs.append(dict(**idcols,**cap,**log))
        (dest/'training.json').write_text(json.dumps(logs[-1],indent=2),encoding='utf-8')
        print(f'Completed {len(logs)}/{len(arms)} {run_id}: {log["training_seconds"]:.2f}s',flush=True)
    for key,frames in all_metrics.items():pd.concat(frames,ignore_index=True).to_csv(out/(key+'.csv'),index=False)
    pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,list)} for row in logs]).to_csv(out/'training_metadata.csv',index=False)
    (out/'replay_checks.json').write_text(json.dumps(replays,indent=2),encoding='utf-8')
    (out/'COMPLETE.json').write_text(json.dumps(dict(runs=len(logs),finished_utc=pd.Timestamp.now(tz='UTC').isoformat())),encoding='utf-8')


if __name__=='__main__':main()
