"""Train FlyBrake and matched controls on UCI room occupancy CSVs.

This script requires REAL BANC v888 files transformed with build_graph.py.
Synthetic sensor corruption is used only for controlled robustness tests.
"""
import argparse
from pathlib import Path
import json
import platform
import importlib.metadata
from src.audit import sha256, audit_split
from src.evaluation import decision_metrics, coverage_rows
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.model import SignedConnectome
from src.data import load_occ, SensorPrep, augment, sensor_attack
from src.connectome import rewire_preserving_degrees

p=argparse.ArgumentParser()
p.add_argument('--graph', default='data/banc_graph.npz')
p.add_argument('--train', required=True, help='UCI datatraining.txt')
p.add_argument('--test', required=True, help='UCI datatest2.txt; NEVER used for model selection')
p.add_argument('--epochs',type=int,default=18)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--outdir',default='results/revised_single')
p.add_argument('--extended-controls', action='store_true', help='Add rewired all-positive and BANC/rewired free-sign controls')
p.add_argument('--attack-seed', type=int, default=42)
a=p.parse_args()
if a.epochs < 1: p.error('--epochs must be positive')
out = Path(a.outdir)
if out.exists() and any(out.iterdir()):
    p.error('Output directory is not empty; use a new directory to preserve previous results')
audit = audit_split(a.train, a.test)
if not audit['passes_basic_checks']:
    raise ValueError(f'Data split failed basic leakage checks: {audit}')
torch.use_deterministic_algorithms(True)

seed = a.seed
rng=np.random.default_rng(seed)
np.random.seed(seed);torch.manual_seed(seed)
torch.set_num_threads(min(torch.get_num_threads(),4))
G=np.load(a.graph,allow_pickle=False)
if str(G['source']) != 'BANC v888 user-supplied inputs (not independently authenticated)':
    raise ValueError('Unexpected graph format; check build_graph.py and original data provenance.')
mask = G['mask']
signs = G['signs']
other = rewire_preserving_degrees(mask, seed=seed + 1000)
assert mask.shape==other.shape
assert (mask!=other).any(), 'Rewiring failed: choose a graph with more edges'
X,y=load_occ(a.train)
cut=int(len(X)*.8)  # chronological holdout: no randomized temporal leakage
Xtr,ytr=X[:cut],y[:cut]
Xval,yval=X[cut:],y[cut:]
Xt,yt=load_occ(a.test)
prep=SensorPrep().fit(Xtr)
Xtraug=augment(Xtr,rng)
Ftr,_=prep.transform(Xtraug)
Fval,_=prep.transform(Xval)
Xtest,_=prep.transform(Xt)
Path(a.outdir).mkdir(parents=True,exist_ok=True)
joblib.dump(prep,Path(a.outdir)/'sensor_prep.joblib')

# Equal topology dimensions, train budget, and initialization for all NN arms.
# For simple baseline, logistic regression uses the same engineered sensor inputs.
lr=LogisticRegression(max_iter=2000,class_weight='balanced')
lr.fit(Ftr,ytr)
joblib.dump(lr,Path(a.outdir)/'logistic.joblib')

models={}
training_log={}
arms = [
    ('banc_signed',mask,signs),
    ('rewired_signed',other,signs),
    ('banc_unsigned',mask,np.ones_like(signs)),
]
if a.extended_controls:
    arms += [('rewired_unsigned',other,np.ones_like(signs)),
             ('banc_free',mask,signs), ('rewired_free',other,signs)]
for name,topology,ntsigns in arms:
    torch.manual_seed(seed)
    model=SignedConnectome(topology,ntsigns, input_dim=Ftr.shape[1], constraint="free" if name.endswith("_free") else "signed")
    batch_rng = np.random.default_rng(seed + 2000)
    optimizer=torch.optim.Adam(model.parameters(),lr=.006,weight_decay=1e-4)
    pos=np.sum(ytr==1);neg=np.sum(ytr==0)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(max(1,neg/max(pos,1)),dtype=torch.float32))
    best=float('inf');state=None;patience=0
    for epoch in range(a.epochs):
        model.train()
        order=batch_rng.permutation(len(Ftr))
        for pos0 in range(0,len(Ftr),256):
            ii=order[pos0:pos0+256]
            x=torch.tensor(Ftr[ii],dtype=torch.float32)
            t=torch.tensor(ytr[ii],dtype=torch.float32)
            optimizer.zero_grad()
            loss=loss_fn(model(x),t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score=loss_fn(model(torch.tensor(Fval,dtype=torch.float32)),torch.tensor(yval,dtype=torch.float32)).item()
        if score < best-1e-4:
            best_epoch=epoch+1
            best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};patience=0
        else:
            patience+=1
            if patience>=5: break
    model.load_state_dict(state)
    torch.save(model.state_dict(),Path(a.outdir)/f'{name}.pt')
    models[name]=model
    training_log[name] = dict(best_epoch=best_epoch, epochs_run=epoch+1, validation_bce=best, constraint=model.constraint, active_recurrent_weights=int(topology.sum()))
    print(f'{name}: validation BCE={best:.3f}')

def infer(name,frame):
    if name=='logistic': return lr.predict_proba(frame)[:,1]
    with torch.no_grad():
        return torch.sigmoid(models[name](torch.tensor(frame,dtype=torch.float32))).numpy()

def score(y,probs,invalid,guard):
    # positive = OCCUPIED. 'empty' can only be returned with high certainty.
    if guard:
        decisions=np.where(invalid, -1, np.where(probs>=.8,1,np.where(probs<=.2,0,-1)))
    else:
        decisions=(probs>=.5).astype(int)
    return decision_metrics(y, decisions)

records=[]
coverage_records=[]
threshold_records=[]
for scenario in ['normal','missing_light','co2_spike']:
    raw=sensor_attack(Xt,scenario,seed=a.attack_seed)
    frame,invalid=prep.transform(raw)
    for name in ['logistic'] + list(models):
        prob=infer(name,frame)
        for row in coverage_rows(yt, prob, invalid):
            coverage_records.append(dict(model=name, scenario=scenario, **row))
        # Deployment-style thresholds chosen on validation scores only.
        vf, vi = prep.transform(sensor_attack(Xval,scenario,seed=a.attack_seed + 1))
        vp = infer(name,vf)
        for target in [.25,.5,.65,.75,.9,1.]:
            threshold = float(np.quantile(np.abs(vp-.5), 1-target))
            decisions = np.where(np.abs(prob-.5)>=threshold, (prob>=.5).astype(int), -1)
            threshold_records.append(dict(model=name, scenario=scenario, target_coverage=target,
                validation_confidence_threshold=threshold, safety_gate=False,
                method='validation_threshold_test_coverage_not_guaranteed', **decision_metrics(yt,decisions)))
        for guard in [False,True]:
            records.append({'model':name,'scenario':scenario,'safety_gate':guard,
                            **score(yt,prob,invalid,guard)})
res=pd.DataFrame(records)
res.to_csv(Path(a.outdir)/'metrics.csv',index=False)
pd.DataFrame(coverage_records).to_csv(out/'coverage_metrics.csv',index=False)
pd.DataFrame(threshold_records).to_csv(out/'validation_threshold_metrics.csv',index=False)
np.savez_compressed(out/'experiment_graph.npz', **{k:G[k] for k in G.files}, trained_rewired_mask=other)
(Path(a.outdir)/'experiment_info.json').write_text(json.dumps({
    'protocol':'revised_v1', 'seed':seed, 'attack_seed':a.attack_seed,
    'epochs_requested':a.epochs, 'extended_controls':a.extended_controls,
    'python':platform.python_version(), 'platform':platform.platform(),
    'versions':{k:importlib.metadata.version(k) for k in ['numpy','pandas','torch','scikit-learn','joblib','networkx']},
    'input_sha256':{str(Path(x).resolve()):sha256(x) for x in [a.graph,a.train,a.test]},
    'code_sha256':{str(x.relative_to(Path(__file__).parent)):sha256(x) for x in [Path(__file__)] + sorted((Path(__file__).parent/'src').glob('*.py'))},
    'split_audit':audit, 'training_log':training_log,
    'batch_seed':seed+2000, 'rewire_seed':seed+1000, 'torch_threads':torch.get_num_threads(),
    'unsigned_means':'all-positive softplus, NOT unconstrained',
    'BANC_source_version':'v888',
    'BANC_neurons':int(len(signs)),
    'BANC_edges':int(mask.sum()),
    'BANC_predicted_GABA_neurons':int((signs==-1).sum()),
    'training_rows':int(len(Xtr)),
    'validation_rows':int(len(Xval)),
    'test_rows':int(len(Xt)),
    'sensor_scenarios':{'missing_light':'25% samples set to NaN','co2_spike':'25% samples +3000 ppm'},
    'limitations':'GABA inhibitory sign is an engineering assumption; test office data is not school data; output not certified safety device.'
},indent=2),encoding='utf-8')
print(res[['model','scenario','safety_gate','coverage','false_alarm_per_empty','unsafe_false_empty_per_occupied']].to_string(index=False))
print(f'\nResults: {a.outdir}/metrics.csv')
