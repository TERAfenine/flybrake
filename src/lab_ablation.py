"""Read-only, single-neuron intervention on the saved BANC signed network.

No fitting, checkpoint writes, or changes to src.model. The selected recurrent
activation is zeroed after EACH tanh update, including the final readout step.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import re
import warnings
import joblib
import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
import torch
from torch.nn import functional as F
from src.model import SignedConnectome
from src.data import SensorPrep

SOURCE = 'BANC v888 user-supplied inputs (not independently authenticated)'

def load_graph(path):
    with np.load(path, allow_pickle=False) as archive:
        graph = {key: archive[key].copy() for key in archive.files}
    required = {'ids', 'mask', 'signs', 'source'}
    if not required.issubset(graph) or str(graph['source']) != SOURCE:
        raise ValueError('지원하는 BANC 그래프 형식이 아닙니다.')
    ids, mask, signs = graph['ids'], graph['mask'], graph['signs']
    n = len(ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(set(ids)) != n:
        raise ValueError('뉴런 ID는 중복 없는 정수여야 합니다.')
    if mask.shape != (n,n) or signs.shape != (n,):
        raise ValueError('뉴런 ID와 연결/부호 배열의 크기가 다릅니다.')
    if not np.isin(mask,[0,1]).all() or not np.isin(signs,[-1,1]).all():
        raise ValueError('이진 연결 마스크와 GABA/ACh 부호를 확인하세요.')
    if 'counts' in graph:
        counts = graph['counts']
        if counts.shape != mask.shape or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError('시냅스 수 배열이 유효하지 않습니다.')
    return graph

def run_folders(results):
    root = Path(results)
    if not root.is_dir():
        return {}
    bundles = [root] + sorted(p for p in root.iterdir() if p.is_dir() and not re.fullmatch(r'seed\d+',p.name))
    found = {}
    for bundle in bundles:
        seeds = {int(p.name[4:]): p for p in bundle.glob('seed*')
                 if p.is_dir() and re.fullmatch(r'seed\d+',p.name)
                 and (p/'banc_signed.pt').is_file() and (p/'sensor_prep.joblib').is_file()}
        if seeds:
            found[bundle] = dict(sorted(seeds.items()))
    return found

def graph_path(run, project):
    saved = Path(run)/'experiment_graph.npz'
    return saved if saved.is_file() else Path(project)/'data/banc_graph.npz'

def file_signature(paths):
    return tuple((str(Path(p).resolve()),Path(p).stat().st_mtime_ns,Path(p).stat().st_size)
                 for p in paths if Path(p).is_file())

@dataclass
class LoadedExperiment:
    run: Path
    seed: int
    graph: dict
    model: SignedConnectome
    prep: SensorPrep
    info: dict

def load_experiment(run, project):
    run = Path(run)
    match = re.fullmatch(r'seed(\d+)',run.name)
    if not match:
        raise ValueError('시드별 폴더를 선택해야 합니다.')
    seed = int(match[1])
    info_file = run/'experiment_info.json'
    info = json.loads(info_file.read_text(encoding='utf-8')) if info_file.exists() else {}
    if info.get('seed',seed) != seed:
        raise ValueError('폴더 이름과 기록된 학습 시드가 다릅니다.')
    recorded = info.get('versions',{}).get('scikit-learn')
    if recorded and recorded != sklearn.__version__:
        raise ValueError(f'전처리 저장 버전 scikit-learn {recorded}, 현재 {sklearn.__version__}: '
                         '호환 환경에서 실행해야 합니다. 현재 환경에서는 기존 results 묶음을 선택하세요. '
                         '전처리를 다시 학습하거나 다른 시드에서 가져오지 않습니다.')
    graph = load_graph(graph_path(run,project))
    state = torch.load(run/'banc_signed.pt',map_location='cpu',weights_only=True)
    for key in ['mask','signs']:
        if key not in state or not np.array_equal(state[key].numpy(),graph[key]):
            raise ValueError(f'체크포인트와 그래프의 {key}가 다릅니다. 추론을 중단합니다.')
    if not all(torch.isfinite(t).all() for t in state.values()):
        raise ValueError('체크포인트에 유한하지 않은 값이 있습니다.')
    degree = np.maximum(graph['mask'].sum(axis=1),1)
    if not np.array_equal(state['degree'].numpy(),degree):
        raise ValueError('체크포인트의 차수 정규화가 현재 그래프와 다릅니다.')
    # Construction must not consume the caller's global random stream.
    with torch.random.fork_rng():
        model = SignedConnectome(graph['mask'],graph['signs'],input_dim=state['input_map.weight'].shape[1])
    model.load_state_dict(state,strict=True)
    model.eval().requires_grad_(False)
    with warnings.catch_warnings():
        warnings.simplefilter('error',InconsistentVersionWarning)
        try:
            prep = joblib.load(run/'sensor_prep.joblib')
        except InconsistentVersionWarning as error:
            raise ValueError('전처리 저장 버전이 현재 scikit-learn과 다릅니다. 호환 환경이 필요합니다.') from error
    if not isinstance(prep,SensorPrep):
        raise ValueError('지원하는 SensorPrep 객체가 아닙니다.')
    probe,_ = prep.transform(np.asarray(prep.medians).reshape(1,-1))
    if probe.shape != (1,model.input_map.in_features) or not np.isfinite(probe).all():
        raise ValueError('전처리 입력 차원 또는 변환값이 모델과 다릅니다.')
    return LoadedExperiment(run,seed,graph,model,prep,info)

@torch.inference_mode()
def ablated_logits(model, features, neuron=None, return_states=False):
    if model.constraint != 'signed':
        raise ValueError('이번 실험은 BANC signed 모델만 지원합니다.')
    n = len(model.signs)
    if neuron is not None and (isinstance(neuron,bool) or not isinstance(neuron,(int,np.integer)) or not 0<=neuron<n):
        raise ValueError('유효하지 않은 뉴런 인덱스입니다.')
    x = torch.as_tensor(features,dtype=torch.float32,device=model.mask.device)
    if x.ndim != 2 or x.shape[1] != model.input_map.in_features or not torch.isfinite(x).all():
        raise ValueError('모델 입력은 유한한 전처리된 2차원 배열이어야 합니다.')
    h = x.new_zeros((len(x),n))
    drive = x.new_zeros((len(x),n))
    drive[:,:model.n_inputs] = torch.tanh(model.input_map(x))
    weights = F.softplus(model.raw_weights)*model.mask*model.signs.view(1,-1)
    weights = .35*weights/model.degree.sqrt().view(-1,1)
    states = []
    for _ in range(model.steps):
        h = torch.tanh(drive + h @ weights.T + model.bias)
        if neuron is not None:
            h[:,int(neuron)] = 0.
        if return_states:
            states.append(h.clone())
    logits = model.output_map(h[:,-model.n_outputs:]).squeeze(-1)
    return (logits,states) if return_states else logits

def decision(probability, invalid, guard=True):
    if not np.isfinite(probability) or not 0<=probability<=1:
        raise ValueError('유효하지 않은 모델 점수입니다.')
    if guard:
        if invalid:
            return '판단 보류',True,'입력 결손 또는 학습 범위 밖'
        if .2 < probability < .8:
            return '판단 보류',True,'점수 불확실 구간 (0.2~0.8 사이)'
        return ('재실' if probability>=.8 else '비재실'),False,'고정 0.2/0.8 규칙'
    return ('재실' if probability>=.5 else '비재실'),False,'안전장치 OFF · 0.5 기준'

def demo_input(values, scenario='normal'):
    raw = np.asarray(values,dtype=float).reshape(1,4).copy()
    # Single-input demonstration: always modify this input, NOT the 25% batch attack.
    if scenario == 'missing_light':
        raw[0,2] = np.nan
    elif scenario == 'co2_spike':
        raw[0,3] += 3000.
    elif scenario != 'normal':
        raise ValueError('지원하지 않는 센서 시연 조건입니다.')
    return raw

@torch.inference_mode()
def compare_input(experiment, raw, neuron=None, guard=True):
    raw = np.asarray(raw,dtype=float)
    if raw.shape != (1,4):
        raise ValueError('이번 시연은 한 개의 4센서 입력만 지원합니다.')
    features, invalid = experiment.prep.transform(raw)
    x = torch.tensor(features,dtype=torch.float32)
    original = float(torch.sigmoid(experiment.model(x)).item())
    altered = float(torch.sigmoid(ablated_logits(experiment.model,x,neuron)).item())
    return dict(original=original,altered=altered,delta=altered-original,
                original_decision=decision(original,bool(invalid[0]),guard),
                altered_decision=decision(altered,bool(invalid[0]),guard),
                invalid=bool(invalid[0]),neuron=neuron,
                banc_id=str(experiment.graph['ids'][neuron]) if neuron is not None else None,
                seed=experiment.seed,run=str(experiment.run),disabled_count=int(neuron is not None))
