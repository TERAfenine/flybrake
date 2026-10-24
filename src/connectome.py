"""Fixed BANC connectivity + presynaptic NT-based SIGN ASSUMPTION.

Source files: BANC v888, not example/synthetic graphs. ACh=+1, GABA=-1
is an explicitly simplified modelling assumption, not a verified synaptic sign.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx


def table(path, columns=None):
    path = Path(path)
    if path.suffix == '.feather':
        return pd.read_feather(path, columns=columns)
    if path.suffix == '.csv':
        df = pd.read_csv(path)
        return df if columns is None else df[columns]
    raise ValueError(f'Expected csv or feather: {path}')


def build_graph(meta_path, nt_path, edge_path, *, n=96, region='ventral_nerve_cord',
                min_score=.8, min_nt_count=20, min_synapses=5, seed=42):
    meta = table(meta_path, ['banc_888_id', 'region', 'super_class'])
    meta = meta[
        (meta["region"] == region) &
        (meta["super_class"] == "ventral_nerve_cord_intrinsic")
        ]
    nts = table(nt_path, ['root_id','neurotransmitter_predicted','neurotransmitter_score','count'])
    nts = nts[(nts['neurotransmitter_predicted'].isin(['acetylcholine','gaba'])) &
              (nts['neurotransmitter_score'] >= min_score) &
              (nts['count'] >= min_nt_count)]

    # 두 파일의 뉴런 ID를 정수형으로 통일
    meta["banc_888_id"] = meta["banc_888_id"].astype("int64")
    nts["root_id"] = nts["root_id"].astype("int64")

    # 동일한 뉴런 ID를 기준으로 결합
    neurons = meta.merge(
        nts, left_on="banc_888_id", right_on="root_id"
    )

    print("매칭된 뉴런 수:", len(neurons))

    if len(neurons) < n:
        raise ValueError(f'Only {len(neurons)} eligible neurons; lower n or loosen filters.')
    good = set(neurons['root_id'].astype('int64').tolist())
    # Full file is ~285MB; this read may need 1-2GB RAM. Use a local Codex export if needed.

    # 실제 BANC 뉴런 연결 데이터 읽기
    edges = table(edge_path, ["pre", "post", "count"])

    print("전체 연결 수:", len(edges))
    print("변환 전 자료형:")
    print(edges.dtypes)

    # 뉴런 ID를 정수형으로 통일
    edges["pre"] = pd.to_numeric(
        edges["pre"], errors="raise"
    ).astype("int64")

    edges["post"] = pd.to_numeric(
        edges["post"], errors="raise"
    ).astype("int64")

    # 시냅스 수를 숫자형으로 변환
    edges["count"] = pd.to_numeric(
        edges["count"], errors="raise"
    )

    # 선택된 뉴런과 일치하는 연결 확인
    pre_match = edges["pre"].isin(good)
    post_match = edges["post"].isin(good)

    print("출발 뉴런 ID 일치:", pre_match.sum())
    print("도착 뉴런 ID 일치:", post_match.sum())
    print("양쪽 모두 일치:", (pre_match & post_match).sum())

    # 조건에 맞는 실제 뉴런 연결만 선택
    edges = edges[
        (edges["count"] >= min_synapses) &
        pre_match &
        post_match &
        (edges["pre"] != edges["post"])
        ].copy()

    print("최종 선택된 연결 수:", len(edges))

    if edges.empty:
        raise ValueError('No compatible BANC connections found. Check data version/region.')
    G = nx.DiGraph()
    G.add_nodes_from(good)
    G.add_weighted_edges_from((int(r.pre), int(r.post), float(r.count))
                              for r in edges.itertuples(index=False))
    # Dense connected-neighborhood sampling: real neurons, not arbitrary random IDs.
    rng = np.random.default_rng(seed)
    choices = [(u, G.degree(u)) for u in G if G.degree(u) > 0]
    hub = max(choices, key=lambda z: z[1])[0]
    selected = [hub]
    chosen = {hub}
    und = G.to_undirected()
    while len(selected) < n:
        candidates = set().union(*(set(und.neighbors(v)) for v in selected)) - chosen
        if not candidates:
            raise ValueError(f'Disconnected selection stopped at {len(selected)} neurons')
        # Add candidate sharing the highest sum of synapse strengths with chosen set.
        weights = {v: sum(und[v][u].get('weight', 1) for u in und.neighbors(v) if u in chosen)
                   for v in candidates}
        high = max(weights.values())
        best = min(v for v, w in weights.items() if w == high)  # deterministic
        selected.append(best)
        chosen.add(best)
    idx = {v: i for i, v in enumerate(selected)}
    nt_lookup = neurons.set_index('root_id')['neurotransmitter_predicted'].to_dict()
    signs = np.array([-1. if nt_lookup[i] == 'gaba' else 1. for i in selected], dtype=np.float32)
    mask = np.zeros((n,n), dtype=np.float32)  # postsynaptic row, presynaptic column
    raw_counts = np.zeros((n,n), dtype=np.float32)
    for r in edges.itertuples(index=False):
        a, b = int(r.pre), int(r.post)
        if a in idx and b in idx:
            mask[idx[b],idx[a]] = 1.
            raw_counts[idx[b],idx[a]] = float(r.count)
    if mask.sum() < n: 
        raise ValueError(f'Subgraph too sparse ({mask.sum():.0f} edges for {n} neurons). Choose a denser circuit.')
    gaba_fraction = (signs == -1).mean()
    if not (0.05 <= gaba_fraction <= .9):
        raise ValueError(f'GABA fraction {gaba_fraction:.2f}: choose a mixed local circuit.')
    return {'mask':mask, 'signs':signs, 'ids':np.asarray(selected, dtype=np.int64),
            'counts':raw_counts, 'source':'BANC v888 user-supplied inputs (not independently authenticated)',
            'region':region, 'seed':seed}


def rewire_preserving_degrees(mask, seed=7, attempts_per_edge=25):
    """Directed double-edge swaps preserve each neuron's in/out degree."""
    rng = np.random.default_rng(seed)
    dest, source = np.where(mask > 0)
    edges = [(int(s), int(d)) for s,d in zip(source,dest)]
    E = set(edges)
    if len(edges) < 4:
        return mask.copy()
    for _ in range(attempts_per_edge * len(edges)):
        i, j = rng.choice(len(edges), 2, replace=False)
        a,b = edges[i]
        c,d = edges[j]
        if a == c or b == d or a == d or c == b: continue
        if (a,d) in E or (c,b) in E: continue
        E.remove((a,b)); E.remove((c,d)); E.add((a,d)); E.add((c,b))
        edges[i] = (a,d); edges[j] = (c,b)
    out = np.zeros_like(mask)
    for s,d in edges:
        out[d,s] = 1.
    assert np.array_equal((mask > 0).sum(0), (out > 0).sum(0))
    assert np.array_equal((mask > 0).sum(1), (out > 0).sum(1))
    return out
