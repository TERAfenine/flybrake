"""Selective prediction: full-class denominators and label-free selection."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

KEYS = ['model', 'scenario', 'safety_gate']
METRICS = ['coverage', 'false_alarm_per_empty', 'unsafe_false_empty_per_occupied',
           'abstain_rate', 'f1_decided_only', 'precision_decided_only', 'recall_decided_only']

def summarize(df, keys=KEYS, metrics=METRICS, expected_seeds=None):
    if df.empty or df[keys + ['seed']].isna().any().any():
        raise ValueError('Empty results or missing group keys/seed')
    if df.duplicated(keys + ['seed']).any():
        raise ValueError('Duplicate model/condition/seed rows')
    expected = set(df.seed.unique()) if expected_seeds is None else set(expected_seeds)
    for _, group in df.groupby(keys, dropna=False):
        if set(group.seed) != expected:
            raise ValueError('Incomplete seed group')
    result = df.groupby(keys, dropna=False)[metrics].agg(['mean', 'std'])
    result.columns = [f'{metric}_{stat}' for metric, stat in result.columns]
    result['n_seeds'] = df.groupby(keys, dropna=False).seed.nunique()
    return result.reset_index()

def decision_metrics(y, decisions):
    y, decisions = np.asarray(y), np.asarray(decisions)
    chosen = decisions >= 0
    neg, pos = y == 0, y == 1
    fp = int(np.sum(neg & (decisions == 1)))
    fn = int(np.sum(pos & (decisions == 0)))
    ratio = lambda a, b: float(a / b) if b else float('nan')
    return dict(coverage=float(chosen.mean()), abstain_rate=float((~chosen).mean()),
        false_alarm_per_empty=ratio(fp, neg.sum()), unsafe_false_empty_per_occupied=ratio(fn, pos.sum()),
        false_alarm_decided_empty=ratio(fp, (neg & chosen).sum()),
        miss_decided_occupied=ratio(fn, (pos & chosen).sum()),
        empty_coverage=ratio((neg & chosen).sum(), neg.sum()),
        occupied_coverage=ratio((pos & chosen).sum(), pos.sum()),
        n_decided=int(chosen.sum()), n_empty=int(neg.sum()), n_occupied=int(pos.sum()), fp=fp, fn=fn,
        **{name + '_decided_only': float(fn_score(y[chosen], decisions[chosen], zero_division=0)) if chosen.any() else float('nan')
           for name, fn_score in [('f1', f1_score), ('precision', precision_score), ('recall', recall_score)]})

def select_exact(probs, invalid, coverage, guard=False, tie_seed=2026):
    """Retrospective, label-free top-k; NOT a deployable fixed threshold."""
    if not 0 <= coverage <= 1:
        raise ValueError('coverage must be in [0,1]')
    probs = np.asarray(probs)
    if not np.isfinite(probs).all():
        raise ValueError('Nonfinite predictions')
    eligible = np.flatnonzero(~np.asarray(invalid, bool) if guard else np.ones(len(probs), bool))
    k = int(np.floor(coverage * len(probs)))
    if k > len(eligible):
        return None
    ties = np.random.default_rng(tie_seed).random(len(probs))
    order = np.lexsort((ties[eligible], -np.abs(probs[eligible] - .5)))
    chosen = eligible[order[:k]]
    decisions = np.full(len(probs), -1, dtype=int)
    decisions[chosen] = (probs[chosen] >= .5).astype(int)
    return decisions

def coverage_rows(y, probs, invalid, targets=(.25,.5,.65,.75,.9,1.)):
    rows = []
    for guard in [False, True]:
        for target in targets:
            decisions = select_exact(probs, invalid, target, guard)
            row = dict(safety_gate=guard, target_coverage=target, feasible=decisions is not None,
                       method='test_score_top_k_no_labels')
            if decisions is not None:
                row.update(decision_metrics(y, decisions))
            rows.append(row)
    return rows
