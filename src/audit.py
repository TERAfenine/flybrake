"""Provenance and chronological split checks; does not prove absence of all leakage."""
import hashlib
from pathlib import Path
import pandas as pd

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def audit_split(train, test):
    a, b = pd.read_csv(train), pd.read_csv(test)
    for frame in (a,b):
        if not frame.Occupancy.isin([0,1]).all():
            raise ValueError('Labels must be 0/1')
    ta, tb = pd.to_datetime(a.date), pd.to_datetime(b.date)
    overlap = len(set(ta) & set(tb))
    cut = int(len(a)*.8)
    checks = dict(train_time_sorted=bool(ta.is_monotonic_increasing),
        test_time_sorted=bool(tb.is_monotonic_increasing), overlapping_timestamps=overlap,
        validation_after_training=bool(ta.iloc[cut] > ta.iloc[cut-1]),
        test_after_validation=bool(tb.min() > ta.max()))
    cols = ['Temperature','Humidity','Light','CO2']
    checks['shared_feature_vectors'] = len(set(map(tuple,a[cols].to_numpy())) & set(map(tuple,b[cols].to_numpy())))
    checks['train_range'] = [str(ta.min()),str(ta.max())]
    checks['test_range'] = [str(tb.min()),str(tb.max())]
    checks['validation_gap_seconds'] = (ta.iloc[cut]-ta.iloc[cut-1]).total_seconds()
    checks['passes_basic_checks'] = bool(not overlap and all(checks[k] for k in
        ['train_time_sorted','test_time_sorted','validation_after_training','test_after_validation']))
    return checks
