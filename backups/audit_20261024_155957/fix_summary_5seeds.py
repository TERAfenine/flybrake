"""Restore model/scenario/safety_gate columns in the five-seed summary.
Run from D:\\flybrake using: python fix_summary_5seeds.py
Does not retrain models; reads existing results/metrics_5seeds.csv.
"""
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
path = root / 'results' / 'metrics_5seeds.csv'
if not path.exists():
    raise SystemExit(f'입력 파일이 없습니다: {path} (파일을 현재 프로젝트 results 폴더에 두세요)')

df = pd.read_csv(path)
keys = ['model', 'scenario', 'safety_gate']
metrics = [
    'coverage',
    'false_alarm_per_empty',
    'unsafe_false_empty_per_occupied',
    'abstain_rate',
    'f1_decided_only',
    'precision_decided_only',
    'recall_decided_only',
]
needed = keys + metrics + ['seed']
missing = sorted(set(needed) - set(df.columns))
if missing:
    raise SystemExit(f'필요한 열이 없습니다: {missing}')

# Group key columns are preserved by reset_index() AFTER aggregation.
summary = df.groupby(keys, sort=True)[metrics].agg(['mean', 'std'])
summary.columns = [f'{column}_{stat}' for column, stat in summary.columns]
summary = summary.reset_index()
out = root / 'results' / 'summary_5seeds_fixed.csv'
summary.to_csv(out, index=False, encoding='utf-8-sig')
print(f'완료: {out}')
print(summary[['model','scenario','safety_gate','false_alarm_per_empty_mean',
               'false_alarm_per_empty_std','coverage_mean']].to_string(index=False))
