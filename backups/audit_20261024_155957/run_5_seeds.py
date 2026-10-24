"""FlyBrake: run five training seeds and summarize experimental results.

Place this file next to train.py in your flybrake project folder.
Requires train.py to accept --seed, as described in the setup instructions.
"""
from pathlib import Path
import subprocess
import sys

import pandas as pd

BASE = Path(__file__).resolve().parent
RESULT_DIR = BASE / 'results'
SEEDS = [42, 43, 44, 45, 46]

required = [
    BASE / 'train.py',
    BASE / 'data' / 'banc_graph.npz',
    BASE / 'data' / 'datatraining.txt',
    BASE / 'data' / 'datatest2.txt',
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit('먼저 다음 파일을 확인하세요:\n' + '\n'.join(missing))

RESULT_DIR.mkdir(exist_ok=True)
all_results = []

for index, seed in enumerate(SEEDS, start=1):
    outdir = RESULT_DIR / f'seed{seed}'
    cmd = [
        sys.executable,
        str(BASE / 'train.py'),
        '--graph', str(BASE / 'data' / 'banc_graph.npz'),
        '--train', str(BASE / 'data' / 'datatraining.txt'),
        '--test', str(BASE / 'data' / 'datatest2.txt'),
        '--epochs', '18',
        '--outdir', str(outdir),
        '--seed', str(seed),
    ]
    print(f'\n[{index}/{len(SEEDS)}] 시드 {seed} 학습 시작', flush=True)
    subprocess.run(cmd, check=True, cwd=BASE)
    results_file = outdir / 'metrics.csv'
    if not results_file.exists():
        raise SystemExit(f'결과 파일이 없습니다: {results_file}')
    df = pd.read_csv(results_file)
    df['seed'] = seed
    all_results.append(df)
    print(f'시드 {seed} 완료: {results_file}', flush=True)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(RESULT_DIR / 'metrics_5seeds.csv', index=False, encoding='utf-8-sig')

metrics = [
    'coverage',
    'false_alarm_per_empty',
    'unsafe_false_empty_per_occupied',
    'abstain_rate',
    'f1_decided_only',
    'precision_decided_only',
    'recall_decided_only',
]
summary = (
    combined.groupby(['model', 'scenario', 'safety_gate'], as_index=False)[metrics]
    .agg(['mean', 'std'])
)
# The grouped columns have a 2-level MultiIndex; flatten for CSV readability.
summary.columns = [
    '_'.join(str(x) for x in c if x and x != '') if isinstance(c, tuple) else str(c)
    for c in summary.columns
]
summary.to_csv(RESULT_DIR / 'summary_5seeds.csv', index=False, encoding='utf-8-sig')

print('\n=== 5회 반복 실험 완료 ===')
print('전체 결과 :', RESULT_DIR / 'metrics_5seeds.csv')
print('평균/표준편차 :', RESULT_DIR / 'summary_5seeds.csv')
print('\n정상 환경(안전 장치 OFF) 오경보율 평균 / 표준편차')
subset = combined[(combined['scenario'] == 'normal') & (~combined['safety_gate'])]
print(subset.groupby('model')['false_alarm_per_empty'].agg(['mean','std']).round(4).to_string())
