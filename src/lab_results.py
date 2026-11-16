"""Read-only access to measured runs. Never trains, imputes metrics or joins bundles."""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd
from src.evaluation import summarize

LABELS = {
    'false_alarm_per_empty': '오경보율',
    'unsafe_false_empty_per_occupied': '미탐지율 (False-empty)',
    'f1_decided_only': 'F1 (판단 표본)',
    'recall_decided_only': 'Recall (판단 표본)',
    'coverage': 'Coverage',
    'abstain_rate': '판단 보류율',
}
METRICS = list(LABELS)
SCENARIOS = {'normal': '정상 입력', 'missing_light': '조도 센서 결손', 'co2_spike': 'CO₂ 센서 이상치'}
KINDS = {'metrics', 'coverage_metrics', 'validation_threshold_metrics'}
BASE_KEYS = ['model', 'scenario', 'safety_gate', 'seed']

@dataclass
class ResultTable:
    frame: pd.DataFrame
    sources: tuple
    notices: tuple = ()

def source_files(bundle, kind='metrics'):
    """Combined raw table takes precedence; never append its per-seed duplicates."""
    if kind not in KINDS:
        raise ValueError(f'Unknown table: {kind}')
    bundle = Path(bundle)
    combined = bundle / f'{kind}_5seeds.csv'
    if combined.is_file():
        return (combined,)
    return tuple(sorted(p for p in bundle.glob(f'seed*/{kind}.csv')
                        if re.fullmatch(r'seed\d+', p.parent.name)))

def discover_bundles(results):
    """Only the root and immediate non-seed folders are independent bundles."""
    root = Path(results)
    if not root.is_dir():
        return []
    candidates = [root] + sorted(p for p in root.iterdir() if p.is_dir()
                                and not re.fullmatch(r'seed\d+', p.name))
    return [p for p in candidates if source_files(p)]

def source_signature(bundle, kind='metrics'):
    # Re-evaluated on each UI rerun; a changed file invalidates st.cache_data.
    return tuple((str(p.resolve()), p.stat().st_mtime_ns, p.stat().st_size)
                 for p in source_files(bundle, kind))

def _booleans(series, name):
    normalized = series.astype(str).str.strip().str.lower().map(
        {'true': True, 'false': False, '1': True, '0': False, '1.0': True, '0.0': False})
    if normalized.isna().any():
        raise ValueError(f'{name}: True/False 또는 1/0 이외의 값이 있습니다.')
    return normalized.astype(bool)

def load_results(bundle, kind='metrics'):
    sources = source_files(bundle, kind)
    if not sources:
        return ResultTable(pd.DataFrame(), (), ('저장된 시드별 결과 파일이 없습니다.',))
    frames, notices = [], []
    for path in sources:
        try:
            df = pd.read_csv(path)
        except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            raise ValueError(f'{path.name}: 결과 CSV를 읽을 수 없습니다 ({error}).') from error
        if df.empty:
            raise ValueError(f'{path.name}: 결과 CSV에 측정 행이 없습니다.')
        folder_seed = re.fullmatch(r'seed(\d+)', path.parent.name)
        if 'seed' not in df and folder_seed:
            df['seed'] = int(folder_seed[1])
        missing = set(BASE_KEYS) - set(df)
        if missing:
            raise ValueError(f'{path.name}: 필수 열 누락 {sorted(missing)}. 요약 CSV는 개별 결과로 사용하지 않습니다.')
        if df[BASE_KEYS].isna().any().any():
            raise ValueError(f'{path.name}: 모델/조건/시드 값이 비어 있습니다.')
        for col in ['model', 'scenario']:
            df[col] = df[col].astype(str).str.strip()
            if df[col].eq('').any():
                raise ValueError(f'{col}: 빈 이름이 있습니다.')
        seeds = pd.to_numeric(df.seed, errors='raise')
        if not np.isfinite(seeds).all() or (seeds % 1 != 0).any():
            raise ValueError('학습 시드는 유한한 정수여야 합니다.')
        df['seed'] = seeds.astype(int)
        if folder_seed and not df.seed.eq(int(folder_seed[1])).all():
            raise ValueError(f'{path}: 폴더 시드와 CSV 시드가 다릅니다.')
        df['safety_gate'] = _booleans(df.safety_gate, 'safety_gate')
        if kind != 'metrics':
            required = {'target_coverage', 'method'}
            if kind == 'coverage_metrics':
                required.add('feasible')
            if not required.issubset(df):
                raise ValueError(f'{path.name}: 분석 종류 확인에 필요한 열 누락 {sorted(required-set(df))}')
            df['target_coverage'] = pd.to_numeric(df.target_coverage, errors='raise')
            if not df.target_coverage.between(0, 1).all() or df.method.isna().any():
                raise ValueError('유효하지 않은 목표 coverage 또는 분석 방법입니다.')
            if kind == 'coverage_metrics':
                df['feasible'] = _booleans(df.feasible, 'feasible')
        for metric in METRICS:
            if metric not in df:
                df[metric] = np.nan
                notices.append(f'{metric}: 저장된 열이 없어 N/A로 표시합니다.')
            df[metric] = pd.to_numeric(df[metric], errors='raise')
            nonnull = df[metric].dropna()
            if not nonnull.between(0, 1).all():
                raise ValueError(f'{metric}: 0~1 범위를 벗어난 값입니다.')
        frames.append(df)
    frame = pd.concat(frames, ignore_index=True)
    keys = BASE_KEYS + (['method', 'target_coverage'] if kind != 'metrics' else [])
    if frame.duplicated(keys).any():
        raise ValueError('같은 모델·조건·시드의 중복 측정 행이 있습니다. 자동으로 평균내지 않습니다.')
    return ResultTable(frame, tuple(str(p) for p in sources), tuple(dict.fromkeys(notices)))

def select_rows(frame, models, scenario, safety_gate, seed=None, target=None, method=None):
    if frame.empty:
        return frame.copy()
    selected = frame[frame.model.isin(models) & frame.scenario.eq(scenario)
                     & frame.safety_gate.eq(bool(safety_gate))]
    if seed is not None:
        selected = selected[selected.seed.eq(int(seed))]
    if target is not None:
        selected = selected[np.isclose(selected.target_coverage, target, rtol=0, atol=1e-10)]
    if method is not None:
        selected = selected[selected.method.eq(method)]
    return selected.copy()

def comparison_view(rows, models, expected_seeds, average):
    """One selected condition. Missing seeds/infeasible runs are never silently dropped.

    Reuses the research summarizer for mean and sample SD (ddof=1). The view
    marks undefined metrics N/A; measured CSVs and their definitions are untouched.
    """
    expected = set(int(s) for s in expected_seeds)
    records = []
    for model in models:
        group = rows[rows.model.eq(model)].copy() if not rows.empty else rows.copy()
        status = '측정값'
        if group.empty:
            status = '해당 조건의 측정값 없음'
        elif group.seed.duplicated().any():
            raise ValueError('비교 전에 하나의 센서/안전장치/분석 방법/coverage 조건을 선택하세요.')
        elif set(group.seed) != expected:
            status = f'시드 불완전 ({len(group)}/{len(expected)})'
        elif 'feasible' in group and not group.feasible.all():
            status = '달성 불가능한 coverage 포함'
        usable = status == '측정값'
        # Legacy files could store zero F1 for no decided samples. Display N/A.
        if usable:
            zero = group.coverage.eq(0)
            if 'n_decided' in group:
                zero |= group.n_decided.eq(0)
            group.loc[zero, ['f1_decided_only', 'recall_decided_only']] = np.nan
            if average:
                summary = summarize(group, keys=['model'], metrics=METRICS,
                                    expected_seeds=expected).iloc[0]
        for metric in METRICS:
            n_valid = int(group[metric].notna().sum()) if metric in group else 0
            complete = usable and n_valid == len(expected)
            value = (float(summary[f'{metric}_mean']) if average else float(group[metric].iloc[0])) if complete else np.nan
            std = float(summary[f'{metric}_std']) if complete and average else np.nan
            records.append(dict(model=model, metric=metric, label=LABELS[metric], value=value,
                                std=std, n_seeds=len(group), n_valid=n_valid,
                                status=status if not usable else ('측정값' if complete else '정의 불가 또는 결측 (N/A)')))
    return pd.DataFrame(records)

def display_value(value, std=np.nan, average=False):
    if pd.isna(value):
        return 'N/A'
    if average:
        return f'{value:.2%} ± {std:.2%}' if pd.notna(std) else f'{value:.2%} ± N/A'
    return f'{value:.2%}'
