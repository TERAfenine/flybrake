"""FlyBrake Lab phase 1: browse recorded measurements, never retrain."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from src.lab_results import (LABELS, METRICS, SCENARIOS, discover_bundles, load_results,
                             source_signature, select_rows, comparison_view, display_value)
from src.lab_charts import comparison_figure
from src.lab_experience import show_gate_comparison
from src.lab_ux import setup,navigation,glossary,model_name,HELP

BASE = Path(os.environ.get('FLYBRAKE_ROOT', PROJECT))
st.set_page_config(page_title='FlyBrake Lab', page_icon='🧠', layout='wide')
setup()
navigation('pages/1_FlyBrake_Lab.py')
st.title('FlyBrake Lab')
st.caption('연구자 모드 · 모델 성능 비교')
glossary()
st.caption('사전 학습 결과 탐색 모드 · 설정을 바꾸면 저장된 측정값을 비교합니다. 새로운 학습이나 추론을 실행하지 않습니다.')

@st.cache_data(show_spinner='저장된 측정값을 읽는 중… 새 학습은 실행하지 않습니다.')
def cached_results(bundle, kind, signature):
    # signature intentionally participates in the cache key, even though the
    # reader itself needs only bundle/kind. Filters are applied AFTER loading.
    return load_results(bundle, kind)

def read_table(bundle, kind):
    try:
        return cached_results(str(bundle), kind, source_signature(bundle, kind))
    except (ValueError, OSError, TypeError) as error:
        st.warning(f'결과를 표시할 수 없습니다: {error}')
        return None

def table_view(view, average):
    output = pd.DataFrame({'지표': list(LABELS.values())})
    model_names = list(view.model.unique())
    values = []
    for model in model_names:
        data = view[view.model.eq(model)].set_index('metric').loc[METRICS]
        output[model] = [display_value(r.value, r['std'], average) for _, r in data.iterrows()]
        values.append(data.value.to_numpy())
    if len(values) == 2:
        output['차이 A − B (%p)'] = [f'{v * 100:+.3f}' if np.isfinite(v) else 'N/A'
                                      for v in values[0] - values[1]]
    return output

def show_comparison(rows, models, expected, average, prefix, cards=False):
    try:
        view = comparison_view(rows, models, expected, average)
    except ValueError as error:
        st.warning(str(error))
        return
    for model in models:
        states = view[view.model.eq(model)].status.unique()
        for state in states:
            if state != '측정값':
                st.info(f'{model}: {state}. N/A를 0으로 해석하지 마세요.')
    if cards:
        with st.expander('모델별 상세 지표 카드 · 평균과 표준편차'):
            for column, model, letter in zip(st.columns(2), models, ['A', 'B']):
                with column:
                    st.subheader(f'{letter} · {model_name(model)}')
                    st.caption(model)
                    group = view[view.model.eq(model)].set_index('metric')
                    for metric in METRICS:
                        row = group.loc[metric]
                        st.metric(LABELS[metric], display_value(row.value, row['std'], average),help=HELP.get(metric))
    st.dataframe(table_view(view, average), hide_index=True, use_container_width=True)
    if view.value.notna().any():
        left, right = st.columns(2)
        with left:
            st.markdown('**오경보·미탐지 비교 · 단위 %**')
            st.plotly_chart(comparison_figure(view, METRICS[:2], average), use_container_width=True,
                            key=prefix+'_errors')
        with right:
            st.markdown('**자동 판단·보류 비교 · 단위 %**')
            st.plotly_chart(comparison_figure(view, ['coverage', 'abstain_rate'], average),
                            use_container_width=True, key=prefix+'_coverage')
    else:
        st.info('선택 조건에 표시 가능한 수치가 없어 그래프를 생략했습니다.')
    st.caption('평균 모드: 시드별 측정값의 산술평균 ± 표본 표준편차(ddof=1). 오차막대는 표준편차입니다. '
               '개별 모드: 해당 시드의 측정값만 표시합니다. 차이는 A − B이며 신뢰구간이나 유의성 검정이 아닙니다.')

bundles = discover_bundles(BASE/'results')
if not bundles:
    st.info('조회 가능한 실험 결과 묶음이 없습니다. results의 metrics_5seeds.csv 또는 seed*/metrics.csv를 확인하세요.')
    st.stop()
with st.sidebar:
    st.header('상세 비교 설정')
    st.caption('7개 모델·동일 Coverage는 revised_audit_20261024 묶음에서 확인합니다.')
    if st.button('결과 파일 새로 읽기'):
        cached_results.clear()
    bundle = st.selectbox('결과 묶음', bundles,
                          format_func=lambda p: '기존 결과 · results' if p == BASE/'results' else p.name,
                          key='lab_bundle')
table = read_table(bundle, 'metrics')
if table is None or table.frame.empty:
    st.info('선택한 묶음에 읽을 수 있는 측정값이 없습니다. 다른 묶음을 선택하세요.')
    st.stop()
frame = table.frame
for notice in table.notices:
    st.info(notice)
seeds = sorted(int(s) for s in frame.seed.unique())
models = sorted(frame.model.unique())
if len(models) < 2:
    st.info('두 모델을 비교하려면 실제 결과에 최소 2개 모델이 있어야 합니다.')
    st.stop()
with st.sidebar:
    seed = st.selectbox('학습 시드', [None] + seeds,
                        format_func=lambda s: f'전체 {len(seeds)}시드 평균' if s is None else str(s),
                        key=f'lab_seed_{bundle.name}')
    st.session_state.setdefault(f'lab_a_{bundle.name}', 'banc_signed' if 'banc_signed' in models else models[0])
    model_a = st.selectbox('모델 A', models,
                          key=f'lab_a_{bundle.name}',format_func=model_name)
    other_models = [m for m in models if m != model_a]
    st.session_state.setdefault(f'lab_b_{bundle.name}_{model_a}', 'rewired_signed' if 'rewired_signed' in other_models else other_models[0])
    model_b = st.selectbox('모델 B', other_models,
                          key=f'lab_b_{bundle.name}_{model_a}',format_func=model_name)
    scenarios = sorted(frame.scenario.unique())
    st.session_state.setdefault(f'lab_scenario_{bundle.name}', 'normal' if 'normal' in scenarios else scenarios[0])
    scenario = st.selectbox('센서 조건', scenarios, format_func=lambda s: SCENARIOS.get(s,s),
                            key=f'lab_scenario_{bundle.name}')
    st.session_state.setdefault('lab_gate',True)
    gate = st.checkbox('안전장치 ON', key='lab_gate')
    st.caption('기본 비교 ON: 입력 오류 또는 점수 0.2~0.8이면 보류. OFF: 점수 0.5 기준으로 분류.')
    st.caption('학습 시드, 회로 선택 시드, 재배선 시드는 서로 다른 설정입니다. 이 페이지는 회로를 새로 추출하지 않습니다.')

average = seed is None
expected = seeds if average else [seed]
pair = [model_a, model_b]
st.caption(f'A: {model_name(model_a)} ({model_a}) · B: {model_name(model_b)} ({model_b})')
st.subheader('두 AI 모델 직접 비교')
st.caption('설정은 왼쪽 사이드바에 있습니다. 좁은 화면에서는 왼쪽 위 화살표로 펼치세요.')
st.write(f'**{bundle.name}** · {len(models)}개 모델 · '
         f'{"전체 " + str(len(seeds)) + "시드 평균" if average else "학습 시드 " + str(seed)} · '
         f'{SCENARIOS.get(scenario,scenario)} · 안전장치 {"ON" if gate else "OFF"}')
st.caption('banc_unsigned / rewired_unsigned는 모든 재귀 연결 양수 제약, *_free는 부호 자유 모델입니다.')
selected = select_rows(frame, pair, scenario, gate, seed)
show_comparison(selected, pair, expected, average, 'main', cards=True)
st.info('오경보율이 낮아져도 미탐지율 또는 판단 보류율이 늘 수 있습니다. Coverage와 함께 해석하세요.')

with st.expander('지표의 분모와 보류 표본 처리'):
    st.markdown('''- **오경보율** = FP / 전체 실제 비재실 표본 수.
- **미탐지율 (False-empty)** = FN / 전체 실제 재실 표본 수.
- 위 두 지표는 보류 표본도 분모에 포함하지만 보류를 FP/FN으로 세지 않습니다.
- **F1 · Recall**은 자동 판단한 표본에서만 계산한 저장값입니다. 보류 표본은 계산에서 제외합니다.
- **Coverage** = 판단한 표본 / 전체 표본. **판단 보류율** = 보류 표본 / 전체 표본.
- 판단 표본이 없거나 값이 저장되지 않았으면 **N/A**입니다. 판단 표본 안에 양성 정답이 없을 때의 Recall은 기존 평가 코드의 `zero_division=0` 정의를 따릅니다.
- 평균은 시드별 지표를 평균낸 값이며 여러 시드의 예측을 합쳐 새로 계산한 F1이 아닙니다. 선택 모델에 시드가 빠지면 전체 평균 대신 N/A를 표시합니다.''')

st.divider()
st.subheader('같은 Coverage에서 비교')
st.caption('테스트 점수 순위로 같은 수를 판단한 사후 분석입니다. 선택에 테스트 정답을 사용하지 않았습니다. '
           'Coverage가 같아도 모델이 판단한 표본은 다를 수 있습니다. 운영 임계값 분석과 구분합니다.')
coverage = read_table(bundle, 'coverage_metrics')
if coverage is None or coverage.frame.empty:
    st.info('이 결과 묶음에는 동일 coverage 측정 파일이 없습니다. 다른 묶음의 결과로 대신 채우지 않습니다.')
else:
    for notice in coverage.notices:
        st.info(notice)
    cov = select_rows(coverage.frame, pair, scenario, gate, seed,
                      method='test_score_top_k_no_labels')
    st.caption(f'이 섹션의 안전장치 {"ON" if gate else "OFF"}: 입력 오류 강제 보류만 적용하는 설정입니다. '
               '기본 비교의 고정 0.2/0.8 점수 보류와 다릅니다.')
    targets = sorted(cov.target_coverage.unique())
    if not targets:
        st.info('선택한 모델/시드/센서/안전장치에 해당하는 동일 coverage 측정이 없습니다.')
    else:
        if st.session_state.get('lab_target') not in targets:
            st.session_state['lab_target']=.5 if .5 in targets else targets[0]
        target = st.select_slider('목표 Coverage', options=targets, key='lab_target')
        cov_selected = select_rows(cov, pair, scenario, gate, seed, target,
                                   'test_score_top_k_no_labels')
        show_comparison(cov_selected, pair, expected, average, 'matched')
        with st.expander('동일 Coverage 원본 행 · 클래스별 Coverage · feasible 확인'):
            st.dataframe(cov_selected, hide_index=True, use_container_width=True)
            st.caption('feasible=False는 해당 목표가 입력 오류 강제 보류 때문에 달성 불가능하다는 기록입니다. '
                       '일부 시드만 가능해도 이를 전체 평균으로 표시하지 않습니다.')

with st.expander('별도 분석: 검증 데이터 기반 고정 임계값'):
    st.caption('검증 점수로 임계값을 정한 뒤 테스트에 고정 적용한 결과입니다. 테스트 Coverage는 같지 않을 수 있습니다. '
               '상단의 학습 시드·모델·센서 조건을 적용하며, 이 파일은 안전장치 OFF 조건만 기록합니다.')
    thresholds = read_table(bundle, 'validation_threshold_metrics')
    if thresholds is None or thresholds.frame.empty:
        st.info('이 묶음에는 검증 임계값 분석 파일이 없습니다.')
    else:
        threshold_rows = select_rows(thresholds.frame, pair, scenario, False, seed,
                                     method='validation_threshold_test_coverage_not_guaranteed')
        st.caption('안전장치 OFF 고정 · 상단 ON/OFF와 별개인 분석임을 표시합니다.')
        if threshold_rows.empty:
            st.info('선택한 모델·시드·센서의 측정값이 없습니다.')
        else:
            targets = sorted(threshold_rows.target_coverage.unique())
            if st.session_state.get('lab_val_target') not in targets:
                st.session_state['lab_val_target']=.5 if .5 in targets else targets[0]
            target = st.selectbox('검증 목표 Coverage', targets, key='lab_val_target')
            threshold_rows = select_rows(threshold_rows, pair, scenario, False, seed, target)
            show_comparison(threshold_rows, pair, expected, average, 'validation')
            st.dataframe(threshold_rows, hide_index=True, use_container_width=True)

with st.expander('선택한 기본 비교 원본 · 파일 출처 · 시드 정보'):
    st.write('이번 화면이 읽은 기본 지표 파일:', list(table.sources))
    st.dataframe(selected, hide_index=True, use_container_width=True)
    st.download_button('선택한 원본 행 다운로드', selected.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f'lab_{bundle.name}_{seed if seed is not None else "all"}.csv', mime='text/csv')
    provenance = []
    for s in expected:
        info_path = bundle / f'seed{s}' / 'experiment_info.json'
        try:
            info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.is_file() else {}
            provenance.append({'학습 시드': s, '재배선 시드 (기록값)': info.get('rewire_seed','기록 없음'),
                               'Python (기록값)': info.get('python','기록 없음')})
        except (OSError, ValueError):
            provenance.append({'학습 시드': s, '재배선 시드 (기록값)': '정보 파일 읽기 실패'})
    st.dataframe(pd.DataFrame(provenance), hide_index=True)
    st.caption('회로 선택 시드는 그래프 추출 설정입니다. 여기서 고르는 학습 시드로 대체하거나 추정하지 않습니다.')

st.divider()
st.subheader('연구상 한계')
st.markdown('''BANC 연결 구조는 실제 데이터에서 추출한 VNC 부분 연결망입니다. GABA/ACh 부호 제약은
신경전달물질 **예측**을 활용한 공학적 모델링 가정이며, FlyBrake가 실제 초파리 뇌 활동을 완전히 재현하지는 않습니다.
UCI Occupancy Detection은 **사무실 재실 데이터**로 학교 현장에서 검증된 데이터가 아닙니다.
센서 오류 조건은 인위적으로 주입한 장애입니다. 오경보율 감소가 곧 안전성 전체의 향상을 의미하지 않습니다.
5개 시드의 평균±표준편차는 통계적 유의성을 입증하지 않습니다. 기존 결과와 수정 결과를 혼합하지 않습니다.''')

with st.expander('기본 안전장치 ON/OFF의 절충 관계 자세히 보기'):
    show_gate_comparison(frame,pair,scenario,expected,'research',seed)
