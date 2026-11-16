"""Read-only presentation helpers for graph controls and stored ON/OFF comparisons."""
import numpy as np
import pandas as pd
import streamlit as st
from src.lab_results import select_rows,comparison_view,display_value
from src.lab_ux import model_name

def reset_guide():
    defaults={'ux_node':0,'ux_disabled':False,'ux_scenario':'normal','ux_guard':True,
        'ux_graph_mode':'선택 뉴런 중심','ux_nt_filter':'전체','ux_compare_scenario':'normal',
        'ux_compare_guard':True,'ux_tradeoff_model':'banc_signed'}
    for label,value in [('온도 (°C)',22.5),('습도 (%)',45.),('조도 (lux)',150.),('CO₂ (ppm)',700.)]:
        defaults['ux_sensor_'+label]=value
    st.session_state.update(defaults)
    st.session_state['ux_zoom']=1.
    st.session_state['ux_view_revision']=st.session_state.get('ux_view_revision',0)+1
    st.session_state['ux_reset_notice']=True

def filter_indices(graph,prefix,node_key,disabled_key):
    nt=st.selectbox('표시할 전달물질 예측',['전체','GABA 예측만'],key=prefix+'_nt_filter',
        help='GABA 예측만: GABA로 예측된 뉴런과 그 사이의 실제 연결만 표시합니다. ACh와의 연결은 화면에서만 숨깁니다.')
    indices=np.flatnonzero(graph['signs']<0).tolist() if nt=='GABA 예측만' else list(range(len(graph['ids'])))
    if not indices:
        st.info('이 부분 연결망에는 GABA 예측 뉴런이 없습니다. 전체 필터를 선택하세요.')
        st.stop()
    if st.session_state.get(node_key,indices[0]) not in indices:
        st.session_state[node_key]=indices[0]
        st.session_state[disabled_key]=False
        st.caption('필터에 맞는 첫 뉴런으로 선택을 옮기고 개입을 해제했습니다.')
    return nt,indices

def graph_controls(prefix,mode_key=None):
    def adjust(factor):
        st.session_state[prefix+'_zoom']=min(4.,max(.5,st.session_state.get(prefix+'_zoom',1.)*factor))
    def fit():
        st.session_state[prefix+'_zoom']=1.
        st.session_state[prefix+'_view_revision']=st.session_state.get(prefix+'_view_revision',0)+1
    def focus():
        st.session_state[mode_key]='선택 뉴런 중심'
        fit()
    if mode_key and st.session_state.get(mode_key)!='선택 뉴런 중심':
        st.button('선택 뉴런 중심으로 보기',key=prefix+'_focus',on_click=focus,use_container_width=True)
    with st.container(key=prefix+'_graph_tools'):
        a,b,c=st.columns(3)
        a.button('＋ 확대',key=prefix+'_zoom_in',on_click=adjust,args=(1.5,),use_container_width=True)
        b.button('− 축소',key=prefix+'_zoom_out',on_click=adjust,args=(1/1.5,),use_container_width=True)
        c.button('보기 초기화',key=prefix+'_fit',on_click=fit,use_container_width=True)
    return st.session_state.get(prefix+'_zoom',1.),st.session_state.get(prefix+'_view_revision',0)

def network_legend(nt_filter,nodes,edges):
    st.markdown('<div class="ux-legend"><strong>● 보라: GABA 예측</strong> · <strong>● 주황: ACh 예측</strong><br>'
        '<strong>선택 번호 + 큰 점</strong>: 현재 선택 · <strong>× 빨강</strong>: 계산상 비활성화<br>'
        '<strong>파란 선: 선택 뉴런으로 들어옴</strong> · <strong>갈색 선: 선택 뉴런에서 나감</strong><br>'
        '선 위 삼각형이 출발 → 도착 방향을 가리킵니다.</div>',unsafe_allow_html=True)
    st.caption(f'현재 표시: {nodes}개 뉴런 · {edges}개 실제 연결 · 필터: {nt_filter}. '
        '필터는 화면에만 적용하며 원본 연결과 AI 계산은 바꾸지 않습니다. 입출력 연결 수 카드는 원본 96개 부분 연결망 기준입니다.')

TRADEOFF_METRICS=['false_alarm_per_empty','unsafe_false_empty_per_occupied','coverage','abstain_rate']
TRADEOFF_LABELS=['오경보율','미탐지율','자동 판단 비율','판단 보류율']

def gate_comparison(frame,model,scenario,seeds,seed=None):
    """Same model/condition/seeds, existing strict aggregation, display ON minus OFF."""
    views={gate:comparison_view(select_rows(frame,[model],scenario,gate,seed),[model],seeds,seed is None).set_index('metric')
           for gate in [False,True]}
    rows=[]
    for metric,label in zip(TRADEOFF_METRICS,TRADEOFF_LABELS):
        off,on=views[False].loc[metric],views[True].loc[metric]
        rows.append({'metric':metric,'지표':label,'OFF':off.value,'ON':on.value,
            '변화 (%p)':100*(on.value-off.value),'OFF 상태':off.status,'ON 상태':on.status})
    return pd.DataFrame(rows)

def show_gate_comparison(frame,models,scenario,seeds,prefix,seed=None):
    st.subheader('안전장치를 켜면 무엇이 달라질까?')
    model_key=prefix+'_tradeoff_model'
    if model_key in st.session_state and st.session_state[model_key] not in models:
        st.session_state[model_key]=models[0]
    model=st.selectbox('안전장치 효과를 볼 모델',models,format_func=model_name,key=model_key)
    view=gate_comparison(frame,model,scenario,seeds,seed)
    shown=view[['지표']].copy()
    for gate in ['OFF','ON']:
        shown[gate]=view[gate].map(lambda x:display_value(x,np.nan,False))
    shown['변화 (%p)']=view['변화 (%p)'].map(lambda x:f'{x:+.3f}' if np.isfinite(x) else 'N/A')
    st.table(shown.set_index('지표'))
    st.caption('OFF: 점수 0.5 기준 자동 판단 · ON: 입력 오류 또는 0.2 < 점수 < 0.8이면 보류. 차이는 반올림 전 값에서 계산합니다.')
    st.caption(f'{model_name(model)} · 같은 센서 조건({scenario}) · '+
        ('같은 시드들의 평균' if seed is None else f'시드 {seed}')+' · 변화는 ON − OFF. 상단 체크박스와 관계없이 두 저장 조건을 함께 비교합니다.')
    for column in ['OFF 상태','ON 상태']:
        for state in view[column].unique():
            if state!='측정값':st.info(f'{column}: {state}. 누락 값을 추정하지 않습니다.')
    st.write('오경보·미탐지가 감소해도 보류가 늘면 자동으로 처리한 입력이 줄어든 것입니다. '
             '오류율만으로 안전성을 순위 매기지 마세요. 보류는 정답을 맞힌 것으로 세지 않습니다.')
    st.caption('평균 차이는 인과효과나 유의성 검정이 아닙니다. 기본 안전장치의 입력 오류·0.2/0.8 규칙이며 동일 Coverage 사후 분석과 구분합니다.')
