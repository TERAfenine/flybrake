"""BANC exploration and single-input, single-neuron computational intervention."""
from pathlib import Path
import inspect
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0,str(PROJECT))
from src.lab_ablation import (run_folders,graph_path,file_signature,load_graph,
                              load_experiment,compare_input,demo_input)
from src.lab_network import network_layout,network_figure,neuron_connections,node_from_event,visible_connections
from src.lab_ux import setup,navigation,selection_banner,score_figure,hold_explanation
from src.lab_experience import filter_indices,graph_controls,network_legend

BASE=Path(os.environ.get('FLYBRAKE_ROOT',PROJECT))
st.set_page_config(page_title='FlyBrake Lab · 뉴런 실험',page_icon='🧠',layout='wide')
setup()
navigation('pages/2_FlyBrake_Neurons.py')
st.title('FlyBrake Lab · 뉴런 실험')
st.caption('연구자 모드 · 왼쪽 설정에서 시드·뉴런을 선택하세요. 좁은 화면은 왼쪽 위 화살표로 설정을 펼칩니다.')
st.caption('실제 BANC 연결망 탐색 · 학습된 banc_signed 모델의 단일 뉴런 활성값 개입 · 재학습 없음')
st.info('본 실험은 실제 BANC 연결 데이터를 이용해 학습한 인공신경망에서 일부 뉴런의 활성값을 제한하는 계산 실험입니다. '
        '실제 초파리에서 해당 뉴런을 제거하거나 기능을 억제한 생물학적 실험이 아니며, AI의 출력 변화가 초파리의 실제 신경 기능을 입증하는 것은 아닙니다.')

@st.cache_data(show_spinner='실제 연결망을 읽는 중…')
def cached_graph(path,signature):
    graph=load_graph(path)
    return graph,network_layout(graph)

@st.cache_resource(show_spinner='저장된 모델·전처리 호환성을 확인하는 중…')
def cached_experiment(run,project,signature):
    return load_experiment(run,project)

def reset_inputs():
    st.session_state.update(neuron_index=0,neuron_disabled=False,neuron_scenario='normal',neuron_guard=True,neuron_nt_filter='전체',neuron_view='선택 뉴런 중심',neuron_zoom=1.)
    for label,value in [('Temperature (°C)',22.5),('Humidity (%)',45.),('Light (lux)',150.),('CO₂ (ppm)',700.)]:
        st.session_state['sensor_'+label]=value
    st.session_state['neuron_view_revision']=st.session_state.get('neuron_view_revision',0)+1

def reset_ablation():
    st.session_state['neuron_disabled']=False

bundles=run_folders(BASE/'results')
if not bundles:
    st.warning('시드별 banc_signed.pt와 sensor_prep.joblib가 함께 있는 결과 폴더가 없습니다.')
    st.stop()
with st.sidebar:
    st.header('뉴런 실험 설정')
    bundle=st.selectbox('체크포인트 묶음',list(bundles),key='neuron_bundle',on_change=reset_ablation,
                        format_func=lambda p:'기존 결과 · results' if p==BASE/'results' else p.name)
    seed=st.selectbox('학습 시드',list(bundles[bundle]),key=f'neuron_seed_{bundle.name}',on_change=reset_ablation)
    st.caption('모델: banc_signed 고정. 전체 시드 평균이나 다른 모델의 전처리 객체를 사용하지 않습니다.')
run=bundles[bundle][seed]
try:
    path=graph_path(run,BASE)
    graph,positions=cached_graph(str(path),file_signature([path]))
except (OSError,ValueError,KeyError) as error:
    st.warning(f'실제 연결망을 읽을 수 없습니다: {error}')
    st.stop()
n=len(graph['ids'])
context=str(run)
if st.session_state.get('neuron_run_context')!=context:
    st.session_state['neuron_run_context']=context
    st.session_state['neuron_index']=0
    reset_ablation()
with st.sidebar:
    nt_filter,indices=filter_indices(graph,'neuron','neuron_index','neuron_disabled')
    index=st.selectbox('뉴런 선택',indices,key='neuron_index',on_change=reset_ablation,
                       format_func=lambda i:f'{i} · BANC {graph["ids"][i]}')
    st.session_state.setdefault('neuron_view','선택 뉴런 중심')
    mode=st.selectbox('연결망 보기',[f'전체 {n}개 뉴런','선택 뉴런 중심','GABA 예측 강조'],key='neuron_view')
    disabled=st.checkbox('선택 뉴런 비활성화 ON',key='neuron_disabled')
    st.caption('뉴런이나 체크포인트를 바꾸면 비활성화가 OFF로 초기화됩니다. ON은 각 재귀 단계의 활성값을 0으로 고정합니다.')
    st.button('비활성화 해제',on_click=reset_ablation)
    st.button('입력·뉴런 설정 초기화',on_click=reset_inputs,help='현재 체크포인트·시드는 유지하고 뉴런과 센서 입력·보류·필터를 기본값으로 복원합니다.')

st.subheader('뉴런 탐색')
selection_banner(index,graph['ids'][index],disabled,graph)
a,b,c=st.columns(3)
a.metric('실제 뉴런',n);b.metric('방향성 연결',int(graph['mask'].sum()));c.metric('GABA 예측 뉴런',int((graph['signs']<0).sum()))
st.caption('화살표: 출발(pre) → 도착(post), mask[post, pre]. 좌표는 연결 관계를 보기 위한 배치이며 해부학적 위치가 아닙니다. '
           '비활성화 표시가 있어도 원본 연결선과 NPZ는 삭제하거나 변경하지 않습니다.')
st.caption('범례: 보라 = GABA 예측 · 주황 = ACh 예측 · 큰 점과 진한 테두리 = 선택 · 청록 테두리 = 직접 연결된 이웃 · 빨간 × = 비활성화. 선택 중심 보기로 복잡도를 줄일 수 있습니다.')
zoom,revision=graph_controls('neuron','neuron_view')
chart_key=f'neuron_graph_{bundle.name}_{seed}_{mode}_{nt_filter}_{revision}'
def pick_node():
    chosen=node_from_event(st.session_state.get(chart_key,{}),graph['ids'])
    if chosen is not None and chosen!=st.session_state['neuron_index']:
        st.session_state['neuron_index']=chosen
        reset_ablation()

figure=network_figure(graph,index,mode,index if disabled else None,positions,nt_filter,zoom,revision)
if 'on_select' in inspect.signature(st.plotly_chart).parameters:
    st.plotly_chart(figure,use_container_width=True,key=chart_key,on_select=pick_node,selection_mode='points')
    st.caption('노드 한 개를 클릭하여 선택하세요. 선택되지 않으면 왼쪽 뉴런 드롭다운을 사용하세요. 선/화살표 클릭은 뉴런 선택으로 처리하지 않습니다.')
else:
    st.plotly_chart(figure,use_container_width=True,key=chart_key)
    st.caption('현재 Streamlit은 그래프 선택 이벤트를 지원하지 않습니다. 왼쪽 뉴런 드롭다운을 사용하세요.')

nodes,post,pre=visible_connections(graph,index,mode,nt_filter)
network_legend(nt_filter,len(nodes),len(post))
selected_id=str(graph['ids'][index])
st.write(f'**선택 뉴런 BANC ID: {selected_id}** · 모델 인덱스 {index} · '
         f'{"GABA" if graph["signs"][index]<0 else "ACh"} 예측 · '
         f'{"활성값 0 고정 요청" if disabled else "비활성화 OFF"}')
a,b=st.columns(2)
a.metric('들어오는 연결',int(graph['mask'][index].sum()))
b.metric('나가는 연결',int(graph['mask'][:,index].sum()))
links=neuron_connections(graph,index)
with st.expander('연결된 뉴런 · 방향 · 시냅스 수',expanded=False):
    if links.empty:
        st.info('추출된 부분 그래프 안에는 해당 뉴런의 입출력 연결이 없습니다.')
    else:
        st.dataframe(links,hide_index=True,use_container_width=True)
    if 'counts' not in graph:
        st.caption('이 NPZ에는 시냅스 수가 저장되어 있지 않아 N/A입니다.')
    st.caption('표의 시냅스 수는 원본 연결 정보입니다. 학습된 AI 가중치의 크기와 동일하지 않습니다.')

st.divider()
st.subheader('뉴런 비활성화 · 단일 입력 비교')
try:
    signature=file_signature([path,run/'banc_signed.pt',run/'sensor_prep.joblib',run/'experiment_info.json'])
    experiment=cached_experiment(str(run),str(BASE),signature)
except Exception as error:
    # Local artifact load errors must not replace the experiment with other data.
    st.warning(f'이 체크포인트의 추론을 실행하지 않았습니다: {error}')
    st.caption('연결망의 선택/표시는 가능하지만 현재 비활성화 요청의 AI 출력은 계산되지 않았습니다.')
    st.stop()

st.caption('점수는 sigmoid 출력이며 보정된 재실 확률이 아닙니다. 동일한 저장 전처리 객체와 동일 입력을 두 계산에 사용합니다.')
inputs=st.columns(4)
values=[]
for col,label,maximum,default in zip(inputs,['Temperature (°C)','Humidity (%)','Light (lux)','CO₂ (ppm)'],
                                    [50.,100.,20000.,20000.],[22.5,45.,150.,700.]):
    with col:
        st.session_state.setdefault('sensor_'+label,default)
        values.append(st.number_input(label,min_value=0.,max_value=maximum,key='sensor_'+label))
scenario=st.selectbox('단일 입력 시연 조건',['normal','missing_light','co2_spike'],key='neuron_scenario',
                      format_func=lambda s:{'normal':'입력값 그대로','missing_light':'이 입력의 조도 결손','co2_spike':'이 입력의 CO₂ +3000 ppm'}[s])
st.session_state.setdefault('neuron_guard',True)
guard=st.checkbox('판단 보류 안전장치 ON',key='neuron_guard')
hold_explanation()
st.caption('이 시연은 선택한 한 입력에 장애를 항상 적용합니다. 기존 평가의 “전체 표본 중 난수로 약 25% 선택”과 다른 단일 입력 시연입니다. '
           '안전장치 ON: 결손/범위 밖 입력 또는 0.2 < 점수 < 0.8이면 보류. OFF: 0.5 기준 판단.')
raw=demo_input(values,scenario)
try:
    result=compare_input(experiment,raw,index if disabled else None,guard)
except (ValueError,RuntimeError) as error:
    st.warning(f'입력 비교를 계산할 수 없습니다: {error}')
    st.stop()
left,right=st.columns(2)
for col,label,score,decision in [(left,'원래 FlyBrake',result['original'],result['original_decision']),
                               (right,'뉴런 비활성화 FlyBrake' if disabled else '비활성화 OFF · 원본 계산',result['altered'],result['altered_decision'])]:
    with col:
        st.markdown(f'**{label}**')
        st.metric('재실 예측 점수',f'{score:.8f}')
        st.write(f'판단: **{decision[0]}** · 판단 보류: {"예" if decision[1] else "아니오"}')
        st.caption(decision[2])
left,right=st.columns(2)
left.metric('점수 변화량 (개입 후 − 원래)',f'{result["delta"]:+.8f}')
right.metric('실제 비활성화 뉴런 수',result['disabled_count'])
st.plotly_chart(score_figure(['원래 FlyBrake','개입 후' if disabled else '비활성화 OFF'],[result['original'],result['altered']],'선택한 입력의 뉴런 개입 전·후'),use_container_width=True,key='neuron_score_comparison')
st.write('두 모델의 판단은 **'+('같습니다' if result['original_decision'][0]==result['altered_decision'][0] else '달라졌습니다')+'**. 점수 변화는 이 입력에서의 계산 변화이며 성능 개선·악화나 생물학적 기능을 뜻하지 않습니다.')
st.caption(f'센서 조건: {scenario} · 판단 보류: {"ON" if guard else "OFF"} · 모델: banc_signed')
st.caption(f'학습 시드 {seed} · 선택 BANC ID {selected_id} · 각 {experiment.model.steps}단계의 활성값 제한 · 재학습/가중치 변경 없음')
if result['delta']==0 and disabled:
    st.info('이 입력에서 측정한 점수 변화는 0입니다. 입력·출력 매핑 및 계산 단계에 따라 영향이 전달되지 않을 수 있으며, 생물학적으로 기능이 없다는 뜻이 아닙니다.')
st.info('이 한 입력에는 정답 라벨과 전체 평가 표본이 없습니다. 오경보율·미탐지율·F1·Coverage의 비활성화 전후 성능은 계산하거나 표시하지 않습니다.')
with st.expander('실제 입력 · 사용한 파일 · 시드 구분'):
    st.dataframe(pd.DataFrame(raw,columns=['Temperature','Humidity','Light','CO2']),hide_index=True)
    st.code('\n'.join(str(p) for p in [path,run/'banc_signed.pt',run/'sensor_prep.joblib']))
    st.write('회로 선택 시드 (NPZ 기록):',str(graph.get('seed','기록 없음')))
    st.write('학습 시드 (선택 폴더):',seed)
    st.write('재배선 시드 (정보 파일 기록):',experiment.info.get('rewire_seed','기록 없음'))
    st.caption('이번 모델은 BANC signed 원래 연결을 사용합니다. 기록된 재배선 시드는 대조군용이며 현재 연결망을 바꾸지 않습니다. '
               '구형 체크포인트에는 뉴런 ID/학습 시드 자체가 포함되지 않아, 해당 폴더 및 NPZ와 mask/signs 일치 검사를 근거로 대응합니다.')
st.caption('GABA/ACh 부호 제약은 신경전달물질 예측을 바탕으로 한 공학적 가정입니다. '
           'GABA 뉴런 개입에 따른 AI 점수 변화는 생물학적 안전 회로를 입증하지 않습니다. 사무실 재실 모델이며 학교 현장 검증 결과가 아닙니다.')
