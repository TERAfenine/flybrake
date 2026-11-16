"""Four guided experiments; all calculations delegate to existing research APIs."""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0,str(PROJECT))
from src.lab_ux import setup,navigation,glossary,model_name,HELP,selection_banner,score_figure,page_link,hold_explanation
from src.lab_ablation import (load_experiment,load_graph,graph_path,file_signature,compare_input,demo_input)
from src.lab_network import network_figure,network_layout,node_from_event,neuron_connections,visible_connections
from src.lab_results import load_results,source_signature,select_rows,comparison_view,display_value,SCENARIOS
from src.lab_charts import comparison_figure
from src.lab_experience import reset_guide,filter_indices,graph_controls,network_legend,show_gate_comparison

BASE=Path(os.environ.get('FLYBRAKE_ROOT',PROJECT))
st.set_page_config(page_title='FlyBrake · 처음 사용하기',page_icon='🧭',layout='wide')
setup()
navigation('pages/0_Start_Experiment.py')
st.title('처음 사용하기')
st.caption('센서 오류부터 모델 비교까지 · 같은 실제 자료와 기존 계산을 사용하는 4가지 실험')
with st.sidebar:
    st.button('실험 초기화',on_click=reset_guide,help='현재 단계를 유지하며 뉴런·센서·보류·필터를 기본값으로 복원합니다.')
if st.session_state.pop('ux_reset_notice',False):
    st.success('입력과 선택을 기본값으로 복원했습니다. 현재 실험 단계는 유지합니다.')
STEPS=['1 · 센서 오류','2 · 연결망 탐색','3 · 뉴런 비활성화','4 · 모델 비교']
step=st.radio('실험 단계',STEPS,key='ux_step',horizontal=True)
number=STEPS.index(step)
st.progress((number+1)/4,text=f'실험 {number+1} / 4')
st.caption('기존 결과 사용 · 단일 입력: 시드 42 · 성능: 5시드 평균. 상세 조건은 연구 결과에서 선택합니다.')
with st.expander('현재 단계의 시연 포인트 · 연구 범위'):
    st.write(['센서 오류를 고르고 보류를 켜고 끄세요. 점수와 자동 판단 여부가 별개임을 관찰합니다.', '뉴런을 클릭하세요. 선택 번호·BANC ID·입출력 연결과 방향이 함께 바뀝니다.', '활성값 0 고정을 켜세요. 점수가 거의 같거나 판단이 같아도 정상적인 관찰 결과입니다.', '오경보·미탐지를 자동 판단 비율과 함께 비교하세요. 안전장치 ON/OFF 표는 같은 모델과 같은 조건을 사용합니다.'][number])
    st.caption('단일 입력은 results/seed42의 banc_signed, 평가 비교는 기존 results의 저장된 5시드입니다. 실제 신경 기능이나 학교 현장 안전성을 검증하는 실험이 아닙니다.')

@st.cache_resource(show_spinner='저장된 모델과 전처리 객체를 읽는 중…')
def experiment_cached(run,root,signature):
    return load_experiment(run,root)

@st.cache_data(show_spinner='실제 연결망을 읽는 중…')
def graph_cached(path,signature):
    graph=load_graph(path)
    return graph,network_layout(graph)

@st.cache_data(show_spinner='저장된 측정값을 읽는 중…')
def results_cached(bundle,signature):
    return load_results(bundle,'metrics')

def set_step(value):
    st.session_state['ux_step']=value

def reset_node():
    st.session_state['ux_disabled']=False

def sensor_controls():
    scenario=st.selectbox('센서에 어떤 오류를 넣을까요?', ['normal','missing_light','co2_spike'],
        format_func=lambda s:{'normal':'오류 없음 · 입력값 그대로','missing_light':'조도 센서 값 누락','co2_spike':'CO₂ 센서 값 +3,000 ppm'}[s],key='ux_scenario')
    st.session_state.setdefault('ux_guard',True)
    guard=st.checkbox('불확실하면 판단 보류하기',key='ux_guard',
        help='ON: 결손·학습 범위 밖 입력 또는 0.2 < 점수 < 0.8이면 보류. OFF: 점수 0.5 기준으로 판단합니다.')
    with st.expander('센서 기본값 직접 바꾸기'):
        values=[]
        for label,default,maximum in [('온도 (°C)',22.5,50.),('습도 (%)',45.,100.),('조도 (lux)',150.,20000.),('CO₂ (ppm)',700.,20000.)]:
            st.session_state.setdefault('ux_sensor_'+label,default)
            values.append(st.number_input(label,min_value=0.,max_value=maximum,key='ux_sensor_'+label))
    st.button('입력·뉴런 설정 초기화',key='guide_reset',on_click=reset_guide)
    return values,scenario,guard

def load_demo():
    run=BASE/'results/seed42'
    path=graph_path(run,BASE)
    return experiment_cached(str(run),str(BASE),file_signature([path,run/'banc_signed.pt',run/'sensor_prep.joblib',run/'experiment_info.json']))

def decision_card(title,score,outcome):
    with st.container(border=True):
        st.markdown(f'**{title}**')
        st.metric('모델 출력 점수',f'{score:.8f}')
        st.write(f'판단: **{outcome[0]}**')
        st.caption(f'판단 보류: {"예" if outcome[1] else "아니오"} · {outcome[2]}')

def sensor_step():
    st.header('센서가 고장 나면 AI는 어떻게 판단할까?')
    st.write('① 센서 오류를 고르세요. ② 판단 보류를 켜고 꺼 보세요. 아래에서 같은 기본 입력의 오류 전·후를 비교합니다.')
    values,scenario,guard=sensor_controls()
    try:
        experiment=load_demo()
        normal=compare_input(experiment,demo_input(values,'normal'),guard=guard)
        changed=compare_input(experiment,demo_input(values,scenario),guard=guard)
    except Exception as error:
        st.warning(f'이 모델의 입력 비교를 실행하지 못했습니다: {error}')
        st.info('파일과 실행 환경을 확인하거나 연구자 모드에서 저장된 성능 결과를 살펴보세요.')
        return
    hold_explanation()
    a,b=st.columns(2)
    with a: decision_card('오류 주입 전',normal['original'],normal['original_decision'])
    with b: decision_card('선택한 센서 조건',changed['original'],changed['original_decision'])
    st.plotly_chart(score_figure(['오류 주입 전','선택한 센서 조건'],[normal['original'],changed['original']],
        '같은 입력의 센서 오류 전·후'),use_container_width=True,key='ux_sensor_plot',config={'displayModeBar':False})
    st.caption(f'results / seed42 / banc_signed · {SCENARIOS[scenario]} · 판단 보류 {"ON" if guard else "OFF"} · 점수는 보정된 실제 재실 확률이 아닙니다.')
    if changed['original_decision'][1]:
        st.info('선택한 조건에서 자동 판단을 보류했습니다. 보류는 정답을 맞혔다는 뜻이 아니라, 자동 결정을 내리지 않았다는 뜻입니다.')
    elif changed['original_decision'][0]==normal['original_decision'][0]:
        st.info('두 조건의 판단은 같습니다. 점수 차이만으로 성능이 좋아지거나 나빠졌다고 말할 수 없습니다.')
    else:
        st.info('이 입력에서 두 조건의 판단이 달라졌습니다. 정답 라벨이 없으므로 어느 판단이 옳은지는 이 시연으로 알 수 없습니다.')
    with st.expander('계산에 사용한 입력과 시연 범위'):
        st.dataframe(pd.DataFrame(np.concatenate([demo_input(values),demo_input(values,scenario)]),
            columns=['Temperature','Humidity','Light','CO2'],index=['오류 주입 전','선택 조건']),use_container_width=True)
        st.write('오류 주입 전에도 입력이 학습 범위 밖이면 보류할 수 있습니다. 이 시연은 단일 입력에 오류를 항상 적용합니다. 기존 전체 평가의 약 25% 오류 주입과 다릅니다.')

def neuron_step(ablation):
    st.header('뉴런을 비활성화하면 AI 판단이 바뀔까?' if ablation else '실제 초파리 뉴런의 연결을 탐색해 보자')
    st.write('뉴런을 고르면 선택 표시와 연결 정보가 함께 바뀝니다. 큰 테두리는 선택, 작은 점은 주변 뉴런, 빨간 ×는 계산상 비활성화입니다.')
    try:
        path=graph_path(BASE/'results/seed42',BASE)
        graph,positions=graph_cached(str(path),file_signature([path]))
    except (OSError,ValueError,KeyError) as error:
        st.warning(f'실제 연결망을 읽을 수 없습니다: {error}')
        return
    nt_filter,indices=filter_indices(graph,'ux','ux_node','ux_disabled')
    index=st.selectbox('탐색할 뉴런',indices,key='ux_node',on_change=reset_node,
        format_func=lambda i:f'뉴런 {i} · BANC {graph["ids"][i]}')
    if ablation:
        disabled=st.checkbox('선택한 뉴런의 활성값을 0으로 고정',key='ux_disabled')
        st.caption('한 번에 한 뉴런만 개입합니다. 다른 뉴런을 고르면 개입을 해제합니다. 저장된 연결·가중치는 그대로입니다.')
    else:
        disabled=st.session_state.get('ux_disabled',False)
        if disabled:
            st.button('이전 실험의 비활성화 설정 해제',on_click=reset_node)
    selection_banner(index,graph['ids'][index],disabled,graph)
    if ablation:
        values,scenario,guard=sensor_controls()
        hold_explanation()
        try:
            experiment=load_demo()
            result=compare_input(experiment,demo_input(values,scenario),index if disabled else None,guard)
        except Exception as error:
            st.warning(f'비활성화 계산을 실행하지 못했습니다: {error}')
        else:
            a,b=st.columns(2)
            with a: decision_card('원래 모델',result['original'],result['original_decision'])
            with b: decision_card('비활성화 적용' if disabled else '비활성화 OFF',result['altered'],result['altered_decision'])
            st.metric('점수 변화량 (개입 후 − 원래)',f'{result["delta"]:+.8f}',help='양수·음수는 점수 변화 방향입니다. 개선·악화를 의미하지 않습니다.')
            st.plotly_chart(score_figure(['원래 모델','개입 후' if disabled else '비활성화 OFF'],[result['original'],result['altered']],
                '선택한 한 입력의 계산 결과'),use_container_width=True,key='ux_ablation_plot',config={'displayModeBar':False})
            same=result['original_decision'][0]==result['altered_decision'][0]
            st.info(f'이 입력에서 판단은 {"같습니다" if same else "달라졌습니다"}. 점수 변화량은 {result["delta"]:+.8f}입니다. '
                '실제 초파리 신경 기능이나 전체 평가 성능의 변화를 뜻하지 않습니다.')
            st.caption(f'results / seed42 · BANC {graph["ids"][index]} · {SCENARIOS[scenario]} · 보류 {"ON" if guard else "OFF"} · 개입 뉴런 {result["disabled_count"]}개 · 점수는 확률 보정 미검증')
    with st.expander('개입된 뉴런의 연결망 다시 보기',expanded=False) if ablation else st.container():
        mode=st.selectbox('연결 범위', ['선택 뉴런 중심',f'전체 {len(graph["ids"])}개 뉴런','GABA 예측 강조'],key='ux_graph_mode')
        zoom,revision=graph_controls('ux','ux_graph_mode')
        chart_key='ux_network_'+str(number)+'_'+mode+'_'+nt_filter+'_'+str(revision)
        def picked():
            chosen=node_from_event(st.session_state.get(chart_key,{}),graph['ids'])
            if chosen is not None and chosen!=st.session_state['ux_node']:
                st.session_state['ux_node']=chosen
                reset_node()
        st.plotly_chart(network_figure(graph,index,mode,index if disabled else None,positions,nt_filter,zoom,revision),
            use_container_width=True,key=chart_key,on_select=picked,selection_mode='points',config={'displayModeBar':False})
        with st.container(key='guide_graph_selection'):
            selection_banner(index,graph['ids'][index],disabled,graph)
        nodes,post,pre=visible_connections(graph,index,mode,nt_filter)
        network_legend(nt_filter,len(nodes),len(post))
        st.caption('클릭: 뉴런 선택 · 드래그: 이동 · 확대/축소: 선택 뉴런 중심 · 보기 초기화: 표시 범위에 맞추기. 클릭이 어려우면 위의 탐색할 뉴런 선택기를 사용하세요. 선/화살표 클릭은 뉴런 선택이 아닙니다.')
        st.caption(f'BANC VNC 부분 연결망 · {len(graph["ids"])}개 뉴런 / {int(graph["mask"].sum()):,}개 방향성 연결 중 {mode} · 화살표: 출발 → 도착 · 보라: GABA 예측 / 주황: ACh 예측. 배치는 해부학적 좌표가 아닙니다.')
        st.caption('제공된 BANC 파일의 연결·ID를 표시합니다. 외부 원본 인증은 별도 필요하며, GABA 음수 / ACh 양수 부호 제약은 공학적 가정입니다.')
    with st.expander('연결된 BANC ID · 방향 · 시냅스 수 보기'):
        st.dataframe(neuron_connections(graph,index),hide_index=True,use_container_width=True)
        st.caption('시냅스 수는 연결 데이터의 값이며 학습된 AI 가중치가 아닙니다.')

def comparison_step():
    st.header('어떤 모델이 더 안전할까?')
    st.write('오경보·미탐지가 줄어도 자동 판단을 더 많이 포기했을 수 있습니다. 네 지표를 함께 비교하세요.')
    scenario=st.selectbox('비교할 센서 조건',list(SCENARIOS),format_func=SCENARIOS.get,key='ux_compare_scenario')
    st.session_state.setdefault('ux_compare_guard',True)
    guard=st.checkbox('성능 비교의 판단 보류 안전장치 ON',key='ux_compare_guard')
    try:
        table=results_cached(str(BASE/'results'),source_signature(BASE/'results','metrics'))
        frame=table.frame
        if frame.empty:
            st.info('이 묶음에는 저장된 성능 결과가 없습니다.'); return
        models=['banc_signed','logistic','rewired_signed']
        seeds=sorted(frame.seed.unique())
        view=comparison_view(select_rows(frame,models,scenario,guard),models,seeds,True)
    except (OSError,ValueError,KeyError,TypeError) as error:
        st.warning(f'저장된 성능 결과를 표시할 수 없습니다: {error}'); return
    for notice in table.notices: st.info(notice)
    metrics=['false_alarm_per_empty','unsafe_false_empty_per_occupied','coverage','abstain_rate']
    names=['오경보율','미탐지율','자동 판단 비율','판단 보류율']
    st.caption(f'기존 결과 results · {len(seeds)}개 시드 평균 ± 표본 표준편차 · {SCENARIOS[scenario]} · 안전장치 {"ON" if guard else "OFF"}. 아래는 저장된 평가 결과이며 앞 단계의 단일 입력/비활성화 결과가 아닙니다.')
    with st.expander('모델별 상세 수치 · 평균과 표준편차'):
        for col,model in zip(st.columns(3),models):
            with col,st.container(border=True):
                st.markdown(f'**{model_name(model)}**')
                group=view[view.model.eq(model)].set_index('metric')
                for metric,label in zip(metrics,names):
                    row=group.loc[metric]
                    st.metric(label,display_value(row.value,row['std'],True),help=HELP[metric])
                for state in group.status.unique():
                    if state!='측정값': st.info(state+' · N/A를 0으로 해석하지 마세요.')
    for subset,title,key in [(metrics[:2],'오경보·미탐지 비교','errors'),
                              (metrics[2:],'자동 판단·판단 보류 비교','coverage')]:
        st.caption('오경보: 빈 공간을 재실로 오판 · 미탐지: 사람이 있는 공간을 비었다고 오판. 각 실제 클래스 전체가 분모입니다.' if key=='errors' else '자동 판단 비율(Coverage)과 판단 보류율의 합은 100%입니다. 전체 입력이 분모입니다.')
        fig=comparison_figure(view,subset,True)
        for trace in fig.data: trace.name=model_name(trace.name)
        fig.update_layout(title=dict(text=title,font=dict(size=16)),height=440,margin=dict(t=65,b=110),legend=dict(y=-.3,yanchor='top',font=dict(size=10)),
                          yaxis_title='표본 비율 (%)')
        st.plotly_chart(fig,use_container_width=True,key='ux_compare_'+key,config={'displayModeBar':False})
    show_gate_comparison(frame,models,scenario,seeds,'ux')
    st.info('이 조건의 차이는 모델 간 관찰된 차이입니다. 특정 모델이 항상 우수하다는 뜻이 아니며, 5시드 표준편차는 통계적 유의성을 입증하지 않습니다.')
    page_link('pages/1_FlyBrake_Lab.py',label='7개 모델·같은 자동 판단 비율로 더 자세히 비교 →',use_container_width=True)
    st.caption('연구자 모드에서 revised_audit_20261024 묶음을 선택하면 7개 모델과 동일 Coverage 분석을 볼 수 있습니다. 기존 결과와 수정 결과는 합치지 않습니다.')

if number==0: sensor_step()
elif number in (1,2): neuron_step(number==2)
else: comparison_step()
glossary()
st.divider()
left,right=st.columns(2)
left.button('← 이전 실험',disabled=number==0,on_click=set_step,args=(STEPS[max(0,number-1)],),use_container_width=True)
right.button('다음 실험 →' if number<3 else '처음부터 다시 살펴보기',type='primary',on_click=set_step,
             args=(STEPS[number+1] if number<3 else STEPS[0],),use_container_width=True)
st.caption('실제 초파리 신경 기능을 입증하는 생물학 실험이 아닙니다. 사무실 데이터와 합성 오류를 사용한 AI 계산 실험이며 학교 현장 안전성을 검증하지 않았습니다.')
