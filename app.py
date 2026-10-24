"""FlyBrake: measured results and supplied BANC connectivity; advisory only."""
from pathlib import Path
import os
import json
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from src.model import SignedConnectome

BASE=Path(os.environ.get('FLYBRAKE_ROOT',Path(__file__).resolve().parent))
st.set_page_config(page_title='FlyBrake',page_icon='🧠',layout='wide')
st.title('FlyBrake | 커넥톰 제약 AI 연구')
st.caption('VNC 부분 연결망 · 사무실 재실 데이터 · 합성 센서 오류 실험. 생물학적 기능 복제나 실제 안전 인증 결과가 아닙니다.')
results=BASE/'results'
options={'기존 5회 실험':results}
for folder in sorted(results.iterdir()) if results.exists() else []:
    if folder.is_dir() and (folder/'metrics_5seeds.csv').exists(): options[folder.name]=folder
selected=st.selectbox('실험 결과 묶음',list(options))
root=options[selected]
seed_dirs=[d for d in sorted(root.glob('seed*')) if d.is_dir() and (d/'banc_signed.pt').exists()]
run=st.selectbox('모델 / 그래프 확인 시드',[str(d.name) for d in seed_dirs]) if seed_dirs else None
checkpoint=root/run if run else root
path=checkpoint/'experiment_graph.npz'
if not path.exists():path=BASE/'data/banc_graph.npz'
if not path.exists():
    st.error('연결망 파일이 없습니다. build_graph.py를 먼저 실행하세요.');st.stop()
g=np.load(path,allow_pickle=False)
if str(g['source'])!='BANC v888 user-supplied inputs (not independently authenticated)':
    st.error('지원하지 않는 연결망 형식입니다.');st.stop()
a,b,c=st.columns(3)
a.metric('추출 뉴런',len(g['ids']));b.metric('방향성 연결',int(g['mask'].sum()));c.metric('GABA 예측 뉴런',int((g['signs']<0).sum()))
st.info('연결·ID는 제공된 BANC 파일에서 추출했습니다(외부 원본 인증은 별도 필요). GABA 음수/ACh 양수는 공학적 가정이며 실제 수용체 반응을 확인한 값이 아닙니다.')
t_graph,t_results,t_coverage,t_demo=st.tabs(['뉴런 연결망','모델 비교','같은 판단 가능 비율','센서 입력 시연'])
with t_graph:
    st.caption('행=도착(post), 열=출발(pre). 아래 그림은 해부학적 좌표가 아닌 연결 관계 표시입니다.')
    fig,ax=plt.subplots(figsize=(8,6))
    ax.imshow(g['mask']*g['signs'][None,:],cmap='bwr',vmin=-1,vmax=1)
    ax.set(xlabel='Presynaptic neuron index',ylabel='Postsynaptic neuron index',title='Adjacency: assumed GABA (-) / ACh (+)')
    st.pyplot(fig);plt.close(fig)
    index=st.selectbox('뉴런 ID 선택',range(len(g['ids'])),format_func=lambda i:f'{i}: {g["ids"][i]}')
    post,pre=np.where(g['mask']>0)
    edges=pd.DataFrame({'pre_index':pre,'post_index':post,'pre_id':g['ids'][pre].astype(str),'post_id':g['ids'][post].astype(str),'synapse_count':g['counts'][post,pre]})
    related=edges[(edges.pre_index==index)|(edges.post_index==index)]
    graph=nx.DiGraph();graph.add_node(index);graph.add_edges_from(zip(related.pre_index,related.post_index))
    fig,ax=plt.subplots(figsize=(10,6))
    nx.draw_networkx(graph,pos=nx.spring_layout(graph,seed=42),ax=ax,node_size=240,font_size=7,
        node_color=['#617bdb' if g['signs'][i]<0 else '#df7861' for i in graph],arrows=True,arrowsize=10)
    ax.set_title('Selected neuron: all incoming and outgoing edges');ax.axis('off')
    st.pyplot(fig);plt.close(fig)
    st.dataframe(related,hide_index=True)
    st.dataframe(pd.DataFrame({'index':range(len(g['ids'])),'neuron_id':g['ids'].astype(str),'NT_prediction':np.where(g['signs']<0,'GABA','ACh'),'assumed_sign':g['signs']}),hide_index=True)
    st.download_button('전체 연결 CSV',edges.to_csv(index=False).encode('utf-8-sig'),'banc_edges.csv','text/csv')
with t_results:
    st.caption('banc_unsigned = 모든 재귀 연결 양수 제약. banc_free = 부호 자유. 서로 다른 실험 묶음의 결과를 합쳐 평균내지 않습니다.')
    metric_path=root/'metrics_5seeds.csv'
    if not metric_path.exists():metric_path=checkpoint/'metrics.csv'
    if metric_path.exists():
        df=pd.read_csv(metric_path)
        scenario=st.selectbox('조건',sorted(df.scenario.unique()),key='scenario')
        gate=st.checkbox('고정 안전장치 ON (0.2/0.8 + 입력 오류 보류)',value=True)
        part=df[(df.scenario==scenario)&(df.safety_gate==gate)]
        columns=['false_alarm_per_empty','unsafe_false_empty_per_occupied','coverage']
        stats=part.groupby('model')[columns].agg(['mean','std'])
        fig,axs=plt.subplots(1,3,figsize=(15,5))
        for ax,col in zip(axs,columns):
            ax.bar(stats.index,stats[(col,'mean')],yerr=stats[(col,'std')].fillna(0),capsize=3)
            ax.set_title(col);ax.tick_params(axis='x',rotation=65);ax.set_ylim(0,1)
        fig.tight_layout();st.pyplot(fig);plt.close(fig)
        st.caption('막대=시드 평균, 오차막대=시드 표준편차(신뢰구간 아님). 오류율의 분모는 해당 실제 클래스 전체입니다.')
        st.dataframe(part,hide_index=True)
    else:st.info('측정 결과가 아직 없습니다.')
with t_coverage:
    st.caption('테스트 점수의 |p−0.5| 순위로 각 모델에서 같은 수를 선택합니다. 정답은 선택에 사용하지 않습니다. 사후 비교용이며 운영 임계값이나 보정된 확률이 아닙니다. 동점은 고정 난수로 처리합니다.')
    path=root/'coverage_metrics_5seeds.csv'
    if not path.exists():path=checkpoint/'coverage_metrics.csv'
    if path.exists():
        df=pd.read_csv(path)
        sc=st.selectbox('센서 오류 조건',sorted(df.scenario.unique()),key='coverage_scenario')
        guard=st.checkbox('입력 오류 강제 보류 적용',value=False)
        subset=df[(df.scenario==sc)&(df.safety_gate==guard)]
        target=st.select_slider('목표 coverage',options=sorted(subset.target_coverage.unique()),value=.5)
        at_target=subset[np.isclose(subset.target_coverage,target)]
        if not at_target.feasible.all():st.warning('입력 오류 강제 보류로 이 coverage를 달성할 수 없는 조건이 있습니다. 해당 결과는 계산하지 않습니다.')
        st.dataframe(at_target,hide_index=True)
        good=subset[subset.feasible]
        fig,axs=plt.subplots(1,2,figsize=(12,5))
        for model,part in good.groupby('model'):
            grouped=part.groupby('target_coverage')
            x=grouped.coverage.mean()
            for ax,col in zip(axs,['false_alarm_per_empty','unsafe_false_empty_per_occupied']):
                ax.errorbar(x,grouped[col].mean(),yerr=grouped[col].std().fillna(0),marker='o',label=model,capsize=2)
                ax.set(xlabel='Actual coverage',ylabel=col,ylim=(0,1))
        axs[0].legend(fontsize=7);fig.tight_layout();st.pyplot(fig);plt.close(fig)
        st.caption('전체 coverage 외에 empty_coverage·occupied_coverage도 비교하세요. 서로 다른 모델은 서로 다른 표본을 보류할 수 있습니다.')
        st.download_button('Coverage 결과 CSV',df.to_csv(index=False).encode('utf-8-sig'),'coverage_metrics.csv','text/csv')
        vp=root/'validation_threshold_metrics_5seeds.csv'
        if not vp.exists():vp=checkpoint/'validation_threshold_metrics.csv'
        if vp.exists():
            with st.expander('검증 데이터에서 고정한 임계값의 별도 테스트 결과'):
                st.caption('검증 점수로 정한 임계값을 적용하므로 테스트 coverage는 모델 간 같지 않을 수 있습니다. 입력 오류 강제 보류 OFF 조건입니다.')
                st.dataframe(pd.read_csv(vp),hide_index=True)
    else:st.info('같은 coverage 결과가 없습니다. 수정된 train.py 또는 run_5_seeds.py를 새 결과 폴더로 실행하세요.')
with t_demo:
    if not (checkpoint/'banc_signed.pt').exists():st.info('선택한 실험의 모델 체크포인트가 없습니다.')
    else:
        info_path=checkpoint/'experiment_info.json'
        info=json.loads(info_path.read_text(encoding='utf-8')) if info_path.exists() else {}
        recorded=info.get('versions',{}).get('scikit-learn')
        current=importlib.metadata.version('scikit-learn')
        if recorded and recorded!=current:
            st.warning(f'이 체크포인트의 scikit-learn 버전은 {recorded}, 현재는 {current}입니다. 시연은 동일 환경에서 실행하거나 현재 환경에서 재학습하세요. 저장된 비교표는 계속 볼 수 있습니다.')
        else:
            prep=joblib.load(checkpoint/'sensor_prep.joblib')
            model=SignedConnectome(g['mask'],g['signs'])
            model.load_state_dict(torch.load(checkpoint/'banc_signed.pt',map_location='cpu',weights_only=True));model.eval()
            vals=[st.number_input(label,min_value=0.,max_value=limit,value=default) for label,limit,default in [('온도 (°C)',50.,22.5),('습도 (%)',100.,45.),('조도 (lux)',20000.,150.),('CO₂ (ppm)',20000.,700.)]]
            names=['Temperature','Humidity','Light','CO2']
            broken=st.multiselect('데이터가 없는 센서',names)
            raw=np.array([vals]);raw[0,[names.index(n) for n in broken]]=np.nan
            f,invalid=prep.transform(raw)
            with torch.no_grad():prob=float(torch.sigmoid(model(torch.tensor(f,dtype=torch.float32))).item())
            if invalid[0] or .2<prob<.8:st.warning(f'판단 보류 · 모델 점수 {prob:.2f}')
            else:st.info(f'{"사용 중일 가능성" if prob>=.8 else "비어 있을 가능성"} · 모델 점수 {prob:.2f}')
            st.caption('점수는 보정된 확률이 아닙니다. 시연용이며 실제 장치를 제어하지 않습니다.')
