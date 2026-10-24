"""Streamlit demo: advisory-only, NEVER controls physical devices."""
from pathlib import Path
import joblib
import numpy as np
import torch
import streamlit as st
from src.model import SignedConnectome

st.set_page_config(page_title='FlyBrake',page_icon='🧠',layout='centered')
st.title('FlyBrake | 신경회로 기반 안전 AI')
st.caption('실험용 프로토타입 — 실시간 안전 인증 또는 실제 교실 적용 결과가 아닙니다.')
if not (Path('data/banc_graph.npz').exists() and Path('results/banc_signed.pt').exists()):
    st.error('실제 BANC v888 파일로 build_graph.py와 train.py를 먼저 실행해야 합니다.')
    st.stop()
G=np.load('data/banc_graph.npz',allow_pickle=False)
assert str(G['source'])=='BANC v888 user-supplied inputs (not independently authenticated)'
prep=joblib.load('results/sensor_prep.joblib')
model=SignedConnectome(G['mask'],G['signs'])
model.load_state_dict(torch.load('results/banc_signed.pt',map_location='cpu',weights_only=True))
model.eval()
st.metric('실제 BANC 뉴런 수',len(G['ids']))
st.metric('사용한 실제 연결 수',int(G['mask'].sum()))
st.caption('GABA 음의 가중치 가정 · 아세틸콜린 양의 가중치 가정. 수용체별 신호 효과는 재현하지 않습니다.')
t=st.number_input('온도 (°C)',0.0,50.0,22.5,0.5)
h=st.number_input('상대습도 (%)',0.0,100.0,45.0,1.0)
l=st.number_input('조도 (lux)',0.0,20000.0,150.0,25.0)
c=st.number_input('CO₂ (ppm)',0.0,20000.0,700.0,50.0)
broken=st.multiselect('고장난 센서(데이터 없음)',['Temperature','Humidity','Light','CO2'])
raw=np.array([[t,h,l,c]],dtype=float)
for feature in broken:
    raw[0,['Temperature','Humidity','Light','CO2'].index(feature)] = np.nan
F,invalid=prep.transform(raw)
with torch.no_grad():
    probability=torch.sigmoid(model(torch.tensor(F,dtype=torch.float32))).item()
if invalid[0]:
    st.warning('판단 보류 — 센서 오류 또는 학습 범위 밖 입력. 직접 확인하세요.')
elif probability>=.8:
    st.info(f'사용 중일 가능성 있음 (모델 점수 {probability:.2f})')
elif probability<=.2:
    st.info(f'비어 있을 가능성 있음 (모델 점수 {probability:.2f})')
else:
    st.warning(f'판단 보류 (모델 점수 {probability:.2f}) — 직접 확인하세요.')
st.caption('이 점수는 보정된 확률이 아닙니다. 사람의 확인 없이 출입·환기·조명을 제어하지 마세요.')
if Path('results/metrics.csv').exists():
    import pandas as pd
    st.subheader('성능 검증 결과')
    st.dataframe(pd.read_csv('results/metrics.csv'),hide_index=True)
