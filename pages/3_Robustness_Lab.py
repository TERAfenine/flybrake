from pathlib import Path
import sys,uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from analysis.phase8.robustness import run,summarize,analyze,DEFAULT_OUT,MODELS,RATES,ROOT
from src.lab_ux import setup,navigation,glossary
st.set_page_config(page_title='강건성 실험실',page_icon='🧪',layout='wide')
setup();navigation('pages/3_Robustness_Lab.py')
st.title('강건성 실험실 (Robustness Lab)')
st.write('센서가 고장 나는 표본의 비율이 늘어나면 AI의 오류와 자동 판단 비율은 어떻게 바뀔까요?')
glossary()
models=st.multiselect('비교 모델',MODELS,default=MODELS,key='rob_models')
fault=st.selectbox('센서 오류 유형',['missing_light','co2_spike'],format_func=lambda v:{'missing_light':'조도 결손','co2_spike':'CO₂ +3000 ppm'}[v],key='rob_fault')
rates=st.multiselect('고장 발생률',RATES,default=RATES,format_func=lambda v:f'{v:.0%}',key='rob_rates')
seeds=st.multiselect('학습 시드',list(range(42,47)),default=list(range(42,47)),key='rob_seeds')
attacks=st.multiselect('오류 주입 시드',[700,701,702],default=[700,701,702],key='rob_attacks')
gate=st.toggle('안전장치 ON',value=True,key='rob_gate')
mode=st.radio('결과 사용 방식',['저장 결과 즉시 조회','선택 조건 새 평가'],key='rob_mode')
@st.cache_data(show_spinner='저장 결과를 읽는 중…')
def read(path,signature):return pd.read_csv(path)
bundles=[DEFAULT_OUT]+sorted((ROOT/'results').glob('phase8_ui_*'))
bundles=[p for p in bundles if (p/'COMPLETE.json').exists()]
if mode=='선택 조건 새 평가':
    st.info('선택 모델·학습·오류 시드와 오류 유형을 다시 추론합니다. 정상 대비 비교를 위해 0%를 포함하고 안전장치 ON/OFF를 함께 저장합니다. 재배선은 3종 모두 사용합니다.')
    if st.button('선택 조건 평가 실행',type='primary',disabled=not(models and rates and seeds and attacks)):
        dest=ROOT/'results'/('phase8_ui_'+pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        bar=st.progress(0.,text='체크포인트 검사 중…')
        try:
            result=run(dest,models,seeds,attacks,[fault],sorted(set(rates+[0.])),progress=lambda p,r:bar.progress(p,text=f'평가 중: {r}'))
            analyze(result,dest/'analysis');st.success(f'평가 완료: {dest.name}');bundles.append(dest)
        except Exception as e:st.error(f'평가를 완료하지 못했습니다: {e}')
if not bundles:st.warning('완료된 평가가 없습니다. 새 평가를 실행해 주세요.');st.stop()
bundle=st.selectbox('저장된 평가 묶음',bundles,index=len(bundles)-1,format_func=lambda p:p.name)
try:df=read(str(bundle/'metrics.csv'),(bundle/'metrics.csv').stat().st_mtime_ns)
except Exception as e:st.error(f'저장 결과 로딩 실패: {e}');st.stop()
if not(models and rates and seeds and attacks):st.info('모델·발생률·학습 시드·오류 시드를 각각 하나 이상 선택해 주세요.');st.stop()
view=df[df.model.isin(models)&df.train_seed.isin(seeds)&df.attack_seed.isin(attacks)&df.fault.eq(fault)&df.safety_gate.eq(gate)]
if view.empty:st.warning('이 묶음에는 선택 조건의 결과가 없습니다.');st.stop()
st.caption(f'출처: {bundle.name} / Phase7 체크포인트 · 동일 테스트 9,752행. 선택에 해당하는 저장된 반복만 표시합니다.')
s=summarize(view[view.fault_rate.isin(rates)])
st.caption('점=학습 시드별 오류·재배선 평균의 평균, 오차막대=그 학습 시드 값의 표본 SD(CI 아님). 시드 1개면 SD를 표시하지 않습니다. 오류 시드별 변동은 아래 원시 표에서 확인하세요.')
for metric,label in [('false_alarm_per_empty','오경보율 ↓'),('unsafe_false_empty_per_occupied','미탐지율 ↓'),('coverage','자동 판단 비율 (Coverage) ↑'),('f1_decided_only','판단 표본 F1 ↑')]:
    fig=go.Figure()
    for model,g in s[s.metric.eq(metric)].groupby('model'):
        g=g.sort_values('fault_rate');fig.add_scatter(x=g.fault_rate*100,y=g['mean']*100,name=model,mode='lines+markers',error_y=dict(type='data',array=g['std']*100))
    fig.update_layout(title=label,xaxis_title='목표 고장 발생률 (%)',yaxis_title='지표 (%)',height=330,margin=dict(l=30,r=15,t=55,b=40),legend=dict(orientation='h'))
    st.plotly_chart(fig,use_container_width=True)
st.caption('오류율은 낮을수록 좋습니다. Coverage/F1은 같은 평가 조건에서 높을수록 유용하지만, Coverage를 높이면 오류도 늘 수 있고 F1은 보류된 표본을 제외합니다.')
with st.expander('모델별 요약 · 시드별 측정값 · 정상 대비 변화'):
    st.dataframe(s,use_container_width=True)
    st.dataframe(view[view.fault_rate.isin(rates)],use_container_width=True)
    keys=['model','train_seed','rewire_seed','attack_seed','fault','safety_gate']
    j=view.merge(view[view.fault_rate.eq(0)],on=keys,suffixes=('','_normal'))
    delta=j[keys+['fault_rate']].copy()
    for m in ['false_alarm_per_empty','unsafe_false_empty_per_occupied','coverage']:delta[m+'_change_pp']=100*(j[m]-j[m+'_normal'])
    st.dataframe(delta,use_container_width=True)
    st.download_button('선택한 개별 결과 CSV',view.to_csv(index=False).encode('utf-8-sig'),'robustness_results.csv','text/csv')
with st.expander('실험 조건 · 데이터 정보 · 해석 한계'):
    st.write('10/20/30%는 각각 975/1950/2925행에 오류를 주입합니다. 조도는 NaN, CO₂는 정규화 전 +3000ppm이며 정답과 기존 전처리·임계값은 그대로입니다. 모델별 오류 마스크는 같습니다.')
    st.write('같은 사무실 테스트를 반복 사용한 탐색적 계산 실험입니다. 실제 교실 고장이나 생물학적 방어 회로의 검증이 아닙니다. BANC 연결은 제공 데이터이며 부호 제약과 동작은 공학적 가정입니다.')
