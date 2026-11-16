from analysis.phase8.robustness import *

def main():
    out=ROOT/'analysis/phase8';df=pd.read_csv(DEFAULT_OUT/'metrics.csv');s=analyze(df,out/'tables')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
    (out/'figures').mkdir(exist_ok=True)
    names={'false_alarm_per_empty':'오경보율','unsafe_false_empty_per_occupied':'미탐지율','coverage':'Coverage','f1_decided_only':'판단 표본 F1'}
    for metric,label in names.items():
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for i,fault in enumerate(FAULTS):
            for j,gate in enumerate([False,True]):
                ax=axes[i,j]
                for model in MODELS:
                    g=s[s.model.eq(model)&s.fault.eq(fault)&s.safety_gate.eq(gate)&s.metric.eq(metric)].sort_values('fault_rate')
                    ax.errorbar(g.fault_rate*100,g['mean']*100,yerr=g['std']*100,marker='o',capsize=3,label=model)
                ax.set(title=f'{fault} · gate {"ON" if gate else "OFF"}',xlabel='목표 오류 발생률 (%)',ylabel=label+' (%)',xticks=[0,10,20,30]);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'{label} · phase8_20261116 · 5학습 × 3오류 시드 (재배선3종)',fontsize=12)
        fig.text(.02,.015,'오차막대: 오류/재배선 평균 후 5학습 시드의 표본 SD, 신뢰구간 아님. 패널별 축 범위가 다름.',fontsize=9)
        fig.tight_layout(rect=[0,.04,1,.95])
        for ext in ['png','svg']:fig.savefig(out/'figures'/f'{metric}.{ext}',dpi=160)
        plt.close(fig)
    lines=['# FlyBrake Robustness Lab — Phase 8 결과 보고서','작성일: 2026-11-16. 신규 학습 없이 Phase7 체크포인트로 수행한 탐색적 평가.',
        '## 구현과 연구 연속성',
        '기존 src/model.py, src/data.py, src/evaluation.py, 학습 코드, BANC 원본과 Phase1–7 결과는 보존했다. 새 메뉴를 위해 src/lab_ux.py의 메뉴와 설정 보존 조건만 확장했다. Phase7 실험30개 및52테스트 기록은 있으나 최종 Phase7 보고서는 이번 작업 시작 시 없었다. 기존 계획·실측 CSV를 근거로 체크포인트를 선택했다.',
        '## 실험 설계',
        '체크포인트: results/phase7_20261105. Logistic/BANC signed 각 학습42–46, Rewired signed는 학습42–46 × 재배선1042/1043/1044. 총25개 체크포인트. 오류 주입 시드700/701/702는 학습과 독립이다. 같은 오류시드의 무작위 순열 앞 floor(N×rate)행을 선택하므로 모든 모델의 오류 마스크는 같다. 비율끼리는 포함관계이고 독립 데이터가 아니다.',
        '테스트는 datatest2.txt의9,752행(비재실7,703,재실2,049). 발생률0/10/20/30%는 실제0/975/1950/2925행(0/9.99795/19.99590/29.99385%)이다. 조도NaN 또는CO2+3000ppm을 원자료에 적용하고 저장된 전처리를 사용한다. 강도와 발생률을 구분한다. 정답/분할/가중치/임계값을 변경하지 않았다.',
        '기본OFF는 .5, ON은 invalid 또는 .2와.8사이 점수 보류. 전체 분모 오경보=FP/7703,미탐지=FN/2049. 조건부 오류는 판단한 해당 클래스 분모다. F1/Recall은 판단 표본만 사용한다. Coverage+보류율=1. 보류는 정답이 아니다.',
        '평균은 학습 시드 안에서 오류3개 및 재배선3개의 결과를 먼저 평균하고 그5값으로 평균±표본SD를 계산한다. SD는CI나유의성이 아니다. 오류시드별 별도표 및 전체1200행을 보존했다. 0% 반복은 같은정상결과의복사이므로추가독립증거가 아니다.',
        '## 실제 측정 결과','모든 표의 단위는 %. Rewired는 위의 계층 평균.']
    for fault in FAULTS:
        for gate in [False,True]:
            lines += [f'### {fault} · gate {gate}','| 모델 | 발생률 | 오경보 | 미탐지 | Coverage | F1 | Recall | 보류 |','|---|---|---|---|---|---|---|---|']
            for model in MODELS:
                for rate in RATES:
                    g=s[s.model.eq(model)&s.fault.eq(fault)&s.fault_rate.eq(rate)&s.safety_gate.eq(gate)].set_index('metric')
                    cells=[f'{100*g.loc[m,"mean"]:.3f} ± {100*g.loc[m,"std"]:.3f}' for m in ['false_alarm_per_empty','unsafe_false_empty_per_occupied','coverage','f1_decided_only','recall_decided_only','abstain_rate']]
                    lines.append('| '+' | '.join([model,f'{rate:.0%}']+cells)+' |')
    for metric,label in names.items():lines+= [f'![{label}](analysis/phase8/figures/{metric}.png)']
    lines+=['## 시드별 결과와 해석',
        '개별값: results/phase8_20261116/metrics.csv. 정상 대비 변화: analysis/phase8/tables/normal_deltas_pp.csv. BANC−Logistic/Rewired의 같은학습·오류시드 paired차이: paired_differences_pp.csv. 오류시드별 요약: attack_700/701/702_summary.csv. 각실행 NPZ에는 점수, 정답, 표본인덱스, 오류마스크,ON/OFF결정이 있다. 결정-1은보류다.',
        '## 관측 결과의 해석',
        '30% 조도 결손·OFF에서 BANC의 평균 오경보/미탐지는22.201%/9.497%, Logistic은19.395%/10.558%, Rewired는22.147%/9.489%다. BANC−Logistic 차이는오경보+2.806%p,미탐지−1.061%p로 오류시드평균후5학습시드모두 같은방향이다. 따라서 미탐지이득과오경보불이익이함께있다.',
        '30% CO₂·OFF에서는 BANC40.248%/0.179%, Logistic39.145%/0.358%, Rewired40.225%/0.183%다. BANC−Logistic은오경보+1.103%p,미탐지−0.179%p로5학습시드모두같은방향이다. CO₂에서미탐지가낮다는사실만으로강건하다고볼수없다. 오경보가크게증가한다.',
        '기본정상OFF대비30%에서 BANC오경보증가는조도+7.757%p/CO₂+25.804%p이고 Logistic은+6.608%p/+26.358%p다. CO₂오경보의증가폭만보면BANC가작지만오류발생후절대오경보율은여전히높다. 절대값과저하량을함께봐야한다.',
        '30%·ON에서는두센서조건의결과가같다. BANC오경보2.668%,미탐지0.049%,Coverage61.195%; Logistic1.225%,0.114%,59.058%; Rewired2.304%,0.049%,61.115%다. 같은오류행을입력검사로보류한결과이며센서오류를복원했다는증거가아니다. BANC정상ON Coverage87.199%에서61.195%로약26.003%p감소한다.',
        'BANC−Rewired의30%OFF오경보차이는조도+0.054%p/CO₂+0.023%p로작고학습시드방향이혼재한다(각2낮음/3높음). ON오경보는+0.364%p로5학습시드모두높고미탐지는같다. 따라서실제BANC구조의일관된강건성우월성은확인되지않았다. 재배선3개와학습5개의결과를독립15표본처럼검정하지않았다.',
        'Logistic의학습시드SD가0이어도오류행선택의불확실성이0이라는뜻은아니다. 같은모델과증강으로학습결과가같기때문이며오류시드별표를함께제공한다. 유의성/신뢰구간/일반환경우월성을주장하지않는다.',
        '## UI 및 재실행',
        '강건성 실험실 메뉴에서 모델,발생률,학습시드,오류시드,오류유형,안전장치를 선택한다. 저장조회는캐시를사용한다. 새평가는버튼을눌러기존모델을추론하고진행상태를표시하며 results/phase8_ui_시간_고유값에저장한다. 정상비교용0% 및ON/OFF는함께계산한다. 기존평가폴더덮어쓰기는거부한다.',
        '실행: `.venv/Scripts/python.exe -m analysis.phase8.robustness --out results/새폴더` 후 `.venv/Scripts/python.exe -m analysis.phase8.report`. 분석코드는 기본완료묶음 phase8_20261116을읽는다.',
        '## 검증 결과와 보존',
        '기존52개와 신규4개, 총56개 테스트가 모두 통과했다(48.747초, analysis/phase8/tests.log). 1,200행의 지표를 저장된 개별 판단에서 복원했고, 마스크 건수/동일성/포함관계, Coverage+보류율=1, 정상300조건의 기존값 일치, BANC42의 선택 조건 재추론 일치를 확인했다. 기존 뉴런 페이지와 모델 비교 페이지의 AppTest도 통과했다.',
        '새 화면의 필터 변경, 빈 선택 안내,4개 그래프/표 표시를 AppTest로 확인했다. 별도로 새 평가 버튼을 실제로 눌러 Logistic42·오류시드700·0/10% 조건을 새 폴더에 계산/저장하고 성공 메시지와4개 그래프를 확인했다(ui_run_check.log). 이는 자동 실행 검사이며 실제 사람의 브라우저 사용성 평가가 아니다.',
        '전체 평가에는25체크포인트·2오류유형·4발생률·3오류시드·ON/OFF가 포함되며1200행, 실제 실행시간24.638초였다. 환경은Python3.11.9, numpy1.26.0,pandas2.0.0,torch2.1.0,scikit-learn1.3.0,joblib1.3.0,CPU4threads다. 실행시각/체크포인트·전처리·테스트·코드해시는 새결과 manifest.json에 있다.',
        '보호목록501개 중500개가 SHA256으로 변경 없음을 확인했다. 유일한 허용 변경은 메뉴 추가/선택 유지용 src/lab_ux.py이며 원래 버전은 analysis/phase8/lab_ux_before.py에 보존했다. 기존 데이터/체크포인트/CSV/NPZ/Phase6·7 분석 결과에는 변경이 없다. 보존 기록: analysis/phase8/preservation.json. 작업 브랜치: codex/flybrake-phase8-robustness.',
        '## 한계 및 미검증 범위',
        '같은사무실의 이미열람한테스트를재사용했으므로새독립확증이아니다. 실제교실고장/사고율/생물학적방어회로를검증하지않았다. 전달물질은예측이며부호제약은공학가정이다. 여러재배선/학습/오류시드는같은데이터에대한반복이지독립장소표본이아니다. 높은Coverage는자동판단증가이지단독안전성보장이아니고조건부F1상승도보류표본변화와함께읽어야한다.',
        '브라우저 직접 조작 도구를 사용할 수 없어 실제 데스크톱/모바일 시각 검증은 수행하지 않았다. Streamlit AppTest 결과와 실제 브라우저 사용자 평가는 구분한다. 새UI의임의오류시드입력/동시오류는추가하지않았으며저장된700/701/702시드를선택할수있다.']
    text='\n\n'.join(lines).replace('|\n\n|','|\n|')
    (ROOT/'LAB_ROBUSTNESS_REPORT_KO.md').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
