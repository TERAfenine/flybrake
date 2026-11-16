"""Presentation only: shared navigation, wording and chart styling."""
from html import escape
import streamlit as st
import plotly.graph_objects as go
from streamlit.errors import StreamlitPageNotFoundError

MODEL_NAMES = {
    'logistic': '일반 AI · 로지스틱 회귀',
    'banc_signed': '실제 초파리 연결망 + 전달물질 부호 제약',
    'rewired_signed': '무작위 재배선 + 전달물질 부호 제약',
    'banc_unsigned': '실제 초파리 연결망 + 모든 연결 양수',
    'rewired_unsigned': '무작위 재배선 + 모든 연결 양수',
    'banc_free': '실제 초파리 연결망 + 부호 자유',
    'rewired_free': '무작위 재배선 + 부호 자유',
}
HELP = {
    'false_alarm_per_empty': '사람이 없는 공간을 사람이 있다고 잘못 판단한 비율. 분모는 실제 비재실 표본 전체입니다.',
    'unsafe_false_empty_per_occupied': '사람이 있는 공간을 비어 있다고 잘못 판단한 비율. 분모는 실제 재실 표본 전체입니다.',
    'coverage': 'AI가 판단을 보류하지 않고 자동 판단한 비율입니다.',
    'abstain_rate': '전체 입력 중 자동 판단을 보류한 비율입니다.',
    'f1_decided_only': '자동 판단한 표본에서만 계산한 정밀도와 재현율의 조화평균입니다.',
    'recall_decided_only': '자동 판단한 실제 재실 표본 중 재실로 맞힌 비율입니다.',
}

def model_name(name):
    return MODEL_NAMES.get(name, name)

def setup():
    # Keep control values across pages; never reassign chart events or buttons.
    research_keys = {'lab_bundle','lab_gate','lab_target','lab_val_target','research_tradeoff_model',
                     'neuron_bundle','neuron_index','neuron_view','neuron_disabled',
                     'neuron_nt_filter','neuron_scenario','neuron_guard'}
    research_prefixes = ('lab_seed_','lab_a_','lab_b_','lab_scenario_',
                         'neuron_seed_','sensor_')
    for key in list(st.session_state):
        guide_control = key.startswith('ux_') and not any(part in key for part in ('network_', '_plot', 'compare_errors', 'compare_coverage', '_zoom_in', '_zoom_out', '_fit', '_focus'))
        if guide_control or key in research_keys or key.startswith(research_prefixes) or key.startswith('rob_'):
            st.session_state[key] = st.session_state[key]
    st.markdown('''<style>
    .block-container {max-width:1240px;padding-top:2.4rem;padding-bottom:3rem;}
    h1 {letter-spacing:-.04em;line-height:1.25!important;word-break:keep-all;}
    h2,h3 {letter-spacing:-.025em;word-break:keep-all;}
    [data-testid="stMetric"] {border:1px solid #81969b55;border-radius:12px;padding:14px;}
    [data-testid="stMetricValue"] {font-size:1.65rem;}
    [data-testid="stCaptionContainer"] {line-height:1.65;}
    .ux-hero {background:#102d3b;color:#f2f8fa;border-radius:20px;padding:30px;margin:12px 0 24px;}
    .ux-hero h1 {color:#fff;font-size:2.5rem;max-width:850px;margin:12px 0;}
    .ux-hero p {color:#c7dce2;max-width:850px;line-height:1.8;}
    .ux-eyebrow {color:#77ded0;font-weight:700;letter-spacing:.14em;font-size:.8rem;}
    .ux-selection {border-left:4px solid #0f766e;background:#e8f5f2;color:#123c39;padding:14px 18px;border-radius:8px;overflow-wrap:anywhere;}
    .st-key-mobile_menu {display:none;}
    .ux-hold {border-left:4px solid #996100;padding:10px 14px;background:#fff5d9;color:#634400;border-radius:6px;}
    .ux-legend {font-size:.9rem;line-height:1.8;margin:8px 0;}
    .ux-legend strong {white-space:nowrap;}
    @media(max-width:640px){.block-container{padding:4.5rem 1rem 2rem;}
      .st-key-mobile_menu{display:block;}
      .st-key-ux_graph_tools [data-testid="stHorizontalBlock"],.st-key-neuron_graph_tools [data-testid="stHorizontalBlock"]{flex-wrap:nowrap;gap:.4rem;}
      .st-key-ux_graph_tools [data-testid="stColumn"],.st-key-neuron_graph_tools [data-testid="stColumn"]{min-width:0!important;flex:1 1 0!important;}
      .ux-hero{padding:22px 18px;}.ux-hero h1{font-size:1.8rem;}
      h1{font-size:2rem!important;}[data-testid="stMetricValue"]{font-size:1.5rem;}}
    </style>''', unsafe_allow_html=True)

def navigation(current):
    pages=[('app.py','홈'),('pages/0_Start_Experiment.py','시작하기 · AI 실험실'),
           ('pages/2_FlyBrake_Neurons.py','뉴런 탐색'),('pages/1_FlyBrake_Lab.py','연구 결과'),
           ('pages/3_Robustness_Lab.py','강건성 실험실 (Robustness Lab)')]
    with st.sidebar:
        st.markdown('### FlyBrake Lab')
        for page,label in pages:
            page_link(page,label=label,disabled=current==page,use_container_width=True)
        st.caption('메뉴로 이동해도 선택 조건은 유지됩니다. 새로고침·새 세션에서는 초기화될 수 있습니다.')
        st.divider()
    with st.container(key='mobile_menu'):
        with st.popover('메뉴 · 페이지 이동',use_container_width=True):
            for page,label in pages:
                page_link(page,label=label,disabled=current==page,use_container_width=True)

def page_link(page, **kwargs):
    try:
        st.page_link(page, **kwargs)
    except StreamlitPageNotFoundError:
        # Standalone page entrypoints (including AppTest) have no app page registry.
        # Keep the research page usable; navigation is available via app.py.
        st.caption(kwargs.get('label',page)+' · app.py에서 실행 시 이동 가능')

def glossary():
    with st.expander('용어가 낯선가요? 지표와 모델 이름 알아보기'):
        for key,label in [('false_alarm_per_empty','오경보율'),('unsafe_false_empty_per_occupied','미탐지율'),
                          ('coverage','자동 판단 비율 (Coverage)'),('abstain_rate','판단 보류율')]:
            st.markdown(f'**{label}** — {HELP[key]}')
        st.caption('오경보·미탐지의 분모에는 보류 표본도 포함되지만, 보류를 오류로 세지 않습니다. 오류율과 보류율을 함께 읽으세요.')
        for key,label in MODEL_NAMES.items():
            st.markdown(f'`{key}` · {label}')
        st.caption('전달물질은 예측값입니다. GABA 음수 / ACh 양수 제약은 공학적 가정이며, unsigned는 부호 자유가 아니라 모든 연결 양수 제약입니다.')

def selection_banner(index, neuron_id, disabled=False, graph=None):
    state='활성값 0 고정' if disabled else '원래 활성값 사용'
    details=''
    if graph is not None:
        prediction='GABA' if graph['signs'][index]<0 else 'ACh'
        details=f'<br>{prediction} 예측 · 들어옴 {int(graph["mask"][index].sum())}개 / 나감 {int(graph["mask"][:,index].sum())}개'
    st.markdown(f'<div class="ux-selection"><b>{"× 개입 설정 ON" if disabled else "● 선택됨"} · 뉴런 {int(index)}</b><br>BANC {escape(str(neuron_id))}{details}<br>{state}</div>',unsafe_allow_html=True)

def hold_explanation():
    st.markdown('<div class="ux-hold"><b>판단 보류 = 자동 결정을 내리지 않음</b><br>사람이 있다/없다의 세 번째 정답이 아닙니다. 입력 오류 또는 불확실한 점수를 만났을 때 자동 판단을 멈춥니다.</div>',unsafe_allow_html=True)

def score_figure(labels, scores, title):
    fig=go.Figure(go.Bar(x=scores,y=labels,orientation='h',marker_color=['#176b86','#c46a23'],
        text=[f'{p:.8f}' for p in scores],textposition='auto',
        hovertemplate='%{y}<br>모델 출력 점수 %{x:.8f}<extra></extra>',name='모델 출력 점수'))
    fig.update_layout(title=title,height=260,margin=dict(l=10,r=20,t=55,b=40),
        xaxis=dict(title='모델 출력 점수 (0–1)<br>확률 보정 미검증',range=[0,1]),
        yaxis=dict(autorange='reversed'),template='plotly_white',showlegend=False)
    return fig

def landing():
    setup()
    navigation('app.py')
    st.markdown('''<section class="ux-hero"><span class="ux-eyebrow">FLYBRAKE LAB / INTERACTIVE RESEARCH</span>
    <h1>실제 초파리 신경 연결 구조를 활용한 AI는 센서 오류 상황에서 어떻게 판단할까?</h1>
    <p>센서를 바꾸고, 뉴런을 선택하고, AI의 판단을 비교해 보세요.<br>
    제공된 BANC v888의 VNC 부분 연결망 · 96개 뉴런 · 1,733개 방향성 연결 · GABA 예측 50개</p></section>''',unsafe_allow_html=True)
    left,right=st.columns(2)
    with left,st.container(border=True):
        st.subheader('처음 사용하기')
        st.write('준비된 기본값으로 4가지 실험을 따라가세요. 먼저 센서에 오류를 넣고 판단이 어떻게 달라지는지 확인합니다.')
        st.page_link('pages/0_Start_Experiment.py',label='실험 시작하기 →',icon='🧭',use_container_width=True)
    with right,st.container(border=True):
        st.subheader('연구자 모드')
        st.write('결과 묶음, 모델, 시드와 센서 조건을 직접 선택하세요. 동일 Coverage 분석과 원본 측정값까지 확인할 수 있습니다.')
        st.page_link('pages/1_FlyBrake_Lab.py',label='상세 모델 비교 열기 →',icon='📊',use_container_width=True)
        st.page_link('pages/2_FlyBrake_Neurons.py',label='뉴런 탐색·비활성화 열기 →',use_container_width=True)
    st.caption('관측 자료: 제공된 연결·뉴런 ID 및 전달물질 예측. 외부 원본 인증은 별도 필요합니다. '
               '모델링 가정: GABA 음수 / ACh 양수. 사무실 재실 데이터와 합성 센서 오류를 사용하는 계산 실험이며 생물학적 기능 복제나 실제 안전 인증이 아닙니다.')

