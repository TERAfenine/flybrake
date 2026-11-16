"""Render Korean report tables from Phase 6 CSVs; no inference or new metrics."""
from pathlib import Path
import pandas as pd
from analyze import OUT, ROOT, MODELS, SCENARIOS, PRIMARY, TARGETS

LABELS = ['오경보율', '미탐지율', 'F1(판단 표본)', 'Recall(판단 표본)', 'Coverage', '보류율']
SC = {'normal':'정상', 'missing_light':'조도 결손', 'co2_spike':'CO₂ 이상치'}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def cell(g, metric, scale=100):
    r = g[g.metric.eq(metric)].iloc[0]
    if r.status != 'complete':
        return f'NA (유효 {int(r.n_valid)}/5)'
    return f'{scale*r["mean"]:.3f} ± {scale*r["std"]:.3f}'


def summary_sections(kind):
    df = pd.read_csv(OUT/'tables'/f'{kind}_summary.csv')
    sections=[]
    for s in SCENARIOS:
        for gate in ([False] if kind=='validation' else [False,True]):
            for target in ([None] if kind=='basic' else TARGETS):
                g=df[df.scenario.eq(s)&df.safety_gate.eq(gate)]
                if target is not None:g=g[g.target_coverage.eq(target)]
                heading=f'### {SC[s]} · '+('기본 안전장치' if kind=='basic' else '입력 오류 강제 보류')+(' ON' if gate else ' OFF')
                if target is not None:heading+=f' · 목표 {target:.0%}'
                rows=[]
                for m in MODELS:
                    h=g[g.model.eq(m)]
                    label=m
                    if h.status.eq('infeasible').all(): label+=' (feasible=False)'
                    rows.append([label]+[cell(h,k) for k in PRIMARY])
                sections.append(heading+'\n\n'+table(['모델']+LABELS,rows))
    return '\n'.join(sections)


def pair_cell(g, metric):
    r=g[g.metric.eq(metric)].iloc[0]
    if pd.isna(r.mean_delta_pp):return f'NA (쌍 {r.n_pairs}/5)'
    return f'{r.mean_delta_pp:+.3f} ± {r.std_delta_pp:.3f} ({r.lower}/{r.equal}/{r.higher})'


def basic_pairs():
    p=pd.read_csv(OUT/'tables/basic_paired_summary.csv');rows=[]
    for other in ['rewired_signed','logistic','banc_unsigned','banc_free']:
        for s in SCENARIOS:
            for gate in [False,True]:
                g=p[p.contrast.eq('banc_signed minus '+other)&p.scenario.eq(s)&p.safety_gate.eq(gate)]
                rows.append([other,SC[s],'ON' if gate else 'OFF']+[pair_cell(g,m) for m in PRIMARY[:2]+['coverage']])
    return table(['비교 대상', '센서', '기본 보류', 'Δ오경보율', 'Δ미탐지율', 'ΔCoverage'],rows)


def posthoc_pairs():
    p=pd.read_csv(OUT/'tables/posthoc_paired_summary.csv');rows=[]
    for s in SCENARIOS:
        for gate in [False,True]:
            for t in [.5,.75,.9]:
                g=p[p.contrast.eq('banc_signed minus rewired_signed')&p.scenario.eq(s)&p.safety_gate.eq(gate)&p.target_coverage.eq(t)]
                rows.append([SC[s], 'ON' if gate else 'OFF',f'{t:.0%}']+[pair_cell(g,m) for m in PRIMARY[:2]])
    return table(['센서','입력 오류 강제 보류','목표 Coverage','Δ오경보율','Δ미탐지율'],rows)


def selected_summary(kind, scenario=None, targets=(.5,)):
    df=pd.read_csv(OUT/'tables'/f'{kind}_summary.csv');rows=[]
    for s in ([scenario] if scenario else SCENARIOS):
        for t in targets:
            for model in MODELS[:3]:
                g=df[df.scenario.eq(s)&(~df.safety_gate)&df.target_coverage.eq(t)&df.model.eq(model)]
                metrics=['coverage']+PRIMARY[:2]+['empty_coverage','occupied_coverage','false_alarm_decided_empty','miss_decided_occupied']
                rows.append([SC[s],model,f'{t:.0%}']+[cell(g,m) for m in metrics])
    return table(['센서','모델','목표','실제 Coverage','오경보율','미탐지율','빈 공간 Coverage','재실 Coverage','조건부 오경보율','조건부 미탐지율'],rows)


def count_example():
    df=pd.read_csv(OUT/'tables/basic_individual.csv');rows=[]
    for s in SCENARIOS:
        for gate in [False,True]:
            r=df[df.model.eq('banc_signed')&df.seed.eq(42)&df.scenario.eq(s)&df.safety_gate.eq(gate)].iloc[0]
            rows.append([SC[s],'ON' if gate else 'OFF',f'{int(r.fp)}/7703',f'{int(r.fn)}/2049',
                         f'{int(r.fp)}/{int(r.n_decided_empty)}',f'{int(r.fn)}/{int(r.n_decided_occupied)}',
                         f'{100*r.empty_coverage:.3f}',f'{100*r.occupied_coverage:.3f}',f'{100*r.abstain_rate:.3f}'])
    return table(['센서','기본 보류','FP/전체 빈 공간','FN/전체 재실','FP/판단 빈 공간','FN/판단 재실','빈 공간 Coverage %','재실 Coverage %','보류 %'],rows)


def main():
    appendix='# Phase 6 전체 기본 평가 표\n\n데이터: revised_audit_20261024. 시드 42–46. 모든 비율은 %, 평균 ± 표본 SD(ddof=1). SD는 신뢰구간/유의성이 아니다. F1/Recall은 판단 표본에서 계산한다.\n\n'
    basic=summary_sections('basic');appendix+=basic+'\n## 개별 5시드 결과 (210행)\n\n'
    b=pd.read_csv(OUT/'tables/basic_individual.csv')
    for s in SCENARIOS:
        for gate in [False,True]:
            g=b[b.scenario.eq(s)&b.safety_gate.eq(gate)].sort_values(['model','seed'])
            appendix+=f'### {SC[s]} · 기본 안전장치 '+('ON' if gate else 'OFF')+'\n\n'
            appendix+=table(['모델','시드']+LABELS,[[r.model,r.seed]+[f'{100*r[k]:.5f}' if pd.notna(r[k]) else 'NA' for k in PRIMARY] for _,r in g.iterrows()])+'\n'
    appendix+='## 클래스별 판단 비율과 조건부 오류율\n\n전체 클래스 분모 오류율과 함께 읽는다. 모든 수치는 %의 5시드 평균 ± SD. 실제 건수/분모는 basic_individual.csv에 있다.\n\n'
    s=pd.read_csv(OUT/'tables/basic_summary.csv')
    for scenario in SCENARIOS:
        for gate in [False,True]:
            g=s[s.scenario.eq(scenario)&s.safety_gate.eq(gate)]
            appendix+=f'### {SC[scenario]} · 기본 안전장치 '+('ON' if gate else 'OFF')+'\n\n'
            appendix+=table(['모델','빈 공간 Coverage','재실 Coverage','조건부 오경보율','조건부 미탐지율'],
                [[m]+[cell(g[g.model.eq(m)],k) for k in ['empty_coverage','occupied_coverage','false_alarm_decided_empty','miss_decided_occupied']] for m in MODELS])+'\n'
    (OUT/'tables/BASIC_ALL_KO.md').write_text(appendix,encoding='utf-8')
    for kind,title in [('posthoc','사후 동일 Coverage'),('validation','검증 임계값 고정 적용')]:
        content=f'# Phase 6 {title} 전체 요약\n\nrevised_audit_20261024 · 시드 42–46. 비율 %, 평균 ± 표본 SD. 유효값 5개가 아니면 평균/SD는 NA. NA는 0이 아니다.\n\n'
        content+=('사후 순위 분석이며 운영 성능 보장이 아니다. feasible=False는 평가할 수 없는 목표다.\n\n' if kind=='posthoc' else '검증 목표와 테스트 실제 Coverage는 다르다. 판단한 양성 표본이 없을 때 저장된 Recall=0은 코드의 zero_division 규칙이며 재실 평가 근거가 아니다.\n\n')
        (OUT/'tables'/(kind.upper()+'_ALL_KO.md')).write_text(content+summary_sections(kind),encoding='utf-8')
    replacements={'basic_tables':basic,'basic_pairs':basic_pairs(),'posthoc_pairs':posthoc_pairs(),
                  'posthoc_co2':selected_summary('posthoc','co2_spike',(.5,.75)),
                  'validation_50':selected_summary('validation'),'counts':count_example()}
    text=(OUT/'report_template.md').read_text(encoding='utf-8')
    for k,v in replacements.items():
        marker='{{'+k+'}}'
        assert text.count(marker)==1,marker
        text=text.replace(marker,v)
    assert '{{' not in text
    (ROOT/'LAB_PHASE6_PERFORMANCE_REPORT_KO.md').write_text(text,encoding='utf-8')
    print('Rendered root report and 3 complete Markdown table appendices.')


if __name__=='__main__':main()
