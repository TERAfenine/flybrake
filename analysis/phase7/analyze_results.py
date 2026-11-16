"""Read completed Phase 7 outputs; nested rewire/train summaries, no new inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BUNDLE=ROOT/'results/phase7_20261105'
SEEDS=[42,43,44,45,46]
REWIRES=[1042,1043,1044]
MODELS=['banc_signed','mlp_matched','rewired_signed','logistic']
SCENARIOS=['normal','missing_light','co2_spike']
SC={'normal':'정상','missing_light':'조도 결손','co2_spike':'CO₂ 이상치'}
PRIMARY=['false_alarm_per_empty','unsafe_false_empty_per_occupied','f1_decided_only','recall_decided_only','coverage','abstain_rate']
RATES=PRIMARY+['false_alarm_decided_empty','miss_decided_occupied','empty_coverage','occupied_coverage','precision_decided_only']
COUNTS=['fp','fn','n_empty','n_occupied','n_decided','n_decided_empty','n_decided_occupied']
METRICS=RATES+COUNTS
NAMES={'basic':'metrics','posthoc':'coverage_metrics','validation':'validation_threshold_metrics'}


def save(frame,name):
    frame.to_csv(OUT/'tables'/name,index=False,encoding='utf-8-sig',na_rep='NA',float_format='%.16g')


def load(kind):
    df=pd.read_csv(BUNDLE/(NAMES[kind]+'.csv'))
    for col in ['safety_gate','feasible']:
        if col in df:
            converted=df[col].astype(str).str.lower().map({'true':True,'false':False})
            if converted.isna().any():raise ValueError(f'Invalid {col}')
            df[col]=converted.astype(bool)
    if 'feasible' not in df:df['feasible']=True
    context=['scenario','safety_gate']+(['method','target_coverage'] if kind!='basic' else [])
    if df.duplicated(['model','train_seed','rewire_seed']+context).any():raise ValueError('Duplicate run/condition')
    df['source_bundle']='phase7_20261105'
    df['source_csv_row']=np.arange(2,len(df)+2)
    return df,context


def collapse_rewires(df,context,metrics=METRICS):
    """Every metric needs all planned rewires before becoming a train-seed value."""
    rows=[]
    for key,g in df.groupby(['model','train_seed']+context,dropna=False):
        base=dict(zip(['model','train_seed']+context,key))
        rewired=base['model']=='rewired_signed'
        expected=3 if rewired else 1
        identifiers_ok=(set(g.rewire_seed)==set(REWIRES)) if rewired else g.rewire_seed.isna().all()
        complete=len(g)==expected and identifiers_ok
        for m in metrics:
            valid=g.feasible & g[m].notna()
            rows.append(dict(**base,metric=m,value=g.loc[valid,m].mean() if complete and valid.all() else np.nan,
                n_valid_runs=int(valid.sum()),expected_runs=expected,n_runs=len(g),
                status='complete' if complete and valid.all() else 'infeasible' if not g.feasible.any() else 'incomplete_or_undefined'))
    return pd.DataFrame(rows)


def summarize_long(values,keys,value_col='value',expected=SEEDS):
    rows=[]
    for key,g in values.groupby(keys,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        valid=g[value_col].dropna()
        complete=set(g.train_seed)==set(expected) and len(g)==len(expected) and len(valid)==len(expected)
        rows.append(dict(zip(keys,key),n_rows=len(g),n_valid=len(valid),n_train_seeds=g.train_seed.nunique(),
            mean=valid.mean() if complete else np.nan,std=valid.std(ddof=1) if complete else np.nan,
            status='complete' if complete else 'incomplete_or_undefined'))
    return pd.DataFrame(rows)


def per_rewire_summary(df,context):
    d=df[df.model.eq('rewired_signed')].copy()
    rows=[]
    for _,r in d.iterrows():
        for m in METRICS:
            rows.append(dict(**{k:r[k] for k in ['model','rewire_seed','train_seed']+context},
                             metric=m,value=r[m] if r.feasible else np.nan))
    return summarize_long(pd.DataFrame(rows),['model','rewire_seed']+context+['metric'])


def paired(df,context):
    a=df[df.model.eq('banc_signed')]
    rows=[]
    for other in MODELS[1:]:
        b=df[df.model.eq(other)]
        join=a.merge(b,on=['train_seed']+context,suffixes=('_a','_b'),validate='one_to_many')
        for _,r in join.iterrows():
            for m in RATES:
                good=r.feasible_a and r.feasible_b and pd.notna(r[m+'_a']) and pd.notna(r[m+'_b'])
                rows.append(dict(**{k:r[k] for k in ['train_seed']+context},contrast='banc_signed minus '+other,
                    rewire_seed=r.rewire_seed_b,metric=m,value_a=r[m+'_a'],value_b=r[m+'_b'],
                    delta_pp=100*(r[m+'_a']-r[m+'_b']) if good else np.nan,
                    status='complete' if good else 'infeasible_or_undefined'))
    return pd.DataFrame(rows)


def collapse_pairs(p,context):
    rows=[]
    for key,g in p.groupby(['contrast','train_seed']+context+['metric'],dropna=False):
        base=dict(zip(['contrast','train_seed']+context+['metric'],key))
        expected=3 if base['contrast'].endswith('rewired_signed') else 1
        complete=len(g)==expected and g.delta_pp.notna().all()
        rows.append(dict(**base,value=g.delta_pp.mean() if complete else np.nan,
            n_valid_pairs=int(g.delta_pp.notna().sum()),expected_pairs=expected))
    return pd.DataFrame(rows)


def pair_summary(p,context):
    d=collapse_pairs(p,context)
    s=summarize_long(d,['contrast']+context+['metric']).rename(columns={'mean':'mean_delta_pp','std':'std_delta_pp'})
    counts=[]
    for key,g in d.groupby(['contrast']+context+['metric'],dropna=False):
        vals=g.value.dropna()
        counts.append(dict(zip(['contrast']+context+['metric'],key),lower=int((vals < -1e-10).sum()),
                           equal=int((vals.abs()<=1e-10).sum()),higher=int((vals>1e-10).sum())))
    return d,s.merge(pd.DataFrame(counts),on=['contrast']+context+['metric'],validate='one_to_one')


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def render_basic(s,p):
    parts=['# Phase 7 기본 평가\n\n단위 %, 평균 ± 표본 SD(ddof=1). BANC/MLP/Logistic은 학습 시드 5개. Rewired는 각 학습 시드 안에서 3재배선 평균을 구한 후 5학습 시드의 평균±SD. 따라서 Rewired SD는 15개 실행을 독립으로 간주한 SD가 아니다.\n']
    for scenario in SCENARIOS:
        for gate in [False,True]:
            parts.append(f'## {SC[scenario]} · 기본 안전장치 '+('ON' if gate else 'OFF')+'\n')
            rows=[]
            for model in MODELS:
                g=s[s.model.eq(model)&s.scenario.eq(scenario)&s.safety_gate.eq(gate)]
                cells=[]
                for m in PRIMARY:
                    r=g[g.metric.eq(m)].iloc[0]
                    cells.append(f'{100*r["mean"]:.3f} ± {100*r["std"]:.3f}' if r.status=='complete' else f'NA ({r.n_valid}/5)')
                rows.append([model]+cells)
            parts.append(md_table(['모델','오경보율','미탐지율','F1(판단 표본)','Recall(판단 표본)','Coverage','보류율'],rows))
    (OUT/'tables/BASIC_SUMMARY_KO.md').write_text('\n'.join(parts),encoding='utf-8')
    parts=['# Phase 7 동일 학습 시드 paired 차이\n\nBANC signed − 비교 대상. 단위 %p, 평균 ± 표본 SD. 괄호는 낮음/동일/높음 학습 시드 수. Rewired는 같은 학습 시드 내 3재배선 차이를 평균한 후 학습 시드 5개를 요약한다. 오류율 음수는 BANC가 낮은 오류. Coverage에는 단일 우열을 부여하지 않는다.\n']
    for other in MODELS[1:]:
        parts.append('## '+other+'\n');rows=[]
        for scenario in SCENARIOS:
            for gate in [False,True]:
                g=p[p.contrast.eq('banc_signed minus '+other)&p.scenario.eq(scenario)&p.safety_gate.eq(gate)]
                cells=[]
                for m in PRIMARY[:2]+['coverage']:
                    r=g[g.metric.eq(m)].iloc[0]
                    cells.append(f'{r.mean_delta_pp:+.3f} ± {r.std_delta_pp:.3f} ({r.lower}/{r.equal}/{r.higher})' if r.status=='complete' else f'NA ({r.n_valid}/5)')
                rows.append([SC[scenario],'ON' if gate else 'OFF']+cells)
        parts.append(md_table(['센서','기본 보류','Δ오경보율','Δ미탐지율','ΔCoverage'],rows))
    (OUT/'tables/PAIRS_KO.md').write_text('\n'.join(parts),encoding='utf-8')


def figures(summary,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':9})
    colors=['#126b75','#b25c20','#7476a8','#555b66']
    labels={'false_alarm_per_empty':'오경보율','unsafe_false_empty_per_occupied':'미탐지율','coverage':'Coverage'}
    def finish(fig,name,title,foot):
        fig.suptitle(title+'\nphase7_20261105 · train_seed 42–46 · rewire_seed 1042/1043/1044',fontsize=13)
        fig.text(.015,.015,foot,fontsize=9);fig.tight_layout(rect=[0,.065,1,.93])
        for ext in ['png','svg']:fig.savefig(OUT/'figures'/(name+'.'+ext),dpi=180)
        plt.close(fig)
    for gate in [False,True]:
        fig,axes=plt.subplots(3,3,figsize=(13,10))
        for i,m in enumerate(PRIMARY[:2]+['coverage']):
            for j,scenario in enumerate(SCENARIOS):
                ax=axes[i,j];g=summary[summary.scenario.eq(scenario)&summary.safety_gate.eq(gate)&summary.metric.eq(m)].set_index('model').loc[MODELS]
                ax.bar(range(4),g['mean']*100,yerr=g['std']*100,color=colors,capsize=3)
                ax.set_xticks(range(4),MODELS,rotation=25,ha='right',fontsize=8);ax.set_ylabel(labels[m]+' (%)');ax.set_title(SC[scenario]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
                if m=='coverage':ax.set_ylim(0,105)
        finish(fig,'basic_gate_'+str(gate),'기본 평가: 안전장치 '+('ON' if gate else 'OFF'),
            '막대/오차막대: 5학습 시드 평균/SD. Rewired는 시드 내 3재배선 평균 후 요약. SD는 CI/유의성이 아님. 오류 분모는 전체 실제 클래스.')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for i,gate in enumerate([False,True]):
        for j,m in enumerate(PRIMARY[:2]+['coverage']):
            ax=axes[i,j]
            for scenario in SCENARIOS:
                g=pairs[pairs.contrast.eq('banc_signed minus mlp_matched')&pairs.scenario.eq(scenario)&pairs.safety_gate.eq(gate)&pairs.metric.eq(m)].sort_values('train_seed')
                ax.plot(g.train_seed,g.delta_pp,'o-',label=SC[scenario])
            ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(SEEDS);ax.set_xlabel('학습 시드');ax.set_ylabel(labels[m]+' 차이 (%p)');ax.set_title('기본 안전장치 '+('ON' if gate else 'OFF'));ax.grid(alpha=.2)
    axes[0,0].legend()
    finish(fig,'paired_banc_mlp','BANC signed - 용량을 맞춘 MLP: 시드별 차이','오류율에서 음수는 BANC의 낮은 오류, 양수는 높은 오류. Coverage 차이와 함께 해석. 같은 테스트를 쓴 탐색적 비교.')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    q=pairs[pairs.contrast.eq('banc_signed minus rewired_signed')&pairs.metric.eq('false_alarm_per_empty')]
    vmax=max(float(q.delta_pp.abs().max()),1e-3)
    for i,gate in enumerate([False,True]):
        for j,scenario in enumerate(SCENARIOS):
            ax=axes[i,j];g=q[q.scenario.eq(scenario)&q.safety_gate.eq(gate)].pivot(index='rewire_seed',columns='train_seed',values='delta_pp').reindex(index=REWIRES,columns=SEEDS)
            im=ax.imshow(g,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
            ax.set_xticks(range(5),SEEDS);ax.set_yticks(range(3),REWIRES);ax.set_xlabel('학습 시드');ax.set_ylabel('재배선 시드');ax.set_title(SC[scenario]+' · '+('ON' if gate else 'OFF'))
            for r in range(3):
                for c in range(5):ax.text(c,r,f'{g.iloc[r,c]:+.2f}',ha='center',va='center',color='white' if abs(g.iloc[r,c])>vmax*.55 else 'black',fontsize=8)
            fig.colorbar(im,ax=ax,label='오경보율 차이 (%p)',shrink=.8)
    finish(fig,'rewire_train_grid','BANC signed - Rewired signed: 3×5 교차 차이','각 칸은 같은 학습 시드 BANC와 해당 재배선의 차이. 15칸은 독립 데이터 15개가 아님. 음수: BANC 오경보율이 낮음.')


def main():
    if not (BUNDLE/'COMPLETE.json').exists():raise RuntimeError('Wait for all prespecified runs to finish')
    for d in ['tables','figures']:(OUT/d).mkdir(exist_ok=True)
    details={};summaries={};pairsets={};pair_summaries={}
    audit={}
    for kind in NAMES:
        d,context=load(kind);details[kind]=d
        expected={'basic':180,'posthoc':1080,'validation':540}[kind]
        assert len(d)==expected,(kind,len(d),expected)
        v=collapse_rewires(d,context);s=summarize_long(v,['model']+context+['metric']);summaries[kind]=s
        p=paired(d,context);pairsets[kind]=p
        pv,ps=pair_summary(p,context);pair_summaries[kind]=ps
        pr=p[p.rewire_seed.notna()].rename(columns={'delta_pp':'value'})
        prs=summarize_long(pr,['contrast','rewire_seed']+context+['metric']).rename(columns={'mean':'mean_delta_pp','std':'std_delta_pp'})
        save(d,kind+'_individual.csv');save(v,kind+'_within_train.csv');save(s,kind+'_summary.csv')
        save(per_rewire_summary(d,context),kind+'_per_rewire_summary.csv')
        save(p,kind+'_paired_individual.csv');save(pv,kind+'_paired_within_train.csv');save(ps,kind+'_paired_summary.csv');save(prs,kind+'_paired_per_rewire_summary.csv')
        audit[kind]={'rows':len(d),'infeasible_rows':int((~d.feasible).sum()),'summary_incomplete_metrics':int(s.status.ne('complete').sum())}
    render_basic(summaries['basic'],pair_summaries['basic']);figures(summaries['basic'],pairsets['basic'])
    # Historical environment + augmentation changed; never pooled with Phase 7.
    old=pd.read_csv(ROOT/'results/revised_audit_20261024/metrics_5seeds.csv');old=old[old.model.eq('banc_signed')].rename(columns={'seed':'train_seed'})
    new=details['basic'][details['basic'].model.eq('banc_signed')]
    joined=new.merge(old,on=['model','train_seed','scenario','safety_gate'],suffixes=('_phase7','_phase6'),validate='one_to_one')
    historical=[]
    for _,r in joined.iterrows():
        for m in RATES:
            historical.append(dict(model=r.model,train_seed=r.train_seed,scenario=r.scenario,safety_gate=r.safety_gate,
                metric=m,phase6=r[m+'_phase6'],phase7=r[m+'_phase7'],delta_pp=100*(r[m+'_phase7']-r[m+'_phase6']),
                caveat='environment and augmentation changed; descriptive historical comparison, not causal improvement'))
    save(pd.DataFrame(historical),'historical_banc_phase7_minus_phase6.csv')
    meta=pd.read_csv(BUNDLE/'training_metadata.csv');rows=[]
    for model,g in meta.groupby('model'):
        for metric in ['training_seconds','epochs_run','best_epoch','allocated','mask_active','output_effective','dense_matmul_macs','masked_matmul_macs']:
            v=g[metric].dropna()
            rows.append(dict(model=model,metric=metric,n=len(v),mean=v.mean(),std=v.std(ddof=1),minimum=v.min(),maximum=v.max(),sum=v.sum() if len(v) else np.nan))
    save(pd.DataFrame(rows),'timing_capacity_summary.csv')
    inf=details['basic'][~details['basic'].safety_gate][['model','train_seed','rewire_seed','scenario','inference_batch_seconds']]
    save(inf,'inference_batch_timings.csv')
    (OUT/'analysis_checks.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
