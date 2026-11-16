"""Read-only Phase 6 analysis of recorded revised_v1 results. No model loading/training.

Run from project root: .venv/Scripts/python.exe analysis/phase6/analyze.py
All generated tables/figures stay beside this script. Rates remain fractions in CSV;
paired differences are percentage points. Complete five-seed means only.
"""
from pathlib import Path
import hashlib
import itertools
import json
import platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BUNDLE = ROOT/'results/revised_audit_20261024'
SEEDS = [42,43,44,45,46]
MODELS = ['banc_signed','rewired_signed','logistic','banc_unsigned','banc_free','rewired_unsigned','rewired_free']
SCENARIOS = ['normal','missing_light','co2_spike']
TARGETS = [.25,.5,.65,.75,.9,1.]
KEYS = ['model','scenario','safety_gate']
PRIMARY = ['false_alarm_per_empty','unsafe_false_empty_per_occupied','f1_decided_only',
           'recall_decided_only','coverage','abstain_rate']
RATES = PRIMARY + ['false_alarm_decided_empty','miss_decided_occupied','empty_coverage',
                   'occupied_coverage','precision_decided_only']
COUNTS = ['fp','fn','n_empty','n_occupied','n_decided','n_decided_empty','n_decided_occupied']
PAIRS = [('banc_signed',b) for b in ['rewired_signed','logistic','banc_unsigned','banc_free']]+[
    ('banc_unsigned','rewired_unsigned'),('banc_free','rewired_free')]
SOURCES = {'basic':'metrics_5seeds.csv','posthoc':'coverage_metrics_5seeds.csv',
           'validation':'validation_threshold_metrics_5seeds.csv'}
METHODS = {'basic':'fixed_0.2_0.8_or_0.5','posthoc':'test_score_top_k_no_labels',
           'validation':'validation_threshold_test_coverage_not_guaranteed'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bool_column(series):
    mapped = series.astype(str).str.lower().map({'true':True,'false':False})
    if mapped.isna().any():
        raise ValueError('Unknown/missing boolean value')
    return mapped.astype(bool)


def load_recorded(kind, bundle=BUNDLE):
    path = bundle/SOURCES[kind]
    df = pd.read_csv(path)
    df['safety_gate'] = bool_column(df.safety_gate)
    if 'feasible' in df:
        df['feasible'] = bool_column(df.feasible)
    else:
        # No feasibility flag in these protocols, distinct from posthoc feasibility.
        df['feasible'] = True
    df['method'] = df['method'] if 'method' in df else METHODS[kind]
    df['analysis_kind'] = kind
    df['source_file'] = path.relative_to(ROOT).as_posix()
    df['source_csv_row'] = np.arange(2,len(df)+2)
    keys = KEYS+(['target_coverage'] if kind!='basic' else [])
    if df[keys+['seed']].isna().any().any() or df.duplicated(keys+['seed']).any():
        raise ValueError(f'{kind}: missing/duplicate keys')
    if set(df.model)!=set(MODELS) or set(df.seed)!=set(SEEDS) or set(df.scenario)!=set(SCENARIOS):
        raise ValueError(f'{kind}: unexpected model/seed/scenario set')
    if set(df.method)!={METHODS[kind]}:
        raise ValueError(f'{kind}: mixed methods')
    if kind=='validation' and df.safety_gate.any():
        raise ValueError('Validation protocol must have safety_gate=False')
    for metric in RATES:
        finite = df[metric].dropna()
        if not np.isfinite(finite).all() or not finite.between(0,1).all():
            raise ValueError(f'Invalid rate: {metric}')
    for klass in ['empty','occupied']:
        values = df[f'n_{klass}']*df[f'{klass}_coverage']
        if not np.allclose(values.dropna(),np.rint(values.dropna()),atol=1e-7,rtol=0):
            raise ValueError('Class decided count cannot be recovered as integer')
        df['n_decided_'+klass] = np.rint(values)
    df['row_status'] = np.select([~df.feasible,df.n_decided.eq(0),df.n_decided_empty.eq(0)|df.n_decided_occupied.eq(0)],
                                 ['infeasible','no_decisions','one_decided_class_empty'],default='measured')
    return df


def summary(df, keys, metrics=RATES+COUNTS, expected=SEEDS):
    """No silent skip-NA five-seed mean. n_valid and status explain N/A."""
    if df.duplicated(keys+['seed']).any():
        raise ValueError('Duplicate seed in summary group')
    rows=[]
    for group_key,g in df.groupby(keys,dropna=False,sort=True):
        values = group_key if isinstance(group_key,tuple) else (group_key,)
        base=dict(zip(keys,values))
        feasible=g.feasible if 'feasible' in g else pd.Series(True,index=g.index)
        for metric in metrics:
            valid=g[metric].notna()&feasible
            status=('missing_seed' if set(g.seed)!=set(expected) else
                    'infeasible' if not feasible.any() else
                    'partly_infeasible' if not feasible.all() else
                    'undefined_or_missing_metric' if not valid.all() else 'complete')
            x=g.loc[valid,metric]
            rows.append(dict(**base,metric=metric,n_rows=len(g),n_seeds=g.seed.nunique(),n_valid=len(x),
                n_feasible=int(feasible.sum()),status=status,
                mean=x.mean() if status=='complete' else np.nan,
                std=x.std(ddof=1) if status=='complete' else np.nan))
    return pd.DataFrame(rows)


def paired(df, context, pairs=PAIRS, metrics=PRIMARY+RATES[6:10]):
    """Exact seed join. Missing/infeasible/undefined sides never become zero."""
    rows=[]
    if df.duplicated(['model','seed']+context).any():
        raise ValueError('Duplicate paired keys')
    for a,b in pairs:
        left=df[df.model.eq(a)];right=df[df.model.eq(b)]
        merged=left.merge(right,on=['seed']+context,how='outer',suffixes=('_a','_b'),indicator=True,validate='one_to_one')
        for _,r in merged.iterrows():
            present=r['_merge']=='both'
            feasible=present and bool(r.feasible_a) and bool(r.feasible_b)
            for m in metrics:
                av,bv=r[m+'_a'],r[m+'_b']
                valid=feasible and pd.notna(av) and pd.notna(bv)
                rows.append(dict(**{k:r[k] for k in ['seed']+context},contrast=a+' minus '+b,metric=m,
                    value_a=av,value_b=bv,delta_pp=100*(av-bv) if valid else np.nan,
                    feasible=feasible,status='measured' if valid else 'missing_pair' if not present else 'infeasible_or_undefined'))
    return pd.DataFrame(rows)


def paired_summary(df,context):
    keys=['contrast']+context+['metric']
    records=[]
    for key,g in df.groupby(keys,dropna=False,sort=True):
        valid=g.delta_pp.dropna();complete=len(valid)==len(SEEDS) and set(g.seed)==set(SEEDS)
        records.append(dict(zip(keys,key),n_pairs=len(valid),n_rows=len(g),
            mean_delta_pp=valid.mean() if complete else np.nan,std_delta_pp=valid.std(ddof=1) if complete else np.nan,
            lower=int((valid < -1e-10).sum()),equal=int((valid.abs()<=1e-10).sum()),higher=int((valid>1e-10).sum()),
            status='complete' if complete else 'incomplete_or_undefined'))
    return pd.DataFrame(records)


def verify_arithmetic(df):
    """Independent reconstruction from saved confusion counts, not model inference."""
    checks=0
    for _,r in df.iterrows():
        if not r.feasible:
            if r[RATES+COUNTS].notna().any():raise AssertionError('Infeasible row has measured metrics')
            continue
        ne,np_,de,dp,fp,fn,nd = [r[k] for k in ['n_empty','n_occupied','n_decided_empty','n_decided_occupied','fp','fn','n_decided']]
        if any(pd.isna(x) for x in [ne,np_,de,dp,fp,fn,nd]):continue
        if not (0<=fp<=de<=ne and 0<=fn<=dp<=np_ and de+dp==nd):raise AssertionError('Invalid counts')
        tp=dp-fn
        ratio=lambda a,b: a/b if b else np.nan
        expected={'coverage':ratio(nd,ne+np_),'abstain_rate':1-ratio(nd,ne+np_),
            'false_alarm_per_empty':ratio(fp,ne),'unsafe_false_empty_per_occupied':ratio(fn,np_),
            'false_alarm_decided_empty':ratio(fp,de),'miss_decided_occupied':ratio(fn,dp),
            'empty_coverage':ratio(de,ne),'occupied_coverage':ratio(dp,np_),
            'recall_decided_only':(tp/dp if dp else 0.) if nd else np.nan,
            'precision_decided_only':(tp/(tp+fp) if tp+fp else 0.) if nd else np.nan,
            'f1_decided_only':(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.) if nd else np.nan}
        for m,value in expected.items():
            if not np.isclose(r[m],value,atol=1e-12,rtol=1e-10,equal_nan=True):raise AssertionError(f'{m}: {r[m]} != {value}')
            checks+=1
    return checks


def verify_inputs(frames):
    audit={'bundle':BUNDLE.relative_to(ROOT).as_posix(),'seed_csv_equality':{},'arithmetic_checks':{},'grid':{},'source_summary_checks':{},'graph_checks':[]}
    partial=[]
    for kind,df in frames.items():
        original=pd.read_csv(BUNDLE/SOURCES[kind])
        name={'basic':'metrics','posthoc':'coverage_metrics','validation':'validation_threshold_metrics'}[kind]
        individual=[]
        for seed in SEEDS:
            f=pd.read_csv(BUNDLE/f'seed{seed}'/(name+'.csv'));f['seed']=seed;individual.append(f)
        rebuilt=pd.concat(individual,ignore_index=True)
        keys=KEYS+(['target_coverage'] if kind!='basic' else [])
        sort=keys+['seed']
        pd.testing.assert_frame_equal(original.sort_values(sort).reset_index(drop=True),rebuilt[original.columns].sort_values(sort).reset_index(drop=True),check_exact=True)
        audit['seed_csv_equality'][kind]=len(original)
        expected=list(itertools.product(MODELS,SCENARIOS,[False] if kind=='validation' else [False,True],*([TARGETS] if kind!='basic' else []),SEEDS))
        actual=set(map(tuple,df[sort].itertuples(index=False,name=None)))
        missing=[x for x in expected if x not in actual]
        audit['grid'][kind]={'expected':len(expected),'actual':len(df),'missing':missing}
        if missing:raise AssertionError('Missing condition rows in supplied bundle')
        audit['arithmetic_checks'][kind]=verify_arithmetic(df)
        stored=pd.read_csv(BUNDLE/('summary_5seeds.csv' if kind=='basic' else name+'_summary.csv'))
        skeys=KEYS+(['method','target_coverage'] if kind!='basic' else [])
        groups=df[df.feasible].groupby(skeys,dropna=False)
        count=0
        for _,row in stored.iterrows():
            g=groups.get_group(tuple(row[k] for k in skeys))
            for m in PRIMARY+['precision_decided_only']:
                for stat in ['mean','std']:
                    calculated=g[m].mean() if stat=='mean' else g[m].std(ddof=1)
                    if not np.isclose(row[m+'_'+stat],calculated,rtol=1e-10,atol=1e-12,equal_nan=True):raise AssertionError('Stored summary mismatch')
                    count+=1
                if g[m].notna().sum()!=len(SEEDS):
                    partial.append(dict(kind=kind,**{k:row[k] for k in skeys},metric=m,n_valid=int(g[m].notna().sum()),stored_mean=row[m+'_mean']))
        audit['source_summary_checks'][kind]=count
    g=np.load(ROOT/'data/banc_graph.npz',allow_pickle=False)
    logs=[];metadata=[]
    for seed in SEEDS:
        f=BUNDLE/f'seed{seed}'
        info=json.loads((f/'experiment_info.json').read_text())
        npz=np.load(f/'experiment_graph.npz',allow_pickle=False)
        assert np.array_equal(g['mask'],npz['mask']) and np.array_equal(g['ids'],npz['ids']) and np.array_equal(g['signs'],npz['signs'])
        mask=npz['trained_rewired_mask']
        assert np.array_equal(mask.sum(0),g['mask'].sum(0)) and np.array_equal(mask.sum(1),g['mask'].sum(1))
        assert mask.sum()==1733 and not np.diag(mask).any()
        audit['graph_checks'].append({'seed':seed,'neurons':len(g['ids']),'edges':int(mask.sum()),'gaba':int((g['signs']<0).sum()),'degree_preserved':True,'different_mask_entries':int((mask!=g['mask']).sum())})
        for path,digest in info['code_sha256'].items():assert sha(ROOT/path)==digest, f'Code provenance mismatch {path}'
        for path,digest in info['input_sha256'].items():assert sha(path)==digest, f'Input provenance mismatch {path}'
        metadata.append({k:info[k] for k in ['seed','attack_seed','batch_seed','rewire_seed','training_rows','validation_rows','test_rows','python','versions','split_audit']})
        for model,log in info['training_log'].items():logs.append(dict(seed=seed,model=model,**log))
    audit['metadata']=metadata
    audit['source_summary_partial_groups']=len(partial)
    return audit,pd.DataFrame(partial),pd.DataFrame(logs)


def gate_effect(df):
    a=df[df.safety_gate].copy();b=df[~df.safety_gate].copy()
    joined=a.merge(b,on=['model','scenario','seed'],suffixes=('_on','_off'),validate='one_to_one')
    records=[]
    for _,r in joined.iterrows():
        row={k:r[k] for k in ['model','scenario','seed']}
        for m in RATES:
            row[m+'_off']=r[m+'_off'];row[m+'_on']=r[m+'_on'];row[m+'_delta_pp']=100*(r[m+'_on']-r[m+'_off'])
        for m in COUNTS:
            row[m+'_off']=r[m+'_off'];row[m+'_on']=r[m+'_on'];row[m+'_delta']=r[m+'_on']-r[m+'_off']
        records.append(row)
    return pd.DataFrame(records)


def save(df,name):
    df.to_csv(OUT/'tables'/name,index=False,encoding='utf-8-sig',na_rep='NA',float_format='%.15g')


def make_figures(frames,summaries,pair_summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#126b75','#c65c32','#555b66','#6974b7','#9a5b8b','#a18a3d','#539b70']
    scenario_names={'normal':'정상','missing_light':'조도 결손','co2_spike':'CO₂ 이상치'}
    def finish(fig,name,title,foot):
        fig.suptitle(title+'\nrevised_audit_20261024 · 시드 42–46 (n=5)',fontsize=13)
        fig.text(.02,.015,foot,fontsize=9)
        fig.tight_layout(rect=[0,.06,1,.92])
        fig.savefig(OUT/'figures'/(name+'.png'),dpi=180)
        fig.savefig(OUT/'figures'/(name+'.svg'))
        plt.close(fig)
    def stats(frame,keys,metric):
        g=frame
        for k,v in keys.items():g=g[g[k].eq(v)]
        r=g[g.metric.eq(metric)].iloc[0]
        return 100*r['mean'],100*r['std']
    # 1. All seven models, both gates, both class errors.
    for gate in [False,True]:
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for j,scenario in enumerate(SCENARIOS):
            for i,m in enumerate(PRIMARY[:2]):
                ax=axes[i,j]
                vals=[stats(summaries['basic'],dict(model=model,scenario=scenario,safety_gate=gate),m) for model in MODELS]
                ax.bar(np.arange(7),[v[0] for v in vals],yerr=[v[1] for v in vals],color=colors,capsize=3)
                ax.set_xticks(np.arange(7),MODELS,rotation=60,ha='right',fontsize=8)
                ax.set_title(scenario_names[scenario]);ax.set_ylabel(('오경보율' if i==0 else '미탐지율')+' (%)')
                ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        finish(fig,'01_errors_gate_'+str(gate),'기본 평가: 모델별 오류율 · 안전장치 '+('ON' if gate else 'OFF'),
               '오차막대: 5시드 표본 표준편차(ddof=1), 신뢰구간 아님. 분모는 실제 클래스 전체. 패널별 y축 범위가 다름.')
    # 2. Sensors and F1/recall/coverage, both gates.
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for i,gate in enumerate([False,True]):
        for j,m in enumerate(['f1_decided_only','recall_decided_only','coverage']):
            ax=axes[i,j]
            for model,color in zip(MODELS,colors):
                yy=[stats(summaries['basic'],dict(model=model,scenario=s,safety_gate=gate),m)[0] for s in SCENARIOS]
                ax.plot(range(3),yy,'o-',label=model,color=color,linewidth=1)
            ax.set_xticks(range(3),[scenario_names[s] for s in SCENARIOS]);ax.set_ylabel({'f1_decided_only':'판단 표본 F1','recall_decided_only':'판단 표본 Recall','coverage':'Coverage'}[m]+' (%)')
            ax.set_title('기본 안전장치 '+('ON' if gate else 'OFF'));ax.grid(alpha=.2)
            if m=='coverage':
                ax.set_ylim(0,103);ax.set_yticks([0,25,50,75,100])
    axes[0,0].legend(fontsize=7)
    finish(fig,'02_sensor_conditions','센서 조건별 평균 F1·Recall·Coverage','F1/Recall은 판단 표본에 한정. 패널별 y축 범위가 다름. 선은 5시드 평균이며 인과효과·유의성을 뜻하지 않음.')
    # 3. Posthoc 50/75/90 and feasibility, all seven models, same panel conditions.
    for gate in [False,True]:
        fig,axes=plt.subplots(2,3,figsize=(13,8))
        for j,scenario in enumerate(SCENARIOS):
            for i,m in enumerate(PRIMARY[:2]):
                ax=axes[i,j]
                for model,color in zip(MODELS,colors):
                    yy=[stats(summaries['posthoc'],dict(model=model,scenario=scenario,safety_gate=gate,target_coverage=t),m)[0] for t in [.5,.75,.9]]
                    ax.plot([50,75,90],yy,'o-',label=model,color=color,linewidth=1)
                ax.set_xticks([50,75,90]);ax.set_xlabel('저장된 목표 Coverage (%)')
                ax.set_ylabel(('오경보율' if i==0 else '미탐지율')+' (%)');ax.set_title(scenario_names[scenario]);ax.grid(alpha=.2)
                if gate and scenario!='normal':ax.text(.55,.75,'75%, 90%\nfeasible=False',transform=ax.transAxes,ha='center',color='#a03a25')
        axes[0,0].legend(fontsize=7)
        finish(fig,'03_posthoc_gate_'+str(gate),'사후 동일 Coverage 오류율 · 입력 오류 강제 보류 '+('ON' if gate else 'OFF'),
               '점수 순위 기반 사후 분석. 기본 0.2/0.8 보류와 다름. 결측/불가능 지점은 선을 잇지 않음. 판단 표본은 모델별로 다를 수 있음.')
    # 4. Gate removes errors and also removes decisions.
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for j,scenario in enumerate(SCENARIOS):
        for i,m in enumerate(PRIMARY[:2]):
            ax=axes[i,j]
            for model,color in zip(MODELS,colors):
                x=[stats(summaries['basic'],dict(model=model,scenario=scenario,safety_gate=g),'abstain_rate')[0] for g in [False,True]]
                y=[stats(summaries['basic'],dict(model=model,scenario=scenario,safety_gate=g),m)[0] for g in [False,True]]
                ax.plot(x,y,'o-',color=color,label=model)
            ax.set_title(scenario_names[scenario]);ax.set_xlabel('판단 보류율 (%)');ax.set_ylabel(('오경보율' if i==0 else '미탐지율')+' (%)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    finish(fig,'04_gate_tradeoff','기본 안전장치 OFF → ON: 오류와 보류의 절충','점: 5시드 평균. 왼쪽(보류율 0%)은 OFF, 오른쪽은 ON. 선은 동일 모델의 두 저장 조건. 일부 결정을 보류한 결과이며 예측 점수 개선이 아님.')
    # 5. All seed paired differences BANC minus rewired, error and coverage.
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    base=frames['basic'];p=paired(base,['scenario','safety_gate'])
    for i,gate in enumerate([False,True]):
        for j,m in enumerate(PRIMARY[:2]+['coverage']):
            ax=axes[i,j]
            for k,s in enumerate(SCENARIOS):
                g=p[(p.contrast=='banc_signed minus rewired_signed')&(p.scenario==s)&(p.safety_gate==gate)&(p.metric==m)].sort_values('seed')
                ax.plot(g.seed,g.delta_pp,'o-',label=scenario_names[s])
            ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(SEEDS);ax.set_xlabel('학습/재배선 연동 시드')
            ax.set_ylabel({'false_alarm_per_empty':'오경보율','unsafe_false_empty_per_occupied':'미탐지율','coverage':'Coverage'}[m]+' 차이 (%p)');ax.set_title('기본 안전장치 '+('ON' if gate else 'OFF'));ax.grid(alpha=.2)
    axes[0,0].legend()
    finish(fig,'05_paired_banc_rewired','동일 시드 차이: BANC signed - Rewired signed','오류율 음수는 BANC의 낮은 오류, 양수는 높은 오류. Coverage 차이에는 우열 방향을 부여하지 않음. 유의성 검정 아님.')
    # 6. Validation targets do not guarantee test coverage.
    fig,axes=plt.subplots(1,3,figsize=(13,5))
    for ax,s in zip(axes,SCENARIOS):
        for model,color in zip(MODELS,colors):
            y=[stats(summaries['validation'],dict(model=model,scenario=s,safety_gate=False,target_coverage=t),'coverage')[0] for t in TARGETS]
            ax.plot(np.array(TARGETS)*100,y,'o-',label=model,color=color,linewidth=1)
        ax.plot([0,100],[0,100],'k--',linewidth=.7,label='목표=테스트');ax.set_title(scenario_names[s]);ax.set_xlabel('검증 목표 (%)');ax.set_ylabel('실제 테스트 Coverage (%)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    finish(fig,'06_validation_coverage','검증 점수 임계값을 테스트에 고정 적용','입력 오류 강제 보류 OFF. 목표 대비 실제 테스트 Coverage가 다르므로 동일 Coverage 성능이라고 부르지 않음.')


def main():
    for d in ['tables','figures']:(OUT/d).mkdir(exist_ok=True)
    frames={kind:load_recorded(kind) for kind in SOURCES}
    audit,partial,training=verify_inputs(frames)
    summaries={};pair_summaries={}
    for kind,df in frames.items():
        context=['scenario','safety_gate']+(['method','target_coverage'] if kind!='basic' else [])
        s=summary(df,['model']+context);summaries[kind]=s
        p=paired(df,context);ps=paired_summary(p,context);pair_summaries[kind]=ps
        save(df,kind+'_individual.csv');save(s,kind+'_summary.csv');save(p,kind+'_paired_individual.csv');save(ps,kind+'_paired_summary.csv')
    gates=gate_effect(frames['basic']);save(gates,'gate_effect_individual.csv')
    save(summary(gates,['model','scenario'],[m+'_delta_pp' for m in RATES]+[m+'_delta' for m in COUNTS]),'gate_effect_summary.csv')
    # Fault minus normal, paired within model and seed, not cross-seed means.
    changes=[]
    for scenario in SCENARIOS[1:]:
        left=frames['basic'][frames['basic'].scenario.eq(scenario)];right=frames['basic'][frames['basic'].scenario.eq('normal')]
        join=left.merge(right,on=['model','seed','safety_gate'],suffixes=('_fault','_normal'),validate='one_to_one')
        for _,r in join.iterrows():
            changes.append(dict(model=r.model,seed=r.seed,safety_gate=r.safety_gate,scenario=scenario,
                **{m:100*(r[m+'_fault']-r[m+'_normal']) for m in PRIMARY}))
    changes=pd.DataFrame(changes);save(changes,'fault_minus_normal_individual_pp.csv')
    save(summary(changes,['model','scenario','safety_gate'],PRIMARY),'fault_minus_normal_summary_pp.csv')
    save(partial,'source_summary_partial_metric_groups.csv');save(training,'training_metadata.csv')
    p=paired(frames['posthoc'],['scenario','safety_gate','target_coverage'])
    old=pd.read_csv(BUNDLE/'paired_coverage_differences.csv')
    n=0
    for _,r in old.iterrows():
        g=p[(p.seed==r.seed)&(p.scenario==r.scenario)&(p.safety_gate==r.safety_gate)&(p.target_coverage==r.target_coverage)&(p.contrast==r.contrast)]
        for m in PRIMARY[:2]+['coverage']:
            assert len(g[g.metric.eq(m)])==1
            assert np.isclose(g[g.metric.eq(m)].delta_pp.iloc[0]/100,r[m+'_difference'],atol=1e-12,rtol=1e-10)
            n+=1
    audit['saved_paired_values_verified']=n
    audit['infeasible_posthoc_rows']=int((~frames['posthoc'].feasible).sum())
    audit['zero_decision_validation_rows']=int(frames['validation'].n_decided.eq(0).sum())
    audit['prediction_identity_limit']='No row-level prediction/decision vectors saved. Same selected sample identities cannot be verified.'
    (OUT/'verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    inputs=sorted(BUNDLE.glob('*.csv'))+sorted(BUNDLE.glob('seed*/*.csv'))+sorted(BUNDLE.glob('seed*/experiment_info.json'))+sorted(BUNDLE.glob('seed*/experiment_graph.npz'))
    (OUT/'sources.json').write_text(json.dumps({'bundle':str(BUNDLE),'python':platform.python_version(),'files':{str(p.relative_to(ROOT)):sha(p) for p in inputs}},ensure_ascii=False,indent=2),encoding='utf-8')
    make_figures(frames,summaries,pair_summaries)
    print(json.dumps({k:audit[k] for k in ['seed_csv_equality','arithmetic_checks','infeasible_posthoc_rows','zero_decision_validation_rows','source_summary_partial_groups','saved_paired_values_verified']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
