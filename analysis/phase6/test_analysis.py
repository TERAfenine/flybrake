"""Independent numerical checks plus missing/infeasible/zero-decision cases."""
import math
import statistics
import unittest
import numpy as np
import pandas as pd
from analysis.phase6 import analyze as a


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames={kind:a.load_recorded(kind) for kind in a.SOURCES}

    def test_all_primary_aggregates_against_independent_scalar_statistics(self):
        for kind,df in self.frames.items():
            keys=a.KEYS+(['target_coverage'] if kind!='basic' else [])
            result=a.summary(df,keys,a.PRIMARY)
            for _,row in result.iterrows():
                g=df
                for k in keys:g=g[g[k].eq(row[k])]
                values=g.loc[g.feasible,row.metric].dropna().tolist()
                self.assertEqual(row.n_valid,len(values))
                if len(values)==5:
                    self.assertAlmostEqual(row['mean'],math.fsum(values)/5,places=13)
                    self.assertAlmostEqual(row['std'],statistics.stdev(values),places=13)
                else:
                    self.assertTrue(pd.isna(row['mean']))
                    self.assertTrue(pd.isna(row['std']))

    def test_infeasible_and_zero_decided_class_are_not_filled(self):
        c=self.frames['posthoc'];v=self.frames['validation']
        self.assertEqual((~c.feasible).sum(),245)
        self.assertTrue(c.loc[~c.feasible,a.RATES].isna().all().all())
        self.assertEqual(v.n_decided.eq(0).sum(),10)
        self.assertTrue(v.loc[v.n_decided.eq(0),'f1_decided_only'].isna().all())
        self.assertTrue(v.loc[v.n_decided_occupied.eq(0),'miss_decided_occupied'].isna().all())
        # No positives among decided rows: sklearn's recorded recall=0 is preserved,
        # but conditional miss is undefined, never reinterpreted as successful detection.
        subset=c[c.feasible & c.n_decided_occupied.eq(0) & c.n_decided.gt(0)]
        self.assertGreater(len(subset),0)
        self.assertTrue(subset.recall_decided_only.eq(0).all())
        self.assertTrue(subset.miss_decided_occupied.isna().all())

    def test_missing_seed_or_metric_and_partial_feasibility_leave_mean_unavailable(self):
        g=self.frames['basic'].query("model=='banc_signed' and scenario=='normal' and safety_gate==False").copy()
        for broken in [g.iloc[:-1],g.assign(false_alarm_per_empty=g.false_alarm_per_empty.mask(g.seed.eq(42))),
                       g.assign(feasible=~g.seed.eq(42))]:
            s=a.summary(broken,a.KEYS,['false_alarm_per_empty']).iloc[0]
            self.assertTrue(pd.isna(s['mean']))
            self.assertEqual(s.n_valid,4)

    def test_paired_join_uses_seed_not_position_and_missing_side_is_na(self):
        d=self.frames['basic'];pair=[('banc_signed','rewired_signed')]
        p=a.paired(d.sample(frac=1,random_state=81),['scenario','safety_gate'],pair,a.PRIMARY)
        lookup=d.set_index(['model','scenario','safety_gate','seed'])
        for _,r in p.iterrows():
            x=lookup.loc[('banc_signed',r.scenario,r.safety_gate,r.seed),r.metric]
            y=lookup.loc[('rewired_signed',r.scenario,r.safety_gate,r.seed),r.metric]
            self.assertAlmostEqual(r.delta_pp,100*(x-y),places=13)
        broken=d[~(d.model.eq('rewired_signed')&d.seed.eq(42)&d.scenario.eq('normal')&~d.safety_gate)]
        p=a.paired(broken,['scenario','safety_gate'],pair,a.PRIMARY)
        miss=p[(p.seed==42)&(p.scenario=='normal')&~p.safety_gate]
        self.assertTrue(miss.delta_pp.isna().all());self.assertTrue(miss.status.eq('missing_pair').all())
        s=a.paired_summary(p,['scenario','safety_gate'])
        self.assertTrue(s[(s.scenario=='normal')&~s.safety_gate].mean_delta_pp.isna().all())

    def test_invalid_boolean_and_duplicate_keys_fail_loudly(self):
        with self.assertRaises(ValueError):a.bool_column(pd.Series(['True','unknown']))
        with self.assertRaises(ValueError):a.bool_column(pd.Series([True,None]))
        d=self.frames['basic'];dup=pd.concat([d,d.iloc[:1]])
        with self.assertRaises(ValueError):a.summary(dup,a.KEYS)
        with self.assertRaises(ValueError):a.paired(dup,['scenario','safety_gate'])

    def test_saved_counts_and_preexisting_paired_file_reconcile(self):
        for df in self.frames.values():self.assertGreater(a.verify_arithmetic(df),0)
        old=pd.read_csv(a.BUNDLE/'paired_coverage_differences.csv')
        d=self.frames['posthoc'].set_index(['model','seed','scenario','safety_gate','target_coverage'])
        for _,r in old.iterrows():
            first,second=r.contrast.split(' minus ')
            idx=(r.seed,r.scenario,r.safety_gate,r.target_coverage)
            for metric in a.PRIMARY[:2]+['coverage']:
                expected=d.loc[(first,)+idx,metric]-d.loc[(second,)+idx,metric]
                self.assertAlmostEqual(expected,r[metric+'_difference'],places=13)
        # Confusion count corruption is caught independently of source summary means.
        corrupt=self.frames['basic'].iloc[:1].copy();corrupt.loc[corrupt.index[0],'fp']+=1
        with self.assertRaises(AssertionError):a.verify_arithmetic(corrupt)


if __name__=='__main__':unittest.main()
