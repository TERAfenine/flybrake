"""Read-only regression checks using real measurements and isolated failure fixtures."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.lab_results import (discover_bundles, load_results, source_signature, select_rows,
                             comparison_view, display_value, METRICS)
from src.lab_charts import comparison_figure

ROOT = Path(__file__).resolve().parent

class LabResultsTests(unittest.TestCase):
    def setUp(self):
        self.original = load_results(ROOT/'results').frame
        self.revised = load_results(ROOT/'results/revised_audit_20261024').frame

    def test_real_discovery_and_bundle_separation(self):
        found = discover_bundles(ROOT/'results')
        self.assertIn(ROOT/'results', found)
        self.assertIn(ROOT/'results/revised_audit_20261024', found)
        self.assertFalse(any(p.name.startswith('seed') for p in found))
        self.assertEqual(self.original.model.nunique(), 4)
        self.assertEqual(self.revised.model.nunique(), 7)
        self.assertEqual(len(self.original), 120)
        self.assertEqual(len(self.revised), 210)

    def test_individual_seed_42_43_and_every_filter_equals_csv(self):
        raw = pd.read_csv(ROOT/'results/revised_audit_20261024/metrics_5seeds.csv')
        for seed in [42,43]:
            for scenario in raw.scenario.unique():
                for gate in [False,True]:
                    for model in raw.model.unique():
                        rows = select_rows(self.revised, [model], scenario, gate, seed)
                        view = comparison_view(rows,[model],[seed],False)
                        expected = raw[(raw.model==model)&(raw.scenario==scenario)&
                                       (raw.safety_gate==gate)&(raw.seed==seed)].iloc[0]
                        np.testing.assert_allclose(view.value, expected[METRICS].astype(float), rtol=0,atol=0)
                        self.assertTrue(view['std'].isna().all())
        a=select_rows(self.revised,['banc_signed'],'normal',True,42).iloc[0]
        b=select_rows(self.revised,['banc_signed'],'normal',True,43).iloc[0]
        self.assertNotEqual(a.coverage,b.coverage)

    def test_mean_sd_and_plot_are_exact(self):
        pair=['banc_signed','logistic']
        rows=select_rows(self.revised,pair,'co2_spike',True)
        view=comparison_view(rows,pair,[42,43,44,45,46],True)
        fig=comparison_figure(view,METRICS[:2],True)
        self.assertEqual([t.name for t in fig.data],pair)
        for i,model in enumerate(pair):
            raw=rows[rows.model==model]
            data=view[view.model==model]
            np.testing.assert_allclose(data.value,raw[METRICS].mean(),rtol=1e-14)
            # pandas groupby.std and Series.std differ by floating roundoff.
            np.testing.assert_allclose(data['std'],raw[METRICS].std(ddof=1),rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(fig.data[i].y,data.value.iloc[:2],rtol=0,atol=0)
            np.testing.assert_allclose(fig.data[i].error_y.array,data['std'].iloc[:2],rtol=0,atol=0)

    def test_coverage_and_thresholds_remain_separate(self):
        bundle=ROOT/'results/revised_audit_20261024'
        cov=load_results(bundle,'coverage_metrics').frame
        val=load_results(bundle,'validation_threshold_metrics').frame
        self.assertEqual(set(cov.method),{'test_score_top_k_no_labels'})
        self.assertEqual(set(val.method),{'validation_threshold_test_coverage_not_guaranteed'})
        pair=['banc_signed','logistic']
        rows=select_rows(cov,pair,'missing_light',True,None,1.,'test_score_top_k_no_labels')
        view=comparison_view(rows,pair,range(42,47),True)
        self.assertTrue(view.value.isna().all())
        self.assertTrue(view.status.str.contains('불가능').all())
        rows=select_rows(cov,pair,'normal',False,42,.5)
        view=comparison_view(rows,pair,[42],False)
        self.assertEqual(view[view.metric=='coverage'].value.nunique(),1)

    def test_missing_rows_not_silently_averaged(self):
        rows=select_rows(self.revised,['logistic'],'normal',True)
        view=comparison_view(rows[rows.seed!=42],['logistic'],range(42,47),True)
        self.assertTrue(view.value.isna().all())
        self.assertTrue(view.status.str.contains('불완전').all())
        view=comparison_view(rows.iloc[:0],['unknown'],[42],False)
        self.assertTrue(view.value.isna().all())

    def test_undefined_no_decisions_shown_na_without_mutation(self):
        rows=select_rows(self.original,['logistic'],'normal',True,42)
        rows['coverage']=0.;rows['f1_decided_only']=0.;rows['recall_decided_only']=0.
        before=rows.copy(deep=True)
        view=comparison_view(rows,['logistic'],[42],False).set_index('metric')
        self.assertTrue(pd.isna(view.loc['f1_decided_only','value']))
        pd.testing.assert_frame_equal(rows,before)
        self.assertEqual(display_value(np.nan),'N/A')

    def test_files_missing_empty_invalid_and_boolean_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(discover_bundles(root),[])
            self.assertTrue(load_results(root).frame.empty)
            file=root/'metrics_5seeds.csv'
            file.write_text('')
            with self.assertRaises(ValueError):load_results(root)
            row=self.original.iloc[:1].copy()
            for value in ['False','false','0',0,False]:
                row['safety_gate']=value;row.to_csv(file,index=False)
                self.assertFalse(load_results(root).frame.safety_gate.iloc[0])
            row['safety_gate']='unknown';row.to_csv(file,index=False)
            with self.assertRaises(ValueError):load_results(root)
            row.drop(columns=['model']).to_csv(file,index=False)
            with self.assertRaises(ValueError):load_results(root)

    def test_per_seed_fallback_no_duplicate_combined_and_file_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'seed42').mkdir()
            original=self.original[self.original.seed==42].copy()
            path=root/'seed42/metrics.csv'
            original.drop(columns=['seed']).to_csv(path,index=False)
            self.assertEqual(len(load_results(root).frame),24)
            original.to_csv(root/'metrics_5seeds.csv',index=False)
            table=load_results(root)
            self.assertEqual(len(table.frame),24)
            self.assertEqual(len(table.sources),1)
            before=source_signature(root)
            original.iloc[:1].to_csv(root/'metrics_5seeds.csv',index=False)
            self.assertNotEqual(source_signature(root),before)

    def test_duplicate_and_missing_metric_handling(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'metrics_5seeds.csv'
            rows=self.original.iloc[:1]
            pd.concat([rows,rows]).to_csv(file,index=False)
            with self.assertRaises(ValueError):load_results(tmp)
            rows.drop(columns=['f1_decided_only']).to_csv(file,index=False)
            table=load_results(tmp)
            self.assertTrue(table.frame.f1_decided_only.isna().all())
            self.assertTrue(table.notices)

if __name__=='__main__':
    unittest.main()
