"""Streamlit interaction checks, including displayed numbers and Plotly traces."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.lab_results import METRICS, display_value

ROOT=Path(__file__).resolve().parent
PAGE=ROOT/'pages/1_FlyBrake_Lab.py'

def box(at,label):
    return next(w for w in at.selectbox if w.label==label)

class LabUITests(unittest.TestCase):
    def check_ok(self,at):
        self.assertFalse(at.exception,[e.message for e in at.exception])

    def assert_main_numbers(self,at,bundle,seed,scenario,gate,pair):
        raw=pd.read_csv(bundle/'metrics_5seeds.csv')
        expected=[]
        for model in pair:
            group=raw[(raw.model==model)&(raw.scenario==scenario)&(raw.safety_gate==gate)]
            if seed is not None:group=group[group.seed==seed]
            for metric in METRICS:
                value=group[metric].mean() if seed is None else group[metric].iloc[0]
                sd=group[metric].std(ddof=1) if seed is None else np.nan
                expected.append(display_value(value,sd,seed is None))
        self.assertEqual([m.value for m in at.metric],expected)
        spec=json.loads(at.get('plotly_chart')[0].proto.spec)
        self.assertEqual([trace['name'] for trace in spec['data']],pair)
        for model,trace in zip(pair,spec['data']):
            group=raw[(raw.model==model)&(raw.scenario==scenario)&(raw.safety_gate==gate)]
            if seed is not None:group=group[group.seed==seed]
            np.testing.assert_allclose(trace['y'],group[METRICS[:2]].mean(),rtol=1e-14)

    def test_legacy_navigation_and_lab_interactions(self):
        with patch.dict(os.environ,{'FLYBRAKE_ROOT':str(ROOT)}):
            at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
            self.check_ok(at)
            self.assertEqual(len(at.tabs),4)
            at.switch_page('pages/1_FlyBrake_Lab.py').run()
            self.check_ok(at)
            self.assertEqual(at.title[0].value,'FlyBrake Lab')
            self.assert_main_numbers(at,ROOT/'results',None,'normal',True,['banc_signed','rewired_signed'])
            bundle=ROOT/'results/revised_audit_20261024'
            box(at,'결과 묶음').select(bundle).run()
            self.check_ok(at)
            self.assert_main_numbers(at,bundle,None,'normal',True,['banc_signed','rewired_signed'])
            for seed in [42,43]:
                box(at,'학습 시드').select(seed).run()
                self.check_ok(at)
                self.assert_main_numbers(at,bundle,seed,'normal',True,['banc_signed','rewired_signed'])
            box(at,'모델 A').select('banc_free').run()
            box(at,'모델 B').select('logistic').run()
            box(at,'센서 조건').select('co2_spike').run()
            at.checkbox(key='lab_gate').uncheck().run()
            self.check_ok(at)
            self.assert_main_numbers(at,bundle,43,'co2_spike',False,['banc_free','logistic'])
            box(at,'학습 시드').select(None).run()
            self.check_ok(at)
            self.assert_main_numbers(at,bundle,None,'co2_spike',False,['banc_free','logistic'])
            at.checkbox(key='lab_gate').check().run()
            at.select_slider(key='lab_target').set_value(1.).run()
            self.check_ok(at)
            self.assertTrue(any('달성 불가능' in x.value for x in at.info))
            # The identical coverage table must show N/A, not zeros.
            matched=at.dataframe[1].value
            self.assertTrue(matched['banc_free'].eq('N/A').all())

    def test_missing_and_empty_results_ui(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'FLYBRAKE_ROOT':tmp}):
            at=AppTest.from_file(str(PAGE),default_timeout=60).run()
            self.check_ok(at)
            self.assertTrue(any('묶음이 없습니다' in x.value for x in at.info))
            result=Path(tmp)/'results';result.mkdir()
            (result/'metrics_5seeds.csv').write_text('')
            at.run();self.check_ok(at)
            self.assertTrue(at.warning)

    def test_file_change_invalidates_cache_and_missing_condition(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'FLYBRAKE_ROOT':tmp}):
            result=Path(tmp)/'results';result.mkdir()
            file=result/'metrics_5seeds.csv'
            raw=pd.read_csv(ROOT/'results/metrics_5seeds.csv')
            raw.to_csv(file,index=False)
            at=AppTest.from_file(str(PAGE),default_timeout=60).run()
            self.check_ok(at)
            previous=at.metric[0].value
            # Remove a measured row rather than inventing a performance value.
            raw=raw[~((raw.model=='banc_signed')&(raw.scenario=='normal')&raw.safety_gate&(raw.seed==42))]
            raw.to_csv(file,index=False)
            at.run();self.check_ok(at)
            self.assertNotEqual(at.metric[0].value,previous)
            self.assertEqual(at.metric[0].value,'N/A')
            self.assertTrue(any('시드 불완전' in x.value for x in at.info))

if __name__=='__main__':unittest.main()
