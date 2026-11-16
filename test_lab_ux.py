"""Phase 3 journeys verify actual displayed calculations, state and unavailable data."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.lab_ablation import load_experiment,compare_input,demo_input
from src.lab_results import display_value
from src.lab_ux import MODEL_NAMES

ROOT=Path(__file__).resolve().parent
STEPS=['1 · 센서 오류','2 · 연결망 탐색','3 · 뉴런 비활성화','4 · 모델 비교']

class UXTests(unittest.TestCase):
    def start(self):
        at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        self.assertFalse(at.exception)
        at.switch_page('pages/0_Start_Experiment.py').run()
        self.ok(at)
        return at

    def ok(self,at):
        self.assertFalse(at.exception,[e.message for e in at.exception])

    def step(self,at,n):
        at.radio(key='ux_step').set_value(STEPS[n]).run();self.ok(at)

    def test_sensor_gate_actual_scores(self):
        at=self.start()
        experiment=load_experiment(ROOT/'results/seed42',ROOT)
        for scenario in ['normal','missing_light','co2_spike']:
            for guard in [True,False]:
                at.selectbox(key='ux_scenario').select(scenario).run()
                at.checkbox(key='ux_guard').set_value(guard).run();self.ok(at)
                expected=[compare_input(experiment,demo_input([22.5,45,150,700],s),guard=guard) for s in ['normal',scenario]]
                self.assertEqual([m.value for m in at.metric],[f'{r["original"]:.8f}' for r in expected])
                spec=json.loads(at.get('plotly_chart')[0].proto.spec)
                np.testing.assert_allclose(spec['data'][0]['x'],[r['original'] for r in expected],rtol=1e-14)
                self.assertTrue(any(('판단 보류: 예' if expected[1]['original_decision'][1] else '판단 보류: 아니오') in c.value for c in at.caption))

    def test_next_previous_node_ablation_and_cross_page_settings(self):
        at=self.start()
        at.selectbox(key='ux_scenario').select('missing_light').run()
        next(b for b in at.button if b.label=='다음 실험 →').click().run();self.ok(at)
        self.assertEqual(at.radio(key='ux_step').value,STEPS[1])
        self.assertEqual(at.selectbox(key='ux_graph_mode').value,'선택 뉴런 중심')
        at.selectbox(key='ux_node').select(95).run()
        next(b for b in at.button if b.label=='다음 실험 →').click().run();self.ok(at)
        self.assertEqual(at.selectbox(key='ux_node').value,95)
        self.assertEqual(at.selectbox(key='ux_scenario').value,'missing_light')
        at.checkbox(key='ux_disabled').check().run();self.ok(at)
        expected=compare_input(load_experiment(ROOT/'results/seed42',ROOT),demo_input([22.5,45,150,700],'missing_light'),95)
        self.assertEqual([m.value for m in at.metric if m.label=='모델 출력 점수'],[f'{expected[k]:.8f}' for k in ['original','altered']])
        at.selectbox(key='ux_node').select(85).run();self.ok(at)
        self.assertFalse(at.checkbox(key='ux_disabled').value)
        at.switch_page('pages/1_FlyBrake_Lab.py').run();self.ok(at)
        at.switch_page('pages/0_Start_Experiment.py').run();self.ok(at)
        self.assertEqual(at.radio(key='ux_step').value,STEPS[2])
        self.assertEqual(at.selectbox(key='ux_node').value,85)
        self.assertEqual(at.selectbox(key='ux_scenario').value,'missing_light')
        next(b for b in at.button if b.label=='← 이전 실험').click().run();self.ok(at)
        self.assertEqual(at.radio(key='ux_step').value,STEPS[1])

    def test_comparison_reads_same_csv_and_preserves_internal_names(self):
        at=self.start();self.step(at,3)
        at.selectbox(key='ux_compare_scenario').select('co2_spike').run()
        at.checkbox(key='ux_compare_guard').uncheck().run();self.ok(at)
        raw=pd.read_csv(ROOT/'results/metrics_5seeds.csv')
        metrics=['false_alarm_per_empty','unsafe_false_empty_per_occupied','coverage','abstain_rate']
        expected=[]
        for model in ['banc_signed','logistic','rewired_signed']:
            group=raw[(raw.model==model)&(raw.scenario=='co2_spike')&(~raw.safety_gate)]
            expected.extend(display_value(group[k].mean(),group[k].std(ddof=1),True) for k in metrics)
        self.assertEqual([m.value for m in at.metric],expected)
        spec=json.loads(at.get('plotly_chart')[0].proto.spec)
        self.assertEqual([t['name'] for t in spec['data']],[MODEL_NAMES[k] for k in ['banc_signed','logistic','rewired_signed']])
        self.assertEqual(len(MODEL_NAMES),7)

    def test_missing_data_is_message_not_fabricated_result(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'FLYBRAKE_ROOT':temp}):
            at=AppTest.from_file(str(ROOT/'pages/0_Start_Experiment.py'),default_timeout=90).run()
            self.ok(at);self.assertTrue(at.warning);self.assertFalse(at.metric)
            self.step(at,1);self.assertTrue(at.warning);self.assertFalse(at.get('plotly_chart'))
            self.step(at,3);self.assertFalse(at.metric)
            self.assertTrue(at.info or at.warning)

if __name__=='__main__':unittest.main()
