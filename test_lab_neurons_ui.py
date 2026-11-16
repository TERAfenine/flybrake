import os
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.lab_ablation import load_experiment,compare_input,demo_input

ROOT=Path(__file__).resolve().parent

class NeuronUITests(unittest.TestCase):
    def check_ok(self,at):self.assertFalse(at.exception,[e.message for e in at.exception])

    def test_navigation_ablation_restore_seed_and_sensor(self):
        with patch.dict(os.environ,{'FLYBRAKE_ROOT':str(ROOT)}):
            at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
            self.check_ok(at)
            at.switch_page('pages/2_FlyBrake_Neurons.py').run();self.check_ok(at)
            at.selectbox(key='neuron_index').select(95).run();self.check_ok(at)
            scores=lambda:[m.value for m in at.metric if m.label=='재실 예측 점수']
            self.assertEqual(scores()[0],scores()[1])
            at.checkbox(key='neuron_disabled').check().run();self.check_ok(at)
            expected=compare_input(load_experiment(ROOT/'results/seed42',ROOT),demo_input([22.5,45,150,700]),95)
            self.assertEqual(scores(),[f'{expected[k]:.8f}' for k in ['original','altered']])
            self.assertNotEqual(scores()[0],scores()[1])
            at.checkbox(key='neuron_disabled').uncheck().run();self.check_ok(at)
            self.assertEqual(scores()[0],scores()[1])
            at.checkbox(key='neuron_disabled').check().run()
            at.selectbox(key='neuron_seed_results').select(43).run();self.check_ok(at)
            self.assertFalse(at.checkbox(key='neuron_disabled').value)
            expected=compare_input(load_experiment(ROOT/'results/seed43',ROOT),demo_input([22.5,45,150,700]))
            self.assertEqual(scores()[0],f'{expected["original"]:.8f}')
            at.selectbox(key='neuron_scenario').select('missing_light').run();self.check_ok(at)
            self.assertTrue(any('판단 보류: 예' in m.value for m in at.markdown))
            for mode in ['선택 뉴런 중심','GABA 예측 강조']:
                at.selectbox(key='neuron_view').select(mode).run();self.check_ok(at)
            at.selectbox(key='neuron_index').select(48).run();self.check_ok(at)

    def test_incompatible_bundle_keeps_graph_without_inference(self):
        with patch.dict(os.environ,{'FLYBRAKE_ROOT':str(ROOT)}):
            at=AppTest.from_file(str(ROOT/'pages/2_FlyBrake_Neurons.py'),default_timeout=90).run()
            at.selectbox(key='neuron_bundle').select(ROOT/'results/revised_audit_20261024').run()
            self.check_ok(at)
            self.assertTrue(any('호환' in w.value for w in at.warning))
            self.assertEqual(len(at.get('plotly_chart')),1)
            self.assertFalse(any(m.label=='재실 예측 점수' for m in at.metric))

if __name__=='__main__':unittest.main()
