"""Phase 5 regression: navigation must not silently change scientific conditions."""
from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest
from src.lab_ablation import load_experiment, compare_input, demo_input

ROOT = Path(__file__).resolve().parent


class Phase5Tests(unittest.TestCase):
    def ok(self, app):
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(any('default value' in w.value for w in app.warning))

    def start(self, page):
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=90).run()
        app.switch_page(page).run()
        self.ok(app)
        return app

    def test_research_roundtrip_keeps_individual_seed_bundle_and_measurements(self):
        page = 'pages/1_FlyBrake_Lab.py'
        app = self.start(page)
        bundle = ROOT/'results/revised_audit_20261024'
        app.selectbox(key='lab_bundle').select(bundle).run()
        app.selectbox(key='lab_seed_'+bundle.name).select(43).run()
        app.selectbox(key='lab_a_'+bundle.name).select('banc_free').run()
        app.selectbox(key='lab_scenario_'+bundle.name).select('co2_spike').run()
        app.checkbox(key='lab_gate').uncheck().run()
        app.select_slider(key='lab_target').set_value(.75).run()
        app.selectbox(key='research_tradeoff_model').select('rewired_signed').run()
        self.ok(app)
        before = app.dataframe[0].value.copy()
        app.switch_page('app.py').run(); self.ok(app)
        app.switch_page('pages/0_Start_Experiment.py').run(); self.ok(app)
        app.switch_page(page).run(); self.ok(app)
        self.assertEqual(app.selectbox(key='lab_bundle').value, bundle)
        self.assertEqual(app.selectbox(key='lab_seed_'+bundle.name).value, 43)
        self.assertEqual(app.selectbox(key='lab_a_'+bundle.name).value, 'banc_free')
        self.assertEqual(app.selectbox(key='lab_scenario_'+bundle.name).value, 'co2_spike')
        self.assertFalse(app.checkbox(key='lab_gate').value)
        self.assertEqual(app.select_slider(key='lab_target').value, .75)
        self.assertEqual(app.selectbox(key='research_tradeoff_model').value, 'rewired_signed')
        self.assertTrue(before.equals(app.dataframe[0].value))
        # A fresh browser session still starts with the documented defaults.
        fresh = self.start(page)
        self.assertIsNone(fresh.selectbox(key='lab_seed_results').value)

    def test_neuron_roundtrip_keeps_input_and_intervention_but_seed_change_resets_it(self):
        page = 'pages/2_FlyBrake_Neurons.py'
        app = self.start(page)
        app.selectbox(key='neuron_seed_results').select(43).run()
        app.selectbox(key='neuron_index').select(95).run()
        app.number_input(key='sensor_Temperature (°C)').set_value(24.).run()
        app.selectbox(key='neuron_scenario').select('co2_spike').run()
        app.checkbox(key='neuron_guard').uncheck().run()
        app.checkbox(key='neuron_disabled').check().run()
        next(b for b in app.button if b.label=='＋ 확대').click().run()
        self.ok(app)
        app.switch_page('app.py').run(); self.ok(app)
        app.switch_page(page).run(); self.ok(app)
        self.assertEqual(app.selectbox(key='neuron_seed_results').value, 43)
        self.assertEqual(app.selectbox(key='neuron_index').value, 95)
        self.assertEqual(app.number_input(key='sensor_Temperature (°C)').value, 24.)
        self.assertEqual(app.selectbox(key='neuron_scenario').value, 'co2_spike')
        self.assertFalse(app.checkbox(key='neuron_guard').value)
        self.assertTrue(app.checkbox(key='neuron_disabled').value)
        self.assertEqual(app.session_state['neuron_zoom'], 1.5)
        expected = compare_input(load_experiment(ROOT/'results/seed43', ROOT),
                                 demo_input([24.,45.,150.,700.], 'co2_spike'),95,False)
        self.assertEqual([m.value for m in app.metric if m.label=='재실 예측 점수'],
                         [f'{expected[k]:.8f}' for k in ['original','altered']])
        app.selectbox(key='neuron_seed_results').select(44).run(); self.ok(app)
        self.assertFalse(app.checkbox(key='neuron_disabled').value)
        # Compatibility failure stays a failure: graph only, no substitute inference.
        app.selectbox(key='neuron_bundle').select(ROOT/'results/revised_audit_20261024').run()
        self.ok(app)
        self.assertTrue(any('호환' in w.value for w in app.warning))
        self.assertFalse(any(m.label=='재실 예측 점수' for m in app.metric))

    def test_guide_selection_feedback_matches_node_and_intervention(self):
        app = self.start('pages/0_Start_Experiment.py')
        app.radio(key='ux_step').set_value('2 · 연결망 탐색').run()
        app.selectbox(key='ux_node').select(95).run(); self.ok(app)
        banners = [m.value for m in app.markdown if 'class="ux-selection"' in m.value]
        self.assertEqual(len(banners), 2)
        self.assertEqual(banners[0], banners[1])
        self.assertIn('뉴런 95', banners[1])
        app.radio(key='ux_step').set_value('3 · 뉴런 비활성화').run()
        app.checkbox(key='ux_disabled').check().run(); self.ok(app)
        banners = [m.value for m in app.markdown if 'class="ux-selection"' in m.value]
        self.assertTrue(all('× 개입 설정 ON' in b and '뉴런 95' in b for b in banners))


if __name__ == '__main__':
    unittest.main()
