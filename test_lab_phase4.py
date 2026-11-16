"""Phase 4: view-only filtering, explicit reset, measured ON/OFF comparison."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.lab_ablation import load_graph
from src.lab_network import visible_connections,network_figure
from src.lab_experience import gate_comparison,TRADEOFF_METRICS

ROOT=Path(__file__).resolve().parent

class Phase4Tests(unittest.TestCase):
    def ok(self,at):self.assertFalse(at.exception,[e.message for e in at.exception])

    def guide(self):
        at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        self.ok(at)
        at.switch_page('pages/0_Start_Experiment.py').run();self.ok(at)
        return at

    def test_gaba_filter_is_exact_induced_graph_without_mutation(self):
        g=load_graph(ROOT/'data/banc_graph.npz')
        before={k:v.copy() for k,v in g.items() if isinstance(v,np.ndarray)}
        index=int(np.flatnonzero(g['signs']<0)[0])
        nodes,post,pre=visible_connections(g,index,'전체 96개 뉴런','GABA 예측만')
        self.assertEqual(nodes,np.flatnonzero(g['signs']<0).tolist())
        self.assertEqual(len(nodes),50)
        expected_post,expected_pre=np.where((g['mask']>0)&(g['signs'][:,None]<0)&(g['signs'][None,:]<0))
        np.testing.assert_array_equal(post,expected_post);np.testing.assert_array_equal(pre,expected_pre)
        local,lp,ls=visible_connections(g,index,'선택 뉴런 중심','GABA 예측만')
        self.assertTrue(np.all((lp==index)|(ls==index)))
        self.assertTrue(all(g['signs'][i]<0 for i in local))
        for zoom in [1.,2.]:
            fig=network_figure(g,index,'전체 96개 뉴런',index,nt_filter='GABA 예측만',zoom=zoom)
            records=[row for trace in fig.data[2:4] for row in trace.customdata]
            self.assertEqual([int(r[0]) for r in records],nodes)
            self.assertEqual(len(fig.data[1].x),len(pre))
            self.assertEqual(fig.data[-1].marker.symbol,'x')
        for key,value in before.items():np.testing.assert_array_equal(g[key],value)

    def test_filter_zoom_selection_and_full_experiment_reset(self):
        at=self.guide()
        at.radio(key='ux_step').set_value('3 · 뉴런 비활성화').run()
        at.selectbox(key='ux_nt_filter').select('GABA 예측만').run();self.ok(at)
        self.assertEqual(len(at.selectbox(key='ux_node').options),50)
        at.checkbox(key='ux_disabled').check().run()
        at.selectbox(key='ux_scenario').select('co2_spike').run()
        at.checkbox(key='ux_guard').uncheck().run()
        at.number_input(key='ux_sensor_온도 (°C)').set_value(24.).run()
        at.button(key='ux_zoom_in').click().run();self.ok(at)
        self.assertEqual(at.session_state['ux_zoom'],1.5)
        at.button(key='ux_fit').click().run()
        self.assertEqual(at.session_state['ux_zoom'],1.)
        at.button(key='guide_reset').click().run();self.ok(at)
        self.assertEqual(at.radio(key='ux_step').value,'3 · 뉴런 비활성화')
        self.assertEqual(at.selectbox(key='ux_nt_filter').value,'전체')
        self.assertEqual(at.selectbox(key='ux_node').value,0)
        self.assertFalse(at.checkbox(key='ux_disabled').value)
        self.assertTrue(at.checkbox(key='ux_guard').value)
        self.assertEqual(at.selectbox(key='ux_scenario').value,'normal')
        self.assertEqual(at.number_input(key='ux_sensor_온도 (°C)').value,22.5)

    def test_measured_gate_table_matches_csv_and_missing_seed_stays_na(self):
        raw=pd.read_csv(ROOT/'results/metrics_5seeds.csv')
        for model in raw.model.unique():
            for scenario in raw.scenario.unique():
                view=gate_comparison(raw,model,scenario,[42,43,44,45,46])
                for gate in [False,True]:
                    group=raw[(raw.model==model)&(raw.scenario==scenario)&(raw.safety_gate==gate)]
                    np.testing.assert_allclose(view['ON' if gate else 'OFF'],group[TRADEOFF_METRICS].mean().values,rtol=1e-13,atol=1e-15)
                np.testing.assert_allclose(view['변화 (%p)'],100*(view.ON-view.OFF))
        incomplete=raw[~((raw.model=='banc_signed')&(raw.seed==42)&raw.safety_gate)]
        view=gate_comparison(incomplete,'banc_signed','normal',[42,43,44,45,46])
        self.assertTrue(view.ON.isna().all());self.assertTrue(view['변화 (%p)'].isna().all())
        self.assertTrue(view.OFF.notna().all())

    def test_research_neuron_filter_reset_keeps_checkpoint(self):
        at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        at.switch_page('pages/2_FlyBrake_Neurons.py').run()
        at.selectbox(key='neuron_seed_results').select(43).run()
        at.selectbox(key='neuron_nt_filter').select('GABA 예측만').run();self.ok(at)
        self.assertEqual(len(at.selectbox(key='neuron_index').options),50)
        at.checkbox(key='neuron_disabled').check().run()
        next(b for b in at.button if b.label=='입력·뉴런 설정 초기화').click().run();self.ok(at)
        self.assertEqual(at.selectbox(key='neuron_seed_results').value,43)
        self.assertEqual(at.selectbox(key='neuron_index').value,0)
        self.assertFalse(at.checkbox(key='neuron_disabled').value)

if __name__=='__main__':unittest.main()
