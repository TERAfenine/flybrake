"""Real saved checkpoints: equivalence, zero activation, pairing and graph identity."""
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.lab_ablation import (load_graph,load_experiment,ablated_logits,compare_input,
                              demo_input,decision,run_folders)
from src.lab_network import neuron_connections,node_from_event,network_figure
from src.data import load_occ

ROOT=Path(__file__).resolve().parent

class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs={s:load_experiment(ROOT/f'results/seed{s}',ROOT) for s in range(42,47)}
        cls.raw,_=load_occ(ROOT/'data/datatest2.txt')

    def test_no_ablation_matches_original_all_five_seeds(self):
        for experiment in self.runs.values():
            for raw in [self.raw[:32],demo_input([22.5,45,150,700],'missing_light'),
                        demo_input([22.5,45,150,700],'co2_spike')]:
                features,_=experiment.prep.transform(raw)
                x=torch.tensor(features,dtype=torch.float32)
                with torch.inference_mode():expected=experiment.model(x)
                actual=ablated_logits(experiment.model,x)
                self.assertTrue(torch.equal(expected,actual))

    def test_clamp_every_step_including_readout_and_off_restore(self):
        experiment=self.runs[42]
        features,_=experiment.prep.transform(self.raw[:16])
        original,original_states=ablated_logits(experiment.model,features,return_states=True)
        for neuron in [0,48,95]:
            altered,states=ablated_logits(experiment.model,features,neuron,True)
            self.assertEqual(len(states),4)
            for state in states:self.assertTrue(torch.all(state[:,neuron]==0))
            other=[i for i in range(96) if i!=neuron]
            torch.testing.assert_close(states[0][:,other],original_states[0][:,other],rtol=0,atol=0)
            self.assertTrue(torch.equal(altered,experiment.model.output_map(states[-1][:,-24:]).squeeze(-1)))
        self.assertTrue(torch.equal(original,ablated_logits(experiment.model,features)))

    def test_weights_and_graph_unchanged_during_computation(self):
        e=self.runs[42];before={k:v.clone() for k,v in e.model.state_dict().items()}
        graph=e.graph['mask'].copy()
        r=compare_input(e,demo_input([22.5,45,150,700]),95)
        self.assertNotEqual(r['delta'],0.)  # Actual saved model, not a synthetic performance claim.
        for k,v in e.model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        np.testing.assert_array_equal(graph,e.graph['mask'])
        restored=compare_input(e,demo_input([22.5,45,150,700]))
        self.assertEqual(restored['delta'],0.)

    def test_seed_pairing_preprocessing_and_no_fit(self):
        with patch('src.data.SensorPrep.fit',side_effect=AssertionError('must not fit')):
            for seed,e in self.runs.items():
                self.assertEqual(e.seed,seed);self.assertEqual(e.run.name,f'seed{seed}')
                disk=torch.load(e.run/'banc_signed.pt',map_location='cpu',weights_only=True)
                self.assertTrue(torch.equal(disk['raw_weights'],e.model.raw_weights))
                raw=demo_input([22.5,45,150,700],'missing_light')
                features,invalid=e.prep.transform(raw)
                self.assertTrue(invalid[0]);self.assertEqual(features[0,6],1.)
                result=compare_input(e,raw)
                with torch.no_grad():expected=torch.sigmoid(e.model(torch.tensor(features,dtype=torch.float32))).item()
                self.assertEqual(expected,result['original'])
                self.assertTrue(result['original_decision'][1])
        self.assertFalse(torch.equal(self.runs[42].model.raw_weights,self.runs[43].model.raw_weights))

    def test_decision_threshold_boundaries(self):
        for p,expected in [(0.,'비재실'),(.2,'비재실'),(.200001,'판단 보류'),(.799999,'판단 보류'),(.8,'재실'),(1.,'재실')]:
            self.assertEqual(decision(p,False)[0],expected)
            self.assertEqual(decision(p,True)[0],'판단 보류')
        self.assertEqual(decision(.5,True,False)[0],'재실')
        self.assertEqual(decision(.499,False,False)[0],'비재실')

    def test_demo_faults_are_single_input_and_do_not_mutate(self):
        values=np.array([22.5,45,150,700]);copy=values.copy()
        self.assertTrue(np.isnan(demo_input(values,'missing_light')[0,2]))
        self.assertEqual(demo_input(values,'co2_spike')[0,3],3700)
        np.testing.assert_array_equal(values,copy)

    def test_id_direction_counts_and_click_precision(self):
        g=self.runs[42].graph
        self.assertEqual(len(g['ids']),96);self.assertEqual(g['mask'].sum(),1733)
        for index in [0,48,95]:
            links=neuron_connections(g,index)
            self.assertEqual(sum(links.post_index==index),int(g['mask'][index].sum()))
            self.assertEqual(sum(links.pre_index==index),int(g['mask'][:,index].sum()))
            for _,row in links.iterrows():
                self.assertEqual(row['출발 BANC ID'],str(g['ids'][row.pre_index]))
                self.assertEqual(row['도착 BANC ID'],str(g['ids'][row.post_index]))
                self.assertEqual(row['시냅스 수'],g['counts'][row.post_index,row.pre_index])
            event={'selection':{'points':[{'customdata':[str(index),str(g['ids'][index])]}]}}
            self.assertEqual(node_from_event(event,g['ids']),index)
        self.assertIsNone(node_from_event({'selection':{'points':[{'customdata':['0',float(g['ids'][0])]}]}},g['ids']))
        self.assertIsNone(node_from_event({'selection':{'points':[{'curve_number':0}]}},g['ids']))

    def test_figure_all_nodes_edges_and_disabled_visibility(self):
        g=self.runs[42].graph
        for mode in ['전체 96개 뉴런','선택 뉴런 중심','GABA 예측 강조']:
            fig=network_figure(g,95,mode,95)
            nodes=[r for trace in fig.data[2:4] for r in trace.customdata]
            self.assertTrue(all(isinstance(r[1],str) for r in nodes))
            self.assertEqual(fig.data[-1].marker.symbol,'x')
            if mode!='선택 뉴런 중심':
                self.assertEqual(len(nodes),96);self.assertEqual(len(fig.data[1].x),1733)
            else:self.assertEqual(len(fig.data[1].x),len(neuron_connections(g,95)))

    def test_incompatible_and_mismatched_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError,'호환'):
            load_experiment(ROOT/'results/revised_audit_20261024/seed42',ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'seed42';run.mkdir()
            (run/'experiment_info.json').write_text(json.dumps({'seed':43}))
            with self.assertRaisesRegex(ValueError,'학습 시드'):load_experiment(run,ROOT)
        with patch('src.lab_ablation.torch.load') as mocked:
            state={k:v.clone() for k,v in self.runs[42].model.state_dict().items()}
            state['signs'][0]*=-1;mocked.return_value=state
            with self.assertRaisesRegex(ValueError,'signs'):load_experiment(ROOT/'results/seed42',ROOT)

    def test_invalid_neurons_and_inputs(self):
        e=self.runs[42];features,_=e.prep.transform(self.raw[:1])
        for index in [-1,96,1.5,True,[0,1]]:
            with self.assertRaises(ValueError):ablated_logits(e.model,features,index)
        with self.assertRaises(ValueError):compare_input(e,self.raw[:2])
        self.assertEqual(list(run_folders(ROOT/'results')[ROOT/'results']),[42,43,44,45,46])

if __name__=='__main__':unittest.main()
