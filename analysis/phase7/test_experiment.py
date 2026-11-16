"""Phase 7: independent capacity, seed, denominator and saved-prediction checks."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from analysis.phase7 import experiment as e


def independent_metrics(y, d):
    chosen=d>=0;neg=y==0;pos=y==1
    ne=int(neg.sum());np_=int(pos.sum());de=int((neg&chosen).sum());dp=int((pos&chosen).sum())
    fp=int((neg&(d==1)).sum());fn=int((pos&(d==0)).sum());tp=dp-fn;nd=de+dp
    ratio=lambda x,n: x/n if n else np.nan
    return dict(n_empty=ne,n_occupied=np_,n_decided=nd,n_decided_empty=de,n_decided_occupied=dp,fp=fp,fn=fn,
        coverage=nd/len(y),abstain_rate=1-nd/len(y),empty_coverage=ratio(de,ne),occupied_coverage=ratio(dp,np_),
        false_alarm_per_empty=ratio(fp,ne),unsafe_false_empty_per_occupied=ratio(fn,np_),
        false_alarm_decided_empty=ratio(fp,de),miss_decided_occupied=ratio(fn,dp),
        precision_decided_only=(ratio(tp,tp+fp) if tp+fp else 0.) if nd else np.nan,
        recall_decided_only=(ratio(tp,dp) if dp else 0.) if nd else np.nan,
        f1_decided_only=(ratio(2*tp,2*tp+fp+fn) if 2*tp+fp+fn else 0.) if nd else np.nan)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        with np.load(e.ROOT/'data/banc_graph.npz',allow_pickle=False) as g:
            cls.mask=g['mask'].copy();cls.signs=g['signs'].copy()
        cls.result=e.ROOT/'results/phase7_20261105'

    def test_parameter_counts_and_generic_autograd_reachability(self):
        self.assertEqual(sum(p.numel() for p in e.MatchedMLP().parameters()),1961)
        c=e.capacities(self.mask)
        self.assertEqual((c['allocated'],c['mask_active'],c['output_effective']),(9553,2070,1965))
        self.assertEqual(c['backward_reach_sizes'],[24,86,93,93])
        for topology in [self.mask]+[e.rewire_preserving_degrees(self.mask,s) for s in e.CONFIG['rewire_seeds']]:
            model=e.initialize('banc_signed',19,topology,self.signs).double()
            # Initial zero bias hides potential paths; generic small learned bias exposes them.
            with torch.no_grad():model.bias.copy_(torch.linspace(.011,.109,len(self.signs),dtype=torch.float64))
            x=torch.randn(7,8,dtype=torch.float64)
            model(x).sum().backward()
            observed=sum(int(torch.count_nonzero(p.grad)) for p in model.parameters())
            self.assertEqual(observed,e.capacities(topology)['output_effective'])
            self.assertEqual(sum(p.numel() for p in model.parameters()),e.capacities(topology)['allocated'])
            self.assertEqual(int(torch.count_nonzero(model.raw_weights.grad)),e.capacities(topology)['effective_recurrent'])

    def test_rewiring_and_training_initialization_are_independent(self):
        original=self.mask.copy();a=e.rewire_preserving_degrees(self.mask,1042);b=e.rewire_preserving_degrees(self.mask,1043)
        self.assertFalse(np.array_equal(a,b));np.testing.assert_array_equal(self.mask,original)
        for topology in [a,b]:
            np.testing.assert_array_equal(topology.sum(0),original.sum(0))
            np.testing.assert_array_equal(topology.sum(1),original.sum(1))
            self.assertFalse(np.diag(topology).any())
        x=e.initialize('banc_signed',42,self.mask,self.signs)
        z=e.initialize('rewired_signed',42,a,self.signs)
        for (name,p),(other,q) in zip(x.named_parameters(),z.named_parameters()):
            self.assertEqual(name,other);self.assertTrue(torch.equal(p,q))
        e.initialize('rewired_signed',46,b,self.signs)
        np.testing.assert_array_equal(e.rewire_preserving_degrees(self.mask,1042),a)
        changed=e.initialize('banc_signed',43,self.mask,self.signs)
        self.assertFalse(torch.equal(x.raw_weights,changed.raw_weights))

    def test_training_preprocessing_and_augmentation_use_only_training_file(self):
        calls=[];original=e.load_occ
        def tracked(path):
            calls.append(Path(path).name)
            return original(path)
        with patch.object(e,'load_occ',side_effect=tracked):
            first=e.load_training();second=e.load_training()
        self.assertEqual(calls,['datatraining.txt','datatraining.txt'])
        for a,b in zip(first[1:],second[1:]):np.testing.assert_array_equal(a,b)
        prep,Ftr,ytr,Fval,yval,_=first
        self.assertEqual((Ftr.shape,Fval.shape),((6514,8),(1629,8)))
        X,_=original(e.ROOT/'data/datatraining.txt');tr=X[:6514]
        np.testing.assert_allclose(prep.medians,np.nanmedian(tr,axis=0))
        np.testing.assert_allclose(prep.scaler.mean_,np.where(np.isfinite(tr),tr,np.nanmedian(tr,axis=0)).mean(0))
        altered=e.augment(tr,np.random.default_rng(e.CONFIG['augmentation_seed']))
        np.testing.assert_array_equal(Ftr[:,4:],~np.isfinite(altered))

    def test_guard_boundaries_and_class_denominators(self):
        scores=np.array([.2,.8,.5,.1,.9,.7]);bad=np.array([False,False,False,True,False,False])
        y=np.array([1,0,1,0,1,0])
        np.testing.assert_array_equal(e.basic_decisions(scores,bad,True),[0,1,-1,-1,1,-1])
        np.testing.assert_array_equal(e.basic_decisions(scores,bad,False),[0,1,1,0,1,1])
        measured=e.enrich(y,e.basic_decisions(scores,bad,True))
        self.assertAlmostEqual(measured['false_alarm_per_empty'],1/3)
        self.assertEqual(measured['false_alarm_decided_empty'],1.)
        self.assertAlmostEqual(measured['unsafe_false_empty_per_occupied'],1/3)
        self.assertEqual(measured['miss_decided_occupied'],.5)
        self.assertIsNone(e.select_exact(scores,bad,1.,True))
        empty=e.enrich(y,np.full(len(y),-1))
        self.assertTrue(np.isnan(empty['f1_decided_only']))
        self.assertTrue(np.isnan(empty['miss_decided_occupied']))
        no_pos=e.enrich(y,np.where(y==0,0,-1))
        self.assertEqual(no_pos['recall_decided_only'],0.)
        self.assertTrue(np.isnan(no_pos['miss_decided_occupied']))

    def test_output_refuses_overwrite_and_unrelated_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(e,'ROOT',root):
                with self.assertRaises(ValueError):e.prepare_output(root/'unrelated')
                out=e.prepare_output(root/'results/new_phase7')
                sentinel=out/'keep.txt';sentinel.write_text('original',encoding='utf-8')
                with self.assertRaises(FileExistsError):e.prepare_output(out)
                self.assertEqual(sentinel.read_text(encoding='utf-8'),'original')

    def test_saved_predictions_reconstruct_all_metrics(self):
        if not (self.result/'COMPLETE.json').exists():self.skipTest('Full Phase 7 run has not completed')
        source=pd.read_csv(e.ROOT/'data/datatest2.txt')
        for run in pd.read_csv(self.result/'training_metadata.csv').itertuples():
            folder=self.result/run.run_id
            pred=pd.read_csv(folder/'predictions.csv.gz',float_precision='round_trip')
            val=pd.read_csv(folder/'validation_predictions.csv.gz',float_precision='round_trip')
            basic=pd.read_csv(folder/'metrics.csv');cov=pd.read_csv(folder/'coverage_metrics.csv');thresholds=pd.read_csv(folder/'validation_threshold_metrics.csv')
            self.assertEqual(len(pred),len(source)*3);self.assertEqual(len(val),1629*3)
            self.assertFalse(pred.duplicated(['scenario','row_index']).any())
            for frame in [pred,val]:
                self.assertEqual(set(frame.run_id),{run.run_id})
                self.assertEqual(set(frame.train_seed),{run.train_seed})
                self.assertTrue(frame.score.between(0,1).all())
            # Restore inference dtype before threshold arithmetic; NN scores were float32.
            dtype=np.float64 if run.model=='logistic' else np.float32
            for scenario in e.SCENARIOS:
                g=pred[pred.scenario.eq(scenario)].sort_values('row_index');v=val[val.scenario.eq(scenario)].sort_values('row_index')
                np.testing.assert_array_equal(g.row_index,np.arange(len(source)))
                np.testing.assert_array_equal(g.y_true,source.Occupancy)
                np.testing.assert_array_equal(g.timestamp,source.date)
                y=g.y_true.to_numpy();score=g.score.to_numpy(dtype=dtype);invalid=g.invalid.to_numpy(bool)
                expected_fault=(np.random.default_rng(e.CONFIG['attack_seed']).random(len(y))<.25) if scenario!='normal' else np.zeros(len(y),bool)
                np.testing.assert_array_equal(g.corrupted,expected_fault)
                for gate in [False,True]:
                    d=np.where(invalid,-1,np.where(score>=.8,1,np.where(score<=.2,0,-1))) if gate else (score>=.5).astype(int)
                    np.testing.assert_array_equal(g['decision_on' if gate else 'decision_off'],d)
                    row=basic[basic.scenario.eq(scenario)&basic.safety_gate.eq(gate)].iloc[0]
                    self.compare_metrics(row,y,d)
                    eligible=np.flatnonzero(~invalid if gate else np.ones(len(y),bool))
                    tie=np.random.default_rng(e.CONFIG['tie_seed']).random(len(y))
                    ordered=eligible[np.lexsort((tie[eligible],-np.abs(score[eligible]-.5)))]
                    for target in e.TARGETS:
                        row=cov[cov.scenario.eq(scenario)&cov.safety_gate.eq(gate)&cov.target_coverage.eq(target)].iloc[0]
                        k=int(np.floor(target*len(y)));col=f'posthoc_guard{int(gate)}_target{int(target*100)}'
                        self.assertEqual(bool(row.feasible),k<=len(eligible))
                        if k>len(eligible):
                            self.assertTrue(g[col].isna().all());self.assertTrue(pd.isna(row.coverage));continue
                        d=np.full(len(y),-1);chosen=ordered[:k];d[chosen]=(score[chosen]>=.5).astype(int)
                        np.testing.assert_array_equal(g[col],d);self.compare_metrics(row,y,d)
                for target in e.TARGETS:
                    row=thresholds[thresholds.scenario.eq(scenario)&thresholds.target_coverage.eq(target)].iloc[0]
                    threshold=float(np.quantile(np.abs(v.score.to_numpy(dtype=dtype)-.5),1-target))
                    self.assertAlmostEqual(row.validation_confidence_threshold,threshold,places=13)
                    d=np.where(np.abs(score-.5)>=threshold,(score>=.5).astype(int),-1)
                    np.testing.assert_array_equal(g[f'validation_target{int(target*100)}'],d)
                    self.compare_metrics(row,y,d)

    def compare_metrics(self,row,y,d):
        for key,value in independent_metrics(y,d).items():
            if np.isnan(value):self.assertTrue(pd.isna(row[key]),key)
            else:self.assertAlmostEqual(float(row[key]),value,places=12,msg=key)

    def test_completed_grid_replay_and_provenance(self):
        if not (self.result/'COMPLETE.json').exists():self.skipTest('Full Phase 7 run has not completed')
        manifest=json.loads((self.result/'manifest.json').read_text())
        cfg=manifest['config'];runs=pd.read_csv(self.result/'training_metadata.csv')
        expected={(model,s,None) for model in ['banc_signed','mlp_matched','logistic'] for s in cfg['train_seeds']}
        expected|={('rewired_signed',s,r) for s in cfg['train_seeds'] for r in cfg['rewire_seeds']}
        actual={(r.model,r.train_seed,int(r.rewire_seed) if pd.notna(r.rewire_seed) else None) for r in runs.itertuples()}
        self.assertEqual(actual,expected);self.assertEqual(len(runs),len(expected))
        replays=json.loads((self.result/'replay_checks.json').read_text())
        self.assertEqual({r['run_id'] for r in replays},{'banc_signed_t42','mlp_matched_t42'})
        self.assertTrue(all(r['state_exact'] and r['validation_scores_exact'] for r in replays))
        for filename,digest in manifest['files'].items():self.assertEqual(e.sha256(e.ROOT/filename),digest)
        for name,key in [('PROTOCOL_KO.md','protocol_sha256'),('EXECUTION_SCALE_KO.md','scale_sha256')]:
            self.assertEqual(e.sha256(e.ROOT/'analysis/phase7'/name),manifest[key])


if __name__=='__main__':unittest.main()
