"""Regression tests for scientific controls; artificial inputs, not performance claims."""
import unittest
import numpy as np
import pandas as pd
import torch
from src.evaluation import summarize, decision_metrics, select_exact, METRICS
from src.connectome import rewire_preserving_degrees
from src.model import SignedConnectome

class ResearchTests(unittest.TestCase):
    def test_summary_preserves_conditions_and_rejects_missing_seeds(self):
        rows=[dict(model=m,scenario=s,safety_gate=g,seed=seed,**{k:float(seed) for k in METRICS})
              for m in ['a','b'] for s in ['normal','missing'] for g in [True,False] for seed in [42,43]]
        frame=pd.DataFrame(rows)
        result=summarize(frame,expected_seeds=[42,43])
        self.assertEqual(len(result),8)
        self.assertTrue((result.n_seeds==2).all())
        self.assertAlmostEqual(result.coverage_mean.iloc[0],42.5)
        with self.assertRaises(ValueError):summarize(frame.iloc[1:],expected_seeds=[42,43])
        with self.assertRaises(ValueError):summarize(pd.concat([frame,frame.iloc[:1]]))

    def test_equal_coverage_ties_and_infeasible_guard(self):
        invalid=np.array([True,True,False,False,False,False])
        p=np.full(6,.8)
        a=select_exact(p,invalid,.5)
        self.assertEqual((a>=0).sum(),3)
        np.testing.assert_array_equal(a,select_exact(p,invalid,.5))
        self.assertIsNone(select_exact(p,invalid,1.,True))
        self.assertEqual((select_exact(p,invalid,0.)>=0).sum(),0)
        guarded=select_exact(p,invalid,.5,True)
        self.assertTrue((guarded[invalid]==-1).all())

    def test_error_denominators(self):
        m=decision_metrics([0,0,1,1],[1,-1,0,-1])
        self.assertEqual(m['false_alarm_per_empty'],.5)
        self.assertEqual(m['unsafe_false_empty_per_occupied'],.5)
        self.assertEqual(m['false_alarm_decided_empty'],1.)
        self.assertEqual(m['coverage'],.5)
        self.assertTrue(np.isnan(decision_metrics([0,1],[-1,-1])['f1_decided_only']))

    def test_graph_and_initialization_controls(self):
        rng=np.random.default_rng(1)
        mask=(rng.random((24,24))<.3).astype('float32');np.fill_diagonal(mask,0)
        rewired=rewire_preserving_degrees(mask)
        np.testing.assert_equal(mask.sum(0),rewired.sum(0));np.testing.assert_equal(mask.sum(1),rewired.sum(1))
        self.assertTrue(np.any(mask!=rewired));self.assertEqual(np.trace(rewired),0)
        signs=np.where(np.arange(24)%2,-1,1).astype('float32')
        torch.manual_seed(1);signed=SignedConnectome(mask,signs)
        torch.manual_seed(1);free=SignedConnectome(mask,signs,constraint='free')
        x=torch.randn(8,8)
        torch.testing.assert_close(signed(x),free(x),rtol=0,atol=0)
        signed(x).sum().backward()
        self.assertTrue(torch.all(signed.raw_weights.grad[signed.mask==0]==0))

if __name__=='__main__':unittest.main()
