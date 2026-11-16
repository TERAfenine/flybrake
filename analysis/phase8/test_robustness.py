import unittest,json
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from analysis.phase8 import robustness as r

class RobustnessTests(unittest.TestCase):
    def test_exact_nested_masks_and_raw_changes(self):
        x=np.full((9752,4),400.);before=x.copy();previous=np.zeros(len(x),bool)
        for rate,count in zip(r.RATES,[0,975,1950,2925]):
            a,m=r.inject(x,'missing_light',rate,700);b,n=r.inject(x,'co2_spike',rate,700)
            self.assertEqual(m.sum(),count);np.testing.assert_array_equal(m,n)
            self.assertTrue((~previous|m).all());previous=m
            np.testing.assert_array_equal(np.isnan(a[:,2]),m)
            np.testing.assert_array_equal(b[:,3]-x[:,3],m*3000)
            np.testing.assert_array_equal(x,before)
            np.testing.assert_array_equal(a,r.inject(x,'missing_light',rate,700)[0])
        self.assertFalse(np.array_equal(m,r.inject(x,'missing_light',.3,701)[1]))

    def test_all_saved_predictions_and_masks(self):
        df=pd.read_csv(r.DEFAULT_OUT/'metrics.csv');self.assertEqual(len(df),1200)
        self.assertTrue(np.allclose(df.coverage+df.abstain_rate,1))
        self.assertTrue((df.n_total==df.n_decided+df.n_abstained).all())
        for _,g in df.groupby(['attack_seed','fault','fault_rate']):self.assertEqual(g.mask_sha256.nunique(),1)
        for path in r.DEFAULT_OUT.glob('*_predictions.npz'):
            z=np.load(path,allow_pickle=False);rid=path.name.removesuffix('_predictions.npz')
            for attack in [700,701,702]:
                for fault in r.FAULTS:
                    for rate in r.RATES:
                        key=f'a{attack}_{fault}_p{int(rate*100)}'
                        self.assertEqual(z[key+'_mask'].sum(),int(rate*len(z['y_true'])))
                        for gate in [False,True]:
                            d=z[key+f'_decision{int(gate)}'];m=r.decision_metrics(z['y_true'],d)
                            name=rid.split('_t')[0];seed=int(rid.split('_t')[1][:2]);rewire=int(rid.split('_r')[1]) if '_r' in rid else None
                            g=df[df.model.eq(name)&df.train_seed.eq(seed)&df.attack_seed.eq(attack)&df.fault.eq(fault)&df.fault_rate.eq(rate)&df.safety_gate.eq(gate)]
                            if rewire:g=g[g.rewire_seed.eq(rewire)]
                            self.assertEqual(len(g),1)
                            for metric in r.METRICS:self.assertTrue(np.isclose(m[metric],g.iloc[0][metric],atol=1e-12,equal_nan=True))

    def test_checkpoint_replay_without_training(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=r.ROOT/'results') as tmp:
            out=r.Path(tmp)/'replay'
            df=r.run(out,models=['banc_signed'],train_seeds=[42],attack_seeds=[700],faults=['missing_light'],rates=[0.,.3])
            old=pd.read_csv(r.DEFAULT_OUT/'metrics.csv')
            old=old[old.model.eq('banc_signed')&old.train_seed.eq(42)&old.attack_seed.eq(700)&old.fault.eq('missing_light')&old.fault_rate.isin([0.,.3])]
            np.testing.assert_allclose(df.sort_values(['fault_rate','safety_gate'])[r.METRICS],old.sort_values(['fault_rate','safety_gate'])[r.METRICS],atol=1e-12)

    def test_ui_filters_and_existing_pages(self):
        at=AppTest.from_file(str(r.ROOT/'app.py'),default_timeout=90).run();self.assertFalse(at.exception)
        at.switch_page('pages/3_Robustness_Lab.py').run();self.assertFalse(at.exception)
        self.assertEqual(len(at.get('plotly_chart')),4)
        at.multiselect(key='rob_seeds').set_value([42]).run();self.assertFalse(at.exception)
        at.selectbox(key='rob_fault').set_value('co2_spike').run();self.assertFalse(at.exception)
        at.multiselect(key='rob_models').set_value([]).run();self.assertFalse(at.exception);self.assertTrue(at.info)
        at.switch_page('pages/2_FlyBrake_Neurons.py').run();self.assertFalse(at.exception)
        at.switch_page('pages/1_FlyBrake_Lab.py').run();self.assertFalse(at.exception)

if __name__=='__main__':unittest.main()
