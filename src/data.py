"""UCI Occupancy Detection (ID 357), independent of BANC structural data."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
FEATURES=['Temperature','Humidity','Light','CO2']

def load_occ(path):
    df=pd.read_csv(path)
    if not set(FEATURES+['Occupancy']).issubset(df.columns):
        raise ValueError('Expected UCI dataset 357: Temperature, Humidity, Light, CO2, Occupancy')
    return df[FEATURES].to_numpy(dtype=float), df['Occupancy'].to_numpy(dtype=np.int64)

class SensorPrep:
    def fit(self, x):
        self.medians=np.nanmedian(x,axis=0)
        self.scaler=StandardScaler().fit(np.where(np.isfinite(x),x,self.medians))
        lo=np.nanpercentile(x,.1,axis=0)
        hi=np.nanpercentile(x,99.9,axis=0)
        iqr=np.nanpercentile(x,75,axis=0)-np.nanpercentile(x,25,axis=0)
        self.lower=lo-3*iqr
        self.upper=hi+3*iqr
        return self
    def transform(self, x):
        finite=np.isfinite(x)
        corrupted=np.where(finite,x,self.medians)
        scaled=self.scaler.transform(corrupted).astype('float32')
        flags=(~finite).astype('float32')
        bad=np.any(~finite | (x < self.lower) | (x > self.upper),axis=1)
        return np.concatenate([scaled,flags],axis=1),bad

def augment(x, rng):
    x=x.copy()
    rows=rng.random(len(x)) < .12
    col=rng.integers(0,x.shape[1],size=len(x))
    x[np.flatnonzero(rows),col[rows]]=np.nan
    return x

def sensor_attack(x, name, seed=42):
    x=x.copy(); rng=np.random.default_rng(seed)
    sel=rng.random(len(x)) < .25
    if name=='missing_light': x[sel,2]=np.nan
    if name=='co2_spike': x[sel,3] += 3000
    if name=='normal': return x
    if name not in ('missing_light','co2_spike','normal'): raise ValueError(name)
    return x
