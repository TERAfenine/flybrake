"""Simple figure builder for genuinely measured model metrics."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
p=Path('results/metrics.csv')
if not p.exists():
    raise SystemExit('Run train.py with original BANC+UCI data first.')
df=pd.read_csv(p)
df=df[(df['scenario']=='co2_spike') & (df['safety_gate']==True)].copy()
models=df['model'].tolist()
labels=['False alarms / empty','Unsafe false-empty / occupied','Abstention']
columns=['false_alarm_per_empty','unsafe_false_empty_per_occupied','abstain_rate']
x=np.arange(len(models)); fig,ax=plt.subplots(figsize=(10,5))
for i,(label,col) in enumerate(zip(labels,columns)):
    ax.bar(x+(i-1)*.24,df[col],width=.23,label=label)
ax.set_xticks(x,models,rotation=20,ha='right')
ax.set_ylabel('Share of all relevant labels or samples')
ax.set_ylim(0,1)
ax.legend()
ax.set_title('Simulated CO2 spike: selective-prediction safety evaluation')
fig.tight_layout()
fig.savefig('results/robustness_comparison.png',dpi=180)
print('Saved results/robustness_comparison.png')
