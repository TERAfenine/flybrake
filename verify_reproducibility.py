"""Repeat the revised protocol in the caller's Python environment; never overwrite."""
import argparse
from pathlib import Path
import subprocess,sys,json
import pandas as pd
import torch

def main():
    base=Path(__file__).resolve().parent
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=base/'data')
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=18)
    a=p.parse_args()
    if a.outdir.exists():p.error('Choose a new output directory')
    a.outdir.mkdir(parents=True)
    report=[]
    for repeat in ['a','b']:
        subprocess.run([sys.executable,str(base/'run_5_seeds.py'),'--data-dir',str(a.data_dir),
            '--outdir',str(a.outdir/repeat),'--epochs',str(a.epochs),'--extended-controls'],check=True)
    for seed in range(42,47):
        first=a.outdir/'a'/f'seed{seed}';second=a.outdir/'b'/f'seed{seed}'
        tables={name:pd.read_csv(first/name).equals(pd.read_csv(second/name)) for name in
            ['metrics.csv','coverage_metrics.csv','validation_threshold_metrics.csv']}
        tensors=True
        for path in first.glob('*.pt'):
            x=torch.load(path,map_location='cpu',weights_only=True)
            y=torch.load(second/path.name,map_location='cpu',weights_only=True)
            tensors &= all(torch.equal(x[k],y[k]) for k in x)
        report.append(dict(seed=seed,tables_equal=tables,checkpoint_tensors_equal=tensors))
    (a.outdir/'reproducibility.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not all(all(r['tables_equal'].values()) and r['checkpoint_tensors_equal'] for r in report):
        raise SystemExit('Reproducibility mismatch; inspect results')
if __name__=='__main__':main()
