"""Run paired seeds without overwriting prior runs."""
from pathlib import Path
import argparse
import subprocess
import sys
import pandas as pd
from src.evaluation import summarize, METRICS

def main():
    base=Path(__file__).resolve().parent
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=base/'data')
    p.add_argument('--outdir',type=Path,default=base/'results/revised_5seeds')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    p.add_argument('--epochs',type=int,default=18)
    p.add_argument('--extended-controls',action='store_true')
    a=p.parse_args()
    if len(set(a.seeds)) != len(a.seeds): p.error('Duplicate seeds')
    if a.outdir.exists() and any(a.outdir.iterdir()): p.error('Use a new output directory')
    a.outdir.mkdir(parents=True,exist_ok=True)
    collected={name:[] for name in ['metrics','coverage_metrics','validation_threshold_metrics']}
    for seed in a.seeds:
        output=a.outdir/f'seed{seed}'
        cmd=[sys.executable,str(base/'train.py'),'--train',str(a.data_dir/'datatraining.txt'),
             '--test',str(a.data_dir/'datatest2.txt'),'--graph',str(a.data_dir/'banc_graph.npz'),
             '--seed',str(seed),'--epochs',str(a.epochs),'--outdir',str(output)]
        if a.extended_controls: cmd.append('--extended-controls')
        subprocess.run(cmd,check=True)
        for name in collected:
            frame=pd.read_csv(output/f'{name}.csv');frame['seed']=seed
            collected[name].append(frame)
    for name,frames in collected.items():
        combined=pd.concat(frames,ignore_index=True)
        combined.to_csv(a.outdir/f'{name}_5seeds.csv',index=False,encoding='utf-8-sig')
        keys=['model','scenario','safety_gate']
        valid=combined
        if name!='metrics': keys+=['method','target_coverage']
        if name=='coverage_metrics': valid=combined[combined.feasible].copy()
        summary=summarize(valid,keys,METRICS,expected_seeds=a.seeds)
        filename='summary_5seeds.csv' if name=='metrics' else f'{name}_summary.csv'
        summary.to_csv(a.outdir/filename,index=False,encoding='utf-8-sig')
    print(f'Completed {len(a.seeds)} seeds: {a.outdir}')

if __name__=='__main__': main()
