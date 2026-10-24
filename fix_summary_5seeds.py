"""Repair summary while preserving the existing CSV in a timestamped backup."""
from pathlib import Path
from datetime import datetime
import argparse
import shutil
import pandas as pd
from src.evaluation import summarize

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    a=p.parse_args()
    df=pd.read_csv(a.results/'metrics_5seeds.csv')
    summary=summarize(df,expected_seeds=[42,43,44,45,46])
    out=a.results/'summary_5seeds.csv'
    if out.exists():
        shutil.copy2(out,out.with_name('summary_5seeds.backup_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv'))
    summary.to_csv(out,index=False,encoding='utf-8-sig')
    print(summary.to_string(index=False))
if __name__=='__main__': main()
