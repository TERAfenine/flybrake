import argparse
from pathlib import Path
import numpy as np
from src.connectome import build_graph, rewire_preserving_degrees
p=argparse.ArgumentParser(description='Extract genuine BANC v888 local network')
p.add_argument('--meta', required=True)
p.add_argument('--nt', required=True)
p.add_argument('--edges', required=True)
p.add_argument('--n', type=int, default=96)
p.add_argument('--out', default='data/banc_graph.npz')
a=p.parse_args()
g=build_graph(a.meta,a.nt,a.edges,n=a.n)
g['rewired_mask']=rewire_preserving_degrees(g['mask'])
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.out,**g)
print(f"Saved {a.out}: {len(g['ids'])} input neurons, {int(g['mask'].sum())} connections, {int((g['signs']<0).sum())} predicted GABA neurons")
