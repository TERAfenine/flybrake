"""Local code verification with ARTIFICIAL GRAPH ONLY. NOT BANC experiment."""
import numpy as np
import torch
from src.connectome import rewire_preserving_degrees
from src.model import SignedConnectome
rng=np.random.default_rng(4)
n=48
M=(rng.random((n,n))<.15).astype('float32')
np.fill_diagonal(M,0.)
S=np.where(rng.random(n)<.25,-1.,1.).astype('float32')
R=rewire_preserving_degrees(M)
assert R.sum()==M.sum()
model=SignedConnectome(M,S)
x=torch.randn(32,8)
logits=model(x)
assert logits.shape==(32,)
loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.randint(0,2,(32,)).float())
loss.backward()
assert all(p.grad is not None for p in model.parameters())
print('PASS: model forward/backprop and matched-degree rewiring. Artificial graph only.')
