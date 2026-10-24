"""Small recurrent network constrained by neuron-neuron connectivity."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class SignedConnectome(nn.Module):
    def __init__(self, mask, signs, input_dim=8, steps=4, constraint="signed"):
        super().__init__()
        n = len(signs)
        assert mask.shape == (n,n)
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))
        self.register_buffer('signs', torch.tensor(signs, dtype=torch.float32))
        self.register_buffer('degree', torch.tensor(np.maximum(mask.sum(axis=1),1),dtype=torch.float32))
        # Input/readout mappings are engineered, NOT from BANC.
        self.n_inputs = max(4, n//4)
        self.n_outputs = max(4, n//4)
        self.input_map = nn.Linear(input_dim, self.n_inputs)
        self.raw_weights = nn.Parameter(torch.randn(n,n)*.1)
        # Free-sign controls start at exactly the same effective recurrent
        # weights as signed controls; subsequent parameterization differs.
        if constraint == 'free':
            with torch.no_grad():
                self.raw_weights.copy_(F.softplus(self.raw_weights) * self.signs.view(1,-1))
        self.bias = nn.Parameter(torch.zeros(n))
        self.output_map = nn.Linear(self.n_outputs,1)
        self.steps = steps
        if constraint not in ("signed", "free"):
            raise ValueError(constraint)
        self.constraint = constraint
    def forward(self, x):
        n = len(self.signs)
        h = x.new_zeros((len(x),n))
        drive = x.new_zeros((len(x),n))
        drive[:,:self.n_inputs] = torch.tanh(self.input_map(x))
        if self.constraint == "free":
            W = self.raw_weights * self.mask
        else:
            W = F.softplus(self.raw_weights) * self.mask * self.signs.view(1,-1)
        W = .35 * W / self.degree.sqrt().view(-1,1)
        for _ in range(self.steps):
            h = torch.tanh(drive + h @ W.T + self.bias)
        return self.output_map(h[:,-self.n_outputs:]).squeeze(-1)
