import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.utils.parametrizations import spectral_norm

ACTIVATIONS = {"leakyrelu": functional.leaky_relu}


class TwoLayerNormalizedResidualNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, activation="leakyrelu", device="cpu"):
        assert activation in ACTIVATIONS
        super().__init__()
        # Note: spectral_norm sets norm to 1, do we need it to be configurable?
        self.l1 = spectral_norm(nn.Linear(d_in, d_hidden))
        self.l2 = spectral_norm(nn.Linear(d_hidden, d_hidden))
        self.act = ACTIVATIONS[activation]
        self.W = torch.randn(d_hidden, d_out // 2).to(device)

    def forward(self, x):
        l1 = self.l1(x)
        l2 = l1 + self.act(l1)
        l3 = l2 + self.act(l2)
        return torch.cat([torch.cos(l3 @ self.W), torch.sin(l3 @ self.W)], 1)


class TwoLayerNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, activation="leakyrelu"):
        assert activation in ACTIVATIONS
        super().__init__()
        self.l1 = nn.Linear(d_in, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_out)
        self.act = ACTIVATIONS[activation]

    def forward(self, x):
        return self.act(self.l2(self.act(self.l1(x))))
