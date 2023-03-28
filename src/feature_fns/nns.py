import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.utils.parametrizations import spectral_norm

ACTIVATIONS = {"leakyrelu": functional.leaky_relu}


class TwoLayerNormalizedResidualNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, activation="leakyrelu"):
        assert activation in ACTIVATIONS
        super().__init__()
        # Note: spectral_norm sets norm to 1, do we need it to be configurable?
        self.l1 = spectral_norm(nn.Linear(d_in, d_hidden))
        self.l2 = spectral_norm(nn.Linear(d_hidden, d_hidden))
        self.act = ACTIVATIONS[activation]
        self.register_buffer("W", torch.randn(d_hidden, d_out // 2))

    def forward(self, x):
        l1 = self.l1(x)
        l2 = l1 + self.act(l1)
        l3 = l2 + self.act(l2)
        return torch.cat([torch.cos(l3 @ self.W), torch.sin(l3 @ self.W)], -1)


class TwoLayerNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, activation="leakyrelu"):
        assert activation in ACTIVATIONS
        super().__init__()
        self.l1 = nn.Linear(d_in, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_out)
        self.act = ACTIVATIONS[activation]

    def forward(self, x):
        return self.act(self.l2(self.act(self.l1(x))))


class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_in, n_hidden_layers, d_hidden, d_out, activation="leakyrelu"):
        assert activation in ACTIVATIONS
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_hidden)])
        for i in range(n_hidden_layers - 1):
            self.layers.append(nn.Linear(d_hidden, d_hidden))
        self.layers.append(nn.Linear(d_hidden, d_out))
        self.act = ACTIVATIONS[activation]

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.act(l(x))
        return self.layers[-1](x)  # no output activation!


class ResidualNetwork(MultiLayerPerceptron):
    def forward(self, x):
        lx1 = self.layers[0](x)
        x = lx1 + self.act(lx1)  # d_in != d_hidden (map, then skip)
        for l in self.layers[1:-1]:  # d_hidden == d_hidden (skip directly)
            lx = l(x)
            x = x + self.act(lx)
        lxn = self.layers[-1](x)
        x = lxn + self.act(lxn)  # d_hidden != d_out (map, then skip)
        return x


class SpectralNormResidualNetwork(nn.Module):
    def __init__(self, d_in, n_hidden_layers, d_hidden, d_out, activation="leakyrelu"):
        assert activation in ACTIVATIONS
        super().__init__()
        # Note: spectral_norm sets norm to 1, do we need it to be configurable?
        self.layers = nn.ModuleList([spectral_norm(nn.Linear(d_in, d_hidden))])
        for i in range(n_hidden_layers - 1):
            self.layers.append(spectral_norm(nn.Linear(d_hidden, d_hidden)))
        self.act = ACTIVATIONS[activation]
        self.register_buffer("W", torch.randn(d_hidden, d_out // 2))

    def forward(self, x):
        lx1 = self.layers[0](x)
        x = lx1 + self.act(lx1)  # d_in != d_hidden (map, then skip)
        for l in self.layers[1:]:  # d_hidden == d_hidden (skip directly)
            lx = l(x)
            x = x + self.act(lx)
        return torch.cat([torch.cos(x @ self.W), torch.sin(x @ self.W)], -1)


class MLP_RFF(nn.Module):
    def __init__(self, d_in, layer_spec, d_out, activation="leakyrelu"):
        assert activation in ACTIVATIONS
        super().__init__()
        self.layers = nn.ModuleList()
        _d_first = d_in
        for d_hidden in layer_spec:
            self.layers.append(nn.Linear(_d_first, d_hidden))
            _d_first = d_hidden
        self.act = ACTIVATIONS[activation]
        self.W = nn.Parameter(torch.randn(_d_first, d_out // 2))  # learnable
        # self.register_buffer("W", torch.randn(_d_first, d_out // 2))  # constant

    def forward(self, x):
        for l in self.layers:
            x = self.act(l(x))
        return torch.cat([torch.cos(x @ self.W), torch.sin(x @ self.W)], -1)
