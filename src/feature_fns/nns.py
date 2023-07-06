import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.utils.parametrizations import spectral_norm

ACTIVATIONS = {"leakyrelu": functional.leaky_relu}


class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_in, layer_spec, d_out, activation="leakyrelu"):
        assert activation in ACTIVATIONS
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, layer_spec[0])])
        for d_h1, d_h2 in zip(layer_spec, layer_spec[1:]):  # pairwise
            self.layers.append(nn.Linear(d_h1, d_h2))
        self.layers.append(nn.Linear(layer_spec[-1], d_out))
        self.act = ACTIVATIONS[activation]

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.act(l(x))
        return self.layers[-1](x)  # no output activation!


class ResidualNetwork(MultiLayerPerceptron):
    def forward(self, x):
        for l in self.layers:
            lx = l(x)
            x = lx + self.act(lx)  # skip activation
        return x


class SpectralNormResidualNetwork(nn.Module):
    def __init__(self, d_in, layer_spec, d_out, activation="leakyrelu"):
        assert activation in ACTIVATIONS
        super().__init__()
        # Note: spectral_norm sets norm to 1, do we need it to be configurable?
        self.layers = nn.ModuleList([spectral_norm(nn.Linear(d_in, layer_spec[0]))])
        for d_h1, d_h2 in zip(layer_spec, layer_spec[1:]):  # pairwise
            self.layers.append(spectral_norm(nn.Linear(d_h1, d_h2)))
        self.act = ACTIVATIONS[activation]
        self.register_buffer("W", torch.randn(layer_spec[-1], d_out // 2))

    def forward(self, x):
        for l in self.layers:
            lx = l(x)
            x = lx + self.act(lx)  # skip activation
        return torch.cat([torch.cos(x @ self.W), torch.sin(x @ self.W)], -1)


class MLP_RFF(nn.Module):
    def __init__(self, d_in, layer_spec, d_out, activation="leakyrelu"):
        assert activation in ACTIVATIONS
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, layer_spec[0])])
        for d_h1, d_h2 in zip(layer_spec, layer_spec[1:]):  # pairwise
            self.layers.append(nn.Linear(d_h1, d_h2))
        self.act = ACTIVATIONS[activation]
        self.W = nn.Parameter(torch.randn(layer_spec[-1], d_out // 2))  # learnable
        # self.register_buffer("W", torch.randn(layer_spec[-1], d_out // 2))  # constant

    def forward(self, x):
        for l in self.layers:
            x = self.act(l(x))
        return torch.cat([torch.cos(x @ self.W), torch.sin(x @ self.W)], -1)
