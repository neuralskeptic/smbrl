import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.utils.whitening import data_whitening

ACTIVATIONS = {
    "leakyrelu": functional.leaky_relu,
    "softplus": functional.softplus,
    "sigmoid": functional.sigmoid,
}


@data_whitening
class DNN3(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, activation="softplus"):
        super().__init__()
        self.dim_x = d_in  # needed for whitening
        self.dim_y = d_out  # needed for whitening
        self.l1 = torch.nn.Linear(d_in, d_hidden)
        self.l2 = torch.nn.Linear(d_hidden, d_hidden)
        self.l3 = torch.nn.Linear(d_hidden, d_out)
        self.act = ACTIVATIONS[activation]

    def forward(self, x):
        return self.l3(self.act(self.l2(self.act(self.l1(x)))))
