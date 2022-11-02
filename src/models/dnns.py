import torch
import torch.nn as nn
import torch.nn.functional as functional

ACTIVATIONS = {
    "leakyrelu": functional.leaky_relu,
    "softplus": functional.softplus,
    "sigmoid": functional.sigmoid,
}


class DNN3(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, lr, device="cpu", activation="softplus"):
        super().__init__()
        self.l1 = torch.nn.Linear(d_in, d_hidden)
        self.l2 = torch.nn.Linear(d_hidden, d_hidden)
        self.l3 = torch.nn.Linear(d_hidden, d_out)
        self.act = ACTIVATIONS[activation]

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.l3(self.act(self.l2(self.act(self.l1(x)))))
