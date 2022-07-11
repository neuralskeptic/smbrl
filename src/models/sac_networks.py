from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetworkQfunction(nn.Module, ABC):
    def __init__(
        self, input_shape, output_shape, n_features, activation="relu", **kwargs
    ):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        if activation == "relu":
            self._activation = F.relu
        elif activation == "tanh":
            self._activation = torch.tanh
        else:
            raise NotImplementedError

        nn.init.xavier_uniform_(
            self._h1.weight, gain=nn.init.calculate_gain(activation)
        )
        nn.init.xavier_uniform_(
            self._h2.weight, gain=nn.init.calculate_gain(activation)
        )
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = self._activation(self._h1(state_action))
        features2 = self._activation(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class CriticNetworkVfunction(nn.Module, ABC):
    def __init__(
        self, input_shape, output_shape, n_features, activation="relu", **kwargs
    ):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        if activation == "relu":
            self._activation = F.relu
        elif activation == "tanh":
            self._activation = torch.tanh
        else:
            raise NotImplementedError

        nn.init.xavier_uniform_(
            self._h1.weight, gain=nn.init.calculate_gain(activation)
        )
        nn.init.xavier_uniform_(
            self._h2.weight, gain=nn.init.calculate_gain(activation)
        )
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, **kwargs):
        features1 = self._activation(self._h1(torch.squeeze(state, 1).float()))
        features2 = self._activation(self._h2(features1))
        v = self._h3(features2)

        return v


class ActorNetwork(nn.Module, ABC):
    def __init__(
        self, input_shape, output_shape, n_features, activation="relu", **kwargs
    ):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        if activation == "relu":
            self._activation = F.relu
        elif activation == "tanh":
            self._activation = torch.tanh
        else:
            raise NotImplementedError

        nn.init.xavier_uniform_(
            self._h1.weight, gain=nn.init.calculate_gain(activation)
        )
        nn.init.xavier_uniform_(
            self._h2.weight, gain=nn.init.calculate_gain(activation)
        )
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state):
        features1 = self._activation(self._h1(torch.squeeze(state, 1).float()))
        features2 = self._activation(self._h2(features1))
        a = self._h3(features2)

        return a
