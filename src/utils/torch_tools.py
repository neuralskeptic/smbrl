from abc import ABC, abstractmethod

import torch

float32_eps = torch.Tensor([torch.finfo(torch.float32).tiny])


def vec(x):
    return x.T.reshape(-1)


def autograd_tensor(x):
    """
    Same as torch.Tensor(x, requires_grad=True), but does not cause warnings.
    detach first and then clonw removes the clone from the computation graph
    """
    return x.detach().clone().requires_grad_(True)


def map_cpu(iterable):
    return map(lambda x: x.to("cpu"), iterable)


class CudaAble(ABC):
    @abstractmethod
    def to(self, device):
        raise NotImplementedError

    def cpu(self):
        return self.to("cpu")


class Stateless:
    def to(self, device):
        return self

    def cpu(self):
        return self


class NoTraining:
    def eval(self):
        return self

    def train(self):
        return self
