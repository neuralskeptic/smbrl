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
