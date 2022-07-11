import torch


def to_torch(x):
    return torch.from_numpy(x).float()


def vec(x):
    """
    TODO pytorch
    https://stackoverflow.com/questions/25248290/most-elegant-implementation-of-matlabs-vec-function-in-numpy
    """
    shape = x.shape
    if len(shape) == 3:
        a, b, c = shape
        return x.reshape((a, b * c), order="F")
    else:
        return x.reshape((-1, 1), order="F")


def autograd_tensor(x):
    """Same as torch.Tensor(x, requires_grad=True), but does not cause warnings."""
    return x.clone().detach().requires_grad_(True)
