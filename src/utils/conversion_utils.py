import pandas as pd
import torch
from mushroom_rl.utils.dataset import parse_dataset


def np2torch(x):
    return torch.from_numpy(x).float()


def df2torch(df):
    return torch.tensor(df.values)


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


def qube_rollout2df(data):
    s, a, r, ss, absorb, last = parse_dataset(data)
    N = len(a)
    df = pd.DataFrame()
    df[["s0", "s1", "s2", "s3", "s4", "s5"]] = s.reshape(N, -1)
    df[["a"]] = a.reshape(N, -1)
    df[["r"]] = r.reshape(N, -1)
    df[["ss0", "ss1", "ss2", "ss3", "ss4", "ss5"]] = ss.reshape(N, -1)
    df[["absorb"]] = absorb.reshape(N, -1)
    df[["last"]] = last.reshape(N, -1)
    return df


def map_cpu(iterable):
    return map(lambda x: x.to("cpu"), iterable)
