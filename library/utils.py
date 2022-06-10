from collections import UserDict

import numpy as np
import torch


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


def matrix_gaussian_kl(mean_1, cov_in_1, cov_out_1, mean_2, cov_in_2, cov_out_2):
    """
    TODO pytorch
    https://statproofbook.github.io/P/matn-kl
    """
    n, p = mean_1.shape
    diff = mean_2 - mean_1
    # MaVN covariances are scale invariant -- so we can safely normalize to prevent numerical issues.
    sf1 = p / np.trace(cov_out_1)
    sf2 = p / np.trace(cov_out_2)
    cov_out_1 = cov_out_1 * sf1
    cov_out_2 = cov_out_2 * sf2
    cov_in_1 = cov_in_1 / sf1
    cov_in_2 = cov_in_2 / sf2
    return (
        0.5
        * (
            n * np.log(max(EPS, np.linalg.det(cov_out_2)))
            - n * np.log(max(EPS, np.linalg.det(cov_out_1)))
            + p * np.log(max(EPS, np.linalg.det(cov_in_2)))
            - p * np.log(max(EPS, np.linalg.det(cov_in_1)))
            + np.trace(
                np.kron(
                    np.linalg.solve(cov_out_2, cov_out_1),
                    np.linalg.solve(cov_in_2, cov_in_1),
                )
            )
            + vec(diff).T
            @ vec(np.linalg.solve(cov_in_2, np.linalg.solve(cov_out_2, diff.T).T))
            - n * p
        ).item()
    )


def to_torch(x):
    return torch.from_numpy(x).float()


def autograd_tensor(x):
    """Same as torch.Tensor(x, requires_grad=True), but does not cause warnings."""
    return x.clone().detach().requires_grad_(True)


class structdict(UserDict):
    def __init__(self, a_dict: dict):
        super().__init__(a_dict)
        [setattr(self, k, v) for k, v in a_dict.items()]

    @classmethod
    def from_keys_values(cls, keys=[], values=[]):
        return cls(dict(zip(keys, values)))

    def __setattr__(self, name, val):
        if name == "data":
            super().__setattr__(name, val)
        else:
            self[name] = val
            self.__dict__[name] = val

    def __setitem__(self, key, item):
        super(structdict, self).__setitem__(key, item)
        self.__dict__[key] = item
