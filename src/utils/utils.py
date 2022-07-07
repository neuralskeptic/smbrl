from collections import UserDict

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

EPS = torch.finfo(torch.float64).tiny


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


def set_random_seed(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


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


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar
        pbar.update(0)

    def _on_step(self):
        # breakpoint()
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        # self._pbar.update(0)
        # self._pbar.update(1)
        self._pbar.refresh()
        # breakpoint()
        # breakpoint()
        # if (self._pbar.n % 10 == 0):
        # print(self._pbar.n)
        # breakpoint()


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(
        self,
    ):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps, leave=True)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


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
