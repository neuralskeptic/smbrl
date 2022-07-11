import numpy as np
import torch

from src.utils.vec import vec

EPS = torch.finfo(torch.float64).tiny


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
