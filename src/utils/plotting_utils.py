from typing import Sequence

import einops
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.i2c.distributions import MultivariateGaussian


def rollout_plot(xs, us, xvars=None, uvars=None, u_max=None, **fig_kwargs):
    dim_x = xs.shape[-1]
    dim_u = us.shape[-1]
    fig, axs = plt.subplots(dim_x + dim_u, **fig_kwargs)
    xs = einops.rearrange(xs, "t ... x -> t (...) x")
    us = einops.rearrange(us, "t ... u -> t (...) u")
    if xvars is not None:
        xvars = einops.rearrange(xvars, "t ... x -> t (...) x")
    if uvars is not None:
        uvars = einops.rearrange(uvars, "t ... u -> t (...) u")
    n_batches = xs.shape[-2]
    colors = plt.cm.brg(np.linspace(0, 1, n_batches))
    for i in range(dim_x):
        for ib, c in zip(range(n_batches), colors):
            if xvars is not None:
                plot_gp(axs[i], xs[:, ib, i], xvars[:, ib, i], color=c)
            else:
                axs[i].plot(xs[:, ib, i], color=c)
        axs[i].set_ylabel(f"x{i}")
        axs[i].grid(True)
    for i in range(dim_u):
        j = dim_x + i
        for ib, c in zip(range(n_batches), colors):
            if uvars is not None:
                plot_gp(axs[j], us[:, ib, i], uvars[:, ib, i], color=c)
            else:
                axs[j].plot(us[:, ib, i], color=c)
        if u_max is not None:
            axs[j].plot(u_max * torch.ones_like(us[:, i]), "k--")
            axs[j].plot(-u_max * torch.ones_like(us[:, i]), "k--")
        axs[j].set_ylabel(f"u{i}")
        axs[j].grid(True)
    axs[-1].set_xlabel("timestep")


def plot_gp(axis, mean, variance, color="b"):
    axis.plot(mean, color=color)
    sqrt = torch.sqrt(variance)
    upper, lower = mean + sqrt, mean - sqrt
    axis.fill_between(
        range(mean.shape[0]), upper, lower, where=upper >= lower, color=color, alpha=0.2
    )


def plot_mvn(
    axis,
    dists: Sequence[MultivariateGaussian],
    dim=slice(None),
    batch=slice(None),
    color=None,
):
    means = torch.stack([d.mean[..., batch, dim] for d in dists])
    variances = torch.stack([d.covariance[..., batch, dim, dim].sqrt() for d in dists])
    plot_gp(axis, means, variances, color=color)


def plot_trajectory_distribution(
    list_of_distributions: Sequence[MultivariateGaussian], title=""
):
    means = torch.stack(tuple(d.mean for d in list_of_distributions), dim=0)
    covariances = torch.stack(tuple(d.covariance for d in list_of_distributions), dim=0)
    # batch_shape = means.shape[1:-1]  # store batch dimensions
    # flatten batch dimensions (for plotting)
    means = einops.rearrange(means, "t ... x -> t (...) x")
    covariances = einops.rearrange(covariances, "t ... x1 x2 -> t (...) x1 x2")
    n_plots = means.shape[-1]
    n_batches = means.shape[1]
    colors = plt.cm.brg(torch.linspace(0, 1, n_batches))
    fig, axs = plt.subplots(n_plots)
    axs[0].set_title(title)
    for i, ax in enumerate(axs):
        for b, c in zip(range(n_batches), colors):
            plot_gp(ax, means[:, b, i], covariances[:, b, i, i], color=c)
    return fig, axs


def plot_gp_2(axis, x, mu, var, color="b", alpha=0.3, label=""):
    axis.plot(x, mu, "-", color=color, label=label)
    for n_std in range(1, 3):
        std = n_std * torch.sqrt(var.squeeze())
        mu = mu.squeeze()
        upper, lower = mu + std, mu - std
        axis.fill_between(
            x.squeeze(),
            upper.squeeze(),
            lower.squeeze(),
            where=upper > lower,
            color=color,
            alpha=alpha,
        )
