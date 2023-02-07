import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from experiment_launcher import run_experiment
from experiment_launcher.utils import save_args
from mushroom_rl.core.logger.logger import Logger
from pytorch_minimize.optim import MinimizeWrapper
from torch.autograd.functional import hessian, jacobian
from tqdm import trange

from src.datasets.mutable_buffer_datasets import ReplayBuffer
from src.models.dnns import DNN3
from src.models.linear_bayesian_models import (
    NeuralLinearModel,
    SpectralNormalizedNeuralGaussianProcess,
)
from src.utils.seeds import fix_random_seed
from src.utils.time_utils import timestamp


@dataclass
class Distribution:
    """No idea what this should be yet!"""


@dataclass
class MultivariateGaussian(Distribution):
    """Multivariate Gaussian object for Markovian settings, i.e. p(y) = \int p(x,y) dx = \int p(y | x) p(x) dx."""

    mean: torch.Tensor
    covariance: torch.Tensor
    cross_covariance: torch.Tensor
    previous_mean: torch.Tensor
    previous_covariance: torch.Tensor

    def marginalize(self, indices):
        """Marginalize out indices"""
        return MultivariateGaussian(
            self.mean[indices], self.covariance[indices, indices], None, None, None
        )

    def full_joint(self, reverse=True):
        """Convert Markovian structure into the joint multivariate Gaussian and forget history."""
        if reverse:
            mean = torch.cat((self.previous_mean, self.mean), axis=0)
            covariance = torch.cat(
                (
                    torch.cat(
                        (self.previous_covariance, self.cross_covariance.T), axis=1
                    ),
                    (torch.cat((self.cross_covariance, self.covariance), axis=1)),
                ),
                axis=0,
            )
        else:
            mean = torch.cat((self.mean, self.previous_mean), axis=0)
            covariance = torch.cat(
                (
                    torch.cat((self.covariance, self.cross_covariance), axis=1),
                    (
                        torch.cat(
                            (self.cross_covariance.T, self.previous_covariance), axis=1
                        )
                    ),
                ),
                axis=0,
            )
        return MultivariateGaussian(mean, covariance, None, None, None)

    def get_previous_loglikelihood(self):
        dim = self.previous_mean.shape[0]
        previous_precision = torch.linalg.inv(self.previous_covariance)

        def llh(x):
            diff = x - self.previous_mean
            return -0.5 * dim * (
                math.log(2.0 * math.pi) - torch.logdet(previous_precision)
            ) - 0.5 * torch.einsum("bi,ij,bj->b", diff, previous_precision, diff)

        return llh

    def reverse(self):
        return MultivariateGaussian(
            self.previous_mean,
            self.previous_covariance,
            self.cross_covariance.T,
            self.mean,
            self.covariance,
        )

    def sample(self):
        return torch.distributions.MultivariateNormal(
            self.mean, self.covariance
        ).sample()

    def to(self, device):
        self.mean = self.mean.to(device)
        self.covariance = self.covariance.to(device)
        if self.cross_covariance is not None:
            self.cross_covariance = self.cross_covariance.to(device)
            self.previous_mean = self.previous_mean.to(device)
            self.previous_covariance = self.previous_covariance.to(device)
        return self


class LinearizationInference(object):
    def __call__(self, function: Callable, distribution: MultivariateGaussian):
        function_ = lambda x: function(x)[0]
        jac = jacobian(function, distribution.mean)
        mean = function(distribution.mean[None, :])[0][0, :]
        covariance = jac @ distribution.covariance @ jac.T
        return MultivariateGaussian(
            mean,
            covariance,
            jac @ distribution.covariance,
            distribution.mean,
            distribution.covariance,
        )

    def mean(
        self, function: Callable, distribution: MultivariateGaussian
    ) -> torch.float32:
        # TODO (joemwatson) only want to use this for R^d->R functions, but what about hessian usage generally?
        hess = hessian(function, distribution.mean[None, :])[0, :, 0, :]
        return function(distribution.mean[None, :]) + torch.trace(
            distribution.covariance @ hess
        )


class LinearizationInnovation(LinearizationInference):
    def __call__(
        self,
        function: Callable,
        inverse_temperature: float,
        distribution: MultivariateGaussian,
    ) -> MultivariateGaussian:
        jac = jacobian(function, distribution.mean[None, :])[0, 0, :]
        hess = hessian(function, distribution.mean[None, :])[0, :, 0, :]
        if (torch.linalg.eigvalsh(hess) < 0.0).any():
            hess = jac[:, None] @ jac[None, :]
        # import pdb; pdb.set_trace()
        precision = inverse_temperature * hess + torch.linalg.inv(
            distribution.covariance
        )
        covariance = torch.linalg.inv(precision)
        mean = distribution.mean - covariance @ (inverse_temperature * jac)
        if (torch.linalg.det(covariance) < 0.0).any():
            import pdb

            pdb.set_trace()
        return MultivariateGaussian(mean, covariance, None, None, None)


@dataclass
class CubatureQuadrature(object):
    """Implements sparse spherical cubature rule (\alpha=1) and the 'unscented' heuristic weights."""

    alpha: float
    beta: float
    kappa: float

    @staticmethod
    def pts(dim: int) -> torch.Tensor:
        return torch.cat(
            (torch.zeros((1, dim)), torch.eye(dim), -torch.eye(dim)), axis=0
        )

    def weights(self, dim):
        assert self.alpha > 0
        lam = self.alpha**2 * (dim + self.kappa) - dim
        sf = math.sqrt(dim + lam)
        w0_sig_extra = 1 - self.alpha**2 + self.beta
        weights_mu = (1 / (2 * (dim + lam))) * torch.ones((1 + 2 * dim,))
        weights_mu[0] = 2 * lam * weights_mu[0]
        weights_sig = weights_mu.clone()
        weights_sig[0] += w0_sig_extra
        print(weights_mu)
        return sf, weights_mu, weights_sig


@dataclass
class GaussHermiteQuadrature(object):
    degree: int

    def __post_init__(self):
        assert self.degree >= 1
        import numpy as np

        self.gh_pts, self.gh_weights = map(
            torch.from_numpy, np.polynomial.hermite.hermgauss(self.degree)
        )

    def pts(self, dim):
        grid = torch.meshgrid(*(self.gh_pts,) * dim, indexing="xy")
        return torch.vstack(tuple(map(torch.ravel, grid))).T

    def weights(self, dim):
        grid = torch.meshgrid(*(self.gh_weights,) * dim, indexing="xy")
        w = torch.vstack(tuple(map(torch.ravel, grid))).T
        w = torch.prod(w, axis=1) / (math.pi ** (dim / 2))
        print(w)
        return math.sqrt(2), w, w


class QuadratureInference(object):
    def __init__(self, dimension: int, params: CubatureQuadrature):
        self.dim = dimension
        self.base_pts = params.pts(dimension)
        self.sf, self.weights_mu, self.weights_sig = params.weights(dimension)
        self.n_points = self.base_pts.shape[0]

    def get_x_pts(self, mu_x, sigma_x):
        try:
            sqrt = torch.linalg.cholesky(sigma_x)
        except torch.linalg.LinAlgError:
            print(
                f"singular covariance: {torch.linalg.eigvalsh(sigma_x)}, state: {mu_x}"
            )
            sqrt = torch.linalg.cholesky(torch.diag(torch.diag(sigma_x)))
        scale = self.sf * sqrt
        try:
            mu_x[None, :] + self.base_pts @ scale.T
        except:
            breakpoint()
        return mu_x[None, :] + self.base_pts @ scale.T

    def __call__(
        self, function: Callable, distribution: MultivariateGaussian
    ) -> MultivariateGaussian:
        assert isinstance(distribution, MultivariateGaussian)
        x_pts = self.get_x_pts(distribution.mean, distribution.covariance)
        self.y_pts, self.m_y, self.sig_y, self.sig_xy = self.forward_pts(
            function, distribution.mean, x_pts
        )
        return MultivariateGaussian(
            self.m_y,
            self.sig_y,
            self.sig_xy.T,
            distribution.mean,
            distribution.covariance,
        )

    def forward_pts(
        self, f: Callable, mu_x: torch.Tensor, x_pts: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        y_pts, x_pts = f(x_pts)  # return updated arg, because env clips u
        mu_y = torch.einsum("b,bi->i", self.weights_mu, y_pts)
        diff_x = x_pts - mu_x[None, :]
        diff_y = y_pts - mu_y[None, :]
        sigma_y = torch.einsum(
            "b,bi,bj->ij", self.weights_sig, diff_y, diff_y
        ) + 1e-7 * torch.eye(y_pts.shape[1]).to(
            mu_x.device
        )  # regularization
        sigma_xy = torch.einsum("b,bi,bj->ij", self.weights_sig, diff_x, diff_y)
        return y_pts, mu_y, sigma_y, sigma_xy

    def mean(
        self, function: Callable, distribution: MultivariateGaussian
    ) -> torch.float32:
        return torch.einsum(
            "b,b->",
            self.weights_mu,
            function(self.get_x_pts(distribution.mean, distribution.covariance))[0],
        )


class QuadratureGaussianInference(QuadratureInference):
    """For a function that returns a mean and covariance."""

    def forward_pts(
        self, f: Callable, mu_x: torch.Tensor, x_pts: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        # breakpoint()
        y_pts, sigma_pts, x_pts = f(x_pts)
        mu_y = torch.einsum("b,bi->i", self.weights_mu, y_pts)
        diff_x = x_pts - mu_x[None, :]
        diff_y = y_pts - mu_y[None, :]
        sigma_y = torch.einsum(
            "b,bi,bj->ij", self.weights_sig, diff_y, diff_y
        ) + torch.einsum("b,bij->ij", self.weights_mu, sigma_pts)
        sigma_xy = torch.einsum("b,bi,bj->ij", self.weights_sig, diff_x, diff_y)
        return y_pts, mu_y, sigma_y, sigma_xy


class QuadratureImportanceSamplingInnovation(QuadratureInference):
    def __call__(
        self,
        function: Callable,
        inverse_temperature: float,
        distribution: MultivariateGaussian,
    ) -> MultivariateGaussian:
        assert isinstance(distribution, MultivariateGaussian)
        x_pts = self.get_x_pts(distribution.mean, distribution.covariance)
        f_pts, self.x_pts = function(x_pts)
        log_weights = -inverse_temperature * f_pts + torch.log(self.weights_mu)
        log_weights -= torch.logsumexp(log_weights, dim=0)
        weights = torch.exp(log_weights)
        mu_x = torch.einsum("b, bi -> i", weights, self.x_pts)
        diff = self.x_pts - mu_x[None, :]
        sigma_x = torch.einsum("b, bi, bj -> ij", weights, diff, diff)
        sigma_x = 0.5 * (sigma_x + sigma_x.T)
        return MultivariateGaussian(mu_x, sigma_x, None, None, None)


def linear_gaussian_smoothing(
    current_prior: MultivariateGaussian,
    predicted_prior: MultivariateGaussian,
    future_posterior: MultivariateGaussian,
) -> MultivariateGaussian:
    """.
    See https://vismod.media.mit.edu/tech-reports/TR-531.pdf , Section 3.2
    """
    J = torch.linalg.solve(
        predicted_prior.covariance, predicted_prior.cross_covariance
    ).T
    mean = current_prior.mean + J @ (future_posterior.mean - predicted_prior.mean)
    covariance = (
        current_prior.covariance
        + J @ (future_posterior.covariance - predicted_prior.covariance) @ J.T
    )
    if covariance.det() <= 0:  # TODO debug
        # covariance = nearest_spd(covariance)
        breakpoint()
    return MultivariateGaussian(
        mean,
        covariance,
        J @ future_posterior.covariance,
        future_posterior.mean,
        future_posterior.covariance,
    )


class Policy(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, t: int) -> torch.Tensor:
        pass

    @abstractmethod
    def update_from_distribution(self, distribution, *args, **kwargs):
        pass


class TimeVaryingLinearGaussian(Policy):
    def __init__(
        self,
        horizon: int,
        dim_x: int,
        dim_u: int,
        action_covariance: torch.Tensor,
    ):
        assert action_covariance.shape == (dim_u, dim_u)
        self.horizon, self.dim_x, self.dim_u = horizon, dim_x, dim_u
        self.k_actual = torch.tile(
            1e-3 * torch.sin(torch.linspace(0, 20 * torch.pi, horizon))[:, None],
            (1, dim_u),
        )
        self.K_actual = torch.zeros((horizon, dim_u, dim_x))
        self.k_opt, self.K_opt = self.k_actual.clone(), self.K_actual.clone()
        self.sigma = torch.tile(action_covariance, (horizon, 1, 1))
        self.chol = torch.tile(
            torch.linalg.cholesky(action_covariance), (horizon, 1, 1)
        )

    def actual(self):
        from copy import deepcopy

        copy = deepcopy(self)  # crashes: only leaf tensors deepcopy-able
        # copy = TimeVaryingLinearGaussian(
        #     self.horizon,
        #     self.dim_x,
        #     self.dim_u,
        #     self.sigma[0, :, :].detach(),
        # )
        copy.k_opt = self.k_actual.detach()
        copy.K_opt = self.K_actual.detach()
        # TODO skip k/K_actual ?
        return copy

    def __call__(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Assumes batch input."""
        return (
            self.k_opt[t, :] + x @ self.K_opt[t, :, :].T,
            torch.tile(self.sigma[t, :, :], (x.shape[0], 1, 1)),
            x,
        )

    def predict(self, x: torch.Tensor, t: int) -> torch.Tensor:
        return self.k_actual[t, :] + x @ self.K_actual[t, :, :].T

    def sample(self, t: int, x: torch.Tensor) -> torch.Tensor:
        mu = self.k[t, :] + self.K[t, :, :] @ x
        eps = torch.randn(self.dim_u)
        return mu + self.chol[t, :, :] @ eps

    def update_from_distribution(self, distribution, *args, **kwargs):
        assert len(distribution) == self.horizon
        if isinstance(distribution[0], MultivariateGaussian):
            for t, dist in enumerate(distribution):
                x, u = dist.marginalize(slice(0, self.dim_x)), dist.marginalize(
                    slice(self.dim_x, self.dim_x + self.dim_u)
                )
                K = torch.linalg.solve(
                    x.covariance, dist.covariance[: self.dim_x, self.dim_x :]
                ).T
                self.K_actual[t, ...] = K
                self.k_actual[t, ...] = u.mean - self.K_actual[t, ...] @ x.mean
                self.k_opt[t, ...] = u.mean - self.K_opt[t, ...] @ x.mean
                self.sigma[t, ...] = u.covariance
                if self.sigma[t, ...].det() <= 0:
                    # self.sigma[t, ...] = nearest_spd(self.sigma[t, ...])
                    breakpoint()
                try:  # TODO debug
                    torch.linalg.cholesky(self.sigma[t, ...])
                except:
                    breakpoint()
                self.chol[t, ...] = torch.linalg.cholesky(self.sigma[t, ...])
        else:
            raise ValueError("Cannot update from this distribution!")

    # def to(self, device):
    #     self.k_actual = self.k_actual.to(device)
    #     self.K_actual = self.K_actual.to(device)
    #     self.k_opt = self.k_opt.to(device)
    #     self.K_opt = self.K_opt.to(device)
    #     self.sigma = self.sigma.to(device)
    #     self.chol = self.chol.to(device)
    #     return self


class Pendulum(object):

    dim_x = 2
    dim_u = 1
    dim_xu = 3
    u_mx = 2.0

    def __call__(self, xu: torch.Tensor) -> torch.Tensor:
        dt = 0.05
        m = 1.0
        l = 1.0
        d = 1e-2  # damping
        g = 9.80665
        x, u = xu[:, :2], xu[:, 2]
        u = torch.clip(u, -self.u_mx, self.u_mx)
        th_dot_dot = -3.0 * g / (2 * l) * torch.sin(x[:, 0] + torch.pi) - d * x[:, 1]
        th_dot_dot += 3.0 / (m * l**2) * u
        # variant from http://underactuated.mit.edu/pend.html
        # th_dot_dot = -g / l * torch.sin(x[:, 0] + torch.pi) - d / (m*l**2) * x[:, 1]
        # th_dot_dot += 1.0 / (m * l**2) * u
        x_dot = x[:, 1] + th_dot_dot * dt  # theta_dot
        x_pos = x[:, 0] + x_dot * dt  # theta

        x2 = torch.stack((x_pos, x_dot), dim=1)
        xu[:, 2] = u
        return x2, xu

    def run(self, initial_state, policy, horizon):
        xs = torch.zeros((horizon, self.dim_x))
        us = torch.zeros((horizon, self.dim_u))
        xxs = torch.zeros((horizon, self.dim_x))
        state = initial_state
        for t in range(horizon):
            action = policy.predict(state, t)
            xu = torch.cat((state, action))[None, :]
            xxs[t, :], _ = self.__call__(xu)
            xs[t, :] = state
            us[t, :] = action
            state = xxs[t, :]
        return xs, us, xxs

    def plot(self, xs, us):
        fig, axs = plt.subplots(self.dim_x + self.dim_u, figsize=(12, 9))
        for i in range(self.dim_x):
            axs[i].plot(xs[:, i])
            axs[i].set_ylabel(f"x{i}")
            axs[i].grid(True)
        for i in range(self.dim_u):
            j = self.dim_x + i
            axs[j].plot(us[:, i])
            axs[j].plot(self.u_mx * torch.ones_like(us[:, i]), "k--")
            axs[j].plot(-self.u_mx * torch.ones_like(us[:, i]), "k--")
            axs[j].set_ylabel(f"u{i}")
            axs[j].grid(True)
        axs[-1].set_xlabel("timestep")


def plot_gp(axis, mean, variance):
    axis.plot(mean, "b")
    sqrt = torch.sqrt(variance)
    upper, lower = mean + sqrt, mean - sqrt
    axis.fill_between(
        range(mean.shape[0]), upper, lower, where=upper >= lower, color="b", alpha=0.3
    )


def plot_trajectory_distribution(list_of_distributions, title=""):
    means = torch.stack(tuple(d.mean for d in list_of_distributions), dim=0)
    covariances = torch.stack(tuple(d.covariance for d in list_of_distributions), dim=0)
    n_plots = means.shape[1]
    fig, axs = plt.subplots(n_plots)
    axs[0].set_title(title)
    for i, ax in enumerate(axs):
        plot_gp(ax, means[:, i].cpu(), covariances[:, i, i].cpu())
    return fig, axs


class PseudoPosteriorSolver(object):

    metrics: Dict

    def __init__(
        self,
        dim_x: int,
        dim_u: int,
        env: Callable,
        cost: Callable,
        horizon: int,
        initial_state_distribution: Distribution,
        policy_prior: Policy,
        approximate_inference_dynamics,
        approximate_inference_policy,
        approximate_inference_cost,
        approximate_inference_smoothing,
        update_temperature_strategy,
    ):
        self.env = env
        self.dim_x, self.dim_u = dim_x, dim_u
        self.cost = cost
        self.horizon = horizon
        self.initial_state = initial_state_distribution
        self.policy_prior = policy_prior
        self.alpha = 0.0
        self.approximate_inference_dynamics = approximate_inference_dynamics
        self.approximate_inference_policy = approximate_inference_policy
        self.approximate_inference_cost = approximate_inference_cost
        self.approximate_inference_smoothing = approximate_inference_smoothing
        self.update_temperature_strategy = update_temperature_strategy

    def forward_pass(
        self,
        env: Callable,
        cost: Callable,
        policy: Policy,
        initial_state: Distribution,
        alpha: float,
    ) -> (List[Distribution], Distribution):
        state = initial_state
        state_action_dist = []
        future_state_dist = []
        for t in range(self.horizon):
            policy_ = partial(policy.__call__, t=t)
            action = self.approximate_inference_policy(policy_, state)
            state_action_policy = action.full_joint(reverse=True)
            state_action_cost = self.approximate_inference_cost(
                cost, alpha, state_action_policy
            )
            if state_action_cost.covariance.det() <= 0:  # TODO debug
                # state_action_cost.covariance = nearest_spd(state_action_cost.covariance)
                breakpoint()
            state_action_dist += [state_action_cost]
            # update state_action_cost with dynamics correlations
            next_state = self.approximate_inference_dynamics(env, state_action_cost)
            if next_state.covariance.det() <= 0:  # TODO debug
                # next_state.covariance = nearest_spd(next_state.covariance)
                breakpoint()
            future_state_dist += [next_state]
            state = next_state
        # breakpoint()  # TODO debug
        # npar = sum([p.numel() for p in env.parameters()])
        # sumpar = sum([p.sum().detach() for p in env.parameters()])
        # meanpar = sumpar/npar
        # plt.plot([dist.mean[0].detach() for dist in future_state_dist])
        # plt.plot([dist.mean[1].detach() for dist in future_state_dist])
        # plt.plot([dist.covariance[0,0].detach() for dist in future_state_dist])
        # plt.plot([dist.covariance[1,1].detach() for dist in future_state_dist])
        # plt.plot([forward_state_action_prior[i].marginalize(slice(self.dim_x, self.dim_x + self.dim_u))
        #           .covariance.item() for i in range(200)])
        return state_action_dist, future_state_dist, state

    def backward_pass(
        self,
        forward_state_action_distribution,
        predicted_state_distribution,
        terminal_state_distribution,
    ):
        dist = []
        future_state_posterior = terminal_state_distribution
        for current_state_action_prior, predicted_state_prior in zip(
            reversed(forward_state_action_distribution),
            reversed(predicted_state_distribution),
        ):
            # breakpoint()
            state_action_posterior = self.approximate_inference_smoothing(
                current_state_action_prior,
                predicted_state_prior,
                future_state_posterior,
            )
            if state_action_posterior.covariance.det() <= 0:  # TODO debug
                # state_action_posterior.covariance = nearest_spd(
                #     state_action_posterior.covariance
                # )
                breakpoint()
            future_state_posterior = state_action_posterior.marginalize(
                slice(0, self.dim_x)
            )  # pass the state posterior to previous timestep
            if future_state_posterior.covariance.det() <= 0:  # TODO debug
                # future_state_posterior.covariance = nearest_spd(
                #     future_state_posterior.covariance
                # )
                breakpoint()
            dist += [state_action_posterior]
        # breakpoint()
        # plt.plot([d.covariance[0,0].detach() for d in dist])
        # plt.plot([d.covariance[1,1].detach() for d in dist])
        return list(reversed(dist))

    def __call__(self, n_iteration: int, plot_posterior: bool = True):
        initial_alpha = 0.0
        self.init_metrics()
        policy = self.policy_prior
        (
            forward_state_action_prior,
            next_state_state_action_prior,
            _,
        ) = self.forward_pass(
            self.env, self.cost, policy, self.initial_state, alpha=initial_alpha
        )
        forward_distribution = [
            dist.reverse() for dist in next_state_state_action_prior
        ]
        self.alpha = self.update_temperature_strategy(
            self.cost,
            forward_distribution,
            forward_distribution,
            current_alpha=initial_alpha,
        )
        # plot_trajectory_distribution(forward_state_action_prior, f"init")
        self.compute_metrics(
            forward_state_action_prior, forward_state_action_prior, initial_alpha
        )
        for i in range(n_iteration):
            print(f"{i} {self.alpha:.2f}")
            (
                forward_state_action_prior,
                predicted_state_prior,
                terminal_state,
            ) = self.forward_pass(
                self.env, self.cost, policy, self.initial_state, self.alpha
            )
            # plot_trajectory_distribution(forward_state_action_prior, f"filter{i}")
            state_action_posterior = self.backward_pass(
                forward_state_action_prior, predicted_state_prior, terminal_state
            )
            state_action_policy, _, _ = self.forward_pass(
                self.env, self.cost, policy.actual(), self.initial_state, alpha=0.0
            )
            self.compute_metrics(
                state_action_posterior, state_action_policy, self.alpha
            )

            # plt.plot([forward_state_action_prior[i].marginalize(slice(self.dim_x, self.dim_x + self.dim_u)).covariance.item() for i in range(200)])
            # plt.plot([predicted_state_prior[i].marginalize(slice(self.dim_x, self.dim_x + self.dim_u)).covariance.item() for i in range(200)])
            # plt.plot([state_action_posterior[i].marginalize(slice(self.dim_x, self.dim_x + self.dim_u)).covariance.item() for i in range(200)])
            # plt.plot([state_action_policy[i].marginalize(slice(self.dim_x, self.dim_x + self.dim_u)).covariance.item() for i in range(200)])
            # breakpoint()  # state_action_posterior broken?
            policy.update_from_distribution(state_action_posterior, state_action_policy)
            self.alpha = self.update_temperature_strategy(
                self.cost,
                state_action_posterior,
                state_action_policy,
                current_alpha=initial_alpha,
            )
            # self.alpha = self.update_temperature_strategy(self.cost, [dist.reverse() for dist in next_state_state_action_prior], current_alpha=initial_alpha)
            # plot_trajectory_distribution(state_action_posterior, f"smooth{i}")
            # plt.show()
            # if converged(metrics):
            #   break
            if plot_posterior:
                # plot_trajectory_distribution(forward_state_action_prior, f"filter {i}")
                plot_trajectory_distribution(state_action_posterior, f"posterior {i}")
                plt.show()
        return policy

    def compute_metrics(self, posterior_distribution, policy_distribution, alpha):
        posterior_cost = self.cost(
            torch.stack(tuple(d.mean for d in posterior_distribution), dim=0)
        )[0].sum()
        policy_cost = self.cost(
            torch.stack(tuple(d.mean for d in policy_distribution), dim=0)
        )[0].sum()
        self.metrics["posterior_cost"] += [posterior_cost]
        self.metrics["policy_cost"] += [policy_cost]
        self.metrics["alpha"] += [alpha]

    def init_metrics(self):
        self.metrics = {
            "posterior_cost": [],
            "policy_cost": [],
            "alpha": [],
        }

    def plot_metrics(self):
        fix, axs = plt.subplots(len(self.metrics))
        for i, (k, v) in enumerate(self.metrics.items()):
            axs[i].plot(v.cpu())
            axs[i].set_ylabel(k)


class TemperatureStrategy(ABC):

    _alpha: torch.float32

    @abstractmethod
    def __call__(
        self,
        cost: Callable,
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        pass


class Constant(TemperatureStrategy):
    """."""

    def __init__(self, alpha):
        assert alpha >= 0.0
        self._alpha = alpha

    def __call__(
        self,
        cost: Callable,
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        return self._alpha


class MaximumLikelihood(TemperatureStrategy):
    """For cost c = f(x, u), c \in [-\infty, 0], then the normalization term integrates to \exp(-\alpha c) / \alpha."""

    def __init__(self, inference: Callable):
        self.inference = inference

    def __call__(
        self,
        cost: Callable,
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        horizon = len(state_action_distribution)
        expected_cost = sum(
            [self.inference.mean(cost, dist) for dist in state_action_distribution]
        ).item()
        return horizon / expected_cost


class KullbackLeiblerDivergence(TemperatureStrategy):
    """."""

    def __init__(self, inference: Callable, epsilon: float):
        assert epsilon > 0.0
        self.inference = inference
        self.epsilon = epsilon

    def __call__(
        self,
        cost: Callable,
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        # fig, axs = plot_trajectory_distribution(state_action_distribution, "kl_constraint")
        # plt.show()
        horizon = len(state_action_distribution)
        # dim_xu = state_action_distribution[0].mean.shape[0]
        # state_action_next_state_distribution = [dist.full_joint(reverse=False) for dist in state_action_distribution]
        alpha_ = torch.ones((1,), requires_grad=True)
        # approximate integral \log\int p(x,u)\exp(-\alpha c(x,u)) dx du over trajectory distribution
        minimizer_args = dict(method="BFGS", options={"disp": True, "maxiter": 100})
        optimizer = MinimizeWrapper([alpha_], minimizer_args)
        # discretize trajectory into samples, weights and evaluations
        # traj_weights, traj_points = [], []
        # for dist in state_action_distribution:
        #     weights, points = inference.weights_mu, inference.get_x_pts(dist.mean, dist.covariance)
        #     traj_weights += [weights]
        #     traj_points += [points]
        # traj_costs = cost(torch.cat(traj_points, axis=0))
        # traj_weights = torch.cat(traj_weights, axis=0)
        list_of_weights, list_of_points = zip(
            *[
                (
                    self.inference.weights_mu,
                    self.inference.get_x_pts(dist.mean, dist.covariance),
                )
                for dist in state_action_distribution
            ]
        )
        # list_of_future_llh_fns = [dist.get_previous_loglikelihood() for dist in state_action_distribution]  # TODO naming sucks!!!!!
        # n_points = list_of_points[0].shape[0]
        # points are (horizon x n_points, dim_xu + dim_x)
        traj_weights, traj_points = map(torch.cat, [list_of_weights, list_of_points])
        traj_costs, _ = cost(traj_points)
        non_zero_weight = traj_weights > 0.0
        log_traj_weights_nz = torch.log(traj_weights[non_zero_weight])
        # traj_future_state_llhs = torch.cat([llh(xunx[:, dim_xu:]) for (llh, xunx) in zip(list_of_future_llh_fns, list_of_points)])
        # import pdb; pdb.set_trace()
        # traj_current_state_llhs = torch.zeros_like(traj_future_state_llhs)
        # the first quadrature points are deterministic, so set weights to 1, otherwise shift future points
        # traj_current_state_llhs[n_points:] = traj_future_state_llhs[:-n_points]
        # import pdb; pdb.set_trace()
        alpha_.data = torch.Tensor(
            [1e-3]
        )  # horizon / torch.einsum("b,b->", traj_weights, traj_costs)
        # n = traj_weights.shape[0]
        # for i, ax in enumerate(axs):
        #     for t, pts in enumerate(list_of_points):
        #         ax.plot(t * torch.ones((pts.shape[0])), pts[: , i], 'g.')
        #         if i < 2:
        #             ax.plot(t * torch.ones((pts.shape[0])), pts[: , dim_xu + i], 'rx')
        advantage_nz = traj_costs[
            non_zero_weight
        ]  # + (traj_future_state_llhs[non_zero_weight] - traj_current_state_llhs[non_zero_weight]) / current_alpha
        alpha_max = 1000.0 / torch.abs(advantage_nz).min()
        print(alpha_max)
        # import pdb; pdb.set_trace()

        def dual(alpha: torch.float64):
            alpha = torch.clamp(torch.abs(alpha), max=alpha_max)
            # optimize advantage
            traj_alpha_cost_and_value = -alpha * advantage_nz
            # return self.epsilon / alpha + torch.log(sum([inference.mean(exp_alpha_cost, dist) for dist in state_action_distribution])) / alpha
            dual = (
                self.epsilon / alpha
                + torch.logsumexp(
                    log_traj_weights_nz + traj_alpha_cost_and_value, dim=0
                )
                / alpha
            )
            if torch.isinf(dual):
                import pdb

                pdb.set_trace()
            return dual

        def closure():
            optimizer.zero_grad()
            loss = dual(alpha_)
            loss.backward()
            return loss

        optimizer.step(closure)
        # import pdb; pdb.set_trace()
        return abs(alpha_.item())


class PolyakStepSize(TemperatureStrategy):
    """."""

    def __init__(self, inference: Callable):
        assert isinstance(
            inference, QuadratureInference
        ), "Computation requires sample discretization"
        self.inference = inference

    def __call__(
        self,
        cost: Callable,
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        list_of_weights, list_of_costs = zip(
            *[
                (
                    self.inference.weights_mu,
                    cost(self.inference.get_x_pts(dist.mean, dist.covariance))[0],
                )
                for dist in state_action_policy
            ]
        )
        traj_weights, traj_costs = map(torch.cat, [list_of_weights, list_of_costs])
        sup_cost = sum([torch.max(costs_t) for costs_t in list_of_costs])
        expected_cost = torch.einsum("b,b->", traj_weights, traj_costs)
        return expected_cost / sup_cost


class QuadraticModel(TemperatureStrategy):
    def __init__(self, inference):
        self.inference = inference
        self.previous_expected_cost = None

    def __call__(
        self,
        cost: Callable,
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        expected_cost_posterior = sum(
            [self.inference.mean(cost, dist) for dist in state_action_distribution]
        )
        expected_cost_policy = sum(
            [self.inference.mean(cost, dist) for dist in state_action_policy]
        )
        if self.previous_expected_cost is None:
            self.previous_expected_cost = expected_cost_posterior
        new_alpha = (
            current_alpha
            * (self.previous_expected_cost - expected_cost_posterior)
            / (2 * (self.previous_expected_cost - expected_cost_policy) + 1e-6)
        )
        self.previous_expected_cost = expected_cost_policy
        return new_alpha


class Annealing(TemperatureStrategy):
    def __init__(self, start, end, iterations):
        assert start > 0.0
        assert end > start
        # self.schedule = iter(torch.exp(torch.linspace(math.log(start), math.log(end), iterations)).tolist())
        self.schedule = iter(
            (
                start
                + (end - start)
                * torch.sqrt(torch.linspace(0, iterations, iterations))
                / math.sqrt(iterations)
            ).tolist()
        )

    def __call__(
        self,
        cost: Callable,
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        return next(self.schedule)


class HitCounter:
    spd_counter = 0  # static

    def spd_fix():
        HitCounter.spd_counter += 1


def nearest_spd(covariance):
    return covariance  # do nothing
    HitCounter.spd_fix()
    L, V = torch.linalg.eig(covariance)
    L[L.real <= 1e-10] = 1e-6  # is it safe to do real?
    return V @ torch.diag(L) @ V.T


def experiment(
    env_type: str = "localPendulum",  # Pendulum
    horizon: int = 200,
    n_rollout_episodes: int = 10,
    batch_size: int = 200 * 10,  # lower if gpu out of memory
    # plot_data: bool = False,
    n_iter: int = 10,  # outer loop
    use_cuda: bool = True,  # for policy/dynamics training (i2c on cpu)
    log_frequency: float = 0.1,  # every p% epochs of n_epochs
    model_save_frequency: float = 0.5,  # every n-th of n_epochs
    ## dynamics ##
    plot_dyn: bool = True,  # plot pointwise and rollout prediction
    # #  D1) true dynamics
    # dyn_model_type: str = "env",
    # #  D2) dnn model
    # dyn_model_type: str = "mlp",
    # n_features_dyn: int = 256,
    # lr_dyn: float = 3e-4,
    # n_epochs_dyn: int = 100,
    # D3) linear regression w/ dnn features
    dyn_model_type: str = "nlm",
    n_features_dyn: int = 128,
    n_hidden_layers_dyn: int = 2,  # 2 ~ [in, h, h, out]
    lr_dyn: float = 1e-4,
    n_epochs_dyn: int = 100,
    # # D4) linear regression w/ sn-dnn & rf features
    # dyn_model_type: str = "snngp",  # TODO why so slow? (x15) paper says x1.2
    # n_features_dyn: int = 128,
    # n_hidden_layers_dyn: int = 2,  # 2 ~ [in, h, h, out]
    # lr_dyn: float = 1e-4,
    # n_epochs_dyn: int = 50,
    ##############
    ## policy ##
    plot_policy: bool = False,  # plot pointwise and rollout prediction
    #  a) no model
    policy_type: str = "tvlg",  # time-varying linear gaussian controllers (i2c)
    ############
    ## i2c solver ##
    n_iter_solver: int = 5,  # how many i2c solver iterations to do
    # plot_posterior: bool = False,  # plot state-action-posterior over time
    plot_posterior: bool = True,  # plot state-action-posterior over time
    plot_local_policy_metrics: bool = False,  # plot time-cum. sa-posterior cost, local policy cost, and alpha per iter
    # plot_local_policy_metrics: bool = True,  # plot time-cum. sa-posterior cost, local policy cost, and alpha per iter
    ############
    ## general ##
    plotting: bool = True,  # if False overrides all other flags
    plot_data: bool = True,  # visualize data trajectories (sns => very slow!)
    log_console: bool = True,  # also log to console (not just log file); FORCE on if debug
    log_wandb: bool = True,  # off if debug
    wandb_project: str = "smbrl_i2c",
    wandb_entity: str = "showmezeplozz",
    wandb_job_type: str = "train",
    seed: int = 0,
    results_dir: str = "logs/tmp/",
    debug: bool = True,  # prepends logdir with 'debug/', disables wandb, enables console logging
    # debug: bool = False,  # prepends logdir with 'debug/', disables wandb, enables console logging
):
    ####################################################################################################################
    #### SETUP (saved to yaml)

    if debug:
        # disable wandb logging and redirect normal logging to ./debug directory
        print("@@@@@@@@@@@@@@@@@ DEBUG: LOGGING DISABLED @@@@@@@@@@@@@@@@@")
        os.environ["WANDB_MODE"] = "disabled"  # close terminal to reset
        log_wandb = False
        log_console = True
        results_dir = "debug" / Path(results_dir)

    # Fix seed
    fix_random_seed(seed)

    # Results directory
    wandb_group: str = f"i2c_{env_type}_{dyn_model_type}_{policy_type}"
    repo_dir = Path.cwd().parent
    results_dir = repo_dir / results_dir / wandb_group / str(seed) / timestamp()
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    # Save arguments
    save_args(results_dir, locals(), git_repo_path="./")

    # logger
    logger = Logger(
        config=locals(),
        log_name=seed,
        results_dir=results_dir,
        project=wandb_project,
        entity=wandb_entity,
        wandb_kwargs={"group": wandb_group, "job_type": wandb_job_type},
        tags=["dyn_model_type", "policy_type", "env_type"],
        log_console=log_console,
        log_wandb=log_wandb,
    )

    ####################################################################################################################
    #### EXPERIMENT SETUP

    logger.info(f"Env: {env_type}, Dyn: {dyn_model_type}, Pol: {policy_type}")
    logger.info(f"Seed: {seed}")

    time_begin = time.time()

    # visialization options
    torch.set_printoptions(precision=7)
    torch.set_default_dtype(torch.float64)

    ### mdp, initial state, cost ###
    if env_type == "localPendulum":
        environment = Pendulum()  # local seed
    dim_xu = environment.dim_xu
    dim_x = environment.dim_x
    dim_u = environment.dim_u
    initial_state = torch.Tensor([torch.pi, 0.0])
    initial_state_distribution = MultivariateGaussian(
        initial_state,
        1e-6 * torch.eye(dim_x),
        # 1e-2 * torch.eye(dim_x),  # more exploration
        None,
        None,
        None,
    )

    def state_action_cost(xu):
        # swing-up from \theta = \pi -> 0
        return (torch.cos(xu[:, 0]) - 1.0) ** 2 + 1e-2 * xu[:, 1] ** 2 + 1e-2 * xu[
            :, 2
        ] ** 2, xu

    train_buffer = ReplayBuffer(
        dim_xu, dim_x, batchsize=batch_size, device=device, max_size=1e4
    )

    test_buffer = ReplayBuffer(
        dim_xu, dim_x, batchsize=batch_size, device=device, max_size=1e4
    )

    ### approximate inference params ###
    quad_params = CubatureQuadrature(1, 0, 0)
    gh_params = GaussHermiteQuadrature(degree=3)
    # if deterministic -> QuadratureInference (unscented gaussian approx)
    # if gaussian -> QuadratureGaussianInference (unscented gaussian approx)

    ### global dynamics model ###
    def clamp_u(xu):
        xu[:, 2] = torch.clamp(xu[:, 2], -environment.u_mx, +environment.u_mx)
        return xu

    def sincos_angle(xu):
        # input angles get split sin/cos
        xu_sincos = torch.zeros((xu.shape[0], xu.shape[1] + 1)).to(xu.device)
        xu_sincos[:, 0] = xu[:, 0].sin()
        xu_sincos[:, 1] = xu[:, 0].cos()
        xu_sincos[:, 2] = xu[:, 1]
        xu_sincos[:, 3] = xu[:, 2]
        return xu_sincos

    if dyn_model_type == "env":
        global_dynamics = environment
        ai_dyn = QuadratureInference(dim_xu, quad_params)
    elif dyn_model_type == "mlp":

        def patch_call(fn):
            def wrap(self, xu):
                xu_sincos = sincos_angle(clamp_u(xu))
                # output states are deltas
                delta_x = fn(self, xu_sincos)
                return xu[:, :dim_x].detach() + delta_x, xu

            return wrap

        DNN3.__call__ = patch_call(DNN3.__call__)
        dim_input = dim_xu + 1  # sin,cos of theta
        global_dynamics = DNN3(dim_input, dim_x, n_features_dyn)
        # global_dynamics.init_whitening(train_buffer.xs, train_buffer.ys)
        loss_fn_dyn = lambda x, y: torch.nn.MSELoss()(global_dynamics(x)[0], y)
        opt_dyn = torch.optim.Adam(global_dynamics.parameters(), lr=lr_dyn)
        ai_dyn = QuadratureInference(dim_xu, quad_params)
    elif dyn_model_type == "nlm":

        def patch_call(fn):
            def wrap(self, xu):
                xu_sincos = sincos_angle(clamp_u(xu))
                # output states are deltas
                delta_mu, cov, cov_in, cov_out = fn(self, xu_sincos)
                # cov_bii = torch.einsum('n,ij->nij', cov_out.diag(), cov_in)
                cov_bij = torch.einsum("ij,n->nij", cov_out, cov_in.diag())
                mu = xu[:, :dim_x].detach() + delta_mu
                return mu, cov_bij, xu

            return wrap

        dim_input = dim_xu + 1  # sin,cos of theta
        NeuralLinearModel.__call__ = patch_call(NeuralLinearModel.__call__)
        global_dynamics = NeuralLinearModel(
            dim_input, dim_x, n_features_dyn, n_hidden_layers_dyn
        )
        # global_dynamics.init_whitening(train_buffer.xs, train_buffer.ys)

        def loss_fn_dyn(xu, x_next):
            xu_sincos = sincos_angle(clamp_u(xu))
            delta_x = x_next - xu[:, :dim_x]
            return -global_dynamics.elbo(xu_sincos, delta_x)

        opt_dyn = torch.optim.Adam(global_dynamics.parameters(), lr=lr_dyn)
        ai_dyn = QuadratureGaussianInference(dim_xu, quad_params)
    elif dyn_model_type == "snngp":

        def patch_call(fn):
            def wrap(self, xu):
                xu_sincos = sincos_angle(clamp_u(xu))
                # output states are deltas
                delta_mu, cov, cov_in, cov_out = fn(self, xu_sincos)
                # cov_bii = torch.einsum('n,ij->nij', cov_out.diag(), cov_in)
                cov_bij = torch.einsum("ij,n->nij", cov_out, cov_in.diag())
                mu = xu[:, :dim_x].detach() + delta_mu
                return mu, cov_bij, xu

            return wrap

        dim_input = dim_xu + 1  # sin,cos of theta
        SpectralNormalizedNeuralGaussianProcess.__call__ = patch_call(
            SpectralNormalizedNeuralGaussianProcess.__call__
        )
        global_dynamics = SpectralNormalizedNeuralGaussianProcess(
            dim_input, dim_x, n_features_dyn, n_hidden_layers_dyn
        )
        # global_dynamics.init_whitening(train_buffer.xs, train_buffer.ys)

        def loss_fn_dyn(xu, x_next):
            xu_sincos = sincos_angle(clamp_u(xu))
            delta_x = x_next - xu[:, :dim_x]
            return -global_dynamics.elbo(xu_sincos, delta_x)

        opt_dyn = torch.optim.Adam(global_dynamics.parameters(), lr=lr_dyn)
        ai_dyn = QuadratureGaussianInference(dim_xu, quad_params)

    ### local (i2c) policy ###
    local_policy = TimeVaryingLinearGaussian(
        horizon,
        dim_x,
        dim_u,
        action_covariance=0.2 * torch.eye(dim_u),
    )

    ### global policy model ###
    if policy_type == "tvlg":
        global_policy = local_policy
        ai_pol = QuadratureGaussianInference(dim_x, quad_params)
    elif policy_type == "mlp":
        pass  # TODO
    elif policy_type == "nlm":
        pass  # TODO
    elif policy_type == "snngp":
        pass  # TODO

    ### rollout policy (mock object) ###
    class MockPolicy:
        def predict(self, *args):
            pass  # fill/override before use

    exploration_policy = MockPolicy()

    ### i2c solver ###
    i2c_solver = PseudoPosteriorSolver(
        dim_x,
        dim_u,
        global_dynamics,
        state_action_cost,
        horizon,
        initial_state_distribution,
        policy_prior=global_policy,
        approximate_inference_dynamics=ai_dyn,
        approximate_inference_policy=ai_pol,
        approximate_inference_cost=QuadratureImportanceSamplingInnovation(
            dim_xu,
            gh_params,
        ),
        # approximate_inference_cost=LinearizationInnovation(),
        approximate_inference_smoothing=linear_gaussian_smoothing,
        # update_temperature_strategy=Constant(alpha),
        # update_temperature_strategy=MaximumLikelihood(QuadratureInference(dim_xu, quad_params)),
        # update_temperature_strategy=QuadraticModel(QuadratureInference(dim_xu, quad_params)),
        # update_temperature_strategy=KullbackLeiblerDivergence(QuadratureInference(dim_xu, gh_params), epsilon=kl_bound),
        # update_temperature_strategy=Annealing(1e-2, 5e1, 15+1),
        update_temperature_strategy=PolyakStepSize(
            QuadratureInference(dim_xu, quad_params)
        ),
    )

    ####################################################################################################################
    #### TRAINING

    # for alpha in [1e-1, 1, 10, 100, 1000]:
    # for kl_bound in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
    # for _ in [None]:
    dyn_loss_trace = []
    dyn_test_loss_trace = []
    for i_iter in range(n_iter):
        logger.strong_line()
        logger.info(f"ITERATION {i_iter + 1}/{n_iter}")
        if dyn_model_type != "env":
            global_dynamics.cpu()  # in-place
            global_dynamics.eval()
            torch.set_grad_enabled(False)

        # # tvlg with feedback
        # exploration_policy = global_policy.actual()

        # # tvlg without feedback
        # exploration_policy = global_policy

        # # constant
        # exploration_policy.predict = lambda self, *args: 1.0 * torch.ones(dim_u)

        # 50-50 %: tvgl-fb or gaussian noise
        def noise_pred(*args):
            dithering = torch.randn(dim_u)
            # if torch.randn(1) > 0.5:
            #     # return global_policy.predict(*args)  # TODO actual makes no difference?
            #     return global_policy.actual().predict(*args) + dithering
            # else:
            #     return torch.normal(torch.zeros(dim_u), 3 * torch.ones(dim_u))
            return global_policy.actual().predict(*args) + dithering
            # return torch.normal(0.0 * torch.ones(dim_u), 1e-2 * torch.ones(dim_u))

        exploration_policy.predict = noise_pred

        # # random gaussian
        # exploration_policy.predict = lambda *_: torch.randn(dim_u)  # N(0,1)
        # exploration_policy.predict = lambda *_: torch.randn(dim_u) * 3

        # train and test rollouts (env & exploration policy)
        logger.weak_line()
        logger.info("START Collecting Rollouts")
        for i in trange(n_rollout_episodes):  # 80 % train
            state = initial_state_distribution.sample()
            s, a, ss = environment.run(state, exploration_policy, horizon)
            train_buffer.add(torch.hstack([s, a]), ss)
        for i in trange(int(n_rollout_episodes / 4)):  # 20 % test
            state = initial_state_distribution.sample()
            s, a, ss = environment.run(state, exploration_policy, horizon)
            test_buffer.add(torch.hstack([s, a]), ss)
        logger.info("END Collecting Rollouts")

        if dyn_model_type != "env":
            # Learn Dynamics
            logger.weak_line()
            logger.info("START Training Dynamics")
            global_dynamics.to(device)  # in-place
            global_dynamics.train()
            torch.set_grad_enabled(True)

            ## initial loss
            # for minibatch in train_buffer:  # TODO not whole buffer!
            #     _x, _y = minibatch
            #     _loss = loss_fn_dyn(_x, _y)
            #     dyn_loss_trace.append(_loss.detach().item())
            #     logger.log_data(
            #         **{
            #             "dynamics/train/loss": dyn_loss_trace[-1],
            #         },
            #     )
            # test_losses = []
            # for minibatch in test_buffer:  # TODO not whole buffer!
            #     _x_test, _y_test = minibatch
            #     _test_loss = loss_fn_dyn(_x_test, _y_test)
            #     test_losses.append(_test_loss.detach().item())
            # dyn_test_loss_trace.append(np.mean(test_losses))
            # logger.log_data(
            #     step=logger._step,  # in sync with training loss
            #     **{
            #         "dynamics/eval/loss": dyn_test_loss_trace[-1],
            #     },
            # )

            # torch.autograd.set_detect_anomaly(True)
            for i_epoch_dyn in trange(n_epochs_dyn + 1):
                for i_minibatch, minibatch in enumerate(train_buffer):
                    x, y = minibatch
                    opt_dyn.zero_grad()
                    loss = loss_fn_dyn(x, y)
                    loss.backward()
                    opt_dyn.step()
                    dyn_loss_trace.append(loss.detach().item())
                    logger.log_data(
                        **{
                            # "dynamics/train/epoch": i_epoch_dyn,
                            "dynamics/train/loss": dyn_loss_trace[-1],
                        },
                    )

                # save logs & test
                logger.save_logs()
                if i_epoch_dyn % (n_epochs_dyn * log_frequency) == 0:
                    with torch.no_grad():
                        # test loss
                        test_buffer.shuffling = False  # TODO only for plotting?
                        test_losses = []
                        for minibatch in test_buffer:  # TODO not whole buffer!
                            _x_test, _y_test = minibatch
                            _test_loss = loss_fn_dyn(_x_test, _y_test)
                            test_losses.append(_test_loss.item())
                        dyn_test_loss_trace.append(np.mean(test_losses))
                        logger.log_data(
                            step=logger._step,  # in sync with training loss
                            **{
                                "dynamics/eval/loss": dyn_test_loss_trace[-1],
                            },
                        )

                        logstring = f"DYN: Epoch {i_epoch_dyn}, Train Loss={dyn_loss_trace[-1]:.2}"
                        logstring += f", Test Loss={dyn_test_loss_trace[-1]:.2}"
                        logger.info(logstring)

                # TODO save model more often than in each global iter?
                # if n % (n_epochs * model_save_frequency) == 0:
                #     # Save the agent
                #     torch.save(model.state_dict(), results_dir / f"agent_{n}_{i_iter}.pth")

            # Save the model after training
            global_dynamics.cpu()  # in-place
            global_dynamics.eval()
            torch.set_grad_enabled(False)
            torch.save(
                global_dynamics.state_dict(),
                results_dir / "dyn_model_{i_iter}.pth",
            )
            logger.info("END Training Dynamics")

            if plot_dyn:
                ## test dynamics model in rollouts
                # TODO extract?
                ## data traj (from buffer)
                # sa_env = test_buffer.xs[:horizon, :].cpu()  # first
                sa_env = test_buffer.xs[-horizon:, :].cpu()  # last
                s_env, a_env = sa_env[:, :dim_x], sa_env[:, dim_x:]
                # ss_env = test_buffer.ys[:horizon, :].cpu()  # first
                ss_env = test_buffer.ys[-horizon:, :].cpu()  # last
                ss_pred_pw = torch.zeros((horizon, dim_x))
                ss_pred_roll = torch.zeros((horizon, dim_x))
                state = s_env[0, :]  # for rollouts: data init state
                for t in range(horizon):
                    # pointwise
                    xu = torch.cat((s_env[t, :], a_env[t, :]))[None, :]  # pred traj
                    if dyn_model_type in ["nlm", "snngp"]:
                        ss_pred_pw[t, :], var, xu = global_dynamics(xu)
                    else:
                        ss_pred_pw[t, :], xu = global_dynamics(xu)
                    # rollout (replay action)
                    xu = torch.cat((state, a_env[t, :]))[None, :]
                    if dyn_model_type in ["nlm", "snngp"]:
                        ss_pred_roll[t, :], var, xu = global_dynamics(xu)
                    else:
                        ss_pred_roll[t, :], xu = global_dynamics(xu)
                    state = ss_pred_roll[t, :]
                # compute rewards (except init state use pred next state)
                r_env, _ = state_action_cost(sa_env)
                s_pred_pw = torch.cat([s_env[:1, :], ss_pred_pw[:-1, :]])
                sa_pred_pw = torch.cat([s_pred_pw, a_env], dim=1)
                r_pw, _ = state_action_cost(sa_pred_pw)
                s_pred_roll = torch.cat([s_env[:1, :], ss_pred_roll[:-1, :]])
                sa_pred_roll = torch.cat([s_pred_roll, a_env], dim=1)
                r_roll, _ = state_action_cost(sa_pred_roll)

                ### plot pointwise and rollout predictions (1 episode) ###
                fig, axs = plt.subplots(dim_u + 1 + dim_x, 2, figsize=(10, 7))
                steps = torch.tensor(range(0, horizon))
                axs[0, 0].set_title("pointwise predictions")
                axs[0, 1].set_title("rollout predictions")
                # plot actions (twice: left & right)
                for ui in range(dim_u):
                    axs[ui, 0].plot(steps, a_env[:, ui], color="b")
                    axs[ui, 0].set_ylabel("action")
                    axs[ui, 1].plot(steps, a_env[:, ui], color="b")
                    axs[ui, 1].set_ylabel("action")
                # plot reward
                ri = dim_u
                axs[ri, 0].plot(steps, r_env, color="b", label="data")
                axs[ri, 0].plot(steps, r_pw, color="r", label=dyn_model_type)
                axs[ri, 0].set_ylabel("reward")
                axs[ri, 1].plot(steps, r_env, color="b", label="data")
                axs[ri, 1].plot(steps, r_roll, color="r", label=dyn_model_type)
                axs[ri, 1].set_ylabel("reward")
                for xi in range(dim_x):
                    xi_ = xi + dim_u + 1  # plotting offset
                    # plot pointwise state predictions
                    axs[xi_, 0].plot(steps, ss_env[:, xi], color="b")
                    axs[xi_, 0].plot(steps, ss_pred_pw[:, xi], color="r")
                    axs[xi_, 0].set_ylabel(f"ss[{xi}]")
                    # plot rollout state predictions
                    axs[xi_, 1].plot(steps, ss_env[:, xi], color="b")
                    axs[xi_, 1].plot(steps, ss_pred_roll[:, xi], color="r")
                    axs[xi_, 1].set_ylabel(f"ss[{xi}]")
                axs[-1, 0].set_xlabel("steps")
                axs[-1, 1].set_xlabel("steps")
                handles, labels = axs[ri, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=2)
                fig.suptitle(
                    f"{dyn_model_type} pointwise and rollout dynamics on 1 episode "
                    f"({int(test_buffer.size/horizon)} "
                    f"episodes, {i_iter * n_epochs_dyn} epochs, lr={lr_dyn})"
                )
                plt.savefig(results_dir / f"dyn_eval_{i_iter}.png", dpi=150)
                if plotting:
                    plt.show()

        # i2c: find local (optimal) tvlg policy
        logger.weak_line()
        logger.info(f"START i2c [{n_iter_solver} iters]")
        # TODO actual?
        i2c_solver.policy_prior = global_policy  # update prior policy
        local_policy = i2c_solver(
            n_iteration=n_iter_solver, plot_posterior=plot_posterior and plotting
        )
        logger.info("END i2c")
        # log i2c metrics
        for i_ in range(n_iter_solver):
            log_dict = {"iter": i_iter, "i2c_iter": i_}
            for (k, v) in i2c_solver.metrics.items():
                log_dict[k] = v[i_]  # one key, n_iter_solver values
            logger.log_data(log_dict)

        ## plot current i2c optimal controller
        if plot_local_policy_metrics and plotting:
            fix, axs = plt.subplots(3)
            for i, (k, v) in enumerate(i2c_solver.metrics.items()):
                v = torch.tensor(v).cpu()
                # axs[i].plot(v, label=f"$\epsilon={kl_bound}$")
                # axs[i].plot(v, label=f"$\\alpha={alpha}$")
                axs[i].plot(v, label=f"$Polyak$")
                # axs[i].plot(v, label=f"Maximum Likelihood")
                # axs[i].plot(v, label="Quadratic Model")
                # axs[i].plot(v, label="Annealing")
                axs[i].set_ylabel(k)
            for ax in axs:
                ax.legend()
            # TODO clean up plot
            plt.savefig(results_dir / "i2c_metrics_{i_iter}.png", dpi=150)
            plt.show()

        # Fit global policy to local policy
        if policy_type == "tvlg":
            global_policy = local_policy
        else:
            pass  # TODO
            # min KL[mean local_policy || global_policy]
            # should do: global_policy.predict(x, t)

    ####################################################################################################################
    #### EVALUATION

    # local_policy.plot_metrics()
    initial_state = torch.Tensor([torch.pi, 0.0])
    # initial_state = torch.Tensor([torch.pi + 0.4, 0.0])  # breaks local_policy!!
    if policy_type == "tvlg":
        global_policy = global_policy.actual()

    ### policy vs env
    # TODO label plot
    xs, us, xxs = environment.run(initial_state, global_policy, horizon)
    environment.plot(xs, us)
    plt.suptitle(f"{policy_type} policy vs env")
    plt.savefig(results_dir / f"{policy_type}_vs_env_{i_iter}.png", dpi=150)

    ### policy vs dyn model
    xs = torch.zeros((horizon, dim_x))
    us = torch.zeros((horizon, dim_u))
    state = initial_state
    for t in range(horizon):
        action = global_policy.predict(state, t)
        xu = torch.cat((state, action))[None, :]
        x_, *_ = global_dynamics(xu)
        xs[t, :] = state
        us[t, :] = action
        state = x_[0, :]
    environment.plot(xs, us)  # env.plot does not use env, it only plots
    plt.suptitle(f"{policy_type} policy vs {dyn_model_type} dynamics")
    plt.savefig(
        results_dir / f"{policy_type}_vs_{dyn_model_type}_{i_iter}.png", dpi=150
    )

    ### plot data space coverage ###
    if plot_data:
        cols = ["theta", "theta_dot"]  # TODO useful to also plot actions?
        # train data
        xs = train_buffer.xs[:, :dim_x]  # only states
        df = pd.DataFrame()
        df[cols] = np.array(xs.cpu())
        df["traj_id"] = df.index // horizon
        g = sns.PairGrid(df, hue="traj_id")
        # g = sns.PairGrid(df)
        g.map_diag(sns.histplot, hue=None)
        g.map_offdiag(plt.plot)
        g.fig.suptitle(f"train data ({df.shape[0] // horizon} episodes)", y=1.01)
        g.savefig(results_dir / "train_data.png", dpi=150)
        # test data
        xs = test_buffer.xs[:, :dim_x]  # only states
        df = pd.DataFrame()
        df[cols] = np.array(xs.cpu())
        df["traj_id"] = df.index // horizon
        g = sns.PairGrid(df, hue="traj_id")
        # g = sns.PairGrid(df)
        g.map_diag(sns.histplot, hue=None)
        g.map_offdiag(plt.plot)
        g.fig.suptitle(f"test data ({df.shape[0] // horizon} episodes)", y=1.01)
        g.savefig(results_dir / "test_data.png", dpi=150)

    # plot training loss
    def scaled_xaxis(y_points, n_on_axis):
        return np.arange(len(y_points)) / len(y_points) * n_on_axis

    if dyn_model_type != "env":
        fig_loss_dyn, ax_loss_dyn = plt.subplots()
        x_train_loss_dyn = scaled_xaxis(dyn_loss_trace, n_iter * n_epochs_dyn)
        ax_loss_dyn.plot(x_train_loss_dyn, dyn_loss_trace, c="k", label="train loss")
        x_test_loss_dyn = scaled_xaxis(dyn_test_loss_trace, n_iter * n_epochs_dyn)
        ax_loss_dyn.plot(x_test_loss_dyn, dyn_test_loss_trace, c="g", label="test loss")
        if dyn_loss_trace[0] > 1 and dyn_loss_trace[-1] < 0.1:
            ax_loss_dyn.set_yscale("symlog")
        ax_loss_dyn.set_xlabel("epochs")
        ax_loss_dyn.set_ylabel("loss")
        ax_loss_dyn.set_title(
            f"DYN {dyn_model_type} loss "
            f"({int(train_buffer.size/horizon)} episodes, lr={lr_dyn:.0e})"
        )
        ax_loss_dyn.legend()
        plt.savefig(results_dir / "dyn_loss.png", dpi=150)
        # TODO plot policy train loss

    if plotting:
        plt.show()

    logger.strong_line()
    logger.info(f"Seed: {seed} - Took {time.time()-time_begin:.2f} seconds")
    logger.info(f"Logs in {results_dir}")
    logger.finish()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
